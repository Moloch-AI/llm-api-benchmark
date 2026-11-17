"""Credentials never reach the terminal, logs or result files."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from tests.helpers import TEST_KEY

from llm_api_benchmark.cli import main
from llm_api_benchmark.redact import RedactingFilter, Redactor, mask, redact_url


def test_mask() -> None:
    assert mask(None) == "(not set)"
    assert mask("short") == "****"
    assert mask(TEST_KEY) == "****" + TEST_KEY[-4:]
    assert TEST_KEY[:-4] not in mask(TEST_KEY)


def test_redactor_patterns() -> None:
    r = Redactor()
    r.add("my-registered-secret")
    r.add("")  # ignored
    text = r(
        "token my-registered-secret, Authorization: Bearer abcdefghijkl, key sk-proj-ABCDEFGHIJKLMNOPQRST, url ?api_key=zzzzzzzz&x=1"
    )
    assert "my-registered-secret" not in text
    assert "abcdefghijkl" not in text
    assert "ABCDEFGHIJKLMNOPQRST" not in text
    assert "zzzzzzzz" not in text
    assert "x=1" in text


def test_redact_url() -> None:
    assert (
        redact_url("https://user:pw@host.test/v1?key=abcdef123&a=b")
        == "https://[redacted]@host.test/v1?key=[redacted]&a=b"
    )


def test_logging_filter() -> None:
    r = Redactor()
    r.add("hunter2-secret")
    record = logging.LogRecord(
        "x", logging.INFO, __file__, 1, "key is %s", ("hunter2-secret",), None
    )
    RedactingFilter(r).filter(record)
    assert record.getMessage() == "key is [redacted]"


def test_key_echoed_by_the_server_never_reaches_output_or_files(
    tmp_path: Path,
    local_server: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The server rejects this key and echoes it back in the error message.
    wrong = "sk-wrong-key-ABCDEFGHIJKLMNOP"
    monkeypatch.setenv("FAKE_API_KEY", wrong)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "name: leak-test\nworkloads: [short-answer]\nrepetitions: 1\nwarmup: 0\nproviders:\n"
        f"  - {{name: local, base_url: '{local_server.base_url}', api_key_env: FAKE_API_KEY, model: m}}\n",  # type: ignore[attr-defined]
        encoding="utf-8",
    )
    code = main(
        ["run", "-c", str(cfg), "--out-dir", str(tmp_path / "out"), "--formats", "json,csv,md,html"]
    )
    assert code == 0
    out = capsys.readouterr()
    assert "http_401" in out.out
    for text in (
        out.out,
        out.err,
        *(p.read_text(encoding="utf-8") for p in (tmp_path / "out").iterdir()),
    ):
        assert wrong not in text
        assert "ABCDEFGHIJKLMNOP" not in text
    files = list((tmp_path / "out").iterdir())
    assert len(files) == 5
