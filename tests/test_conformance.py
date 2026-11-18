"""The conformance suite against the in-process reference endpoint, its quirks and scripted fakes."""

from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.helpers import TEST_KEY, config, provider_dict

from llm_api_benchmark import conformance as conf
from llm_api_benchmark.cli import main
from llm_api_benchmark.config import REFERENCE_QUIRKS, ConfigError, ProviderConfig
from llm_api_benchmark.reports.conformance import (
    conformance_junit,
    conformance_markdown,
    conformance_table,
)

ROOT = Path(__file__).resolve().parents[1]


def reference(quirks: list[str] | None = None) -> ProviderConfig:
    synthetic = {"quirks": quirks} if quirks else {}
    return config(
        [{"name": "ref", "preset": "synthetic", "model": "ref-model", "synthetic": synthetic}]
    ).providers[0]


def run(
    p: ProviderConfig, tests: list[conf.TestSpec] | None = None, **kw: Any
) -> conf.ProviderReport:
    return asyncio.run(
        conf.run_provider(p, tests or list(conf.TESTS), timeout_s=5, connect_timeout_s=2, **kw)
    )


def outcomes(report: conf.ProviderReport) -> dict[str, str]:
    return {r.id: r.outcome for r in report.results}


def test_the_reference_endpoint_passes_everything() -> None:
    report = run(reference())
    assert set(outcomes(report).values()) == {conf.PASS}
    assert sum(r.requests for r in report.results) == conf.planned_requests(
        reference(), list(conf.TESTS)
    )


@pytest.mark.parametrize(
    ("quirk", "expected"),
    [
        ("models_missing", {"models.list": conf.FAIL}),
        (
            "ignore_max_tokens",
            {"params.max_tokens": conf.FAIL, "params.max_completion_tokens": conf.FAIL},
        ),
        ("ignore_stop", {"params.stop": conf.FAIL}),
        ("ignore_n", {"params.n": conf.FAIL}),
        (
            "ignore_response_format",
            {
                "params.response_format.json_object": conf.WARN,
                "params.response_format.json_schema": conf.FAIL,
            },
        ),
        ("ignore_logprobs", {"params.logprobs": conf.FAIL}),
        ("ignore_tool_choice", {"tools.choice_none": conf.FAIL}),
        ("tool_args_object", {"tools.call": conf.FAIL}),
        ("no_stream_usage", {"stream.usage": conf.FAIL}),
        ("no_done", {"stream.basic": conf.FAIL, "tools.stream": conf.FAIL}),
        ("accept_out_of_range", {"params.temperature_range": conf.WARN}),
        ("substitute_model", {"errors.unknown_model": conf.FAIL}),
        ("accept_empty_messages", {"errors.invalid_request": conf.FAIL}),
        ("server_error_on_bad_json", {"errors.malformed_json": conf.FAIL}),
        ("accept_any_key", {"errors.invalid_key": conf.FAIL, "errors.missing_key": conf.FAIL}),
        (
            "bare_errors",
            {
                "errors.invalid_key": conf.WARN,
                "errors.missing_key": conf.WARN,
                "errors.unknown_model": conf.WARN,
                "errors.invalid_request": conf.WARN,
                "errors.malformed_json": conf.WARN,
            },
        ),
    ],
)
def test_each_quirk_is_detected_and_nothing_else_changes(
    quirk: str, expected: dict[str, str]
) -> None:
    got = outcomes(run(reference([quirk])))
    changed = {k: v for k, v in got.items() if v != conf.PASS}
    assert changed == expected


def test_every_quirk_has_a_test() -> None:
    tested = {
        "models_missing",
        "ignore_max_tokens",
        "ignore_stop",
        "ignore_n",
        "ignore_response_format",
        "ignore_logprobs",
        "ignore_tool_choice",
        "tool_args_object",
        "no_stream_usage",
        "no_done",
        "accept_out_of_range",
        "substitute_model",
        "accept_empty_messages",
        "server_error_on_bad_json",
        "accept_any_key",
        "bare_errors",
    }
    assert tested == set(REFERENCE_QUIRKS)


def test_unknown_quirk_is_a_config_error() -> None:
    with pytest.raises(ConfigError) as err:
        reference(["ignore_everything"])
    assert any("unknown quirk" in p for p in err.value.problems)


def test_selection_by_id_category_and_prefix() -> None:
    assert [t.id for t in conf.select_tests(["errors.unknown_model"], None)] == [
        "errors.unknown_model"
    ]
    assert {t.category for t in conf.select_tests(["tools"], None)} == {"tools"}
    picked = conf.select_tests(["params.response_format"], None)
    assert [t.id for t in picked] == [
        "params.response_format.json_object",
        "params.response_format.json_schema",
    ]
    rest = conf.select_tests(None, ["errors", "params"])
    assert not any(
        t.category in ("errors", "parameters") or t.id.startswith("params.") for t in rest
    )
    with pytest.raises(ValueError, match="unknown test or category"):
        conf.select_tests(["nope"], None)


def test_chat_usage_is_skipped_without_chat_basic() -> None:
    report = run(reference(), conf.select_tests(["chat.usage"], None))
    assert report.results[0].outcome == conf.SKIP and report.results[0].requests == 0


# ----------------------------------------------------------------------------- scripted endpoints


def real_provider(**overrides: Any) -> ProviderConfig:
    return config([provider_dict(rate_limit_rpm=None, **overrides)]).providers[0]


def test_a_bad_key_stops_the_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", TEST_KEY)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
        return httpx.Response(
            402, json={"error": {"type": "no_active_plan", "message": "No plan."}}
        )

    report = run(real_provider(), transport=httpx.MockTransport(handler))
    assert report.results[0].outcome == conf.ERROR and "HTTP 402" in report.results[0].summary
    assert all(r.outcome in (conf.ERROR, conf.SKIP) for r in report.results)
    assert len(seen) == 1 and report.stopped
    assert TEST_KEY not in json.dumps(report.to_dict())


def test_rate_limit_retry_after_is_honored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", TEST_KEY)
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "2"}, json={"message": "Too Many Attempts."}
            )
        return httpx.Response(
            200, json={"object": "list", "data": [{"id": "fake-model", "object": "model"}]}
        )

    async def sleep(s: float) -> None:
        sleeps.append(s)

    report = run(
        real_provider(),
        conf.select_tests(["models.list"], None),
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    assert report.results[0].outcome == conf.PASS and report.results[0].requests == 2
    assert sleeps == [2.0]


def test_requests_are_paced_to_the_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", TEST_KEY)
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key", "type": "auth"}})

    p = config([provider_dict(rate_limit_rpm=6)]).providers[0]
    run(
        p,
        conf.select_tests(["errors.invalid_key", "errors.missing_key"], None),
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    assert len(sleeps) == 1 and 9.0 < sleeps[0] <= 10.0


def test_an_explicit_400_for_a_parameter_is_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", TEST_KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {"message": "logprobs is not supported", "type": "invalid_request_error"}
            },
        )

    report = run(
        real_provider(),
        conf.select_tests(["params.logprobs", "params.n"], None),
        transport=httpx.MockTransport(handler),
    )
    assert outcomes(report) == {"params.logprobs": conf.WARN, "params.n": conf.WARN}
    assert "rejected with HTTP 400" in report.results[0].summary


def test_html_error_pages_and_server_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", TEST_KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(503, text="<html><title>503 Service Unavailable</title></html>")
        return httpx.Response(
            524, text="<html><head><title>A timeout occurred</title></head></html>"
        )

    report = run(
        real_provider(),
        conf.select_tests(["models.list", "errors.unknown_model"], None),
        transport=httpx.MockTransport(handler),
    )
    got = {r.id: r for r in report.results}
    assert got["models.list"].outcome == conf.ERROR and "503" in got["models.list"].summary
    assert got["errors.unknown_model"].outcome == conf.FAIL
    assert "server error" in got["errors.unknown_model"].summary


# ----------------------------------------------------------------------------- reports and CLI


def result_for(*providers: tuple[str, list[str]]) -> conf.ConformanceResult:
    reports = []
    for name, quirks in providers:
        p = config(
            [{"name": name, "preset": "synthetic", "model": "m", "synthetic": {"quirks": quirks}}]
        ).providers[0]
        reports.append(run(p))
    return conf.ConformanceResult(reports, {"python": "3", "httpx": "0", "tool_version": "x"}, True)


def test_reports_render_and_round_trip() -> None:
    result = result_for(("good", []), ("bad", ["ignore_n", "bare_errors"]))
    table = conformance_table(result)
    assert "SYNTHETIC FIXTURE DATA" in table and "FAIL  params.n" in table
    md = conformance_markdown(result)
    assert "| `params.n` | n=2 returns two choices | PASS | FAIL |" in md
    root = ET.fromstring(conformance_junit(result).split("\n", 1)[1])
    suites = root.findall("testsuite")
    assert [s.get("name") for s in suites] == ["conformance.good", "conformance.bad"]
    assert suites[1].get("failures") == "1" and len(suites[1].findall("testcase")) == len(
        conf.TESTS
    )
    again = conf.ConformanceResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert conformance_table(again) == table


def test_cli_conformance_exit_codes_and_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = ROOT / "benchmarks" / "synthetic-conformance.yaml"
    out = tmp_path / "out"
    assert (
        main(["conformance", "-c", str(cfg), "-q", "-o", str(out), "--formats", "json,md,junit"])
        == 1
    )
    files = sorted(p.name.split("Z", 1)[1] for p in out.iterdir())
    assert files == [".json", ".junit.xml", ".md"]
    assert "quirky" in capsys.readouterr().out
    assert (
        main(["conformance", "-c", str(cfg), "-q", "-o", str(out), "--providers", "reference"]) == 0
    )
    assert main(["conformance", "-c", str(cfg), "-q", "-o", str(out), "--fail-on", "never"]) == 0
    result_file = next(out.glob("*.json"))
    assert main(["report", str(result_file), "--format", "junit"]) == 0
    assert "<testsuites" in capsys.readouterr().out
    assert main(["conformance", "--list"]) == 0
    assert "errors.malformed_json" in capsys.readouterr().out


def test_cli_conformance_dry_run_needs_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MOLOCHAI_API_KEY", raising=False)
    cfg = ROOT / "benchmarks" / "molochai.yaml"
    assert main(["conformance", "-c", str(cfg), "--dry-run"]) == 0
    err = capsys.readouterr().err
    assert "NOT SET" in err and "21 requests" in err and "nothing was sent" in err
    assert main(["conformance", "-c", str(cfg), "-o", str(tmp_path)]) == 2
