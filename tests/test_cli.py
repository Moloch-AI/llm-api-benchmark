"""The llm-bench commands, end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_api_benchmark import __version__
from llm_api_benchmark.cli import main

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "src" / "llm_api_benchmark" / "templates"


def write_config(tmp_path: Path, body: str, name: str = "c.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def local_config(tmp_path: Path, base_url: str, **extra: str) -> Path:
    lines = [
        "name: local",
        "workloads: [short-answer]",
        f"repetitions: {extra.pop('repetitions', '1')}",
        "warmup: 0",
    ]
    lines += [f"{k}: {v}" for k, v in extra.items()]
    lines += [
        "providers:",
        f"  - {{name: local, base_url: '{base_url}', api_key_env: FAKE_API_KEY, model: local-a}}",
    ]
    return write_config(tmp_path, "\n".join(lines) + "\n")


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_providers(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "molochai" in out and "openai-compatible" in out and "synthetic" in out
    assert "utm_campaign=llm-api-benchmark" in out
    assert main(["providers", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["molochai"]["base_url"].endswith("https://api.molochai.dev/api/v1}")


def test_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["datasets"]) == 0
    assert "core-v1" in capsys.readouterr().out
    assert main(["datasets", "core-v1", "--prompts"]) == 0
    out = capsys.readouterr().out
    assert "short-answer" in out and "capital city of Australia" in out


def test_init_writes_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert (tmp_path / "synthetic-demo.yaml").exists()
    assert main(["init"]) == 2  # refuses to overwrite
    assert main(["init", "molochai", "m.yaml"]) == 0
    assert (tmp_path / "m.yaml").read_text(encoding="utf-8") == (
        TEMPLATES / "molochai.yaml"
    ).read_text(encoding="utf-8")


def test_dry_run_sends_nothing(
    tmp_path: Path, local_server: object, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = local_config(tmp_path, local_server.base_url)  # type: ignore[attr-defined]
    assert main(["run", "-c", str(cfg), "--dry-run"]) == 0
    err = capsys.readouterr().err
    assert "Requests     3 (limit 100)" in err
    assert "Dry run: nothing was sent." in err
    assert "set (****" in err
    assert local_server.seen == []  # type: ignore[attr-defined]


def test_run_against_a_local_server(
    tmp_path: Path, local_server: object, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = local_config(tmp_path, local_server.base_url, repetitions="2")  # type: ignore[attr-defined]
    out_dir = tmp_path / "out"
    assert main(["run", "-c", str(cfg), "-o", str(out_dir), "--formats", "csv,md,html", "-q"]) == 0
    captured = capsys.readouterr()
    assert "short-answer" in captured.out
    assert len(local_server.seen) == 6  # type: ignore[attr-defined]
    files = sorted(p.name.split("-", 1)[1].split(".", 1)[1] for p in out_dir.iterdir())
    assert files == ["html", "json", "md", "requests.csv", "summary.csv"]
    result = json.loads(next(out_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert {r["status"] for r in result["requests"]} == {"ok"}
    assert all(r["checks_passed"] for r in result["requests"] if r["prompt_id"] == "capital")

    json_file = str(next(out_dir.glob("*.json")))
    for fmt, marker in [
        ("table", "Reliability"),
        ("markdown", "## Limitations"),
        ("html", "<svg"),
        ("csv", "latency_median_s"),
        ("requests-csv", "ttft_s"),
        ("json", '"schema"'),
    ]:
        assert main(["report", json_file, "-f", fmt]) == 0
        assert marker in capsys.readouterr().out
    target = tmp_path / "r.md"
    assert main(["report", json_file, "-f", "markdown", "-o", str(target)]) == 0
    assert target.read_text(encoding="utf-8").startswith("# LLM API benchmark")
    assert main(["compare", json_file, json_file]) == 0
    assert "Verdict" in capsys.readouterr().out


def test_run_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = "https://unused.test/v1"
    cfg = local_config(tmp_path, base, repetitions="5", max_requests="10")
    assert main(["run", "-c", str(cfg)]) == 2
    assert "more than max_requests (10)" in capsys.readouterr().err

    monkeypatch.delenv("FAKE_API_KEY")
    cfg2 = local_config(tmp_path, base)
    assert main(["run", "-c", str(cfg2)]) == 2
    assert "FAKE_API_KEY is not set" in capsys.readouterr().err

    bad = write_config(tmp_path, "providers: []\n", "bad.yaml")
    assert main(["run", "-c", str(bad)]) == 2
    assert "list at least one endpoint" in capsys.readouterr().err

    assert main(["run", "-c", str(cfg2), "--concurrency", "99"]) == 2
    assert "concurrency" in capsys.readouterr().err


def test_large_runs_need_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = local_config(tmp_path, "https://unused.test/v1", repetitions="10")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["run", "-c", str(cfg)]) == 2
    assert "Add --yes to confirm" in capsys.readouterr().err


def test_budget_over_worst_case_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = (
        "workloads: [long-response]\nrepetitions: 1\nwarmup: 0\nbudget: 0.000001\n"
        "providers:\n  - name: p\n    base_url: https://unused.test/v1\n    api_key_env: FAKE_API_KEY\n    model: m\n"
        "    pricing: {input_per_mtok: 1, output_per_mtok: 10, source: https://example.com, checked_on: 2026-11-01}\n"
    )
    cfg = write_config(tmp_path, body)
    assert main(["run", "-c", str(cfg), "--yes"]) == 2
    assert "above the budget" in capsys.readouterr().err


def test_models_command(local_server: object, capsys: pytest.CaptureFixture[str]) -> None:
    base = local_server.base_url  # type: ignore[attr-defined]
    assert main(["models", "--base-url", base, "--api-key-env", "FAKE_API_KEY"]) == 0
    out = capsys.readouterr().out
    assert "local-a" in out and "false" in out
    assert main(["models", "--base-url", base, "--api-key-env", "MISSING_KEY_VAR"]) == 2
    assert main(["models", "--preset", "openai-compatible"]) == 2  # no base_url
    assert "base_url" in capsys.readouterr().err


def test_compare_and_report_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / "none.json")
    assert main(["report", missing]) == 2
    assert "cannot read" in capsys.readouterr().err
    assert main(["compare", missing, missing]) == 2


def test_synthetic_demo_template_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = TEMPLATES / "synthetic-demo.yaml"
    out = tmp_path / "out"
    assert (
        main(
            [
                "run",
                "-c",
                str(cfg),
                "--yes",
                "-q",
                "-o",
                str(out),
                "--workloads",
                "short-answer",
                "--repetitions",
                "1",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert text.startswith("SYNTHETIC FIXTURE DATA")
