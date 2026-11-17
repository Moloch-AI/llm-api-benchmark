"""Result files and every report format, built from a deterministic scripted run."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
from pathlib import Path

import httpx
import pytest
from tests.helpers import Recorder, config, default_respond, provider_dict, tiny_dataset

from llm_api_benchmark.reports import html_report, markdown_report, terminal_report
from llm_api_benchmark.results import (
    ResultsError,
    RunResult,
    build_result,
    load_results,
    requests_csv,
    summary_csv,
    to_json,
)
from llm_api_benchmark.runner import Runner, build_plan

PRICING = {
    "input_per_mtok": 1.0,
    "output_per_mtok": 2.0,
    "source": "https://example.com/pricing",
    "checked_on": "2026-11-01",
}


def scripted_result(synthetic: bool = False) -> RunResult:
    def respond(request: httpx.Request, n: int) -> httpx.Response:
        if n == 3:
            return httpx.Response(
                503, json={"error": {"type": "server_busy", "message": "No slot free."}}
            )
        return default_respond(request, n)

    providers = [
        provider_dict(pricing=PRICING),
        provider_dict(name="other", base_url="https://other.test/v1"),
    ]
    if synthetic:
        providers.append(
            {
                "name": "sim",
                "preset": "synthetic",
                "model": "s",
                "synthetic": {"ttft_ms": 5, "tokens_per_s": 5000},
            }
        )
    cfg = config(providers, modes=["stream", "plain"], repetitions=2, warmup=1)
    ds = tiny_dataset(prompts=2, checks=[{"type": "contains", "value": "hello"}])
    rec = Recorder(respond)
    runner = Runner(
        cfg,
        ds,
        build_plan(cfg, ds),
        transport_factory=lambda p: None if p.is_synthetic else rec.transport(),
    )
    asyncio.run(runner.run())
    return build_result(runner, ["example warning"])


def test_json_round_trip(tmp_path: Path) -> None:
    result = scripted_result()
    path = tmp_path / "r.json"
    path.write_text(to_json(result), encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "llm-api-benchmark/results/v1"
    assert data["tool"]["name"] == "llm-api-benchmark"
    assert data["dataset"]["id"] == "tiny"
    assert data["run"]["synthetic"] is False
    assert len(data["requests"]) == result.run["planned_requests"] == 2 + 2 * 2 * 2 * 2
    assert data["limitations"]
    loaded = load_results(path)
    assert [s.to_dict() for s in loaded.summaries()] == [s.to_dict() for s in result.summaries()]
    assert loaded.warnings == ["example warning"]


def test_load_rejects_other_files(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ResultsError, match="not an llm-api-benchmark results file"):
        load_results(p)
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(ResultsError, match="not valid JSON"):
        load_results(p)


def test_summary_order_follows_dataset_endpoints_and_modes() -> None:
    result = scripted_result()
    keys = [(s.workload, s.provider, s.mode) for s in result.summaries()]
    assert keys == [
        ("w1", "fake", "stream"),
        ("w1", "fake", "plain"),
        ("w1", "other", "stream"),
        ("w1", "other", "plain"),
    ]


def test_csv_outputs() -> None:
    result = scripted_result()
    rows = list(csv.DictReader(io.StringIO(requests_csv(result))))
    assert len(rows) == len(result.requests)
    assert {r["status"] for r in rows} == {"ok", "http_error"}
    failed = next(r for r in rows if r["status"] == "http_error")
    assert failed["error_kind"] == "http_503"
    assert failed["error_type"] == "server_busy"
    summary = list(csv.DictReader(io.StringIO(summary_csv(result))))
    assert len(summary) == 4
    assert {"latency_median_s", "ttft_p95_s", "error_rate", "cost"} <= set(summary[0])
    priced = next(r for r in summary if r["provider"] == "fake" and r["mode"] == "stream")
    assert float(priced["cost"]) > 0


def test_terminal_and_markdown_reports() -> None:
    result = scripted_result()
    text = terminal_report(result)
    assert "SYNTHETIC" not in text
    assert "w1  (max_tokens 16, 2 prompts)" in text
    assert "Reliability" in text
    md = markdown_report(result)
    for heading in (
        "## Run",
        "## Endpoints",
        "## Workloads",
        "## Results by workload",
        "### w1",
        "## Reliability",
        "## Limitations",
    ):
        assert heading in md
    assert "server_busy" in md  # error example table
    assert "https://example.com/pricing" in md
    assert "example warning" in md


def test_synthetic_runs_are_labeled_everywhere() -> None:
    result = scripted_result(synthetic=True)
    assert result.synthetic
    assert terminal_report(result).startswith("SYNTHETIC FIXTURE DATA")
    assert "Synthetic fixture data" in markdown_report(result)
    assert "Synthetic fixture data" in html_report(result)
    assert result.limitations()[0].startswith("SYNTHETIC FIXTURE DATA")


def test_html_report_is_self_contained() -> None:
    page = html_report(scripted_result())
    assert page.startswith("<!doctype html>")
    assert "<svg" in page
    assert "prefers-color-scheme:dark" in page
    assert not re.search(
        r"<(script|link|img)[^>]+(src|href)=", page
    )  # nothing loaded from elsewhere
    assert page.count('class="row"') >= 4
