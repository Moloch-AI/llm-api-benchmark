"""The molochai preset against the official MolochAI mock server.

The mock (https://github.com/Moloch-AI/uncensored-ai-api, mock-server/) implements the gateway's
documented behavior: reasoning deltas, keep-alive and queue comments, the usage chunk, in-stream
errors and the error shapes. Replies are canned text, so only protocol handling is tested here,
never speed. Runs when MOLOCHAI_MOCK_BASE_URL is set, e.g.:

    node mock-server/server.mjs --port 4010 --quiet --rate-limit 600
    MOLOCHAI_MOCK_BASE_URL=http://127.0.0.1:4010/api/v1 pytest tests/integration
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from llm_api_benchmark import conformance as conf
from llm_api_benchmark.cli import main
from llm_api_benchmark.config import parse_config
from llm_api_benchmark.datasets import load_dataset, parse_dataset
from llm_api_benchmark.records import RequestRecord
from llm_api_benchmark.runner import Runner, build_plan

MOCK = os.environ.get("MOLOCHAI_MOCK_BASE_URL", "")
ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not MOCK, reason="set MOLOCHAI_MOCK_BASE_URL to run against the mock server"
)


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOLOCHAI_BASE_URL", MOCK)
    monkeypatch.setenv("MOLOCHAI_API_KEY", "mock-key")


def run_prompts(prompts: dict[str, str], model: str, modes: list[str]) -> list[RequestRecord]:
    cfg = parse_config(
        {
            "name": "mock",
            "repetitions": 1,
            "warmup": 0,
            "modes": modes,
            "max_consecutive_failures": 50,
            "providers": [
                {"name": "moloch", "preset": "molochai", "model": model, "rate_limit_rpm": 600}
            ],
        }
    )
    ds = parse_dataset(
        {
            "id": "mock",
            "version": "1.0.0",
            "workloads": [
                {
                    "id": "w",
                    "max_tokens": 64,
                    "prompts": [{"id": k, "prompt": v} for k, v in prompts.items()],
                }
            ],
        },
        source="mock",
    )
    runner = Runner(cfg, ds, build_plan(cfg, ds))
    asyncio.run(runner.run())
    return runner.ordered_records()


def by_prompt(records: list[RequestRecord], mode: str) -> dict[str, RequestRecord]:
    return {r.prompt_id: r for r in records if r.mode == mode}


def test_template_config_runs_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    cfg = ROOT / "benchmarks" / "molochai.yaml"
    assert main(["run", "-c", str(cfg), "-q", "-o", str(out), "--repetitions", "1"]) == 0
    result = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    requests = result["requests"]
    assert len(requests) == 1 + 3 * 3
    assert {r["status"] for r in requests} == {"ok"}
    assert {r["response_model"] for r in requests} == {"llama-3.3-70b-instruct-abliterated"}
    assert all(r["usage_reported"] and not r["tokens_estimated"] for r in requests)
    assert all(r["done_marker"] for r in requests)
    assert "short-answer" in capsys.readouterr().out


def test_reasoning_model_separates_ttft_and_answer() -> None:
    records = run_prompts({"hello": "Say hello."}, "glm-5.3-flash", ["stream", "plain"])
    stream, plain = by_prompt(records, "stream")["hello"], by_prompt(records, "plain")["hello"]
    assert stream.status == "ok" and plain.status == "ok"
    assert stream.reasoning_chars > 0 and plain.reasoning_chars > 0
    assert stream.ttft_s is not None and stream.ttfc_s is not None
    assert stream.ttft_s <= stream.ttfc_s
    assert stream.comment_lines >= 1  # the mock sends ": ka" between reasoning and answer
    assert stream.completion_tokens == plain.completion_tokens


def test_gateway_behaviors_are_classified() -> None:
    records = run_prompts(
        {
            "queue": "[mock:queue] hello",
            "timeout": "[mock:timeout] hello",
            "preempted": "[mock:preempted] hello",
            "busy": "[mock:busy] hello",
            "unavailable": "[mock:unavailable] hello",
            "context": "[mock:context] hello",
            "cap": "[mock:cap] hello",
            "edge": "[mock:cloudflare-524] hello",
        },
        "glm-5.3-flash",
        ["stream", "plain"],
    )
    s, p = by_prompt(records, "stream"), by_prompt(records, "plain")
    assert s["queue"].status == "ok" and s["queue"].comment_lines >= 3
    assert (s["timeout"].status, s["timeout"].error_type) == ("stream_error", "upstream_timeout")
    assert (p["timeout"].http_status, p["timeout"].error_type) == (504, "upstream_timeout")
    assert (s["preempted"].status, s["preempted"].error_type) == ("stream_error", "preempted")
    assert (s["busy"].http_status, s["busy"].error_type) == (503, "server_busy")
    assert (s["unavailable"].http_status, s["unavailable"].error_type) == (503, "model_unavailable")
    assert (s["context"].http_status, s["context"].should_retry) == (413, "false")
    assert s["cap"].finish_reason == "length" and s["cap"].status == "ok"
    assert p["edge"].http_status == 524 and (p["edge"].error_message or "").startswith("(HTML)")


def test_quota_error_stops_the_endpoint() -> None:
    records = run_prompts(
        {"a": "[mock:quota] hello", "b": "hello", "c": "hello"}, "glm-5.3-flash", ["stream"]
    )
    statuses = [r.status for r in records]
    assert statuses[0] == "http_error"
    assert records[0].should_retry == "false"
    assert statuses[1:] == ["skipped", "skipped"]


def test_invalid_key_stops_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOLOCHAI_API_KEY", "mock-no-plan")
    records = run_prompts({"a": "hello", "b": "hello"}, "glm-5.3-flash", ["stream"])
    assert (records[0].http_status, records[0].error_type) == (402, "no_active_plan")
    assert records[1].status == "skipped"


def test_tool_calls_arrive_whole_and_are_recorded() -> None:
    cfg = parse_config(
        {
            "name": "mock-tools",
            "dataset": "tools-v1",
            "workloads": ["tool-select"],
            "repetitions": 2,
            "warmup": 0,
            "modes": ["stream", "plain"],
            "providers": [
                {
                    "name": "moloch",
                    "preset": "molochai",
                    "model": "llama-3.3-70b-instruct-abliterated",
                    "rate_limit_rpm": 600,
                }
            ],
        }
    )
    ds = load_dataset("tools-v1")
    runner = Runner(cfg, ds, build_plan(cfg, ds))
    asyncio.run(runner.run())
    records = runner.ordered_records()
    assert len(records) == 3 * 2 * 2
    assert {r.status for r in records} == {"ok"}
    # The mock, like the gateway, calls the first tool and sends its arguments in one chunk.
    assert {tuple(r.tool_call_names or []) for r in records} == {("get_weather",)}
    assert {r.finish_reason for r in records} == {"tool_calls"}
    stream = [r for r in records if r.mode == "stream"]
    assert all(r.ttfc_s is not None and r.text_chunks == 1 for r in stream)
    assert len({r.tool_signature for r in records}) == 1


def test_conformance_against_the_mock_matches_the_documented_gateway() -> None:
    """The mock mirrors the documented gateway: ignored parameters fail, 422s are not OpenAI-shaped.

    Only outcomes that follow from the documented contract are asserted. The mock accepts any key
    and calls the first tool whenever tools are sent, so those tests are not asserted here.
    """
    cfg = parse_config(
        {
            "name": "mock-conformance",
            "providers": [
                {
                    "name": "moloch",
                    "preset": "molochai",
                    "model": "llama-3.3-70b-instruct-abliterated",
                    "rate_limit_rpm": 600,
                }
            ],
        }
    )
    report = asyncio.run(
        conf.run_provider(cfg.providers[0], list(conf.TESTS), timeout_s=30, connect_timeout_s=5)
    )
    got = {r.id: r.outcome for r in report.results}
    assert got["models.list"] == conf.PASS
    assert got["chat.basic"] == got["chat.usage"] == conf.PASS
    assert got["stream.basic"] == got["stream.usage"] == conf.PASS  # usage is always sent
    for ignored in (
        "params.max_tokens",
        "params.stop",
        "params.n",
        "params.response_format.json_schema",
        "params.logprobs",
    ):
        assert got[ignored] == conf.FAIL, ignored
    assert got["params.temperature_range"] == conf.PASS  # 422
    assert got["tools.call"] == got["tools.stream"] == conf.PASS
    assert got["errors.unknown_model"] == conf.WARN  # 422 with {"message", "errors"}
    assert got["errors.missing_key"] == conf.WARN  # 401 {"message": "Unauthenticated."}
