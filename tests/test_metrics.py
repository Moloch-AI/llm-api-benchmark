from __future__ import annotations

import math

import pytest

from llm_api_benchmark.metrics import describe, percentile, provider_totals, summarize
from llm_api_benchmark.records import RequestRecord


def rec(i: int, **kw: object) -> RequestRecord:
    base: dict[str, object] = {
        "index": i,
        "provider": "p",
        "model": "m",
        "workload": "w",
        "prompt_id": "a",
        "mode": "stream",
        "repetition": 0,
        "status": "ok",
    }
    base.update(kw)
    return RequestRecord(**base)  # type: ignore[arg-type]


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 2.5
    assert percentile(values, 90) == pytest.approx(3.7)
    assert percentile(values, 95) == pytest.approx(3.85)
    assert percentile(values, 100) == 4.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="between"):
        percentile([1.0], 101)


def test_describe_ignores_missing_and_non_finite() -> None:
    s = describe([3.0, None, 1.0, math.nan, 2.0, math.inf])
    assert s is not None
    assert (s.n, s.min, s.median, s.max, s.mean) == (3, 1.0, 2.0, 3.0, 2.0)
    assert describe([None, None]) is None


def test_throughput_definitions() -> None:
    r = rec(0, latency_s=2.0, ttft_s=0.5, last_token_s=1.5, completion_tokens=101)
    assert r.output_tps == pytest.approx(50.5)
    assert r.decode_tps == pytest.approx(100.0)  # (101 - 1) / (1.5 - 0.5)
    plain = rec(1, mode="plain", latency_s=2.0, completion_tokens=100)
    assert plain.output_tps == 50.0
    assert plain.decode_tps is None
    failed = rec(2, status="timeout", latency_s=2.0, completion_tokens=10)
    assert failed.output_tps is None
    single = rec(3, latency_s=1.0, ttft_s=0.5, last_token_s=0.5, completion_tokens=1)
    assert single.decode_tps is None


def test_summarize_excludes_warmup_and_counts_errors_over_executed_requests() -> None:
    records = [
        rec(0, warmup=True, latency_s=9.0),
        rec(1, latency_s=1.0, ttft_s=0.2, completion_tokens=10, cost=0.001, checks_passed=True),
        rec(2, latency_s=3.0, ttft_s=0.4, completion_tokens=30, cost=0.002, checks_passed=False),
        rec(3, status="http_error", http_status=503, latency_s=0.1),
        rec(4, status="skipped", skip_reason="budget"),
        rec(5, status="stream_error", error_type="preempted", latency_s=2.0),
    ]
    [s] = summarize(records)
    assert (s.planned, s.executed, s.ok) == (5, 4, 2)
    assert s.errors == {"http_503": 1, "stream_error:preempted": 1}
    assert s.error_rate == 0.5
    assert s.latency is not None and s.latency.median == 2.0 and s.latency.n == 2
    assert s.ttft is not None and s.ttft.median == pytest.approx(0.3)
    assert s.cost == pytest.approx(0.003)
    assert s.cost_complete is True  # every successful request has a cost
    assert (s.checks_run, s.checks_passed) == (2, 1)


def test_cost_incomplete_when_a_success_has_no_cost() -> None:
    [s] = summarize([rec(0, latency_s=1.0, cost=0.01), rec(1, latency_s=1.0)])
    assert s.cost == pytest.approx(0.01)
    assert s.cost_complete is False


def test_provider_totals_include_warmup_and_skips() -> None:
    records = [
        rec(0, warmup=True, latency_s=1.0),
        rec(1, status="skipped"),
        rec(2, status="timeout"),
    ]
    [t] = provider_totals(records)
    assert (t.planned, t.executed, t.ok, t.skipped) == (3, 2, 1, 1)
    assert t.errors == {"timeout": 1}
    assert t.error_rate == 0.5
