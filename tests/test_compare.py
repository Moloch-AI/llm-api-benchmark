from __future__ import annotations

from typing import Any

import pytest

from llm_api_benchmark.records import RequestRecord
from llm_api_benchmark.reports.compare import (
    CompareError,
    bootstrap_median_diff,
    compare,
    comparison_json,
    comparison_markdown,
    comparison_text,
)
from llm_api_benchmark.results import RunResult


def make(
    name: str, latencies: dict[str, list[float]], dataset_sha: str = "abc", **config: Any
) -> RunResult:
    requests = []
    i = 0
    for provider, values in latencies.items():
        for v in values:
            requests.append(
                RequestRecord(
                    index=i,
                    provider=provider,
                    model="m",
                    workload="w",
                    prompt_id="p",
                    mode="stream",
                    repetition=0,
                    status="ok",
                    latency_s=v,
                    ttft_s=v / 4,
                    completion_tokens=100,
                )
            )
            i += 1
    return RunResult(
        run={"name": name, "started_at": "2026-11-17T10:00:00+00:00"},
        config={"modes": ["stream"], "temperature": None, **config},
        dataset={"id": "core", "version": "1.0.0", "sha256": dataset_sha, "workloads": {"w": {}}},
        providers=[{"name": p} for p in latencies],
        environment={"platform": "x"},
        requests=requests,
    )


FAST = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01]
SLOW = [2.0, 2.1, 1.9, 2.05, 1.95, 2.02, 1.98, 2.01]


def test_bootstrap_is_deterministic_and_brackets_the_difference() -> None:
    lo, hi = bootstrap_median_diff(FAST, SLOW)
    assert (lo, hi) == bootstrap_median_diff(FAST, SLOW)
    assert 0.8 < lo <= hi < 1.2


def test_clear_difference_between_two_runs() -> None:
    c = compare(make("a", {"x": FAST}), make("b", {"x": SLOW}))
    latency = next(r for r in c.rows if r.metric == "latency_s")
    assert latency.verdict == "B slower"
    assert latency.delta == pytest.approx(1.0, abs=0.05)
    tps = next(r for r in c.rows if r.metric == "output_tps")
    assert tps.verdict == "B lower"
    assert c.warnings == []


def test_overlapping_samples_give_no_clear_difference() -> None:
    noisy_a = [1.0, 3.0, 1.5, 2.5, 2.0, 1.2, 2.8]
    noisy_b = [1.1, 2.9, 1.6, 2.4, 2.1, 1.3, 2.7]
    c = compare(make("a", {"x": noisy_a}), make("b", {"x": noisy_b}))
    assert {r.verdict for r in c.rows} == {"no clear difference"}


def test_too_few_samples() -> None:
    c = compare(make("a", {"x": FAST[:3]}), make("b", {"x": SLOW[:3]}))
    assert {r.verdict for r in c.rows} == {"too few samples"}
    assert all(r.ci_low is None for r in c.rows)


def test_two_endpoints_in_one_run_and_name_matching() -> None:
    both = make("one", {"fast": FAST, "slow": SLOW})
    c = compare(both, both, provider_a="fast", provider_b="slow")
    assert next(r for r in c.rows if r.metric == "ttft_s").verdict == "B slower"
    with pytest.raises(CompareError, match="pick one endpoint"):
        compare(both, both, provider_a="fast")
    # Same names in both runs are matched by name.
    c2 = compare(make("a", {"fast": FAST, "slow": SLOW}), make("b", {"fast": SLOW, "slow": FAST}))
    assert {(r.a, r.verdict) for r in c2.rows if r.metric == "latency_s"} == {
        ("fast", "B slower"),
        ("slow", "B faster"),
    }
    with pytest.raises(CompareError, match="share no endpoint names"):
        compare(make("a", {"p": FAST, "q": FAST}), make("b", {"r": SLOW, "s": SLOW}))


def test_context_warnings_and_renderers() -> None:
    c = compare(make("a", {"x": FAST}, temperature=0), make("b", {"x": SLOW}, dataset_sha="def"))
    assert any("different datasets" in w for w in c.warnings)
    assert any("Different temperature" in w for w in c.warnings)
    assert "B slower" in comparison_text(c)
    assert "| w | stream | Latency (s) |" in comparison_markdown(c)
    assert '"verdict": "B slower"' in comparison_json(c)
