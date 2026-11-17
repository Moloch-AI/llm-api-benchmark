"""Planning and running: order, bounds, pacing, stopping rules, budget, cancellation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from tests.helpers import (
    Recorder,
    config,
    default_respond,
    ok_stream_response,
    provider_dict,
    tiny_dataset,
)

from llm_api_benchmark.config import RunConfig
from llm_api_benchmark.datasets import Dataset, load_dataset
from llm_api_benchmark.records import RequestRecord
from llm_api_benchmark.runner import Runner, build_plan, missing_keys

PRICING = {
    "input_per_mtok": 1.0,
    "output_per_mtok": 10.0,
    "currency": "USD",
    "source": "https://example.com/pricing",
    "checked_on": "2026-11-01",
}


class FakeTime:
    """A clock that only moves when the runner sleeps (for pacing tests)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds
        await asyncio.sleep(0)


def run(
    cfg: RunConfig,
    dataset: Dataset,
    recorder: Recorder,
    *,
    fake_time: FakeTime | None = None,
    progress: list[RequestRecord] | None = None,
) -> Runner:
    plan = build_plan(cfg, dataset)
    kwargs: dict[str, Any] = {}
    if fake_time:
        kwargs = {"clock": fake_time.clock, "sleep": fake_time.sleep}
    runner = Runner(
        cfg,
        dataset,
        plan,
        transport_factory=lambda p: recorder.transport(),
        progress=(lambda r, d, t: progress.append(r)) if progress is not None else None,
        **kwargs,
    )
    asyncio.run(runner.run())
    return runner


def test_plan_size_order_and_warmup() -> None:
    ds = load_dataset("core-v1")
    cfg = config(
        [provider_dict(name="a"), provider_dict(name="b")],
        workloads=["short-answer", "reasoning"],
        modes=["stream", "plain"],
        repetitions=2,
        warmup=1,
        seed=7,
    )
    plan = build_plan(cfg, ds)
    assert (
        len(plan) == 2 * 1 + 2 * 2 * 3 * 2 * 2
    )  # warm-ups + providers x workloads x prompts x modes x reps
    assert [r.warmup for r in plan.requests[:2]] == [True, True]
    assert plan.by_provider() == {"a": 25, "b": 25}
    again = build_plan(cfg, ds)
    key = [(r.provider.name, r.workload.id, r.prompt.id, r.mode) for r in plan.requests]
    assert key == [(r.provider.name, r.workload.id, r.prompt.id, r.mode) for r in again.requests]
    other_seed = build_plan(
        config(
            [provider_dict(name="a"), provider_dict(name="b")],
            workloads=["short-answer", "reasoning"],
            modes=["stream", "plain"],
            repetitions=2,
            warmup=1,
            seed=8,
        ),
        ds,
    )
    assert key != [
        (r.provider.name, r.workload.id, r.prompt.id, r.mode) for r in other_seed.requests
    ]
    # Each repetition block contains every combination exactly once.
    block = plan.requests[2:26]
    assert len({(r.provider.name, r.workload.id, r.prompt.id, r.mode) for r in block}) == 24


def test_worst_case_cost_and_unbounded_output() -> None:
    ds = tiny_dataset(prompts=1)
    cfg = config([provider_dict(pricing=PRICING), provider_dict(name="free")])
    plan = build_plan(cfg, ds)
    cost, unpriced = plan.worst_case_cost(cfg)
    r = plan.requests[0]
    assert unpriced == ["free"]
    assert cost == pytest.approx((r.estimated_prompt_tokens() * 1.0 + 16 * 10.0) / 1e6)
    moloch = config([{"name": "m", "preset": "molochai", "model": "x"}])
    mplan = build_plan(moloch, ds)
    assert mplan.unbounded_output(moloch) == ["m"]
    assert mplan.requests[0].worst_output_tokens(moloch) == 9000


def test_successful_run_records_everything() -> None:
    ds = tiny_dataset(prompts=2, checks=[{"type": "contains", "value": "hello"}])
    cfg = config([provider_dict(pricing=PRICING)], repetitions=2, warmup=1, save_outputs=True)
    rec = Recorder(default_respond)
    seen: list[RequestRecord] = []
    runner = run(cfg, ds, rec, progress=seen)
    records = runner.ordered_records()
    assert len(records) == len(rec.requests) == 5
    assert len(seen) == 5
    assert all(r.status == "ok" for r in records)
    assert records[0].warmup
    assert all(r.cost == pytest.approx((7 * 1.0 + 3 * 10.0) / 1e6) for r in records)
    assert runner.budget.spent == pytest.approx(5 * (7 + 30) / 1e6)
    assert all(r.checks_passed is True for r in records if not r.warmup)
    assert all(r.output == "Hello." for r in records)
    assert all(r.started_at for r in records)


def test_concurrency_limit_is_respected() -> None:
    ds = tiny_dataset(prompts=4)
    cfg = config(concurrency=3, repetitions=3)
    rec = Recorder(default_respond)
    run(cfg, ds, rec)
    assert len(rec.requests) == 12
    assert rec.max_in_flight <= 3


def test_rate_limit_spacing() -> None:
    ds = tiny_dataset(prompts=3)
    cfg = config([provider_dict(rate_limit_rpm=30)])
    ft = FakeTime()
    run(cfg, ds, Recorder(default_respond), fake_time=ft)
    assert ft.sleeps == [2.0, 2.0]  # 60 / 30 seconds between request starts


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_permanent_errors_stop_the_endpoint(status: int) -> None:
    ds = tiny_dataset(prompts=3)
    cfg = config(
        [provider_dict(name="bad"), provider_dict(name="good", base_url="https://good.test/v1")],
        repetitions=2,
    )

    def respond(request: httpx.Request, n: int) -> httpx.Response:
        if request.url.host == "fake.test":
            return httpx.Response(status, json={"message": "nope"})
        return default_respond(request, n)

    runner = run(cfg, ds, Recorder(respond))
    bad = [r for r in runner.ordered_records() if r.provider == "bad"]
    good = [r for r in runner.ordered_records() if r.provider == "good"]
    assert sum(1 for r in bad if r.status == "http_error") == 1
    assert all(
        r.status == "skipped" and "stopped after HTTP" in (r.skip_reason or "")
        for r in bad
        if r.status != "http_error"
    )
    assert all(r.status == "ok" for r in good)


def test_quota_429_with_should_retry_false_stops_but_rate_limit_429_waits() -> None:
    ds = tiny_dataset(prompts=3)

    def quota(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"X-Should-Retry": "false"},
            json={"error": {"type": "quota_exceeded", "message": "used"}},
        )

    runner = run(config(), ds, Recorder(quota))
    statuses = [r.status for r in runner.ordered_records()]
    assert statuses == ["http_error", "skipped", "skipped"]

    def throttled_once(request: httpx.Request, n: int) -> httpx.Response:
        if n == 0:
            return httpx.Response(
                429, headers={"Retry-After": "3"}, json={"message": "Too Many Attempts."}
            )
        return default_respond(request, n)

    ft = FakeTime()
    runner2 = run(config(), ds, Recorder(throttled_once), fake_time=ft)
    assert [r.status for r in runner2.ordered_records()] == ["http_error", "ok", "ok"]
    assert ft.sleeps == [3.0]


def test_long_retry_after_stops_the_endpoint() -> None:
    def respond(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "3600"}, json={"message": "Too Many Attempts."}
        )

    runner = run(config(), tiny_dataset(prompts=2), Recorder(respond))
    assert [r.status for r in runner.ordered_records()] == ["http_error", "skipped"]


def test_consecutive_failures_stop_the_endpoint_with_backoff() -> None:
    def respond(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(503, json={"error": {"type": "server_busy", "message": "busy"}})

    ft = FakeTime()
    cfg = config(max_consecutive_failures=3, repetitions=2)
    runner = run(cfg, tiny_dataset(prompts=3), Recorder(respond), fake_time=ft)
    statuses = [r.status for r in runner.ordered_records()]
    assert statuses == ["http_error"] * 3 + ["skipped"] * 3
    assert ft.sleeps == [1.0, 2.0]  # backoff between failures; the third stops the endpoint


def test_budget_stops_the_run_before_it_is_exceeded() -> None:
    ds = tiny_dataset(prompts=5)
    # Worst case per request is about (prompt estimate + 16 output tokens x 10 USD/M); allow ~2.
    one = build_plan(config([provider_dict(pricing=PRICING)]), ds).requests[0]
    worst = (one.estimated_prompt_tokens() * 1.0 + 16 * 10.0) / 1e6
    cfg = config([provider_dict(pricing=PRICING)], budget=worst * 2.5)
    # Each reply costs (7 x 1 + 15 x 10) / 1M; the third leaves too little room for a fourth.
    rec = Recorder(lambda request, n: ok_stream_response(completion=15))
    runner = run(cfg, ds, rec)
    statuses = [r.status for r in runner.ordered_records()]
    assert statuses.count("ok") == 3
    assert runner.budget.spent <= cfg.budget  # type: ignore[operator]
    assert "skipped" in statuses
    assert runner.stop_reason is not None and runner.stop_reason.startswith("budget")


def test_cancellation_keeps_partial_results() -> None:
    ds = tiny_dataset(prompts=5)
    cfg = config()

    calls = []

    async def slow(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 3:
            await asyncio.sleep(10)
        return ok_stream_response()

    plan = build_plan(cfg, ds)
    runner = Runner(cfg, ds, plan, transport_factory=lambda p: httpx.MockTransport(slow))

    async def go() -> None:
        task = asyncio.create_task(runner.run())
        while sum(1 for r in runner.records.values() if r.status == "ok") < 2:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    records = runner.ordered_records()
    assert runner.interrupted
    assert len(records) == 5
    statuses = sorted(r.status for r in records)
    assert statuses.count("ok") == 2
    assert statuses.count("cancelled") == 1
    assert statuses.count("skipped") == 2
    assert all(r.skip_reason for r in records if r.status == "skipped")


def test_missing_keys_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKE_API_KEY")
    cfg = config(
        [
            provider_dict(headers_env={"X-Org": "FAKE_ORG"}),
            {
                "name": "local",
                "base_url": "http://localhost:1/v1",
                "auth_scheme": "none",
                "model": "m",
            },
        ]
    )
    assert missing_keys(cfg) == [
        "fake: environment variable FAKE_API_KEY is not set",
        "fake: environment variable FAKE_ORG (header X-Org) is not set",
    ]


def test_synthetic_endpoint_runs_offline() -> None:
    cfg = config(
        [
            {
                "name": "sim",
                "preset": "synthetic",
                "model": "s",
                "synthetic": {"ttft_ms": 20, "tokens_per_s": 2000, "output_tokens": 20},
            }
        ],
        modes=["stream", "plain"],
    )
    ds = tiny_dataset(prompts=2)
    plan = build_plan(cfg, ds)
    runner = Runner(cfg, ds, plan)
    asyncio.run(runner.run())
    records = runner.ordered_records()
    assert [r.status for r in records] == ["ok"] * 4
    assert all((r.completion_tokens or 0) > 0 for r in records)
    assert all(r.ttft_s is not None for r in records if r.mode == "stream")


def test_warmups_finish_before_measured_requests_start() -> None:
    ds = tiny_dataset(prompts=3)
    cfg = config(
        [provider_dict(name="a"), provider_dict(name="b", base_url="https://b.test/v1")],
        concurrency=4,
        warmup=1,
    )
    arrivals: list[int] = []
    finished = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal finished
        arrivals.append(finished)
        if len(arrivals) <= 2:  # the two warm-ups are slow
            await asyncio.sleep(0.2)
        finished += 1
        return default_respond(request, 0)

    runner = Runner(
        cfg, ds, build_plan(cfg, ds), transport_factory=lambda p: httpx.MockTransport(handler)
    )
    asyncio.run(runner.run())
    assert arrivals[:2] == [0, 0]  # both warm-ups ran together
    assert min(arrivals[2:]) == 2  # every measured request started after both had finished
    assert all(r.status == "ok" for r in runner.ordered_records())
