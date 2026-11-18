"""Plan and run a bounded benchmark.

The plan is fixed before the first request: every (endpoint, workload, prompt, mode) combination,
repeated ``repetitions`` times, in an order shuffled per repetition with a fixed seed so endpoints
are interleaved over time instead of measured one after another. Warm-up requests run first and
are excluded from statistics.

While running, the runner never sends more than the plan, never exceeds ``concurrency`` requests
in flight, spaces requests to each endpoint by its ``rate_limit_rpm``, honors ``Retry-After``,
stops sending to an endpoint after a permanent error (401, 402, 403, 404, or a 429 with
``X-Should-Retry: false``) or after ``max_consecutive_failures`` failures in a row, and stops the
whole run before a request could push spending past the budget. Requests are never retried: each
attempt is one sample.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from ._version import __version__
from .checks import run_checks
from .config import ProviderConfig, RunConfig, SyntheticProfile
from .datasets import Dataset, Prompt, Workload
from .measure import estimate_tokens, execute
from .pricing import request_cost, worst_case_cost
from .providers import ADAPTERS, Adapter
from .records import (
    CONNECTION_ERROR,
    HTTP_ERROR,
    SKIPPED,
    STREAM_ERROR,
    TIMEOUT,
    RequestRecord,
)
from .redact import REDACTOR, redact

PERMANENT_STATUSES = (401, 402, 403, 404)
MAX_RETRY_AFTER_WAIT_S = 300.0
# Prompt size estimate for budgets: characters / 4, plus a margin for chat formatting and any
# system text the provider adds.
_PROMPT_MARGIN = 1.25
_PROMPT_OVERHEAD = 64
_MAX_SAVED_ARGS = 4000

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
Progress = Callable[[RequestRecord, int, int], None]
TransportFactory = Callable[[ProviderConfig], httpx.AsyncBaseTransport | None]


@dataclass(frozen=True)
class PlannedRequest:
    index: int
    provider: ProviderConfig
    workload: Workload
    prompt: Prompt
    mode: str
    repetition: int
    warmup: bool

    def max_tokens(self, cfg: RunConfig) -> int:
        return cfg.max_tokens or self.workload.max_tokens

    def worst_output_tokens(self, cfg: RunConfig) -> int:
        """Upper bound on output tokens used for budget estimates."""
        p = self.provider
        limit = self.max_tokens(cfg)
        if p.max_tokens_honored and p.max_tokens_param != "none":
            return limit
        return max(limit, p.assumed_max_output_tokens or limit)

    def estimated_prompt_tokens(self) -> int:
        chars = self.prompt.chars() + self.workload.tools_chars()
        return math.ceil(estimate_tokens(chars) * _PROMPT_MARGIN) + _PROMPT_OVERHEAD


@dataclass
class Plan:
    requests: list[PlannedRequest]

    def __len__(self) -> int:
        return len(self.requests)

    def by_provider(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.requests:
            out[r.provider.name] = out.get(r.provider.name, 0) + 1
        return out

    def worst_case_cost(self, cfg: RunConfig) -> tuple[float, list[str]]:
        """Total worst-case cost over priced endpoints, and the names of unpriced ones."""
        total = 0.0
        unpriced: set[str] = set()
        for r in self.requests:
            if r.provider.pricing is None:
                if not r.provider.is_synthetic:
                    unpriced.add(r.provider.name)
                continue
            total += worst_case_cost(
                r.provider.pricing, r.estimated_prompt_tokens(), r.worst_output_tokens(cfg)
            )
        return total, sorted(unpriced)

    def worst_case_output_tokens(self, cfg: RunConfig) -> int:
        return sum(r.worst_output_tokens(cfg) for r in self.requests)

    def unbounded_output(self, cfg: RunConfig) -> list[str]:
        """Endpoints where output length is not bounded by max_tokens."""
        return sorted(
            {
                r.provider.name
                for r in self.requests
                if not r.provider.max_tokens_honored or r.provider.max_tokens_param == "none"
            }
        )


def build_plan(cfg: RunConfig, dataset: Dataset) -> Plan:
    workloads = dataset.select(cfg.workloads)
    items: list[tuple[ProviderConfig, Workload, Prompt, str, int, bool]] = []
    first_wl = workloads[0]
    for p in cfg.providers:
        for w in range(cfg.warmup):
            items.append((p, first_wl, first_wl.prompts[0], cfg.modes[0], w, True))
    rng = random.Random(cfg.seed)
    for rep in range(cfg.repetitions):
        block = [
            (p, wl, pr, mode, rep, False)
            for p in cfg.providers
            for wl in workloads
            for pr in wl.prompts
            for mode in cfg.modes
        ]
        rng.shuffle(block)
        items.extend(block)
    return Plan([PlannedRequest(i, *item) for i, item in enumerate(items)])


class Budget:
    """Reserve the worst case before sending, settle with the actual cost afterwards."""

    def __init__(self, limit: float | None) -> None:
        self.limit = limit
        self.spent = 0.0
        self.reserved = 0.0

    def reserve(self, amount: float) -> bool:
        if self.limit is not None and self.spent + self.reserved + amount > self.limit + 1e-12:
            return False
        self.reserved += amount
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved = max(0.0, self.reserved - reserved)
        self.spent += actual


@dataclass
class _EndpointState:
    rpm: float | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_start: float = 0.0
    paused_until: float = 0.0
    consecutive_failures: int = 0
    aborted: str | None = None

    async def wait_turn(self, clock: Clock, sleep: Sleep) -> None:
        async with self.lock:
            start = max(self.next_start, self.paused_until)
            delay = start - clock()
            if delay > 0:
                await sleep(delay)
            if self.rpm:
                self.next_start = clock() + 60.0 / self.rpm


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        dataset: Dataset,
        plan: Plan,
        *,
        transport_factory: TransportFactory | None = None,
        progress: Progress | None = None,
        clock: Clock = time.perf_counter,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.plan = plan
        self.records: dict[int, RequestRecord] = {}
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.stop_reason: str | None = None
        self.interrupted = False
        self.budget = Budget(cfg.budget)
        self._transport_factory = transport_factory
        self._progress = progress
        self._clock = clock
        self._sleep = sleep
        self._done = 0
        self._state = {p.name: _EndpointState(p.rate_limit_rpm) for p in cfg.providers}

    # ------------------------------------------------------------------ public

    async def run(self) -> None:
        self.started_at = datetime.now(UTC)
        # Warm-ups finish before any measured request starts, even with concurrency above 1.
        phases = [
            deque(r for r in self.plan.requests if r.warmup),
            deque(r for r in self.plan.requests if not r.warmup),
        ]
        try:
            async with contextlib.AsyncExitStack() as stack:
                clients: dict[str, tuple[httpx.AsyncClient, Adapter]] = {}
                for p in self.cfg.providers:
                    client = await stack.enter_async_context(self._client(p))
                    clients[p.name] = (client, self._adapter(p))
                for queue in phases:
                    workers = min(self.cfg.concurrency, len(queue))
                    async with asyncio.TaskGroup() as tg:
                        for _ in range(workers):
                            tg.create_task(self._worker(queue, clients))
        except asyncio.CancelledError:
            self.interrupted = True
            raise
        finally:
            self.finished_at = datetime.now(UTC)
            self._mark_unstarted()

    def ordered_records(self) -> list[RequestRecord]:
        self._mark_unstarted()
        return [self.records[i] for i in sorted(self.records)]

    # ------------------------------------------------------------------ internals

    def _client(self, p: ProviderConfig) -> httpx.AsyncClient:
        transport: httpx.AsyncBaseTransport | None = None
        if self._transport_factory:
            transport = self._transport_factory(p)
        if transport is None and p.is_synthetic:
            from .providers.synthetic import synthetic_transport

            transport = synthetic_transport(p.synthetic or SyntheticProfile(), self.cfg.seed)
        timeout = p.timeout_s or self.cfg.timeout_s
        return httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=self.cfg.connect_timeout_s),
            limits=httpx.Limits(
                max_connections=self.cfg.concurrency, max_keepalive_connections=self.cfg.concurrency
            ),
            follow_redirects=False,
            headers={"User-Agent": f"llm-api-benchmark/{__version__}"},
            trust_env=True,
        )

    def _adapter(self, p: ProviderConfig) -> Adapter:
        key = p.api_key() if p.needs_key else None
        REDACTOR.add(key)
        extra: dict[str, str] = {}
        for header, env in p.headers_env.items():
            value = os.environ.get(env, "").strip()
            if value:
                REDACTOR.add(value)
                extra[header] = value
        return ADAPTERS[p.api](p, key, extra)

    async def _worker(
        self,
        queue: deque[PlannedRequest],
        clients: dict[str, tuple[httpx.AsyncClient, Adapter]],
    ) -> None:
        while queue:
            planned = queue.popleft()
            client, adapter = clients[planned.provider.name]
            await self._one(planned, client, adapter)

    async def _one(
        self, planned: PlannedRequest, client: httpx.AsyncClient, adapter: Adapter
    ) -> None:
        p = planned.provider
        rec = RequestRecord(
            index=planned.index,
            provider=p.name,
            model=p.model,
            workload=planned.workload.id,
            prompt_id=planned.prompt.id,
            mode=planned.mode,
            repetition=planned.repetition,
            warmup=planned.warmup,
        )
        self.records[planned.index] = rec
        state = self._state[p.name]
        if self.stop_reason:
            return self._skip(rec, self.stop_reason)
        if state.aborted:
            return self._skip(rec, state.aborted)

        reserved = 0.0
        if p.pricing is not None:
            reserved = worst_case_cost(
                p.pricing, planned.estimated_prompt_tokens(), planned.worst_output_tokens(self.cfg)
            )
        if not self.budget.reserve(reserved):
            self.stop_reason = (
                f"budget: the next request could exceed the {self.cfg.budget:g} "
                f"{self.cfg.budget_currency} budget"
            )
            return self._skip(rec, self.stop_reason)

        actual = 0.0
        try:
            try:
                await state.wait_turn(self._clock, self._sleep)
            except asyncio.CancelledError:
                rec.skip_reason = "interrupted before it started"
                raise
            if self.stop_reason or state.aborted:
                rec.status = SKIPPED
                rec.skip_reason = self.stop_reason or state.aborted
                return None
            rec.started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            workload = planned.workload
            choice = planned.prompt.tool_choice
            request = adapter.build_request(
                planned.prompt.message_list(),
                stream=planned.mode == "stream",
                max_tokens=planned.max_tokens(self.cfg),
                temperature=self.cfg.temperature,
                tools=[dict(t) for t in workload.tools] or None,
                tool_choice=choice if choice is not None else workload.tool_choice,
            )
            reply = await execute(
                client,
                adapter,
                request,
                rec,
                stream=planned.mode == "stream",
                timeout_s=p.timeout_s or self.cfg.timeout_s,
                prompt_chars=planned.prompt.chars() + workload.tools_chars(),
                clock=self._clock,
            )
            if p.pricing is not None:
                rec.cost = request_cost(p.pricing, rec.prompt_tokens, rec.completion_tokens)
                if rec.cost is not None:
                    actual = rec.cost
                elif rec.status in (TIMEOUT, CONNECTION_ERROR, STREAM_ERROR):
                    actual = reserved  # it may have been billed; assume the worst
            if rec.ok and self.cfg.checks and planned.prompt.checks:
                results = run_checks(
                    reply.text,
                    list(planned.prompt.checks),
                    tool_calls=reply.tool_calls,
                    tools=workload.tool_schemas(),
                )
                rec.checks = [r.to_dict() for r in results]
                rec.checks_passed = all(r.passed for r in results)
            if self.cfg.save_outputs and rec.ok:
                rec.output = redact(reply.text)
                if reply.tool_calls:
                    rec.tool_calls = [
                        {"name": c.name, "arguments": redact(c.arguments)[:_MAX_SAVED_ARGS]}
                        for c in reply.tool_calls
                    ]
            self._after(rec, state)
        finally:
            self.budget.settle(reserved, actual)
            self._report(rec)

    def _after(self, rec: RequestRecord, state: _EndpointState) -> None:
        if rec.ok:
            state.consecutive_failures = 0
            return
        state.consecutive_failures += 1
        status = rec.http_status if rec.status == HTTP_ERROR else None
        label = rec.error_type or (rec.error_message or "")[:80]
        if status in PERMANENT_STATUSES:
            state.aborted = f"stopped after HTTP {status} ({label}); fix the key, plan or base_url"
        elif status == 429 and rec.should_retry == "false":
            state.aborted = f"stopped after HTTP 429 that cannot be retried ({label})"
        elif status == 429 and rec.retry_after_s and rec.retry_after_s > MAX_RETRY_AFTER_WAIT_S:
            state.aborted = f"stopped: rate limited for {rec.retry_after_s:g} s"
        elif state.consecutive_failures >= self.cfg.max_consecutive_failures:
            state.aborted = f"stopped after {state.consecutive_failures} failures in a row"
        if state.aborted:
            return
        n = state.consecutive_failures
        if status == 429:
            pause = rec.retry_after_s if rec.retry_after_s is not None else min(60.0, 5.0 * n)
        else:
            pause = min(30.0, 2.0 ** (n - 1))
        state.paused_until = max(state.paused_until, self._clock() + pause)

    def _skip(self, rec: RequestRecord, reason: str) -> None:
        rec.status = SKIPPED
        rec.skip_reason = reason
        self._report(rec)

    def _report(self, rec: RequestRecord) -> None:
        self._done += 1
        if self._progress:
            self._progress(rec, self._done, len(self.plan))

    def _mark_unstarted(self) -> None:
        for planned in self.plan.requests:
            if planned.index not in self.records:
                self.records[planned.index] = RequestRecord(
                    index=planned.index,
                    provider=planned.provider.name,
                    model=planned.provider.model,
                    workload=planned.workload.id,
                    prompt_id=planned.prompt.id,
                    mode=planned.mode,
                    repetition=planned.repetition,
                    warmup=planned.warmup,
                    status=SKIPPED,
                    skip_reason="interrupted before it started"
                    if self.interrupted
                    else "not started",
                )


def missing_keys(cfg: RunConfig) -> list[str]:
    """Endpoints whose key variable is unset (checked before anything is sent)."""
    out = []
    for p in cfg.providers:
        if p.needs_key and not p.api_key():
            out.append(f"{p.name}: environment variable {p.api_key_env} is not set")
        for header, env in p.headers_env.items():
            if not os.environ.get(env, "").strip():
                out.append(f"{p.name}: environment variable {env} (header {header}) is not set")
    return out
