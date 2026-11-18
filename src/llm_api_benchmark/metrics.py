"""Distribution statistics and per-group summaries.

Percentiles use linear interpolation between the two nearest ranks (the same as NumPy's default
and Excel's PERCENTILE.INC). With few samples, high percentiles sit close to the maximum; every
statistic carries its sample size so readers can judge that.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .records import RequestRecord

# Below this many successful samples, P90/P95 are reported but flagged as unreliable.
MIN_SAMPLES_FOR_TAIL = 20


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


@dataclass(frozen=True)
class Stat:
    n: int
    min: float
    p25: float
    median: float
    p75: float
    p90: float
    p95: float
    max: float
    mean: float

    def to_dict(self, digits: int = 4) -> dict[str, float | int]:
        return {
            "n": self.n,
            "min": round(self.min, digits),
            "p25": round(self.p25, digits),
            "median": round(self.median, digits),
            "p75": round(self.p75, digits),
            "p90": round(self.p90, digits),
            "p95": round(self.p95, digits),
            "max": round(self.max, digits),
            "mean": round(self.mean, digits),
        }


def describe(values: Iterable[float | None]) -> Stat | None:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    return Stat(
        n=len(clean),
        min=clean[0],
        p25=percentile(clean, 25),
        median=percentile(clean, 50),
        p75=percentile(clean, 75),
        p90=percentile(clean, 90),
        p95=percentile(clean, 95),
        max=clean[-1],
        mean=sum(clean) / len(clean),
    )


@dataclass
class GroupSummary:
    provider: str
    model: str
    workload: str
    mode: str
    planned: int
    executed: int
    ok: int
    errors: dict[str, int]
    response_models: list[str]
    latency: Stat | None
    headers: Stat | None
    ttft: Stat | None
    ttfc: Stat | None
    output_tps: Stat | None
    decode_tps: Stat | None
    completion_tokens: Stat | None
    tokens_estimated: bool
    cost: float | None
    cost_complete: bool
    checks_run: int
    checks_passed: int
    finish_reasons: dict[str, int] = field(default_factory=dict)
    # Per check type: [passed, run], over successful replies.
    check_types: dict[str, list[int]] = field(default_factory=dict)
    # Tool workloads only (None otherwise).
    tool_call_rate: float | None = None
    tool_agreement: float | None = None
    tool_agreement_prompts: int = 0

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.provider, self.model, self.workload, self.mode)

    @property
    def error_count(self) -> int:
        return sum(self.errors.values())

    @property
    def error_rate(self) -> float | None:
        return None if self.executed == 0 else self.error_count / self.executed

    @property
    def check_pass_rate(self) -> float | None:
        return None if self.checks_run == 0 else self.checks_passed / self.checks_run

    def to_dict(self) -> dict[str, Any]:
        def s(stat: Stat | None, digits: int = 4) -> dict[str, float | int] | None:
            return stat.to_dict(digits) if stat else None

        return {
            "provider": self.provider,
            "model": self.model,
            "response_models": self.response_models,
            "workload": self.workload,
            "mode": self.mode,
            "planned": self.planned,
            "executed": self.executed,
            "ok": self.ok,
            "errors": self.errors,
            "error_rate": None if self.error_rate is None else round(self.error_rate, 4),
            "latency_s": s(self.latency),
            "headers_s": s(self.headers),
            "ttft_s": s(self.ttft),
            "ttfc_s": s(self.ttfc),
            "output_tps": s(self.output_tps, 2),
            "decode_tps": s(self.decode_tps, 2),
            "completion_tokens": s(self.completion_tokens, 1),
            "tokens_estimated": self.tokens_estimated,
            "cost": None if self.cost is None else round(self.cost, 6),
            "cost_complete": self.cost_complete,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "check_pass_rate": None
            if self.check_pass_rate is None
            else round(self.check_pass_rate, 4),
            "finish_reasons": self.finish_reasons,
            "check_types": self.check_types,
            "tool_call_rate": _r(self.tool_call_rate),
            "tool_agreement": _r(self.tool_agreement),
            "tool_agreement_prompts": self.tool_agreement_prompts,
        }


def summarize(records: Iterable[RequestRecord]) -> list[GroupSummary]:
    """One summary per (provider, model, workload, mode). Warm-up requests are excluded."""
    groups: dict[tuple[str, str, str, str], list[RequestRecord]] = {}
    for r in records:
        if r.warmup:
            continue
        groups.setdefault((r.provider, r.model, r.workload, r.mode), []).append(r)
    out = []
    for (provider, model, workload, mode), recs in groups.items():
        executed = [r for r in recs if r.executed]
        ok = [r for r in executed if r.ok]
        errors = Counter(r.error_kind for r in executed if r.error_kind)
        costs = [r.cost for r in executed]
        known = [c for c in costs if c is not None]
        checked = [r for r in ok if r.checks_passed is not None]
        tooled = [r for r in ok if r.tool_call_names is not None]
        agreement, agreement_n = tool_agreement(tooled)
        out.append(
            GroupSummary(
                provider=provider,
                model=model,
                workload=workload,
                mode=mode,
                planned=len(recs),
                executed=len(executed),
                ok=len(ok),
                errors=dict(sorted(errors.items())),
                response_models=sorted({r.response_model for r in ok if r.response_model}),
                latency=describe(r.latency_s for r in ok),
                headers=describe(r.headers_s for r in ok),
                ttft=describe(r.ttft_s for r in ok) if mode == "stream" else None,
                ttfc=describe(r.ttfc_s for r in ok) if mode == "stream" else None,
                output_tps=describe(r.output_tps for r in ok),
                decode_tps=describe(r.decode_tps for r in ok),
                completion_tokens=describe(
                    float(r.completion_tokens) for r in ok if r.completion_tokens is not None
                ),
                tokens_estimated=any(r.tokens_estimated for r in ok),
                cost=sum(known) if known else None,
                cost_complete=bool(ok) and all(r.cost is not None for r in ok),
                checks_run=len(checked),
                checks_passed=sum(1 for r in checked if r.checks_passed),
                finish_reasons=dict(
                    sorted(Counter(r.finish_reason for r in ok if r.finish_reason).items())
                ),
                check_types=_check_types(checked),
                tool_call_rate=(
                    sum(1 for r in tooled if r.tool_call_names) / len(tooled) if tooled else None
                ),
                tool_agreement=agreement,
                tool_agreement_prompts=agreement_n,
            )
        )
    return out


def tool_agreement(records: Iterable[RequestRecord]) -> tuple[float | None, int]:
    """How often repeated runs of the same prompt make the same tool calls.

    For each prompt with at least two successful replies, the share of replies that match the
    most common call signature (same functions, same parsed arguments, or no call); then the
    mean over those prompts. Returns (mean, number of prompts it covers).
    """
    by_prompt: dict[str, list[str]] = {}
    for r in records:
        if r.tool_signature is not None:
            by_prompt.setdefault(r.prompt_id, []).append(r.tool_signature)
    shares = [
        Counter(sigs).most_common(1)[0][1] / len(sigs)
        for sigs in by_prompt.values()
        if len(sigs) >= 2
    ]
    if not shares:
        return None, 0
    return sum(shares) / len(shares), len(shares)


def _check_types(records: Iterable[RequestRecord]) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for r in records:
        for c in r.checks or []:
            entry = counts.setdefault(str(c.get("type")), [0, 0])
            entry[0] += 1 if c.get("passed") else 0
            entry[1] += 1
    return dict(sorted(counts.items()))


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


@dataclass
class ProviderTotals:
    provider: str
    model: str
    planned: int
    executed: int
    ok: int
    skipped: int
    cancelled: int
    errors: dict[str, int]
    cost: float | None

    @property
    def error_rate(self) -> float | None:
        return None if self.executed == 0 else sum(self.errors.values()) / self.executed

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "planned": self.planned,
            "executed": self.executed,
            "ok": self.ok,
            "skipped": self.skipped,
            "cancelled": self.cancelled,
            "errors": self.errors,
            "error_rate": None if self.error_rate is None else round(self.error_rate, 4),
            "cost": None if self.cost is None else round(self.cost, 6),
        }


def provider_totals(records: Iterable[RequestRecord]) -> list[ProviderTotals]:
    """Reliability across all workloads per endpoint, warm-ups included (they are real requests)."""
    by: dict[tuple[str, str], list[RequestRecord]] = {}
    for r in records:
        by.setdefault((r.provider, r.model), []).append(r)
    out = []
    for (provider, model), recs in by.items():
        executed = [r for r in recs if r.executed]
        known = [r.cost for r in executed if r.cost is not None]
        out.append(
            ProviderTotals(
                provider=provider,
                model=model,
                planned=len(recs),
                executed=len(executed),
                ok=sum(1 for r in executed if r.ok),
                skipped=sum(1 for r in recs if r.status == "skipped"),
                cancelled=sum(1 for r in recs if r.status == "cancelled"),
                errors=dict(
                    sorted(Counter(r.error_kind for r in executed if r.error_kind).items())
                ),
                cost=sum(known) if known else None,
            )
        )
    return out
