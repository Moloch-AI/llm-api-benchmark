"""Compare two runs (or two endpoints in one run) without overstating the difference.

For each matching (workload, mode) group, the difference of medians B - A is reported with a
95% bootstrap confidence interval (2,000 resamples, fixed seed, so the same files always give the
same interval). When the interval includes zero, or either side has fewer than ``min_samples``
successful requests, the verdict is "no clear difference" or "too few samples" rather than a
winner.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..records import RequestRecord
from ..results import RunResult
from . import format as f
from .table import render_table

RESAMPLES = 2000
SEED = 20261117

_METRICS: dict[str, tuple[str, Callable[[RequestRecord], float | None]]] = {
    # name: (label, getter)
    "ttft_s": ("TTFT (s)", lambda r: r.ttft_s),
    "latency_s": ("Latency (s)", lambda r: r.latency_s),
    "output_tps": ("Out tok/s", lambda r: r.output_tps),
}


class CompareError(ValueError):
    pass


@dataclass(frozen=True)
class Row:
    workload: str
    mode: str
    metric: str
    a: str
    b: str
    n_a: int
    n_b: int
    median_a: float | None
    median_b: float | None
    ci_low: float | None
    ci_high: float | None
    verdict: str

    @property
    def delta(self) -> float | None:
        if self.median_a is None or self.median_b is None:
            return None
        return self.median_b - self.median_a

    @property
    def delta_pct(self) -> float | None:
        if self.delta is None or not self.median_a:
            return None
        return self.delta / self.median_a

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "mode": self.mode,
            "metric": self.metric,
            "a": self.a,
            "b": self.b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "median_a": _r(self.median_a),
            "median_b": _r(self.median_b),
            "delta": _r(self.delta),
            "delta_pct": _r(self.delta_pct),
            "ci95": [_r(self.ci_low), _r(self.ci_high)],
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class ErrorRow:
    workload: str
    mode: str
    a: str
    b: str
    errors_a: int
    sent_a: int
    errors_b: int
    sent_b: int


@dataclass(frozen=True)
class Comparison:
    label_a: str
    label_b: str
    rows: list[Row]
    errors: list[ErrorRow]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.label_a,
            "b": self.label_b,
            "method": f"difference of medians (B - A), 95% bootstrap CI, {RESAMPLES} resamples",
            "rows": [r.to_dict() for r in self.rows],
            "errors": [e.__dict__ for e in self.errors],
            "warnings": self.warnings,
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def bootstrap_median_diff(
    a: Sequence[float], b: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED
) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(statistics.median(rb) - statistics.median(ra))
    diffs.sort()
    lo = diffs[int(0.025 * (resamples - 1))]
    hi = diffs[int(0.975 * (resamples - 1))]
    return lo, hi


def _groups(
    result: RunResult, provider: str | None
) -> dict[tuple[str, str, str], list[RequestRecord]]:
    out: dict[tuple[str, str, str], list[RequestRecord]] = {}
    for r in result.requests:
        if r.warmup or (provider and r.provider != provider):
            continue
        out.setdefault((r.provider, r.workload, r.mode), []).append(r)
    return out


def compare(
    a: RunResult,
    b: RunResult,
    *,
    provider_a: str | None = None,
    provider_b: str | None = None,
    min_samples: int = 5,
) -> Comparison:
    warnings = _context_warnings(a, b)
    ga, gb = _groups(a, provider_a), _groups(b, provider_b)
    names_a = sorted({k[0] for k in ga})
    names_b = sorted({k[0] for k in gb})
    if not names_a or not names_b:
        raise CompareError("no measured requests to compare (check the provider names)")

    pairs: list[tuple[str, str]]
    if provider_a or provider_b or (len(names_a) == 1 and len(names_b) == 1):
        if len(names_a) != 1 or len(names_b) != 1:
            raise CompareError(
                "pick one endpoint on each side: --provider-a NAME and --provider-b NAME "
                f"(A has {', '.join(names_a)}; B has {', '.join(names_b)})"
            )
        pairs = [(names_a[0], names_b[0])]
    else:
        pairs = [(n, n) for n in names_a if n in names_b]
        if not pairs:
            raise CompareError(
                "the two runs share no endpoint names; choose the endpoints with --provider-a and --provider-b"
            )

    rows: list[Row] = []
    err_rows: list[ErrorRow] = []
    for pa, pb in pairs:
        wl_order = {w: i for i, w in enumerate(a.dataset.get("workloads", {}))}
        mode_order = {m: i for i, m in enumerate(a.config.get("modes", []))}
        keys = sorted(
            {(wl, mode) for (p, wl, mode) in ga if p == pa and (pb, wl, mode) in gb},
            key=lambda k: (wl_order.get(k[0], 99), k[0], mode_order.get(k[1], 9)),
        )
        if not keys:
            warnings.append(f"{pa} and {pb} share no workload/mode groups")
        for wl, mode in keys:
            ra, rb = ga[(pa, wl, mode)], gb[(pb, wl, mode)]
            err_rows.append(
                ErrorRow(
                    wl,
                    mode,
                    pa,
                    pb,
                    sum(1 for r in ra if r.executed and not r.ok),
                    sum(1 for r in ra if r.executed),
                    sum(1 for r in rb if r.executed and not r.ok),
                    sum(1 for r in rb if r.executed),
                )
            )
            for metric, (_, getter) in _METRICS.items():
                if metric == "ttft_s" and mode != "stream":
                    continue
                va = [v for v in (getter(r) for r in ra if r.ok) if v is not None]
                vb = [v for v in (getter(r) for r in rb if r.ok) if v is not None]
                if not va and not vb:
                    continue
                med_a = statistics.median(va) if va else None
                med_b = statistics.median(vb) if vb else None
                lo = hi = None
                if len(va) < min_samples or len(vb) < min_samples:
                    verdict = "too few samples"
                else:
                    lo, hi = bootstrap_median_diff(va, vb)
                    verdict = _verdict(metric, lo, hi)
                rows.append(
                    Row(wl, mode, metric, pa, pb, len(va), len(vb), med_a, med_b, lo, hi, verdict)
                )
    label_a = f"{a.run.get('name')} ({a.run.get('started_at')})"
    label_b = f"{b.run.get('name')} ({b.run.get('started_at')})"
    return Comparison(label_a, label_b, rows, err_rows, warnings)


def _verdict(metric: str, lo: float, hi: float) -> str:
    if lo <= 0 <= hi:
        return "no clear difference"
    b_higher = lo > 0
    if metric == "output_tps":
        return "B higher" if b_higher else "B lower"
    return "B slower" if b_higher else "B faster"


def _context_warnings(a: RunResult, b: RunResult) -> list[str]:
    w = []
    if a.synthetic or b.synthetic:
        w.append("At least one run contains synthetic fixture data.")
    if a.dataset.get("sha256") != b.dataset.get("sha256"):
        w.append(
            f"The runs used different datasets ({a.dataset.get('id')} {a.dataset.get('version')} vs "
            f"{b.dataset.get('id')} {b.dataset.get('version')}); prompts may differ."
        )
    for key in ("temperature", "max_tokens", "timeout_s", "concurrency"):
        if a.config.get(key) != b.config.get(key):
            w.append(f"Different {key}: {a.config.get(key)} vs {b.config.get(key)}.")
    if a.tool.get("version") != b.tool.get("version"):
        w.append(f"Different tool versions: {a.tool.get('version')} vs {b.tool.get('version')}.")
    if a.environment.get("platform") != b.environment.get("platform"):
        w.append("The runs came from different client platforms; network paths may differ too.")
    return w


def comparison_text(c: Comparison) -> str:
    lines = [f"A: {c.label_a}", f"B: {c.label_b}", ""]
    for w in c.warnings:
        lines.append(f"Warning: {w}")
    if c.warnings:
        lines.append("")
    rows = []
    for r in c.rows:
        label = _METRICS[r.metric][0]
        rows.append(
            [
                r.workload,
                r.mode,
                label,
                f"{r.a} vs {r.b}" if r.a != r.b else r.a,
                f"{r.n_a}/{r.n_b}",
                f.secs(r.median_a) if r.metric != "output_tps" else f.rate(r.median_a),
                f.secs(r.median_b) if r.metric != "output_tps" else f.rate(r.median_b),
                f.pct(r.delta_pct),
                _ci(r),
                r.verdict,
            ]
        )
    lines.append(
        render_table(
            [
                "Workload",
                "Mode",
                "Metric",
                "Endpoint",
                "n A/B",
                "Median A",
                "Median B",
                "Change",
                "95% CI of B-A",
                "Verdict",
            ],
            rows,
            right={4, 5, 6, 7},
        )
    )
    err = [
        [e.workload, e.mode, f"{e.errors_a}/{e.sent_a}", f"{e.errors_b}/{e.sent_b}"]
        for e in c.errors
        if e.errors_a or e.errors_b
    ]
    if err:
        lines += [
            "",
            "Errors (failed/sent)",
            render_table(["Workload", "Mode", "A", "B"], err, right={2, 3}),
        ]
    lines += [
        "",
        f"Medians of successful requests; CI from {RESAMPLES} bootstrap resamples. "
        "'no clear difference' means the interval includes zero.",
    ]
    return "\n".join(lines) + "\n"


def _ci(r: Row) -> str:
    if r.ci_low is None or r.ci_high is None:
        return f.DASH
    if r.metric == "output_tps":
        return f"[{r.ci_low:+.1f}, {r.ci_high:+.1f}]"
    return f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"


def comparison_markdown(c: Comparison) -> str:
    out = ["# Benchmark comparison", "", f"- **A:** {c.label_a}", f"- **B:** {c.label_b}", ""]
    if c.warnings:
        out += ["> " + "  \n> ".join(c.warnings), ""]
    out += [
        "| Workload | Mode | Metric | Endpoint | n A/B | Median A | Median B | Change | 95% CI of B-A | Verdict |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in c.rows:
        fmt = f.rate if r.metric == "output_tps" else f.secs
        endpoint = f"{r.a} vs {r.b}" if r.a != r.b else r.a
        out.append(
            f"| {r.workload} | {r.mode} | {_METRICS[r.metric][0]} | {endpoint} | {r.n_a}/{r.n_b} | "
            f"{fmt(r.median_a)} | {fmt(r.median_b)} | {f.pct(r.delta_pct)} | {_ci(r)} | {r.verdict} |"
        )
    out += [
        "",
        f"Medians of successful requests. The interval is a 95% bootstrap confidence interval of the "
        f"difference of medians ({RESAMPLES} resamples, fixed seed). 'no clear difference' means it "
        "includes zero.",
        "",
    ]
    return "\n".join(out)


def comparison_json(c: Comparison) -> str:
    return json.dumps(c.to_dict(), indent=2) + "\n"
