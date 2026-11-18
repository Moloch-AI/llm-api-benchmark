"""Plain-text tables for the terminal."""

from __future__ import annotations

from collections.abc import Sequence

from ..results import RunResult
from . import format as f


def render_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], right: set[int] | None = None
) -> str:
    right = right if right is not None else set(range(1, len(headers)))
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        parts = [
            c.rjust(widths[i]) if i in right else c.ljust(widths[i]) for i, c in enumerate(cells)
        ]
        return "  ".join(parts).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def terminal_report(result: RunResult) -> str:
    lines: list[str] = []
    run = result.run
    if result.synthetic:
        lines += ["SYNTHETIC FIXTURE DATA - not the performance of any real API.", ""]
    ds = result.dataset
    lines.append(
        f"{run.get('name')}  |  {run.get('started_at')}  |  dataset {ds.get('id')}@{ds.get('version')}"
        f"  |  {run.get('duration_s')} s"
    )
    if run.get("interrupted"):
        lines.append("Interrupted: partial results.")
    if run.get("stop_reason"):
        lines.append(f"Stopped early: {run['stop_reason']}")
    lines.append("")

    summaries = result.summaries()
    workloads = list(dict.fromkeys(s.workload for s in summaries))
    for wl in workloads:
        info = ds.get("workloads", {}).get(wl, {})
        lines.append(
            f"{wl}  (max_tokens {info.get('max_tokens', '?')}, {info.get('prompts', '?')} prompts)"
        )
        rows = []
        for s in (s for s in summaries if s.workload == wl):
            rows.append(
                [
                    s.provider,
                    s.mode,
                    str(s.latency.n if s.latency else 0),
                    f.pct(s.error_rate),
                    f.stat(s.ttft, "median"),
                    f.stat(s.ttft, "p95"),
                    f.stat(s.ttfc, "median"),
                    f.stat(s.latency, "median"),
                    f.stat(s.latency, "p95"),
                    f.tps(s),
                    f.tps(s, which="decode"),
                    f.checks(s),
                ]
            )
        lines.append(
            render_table(
                [
                    "Provider",
                    "Mode",
                    "n",
                    "Errors",
                    "TTFT p50",
                    "TTFT p95",
                    "TTFC p50",
                    "Latency p50",
                    "Latency p95",
                    "Out tok/s",
                    "Decode tok/s",
                    "Checks",
                ],
                rows,
                right=set(range(2, 12)),
            )
        )
        tooled = [s for s in summaries if s.workload == wl and s.tool_call_rate is not None]
        if tooled:
            kinds = sorted({k for s in tooled for k in s.check_types})
            lines.append(
                render_table(
                    ["Provider", "Mode", "Called a tool", "Same calls", *kinds],
                    [
                        [
                            s.provider,
                            s.mode,
                            f.pct(s.tool_call_rate),
                            f.agreement(s),
                            *(f.check_type(s, k) for k in kinds),
                        ]
                        for s in tooled
                    ],
                    right=set(range(2, 4 + len(kinds))),
                )
            )
        lines.append("")

    totals = result.totals()
    lines.append("Reliability (all requests, warm-up included)")
    rows = []
    for t in totals:
        cur = (result.provider(t.provider).get("pricing") or {}).get("currency", "USD")
        rows.append(
            [
                t.provider,
                t.model,
                str(t.executed),
                str(t.ok),
                f.pct(t.error_rate),
                ", ".join(f"{k} x{v}" for k, v in t.errors.items()) or f.DASH,
                str(t.skipped),
                f.money(t.cost, cur),
            ]
        )
    lines.append(
        render_table(
            ["Provider", "Model", "Sent", "OK", "Error rate", "Errors", "Skipped", "Cost"],
            rows,
            right={2, 3, 4, 6, 7},
        )
    )
    reasons = sorted({r.skip_reason for r in result.requests if r.skip_reason})
    if reasons:
        lines += ["", "Skipped because:"] + [f"  - {r}" for r in reasons]
    lines += ["", f.LEGEND]
    if any(s.tool_call_rate is not None for s in summaries):
        lines.append(f.TOOL_LEGEND)
    return "\n".join(lines) + "\n"
