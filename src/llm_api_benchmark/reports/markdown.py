"""Markdown report: the same numbers as the terminal, plus the run's full context."""

from __future__ import annotations

from collections.abc import Sequence

from ..results import RunResult
from . import format as f


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], right_from: int = 1) -> str:
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    align = ["---" if i < right_from else "---:" for i in range(len(headers))]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def markdown_report(result: RunResult) -> str:
    run, cfg, ds = result.run, result.config, result.dataset
    out: list[str] = [f"# LLM API benchmark: {run.get('name')}", ""]
    if result.synthetic:
        out += [
            "> **Synthetic fixture data.** At least one endpoint in this run is the in-process "
            "simulator. These numbers are not the performance of any real API.",
            "",
        ]
    if run.get("interrupted"):
        out += [
            "> **Interrupted.** The run was stopped before it finished; results are partial.",
            "",
        ]
    if run.get("stop_reason"):
        out += [f"> **Stopped early:** {run['stop_reason']}", ""]

    tool = result.tool
    env = result.environment
    meta = [
        ["Started (UTC)", str(run.get("started_at"))],
        ["Finished (UTC)", str(run.get("finished_at"))],
        ["Duration", f"{run.get('duration_s')} s"],
        ["Tool", f"{tool.get('name')} {tool.get('version')}"],
        ["Dataset", f"{ds.get('id')} {ds.get('version')} (sha256 {str(ds.get('sha256'))[:12]}...)"],
        ["Modes", ", ".join(cfg.get("modes", []))],
        ["Repetitions per prompt", str(cfg.get("repetitions"))],
        ["Warm-up requests per endpoint", f"{cfg.get('warmup')} (excluded from statistics)"],
        ["Concurrency", str(cfg.get("concurrency"))],
        [
            "Timeout",
            f"{cfg.get('timeout_s')} s per request (connect {cfg.get('connect_timeout_s')} s)",
        ],
        [
            "Temperature",
            "provider default (not sent)"
            if cfg.get("temperature") is None
            else str(cfg.get("temperature")),
        ],
        ["Seed (request order)", str(cfg.get("seed"))],
        ["Client", f"Python {env.get('python')}, httpx {env.get('httpx')}, {env.get('platform')}"],
    ]
    out += ["## Run", "", _table(["Setting", "Value"], meta, right_from=2), ""]

    prov_rows = []
    summaries = result.summaries()
    for p in result.providers:
        served = sorted(
            {m for s in summaries if s.provider == p["name"] for m in s.response_models}
        )
        pricing = p.get("pricing")
        price = (
            f"{pricing['input_per_mtok']} / {pricing['output_per_mtok']} {pricing['currency']} per 1M "
            f"in/out (source: {pricing['source']}, checked {pricing.get('checked_on') or 'n/a'})"
            if pricing
            else "not provided"
        )
        prov_rows.append(
            [
                p["name"],
                p.get("preset", ""),
                p.get("base_url", ""),
                p.get("model", ""),
                ", ".join(served) or f.DASH,
                price,
            ]
        )
    out += [
        "## Endpoints",
        "",
        _table(
            ["Name", "Preset", "Base URL", "Model requested", "Model reported", "Pricing"],
            prov_rows,
            99,
        ),
        "",
    ]

    wl_rows = [
        [
            wid,
            str(info.get("max_tokens")),
            str(info.get("prompts")),
            str(info.get("description", "")),
        ]
        for wid, info in ds.get("workloads", {}).items()
    ]
    out += [
        "## Workloads",
        "",
        _table(["Workload", "max_tokens", "Prompts", "Description"], wl_rows, 99),
        "",
    ]

    out += ["## Results by workload", "", f.LEGEND, ""]
    for wl in dict.fromkeys(s.workload for s in summaries):
        rows = []
        for s in (s for s in summaries if s.workload == wl):
            rows.append(
                [
                    s.provider,
                    s.mode,
                    str(s.latency.n if s.latency else 0),
                    f.pct(s.error_rate),
                    f.stat(s.ttft, "median"),
                    f.stat(s.ttft, "p90"),
                    f.stat(s.ttft, "p95"),
                    f.stat(s.ttfc, "median"),
                    f.stat(s.latency, "min"),
                    f.stat(s.latency, "median"),
                    f.stat(s.latency, "p90"),
                    f.stat(s.latency, "p95"),
                    f.stat(s.latency, "max"),
                    f.tps(s),
                    f.tps(s, which="decode"),
                    f.rate(s.completion_tokens.median if s.completion_tokens else None),
                    f.checks(s),
                ]
            )
        out += [
            f"### {wl}",
            "",
            _table(
                [
                    "Provider",
                    "Mode",
                    "n",
                    "Errors",
                    "TTFT p50",
                    "TTFT p90",
                    "TTFT p95",
                    "TTFC p50",
                    "Latency min",
                    "Latency p50",
                    "Latency p90",
                    "Latency p95",
                    "Latency max",
                    "Out tok/s p50",
                    "Decode tok/s p50",
                    "Tokens p50",
                    "Checks",
                ],
                rows,
                right_from=2,
            ),
            "",
        ]

    out += ["## Reliability", "", "All requests, warm-up included.", ""]
    rows = []
    for t in result.totals():
        cur = (result.provider(t.provider).get("pricing") or {}).get("currency", "USD")
        rows.append(
            [
                t.provider,
                str(t.planned),
                str(t.executed),
                str(t.ok),
                f.pct(t.error_rate),
                ", ".join(f"{k} x{v}" for k, v in t.errors.items()) or f.DASH,
                str(t.skipped),
                f.money(t.cost, cur),
            ]
        )
    out += [
        _table(
            ["Provider", "Planned", "Sent", "OK", "Error rate", "Errors", "Skipped", "Cost"],
            rows,
            1,
        ),
        "",
    ]

    examples: dict[tuple[str, str], tuple[str, str]] = {}
    for r in result.requests:
        if r.error_kind and (r.provider, r.error_kind) not in examples:
            examples[(r.provider, r.error_kind)] = (r.error_type or f.DASH, r.error_message or "")
    if examples:
        rows = [[p, k, t, m] for (p, k), (t, m) in examples.items()]
        out += [
            "### Errors",
            "",
            _table(["Provider", "Kind", "Type", "First message"], rows, 99),
            "",
        ]
    reasons = sorted({r.skip_reason for r in result.requests if r.skip_reason})
    if reasons:
        out += ["### Skipped requests", ""] + [f"- {r}" for r in reasons] + [""]

    notes = [n for p in result.providers for n in (f"{p['name']}: {x}" for x in p.get("notes", []))]
    out += ["## Limitations", ""] + [f"- {x}" for x in result.limitations()]
    if notes:
        out += ["", "Provider notes:", ""] + [f"- {n}" for n in notes]
    if result.warnings:
        out += ["", "Warnings:", ""] + [f"- {w}" for w in result.warnings]
    out += [
        "",
        "---",
        "",
        "Generated by [llm-api-benchmark](https://github.com/Moloch-AI/llm-api-benchmark). "
        "Methodology: [docs/methodology.md](https://github.com/Moloch-AI/llm-api-benchmark/blob/main/docs/methodology.md).",
        "",
    ]
    return "\n".join(out)
