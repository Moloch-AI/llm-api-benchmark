"""Self-contained static HTML report with inline SVG charts (no external files or scripts).

Each workload gets two range charts (TTFT and end-to-end latency): a thin whisker from min to
max, a bar from P25 to P75, a ring-marked median and a P95 tick, one row per endpoint and mode.
Rows are labeled by name, so identity never depends on color, and every chart has the full table
next to it.
"""

from __future__ import annotations

import html
import math

from ..metrics import GroupSummary, Stat
from ..results import RunResult
from . import format as f

# Categorical slots, validated for color-vision deficiencies against both surfaces
# (light #FFFFFF, dark #19171C). Endpoints past the fourth use the neutral ink.
_SERIES_LIGHT = ("#584CC4", "#C8431A", "#1baf7a", "#eda100")
_SERIES_DARK = ("#9085e9", "#d95926", "#199e70", "#c98500")

_CSS = """
:root{color-scheme:light;--bg:#F6F5F2;--surface:#FFFFFF;--fg:#0E1116;--fg2:#47505E;--muted:#6B7280;
--line:#E3E1DC;--grid:#ECEAE5;--accent:#C8431A;--warn-bg:#FFF4E5;--warn-fg:#7A3E00;
--s1:#584CC4;--s2:#C8431A;--s3:#1baf7a;--s4:#eda100;--s0:#8A8F98}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#100E12;
--surface:#19171C;--fg:#F5F6F8;--fg2:#C9C5CE;--muted:#A5A0AB;--line:#2E2A33;--grid:#26232B;
--accent:#FF6A3D;--warn-bg:#2B1D12;--warn-fg:#FFC38A;--s1:#9085e9;--s2:#d95926;--s3:#199e70;
--s4:#c98500;--s0:#7D7884}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#100E12;--surface:#19171C;--fg:#F5F6F8;--fg2:#C9C5CE;
--muted:#A5A0AB;--line:#2E2A33;--grid:#26232B;--accent:#FF6A3D;--warn-bg:#2B1D12;--warn-fg:#FFC38A;
--s1:#9085e9;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s0:#7D7884}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Instrument Sans",system-ui,-apple-system,
"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:19px;margin:40px 0 12px}
h3{font-size:16px;margin:28px 0 8px}
p,li{color:var(--fg2)}code,.mono{font-family:"JetBrains Mono",ui-monospace,Consolas,monospace;font-size:13px}
.sub{color:var(--muted);margin:0 0 20px}
.banner{border:1px solid var(--accent);background:var(--warn-bg);color:var(--warn-fg);padding:12px 16px;
border-radius:6px;margin:16px 0;font-weight:600}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:12px;align-items:start}
.chart h4{margin:0 0 6px;font-size:13px;color:var(--fg2);font-weight:600}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--fg2);font-size:13px;font-family:inherit}
svg .axis text{fill:var(--muted)}svg .gridline{stroke:var(--grid);stroke-width:1}
svg .hit{fill:transparent}svg .row:hover .hit{fill:var(--grid)}
.legend{color:var(--muted);font-size:12px;margin:8px 0 0}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.45;box-shadow:0 6px 24px rgb(0 0 0/.18);
display:none;z-index:10;white-space:pre}
footer{margin-top:48px;color:var(--muted);font-size:13px}a{color:var(--accent)}
"""

_JS = """
(()=>{const t=document.getElementById('tip');document.querySelectorAll('svg .row').forEach(g=>{
const show=e=>{t.textContent=g.dataset.tip;t.style.display='block';
const x=Math.min(e.clientX+14,innerWidth-t.offsetWidth-8),y=Math.min(e.clientY+14,innerHeight-t.offsetHeight-8);
t.style.left=x+'px';t.style.top=y+'px'};g.addEventListener('mousemove',show);
g.addEventListener('mouseleave',()=>{t.style.display='none'});g.addEventListener('focus',e=>{
const r=g.getBoundingClientRect();show({clientX:r.left+r.width/2,clientY:r.top})});
g.addEventListener('blur',()=>{t.style.display='none'})})})();
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = 10.0 ** math.floor(math.log10(value))
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= step * exp:
            return step * exp
    return 10 * exp


def _ticks(top: float) -> list[float]:
    raw = top / 4
    exp = 10.0 ** math.floor(math.log10(raw))
    step = min((s * exp for s in (1, 2, 2.5, 5, 10) if s * exp >= raw), default=raw)
    out = []
    v = 0.0
    while v <= top + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _range_chart(title: str, rows: list[tuple[str, int, Stat | None]]) -> str:
    """rows: (label, color slot, stat). Returns an SVG range chart."""
    rows = [r for r in rows if r[2] is not None]
    if not rows:
        return ""
    label_w, pad_r, row_h, top_pad, axis_h = 168, 18, 30, 8, 26
    width = 520
    plot_w = width - label_w - pad_r
    height = top_pad + row_h * len(rows) + axis_h
    top = _nice_max(max(s.max for _, _, s in rows if s))
    ticks = _ticks(top)

    def x(v: float) -> float:
        return label_w + plot_w * v / top

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">',
        '<g class="axis">',
    ]
    for t in ticks:
        parts.append(
            f'<line class="gridline" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top_pad}" '
            f'y2="{height - axis_h}"/>'
            f'<text x="{x(t):.1f}" y="{height - 8}" text-anchor="middle">{t:g}</text>'
        )
    parts.append("</g>")
    for i, (label, slot, s) in enumerate(rows):
        assert s is not None
        cy = top_pad + row_h * i + row_h / 2
        color = f"var(--s{slot})"
        tip = (
            f"{label}\nn {s.n}\nmin {f.secs(s.min)}  p25 {f.secs(s.p25)}\nmedian {f.secs(s.median)}"
            f"  p75 {f.secs(s.p75)}\np90 {f.secs(s.p90)}  p95 {f.secs(s.p95)}\nmax {f.secs(s.max)} s"
        )
        box_w = max(x(s.p75) - x(s.p25), 2.0)
        parts.append(
            f'<g class="row" tabindex="0" data-tip="{_e(tip)}">'
            f'<rect class="hit" x="0" y="{cy - row_h / 2:.1f}" width="{width}" height="{row_h}"/>'
            f'<text x="{label_w - 10}" y="{cy + 4:.1f}" text-anchor="end">{_e(label)}</text>'
            f'<line x1="{x(s.min):.1f}" x2="{x(s.max):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" opacity=".55"/>'
            f'<rect x="{x(s.p25):.1f}" y="{cy - 5:.1f}" width="{box_w:.1f}" height="10" rx="4" '
            f'fill="{color}"/>'
            f'<line x1="{x(s.p95):.1f}" x2="{x(s.p95):.1f}" y1="{cy - 8:.1f}" y2="{cy + 8:.1f}" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'<circle cx="{x(s.median):.1f}" cy="{cy:.1f}" r="5" fill="{color}" stroke="var(--surface)" '
            f'stroke-width="2"/>'
            "</g>"
        )
    parts.append("</svg>")
    return f'<div class="chart card"><h4>{_e(title)}</h4>{"".join(parts)}</div>'


def _summary_table(rows: list[GroupSummary]) -> str:
    head = (
        "<tr><th>Provider</th><th>Mode</th><th class=n>n</th><th class=n>Errors</th>"
        "<th class=n>TTFT p50</th><th class=n>TTFT p95</th><th class=n>TTFC p50</th>"
        "<th class=n>Latency min</th><th class=n>p50</th><th class=n>p90</th><th class=n>p95</th>"
        "<th class=n>max</th><th class=n>Out tok/s</th><th class=n>Decode tok/s</th>"
        "<th class=n>Tokens p50</th><th class=n>Checks</th></tr>"
    )
    body = []
    for s in rows:
        cells = [
            f.stat(s.ttft, "median"),
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
        err_title = _e(f.errors(s) or "no errors")
        body.append(
            f"<tr><td>{_e(s.provider)}</td><td>{_e(s.mode)}</td>"
            f"<td class=n>{s.latency.n if s.latency else 0}</td>"
            f'<td class=n title="{err_title}">{f.pct(s.error_rate)}</td>'
            + "".join(f"<td class=n>{_e(c)}</td>" for c in cells)
            + "</tr>"
        )
    return f'<div class="card scroll"><table>{head}{"".join(body)}</table></div>'


def _detail_table(rows: list[GroupSummary]) -> str:
    """Per-check pass counts and tool-call figures, for groups that have them."""
    kinds = sorted({k for s in rows for k in s.check_types})
    tools = any(s.tool_call_rate is not None for s in rows)
    head = "<tr><th>Provider</th><th>Mode</th><th class=n>n</th>"
    if tools:
        head += "<th class=n>Called a tool</th><th class=n>Same calls</th>"
    head += "".join(f"<th class=n>{_e(k)}</th>" for k in kinds) + "</tr>"
    body = []
    for s in rows:
        cells = [f.pct(s.tool_call_rate), f.agreement(s)] if tools else []
        cells += [f.check_type(s, k) for k in kinds]
        body.append(
            f"<tr><td>{_e(s.provider)}</td><td>{_e(s.mode)}</td>"
            f"<td class=n>{s.latency.n if s.latency else 0}</td>"
            + "".join(f"<td class=n>{_e(c)}</td>" for c in cells)
            + "</tr>"
        )
    return f'<div class="card scroll"><table>{head}{"".join(body)}</table></div>'


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows)
    return f'<div class="card scroll"><table>{body}</table></div>'


def html_report(result: RunResult) -> str:
    run, cfg, ds, env, tool = (
        result.run,
        result.config,
        result.dataset,
        result.environment,
        result.tool,
    )
    summaries = result.summaries()
    slots: dict[str, int] = {}
    for p in result.providers:
        slots[p["name"]] = len(slots) + 1 if len(slots) < 4 else 0

    out = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width,initial-scale=1">',
        f"<title>{_e(run.get('name'))} benchmark</title><style>{_CSS}</style></head><body><main>",
        f"<h1>LLM API benchmark: {_e(run.get('name'))}</h1>",
        f'<p class="sub">{_e(run.get("started_at"))} UTC · dataset {_e(ds.get("id"))} {_e(ds.get("version"))}'
        f" · {_e(tool.get('name'))} {_e(tool.get('version'))}</p>",
    ]
    if result.synthetic:
        out.append(
            '<div class="banner">Synthetic fixture data. At least one endpoint in this run is the '
            "in-process simulator; these numbers are not the performance of any real API.</div>"
        )
    if run.get("interrupted"):
        out.append(
            '<div class="banner">Interrupted: the run stopped early and results are partial.</div>'
        )
    if run.get("stop_reason"):
        out.append(f'<div class="banner">Stopped early: {_e(run["stop_reason"])}</div>')

    out.append("<h2>Run</h2>")
    out.append(
        _kv_table(
            [
                ("Started / finished (UTC)", f"{run.get('started_at')} / {run.get('finished_at')}"),
                ("Duration", f"{run.get('duration_s')} s"),
                ("Dataset", f"{ds.get('id')} {ds.get('version')}, sha256 {ds.get('sha256')}"),
                ("Modes", ", ".join(cfg.get("modes", []))),
                ("Repetitions per prompt", str(cfg.get("repetitions"))),
                ("Warm-up per endpoint", f"{cfg.get('warmup')} (excluded from statistics)"),
                ("Concurrency", str(cfg.get("concurrency"))),
                ("Timeout", f"{cfg.get('timeout_s')} s (connect {cfg.get('connect_timeout_s')} s)"),
                (
                    "Temperature",
                    "provider default (not sent)"
                    if cfg.get("temperature") is None
                    else str(cfg.get("temperature")),
                ),
                (
                    "Client",
                    f"Python {env.get('python')}, httpx {env.get('httpx')}, {env.get('platform')}",
                ),
            ]
        )
    )

    out.append("<h2>Endpoints</h2>")
    prov_rows = []
    for p in result.providers:
        served = sorted(
            {m for s in summaries if s.provider == p["name"] for m in s.response_models}
        )
        pr = p.get("pricing")
        price = (
            f"{pr['input_per_mtok']} / {pr['output_per_mtok']} {pr['currency']} per 1M in/out "
            f"({pr['source']}, checked {pr.get('checked_on') or 'n/a'})"
            if pr
            else "not provided"
        )
        prov_rows.append(
            f"<tr><td>{_e(p['name'])}</td><td class=mono>{_e(p.get('base_url'))}</td>"
            f"<td class=mono>{_e(p.get('model'))}</td><td class=mono>{_e(', '.join(served) or '-')}</td>"
            f"<td>{_e(price)}</td></tr>"
        )
    out.append(
        '<div class="card scroll"><table><tr><th>Name</th><th>Base URL</th><th>Model requested</th>'
        f"<th>Model reported</th><th>Pricing</th></tr>{''.join(prov_rows)}</table></div>"
    )

    out.append("<h2>Results by workload</h2>")
    out.append(
        f'<p class="legend">{_e(f.LEGEND)} Charts: whisker min to max, bar P25 to P75, '
        "dot median, tick P95. Hover or focus a row for its numbers.</p>"
    )
    for wl in dict.fromkeys(s.workload for s in summaries):
        info = ds.get("workloads", {}).get(wl, {})
        group = [s for s in summaries if s.workload == wl]
        out.append(
            f"<h3>{_e(wl)}</h3><p class=sub>{_e(info.get('description', ''))} "
            f"max_tokens {_e(info.get('max_tokens'))}, {_e(info.get('prompts'))} prompts.</p>"
        )
        charts = [
            _range_chart(
                f"Time to first token, seconds ({wl})",
                [
                    (f"{s.provider} · stream", slots.get(s.provider, 0), s.ttft)
                    for s in group
                    if s.mode == "stream"
                ],
            ),
            _range_chart(
                f"End-to-end latency, seconds ({wl})",
                [(f"{s.provider} · {s.mode}", slots.get(s.provider, 0), s.latency) for s in group],
            ),
        ]
        out.append(f'<div class="charts">{"".join(c for c in charts if c)}</div>')
        out.append(_summary_table(group))
        detail = [s for s in group if f.has_detail(s)]
        if detail:
            out.append(_detail_table(detail))

    if any(f.has_detail(s) for s in summaries):
        out.append(f'<p class="legend">{_e(f.TOOL_LEGEND)}</p>')
    out.append("<h2>Reliability</h2><p>All requests, warm-up included.</p>")
    rows = []
    for t in result.totals():
        cur = (result.provider(t.provider).get("pricing") or {}).get("currency", "USD")
        errs = ", ".join(f"{k} x{v}" for k, v in t.errors.items()) or "-"
        rows.append(
            f"<tr><td>{_e(t.provider)}</td><td class=n>{t.planned}</td><td class=n>{t.executed}</td>"
            f"<td class=n>{t.ok}</td><td class=n>{f.pct(t.error_rate)}</td><td>{_e(errs)}</td>"
            f"<td class=n>{t.skipped}</td><td class=n>{_e(f.money(t.cost, cur))}</td></tr>"
        )
    out.append(
        '<div class="card scroll"><table><tr><th>Provider</th><th class=n>Planned</th><th class=n>Sent</th>'
        "<th class=n>OK</th><th class=n>Error rate</th><th>Errors</th><th class=n>Skipped</th>"
        f"<th class=n>Cost</th></tr>{''.join(rows)}</table></div>"
    )
    reasons = sorted({r.skip_reason for r in result.requests if r.skip_reason})
    if reasons:
        out.append(
            "<h3>Skipped requests</h3><ul>"
            + "".join(f"<li>{_e(r)}</li>" for r in reasons)
            + "</ul>"
        )

    out.append(
        "<h2>Limitations</h2><ul>" + "".join(f"<li>{_e(x)}</li>" for x in result.limitations())
    )
    for p in result.providers:
        for n in p.get("notes", []):
            out.append(f"<li>{_e(p['name'])}: {_e(n)}</li>")
    for w in result.warnings:
        out.append(f"<li>Warning: {_e(w)}</li>")
    out.append("</ul>")
    out.append(
        '<footer>Generated by <a href="https://github.com/Moloch-AI/llm-api-benchmark">llm-api-benchmark</a>. '
        'Methodology: <a href="https://github.com/Moloch-AI/llm-api-benchmark/blob/main/docs/methodology.md">'
        "docs/methodology.md</a>.</footer>"
    )
    out.append(f'</main><div id="tip" role="tooltip"></div><script>{_JS}</script></body></html>\n')
    return "\n".join(out)
