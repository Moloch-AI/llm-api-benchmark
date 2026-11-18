"""Reports for ``llm-bench conformance``: terminal, Markdown and JUnit XML."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from ..conformance import ERROR, FAIL, PASS, SKIP, TESTS, WARN, ConformanceResult
from .table import render_table

LABEL = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "skip", ERROR: "ERROR"}
LEGEND = (
    "PASS: behaves like the OpenAI API. WARN: works, with a difference worth knowing (for example "
    "an explicit 4xx for an unsupported parameter, or a non-OpenAI error body). FAIL: a difference "
    "that breaks clients, such as a parameter that is accepted and silently ignored. skip: not "
    "applicable. ERROR: could not be judged (network, rate limit, 5xx)."
)
LIMITS = [
    "Each test is one request at one moment. Parameter tests judge the model's reply to a short "
    "prompt, so an unusual reply can look like a server problem; rerun a FAIL before reporting it.",
    "The suite checks the wire format and parameter handling, not answer quality or speed.",
    "A result describes one endpoint and one model. Providers change servers and models without "
    "notice; date every published result.",
]
SYNTHETIC = (
    "SYNTHETIC FIXTURE DATA: at least one endpoint is the in-process reference endpoint. Its "
    "results show how the suite reports, not how any real API behaves."
)


def _titles() -> dict[str, str]:
    return {t.id: t.title for t in TESTS}


def conformance_table(result: ConformanceResult) -> str:
    lines: list[str] = []
    if result.synthetic:
        lines += [SYNTHETIC, ""]
    names = [r.provider["name"] for r in result.reports]
    ids = list(dict.fromkeys(t.id for rep in result.reports for t in rep.results))
    if len(result.reports) == 1:
        rep = result.reports[0]
        p = rep.provider
        lines.append(f"{p['name']}  {p['model']}  {p['base_url']}  {rep.started_at}")
        rows = [[t.id, LABEL[t.outcome], t.summary] for t in rep.results]
        lines.append(render_table(["Test", "Result", "What happened"], rows, right=set()))
    else:
        by = {
            rep.provider["name"]: {t.id: LABEL[t.outcome] for t in rep.results}
            for rep in result.reports
        }
        rows = [[i, *(by[n].get(i, "-") for n in names)] for i in ids]
        lines.append(render_table(["Test", *names], rows, right=set()))
        for rep in result.reports:
            notable = [t for t in rep.results if t.outcome in (WARN, FAIL, ERROR)]
            if notable:
                lines += ["", f"{rep.provider['name']}:"]
                lines += [f"  {LABEL[t.outcome]:5} {t.id}: {t.summary}" for t in notable]
    for rep in result.reports:
        c = rep.counts()
        lines.append("")
        lines.append(
            f"{rep.provider['name']}: {c[PASS]} pass, {c[WARN]} warn, {c[FAIL]} fail, "
            f"{c[SKIP]} skipped, {c[ERROR]} error"
            + (f" (stopped: {rep.stopped})" if rep.stopped else "")
        )
    lines += ["", LEGEND]
    return "\n".join(lines) + "\n"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def conformance_markdown(result: ConformanceResult) -> str:
    titles = _titles()
    out = ["# OpenAI-compatible API conformance", ""]
    if result.synthetic:
        out += [f"> **{SYNTHETIC}**", ""]
    env = result.environment
    out += [
        f"Tool: llm-api-benchmark {env.get('tool_version', '')}, Python {env.get('python', '')}, "
        f"httpx {env.get('httpx', '')}.",
        "",
        "## Endpoints",
        "",
        _md_table(
            ["Name", "Base URL", "Model", "Tested (UTC)", "Pass", "Warn", "Fail", "Skip", "Error"],
            [
                [
                    r.provider["name"],
                    str(r.provider["base_url"]),
                    str(r.provider["model"]),
                    r.started_at,
                    *(str(r.counts()[o]) for o in (PASS, WARN, FAIL, SKIP, ERROR)),
                ]
                for r in result.reports
            ],
        ),
        "",
        "## Results",
        "",
    ]
    names = [r.provider["name"] for r in result.reports]
    by = {r.provider["name"]: {t.id: t for t in r.results} for r in result.reports}
    ids = list(dict.fromkeys(t.id for rep in result.reports for t in rep.results))
    out += [
        _md_table(
            ["Test", "What it checks", *names],
            [
                [
                    f"`{i}`",
                    titles.get(i, ""),
                    *(LABEL[by[n][i].outcome] if i in by[n] else "-" for n in names),
                ]
                for i in ids
            ],
        ),
        "",
        LEGEND,
        "",
    ]
    for rep in result.reports:
        out += [f"## {rep.provider['name']}", ""]
        if rep.stopped:
            out += [f"> Stopped: {rep.stopped}", ""]
        rows = [
            [f"`{t.id}`", LABEL[t.outcome], t.summary, "; ".join(t.details)] for t in rep.results
        ]
        out += [_md_table(["Test", "Result", "What happened", "Notes"], rows), ""]
        if rep.headers_seen:
            out += [
                "Response headers seen: " + ", ".join(f"`{h}`" for h in rep.headers_seen) + ".",
                "",
            ]
    out += ["## Limitations", ""] + [f"- {x}" for x in LIMITS]
    out += [
        "",
        "---",
        "",
        "Generated by [llm-api-benchmark](https://github.com/Moloch-AI/llm-api-benchmark) "
        "(`llm-bench conformance`). Test definitions: "
        "[docs/conformance.md](https://github.com/Moloch-AI/llm-api-benchmark/blob/main/docs/conformance.md).",
        "",
    ]
    return "\n".join(out)


def conformance_junit(result: ConformanceResult) -> str:
    """JUnit XML: one testsuite per endpoint. WARN is a pass with a system-out note."""
    root = ET.Element("testsuites", name="llm-bench conformance")
    for rep in result.reports:
        c = rep.counts()
        suite = ET.SubElement(
            root,
            "testsuite",
            name=f"conformance.{rep.provider['name']}",
            tests=str(len(rep.results)),
            failures=str(c[FAIL]),
            errors=str(c[ERROR]),
            skipped=str(c[SKIP]),
            timestamp=rep.started_at,
        )
        for t in rep.results:
            case = ET.SubElement(
                suite,
                "testcase",
                classname=f"{rep.provider['name']}.{t.category}",
                name=t.id,
                time=f"{t.latency_s or 0:.3f}",
            )
            body = "\n".join([t.summary, *t.details])
            if t.outcome == FAIL:
                ET.SubElement(case, "failure", message=t.summary).text = body
            elif t.outcome == ERROR:
                ET.SubElement(case, "error", message=t.summary).text = body
            elif t.outcome == SKIP:
                ET.SubElement(case, "skipped", message=t.summary)
            elif t.outcome == WARN:
                ET.SubElement(case, "system-out").text = f"WARN: {body}"
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def conformance_json(result: ConformanceResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
