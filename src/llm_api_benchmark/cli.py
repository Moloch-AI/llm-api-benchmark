"""Command-line interface: ``llm-bench``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import httpx

from ._version import __version__
from .config import (
    MAX_CONCURRENCY,
    ConfigError,
    RunConfig,
    load_config,
    parse_config,
)
from .datasets import DatasetError, builtin_names, load_dataset
from .pricing import staleness_warning
from .providers import PRESETS
from .providers.openai_chat import parse_error_body
from .records import RequestRecord
from .redact import REDACTOR, mask, redact
from .reports import (
    CompareError,
    compare,
    comparison_json,
    comparison_markdown,
    comparison_text,
    html_report,
    markdown_report,
    render_table,
    terminal_report,
)
from .results import (
    ResultsError,
    RunResult,
    build_result,
    load_results,
    requests_csv,
    summary_csv,
    to_json,
)
from .runner import Plan, Runner, build_plan, missing_keys

# Runs that send more than this many real requests ask for confirmation (or --yes).
CONFIRM_ABOVE = 20
FORMATS = ("json", "csv", "md", "html")
TEMPLATES = ("synthetic-demo", "molochai", "openai-compatible")


class CliError(Exception):
    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def err(text: str = "") -> None:
    print(redact(text), file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- run


def _apply_overrides(cfg: RunConfig, a: argparse.Namespace) -> RunConfig:
    providers = None
    if a.providers:
        wanted = [n.strip() for n in a.providers.split(",") if n.strip()]
        known = {p.name: p for p in cfg.providers}
        unknown = [n for n in wanted if n not in known]
        if unknown:
            raise CliError(
                f"--providers: {', '.join(unknown)} not in the config (known: {', '.join(known)})"
            )
        providers = tuple(known[n] for n in wanted)
    modes = None
    if a.modes:
        modes = tuple(dict.fromkeys(m.strip() for m in a.modes.split(",") if m.strip()))
        bad = [m for m in modes if m not in ("stream", "plain")]
        if bad:
            raise CliError(f"--modes: {', '.join(bad)} is not stream or plain")
    workloads = (
        tuple(w.strip() for w in a.workloads.split(",") if w.strip()) if a.workloads else None
    )
    try:
        return cfg.with_overrides(
            providers=providers,
            modes=modes,
            workloads=workloads,
            repetitions=a.repetitions,
            warmup=a.warmup,
            concurrency=a.concurrency,
            max_requests=a.max_requests,
            budget=a.budget,
        )
    except ConfigError as e:
        raise CliError("Invalid override:\n  " + "\n  ".join(e.problems)) from e


def _plan_text(cfg: RunConfig, plan: Plan, dataset_ref: str, workloads: list[str]) -> str:
    lines = [f"Plan: {cfg.name}" + (f" ({cfg.source_path})" if cfg.source_path else "")]
    lines.append(f"  Dataset      {dataset_ref}; workloads: {', '.join(workloads)}")
    lines.append(
        f"  Modes        {', '.join(cfg.modes)}; repetitions {cfg.repetitions}; warm-up {cfg.warmup} per "
        f"endpoint; concurrency {cfg.concurrency}; timeout {cfg.timeout_s:g} s"
    )
    temp = "provider default (not sent)" if cfg.temperature is None else f"{cfg.temperature:g}"
    lines.append(f"  Temperature  {temp}")
    lines.append("  Endpoints")
    counts = plan.by_provider()
    for p in cfg.providers:
        if p.is_synthetic:
            key = "synthetic, no key"
        elif not p.needs_key:
            key = "no auth"
        else:
            value = p.api_key()
            key = f"key {p.api_key_env}: " + (f"set ({mask(value)})" if value else "NOT SET")
        rpm = f", {p.rate_limit_rpm:g} req/min" if p.rate_limit_rpm else ""
        lines.append(
            f"    {p.name}: {p.model} at {p.base_url} ({key}{rpm}) - {counts.get(p.name, 0)} requests"
        )
    lines.append(f"  Requests     {len(plan)} (limit {cfg.max_requests})")
    unbounded = plan.unbounded_output(cfg)
    out_line = f"  Output       up to {plan.worst_case_output_tokens(cfg):,} tokens in total"
    if unbounded:
        out_line += f" (assumed caps for {', '.join(unbounded)}, which do not apply max_tokens)"
    lines.append(out_line)
    worst, unpriced = plan.worst_case_cost(cfg)
    if any(p.pricing for p in cfg.providers):
        cost = f"  Cost         at most {worst:.4f} {cfg.budget_currency} (estimated worst case)"
        if cfg.budget is not None:
            cost += f"; budget {cfg.budget:g} {cfg.budget_currency}"
        lines.append(cost)
    if unpriced:
        lines.append(f"  No pricing   {', '.join(unpriced)}: cost is not tracked for these")
    return "\n".join(lines)


def _confirm(n_real: int, yes: bool) -> None:
    if yes or n_real <= CONFIRM_ABOVE:
        return
    if not sys.stdin.isatty():
        raise CliError(
            f"This run sends {n_real} requests. Add --yes to confirm (or use --dry-run)."
        )
    err(f"This run sends {n_real} requests to real endpoints. Continue? [y/N] ")
    if input().strip().lower() not in ("y", "yes"):
        raise CliError("Cancelled; nothing was sent.", code=1)


def _progress(quiet: bool) -> Any:
    def show(rec: RequestRecord, done: int, total: int) -> None:
        if quiet:
            return
        width = len(str(total))
        head = (
            f"[{done:>{width}}/{total}] {rec.provider}  {rec.workload}/{rec.prompt_id}  {rec.mode}"
        )
        if rec.warmup:
            head += " (warm-up)"
        if rec.status == "skipped":
            err(f"{head}  skipped: {rec.skip_reason}")
            return
        parts = [rec.status if rec.ok else (rec.error_kind or rec.status)]
        if rec.latency_s is not None:
            parts.append(f"{rec.latency_s:.2f}s")
        if rec.ttft_s is not None:
            parts.append(f"ttft {rec.ttft_s:.2f}s")
        if rec.completion_tokens is not None:
            parts.append(f"{'~' if rec.tokens_estimated else ''}{rec.completion_tokens} tok")
        if not rec.ok and rec.error_message:
            parts.append(rec.error_message[:120])
        err(f"{head}  " + "  ".join(parts))

    return show


def _write_outputs(result: RunResult, out_dir: Path, formats: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"{result.run.get('name')}-{stamp}"
    written = []
    content: dict[str, list[tuple[str, str]]] = {
        "json": [(".json", to_json(result))],
        "csv": [(".requests.csv", requests_csv(result)), (".summary.csv", summary_csv(result))],
        "md": [(".md", markdown_report(result))],
        "html": [(".html", html_report(result))],
    }
    for fmt in formats:
        for suffix, text in content[fmt]:
            path = base.with_name(base.name + suffix)
            path.write_text(text, encoding="utf-8", newline="\n")
            written.append(path)
    return written


def cmd_run(a: argparse.Namespace) -> int:
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        raise CliError(f"{a.config} is not valid:\n  " + "\n  ".join(e.problems)) from e
    cfg = _apply_overrides(cfg, a)
    try:
        dataset = load_dataset(cfg.dataset, cfg.dataset_base_dir())
        selected = dataset.select(cfg.workloads)
    except DatasetError as e:
        raise CliError("Dataset problem:\n  " + "\n  ".join(e.problems)) from e
    plan = build_plan(cfg, dataset)
    workload_desc = [
        f"{w.id} ({len(w.prompts)} prompts, max_tokens {cfg.max_tokens or w.max_tokens})"
        for w in selected
    ]
    err(_plan_text(cfg, plan, f"{dataset.id} {dataset.version} ({dataset.source})", workload_desc))
    for p in cfg.providers:
        for note in p.notes:
            err(f"  Note         {p.name}: {note}")

    warnings = [
        w for p in cfg.providers if p.pricing for w in [staleness_warning(p.pricing, p.name)] if w
    ]
    for w in warnings:
        err(f"  Warning      {w}")

    problems = []
    if len(plan) > cfg.max_requests:
        problems.append(
            f"The plan has {len(plan)} requests, more than max_requests ({cfg.max_requests}). Lower "
            "repetitions, workloads or modes, or raise max_requests."
        )
    worst, _ = plan.worst_case_cost(cfg)
    if cfg.budget is not None and worst > cfg.budget and not a.stop_at_budget:
        problems.append(
            f"The estimated worst case ({worst:.4f} {cfg.budget_currency}) is above the budget "
            f"({cfg.budget:g}). Reduce the plan, raise the budget, or pass --stop-at-budget to run "
            "until the budget would be exceeded."
        )
    keys = missing_keys(cfg)
    if a.dry_run:
        for k in keys:
            err(f"  Missing      {k}")
        for prob in problems:
            err(f"  Problem      {prob}")
        err("\nDry run: nothing was sent.")
        return 2 if problems else 0
    problems += keys
    if problems:
        raise CliError("Not started:\n  " + "\n  ".join(problems))

    real = sum(1 for r in plan.requests if not r.provider.is_synthetic)
    _confirm(real, a.yes)
    err("")

    runner = Runner(cfg, dataset, plan, progress=_progress(a.quiet))
    interrupted = False
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        interrupted = True
        runner.interrupted = True
        err("\nInterrupted: in-flight requests were cancelled; writing partial results.")
    result = build_result(runner, warnings)
    formats = _formats(a.formats)
    written = _write_outputs(result, Path(a.out_dir), formats)
    print(terminal_report(result), end="")
    for path in written:
        err(f"Wrote {path}")
    return 130 if interrupted else 0


def _formats(value: str) -> list[str]:
    formats = [f.strip() for f in value.split(",") if f.strip()]
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise CliError(f"--formats: unknown {', '.join(bad)} (choose from {', '.join(FORMATS)})")
    return list(dict.fromkeys(["json", *formats]))


# --------------------------------------------------------------------------- other commands


def cmd_providers(a: argparse.Namespace) -> int:
    if a.json:
        print(
            json.dumps(
                {k: v.__dict__ | {"notes": list(v.notes)} for k, v in PRESETS.items()}, indent=2
            )
        )
        return 0
    rows = [
        [
            p.name,
            p.api,
            "(in-process)" if p.api == "synthetic" else p.base_url or "(set base_url)",
            "-" if p.auth_scheme == "none" else p.api_key_env or "(set api_key_env)",
            f"{p.rate_limit_rpm:g}" if p.rate_limit_rpm else "-",
        ]
        for p in PRESETS.values()
    ]
    print(render_table(["Preset", "API", "Base URL", "Key variable", "Req/min"], rows, right={4}))
    for p in PRESETS.values():
        print(f"\n{p.name}: {p.description}\n  Verified: {p.verification}")
        for n in p.notes:
            print(f"  - {n}")
        if p.docs_url:
            print(f"  Docs: {p.docs_url}")
    print(
        "\nAny other OpenAI-compatible endpoint: preset openai-compatible with its base_url (docs/providers.md)."
    )
    return 0


def cmd_datasets(a: argparse.Namespace) -> int:
    if not a.name:
        rows = []
        for name in builtin_names():
            ds = load_dataset(name)
            rows.append(
                [
                    name,
                    ds.version,
                    str(len(ds.workloads)),
                    str(sum(len(w.prompts) for w in ds.workloads)),
                ]
            )
        print(render_table(["Dataset", "Version", "Workloads", "Prompts"], rows))
        print("\nShow one: llm-bench datasets core-v1 [--prompts]")
        return 0
    try:
        ds = load_dataset(a.name, Path.cwd())
    except DatasetError as e:
        raise CliError("\n".join(e.problems)) from e
    print(f"{ds.id} {ds.version} ({ds.source}), sha256 {ds.sha256}\n{ds.description}\n")
    for w in ds.workloads:
        print(f"{w.id}  max_tokens {w.max_tokens}  {w.description}")
        for p in w.prompts:
            checks = ", ".join(c["type"] for c in p.checks) or "none"
            print(f"  - {p.id}  (checks: {checks})")
            if a.prompts:
                for m in p.messages:
                    text = m["content"].replace("\n", "\n      ")
                    print(f"      [{m['role']}] {text}")
    return 0


def _models_target(a: argparse.Namespace) -> tuple[str, str | None, str, str]:
    """(base_url, key, auth_scheme, auth_header) for the models command."""
    if a.config:
        try:
            cfg = load_config(a.config)
        except ConfigError as e:
            raise CliError(f"{a.config} is not valid:\n  " + "\n  ".join(e.problems)) from e
        chosen = [p for p in cfg.providers if not a.provider or p.name == a.provider]
        if len(chosen) != 1:
            raise CliError("--provider: name one endpoint from the config")
        p = chosen[0]
    else:
        data: dict[str, Any] = {
            "preset": a.preset or "openai-compatible",
            "name": "target",
            "model": "-",
        }
        if a.base_url:
            data["base_url"] = a.base_url
        if a.api_key_env:
            data["api_key_env"] = a.api_key_env
        try:
            p = parse_config({"providers": [data]}).providers[0]
        except ConfigError as e:
            raise CliError("\n".join(e.problems)) from e
    if p.is_synthetic:
        raise CliError("the synthetic endpoint has no model list")
    key = p.api_key() if p.needs_key else None
    if p.needs_key and not key:
        raise CliError(f"environment variable {p.api_key_env} is not set")
    REDACTOR.add(key)
    return p.base_url, key, p.auth_scheme, p.auth_header


def cmd_models(a: argparse.Namespace) -> int:
    base, key, scheme, header = _models_target(a)
    headers = {"Accept": "application/json", "User-Agent": f"llm-api-benchmark/{__version__}"}
    if key:
        headers[header] = f"Bearer {key}" if scheme == "bearer" else key
    try:
        resp = httpx.get(f"{base}/models", headers=headers, timeout=20)
    except httpx.HTTPError as e:
        raise CliError(f"GET {base}/models failed: {e}", code=1) from e
    if resp.status_code != 200:
        info = parse_error_body(resp.content, resp.headers.get("content-type", ""))
        detail = " ".join(x for x in (info.type, info.message) if x)
        raise CliError(f"GET {base}/models: HTTP {resp.status_code} {detail}", code=1)
    try:
        data = resp.json()
        items = data["data"] if isinstance(data, dict) else data
        rows = [
            [str(m["id"]), "" if m.get("available") is None else str(m["available"]).lower()]
            for m in items
        ]
    except (ValueError, KeyError, TypeError) as e:
        raise CliError(f"GET {base}/models: unexpected response shape ({e})", code=1) from e
    if any(r[1] for r in rows):
        print(render_table(["Model id", "Available"], rows, right=set()))
    else:
        print("\n".join(r[0] for r in rows))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    result = _load(a.results)
    renderers = {
        "table": terminal_report,
        "markdown": markdown_report,
        "html": html_report,
        "csv": summary_csv,
        "requests-csv": requests_csv,
        "json": to_json,
    }
    text = renderers[a.format](result)
    _emit(text, a.output)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    ra, rb = _load(a.a), _load(a.b)
    try:
        c = compare(
            ra, rb, provider_a=a.provider_a, provider_b=a.provider_b, min_samples=a.min_samples
        )
    except CompareError as e:
        raise CliError(str(e)) from e
    render = {"table": comparison_text, "markdown": comparison_markdown, "json": comparison_json}[
        a.format
    ]
    _emit(render(c), a.output)
    return 0


def cmd_init(a: argparse.Namespace) -> int:
    target = Path(a.path or f"{a.template}.yaml")
    if target.exists() and not a.force:
        raise CliError(f"{target} exists; pass --force to overwrite it")
    text = (resources.files("llm_api_benchmark") / "templates" / f"{a.template}.yaml").read_text(
        encoding="utf-8"
    )
    target.write_text(text, encoding="utf-8", newline="\n")
    err(f"Wrote {target}. Next: llm-bench run --config {target} --dry-run")
    return 0


def _load(path: str) -> RunResult:
    try:
        return load_results(path)
    except ResultsError as e:
        raise CliError(str(e)) from e


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8", newline="\n")
        err(f"Wrote {output}")
    else:
        print(text, end="")


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="llm-bench",
        description="Bounded, reproducible benchmarks for OpenAI-compatible LLM APIs.",
        epilog="Docs: https://github.com/Moloch-AI/llm-api-benchmark",
    )
    ap.add_argument("--version", action="version", version=f"llm-api-benchmark {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = sub.add_parser("run", help="run a benchmark from a configuration file")
    run.add_argument("-c", "--config", required=True, help="YAML or JSON configuration file")
    run.add_argument(
        "--dry-run", action="store_true", help="show the plan and checks; send nothing"
    )
    run.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help=f"do not ask before sending more than {CONFIRM_ABOVE} requests",
    )
    run.add_argument(
        "-o", "--out-dir", default="results", help="where result files go (default: results)"
    )
    run.add_argument(
        "--formats", default="json,csv", help="any of json,csv,md,html (json is always written)"
    )
    run.add_argument("--providers", help="only these endpoint names (comma-separated)")
    run.add_argument("--workloads", help="only these workloads (comma-separated)")
    run.add_argument("--modes", help="stream, plain or stream,plain")
    run.add_argument("--repetitions", type=int, help="repetitions per prompt")
    run.add_argument("--warmup", type=int, help="warm-up requests per endpoint")
    run.add_argument("--concurrency", type=int, help=f"requests in flight (1 to {MAX_CONCURRENCY})")
    run.add_argument("--max-requests", type=int, help="refuse plans larger than this")
    run.add_argument(
        "--budget", type=float, help="spending limit in the budget currency (needs pricing)"
    )
    run.add_argument(
        "--stop-at-budget",
        action="store_true",
        help="run even if the worst case exceeds the budget, stopping before it would be exceeded",
    )
    run.add_argument("-q", "--quiet", action="store_true", help="no per-request progress lines")
    run.set_defaults(fn=cmd_run)

    prov = sub.add_parser("providers", help="list built-in provider presets")
    prov.add_argument("--json", action="store_true")
    prov.set_defaults(fn=cmd_providers)

    ds = sub.add_parser("datasets", help="list datasets, or show one's workloads and prompts")
    ds.add_argument("name", nargs="?", help="built-in name (core-v1) or a file path")
    ds.add_argument("--prompts", action="store_true", help="print the prompt text")
    ds.set_defaults(fn=cmd_datasets)

    mod = sub.add_parser("models", help="list an endpoint's model ids (one GET /models request)")
    mod.add_argument("--preset", choices=[p for p in PRESETS if p != "synthetic"])
    mod.add_argument("--base-url")
    mod.add_argument("--api-key-env")
    mod.add_argument("-c", "--config", help="take the endpoint from a configuration file")
    mod.add_argument("--provider", help="endpoint name in the configuration")
    mod.set_defaults(fn=cmd_models)

    rep = sub.add_parser("report", help="render a results file")
    rep.add_argument("results")
    rep.add_argument(
        "-f",
        "--format",
        default="table",
        choices=["table", "markdown", "html", "csv", "requests-csv", "json"],
    )
    rep.add_argument("-o", "--output", help="write to a file instead of stdout")
    rep.set_defaults(fn=cmd_report)

    cmp_ = sub.add_parser("compare", help="compare two results files (or two endpoints in one)")
    cmp_.add_argument("a", help="results file A (the baseline)")
    cmp_.add_argument("b", help="results file B (may be the same file as A)")
    cmp_.add_argument("--provider-a", help="endpoint name in A")
    cmp_.add_argument("--provider-b", help="endpoint name in B")
    cmp_.add_argument(
        "--min-samples", type=int, default=5, help="fewer successful samples give 'too few samples'"
    )
    cmp_.add_argument("-f", "--format", default="table", choices=["table", "markdown", "json"])
    cmp_.add_argument("-o", "--output")
    cmp_.set_defaults(fn=cmd_compare)

    ini = sub.add_parser("init", help="write a starter configuration file")
    ini.add_argument("template", nargs="?", default="synthetic-demo", choices=TEMPLATES)
    ini.add_argument("path", nargs="?", help="file to write (default: <template>.yaml)")
    ini.add_argument("--force", action="store_true")
    ini.set_defaults(fn=cmd_init)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        code: int = args.fn(args)
    except CliError as e:
        err(f"llm-bench: {e}")
        return e.code
    except KeyboardInterrupt:
        err("Interrupted.")
        return 130
    return code
