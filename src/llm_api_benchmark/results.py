"""Result files: assembly, JSON and CSV output, and loading for reports and comparisons."""

from __future__ import annotations

import csv
import io
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ._version import __version__
from .metrics import GroupSummary, ProviderTotals, provider_totals, summarize
from .records import CSV_COLUMNS, RequestRecord

if TYPE_CHECKING:
    from .runner import Runner

SCHEMA = "llm-api-benchmark/results/v1"


class ResultsError(ValueError):
    pass


@dataclass
class RunResult:
    run: dict[str, Any]
    config: dict[str, Any]
    dataset: dict[str, Any]
    providers: list[dict[str, Any]]
    environment: dict[str, Any]
    requests: list[RequestRecord]
    warnings: list[str] = field(default_factory=list)
    tool: dict[str, Any] = field(
        default_factory=lambda: {"name": "llm-api-benchmark", "version": __version__}
    )

    @property
    def synthetic(self) -> bool:
        return bool(self.run.get("synthetic"))

    def _order(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        providers = {p.get("name", ""): i for i, p in enumerate(self.providers)}
        workloads = {w: i for i, w in enumerate(self.dataset.get("workloads", {}))}
        modes = {m: i for i, m in enumerate(self.config.get("modes", []))}
        return providers, workloads, modes

    def summaries(self) -> list[GroupSummary]:
        """Groups in a stable order: workload as in the dataset, then endpoint, then mode."""
        prov, wl, modes = self._order()
        return sorted(
            summarize(self.requests),
            key=lambda s: (wl.get(s.workload, 99), prov.get(s.provider, 99), modes.get(s.mode, 9)),
        )

    def totals(self) -> list[ProviderTotals]:
        prov, _, _ = self._order()
        return sorted(provider_totals(self.requests), key=lambda t: prov.get(t.provider, 99))

    def provider(self, name: str) -> dict[str, Any]:
        for p in self.providers:
            if p.get("name") == name:
                return p
        return {}

    def limitations(self) -> list[str]:
        return limitations(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "tool": self.tool,
            "run": self.run,
            "environment": self.environment,
            "config": self.config,
            "dataset": self.dataset,
            "providers": self.providers,
            "warnings": self.warnings,
            "limitations": self.limitations(),
            "summary": [s.to_dict() for s in self.summaries()],
            "provider_totals": [t.to_dict() for t in self.totals()],
            "requests": [r.to_dict() for r in self.requests],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ResultsError(f"not an llm-api-benchmark results file (expected schema {SCHEMA})")
        try:
            return cls(
                run=data["run"],
                config=data["config"],
                dataset=data["dataset"],
                providers=data["providers"],
                environment=data.get("environment", {}),
                requests=[RequestRecord.from_dict(r) for r in data["requests"]],
                warnings=list(data.get("warnings", [])),
                tool=data.get("tool", {}),
            )
        except (KeyError, TypeError) as err:
            raise ResultsError(f"the results file is incomplete ({err})") from err


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(terse=True),
        "httpx": httpx.__version__,
    }


def load_results(path: str | Path) -> RunResult:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as err:
        raise ResultsError(f"{p}: cannot read the file ({err.strerror or err})") from err
    except ValueError as err:
        raise ResultsError(f"{p}: not valid JSON ({err})") from err
    return RunResult.from_dict(data)


def to_json(result: RunResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"


def requests_csv(result: RunResult) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in result.requests:
        row = r.to_dict()
        w.writerow({k: _csv_value(row.get(k)) for k in CSV_COLUMNS})
    return buf.getvalue()


SUMMARY_COLUMNS = [
    "provider",
    "model",
    "workload",
    "mode",
    "planned",
    "executed",
    "ok",
    "error_rate",
    "errors",
    "ttft_n",
    "ttft_median_s",
    "ttft_p90_s",
    "ttft_p95_s",
    "ttfc_median_s",
    "latency_n",
    "latency_min_s",
    "latency_median_s",
    "latency_p90_s",
    "latency_p95_s",
    "latency_max_s",
    "output_tps_median",
    "decode_tps_median",
    "completion_tokens_median",
    "tokens_estimated",
    "cost",
    "cost_complete",
    "check_pass_rate",
    "checks_run",
]


def summary_csv(result: RunResult) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    w.writeheader()
    for s in result.summaries():
        w.writerow(
            {
                "provider": s.provider,
                "model": s.model,
                "workload": s.workload,
                "mode": s.mode,
                "planned": s.planned,
                "executed": s.executed,
                "ok": s.ok,
                "error_rate": _csv_value(None if s.error_rate is None else round(s.error_rate, 4)),
                "errors": ";".join(f"{k}={v}" for k, v in s.errors.items()),
                "ttft_n": s.ttft.n if s.ttft else 0,
                "ttft_median_s": _stat(s.ttft, "median"),
                "ttft_p90_s": _stat(s.ttft, "p90"),
                "ttft_p95_s": _stat(s.ttft, "p95"),
                "ttfc_median_s": _stat(s.ttfc, "median"),
                "latency_n": s.latency.n if s.latency else 0,
                "latency_min_s": _stat(s.latency, "min"),
                "latency_median_s": _stat(s.latency, "median"),
                "latency_p90_s": _stat(s.latency, "p90"),
                "latency_p95_s": _stat(s.latency, "p95"),
                "latency_max_s": _stat(s.latency, "max"),
                "output_tps_median": _stat(s.output_tps, "median", 2),
                "decode_tps_median": _stat(s.decode_tps, "median", 2),
                "completion_tokens_median": _stat(s.completion_tokens, "median", 1),
                "tokens_estimated": s.tokens_estimated,
                "cost": _csv_value(None if s.cost is None else round(s.cost, 6)),
                "cost_complete": s.cost_complete,
                "check_pass_rate": _csv_value(
                    None if s.check_pass_rate is None else round(s.check_pass_rate, 4)
                ),
                "checks_run": s.checks_run,
            }
        )
    return buf.getvalue()


def _stat(stat: Any, attr: str, digits: int = 4) -> str:
    return "" if stat is None else str(round(getattr(stat, attr), digits))


def _csv_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, list | dict):
        return json.dumps(v, ensure_ascii=False)
    return v


def limitations(result: RunResult) -> list[str]:
    """Plain statements about what these numbers can and cannot show."""
    out = []
    if result.synthetic:
        out.append(
            "SYNTHETIC FIXTURE DATA: at least one endpoint is the in-process simulator. These "
            "numbers are not the performance of any real API."
        )
    out.append(
        "Measured from one client, at one place and time, over its network path. Latency includes "
        "DNS, TLS on new connections, the network and any queueing at the provider."
    )
    stream = [r for r in result.requests if r.mode == "stream" and r.ok]
    if stream:
        out.append(
            "TTFT is the time to the first streamed text chunk (reasoning or answer); TTFC is the "
            "time to the first answer chunk. Providers stream text chunks, not single tokens."
        )
    if any(r.tokens_estimated for r in result.requests if r.ok):
        out.append(
            "Some token counts were not reported by the provider and are estimated at 4 characters "
            "per token; throughput and cost for those requests are approximate."
        )
    if any(r.reasoning_chars or r.reasoning_tokens for r in result.requests if r.ok):
        out.append(
            "Some replies include reasoning text. Output tokens/s counts every completion token the "
            "provider reports, reasoning included."
        )
    small = [s for s in result.summaries() if s.latency and s.latency.n < 20]
    if small:
        out.append(
            "Groups with fewer than 20 successful samples have unstable P90/P95 values; treat tail "
            "latencies there as indicative only."
        )
    for p in result.providers:
        if not p.get("max_tokens_honored", True):
            out.append(
                f"{p['name']}: the provider does not apply max_tokens, so reply lengths (and latency) "
                "are not capped by the workload limits."
            )
    out.append(
        "Speed and check results say nothing about answer quality beyond the listed mechanical "
        "checks."
    )
    return out


def build_result(runner: Runner, warnings: list[str] | None = None) -> RunResult:
    cfg = runner.cfg
    started = runner.started_at
    finished = runner.finished_at
    duration = (finished - started).total_seconds() if started and finished else None
    return RunResult(
        run={
            "name": cfg.name,
            "started_at": started.isoformat(timespec="seconds") if started else None,
            "finished_at": finished.isoformat(timespec="seconds") if finished else None,
            "duration_s": None if duration is None else round(duration, 2),
            "interrupted": runner.interrupted,
            "stop_reason": runner.stop_reason,
            "synthetic": any(p.is_synthetic for p in cfg.providers),
            "planned_requests": len(runner.plan),
            "spent": round(runner.budget.spent, 6),
        },
        config=cfg.to_public_dict(),
        dataset={
            "id": runner.dataset.id,
            "version": runner.dataset.version,
            "sha256": runner.dataset.sha256,
            "source": runner.dataset.source,
            "workloads": {
                w.id: {
                    "max_tokens": w.max_tokens,
                    "prompts": len(w.prompts),
                    "description": w.description,
                }
                for w in runner.dataset.select(cfg.workloads)
            },
        },
        providers=[p.to_public_dict() for p in cfg.providers],
        environment=environment(),
        requests=runner.ordered_records(),
        warnings=list(warnings or []),
    )
