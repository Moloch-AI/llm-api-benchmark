"""Benchmark configuration: loading, preset merging and validation.

A configuration file (YAML or JSON) names the endpoints to measure, the dataset and workloads,
and the bounds of the run. Validation is strict: unknown keys, wrong types and unsafe values are
errors with the exact path, so a typo cannot silently change what is measured.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .providers.presets import PRESETS, Preset

MODES = ("stream", "plain")
AUTH_SCHEMES = ("bearer", "header", "none")
MAX_TOKENS_PARAMS = ("max_tokens", "max_completion_tokens", "none")

# Hard ceilings. This is a measurement tool, not a load generator.
MAX_CONCURRENCY = 8
MAX_REQUESTS_CEILING = 5000
MAX_TIMEOUT_S = 900.0

DEFAULT_MAX_REQUESTS = 100
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_CONNECT_TIMEOUT_S = 10.0


class ConfigError(ValueError):
    """One or more problems in a configuration file; each message names its path."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


@dataclass(frozen=True)
class Pricing:
    """User-supplied prices per million tokens. Never shipped with the tool."""

    input_per_mtok: float
    output_per_mtok: float
    currency: str
    source: str
    checked_on: date | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_per_mtok": self.input_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "currency": self.currency,
            "source": self.source,
            "checked_on": self.checked_on.isoformat() if self.checked_on else None,
        }


@dataclass(frozen=True)
class SyntheticProfile:
    """Parameters of the in-process synthetic endpoint used for demos and tests."""

    ttft_ms: float = 350.0
    tokens_per_s: float = 60.0
    jitter: float = 0.25
    error_rate: float = 0.0
    output_tokens: int = 120
    time_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "tokens_per_s": self.tokens_per_s,
            "jitter": self.jitter,
            "error_rate": self.error_rate,
            "output_tokens": self.output_tokens,
            "time_scale": self.time_scale,
        }


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    preset: str
    api: str
    base_url: str
    model: str
    api_key_env: str | None
    auth_scheme: str
    auth_header: str
    headers: dict[str, str]
    headers_env: dict[str, str]
    query: dict[str, str]
    params: dict[str, Any]
    pricing: Pricing | None
    rate_limit_rpm: float | None
    timeout_s: float | None
    max_tokens_param: str
    max_tokens_honored: bool
    assumed_max_output_tokens: int | None
    stream_usage: bool
    notes: tuple[str, ...]
    synthetic: SyntheticProfile | None = None

    @property
    def is_synthetic(self) -> bool:
        return self.api == "synthetic"

    @property
    def needs_key(self) -> bool:
        return self.auth_scheme != "none" and not self.is_synthetic

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None

    def to_public_dict(self) -> dict[str, Any]:
        """Everything about the endpoint except credentials (env var names only)."""
        from .redact import redact_url

        return {
            "name": self.name,
            "preset": self.preset,
            "api": self.api,
            "base_url": redact_url(self.base_url),
            "model": self.model,
            "api_key_env": self.api_key_env,
            "auth_scheme": self.auth_scheme,
            "headers": sorted([*self.headers, *self.headers_env]),
            "params": self.params,
            "pricing": self.pricing.to_dict() if self.pricing else None,
            "rate_limit_rpm": self.rate_limit_rpm,
            "timeout_s": self.timeout_s,
            "max_tokens_param": self.max_tokens_param,
            "max_tokens_honored": self.max_tokens_honored,
            "assumed_max_output_tokens": self.assumed_max_output_tokens,
            "stream_usage": self.stream_usage,
            "notes": list(self.notes),
            "synthetic": self.synthetic.to_dict() if self.synthetic else None,
        }


@dataclass(frozen=True)
class RunConfig:
    name: str
    dataset: str
    workloads: tuple[str, ...] | None
    modes: tuple[str, ...]
    repetitions: int
    warmup: int
    concurrency: int
    timeout_s: float
    connect_timeout_s: float
    max_requests: int
    budget: float | None
    budget_currency: str
    temperature: float | None
    max_tokens: int | None
    seed: int
    checks: bool
    save_outputs: bool
    max_consecutive_failures: int
    providers: tuple[ProviderConfig, ...]
    source_path: Path | None = None
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def dataset_base_dir(self) -> Path:
        return self.source_path.parent if self.source_path else Path.cwd()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "dataset": self.dataset,
            "workloads": list(self.workloads) if self.workloads else None,
            "modes": list(self.modes),
            "repetitions": self.repetitions,
            "warmup": self.warmup,
            "concurrency": self.concurrency,
            "timeout_s": self.timeout_s,
            "connect_timeout_s": self.connect_timeout_s,
            "max_requests": self.max_requests,
            "budget": self.budget,
            "budget_currency": self.budget_currency,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "checks": self.checks,
            "save_outputs": self.save_outputs,
            "max_consecutive_failures": self.max_consecutive_failures,
            "source": self.source_path.name if self.source_path else None,
        }

    def with_overrides(self, **changes: Any) -> RunConfig:
        """Apply command-line overrides, re-checking the same bounds as the file."""
        clean = {k: v for k, v in changes.items() if v is not None}
        if not clean:
            return self
        updated = replace(self, **clean)
        problems = _check_run_bounds(updated) + _check_budget_pricing(updated)
        if problems:
            raise ConfigError(problems)
        return updated


# --------------------------------------------------------------------------- loading


def load_config(path: str | Path) -> RunConfig:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as err:
        raise ConfigError([f"{p}: cannot read the file ({err.strerror or err})"]) from err
    try:
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as err:
        raise ConfigError(
            [f"{p}: not valid {'JSON' if p.suffix == '.json' else 'YAML'}: {err}"]
        ) from err
    return parse_config(data, source_path=p)


_TOP_KEYS = {
    "version",
    "name",
    "description",
    "dataset",
    "workloads",
    "modes",
    "repetitions",
    "warmup",
    "concurrency",
    "timeout_s",
    "connect_timeout_s",
    "max_requests",
    "budget",
    "temperature",
    "max_tokens",
    "seed",
    "checks",
    "save_outputs",
    "max_consecutive_failures",
    "providers",
}
_BUDGET_KEYS = {"limit", "currency"}
_PROVIDER_KEYS = {
    "name",
    "preset",
    "base_url",
    "model",
    "api_key_env",
    "auth_scheme",
    "auth_header",
    "headers",
    "headers_env",
    "query",
    "params",
    "pricing",
    "rate_limit_rpm",
    "timeout_s",
    "max_tokens_param",
    "max_tokens_honored",
    "assumed_max_output_tokens",
    "stream_usage",
    "synthetic",
}
_PRICING_KEYS = {"input_per_mtok", "output_per_mtok", "currency", "source", "checked_on"}
_SYNTHETIC_KEYS = {"ttft_ms", "tokens_per_s", "jitter", "error_rate", "output_tokens", "time_scale"}
# Headers that would carry credentials must come from headers_env, never from the file.
_SECRET_HEADER = re.compile(r"(?i)^(authorization|proxy-authorization|x-api-key|api-key|cookie)$")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _Checker:
    def __init__(self) -> None:
        self.problems: list[str] = []

    def error(self, path: str, message: str) -> None:
        self.problems.append(f"{path}: {message}")

    def keys(self, obj: Mapping[str, Any], allowed: set[str], path: str) -> None:
        for key in obj:
            if key not in allowed:
                hint = difflib.get_close_matches(str(key), sorted(allowed), n=1)
                extra = f" (did you mean '{hint[0]}'?)" if hint else ""
                self.error(f"{path}.{key}" if path else str(key), f"unknown key{extra}")

    def mapping(self, value: Any, path: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.error(path, "must be a mapping")
            return {}
        return value

    def string(self, value: Any, path: str, *, required: bool = False) -> str | None:
        if value is None:
            if required:
                self.error(path, "is required")
            return None
        if not isinstance(value, str) or not value.strip():
            self.error(path, "must be a non-empty string")
            return None
        return value.strip()

    def number(
        self,
        value: Any,
        path: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_min: bool = False,
    ) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            self.error(path, "must be a number")
            return None
        v = float(value)
        if minimum is not None and (v <= minimum if exclusive_min else v < minimum):
            self.error(
                path, f"must be {'greater than' if exclusive_min else 'at least'} {minimum:g}"
            )
            return None
        if maximum is not None and v > maximum:
            self.error(path, f"must be at most {maximum:g}")
            return None
        return v

    def integer(
        self, value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            self.error(path, "must be a whole number")
            return None
        if minimum is not None and value < minimum:
            self.error(path, f"must be at least {minimum}")
            return None
        if maximum is not None and value > maximum:
            self.error(path, f"must be at most {maximum}")
            return None
        return value

    def boolean(self, value: Any, path: str) -> bool | None:
        if value is None:
            return None
        if not isinstance(value, bool):
            self.error(path, "must be true or false")
            return None
        return value

    def str_map(self, value: Any, path: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in self.mapping(value, path).items():
            if not isinstance(v, str | int | float) or isinstance(v, bool):
                self.error(f"{path}.{k}", "must be a string")
                continue
            out[str(k)] = str(v)
        return out

    def str_list(self, value: Any, path: str) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            self.error(path, "must be a list of strings")
            return None
        return [v.strip() for v in value]


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}``. Only used for base_url and model."""

    def sub(m: re.Match[str]) -> str:
        found = os.environ.get(m.group(1), "").strip()
        if found:
            return found
        if m.group(2) is not None:
            return m.group(2)
        raise KeyError(m.group(1))

    return _ENV_REF.sub(sub, value)


def parse_config(data: Any, *, source_path: Path | None = None) -> RunConfig:
    c = _Checker()
    if not isinstance(data, dict):
        raise ConfigError(["the configuration must be a mapping at the top level"])
    c.keys(data, _TOP_KEYS, "")
    version = data.get("version", 1)
    if version != 1:
        c.error("version", "only version 1 is supported")

    name = c.string(data.get("name"), "name") or (source_path.stem if source_path else "benchmark")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        c.error("name", "use letters, digits, '.', '_' or '-' (at most 64 characters)")
    description = c.string(data.get("description"), "description") or ""
    dataset = c.string(data.get("dataset"), "dataset") or "core-v1"
    workloads = c.str_list(data.get("workloads"), "workloads")

    modes_raw = c.str_list(data.get("modes"), "modes") or ["stream"]
    for m in modes_raw:
        if m not in MODES:
            c.error("modes", f"'{m}' is not one of {', '.join(MODES)}")
    modes = tuple(dict.fromkeys(modes_raw))

    budget_limit: float | None = None
    budget_currency = "USD"
    budget_raw = data.get("budget")
    if isinstance(budget_raw, dict):
        c.keys(budget_raw, _BUDGET_KEYS, "budget")
        budget_limit = c.number(
            budget_raw.get("limit"), "budget.limit", minimum=0, exclusive_min=True
        )
        budget_currency = c.string(budget_raw.get("currency"), "budget.currency") or "USD"
    elif budget_raw is not None:
        budget_limit = c.number(budget_raw, "budget", minimum=0, exclusive_min=True)

    providers_raw = data.get("providers")
    providers: list[ProviderConfig] = []
    if not isinstance(providers_raw, list) or not providers_raw:
        c.error("providers", "list at least one endpoint to measure")
    else:
        for i, item in enumerate(providers_raw):
            prov = _parse_provider(c, item, f"providers[{i}]")
            if prov:
                providers.append(prov)
        names = [p.name for p in providers]
        for dup in sorted({n for n in names if names.count(n) > 1}):
            c.error("providers", f"the name '{dup}' is used more than once")

    cfg = RunConfig(
        name=name,
        description=description,
        dataset=dataset,
        workloads=tuple(workloads) if workloads else None,
        modes=modes or ("stream",),
        repetitions=_default(
            c.integer(data.get("repetitions"), "repetitions", minimum=1, maximum=1000), 3
        ),
        warmup=_default(c.integer(data.get("warmup"), "warmup", minimum=0, maximum=10), 1),
        concurrency=_default(
            c.integer(data.get("concurrency"), "concurrency", minimum=1, maximum=MAX_CONCURRENCY), 1
        ),
        timeout_s=_default(
            c.number(data.get("timeout_s"), "timeout_s", minimum=1, maximum=MAX_TIMEOUT_S),
            DEFAULT_TIMEOUT_S,
        ),
        connect_timeout_s=_default(
            c.number(data.get("connect_timeout_s"), "connect_timeout_s", minimum=0.5, maximum=120),
            DEFAULT_CONNECT_TIMEOUT_S,
        ),
        max_requests=_default(
            c.integer(
                data.get("max_requests"), "max_requests", minimum=1, maximum=MAX_REQUESTS_CEILING
            ),
            DEFAULT_MAX_REQUESTS,
        ),
        budget=budget_limit,
        budget_currency=budget_currency,
        temperature=c.number(data.get("temperature"), "temperature", minimum=0, maximum=2),
        max_tokens=c.integer(data.get("max_tokens"), "max_tokens", minimum=1, maximum=1_000_000),
        seed=_default(c.integer(data.get("seed"), "seed", minimum=0), 0),
        checks=_default(c.boolean(data.get("checks"), "checks"), True),
        save_outputs=_default(c.boolean(data.get("save_outputs"), "save_outputs"), False),
        max_consecutive_failures=_default(
            c.integer(data.get("max_consecutive_failures"), "max_consecutive_failures", minimum=1),
            5,
        ),
        providers=tuple(providers),
        source_path=source_path,
    )
    c.problems.extend(_check_run_bounds(cfg))
    c.problems.extend(_check_budget_pricing(cfg))
    if c.problems:
        raise ConfigError(c.problems)
    return cfg


def _check_budget_pricing(cfg: RunConfig) -> list[str]:
    if cfg.budget is None:
        return []
    problems = []
    for p in cfg.providers:
        if p.is_synthetic:
            continue
        if p.pricing is None:
            problems.append(
                f"providers[{p.name}].pricing: a budget is set, so every endpoint needs pricing "
                "(the budget cannot be enforced without it); remove the budget or add pricing"
            )
        elif p.pricing.currency != cfg.budget_currency:
            problems.append(
                f"providers[{p.name}].pricing.currency: is {p.pricing.currency} but the budget is "
                f"in {cfg.budget_currency}"
            )
    return problems


def _check_run_bounds(cfg: RunConfig) -> list[str]:
    problems = []
    if not 1 <= cfg.concurrency <= MAX_CONCURRENCY:
        problems.append(f"concurrency: must be between 1 and {MAX_CONCURRENCY}")
    if not 1 <= cfg.max_requests <= MAX_REQUESTS_CEILING:
        problems.append(f"max_requests: must be between 1 and {MAX_REQUESTS_CEILING}")
    if not 1 <= cfg.timeout_s <= MAX_TIMEOUT_S:
        problems.append(f"timeout_s: must be between 1 and {MAX_TIMEOUT_S:g}")
    if cfg.repetitions < 1:
        problems.append("repetitions: must be at least 1")
    if cfg.warmup < 0:
        problems.append("warmup: must be 0 or more")
    if cfg.budget is not None and cfg.budget <= 0:
        problems.append("budget: must be greater than 0")
    return problems


T = TypeVar("T")


def _default(value: T | None, fallback: T) -> T:
    return fallback if value is None else value


def _parse_provider(c: _Checker, item: Any, path: str) -> ProviderConfig | None:
    if not isinstance(item, dict):
        c.error(path, "must be a mapping")
        return None
    c.keys(item, _PROVIDER_KEYS, path)
    preset_name = c.string(item.get("preset"), f"{path}.preset") or "openai-compatible"
    preset: Preset | None = PRESETS.get(preset_name)
    if preset is None:
        hint = difflib.get_close_matches(preset_name, sorted(PRESETS), n=1)
        c.error(
            f"{path}.preset",
            f"unknown preset '{preset_name}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""),
        )
        return None

    name = c.string(item.get("name"), f"{path}.name", required=True) or "?"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,63}", name):
        c.error(
            f"{path}.name", "use letters, digits, spaces, '.', '_' or '-' (at most 64 characters)"
        )

    def pick(key: str) -> Any:
        return item[key] if key in item else getattr(preset, key)

    raw_base = c.string(pick("base_url"), f"{path}.base_url", required=preset.api != "synthetic")
    base_url = ""
    if raw_base:
        try:
            base_url = expand_env(raw_base).rstrip("/")
        except KeyError as missing:
            c.error(f"{path}.base_url", f"environment variable {missing.args[0]} is not set")
        if base_url and preset.api != "synthetic" and not re.match(r"^https?://[^/\s]+", base_url):
            c.error(f"{path}.base_url", "must start with http:// or https://")

    raw_model = c.string(item.get("model"), f"{path}.model", required=True)
    model = ""
    if raw_model:
        try:
            model = expand_env(raw_model)
        except KeyError as missing:
            c.error(f"{path}.model", f"environment variable {missing.args[0]} is not set")

    auth_scheme = c.string(pick("auth_scheme"), f"{path}.auth_scheme") or "bearer"
    if auth_scheme not in AUTH_SCHEMES:
        c.error(f"{path}.auth_scheme", f"must be one of {', '.join(AUTH_SCHEMES)}")
    api_key_env = c.string(pick("api_key_env"), f"{path}.api_key_env")
    if api_key_env and not _ENV_NAME.match(api_key_env):
        c.error(f"{path}.api_key_env", "must be the NAME of an environment variable, not a key")
    if preset.api != "synthetic" and auth_scheme != "none" and not api_key_env:
        c.error(
            f"{path}.api_key_env",
            "name the environment variable that holds the key (or set auth_scheme: none)",
        )
    auth_header = c.string(pick("auth_header"), f"{path}.auth_header") or "Authorization"

    headers = c.str_map(item.get("headers"), f"{path}.headers")
    for h in headers:
        if _SECRET_HEADER.match(h):
            c.error(
                f"{path}.headers.{h}",
                "credentials must not be written in the file; use headers_env",
            )
    headers_env = c.str_map(item.get("headers_env"), f"{path}.headers_env")
    for h, env in headers_env.items():
        if not _ENV_NAME.match(env):
            c.error(f"{path}.headers_env.{h}", "must be the NAME of an environment variable")
    query = c.str_map(item.get("query"), f"{path}.query")
    params = c.mapping(item.get("params"), f"{path}.params")
    for reserved in ("model", "messages", "stream"):
        if reserved in params:
            c.error(f"{path}.params.{reserved}", "is set by the benchmark and cannot be overridden")

    max_tokens_param = (
        c.string(pick("max_tokens_param"), f"{path}.max_tokens_param") or "max_tokens"
    )
    if max_tokens_param not in MAX_TOKENS_PARAMS:
        c.error(f"{path}.max_tokens_param", f"must be one of {', '.join(MAX_TOKENS_PARAMS)}")

    pricing = _parse_pricing(c, item.get("pricing"), f"{path}.pricing")
    synthetic = None
    if preset.api == "synthetic":
        synthetic = _parse_synthetic(c, item.get("synthetic"), f"{path}.synthetic")
    elif "synthetic" in item:
        c.error(f"{path}.synthetic", "only valid with preset: synthetic")

    honored = c.boolean(pick("max_tokens_honored"), f"{path}.max_tokens_honored")
    return ProviderConfig(
        name=name,
        preset=preset_name,
        api=preset.api,
        base_url=base_url or ("synthetic://local" if preset.api == "synthetic" else ""),
        model=model,
        api_key_env=api_key_env,
        auth_scheme=auth_scheme,
        auth_header=auth_header,
        headers=headers,
        headers_env=headers_env,
        query=query,
        params=dict(params),
        pricing=pricing,
        rate_limit_rpm=c.number(
            pick("rate_limit_rpm"), f"{path}.rate_limit_rpm", minimum=0, exclusive_min=True
        ),
        timeout_s=c.number(
            item.get("timeout_s"), f"{path}.timeout_s", minimum=1, maximum=MAX_TIMEOUT_S
        ),
        max_tokens_param=max_tokens_param,
        max_tokens_honored=True if honored is None else honored,
        assumed_max_output_tokens=c.integer(
            pick("assumed_max_output_tokens"), f"{path}.assumed_max_output_tokens", minimum=1
        ),
        stream_usage=_default(c.boolean(pick("stream_usage"), f"{path}.stream_usage"), True),
        notes=tuple(preset.notes),
        synthetic=synthetic,
    )


def _parse_pricing(c: _Checker, value: Any, path: str) -> Pricing | None:
    if value is None:
        return None
    raw = c.mapping(value, path)
    c.keys(raw, _PRICING_KEYS, path)
    inp = c.number(raw.get("input_per_mtok"), f"{path}.input_per_mtok", minimum=0)
    out = c.number(raw.get("output_per_mtok"), f"{path}.output_per_mtok", minimum=0)
    if inp is None and "input_per_mtok" not in raw:
        c.error(f"{path}.input_per_mtok", "is required")
    if out is None and "output_per_mtok" not in raw:
        c.error(f"{path}.output_per_mtok", "is required")
    source = c.string(raw.get("source"), f"{path}.source", required=True)
    currency = (c.string(raw.get("currency"), f"{path}.currency") or "USD").upper()
    checked = raw.get("checked_on")
    checked_on: date | None = None
    if isinstance(checked, date):
        checked_on = checked
    elif isinstance(checked, str):
        try:
            checked_on = date.fromisoformat(checked)
        except ValueError:
            c.error(f"{path}.checked_on", "must be a date like 2026-11-17")
    elif checked is not None:
        c.error(f"{path}.checked_on", "must be a date like 2026-11-17")
    if inp is None or out is None or source is None:
        return None
    return Pricing(inp, out, currency, source, checked_on)


def _parse_synthetic(c: _Checker, value: Any, path: str) -> SyntheticProfile:
    raw = c.mapping(value, path)
    c.keys(raw, _SYNTHETIC_KEYS, path)
    d = SyntheticProfile()
    return SyntheticProfile(
        ttft_ms=_default(
            c.number(raw.get("ttft_ms"), f"{path}.ttft_ms", minimum=0, maximum=60_000), d.ttft_ms
        ),
        tokens_per_s=_default(
            c.number(
                raw.get("tokens_per_s"),
                f"{path}.tokens_per_s",
                minimum=0,
                exclusive_min=True,
                maximum=100_000,
            ),
            d.tokens_per_s,
        ),
        jitter=_default(
            c.number(raw.get("jitter"), f"{path}.jitter", minimum=0, maximum=1), d.jitter
        ),
        error_rate=_default(
            c.number(raw.get("error_rate"), f"{path}.error_rate", minimum=0, maximum=1),
            d.error_rate,
        ),
        output_tokens=_default(
            c.integer(
                raw.get("output_tokens"), f"{path}.output_tokens", minimum=1, maximum=100_000
            ),
            d.output_tokens,
        ),
        time_scale=_default(
            c.number(
                raw.get("time_scale"),
                f"{path}.time_scale",
                minimum=0,
                exclusive_min=True,
                maximum=10,
            ),
            d.time_scale,
        ),
    )
