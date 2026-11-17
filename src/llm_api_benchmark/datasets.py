"""Versioned prompt datasets.

A dataset groups prompts into workloads (short answers, code generation, long replies...). Each
workload has its own output-token limit, and results are always reported per workload: one
prompt type never stands in for "speed" in general.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .checks import validate_spec

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_ROLES = {"system", "user", "assistant"}


class DatasetError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


@dataclass(frozen=True)
class Prompt:
    id: str
    messages: tuple[dict[str, str], ...]
    checks: tuple[dict[str, Any], ...]

    def message_list(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self.messages]

    def chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)


@dataclass(frozen=True)
class Workload:
    id: str
    description: str
    max_tokens: int
    prompts: tuple[Prompt, ...]


@dataclass(frozen=True)
class Dataset:
    id: str
    version: str
    description: str
    workloads: tuple[Workload, ...]
    sha256: str
    source: str

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def workload(self, workload_id: str) -> Workload:
        for w in self.workloads:
            if w.id == workload_id:
                return w
        raise KeyError(workload_id)

    def select(self, workload_ids: tuple[str, ...] | None) -> tuple[Workload, ...]:
        if not workload_ids:
            return self.workloads
        known = {w.id for w in self.workloads}
        unknown = [w for w in workload_ids if w not in known]
        if unknown:
            raise DatasetError(
                [
                    f"workloads: {', '.join(unknown)} not in dataset {self.ref} (known: {', '.join(sorted(known))})"
                ]
            )
        return tuple(self.workload(w) for w in workload_ids)


def builtin_names() -> list[str]:
    root = resources.files("llm_api_benchmark") / "datasets"
    return sorted(p.name.removesuffix(".yaml") for p in root.iterdir() if p.name.endswith(".yaml"))


def load_dataset(ref: str, base_dir: Path | None = None) -> Dataset:
    """Load a built-in dataset by name (``core-v1``) or a YAML/JSON file by path."""
    looks_like_path = any(sep in ref for sep in ("/", "\\")) or ref.endswith(
        (".yaml", ".yml", ".json")
    )
    if not looks_like_path:
        res = resources.files("llm_api_benchmark") / "datasets" / f"{ref}.yaml"
        if not res.is_file():
            raise DatasetError(
                [f"dataset: no built-in dataset '{ref}' (built-in: {', '.join(builtin_names())})"]
            )
        return parse_dataset(
            yaml.safe_load(res.read_text(encoding="utf-8")), source=f"builtin:{ref}"
        )
    path = Path(ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise DatasetError([f"dataset: cannot read {path} ({err.strerror or err})"]) from err
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as err:
        raise DatasetError([f"dataset: {path} is not valid YAML/JSON: {err}"]) from err
    return parse_dataset(data, source=path.name)


def parse_dataset(data: Any, *, source: str) -> Dataset:
    problems: list[str] = []
    if not isinstance(data, dict):
        raise DatasetError([f"{source}: a dataset is a mapping"])
    for key in data:
        if key not in {"id", "version", "description", "license", "workloads"}:
            problems.append(f"{source}: unknown key '{key}'")
    ds_id, version = data.get("id"), data.get("version")
    if not isinstance(ds_id, str) or not _ID.match(ds_id):
        problems.append(f"{source}.id: lowercase letters, digits, '-' or '_'")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        problems.append(
            f"{source}.version: a version like 1.0.0 (change it whenever a prompt changes)"
        )
    workloads: list[Workload] = []
    raw_workloads = data.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        problems.append(f"{source}.workloads: at least one workload")
        raw_workloads = []
    seen_w: set[str] = set()
    for wi, w in enumerate(raw_workloads):
        wpath = f"{source}.workloads[{wi}]"
        if not isinstance(w, dict):
            problems.append(f"{wpath}: must be a mapping")
            continue
        for key in w:
            if key not in {"id", "description", "max_tokens", "prompts"}:
                problems.append(f"{wpath}: unknown key '{key}'")
        wid = w.get("id")
        if not isinstance(wid, str) or not _ID.match(wid):
            problems.append(f"{wpath}.id: lowercase letters, digits, '-' or '_'")
            continue
        if wid in seen_w:
            problems.append(f"{wpath}.id: '{wid}' is used twice")
        seen_w.add(wid)
        max_tokens = w.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            problems.append(f"{wpath}.max_tokens: a whole number of output tokens")
            max_tokens = 1
        prompts: list[Prompt] = []
        raw_prompts = w.get("prompts")
        if not isinstance(raw_prompts, list) or not raw_prompts:
            problems.append(f"{wpath}.prompts: at least one prompt")
            raw_prompts = []
        seen_p: set[str] = set()
        for pi, p in enumerate(raw_prompts):
            ppath = f"{wpath}.prompts[{pi}]"
            prompt = _parse_prompt(p, ppath, problems)
            if prompt:
                if prompt.id in seen_p:
                    problems.append(f"{ppath}.id: '{prompt.id}' is used twice in {wid}")
                seen_p.add(prompt.id)
                prompts.append(prompt)
        workloads.append(
            Workload(wid, str(w.get("description", "")).strip(), int(max_tokens), tuple(prompts))
        )
    if problems:
        raise DatasetError(problems)
    canonical = json.dumps(
        data, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
    )
    return Dataset(
        id=str(ds_id),
        version=str(version),
        description=str(data.get("description", "")).strip(),
        workloads=tuple(workloads),
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        source=source,
    )


def _parse_prompt(p: Any, path: str, problems: list[str]) -> Prompt | None:
    if not isinstance(p, dict):
        problems.append(f"{path}: must be a mapping")
        return None
    for key in p:
        if key not in {"id", "system", "prompt", "messages", "checks"}:
            problems.append(f"{path}: unknown key '{key}'")
    pid = p.get("id")
    if not isinstance(pid, str) or not _ID.match(pid):
        problems.append(f"{path}.id: lowercase letters, digits, '-' or '_'")
        return None
    messages: list[dict[str, str]] = []
    if isinstance(p.get("system"), str):
        messages.append({"role": "system", "content": p["system"].strip()})
    if "prompt" in p and "messages" in p:
        problems.append(f"{path}: use either prompt or messages, not both")
    if isinstance(p.get("prompt"), str):
        messages.append({"role": "user", "content": p["prompt"].strip()})
    elif isinstance(p.get("messages"), list):
        for mi, m in enumerate(p["messages"]):
            if (
                not isinstance(m, dict)
                or m.get("role") not in _ROLES
                or not isinstance(m.get("content"), str)
            ):
                problems.append(
                    f"{path}.messages[{mi}]: needs role (system/user/assistant) and content"
                )
                continue
            messages.append({"role": m["role"], "content": m["content"].strip()})
    if not any(m["role"] == "user" for m in messages):
        problems.append(f"{path}: needs a prompt (or a user message)")
    checks = p.get("checks") or []
    if not isinstance(checks, list):
        problems.append(f"{path}.checks: must be a list")
        checks = []
    for ci, spec in enumerate(checks):
        problems.extend(validate_spec(spec, f"{path}.checks[{ci}]"))
    return Prompt(pid, tuple(messages), tuple(checks))
