"""Tool (function) calls: definitions in datasets, calls in replies, and their fingerprints.

The wire format is OpenAI Chat Completions: a request carries ``tools`` (function definitions
with JSON Schema parameters) and optionally ``tool_choice``; a reply carries
``message.tool_calls`` with each call's ``function.name`` and ``function.arguments``, a JSON
string. Streams send calls as deltas keyed by ``index``: the name usually arrives once and the
arguments arrive in fragments that are concatenated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .schema import check_schema

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
NO_CALL = "no-tool-call"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str  # as sent by the model; may not be valid JSON
    id: str | None = None

    def parsed(self) -> tuple[Any, str | None]:
        """(arguments as JSON, None) or (None, why they did not parse)."""
        text = self.arguments.strip()
        if not text:
            return {}, None  # no arguments at all: an empty object, as OpenAI clients treat it
        try:
            return json.loads(text), None
        except json.JSONDecodeError as err:
            return None, f"arguments are not valid JSON ({err.msg})"


@dataclass(frozen=True)
class ToolCallDelta:
    """One streamed fragment of a call."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


class ToolCallAccumulator:
    """Rebuilds whole calls from stream deltas, in index order."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def add(self, delta: ToolCallDelta) -> None:
        call = self._calls.setdefault(delta.index, {"id": None, "name": "", "arguments": []})
        if delta.id and not call["id"]:
            call["id"] = delta.id
        # Most servers send the name once; some repeat it in every delta. Never concatenate it.
        if delta.name and not call["name"]:
            call["name"] = delta.name
        if delta.arguments:
            call["arguments"].append(delta.arguments)

    def calls(self) -> list[ToolCall]:
        return [
            ToolCall(name=c["name"], arguments="".join(c["arguments"]), id=c["id"])
            for _, c in sorted(self._calls.items())
        ]

    def __bool__(self) -> bool:
        return bool(self._calls)


def signature(calls: list[ToolCall]) -> str:
    """A short fingerprint of what was called with which arguments, for consistency checks.

    Two replies get the same signature when they call the same functions, in any order, with
    equal arguments after JSON parsing (key order and whitespace do not matter). Arguments that
    do not parse are fingerprinted as their stripped text. No content is stored, only the hash.
    """
    if not calls:
        return NO_CALL
    parts = []
    for call in calls:
        value, err = call.parsed()
        canon = (
            json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if err is None
            else call.arguments.strip()
        )
        parts.append(f"{call.name}({canon})")
    digest = hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()
    return digest[:16]


def tool_index(tools: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Function name -> its parameters schema."""
    out: dict[str, Any] = {}
    for tool in tools:
        fn = tool.get("function") or {}
        out[str(fn.get("name"))] = fn.get("parameters") or {"type": "object"}
    return out


def validate_tools(tools: Any, path: str) -> list[str]:
    """Problems with a dataset's tool definitions (OpenAI ``tools`` format)."""
    if not isinstance(tools, list) or not tools:
        return [f"{path}: a non-empty list of tool definitions"]
    problems: list[str] = []
    seen: set[str] = set()
    for i, tool in enumerate(tools):
        where = f"{path}[{i}]"
        if not isinstance(tool, dict) or tool.get("type") != "function":
            problems.append(f"{where}: needs type: function")
            continue
        for key in tool:
            if key not in {"type", "function"}:
                problems.append(f"{where}: unknown key '{key}'")
        fn = tool.get("function")
        if not isinstance(fn, dict):
            problems.append(f"{where}.function: a mapping with name, description, parameters")
            continue
        for key in fn:
            if key not in {"name", "description", "parameters", "strict"}:
                problems.append(f"{where}.function: unknown key '{key}'")
        name = fn.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            problems.append(f"{where}.function.name: letters, digits, '_' or '-', at most 64")
            continue
        if name in seen:
            problems.append(f"{where}.function.name: '{name}' is defined twice")
        seen.add(name)
        params = fn.get("parameters", {"type": "object"})
        problems.extend(check_schema(params, f"{where}.function.parameters"))
        if isinstance(params, dict) and params.get("type") not in (None, "object"):
            problems.append(f"{where}.function.parameters.type: must be object")
    return problems


def validate_tool_choice(choice: Any, names: set[str], path: str) -> list[str]:
    if choice in ("auto", "none", "required"):
        return []
    if isinstance(choice, dict) and choice.get("type") == "function":
        fn = choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name in names:
            return []
        return [
            f"{path}: function.name must be one of the workload's tools ({', '.join(sorted(names))})"
        ]
    return [f"{path}: auto, none, required, or {{type: function, function: {{name: ...}}}}"]


def values_match(found: Any, expected: Any, tolerance: float = 0.0) -> bool:
    """Lenient comparison for expected arguments.

    Strings compare without case and surrounding space; numbers within ``tolerance`` (and an
    integer-valued string such as "3" does not equal 3); lists compare in order; objects compare
    the expected keys only, so extra optional arguments do not fail a check.
    """
    if isinstance(expected, bool) or isinstance(found, bool):
        return found is expected
    if isinstance(expected, int | float) and isinstance(found, int | float):
        return abs(float(found) - float(expected)) <= tolerance
    if isinstance(expected, str) and isinstance(found, str):
        return found.strip().casefold() == expected.strip().casefold()
    if isinstance(expected, list) and isinstance(found, list):
        return len(found) == len(expected) and all(
            values_match(f, e, tolerance) for f, e in zip(found, expected, strict=True)
        )
    if isinstance(expected, dict) and isinstance(found, dict):
        return all(k in found and values_match(found[k], v, tolerance) for k, v in expected.items())
    return bool(found == expected)
