"""Deterministic output checks.

Checks answer narrow, mechanical questions about a reply: did it parse as JSON, does the code
compile, is the requested number on the last line, is it within the word limit. They are not a
measure of quality or intelligence and the reports never present them as one. Text checks run on
the answer text only (not on reasoning), after removing ``<think>...</think>`` blocks. Tool checks
run on the reply's tool calls and the workload's tool definitions.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .schema import validate
from .tools import ToolCall, values_match

_THINK = re.compile(r"(?is)<think>.*?(</think>|$)")
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]*)(?:\s+|$)")


@dataclass(frozen=True)
class CheckResult:
    type: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "passed": self.passed, "detail": self.detail}


class CheckSpecError(ValueError):
    pass


def clean(text: str) -> str:
    return _THINK.sub("", text).strip()


def _code_block(text: str, languages: tuple[str, ...]) -> str:
    blocks = _FENCE.findall(text)
    for lang, body in blocks:
        if lang.lower() in languages:
            return str(body)
    if blocks:
        return str(blocks[0][1])
    return text


def _json_value(text: str) -> Any:
    candidate = _code_block(text, ("json",)).strip()
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    # The first {...} or [...] span, for replies that wrap JSON in a sentence.
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = candidate.find(open_), candidate.rfind(close)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except ValueError:
                continue
    raise ValueError("no JSON value found")


def _words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _sentences(text: str) -> int:
    body = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    if not body:
        return 0
    return max(1, len(_SENTENCE_END.findall(body + " ")))


def _contains(text: str, spec: dict[str, Any]) -> CheckResult:
    value = str(spec["value"])
    hay, needle = (text, value) if spec.get("case_sensitive") else (text.lower(), value.lower())
    return CheckResult("contains", needle in hay, f"looked for {value!r}")


def _contains_any(text: str, spec: dict[str, Any]) -> CheckResult:
    values = [str(v) for v in spec["values"]]
    low = text.lower()
    hit = next((v for v in values if v.lower() in low), None)
    return CheckResult(
        "contains_any", hit is not None, f"found {hit!r}" if hit else f"none of {values}"
    )


def _not_contains(text: str, spec: dict[str, Any]) -> CheckResult:
    value = str(spec["value"])
    return CheckResult(
        "not_contains", value.lower() not in text.lower(), f"must not contain {value!r}"
    )


def _regex(text: str, spec: dict[str, Any]) -> CheckResult:
    pattern = re.compile(str(spec["pattern"]), re.I if spec.get("ignore_case", True) else 0)
    return CheckResult("regex", bool(pattern.search(text)), f"pattern {spec['pattern']!r}")


def _max_words(text: str, spec: dict[str, Any]) -> CheckResult:
    n = _words(text)
    return CheckResult("max_words", n <= int(spec["value"]), f"{n} words, limit {spec['value']}")


def _min_words(text: str, spec: dict[str, Any]) -> CheckResult:
    n = _words(text)
    return CheckResult("min_words", n >= int(spec["value"]), f"{n} words, minimum {spec['value']}")


def _max_sentences(text: str, spec: dict[str, Any]) -> CheckResult:
    n = _sentences(text)
    return CheckResult(
        "max_sentences", n <= int(spec["value"]), f"{n} sentences, limit {spec['value']}"
    )


def _json_valid(text: str, spec: dict[str, Any]) -> CheckResult:
    try:
        value = _json_value(text)
    except ValueError:
        return CheckResult("json_valid", False, "no parseable JSON")
    want = spec.get("kind")
    if want == "object" and not isinstance(value, dict):
        return CheckResult("json_valid", False, "JSON is not an object")
    if want == "array" and not isinstance(value, list):
        return CheckResult("json_valid", False, "JSON is not an array")
    return CheckResult("json_valid", True, "parsed")


def _json_keys(text: str, spec: dict[str, Any]) -> CheckResult:
    keys = [str(k) for k in spec["keys"]]
    try:
        value = _json_value(text)
    except ValueError:
        return CheckResult("json_keys", False, "no parseable JSON")
    if not isinstance(value, dict):
        return CheckResult("json_keys", False, "JSON is not an object")
    missing = [k for k in keys if k not in value]
    return CheckResult(
        "json_keys", not missing, f"missing {missing}" if missing else "all keys present"
    )


def _python_syntax(text: str, spec: dict[str, Any]) -> CheckResult:
    code = _code_block(text, ("python", "py", "python3"))
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        return CheckResult("python_syntax", False, f"SyntaxError line {err.lineno}")
    name = spec.get("defines")
    if name:
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        if name not in names:
            return CheckResult("python_syntax", False, f"parses, but defines no function {name!r}")
    return CheckResult("python_syntax", True, "parses")


def _last_number(text: str, spec: dict[str, Any]) -> CheckResult:
    numbers = _NUMBER.findall(text)
    if not numbers:
        return CheckResult("last_number", False, "no number in the reply")
    found = float(numbers[-1].replace(",", "."))
    expected = float(spec["value"])
    ok = abs(found - expected) <= float(spec.get("tolerance", 0))
    return CheckResult("last_number", ok, f"last number {numbers[-1]}, expected {spec['value']}")


def _tool_call(calls: list[ToolCall], tools: dict[str, Any], spec: dict[str, Any]) -> CheckResult:
    name = str(spec["name"])
    expected = spec.get("arguments")
    named = [c for c in calls if c.name == name]
    if not named:
        called = ", ".join(c.name for c in calls) or "no tool"
        return CheckResult("tool_call", False, f"expected {name}, called {called}")
    if expected is None:
        return CheckResult("tool_call", True, f"called {name}")
    tolerance = float(spec.get("tolerance", 0))
    for call in named:
        value, err = call.parsed()
        if err is None and values_match(value, expected, tolerance):
            return CheckResult("tool_call", True, f"called {name} with the expected arguments")
    return CheckResult("tool_call", False, f"called {name}, arguments differ from {expected}")


def _no_tool_call(
    calls: list[ToolCall], tools: dict[str, Any], spec: dict[str, Any]
) -> CheckResult:
    if not calls:
        return CheckResult("no_tool_call", True, "answered without a tool call")
    return CheckResult("no_tool_call", False, f"called {', '.join(c.name for c in calls)}")


def _tool_args_valid(
    calls: list[ToolCall], tools: dict[str, Any], spec: dict[str, Any]
) -> CheckResult:
    if not calls:
        return CheckResult("tool_args_valid", False, "no tool call")
    for call in calls:
        if call.name not in tools:
            label = call.name or "(no name)"
            return CheckResult("tool_args_valid", False, f"{label} is not a defined tool")
        value, err = call.parsed()
        if err is not None:
            return CheckResult("tool_args_valid", False, f"{call.name}: {err}")
        errors = validate(value, tools[call.name])
        if errors:
            return CheckResult("tool_args_valid", False, f"{call.name}: {errors[0]}")
    return CheckResult("tool_args_valid", True, f"{len(calls)} call(s) match their schemas")


def _tool_call_count(
    calls: list[ToolCall], tools: dict[str, Any], spec: dict[str, Any]
) -> CheckResult:
    n = len(calls)
    low, high = spec.get("min"), spec.get("max")
    ok = (low is None or n >= int(low)) and (high is None or n <= int(high))
    parts = [
        f"at least {low}" if low is not None else "",
        f"at most {high}" if high is not None else "",
    ]
    bounds = " and ".join(p for p in parts if p)
    return CheckResult("tool_call_count", ok, f"{n} call(s), expected {bounds}")


ToolCheck = Callable[[list[ToolCall], dict[str, Any], dict[str, Any]], CheckResult]
_TOOL_CHECKS: dict[str, tuple[ToolCheck, tuple[str, ...]]] = {
    "tool_call": (_tool_call, ("name",)),
    "no_tool_call": (_no_tool_call, ()),
    "tool_args_valid": (_tool_args_valid, ()),
    "tool_call_count": (_tool_call_count, ()),
}

_CHECKS: dict[str, tuple[Callable[[str, dict[str, Any]], CheckResult], tuple[str, ...]]] = {
    "contains": (_contains, ("value",)),
    "contains_any": (_contains_any, ("values",)),
    "not_contains": (_not_contains, ("value",)),
    "regex": (_regex, ("pattern",)),
    "max_words": (_max_words, ("value",)),
    "min_words": (_min_words, ("value",)),
    "max_sentences": (_max_sentences, ("value",)),
    "json_valid": (_json_valid, ()),
    "json_keys": (_json_keys, ("keys",)),
    "python_syntax": (_python_syntax, ()),
    "last_number": (_last_number, ("value",)),
}
_OPTIONAL = {"case_sensitive", "ignore_case", "kind", "defines", "tolerance"}
_TOOL_OPTIONAL = {"arguments", "tolerance", "min", "max"}

CHECK_TYPES = (*_CHECKS, *_TOOL_CHECKS)
TOOL_CHECK_TYPES = tuple(_TOOL_CHECKS)


def validate_spec(spec: Any, path: str, tool_names: set[str] | None = None) -> list[str]:
    """Problems with one check. ``tool_names`` = the workload's tools (None: it has none)."""
    if not isinstance(spec, dict) or "type" not in spec:
        return [f"{path}: a check is a mapping with a 'type'"]
    kind = spec["type"]
    if kind not in _CHECKS and kind not in _TOOL_CHECKS:
        return [f"{path}.type: unknown check '{kind}' (known: {', '.join(CHECK_TYPES)})"]
    if kind in _TOOL_CHECKS:
        return _validate_tool_spec(spec, path, tool_names)
    _, required = _CHECKS[kind]
    problems = [f"{path}.{r}: is required for {kind}" for r in required if r not in spec]
    allowed = {"type", *required, *_OPTIONAL}
    problems += [f"{path}.{k}: unknown key" for k in spec if k not in allowed]
    if kind == "regex" and "pattern" in spec:
        try:
            re.compile(str(spec["pattern"]))
        except re.error as err:
            problems.append(f"{path}.pattern: invalid regular expression ({err})")
    return problems


def _validate_tool_spec(spec: dict[str, Any], path: str, tool_names: set[str] | None) -> list[str]:
    kind = spec["type"]
    if tool_names is None:
        return [f"{path}.type: {kind} needs a workload with tools"]
    _, required = _TOOL_CHECKS[kind]
    problems = [f"{path}.{r}: is required for {kind}" for r in required if r not in spec]
    if kind == "tool_call" and "name" in spec and spec["name"] not in tool_names:
        problems.append(f"{path}.name: '{spec['name']}' is not one of the workload's tools")
    if "arguments" in spec and not isinstance(spec["arguments"], dict):
        problems.append(f"{path}.arguments: a mapping of expected argument values")
    if kind == "tool_call_count" and "min" not in spec and "max" not in spec:
        problems.append(f"{path}: give min, max or both")
    for bound in ("min", "max"):
        v = spec.get(bound)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
            problems.append(f"{path}.{bound}: a whole number")
    allowed = {"type", *required, *_TOOL_OPTIONAL}
    problems += [f"{path}.{k}: unknown key" for k in spec if k not in allowed]
    return problems


def run_checks(
    text: str,
    specs: list[dict[str, Any]],
    *,
    tool_calls: list[ToolCall] | None = None,
    tools: dict[str, Any] | None = None,
) -> list[CheckResult]:
    """Run ``specs`` on a reply. ``tools`` maps each defined function to its parameter schema."""
    answer = clean(text)
    results = []
    for spec in specs:
        if spec["type"] in _TOOL_CHECKS:
            tool_fn, _ = _TOOL_CHECKS[spec["type"]]
            results.append(tool_fn(list(tool_calls or []), dict(tools or {}), spec))
        else:
            fn, _ = _CHECKS[spec["type"]]
            results.append(fn(answer, spec))
    return results
