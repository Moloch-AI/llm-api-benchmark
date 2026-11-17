"""Deterministic output checks.

Checks answer narrow, mechanical questions about a reply: did it parse as JSON, does the code
compile, is the requested number on the last line, is it within the word limit. They are not a
measure of quality or intelligence and the reports never present them as one. They run on the
answer text only (not on reasoning), after removing ``<think>...</think>`` blocks.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

CHECK_TYPES = tuple(_CHECKS)


def validate_spec(spec: Any, path: str) -> list[str]:
    if not isinstance(spec, dict) or "type" not in spec:
        return [f"{path}: a check is a mapping with a 'type'"]
    kind = spec["type"]
    if kind not in _CHECKS:
        return [f"{path}.type: unknown check '{kind}' (known: {', '.join(CHECK_TYPES)})"]
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


def run_checks(text: str, specs: list[dict[str, Any]]) -> list[CheckResult]:
    answer = clean(text)
    results = []
    for spec in specs:
        fn, _ = _CHECKS[spec["type"]]
        results.append(fn(answer, spec))
    return results
