from __future__ import annotations

from typing import Any

import pytest

from llm_api_benchmark.checks import clean, run_checks, validate_spec


def one(text: str, **spec: Any) -> bool:
    [r] = run_checks(text, [dict(spec)])
    return r.passed


def test_think_blocks_are_removed_before_checking() -> None:
    assert clean("<think>long reasoning 999</think>\nCanberra") == "Canberra"
    assert clean("<think>never closed") == ""
    assert one("<think>Sydney?</think> Canberra", type="contains", value="canberra")
    assert not one("<think>Canberra</think> Sydney", type="contains", value="Canberra")


@pytest.mark.parametrize(
    ("text", "spec", "expected"),
    [
        ("Canberra.", {"type": "contains", "value": "canberra"}, True),
        ("Canberra.", {"type": "contains", "value": "canberra", "case_sensitive": True}, False),
        ("we merge it", {"type": "contains_any", "values": ["review", "merge"]}, True),
        ("nothing", {"type": "contains_any", "values": ["review", "merge"]}, False),
        ("fine", {"type": "not_contains", "value": "error"}, True),
        ("Au", {"type": "regex", "pattern": r"\bAu\b", "ignore_case": False}, True),
        ("au", {"type": "regex", "pattern": r"\bAu\b", "ignore_case": False}, False),
        ("one two three", {"type": "max_words", "value": 3}, True),
        ("one two three four", {"type": "max_words", "value": 3}, False),
        ("one two", {"type": "min_words", "value": 3}, False),
        ("First. Second! Third?", {"type": "max_sentences", "value": 3}, True),
        ("First. Second! Third? Fourth.", {"type": "max_sentences", "value": 3}, False),
        ("The trip takes\n155", {"type": "last_number", "value": 155}, True),
        ("about 2.5 hours", {"type": "last_number", "value": 2.5}, True),
        ("no digits", {"type": "last_number", "value": 1}, False),
        ("12.4", {"type": "last_number", "value": 12, "tolerance": 0.5}, True),
    ],
)
def test_text_checks(text: str, spec: dict[str, Any], expected: bool) -> None:
    assert one(text, **spec) is expected


def test_json_checks_accept_fences_and_surrounding_text() -> None:
    fenced = 'Here you go:\n```json\n{"name": "Dana", "city": "Lisbon"}\n```'
    assert one(fenced, type="json_keys", keys=["name", "city"])
    assert not one(fenced, type="json_keys", keys=["name", "email"])
    assert one('Sure: {"a": 1} hope it helps', type="json_valid", kind="object")
    assert one("[1, 2]", type="json_valid", kind="array")
    assert not one("[1, 2]", type="json_valid", kind="object")
    assert not one("{not json", type="json_valid")


def test_python_syntax_and_defines() -> None:
    good = "```python\ndef is_palindrome(text: str) -> bool:\n    return text == text[::-1]\n```"
    assert one(good, type="python_syntax", defines="is_palindrome")
    assert not one(good, type="python_syntax", defines="other")
    assert not one("```python\ndef broken(:\n```", type="python_syntax")


def test_validate_spec_reports_problems() -> None:
    assert validate_spec({"type": "contains", "value": "x"}, "c") == []
    assert validate_spec({"type": "nope"}, "c")[0].startswith("c.type: unknown check")
    assert validate_spec({"type": "contains"}, "c") == ["c.value: is required for contains"]
    assert validate_spec({"type": "regex", "pattern": "("}, "c")[0].startswith("c.pattern: invalid")
    assert validate_spec({"type": "max_words", "value": 3, "extra": 1}, "c") == [
        "c.extra: unknown key"
    ]
    assert validate_spec("x", "c") == ["c: a check is a mapping with a 'type'"]
