"""Tool calling: schemas, parsing, stream accumulation, checks, datasets, agreement, runner."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from tests.helpers import TEST_KEY, Recorder, config, provider, provider_dict, sse, usage_chunk

from llm_api_benchmark.checks import run_checks
from llm_api_benchmark.datasets import DatasetError, load_dataset, parse_dataset
from llm_api_benchmark.measure import execute
from llm_api_benchmark.metrics import summarize, tool_agreement
from llm_api_benchmark.providers.base import InvalidResponse
from llm_api_benchmark.providers.openai_chat import OpenAIChatAdapter
from llm_api_benchmark.records import RequestRecord
from llm_api_benchmark.reports.markdown import markdown_report
from llm_api_benchmark.results import build_result
from llm_api_benchmark.runner import Runner, build_plan
from llm_api_benchmark.schema import check_schema, validate
from llm_api_benchmark.tools import (
    NO_CALL,
    ToolCall,
    ToolCallAccumulator,
    ToolCallDelta,
    signature,
    validate_tool_choice,
    validate_tools,
    values_match,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_local_time",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
}
SCHEMAS = {
    "get_weather": WEATHER["function"]["parameters"],
    "get_local_time": TIME["function"]["parameters"],
}


# ----------------------------------------------------------------------------- schema


def test_schema_accepts_valid_values() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 10},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "x": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["n"],
    }
    assert validate({"n": 3, "tags": ["a"], "code": "EUR", "x": None}, schema) == []
    assert validate({"n": 3.0}, schema) == []  # 3.0 is an integer in JSON Schema


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ({}, "missing required property 'n'"),
        ({"n": 0}, "below the minimum"),
        ({"n": 2.5}, "expected integer"),
        ({"n": True}, "expected integer"),  # booleans are not numbers
        ({"n": 1, "tags": ["a", "b", "c"]}, "more than 2 items"),
        ({"n": 1, "tags": [1]}, "$.tags[0]: expected string"),
        ({"n": 1, "code": "eur"}, "does not match"),
        ({"n": 1, "other": 1}, "unexpected property 'other'"),
    ],
)
def test_schema_reports_each_violation(value: dict[str, Any], fragment: str) -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
        },
        "required": ["n"],
        "additionalProperties": False,
    }
    errors = validate(value, schema)
    assert any(fragment in e for e in errors), errors


def test_schema_refuses_keywords_it_cannot_check() -> None:
    problems = check_schema(
        {"type": "object", "properties": {"a": {"$ref": "#/defs/x"}}, "dependentRequired": {}},
        "p",
    )
    assert any("p.properties.a.$ref" in p for p in problems)
    assert any("p.dependentRequired" in p for p in problems)
    assert check_schema({"type": "object", "description": "fine", "format": "x"}, "p") == []


# ----------------------------------------------------------------------------- calls


def test_accumulator_joins_fragments_by_index_and_keeps_the_first_name() -> None:
    acc = ToolCallAccumulator()
    acc.add(ToolCallDelta(0, id="a", name="get_weather", arguments=""))
    acc.add(ToolCallDelta(1, id="b", name="get_local_time", arguments='{"timezone":'))
    acc.add(ToolCallDelta(0, arguments='{"city": "Oslo", '))
    acc.add(ToolCallDelta(0, name="get_weather", arguments='"unit": "celsius"}'))  # repeated name
    acc.add(ToolCallDelta(1, arguments=' "Asia/Tokyo"}'))
    calls = acc.calls()
    assert [c.name for c in calls] == ["get_weather", "get_local_time"]
    assert json.loads(calls[0].arguments) == {"city": "Oslo", "unit": "celsius"}
    assert calls[1].parsed() == ({"timezone": "Asia/Tokyo"}, None)


def test_signature_ignores_order_and_formatting() -> None:
    a = [ToolCall("f", '{"x": 1, "y": "a"}'), ToolCall("g", "{}")]
    b = [ToolCall("g", ""), ToolCall("f", '{"y":"a","x":1}')]
    assert signature(a) == signature(b)
    assert signature([ToolCall("f", '{"x": 2, "y": "a"}'), ToolCall("g", "{}")]) != signature(a)
    assert signature([]) == NO_CALL
    assert ToolCall("f", "{oops").parsed()[1] is not None


def test_values_match_is_lenient_where_it_should_be() -> None:
    assert values_match({"city": " oslo ", "extra": 1}, {"city": "Oslo"})
    assert values_match({"n": 90.0}, {"n": 90})
    assert not values_match({"n": "90"}, {"n": 90})
    assert values_match({"tags": ["Travel", "documents"]}, {"tags": ["travel", "Documents"]})
    assert not values_match({"tags": ["documents", "travel"]}, {"tags": ["travel", "documents"]})
    assert not values_match({"flag": 1}, {"flag": True})
    assert values_match({"p": 1.04}, {"p": 1}, tolerance=0.05)


def test_tool_definitions_and_choice_are_validated() -> None:
    assert validate_tools([WEATHER, TIME], "t") == []
    assert validate_tools([], "t")
    assert any("defined twice" in p for p in validate_tools([WEATHER, WEATHER], "t"))
    bad_name = {"type": "function", "function": {"name": "has space"}}
    assert any("function.name" in p for p in validate_tools([bad_name], "t"))
    assert validate_tool_choice("required", {"get_weather"}, "c") == []
    assert (
        validate_tool_choice(
            {"type": "function", "function": {"name": "get_weather"}}, {"get_weather"}, "c"
        )
        == []
    )
    assert validate_tool_choice({"type": "function", "function": {"name": "x"}}, {"y"}, "c")
    assert validate_tool_choice("always", {"y"}, "c")


# ----------------------------------------------------------------------------- adapter


def adapter() -> OpenAIChatAdapter:
    return OpenAIChatAdapter(provider(), TEST_KEY, {})


def test_request_carries_tools_and_tool_choice_only_when_given() -> None:
    a = adapter()
    msgs = [{"role": "user", "content": "hi"}]
    plain = a.build_request(msgs, stream=False, max_tokens=8, temperature=None)
    assert "tools" not in plain.body and "tool_choice" not in plain.body
    req = a.build_request(
        msgs, stream=True, max_tokens=8, temperature=None, tools=[WEATHER], tool_choice="required"
    )
    assert req.body["tools"] == [WEATHER] and req.body["tool_choice"] == "required"


def test_plain_reply_tool_calls_are_parsed() -> None:
    info = adapter().parse_completion(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                            },
                            # some servers send arguments as an object
                            {
                                "function": {
                                    "name": "get_local_time",
                                    "arguments": {"timezone": "UTC"},
                                }
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    assert [c.name for c in info.tool_calls] == ["get_weather", "get_local_time"]
    assert info.tool_calls[1].parsed() == ({"timezone": "UTC"}, None)
    assert info.finish_reason == "tool_calls" and info.content == ""


def test_legacy_function_call_and_malformed_calls() -> None:
    legacy = adapter().parse_completion(
        {"choices": [{"message": {"function_call": {"name": "f", "arguments": "{}"}}}]}
    )
    assert [c.name for c in legacy.tool_calls] == ["f"]
    with pytest.raises(InvalidResponse):
        adapter().parse_completion({"choices": [{"message": {"tool_calls": [{"function": {}}]}}]})
    with pytest.raises(InvalidResponse):
        adapter().parse_chunk({"choices": [{"delta": {"tool_calls": "nope"}}]})


# ----------------------------------------------------------------------------- measure


def tool_delta(index: int, args: str, name: str | None = None) -> dict[str, Any]:
    fn: dict[str, Any] = {"arguments": args}
    if name:
        fn["name"] = name
    call: dict[str, Any] = {"index": index, "function": fn}
    if name:
        call.update(id=f"call_{index}", type="function")
    return {"choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]}


def measure(handler: Any, *, stream: bool = True) -> tuple[RequestRecord, Any]:
    a = adapter()
    rec = RequestRecord(0, "fake", "m", "w", "p", "stream" if stream else "plain", 0)

    async def go() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            req = a.build_request(
                [{"role": "user", "content": "x"}],
                stream=stream,
                max_tokens=64,
                temperature=None,
                tools=[WEATHER, TIME],
            )
            return await execute(client, a, req, rec, stream=stream, timeout_s=5, prompt_chars=1)

    return rec, asyncio.run(go())


def test_streamed_tool_calls_are_rebuilt_and_timed() -> None:
    body = sse(
        tool_delta(0, "", name="get_weather"),
        tool_delta(1, "", name="get_local_time"),
        tool_delta(0, '{"city": "Tokyo",'),
        tool_delta(1, '{"timezone": "Asia/Tokyo"}'),
        tool_delta(0, ' "unit": "celsius"}'),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        usage_chunk(completion=20),
    )
    rec, reply = measure(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    assert rec.status == "ok" and rec.finish_reason == "tool_calls"
    assert rec.ttft_s is not None and rec.ttfc_s == rec.ttft_s  # a tool call is the answer
    assert rec.tool_call_names == ["get_weather", "get_local_time"]
    assert rec.tool_signature == signature(reply.tool_calls) != NO_CALL
    assert reply.text == ""
    assert json.loads(reply.tool_calls[0].arguments) == {"city": "Tokyo", "unit": "celsius"}


def test_text_reply_to_a_tool_request_records_no_call() -> None:
    body = sse(
        {"choices": [{"index": 0, "delta": {"content": "Ottawa"}, "finish_reason": "stop"}]},
        usage_chunk(),
    )
    rec, reply = measure(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    assert rec.tool_call_names == [] and rec.tool_signature == NO_CALL
    assert reply.text == "Ottawa"


def test_plain_tool_reply_without_usage_estimates_tokens_from_arguments() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Oslo","unit":"celsius"}',
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    rec, _ = measure(lambda r: httpx.Response(200, json=payload), stream=False)
    assert rec.status == "ok" and rec.tool_call_names == ["get_weather"]
    assert rec.tokens_estimated and rec.completion_tokens and rec.completion_tokens > 0


# ----------------------------------------------------------------------------- checks


def test_tool_checks() -> None:
    calls = [
        ToolCall("get_weather", '{"city": "Oslo", "unit": "celsius"}'),
        ToolCall("get_weather", '{"city": "Nairobi", "unit": "kelvin"}'),
    ]
    specs = [
        {"type": "tool_call", "name": "get_weather", "arguments": {"city": "oslo"}},
        {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Lima"}},
        {"type": "tool_call", "name": "get_local_time"},
        {"type": "tool_call_count", "min": 2},
        {"type": "tool_call_count", "max": 1},
        {"type": "tool_args_valid"},
        {"type": "no_tool_call"},
    ]
    results = run_checks("", specs, tool_calls=calls, tools=SCHEMAS)
    assert [r.passed for r in results] == [True, False, False, True, False, False, False]
    assert "kelvin" in results[5].detail
    unknown = run_checks(
        "", [{"type": "tool_args_valid"}], tool_calls=[ToolCall("nope", "{}")], tools=SCHEMAS
    )
    assert not unknown[0].passed and "not a defined tool" in unknown[0].detail
    none = run_checks(
        "Ottawa", [{"type": "no_tool_call"}, {"type": "tool_args_valid"}], tools=SCHEMAS
    )
    assert [r.passed for r in none] == [True, False]


# ----------------------------------------------------------------------------- datasets


def dataset(**workload: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"id": "w", "max_tokens": 64, "prompts": [{"id": "p", "prompt": "hi"}]}
    base.update(workload)
    return {"id": "t", "version": "1.0.0", "workloads": [base]}


def test_builtin_tools_dataset_loads() -> None:
    ds = load_dataset("tools-v1")
    assert [w.id for w in ds.workloads] == [
        "tool-select",
        "tool-args",
        "tool-parallel",
        "tool-choice",
    ]
    assert all(w.tools and len(w.prompts) == 3 for w in ds.workloads)
    choice = ds.workload("tool-choice")
    assert {p.id: p.tool_choice for p in choice.prompts}["required"] == "required"


@pytest.mark.parametrize(
    ("workload", "fragment"),
    [
        (
            {"prompts": [{"id": "p", "prompt": "hi", "checks": [{"type": "tool_args_valid"}]}]},
            "needs a workload with tools",
        ),
        (
            {
                "tools": [WEATHER],
                "prompts": [
                    {"id": "p", "prompt": "hi", "checks": [{"type": "tool_call", "name": "nope"}]}
                ],
            },
            "not one of the workload's tools",
        ),
        ({"tool_choice": "auto"}, "tool_choice: needs tools"),
        ({"tools": [WEATHER], "tool_choice": "sometimes"}, "tool_choice"),
        (
            {
                "tools": [WEATHER],
                "prompts": [{"id": "p", "prompt": "hi", "checks": [{"type": "tool_call_count"}]}],
            },
            "give min, max or both",
        ),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object", "if": {}}},
                    }
                ]
            },
            "keyword not supported",
        ),
    ],
)
def test_bad_tool_datasets_are_refused(workload: dict[str, Any], fragment: str) -> None:
    with pytest.raises(DatasetError) as err:
        parse_dataset(dataset(**workload), source="t")
    assert any(fragment in p for p in err.value.problems), err.value.problems


# ----------------------------------------------------------------------------- metrics


def rec_with(prompt: str, sig: str, names: list[str]) -> RequestRecord:
    r = RequestRecord(0, "a", "m", "w", prompt, "stream", 0, status="ok", latency_s=1.0)
    r.tool_signature, r.tool_call_names = sig, names
    return r


def test_agreement_is_the_mean_modal_share_over_prompts() -> None:
    recs = [
        rec_with("p1", "x", ["f"]),
        rec_with("p1", "x", ["f"]),
        rec_with("p1", "y", ["g"]),
        rec_with("p2", NO_CALL, []),
        rec_with("p2", NO_CALL, []),
        rec_with("p3", "z", ["f"]),  # one sample: not counted
    ]
    value, prompts = tool_agreement(recs)
    assert prompts == 2 and value == pytest.approx((2 / 3 + 1) / 2)
    summary = summarize(recs)[0]
    assert summary.tool_call_rate == pytest.approx(4 / 6)
    assert summary.tool_agreement_prompts == 2


# ----------------------------------------------------------------------------- runner


def test_runner_sends_tools_and_reports_tool_checks() -> None:
    ds = parse_dataset(
        dataset(
            tools=[WEATHER, TIME],
            prompts=[
                {
                    "id": "weather",
                    "prompt": "Weather in Oslo?",
                    "checks": [
                        {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Oslo"}},
                        {"type": "tool_args_valid"},
                    ],
                },
                {
                    "id": "forced",
                    "prompt": "Time?",
                    "tool_choice": {"type": "function", "function": {"name": "get_local_time"}},
                    "checks": [{"type": "tool_call", "name": "get_local_time"}],
                },
            ],
        ),
        source="t",
    )

    def respond(request: httpx.Request, n: int) -> httpx.Response:
        body = json.loads(request.content)
        forced = isinstance(body.get("tool_choice"), dict)
        name, args = (
            ("get_local_time", '{"timezone":"UTC"}')
            if forced
            else ("get_weather", '{"city":"Oslo","unit":"celsius"}')
        )
        msg = {
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": name, "arguments": args}}
            ]
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": msg, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            },
        )

    recorder = Recorder(respond)
    cfg = config(
        [provider_dict()], dataset="unused", modes=["plain"], repetitions=2, save_outputs=True
    )
    runner = Runner(cfg, ds, build_plan(cfg, ds), transport_factory=lambda p: recorder.transport())
    asyncio.run(runner.run())

    bodies = recorder.bodies()
    assert all(b["tools"][0]["function"]["name"] == "get_weather" for b in bodies)
    forced = [b for b in bodies if "tool_choice" in b]
    assert len(forced) == 2 and forced[0]["tool_choice"]["function"]["name"] == "get_local_time"
    records = runner.ordered_records()
    assert all(r.checks_passed for r in records)
    assert records[0].tool_calls and records[0].tool_calls[0]["name"] in {
        "get_weather",
        "get_local_time",
    }
    summary = summarize(records)[0]
    assert summary.tool_call_rate == 1.0 and summary.tool_agreement == 1.0
    assert summary.check_types == {"tool_args_valid": [2, 2], "tool_call": [4, 4]}

    result = build_result(runner)
    report = markdown_report(result)
    assert "Called a tool" in report and "Same calls" in report and "tool_args_valid" in report
    assert "Tool-call checks test the shape of a call" in report
