from __future__ import annotations

import pytest
from tests.helpers import TEST_KEY, provider

from llm_api_benchmark.providers.base import InvalidResponse
from llm_api_benchmark.providers.openai_chat import OpenAIChatAdapter, parse_error_body
from llm_api_benchmark.redact import REDACTOR

MSG = [{"role": "user", "content": "hi"}]


def adapter(**overrides: object) -> OpenAIChatAdapter:
    return OpenAIChatAdapter(provider(**overrides), TEST_KEY, {})


def test_stream_request_body_and_headers() -> None:
    req = adapter().build_request(MSG, stream=True, max_tokens=32, temperature=0.2)
    assert req.url == "https://fake.test/v1/chat/completions"
    assert req.body == {
        "model": "fake-model",
        "messages": MSG,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 32,
        "temperature": 0.2,
    }
    assert req.headers["Authorization"] == f"Bearer {TEST_KEY}"
    assert req.headers["Accept"] == "text/event-stream"


def test_plain_request_omits_stream_fields_and_unset_temperature() -> None:
    req = adapter(stream_usage=False).build_request(
        MSG, stream=False, max_tokens=8, temperature=None
    )
    assert req.body == {"model": "fake-model", "messages": MSG, "max_tokens": 8}


def test_max_tokens_param_variants() -> None:
    a = adapter(max_tokens_param="max_completion_tokens")
    assert (
        a.build_request(MSG, stream=False, max_tokens=8, temperature=None).body[
            "max_completion_tokens"
        ]
        == 8
    )
    b = adapter(max_tokens_param="none")
    body = b.build_request(MSG, stream=False, max_tokens=8, temperature=None).body
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


def test_params_override_and_null_removes() -> None:
    a = adapter(params={"temperature": None, "top_p": 0.9, "reasoning_effort": "low"})
    body = a.build_request(MSG, stream=False, max_tokens=8, temperature=0.7).body
    assert "temperature" not in body
    assert body["top_p"] == 0.9
    assert body["reasoning_effort"] == "low"


def test_auth_schemes_and_extra_headers() -> None:
    raw = OpenAIChatAdapter(
        provider(auth_scheme="header", auth_header="api-key"), TEST_KEY, {"X-Org": "o1"}
    )
    req = raw.build_request(MSG, stream=False, max_tokens=1, temperature=None)
    assert req.headers["api-key"] == TEST_KEY
    assert "Authorization" not in req.headers
    assert req.headers["X-Org"] == "o1"
    none = OpenAIChatAdapter(provider(auth_scheme="none", api_key_env=None), None, {})
    assert (
        "Authorization"
        not in none.build_request(MSG, stream=False, max_tokens=1, temperature=None).headers
    )
    assert TEST_KEY not in repr(raw)


def test_parse_chunk_variants() -> None:
    a = adapter()
    c = a.parse_chunk(
        {"model": "m", "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
    )
    assert (c.content, c.reasoning, c.model) == ("Hi", "", "m")
    r = a.parse_chunk({"choices": [{"delta": {"reasoning_content": "think"}}]})
    assert r.reasoning == "think"
    r2 = a.parse_chunk({"choices": [{"delta": {"reasoning": "think2"}}]})
    assert r2.reasoning == "think2"
    u = a.parse_chunk(
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}
    )
    assert u.usage is not None and u.usage.completion_tokens == 4
    d = a.parse_chunk(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
    )
    assert d.finish_reason == "stop"
    assert d.usage is not None and d.usage.reasoning_tokens == 1
    e = a.parse_chunk(
        {"error": {"type": "upstream_timeout", "message": "Server busy", "elapsed_s": 181.2}}
    )
    assert e.error is not None and e.error.type == "upstream_timeout"
    parts = a.parse_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
                    }
                }
            ]
        }
    )
    assert parts.content == "AB"


@pytest.mark.parametrize(
    "bad", [[], "x", {"id": "no choices"}, {"choices": "x"}, {"choices": [{"delta": "x"}]}]
)
def test_parse_chunk_invalid(bad: object) -> None:
    with pytest.raises(InvalidResponse):
        adapter().parse_chunk(bad)


def test_parse_completion() -> None:
    info = adapter().parse_completion(
        {
            "model": "served",
            "choices": [
                {
                    "message": {"content": "<think>x</think>ok", "reasoning_content": "r"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }
    )
    assert (info.content, info.reasoning, info.finish_reason, info.model) == (
        "<think>x</think>ok",
        "r",
        "stop",
        "served",
    )
    tool = adapter().parse_completion(
        {"choices": [{"message": {"content": None}, "finish_reason": "tool_calls"}]}
    )
    assert tool.content == ""
    with pytest.raises(InvalidResponse, match="no choices"):
        adapter().parse_completion({"choices": []})
    with pytest.raises(InvalidResponse, match="error body"):
        adapter().parse_completion({"error": {"type": "x", "message": "y"}})


def test_error_body_shapes() -> None:
    a = parse_error_body(
        b'{"error":{"type":"quota_exceeded","message":"Quota used.","resets_at":"x"}}'
    )
    assert (a.type, a.message) == ("quota_exceeded", "Quota used.")
    b = parse_error_body(b'{"message":"The selected model is invalid.","errors":{"model":["x"]}}')
    assert (b.type, b.message) == (None, "The selected model is invalid.")
    c = parse_error_body(b'{"message":"Verify your email.","code":"email_unverified"}')
    assert c.type == "email_unverified"
    d = parse_error_body(
        b"<html><head><title>524: A timeout occurred</title></head></html>", "text/html"
    )
    assert d.message == "(HTML) 524: A timeout occurred"
    assert parse_error_body(b"").message == "(empty body)"
    assert parse_error_body(b"upstream connect error").message == "upstream connect error"
    e = parse_error_body(b'{"error":{"code":"invalid_api_key","message":"bad"}}')
    assert e.type == "invalid_api_key"
    assert len(parse_error_body(b'{"message":"' + b"x" * 1000 + b'"}').message) <= 300


def test_error_messages_are_redacted() -> None:
    REDACTOR.add(TEST_KEY)
    info = parse_error_body(f'{{"error":{{"message":"Incorrect API key {TEST_KEY}"}}}}'.encode())
    assert TEST_KEY not in info.message
    assert "[redacted]" in info.message
