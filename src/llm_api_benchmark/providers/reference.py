"""An in-process reference endpoint for ``llm-bench conformance``.

It answers the conformance suite's requests the way the OpenAI Chat Completions API documents
them: ``GET /models``, plain and streamed replies with usage, ``max_tokens`` and
``max_completion_tokens``, ``stop``, ``n``, JSON mode and JSON schema output, ``logprobs``,
tool calls and tool results, and OpenAI-shaped errors. It does not run a model: replies are
built from the prompt with a few fixed rules, enough for the suite's prompts.

Quirks (``synthetic.quirks`` in the configuration) make it misbehave on purpose, one way each, so
that every kind of failure the suite reports can be shown offline and tested. Results from this
endpoint are synthetic fixture data, never evidence about a real API.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import SyntheticProfile

REFERENCE_KEY = "reference-key"


def reference_transport(profile: SyntheticProfile, model: str) -> httpx.MockTransport:
    quirks = frozenset(profile.quirks)
    state = {"calls": 0}

    def error(status: int, message: str, kind: str, code: str | None = None) -> httpx.Response:
        if "bare_errors" in quirks:
            return httpx.Response(status, json={"message": message})
        body = {"error": {"message": message, "type": kind, "param": None, "code": code}}
        return httpx.Response(status, json=body)

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.rstrip("/")
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {REFERENCE_KEY}" and "accept_any_key" not in quirks:
            return error(
                401, "Incorrect API key provided.", "invalid_request_error", "invalid_api_key"
            )
        if path.endswith("/models") and request.method == "GET":
            if "models_missing" in quirks:
                return error(404, "Not found.", "invalid_request_error")
            data = [{"id": model, "object": "model", "created": 0, "owned_by": "reference"}]
            return httpx.Response(200, json={"object": "list", "data": data})
        if not (path.endswith("/chat/completions") and request.method == "POST"):
            return error(404, "Not found.", "invalid_request_error")
        try:
            body = json.loads(request.content or b"")
        except ValueError:
            if "server_error_on_bad_json" in quirks:
                return httpx.Response(500, text="Internal Server Error")
            return error(
                400, "We could not parse the JSON body of your request.", "invalid_request_error"
            )
        if not isinstance(body, dict):
            return error(400, "The body must be a JSON object.", "invalid_request_error")
        if body.get("model") != model and "substitute_model" not in quirks:
            return error(
                404,
                f"The model `{body.get('model')}` does not exist.",
                "invalid_request_error",
                "model_not_found",
            )
        messages = body.get("messages")
        if not isinstance(messages, list) or (
            not messages and "accept_empty_messages" not in quirks
        ):
            return error(400, "messages must be a non-empty array.", "invalid_request_error")
        temperature = body.get("temperature")
        if (
            isinstance(temperature, int | float)
            and not 0 <= temperature <= 2
            and "accept_out_of_range" not in quirks
        ):
            return error(400, "temperature must be between 0 and 2.", "invalid_request_error")
        state["calls"] += 1
        return _reply(body, messages, model, quirks, state["calls"])

    return httpx.MockTransport(handler)


def _reply(
    body: dict[str, Any], messages: list[Any], model: str, quirks: frozenset[str], n_call: int
) -> httpx.Response:
    cid = f"chatcmpl-reference-{n_call}"
    call = _tool_call(body, messages, quirks, n_call)
    text = "" if call else _text(body, messages, quirks)
    tokens = text.split(" ") if text else []
    finish = "tool_calls" if call else "stop"

    limit = body.get("max_completion_tokens") or body.get("max_tokens")
    if isinstance(limit, int) and "ignore_max_tokens" not in quirks and len(tokens) > limit:
        tokens, finish = tokens[:limit], "length"
    text = " ".join(tokens)
    stop = body.get("stop")
    if stop and "ignore_stop" not in quirks:
        for seq in [stop] if isinstance(stop, str) else list(stop):
            if isinstance(seq, str) and seq and seq in text:
                text, finish = text[: text.index(seq)], "stop"
    n_value = body.get("n")
    choices_n = n_value if isinstance(n_value, int) and n_value > 0 else 1
    if "ignore_n" in quirks:
        choices_n = 1
    prompt_tokens = (
        sum(len(str(m.get("content") or "").split()) for m in messages if isinstance(m, dict)) + 3
    )
    completion = max(1, len(text.split())) if text else (8 if call else 1)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion * choices_n,
        "total_tokens": prompt_tokens + completion * choices_n,
    }
    wants_logprobs = body.get("logprobs") is True and "ignore_logprobs" not in quirks

    if not body.get("stream"):
        choices = []
        for i in range(choices_n):
            message: dict[str, Any] = {"role": "assistant", "content": text or None}
            if call:
                message["tool_calls"] = [call]
            choice: dict[str, Any] = {"index": i, "message": message, "finish_reason": finish}
            if wants_logprobs:
                choice["logprobs"] = {
                    "content": [
                        {"token": t, "logprob": -0.1, "bytes": list(t.encode()), "top_logprobs": []}
                        for t in text.split()
                    ]
                }
            choices.append(choice)
        return httpx.Response(
            200,
            json={
                "id": cid,
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": choices,
                "usage": usage,
            },
        )

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def events() -> AsyncIterator[bytes]:
        def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(obj)}\n\n".encode()

        yield chunk({"role": "assistant", "content": ""})
        if call:
            fn = call["function"]
            args = (
                fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
            )
            head = {"index": 0, "id": call["id"], "type": "function"}
            yield chunk(
                {"tool_calls": [{**head, "function": {"name": fn["name"], "arguments": ""}}]}
            )
            for i in range(0, len(args), 8):
                part = {"index": 0, "function": {"arguments": args[i : i + 8]}}
                yield chunk({"tool_calls": [part]})
        for i, word in enumerate(text.split(" ") if text else []):
            yield chunk({"content": word if i == 0 else " " + word})
        yield chunk({}, finish)
        if include_usage and "no_stream_usage" not in quirks:
            tail = {"id": cid, "object": "chat.completion.chunk", "created": 0, "model": model}
            yield f"data: {json.dumps({**tail, 'choices': [], 'usage': usage})}\n\n".encode()
        if "no_done" not in quirks:
            yield b"data: [DONE]\n\n"

    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())


def _last_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return str(m["content"])
    return ""


def _text(body: dict[str, Any], messages: list[Any], quirks: frozenset[str]) -> str:
    last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    if last.get("role") == "tool":
        try:
            result = json.loads(str(last.get("content")))
        except ValueError:
            result = {}
        temp = result.get("temperature", "?") if isinstance(result, dict) else "?"
        return f"It is {temp} degrees in Paris right now."
    prompt = _last_text(messages)
    fmt = body.get("response_format") if "ignore_response_format" not in quirks else None
    kind = fmt.get("type") if isinstance(fmt, dict) else None
    if kind == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema") or {}  # type: ignore[union-attr]
        return json.dumps(_instance(schema))
    if kind == "json_object":
        return json.dumps({"color": "grey", "legs": 4})
    if "JSON object" in prompt:
        return '```json\n{"color": "grey", "legs": 4}\n```'
    span = re.search(r"from (\d+) to (\d+)", prompt)
    if span:
        a, b = int(span.group(1)), int(span.group(2))
        return " ".join(str(i) for i in range(a, min(b, 200) + 1))
    word = re.search(r"single word: (\w+)", prompt)
    if word:
        return word.group(1)
    if "hello" in prompt.lower():
        return "Hello"
    if "cat" in prompt.lower():
        return "A typical house cat is small, curious and fond of naps."
    return "Hello from the reference endpoint."


def _tool_call(
    body: dict[str, Any], messages: list[Any], quirks: frozenset[str], n_call: int
) -> dict[str, Any] | None:
    tools = [t for t in body.get("tools") or [] if isinstance(t, dict)]
    if not tools:
        return None
    choice = body.get("tool_choice", "auto")
    last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    if choice == "none" and "ignore_tool_choice" not in quirks:
        return None
    if choice == "auto" and last.get("role") == "tool":
        return None
    fn = tools[0].get("function") or {}
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        fn = next(
            (
                (t.get("function") or {})
                for t in tools
                if (t.get("function") or {}).get("name") == name
            ),
            fn,
        )
    prompt = _last_text(messages)
    args = _arguments(fn.get("parameters") or {}, prompt)
    return {
        "id": f"call_reference_{n_call}",
        "type": "function",
        "function": {
            "name": fn.get("name", "tool"),
            "arguments": args if "tool_args_object" in quirks else json.dumps(args),
        },
    }


def _arguments(schema: dict[str, Any], prompt: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    props = schema.get("properties") or {}
    for name in schema.get("required") or list(props):
        sub = props.get(name) or {}
        if sub.get("enum"):
            out[name] = next(
                (v for v in sub["enum"] if str(v).lower() in prompt.lower()), sub["enum"][0]
            )
        elif name == "city":
            m = re.search(r"\bin ([A-Z][a-zA-Z]+)", prompt)
            out[name] = m.group(1) if m else "Paris"
        else:
            out[name] = _instance(sub)
    return out


def _instance(schema: dict[str, Any]) -> Any:
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        return {k: _instance(props.get(k) or {}) for k in schema.get("required") or list(props)}
    if kind == "array":
        return [_instance(schema.get("items") or {})]
    if kind == "integer":
        return int(schema.get("minimum", 4))
    if kind == "number":
        return float(schema.get("minimum", 1.0))
    if kind == "boolean":
        return True
    return "example"
