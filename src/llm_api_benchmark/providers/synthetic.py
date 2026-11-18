"""A simulated OpenAI-compatible endpoint that runs inside the process.

It exists so the tool can be tried, demonstrated and tested without a key or network access.
Timings come from a seeded random generator around the configured profile. Every result that
includes a synthetic endpoint is marked as synthetic fixture data in every report.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import SyntheticProfile

_WORDS = [
    "the",
    "request",
    "travels",
    "to",
    "the",
    "endpoint",
    "where",
    "a",
    "model",
    "reads",
    "the",
    "prompt",
    "and",
    "writes",
    "a",
    "reply",
    "one",
    "piece",
    "at",
    "a",
    "time",
    "while",
    "the",
    "client",
    "measures",
    "when",
    "each",
    "piece",
    "arrives",
]


def synthetic_transport(profile: SyntheticProfile, seed: int = 0) -> httpx.MockTransport:
    counter = itertools.count()

    async def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content or b"{}")
        model = str(body.get("model", "synthetic"))
        n = next(counter)
        rng = random.Random(f"{seed}:{model}:{n}")
        j = profile.jitter
        scale = profile.time_scale
        limit = body.get("max_tokens") or body.get("max_completion_tokens")
        target = min(profile.output_tokens, int(limit)) if limit else profile.output_tokens
        tokens = max(1, round(target * rng.uniform(1 - j, 1 + j)))
        ttft = profile.ttft_ms / 1000 * rng.uniform(max(0.05, 1 - j), 1 + 2 * j) * scale
        per_token = 1 / (profile.tokens_per_s * rng.uniform(1 - j / 2, 1 + j / 2)) * scale
        prompt_chars = sum(len(str(m.get("content") or "")) for m in body.get("messages", []))
        usage = {
            "prompt_tokens": math.ceil(prompt_chars / 4),
            "completion_tokens": tokens,
            "total_tokens": math.ceil(prompt_chars / 4) + tokens,
        }
        words = [_WORDS[i % len(_WORDS)] + " " for i in range(tokens)]

        if rng.random() < profile.error_rate:
            await asyncio.sleep(ttft)
            return httpx.Response(
                503, json={"error": {"type": "server_busy", "message": "Synthetic failure."}}
            )

        cid = f"chatcmpl-synthetic-{n}"
        call = _tool_call(body, rng, n)
        if call is not None:
            return await _tool_reply(body, call, cid, model, usage, ttft, per_token)
        if not body.get("stream"):
            await asyncio.sleep(ttft + tokens * per_token)
            return httpx.Response(
                200,
                json={
                    "id": cid,
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(words)},
                            "finish_reason": "length" if limit and tokens >= int(limit) else "stop",
                        }
                    ],
                    "usage": usage,
                },
            )

        async def events() -> AsyncIterator[bytes]:
            def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
                obj = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                return f"data: {json.dumps(obj)}\n\n".encode()

            await asyncio.sleep(ttft)
            yield chunk({"role": "assistant", "content": ""})
            for i in range(0, tokens, 3):
                if i:
                    await asyncio.sleep(per_token * 3)
                yield chunk({"content": "".join(words[i : i + 3])})
            yield chunk({}, "length" if limit and tokens >= int(limit) else "stop")
            tail = {"id": cid, "object": "chat.completion.chunk", "created": 0, "model": model}
            yield f"data: {json.dumps({**tail, 'choices': [], 'usage': usage})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())

    return httpx.MockTransport(handler)


def _tool_call(body: dict[str, Any], rng: random.Random, n: int) -> dict[str, Any] | None:
    """A plausible-looking call, or None to answer in text.

    The simulator does not read the prompt. It prefers one tool per prompt (picked by a hash of
    the messages) and sometimes picks another, so the agreement figures have something to show.
    """
    tools = [t for t in body.get("tools") or [] if isinstance(t, dict)]
    choice = body.get("tool_choice", "auto")
    if not tools or choice == "none":
        return None
    names = [str((t.get("function") or {}).get("name")) for t in tools]
    if isinstance(choice, dict):
        name = str((choice.get("function") or {}).get("name"))
    else:
        if choice != "required" and rng.random() < 0.15:
            return None  # answer in text now and then, as models do under "auto"
        digest = hashlib.sha256(json.dumps(body.get("messages"), sort_keys=True).encode())
        name = names[digest.digest()[0] % len(names)]
        if rng.random() < 0.2:
            name = rng.choice(names)
    schema = next(
        (
            (t.get("function") or {}).get("parameters") or {}
            for t in tools
            if (t.get("function") or {}).get("name") == name
        ),
        {},
    )
    args = json.dumps(_example(schema, rng), separators=(",", ":"))
    return {"id": f"call_synthetic_{n}", "name": name, "arguments": args}


def _example(schema: dict[str, Any], rng: random.Random) -> Any:
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if schema.get("enum"):
        return rng.choice(schema["enum"])
    if kind == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        return {k: _example(props.get(k) or {}, rng) for k in schema.get("required") or []}
    if kind == "array":
        return [_example(schema.get("items") or {}, rng)]
    if kind == "integer":
        return int(schema.get("minimum", 1))
    if kind == "number":
        return float(schema.get("minimum", 1.5))
    if kind == "boolean":
        return True
    return "example"


async def _tool_reply(
    body: dict[str, Any],
    call: dict[str, Any],
    cid: str,
    model: str,
    usage: dict[str, int],
    ttft: float,
    per_token: float,
) -> httpx.Response:
    tokens = math.ceil(len(call["arguments"]) / 4) + 5
    usage = {**usage, "completion_tokens": tokens, "total_tokens": usage["prompt_tokens"] + tokens}
    if not body.get("stream"):
        await asyncio.sleep(ttft + tokens * per_token)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
            ],
        }
        return httpx.Response(
            200,
            json={
                "id": cid,
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": usage,
            },
        )

    async def events() -> AsyncIterator[bytes]:
        def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(obj)}\n\n".encode()

        await asyncio.sleep(ttft)
        first = {"index": 0, "id": call["id"], "type": "function"}
        yield chunk(
            {
                "role": "assistant",
                "tool_calls": [{**first, "function": {"name": call["name"], "arguments": ""}}],
            }
        )
        args = call["arguments"]
        for i in range(
            0, len(args), 12
        ):  # arguments arrive in fragments, as most servers send them
            await asyncio.sleep(per_token * 3)
            yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[i : i + 12]}}]})
        yield chunk({}, "tool_calls")
        tail = {"id": cid, "object": "chat.completion.chunk", "created": 0, "model": model}
        yield f"data: {json.dumps({**tail, 'choices': [], 'usage': usage})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())
