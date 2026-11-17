"""A simulated OpenAI-compatible endpoint that runs inside the process.

It exists so the tool can be tried, demonstrated and tested without a key or network access.
Timings come from a seeded random generator around the configured profile. Every result that
includes a synthetic endpoint is marked as synthetic fixture data in every report.
"""

from __future__ import annotations

import asyncio
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
