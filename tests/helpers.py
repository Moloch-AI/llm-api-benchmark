"""Test helpers: configurations, scripted HTTP transports and SSE bodies."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

import httpx

from llm_api_benchmark.config import ProviderConfig, RunConfig, parse_config
from llm_api_benchmark.datasets import Dataset, parse_dataset

TEST_KEY = "sk-test-SECRET-key-0123456789abcdef"


def provider_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "fake",
        "preset": "openai-compatible",
        "base_url": "https://fake.test/v1",
        "api_key_env": "FAKE_API_KEY",
        "model": "fake-model",
    }
    base.update(overrides)
    return base


def config(providers: Iterable[dict[str, Any]] | None = None, **top: Any) -> RunConfig:
    data: dict[str, Any] = {
        "name": "test",
        "dataset": "core-v1",
        "repetitions": 1,
        "warmup": 0,
        "providers": list(providers) if providers is not None else [provider_dict()],
    }
    data.update(top)
    return parse_config(data)


def provider(**overrides: Any) -> ProviderConfig:
    return config([provider_dict(**overrides)]).providers[0]


def tiny_dataset(prompts: int = 2, checks: list[dict[str, Any]] | None = None) -> Dataset:
    return parse_dataset(
        {
            "id": "tiny",
            "version": "1.0.0",
            "workloads": [
                {
                    "id": "w1",
                    "max_tokens": 16,
                    "prompts": [
                        {"id": f"p{i}", "prompt": f"Say hello {i}.", "checks": checks or []}
                        for i in range(prompts)
                    ],
                }
            ],
        },
        source="tiny",
    )


def sse(*events: Any, done: bool = True, comments: Iterable[str] = ()) -> bytes:
    out = "".join(f": {c}\n\n" for c in comments)
    for ev in events:
        out += f"data: {ev if isinstance(ev, str) else json.dumps(ev)}\n\n"
    if done:
        out += "data: [DONE]\n\n"
    return out.encode()


def chunk(
    content: str | None = None, *, reasoning: str | None = None, finish: str | None = None
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "model": "fake-model-2026",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def usage_chunk(prompt: int = 7, completion: int = 5) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def completion_body(
    text: str = "Hello there.", prompt: int = 7, completion: int = 3
) -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion",
        "model": "fake-model-2026",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


async def paced(parts: Iterable[bytes], delay: float) -> AsyncIterator[bytes]:
    for i, part in enumerate(parts):
        if i:
            await asyncio.sleep(delay)
        yield part


def ok_stream_response(
    text_parts: Iterable[str] = ("Hel", "lo", "."), completion: int = 3
) -> httpx.Response:
    events = [chunk(t) for t in text_parts] + [
        chunk(finish="stop"),
        usage_chunk(completion=completion),
    ]
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(*events))


class Recorder:
    """Transport handler that answers from a script and records every request."""

    def __init__(self, respond: Callable[[httpx.Request, int], httpx.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return self.respond(request, len(self.requests) - 1)
        finally:
            self.in_flight -= 1

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def default_respond(request: httpx.Request, n: int) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return ok_stream_response()
    return httpx.Response(200, json=completion_body())
