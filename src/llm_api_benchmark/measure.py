"""Send one request and record what happened, with timings taken on the client.

All times are seconds from the moment the request is handed to the HTTP client
(``time.perf_counter``), so they include DNS, TLS (on a new connection), upload, queueing at the
provider and download. Streaming responses are read line by line as they arrive.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from .providers.base import Adapter, ChatRequest, InvalidResponse, Usage
from .records import (
    CANCELLED,
    CONNECTION_ERROR,
    HTTP_ERROR,
    INVALID_RESPONSE,
    OK,
    STREAM_ERROR,
    TIMEOUT,
    RequestRecord,
)
from .redact import redact
from .sse import Comment, SSEParser
from .tools import ToolCall, ToolCallAccumulator, signature

MAX_ERROR_BODY = 64 * 1024
Clock = Callable[[], float]


@dataclass
class Reply:
    """What the model returned, for the checks: the answer text and any tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def estimate_tokens(chars: int) -> int:
    """Rough token count from characters (about 4 per token for English). Labeled as estimated."""
    return math.ceil(chars / 4) if chars > 0 else 0


async def execute(
    client: httpx.AsyncClient,
    adapter: Adapter,
    request: ChatRequest,
    rec: RequestRecord,
    *,
    stream: bool,
    timeout_s: float,
    prompt_chars: int,
    clock: Clock = time.perf_counter,
) -> Reply:
    """Run one request, fill ``rec`` in place and return the answer and tool calls (for checks)."""
    t0 = clock()
    content: list[str] = []
    calls = ToolCallAccumulator()
    plain_calls: list[ToolCall] = []
    reasoning_chars = 0
    usage: Usage | None = None
    try:
        async with asyncio.timeout(timeout_s):
            async with client.stream(
                "POST",
                request.url,
                headers=request.headers,
                params=request.query,
                json=request.body,
            ) as resp:
                rec.headers_s = clock() - t0
                rec.http_status = resp.status_code
                _rate_headers(rec, resp)
                if resp.status_code >= 300:  # redirects are not followed: a wrong base_url
                    body = await _read_limited(resp)
                    rec.latency_s = clock() - t0
                    err = adapter.parse_error(
                        resp.status_code, body, resp.headers.get("content-type", "")
                    )
                    rec.status = HTTP_ERROR
                    rec.error_type, rec.error_message = err.type, err.message
                    return Reply()
                ctype = resp.headers.get("content-type", "")
                if stream:
                    if "text/event-stream" not in ctype:
                        body = await _read_limited(resp)
                        rec.latency_s = clock() - t0
                        rec.status = INVALID_RESPONSE
                        rec.error_message = redact(
                            f"expected text/event-stream, got {ctype or 'no content type'}: "
                            + body[:200].decode("utf-8", "replace")
                        )
                        return Reply()
                    usage, reasoning_chars = await _read_stream(
                        adapter, resp, rec, content, calls, t0, clock
                    )
                else:
                    raw = await resp.aread()
                    rec.ttfb_s = rec.headers_s
                    rec.latency_s = clock() - t0
                    try:
                        info = adapter.parse_completion(json.loads(raw))
                    except ValueError as err:  # JSONDecodeError and InvalidResponse
                        rec.status = INVALID_RESPONSE
                        rec.error_message = redact(
                            str(err) if isinstance(err, InvalidResponse) else "not JSON"
                        )
                        return Reply()
                    content.append(info.content)
                    plain_calls.extend(info.tool_calls)
                    reasoning_chars = len(info.reasoning)
                    usage = info.usage
                    rec.finish_reason = info.finish_reason
                    rec.response_model = info.model
                    rec.status = OK
    except TimeoutError:
        _fail(
            rec, TIMEOUT, "total_timeout", f"no complete response within {timeout_s:g} s", t0, clock
        )
    except httpx.TimeoutException as err:
        kind = {
            httpx.ConnectTimeout: "connect_timeout",
            httpx.ReadTimeout: "read_timeout",
            httpx.WriteTimeout: "write_timeout",
            httpx.PoolTimeout: "pool_timeout",
        }.get(type(err), "timeout")
        _fail(rec, TIMEOUT, kind, str(err) or kind, t0, clock)
    except httpx.ConnectError as err:
        _fail(rec, CONNECTION_ERROR, "connect_error", str(err) or "could not connect", t0, clock)
    except httpx.RemoteProtocolError as err:
        _fail(
            rec,
            CONNECTION_ERROR,
            "remote_protocol_error",
            str(err) or "connection closed",
            t0,
            clock,
        )
    except httpx.HTTPError as err:
        _fail(rec, CONNECTION_ERROR, type(err).__name__, str(err) or type(err).__name__, t0, clock)
    except InvalidResponse as err:
        _fail(rec, INVALID_RESPONSE, None, str(err), t0, clock)
    except asyncio.CancelledError:
        _fail(rec, CANCELLED, None, "cancelled", t0, clock)
        raise

    text = "".join(content)
    tool_calls = plain_calls or calls.calls()
    tool_chars = sum(len(c.name) + len(c.arguments) for c in tool_calls)
    rec.output_chars = len(text)
    rec.reasoning_chars = reasoning_chars
    if "tools" in request.body or tool_calls:
        rec.tool_call_names = [c.name for c in tool_calls]
        if rec.status == OK:
            rec.tool_signature = signature(tool_calls)
    if usage is not None and usage.completion_tokens is not None:
        rec.usage_reported = True
        rec.prompt_tokens = usage.prompt_tokens
        rec.completion_tokens = usage.completion_tokens
        rec.reasoning_tokens = usage.reasoning_tokens
    elif rec.status in (OK, STREAM_ERROR) and (rec.output_chars or reasoning_chars or tool_chars):
        rec.tokens_estimated = True
        rec.completion_tokens = estimate_tokens(rec.output_chars + reasoning_chars + tool_chars)
    if rec.prompt_tokens is None and rec.completion_tokens is not None:
        rec.prompt_tokens = estimate_tokens(prompt_chars)
        rec.tokens_estimated = True
    return Reply(text, tool_calls)


async def _read_stream(
    adapter: Adapter,
    resp: httpx.Response,
    rec: RequestRecord,
    content: list[str],
    calls: ToolCallAccumulator,
    t0: float,
    clock: Clock,
) -> tuple[Usage | None, int]:
    parser = SSEParser()
    usage: Usage | None = None
    reasoning_chars = 0
    done = False
    last = t0
    async for line in resp.aiter_lines():
        now = last = clock()
        if rec.ttfb_s is None:
            rec.ttfb_s = now - t0
        for item in parser.feed(line):
            if isinstance(item, Comment):
                rec.comment_lines += 1
                continue
            if item.data.strip() == "[DONE]":
                done = True
                break
            try:
                obj = json.loads(item.data)
            except ValueError as err:
                raise InvalidResponse("a stream event is not valid JSON") from err
            chunk = adapter.parse_chunk(
                obj if item.event != "error" else {"error": obj.get("error", obj)}
            )
            if chunk.error:
                if rec.status != STREAM_ERROR:
                    rec.status = STREAM_ERROR
                    rec.error_type, rec.error_message = chunk.error.type, chunk.error.message
                continue
            if chunk.reasoning or chunk.content or chunk.tool_calls:
                if rec.ttft_s is None:
                    rec.ttft_s = now - t0
                rec.last_token_s = now - t0
                rec.text_chunks += 1
            if chunk.reasoning:
                reasoning_chars += len(chunk.reasoning)
            if chunk.content or chunk.tool_calls:
                # The answer starts with its first content or tool-call fragment.
                if rec.ttfc_s is None:
                    rec.ttfc_s = now - t0
                if chunk.content:
                    content.append(chunk.content)
                for delta in chunk.tool_calls:
                    calls.add(delta)
            if chunk.usage:
                usage = chunk.usage
            if chunk.model and rec.response_model is None:
                rec.response_model = chunk.model
            if chunk.finish_reason:
                rec.finish_reason = chunk.finish_reason
        if done:
            break
    if not done:
        for item in parser.flush():
            if not isinstance(item, Comment) and item.data.strip() == "[DONE]":
                done = True
    rec.done_marker = done
    rec.latency_s = last - t0
    if rec.status == STREAM_ERROR:
        return usage, reasoning_chars
    if not done and rec.finish_reason is None:
        rec.status = INVALID_RESPONSE
        rec.error_message = "the stream ended without [DONE] and without a finish_reason"
    else:
        rec.status = OK
    return usage, reasoning_chars


async def _read_limited(resp: httpx.Response) -> bytes:
    buf = bytearray()
    async for part in resp.aiter_bytes():
        buf.extend(part)
        if len(buf) >= MAX_ERROR_BODY:
            break
    return bytes(buf[:MAX_ERROR_BODY])


def _rate_headers(rec: RequestRecord, resp: httpx.Response) -> None:
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            rec.retry_after_s = max(0.0, float(retry_after))
        except ValueError:
            rec.retry_after_s = None  # HTTP-date form: not used for pacing
    should = resp.headers.get("x-should-retry")
    if should:
        rec.should_retry = should.strip().lower()


def _fail(
    rec: RequestRecord, status: str, kind: str | None, message: str, t0: float, clock: Clock
) -> None:
    rec.status = status
    rec.error_type = kind
    rec.error_message = redact(message)[:300]
    rec.latency_s = clock() - t0
