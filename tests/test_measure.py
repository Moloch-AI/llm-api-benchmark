"""One request at a time: timings, streaming edge cases, errors, timeouts, cancellation."""

from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest
from tests.helpers import TEST_KEY, chunk, completion_body, paced, provider, sse, usage_chunk

from llm_api_benchmark.measure import estimate_tokens, execute
from llm_api_benchmark.providers.openai_chat import OpenAIChatAdapter
from llm_api_benchmark.records import RequestRecord

MSG = [{"role": "user", "content": "Say hi."}]


def record(mode: str = "stream") -> RequestRecord:
    return RequestRecord(
        index=0,
        provider="fake",
        model="fake-model",
        workload="w",
        prompt_id="p",
        mode=mode,
        repetition=0,
    )


def run(
    handler: object,
    *,
    stream: bool = True,
    timeout_s: float = 5.0,
    base_url: str = "https://fake.test/v1",
) -> tuple[RequestRecord, str]:
    p = provider(base_url=base_url)
    adapter = OpenAIChatAdapter(p, TEST_KEY, {})
    rec = record("stream" if stream else "plain")

    async def go() -> str:
        transport = httpx.MockTransport(handler) if handler is not None else None  # type: ignore[arg-type]
        async with httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(timeout_s)
        ) as client:
            req = adapter.build_request(MSG, stream=stream, max_tokens=16, temperature=None)
            reply = await execute(
                client, adapter, req, rec, stream=stream, timeout_s=timeout_s, prompt_chars=7
            )
            return reply.text

    text = asyncio.run(go())
    return rec, text


def stream_response(
    body: bytes | object, status: int = 200, ctype: str = "text/event-stream"
) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": ctype}, content=body)  # type: ignore[arg-type]


def test_streaming_timings_reasoning_and_usage() -> None:
    parts = [
        b": ka\n\n",
        sse(chunk(reasoning="Thinking"), done=False),
        sse(chunk("Hel"), done=False),
        sse(chunk("lo"), chunk(finish="stop"), usage_chunk(prompt=9, completion=12), done=False),
        b"data: [DONE]\n\n",
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(paced(parts, 0.05))

    rec, text = run(handler)
    assert rec.status == "ok"
    assert text == "Hello"
    assert rec.comment_lines == 1
    assert rec.ttfb_s is not None and rec.ttft_s is not None and rec.ttfc_s is not None
    assert rec.ttfb_s <= rec.ttft_s < rec.ttfc_s <= rec.latency_s  # type: ignore[operator]
    assert rec.ttfc_s - rec.ttft_s >= 0.04  # first answer came one paced chunk after the reasoning
    assert (rec.prompt_tokens, rec.completion_tokens, rec.usage_reported, rec.tokens_estimated) == (
        9,
        12,
        True,
        False,
    )
    assert rec.reasoning_chars == len("Thinking")
    assert rec.output_chars == 5
    assert rec.text_chunks == 3
    assert rec.finish_reason == "stop"
    assert rec.response_model == "fake-model-2026"
    assert rec.done_marker is True


def test_stream_without_usage_is_estimated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(sse(chunk("abcdefgh"), chunk(finish="stop")))

    rec, _ = run(handler)
    assert rec.status == "ok"
    assert rec.tokens_estimated is True
    assert rec.completion_tokens == 2
    assert rec.prompt_tokens == estimate_tokens(7)


def test_in_stream_error_event() -> None:
    err = {
        "error": {
            "type": "upstream_timeout",
            "message": "Server busy or request took too long.",
            "elapsed_s": 181.2,
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(sse(chunk("partial"), err))

    rec, text = run(handler)
    assert rec.status == "stream_error"
    assert rec.error_type == "upstream_timeout"
    assert rec.error_kind == "stream_error:upstream_timeout"
    assert text == "partial"


def test_named_error_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            b'event: error\ndata: {"message":"overloaded","type":"overloaded_error"}\n\n'
        )

    rec, _ = run(handler)
    assert rec.status == "stream_error"
    assert rec.error_type == "overloaded_error"


def test_stream_ending_early() -> None:
    def no_done_with_finish(request: httpx.Request) -> httpx.Response:
        return stream_response(sse(chunk("ok"), chunk(finish="stop"), done=False))

    rec, _ = run(no_done_with_finish)
    assert rec.status == "ok"
    assert rec.done_marker is False

    def cut(request: httpx.Request) -> httpx.Response:
        return stream_response(sse(chunk("partial"), done=False))

    rec2, _ = run(cut)
    assert rec2.status == "invalid_response"
    assert "ended" in (rec2.error_message or "")


def test_stream_with_json_content_type_is_invalid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body())

    rec, _ = run(handler)
    assert rec.status == "invalid_response"
    assert "text/event-stream" in (rec.error_message or "")


def test_bad_json_in_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(b"data: {not json}\n\n")

    rec, _ = run(handler)
    assert rec.status == "invalid_response"


def test_rate_limit_headers_are_captured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "7", "X-Should-Retry": "false"},
            json={"message": "Too Many Attempts."},
        )

    rec, _ = run(handler)
    assert (rec.status, rec.http_status, rec.retry_after_s, rec.should_retry) == (
        "http_error",
        429,
        7.0,
        "false",
    )
    assert rec.error_message == "Too Many Attempts."
    assert rec.error_kind == "http_429"


def test_cloudflare_html_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            524,
            headers={"content-type": "text/html"},
            content=b"<html><title>524: A timeout occurred</title></html>",
        )

    rec, _ = run(handler, stream=False)
    assert rec.http_status == 524
    assert rec.error_message == "(HTML) 524: A timeout occurred"


def test_redirect_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"location": "https://elsewhere.test/"})

    rec, _ = run(handler, stream=False)
    assert rec.status == "http_error"
    assert rec.http_status == 301


def test_plain_success_and_invalid_bodies() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content).get("stream") is None
        return httpx.Response(200, json=completion_body("Canberra", prompt=11, completion=2))

    rec, text = run(ok, stream=False)
    assert (rec.status, text, rec.completion_tokens, rec.prompt_tokens) == ("ok", "Canberra", 2, 11)
    assert rec.headers_s is not None and rec.latency_s is not None
    assert rec.ttft_s is None

    rec2, _ = run(lambda r: httpx.Response(200, content=b"not json"), stream=False)
    assert rec2.status == "invalid_response"
    rec3, _ = run(lambda r: httpx.Response(200, json={"choices": []}), stream=False)
    assert rec3.status == "invalid_response"


def test_total_timeout_on_a_slow_stream() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(paced([sse(chunk("a"), done=False)] * 50, 0.1))

    rec, _ = run(handler, timeout_s=0.35)
    assert rec.status == "timeout"
    assert rec.error_type == "total_timeout"
    assert rec.latency_s is not None and 0.3 <= rec.latency_s < 2.0


def test_read_timeout_from_httpx_is_a_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    rec, _ = run(handler)
    assert (rec.status, rec.error_type) == ("timeout", "read_timeout")


def test_connection_refused() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # The port is closed again now: connecting to it is refused (Windows retries for about 2 s first).
    rec, _ = run(None, base_url=f"http://127.0.0.1:{port}/v1")
    assert rec.status == "connection_error"
    assert rec.error_type == "connect_error"


def test_remote_protocol_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("peer closed connection", request=request)

    rec, _ = run(handler)
    assert (rec.status, rec.error_type) == ("connection_error", "remote_protocol_error")


def test_cancellation_marks_the_record() -> None:
    p = provider()
    adapter = OpenAIChatAdapter(p, TEST_KEY, {})
    rec = record()

    async def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(paced([sse(chunk("a"), done=False)] * 100, 0.05))

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            req = adapter.build_request(MSG, stream=True, max_tokens=4, temperature=None)
            task = asyncio.create_task(
                execute(client, adapter, req, rec, stream=True, timeout_s=10, prompt_chars=7)
            )
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(go())
    assert rec.status == "cancelled"
    assert not rec.executed


def test_real_socket_streaming(local_server: object) -> None:
    base = local_server.base_url  # type: ignore[attr-defined]
    rec, text = run(None, base_url=base)
    assert (rec.status, text, rec.completion_tokens) == ("ok", "Canberra", 2)
    assert rec.ttft_s is not None and rec.latency_s is not None and rec.ttft_s < rec.latency_s
