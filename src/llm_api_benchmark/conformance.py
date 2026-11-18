"""A conformance suite for OpenAI-compatible Chat Completions endpoints.

Each test sends one small request (or none: ``chat.usage`` reads the reply of ``chat.basic``) and
reports one of five outcomes:

``pass``   the endpoint behaved as the OpenAI API does;
``warn``   it works, with a difference worth knowing (a non-OpenAI error body, a parameter that is
           rejected with a clear 4xx instead of being supported, an inconclusive reply);
``fail``   it differs in a way that breaks clients: a parameter accepted and silently ignored, a
           missing field, a malformed stream, a server error on a client mistake;
``skip``   not applicable (for example no auth tests on an endpoint without auth);
``error``  the test could not be judged (network error, rate limit, a 5xx, a stopped run).

The suite tests the wire format and parameter handling, not answer quality. Prompts are short and
harmless, and replies are bounded by the prompt even where ``max_tokens`` is ignored. A result
holds for one endpoint, one model and one moment: rerun it after the provider changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from . import __version__
from .config import ProviderConfig, SyntheticProfile
from .providers.reference import REFERENCE_KEY
from .redact import REDACTOR, redact
from .schema import validate
from .sse import Comment, SSEParser

PASS, WARN, FAIL, SKIP, ERROR = "pass", "warn", "fail", "skip", "error"
OUTCOMES = (PASS, WARN, FAIL, SKIP, ERROR)
SCHEMA = "llm-api-benchmark/conformance/v1"
INVALID_KEY = "invalid-key-for-a-conformance-check"
UNKNOWN_MODEL = "conformance-check-unknown-model"
_THINK = re.compile(r"(?is)<think>.*?(</think>|$)")
_MAX_BODY = 256 * 1024
_HEADER_HINTS = re.compile(r"^(x-request-id|request-id|x-ratelimit-.*|retry-after|x-should-retry)$")

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
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
CAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "enum": ["black", "white", "orange", "grey", "brown", "mixed"]},
        "legs": {"type": "integer"},
    },
    "required": ["color", "legs"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- results


@dataclass
class TestResult:
    id: str
    category: str
    title: str
    outcome: str = SKIP
    summary: str = ""
    details: list[str] = field(default_factory=list)
    http_status: int | None = None
    latency_s: float | None = None
    requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "outcome": self.outcome,
            "summary": self.summary,
            "details": self.details,
            "http_status": self.http_status,
            "latency_s": None if self.latency_s is None else round(self.latency_s, 3),
            "requests": self.requests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestResult:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class ProviderReport:
    provider: dict[str, Any]
    results: list[TestResult]
    started_at: str
    finished_at: str
    headers_seen: list[str]
    stopped: str | None = None

    def counts(self) -> dict[str, int]:
        return {o: sum(1 for r in self.results if r.outcome == o) for o in OUTCOMES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "counts": self.counts(),
            "headers_seen": self.headers_seen,
            "stopped": self.stopped,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ConformanceResult:
    reports: list[ProviderReport]
    environment: dict[str, Any]
    synthetic: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "tool": {"name": "llm-api-benchmark", "version": __version__},
            "synthetic": self.synthetic,
            "environment": self.environment,
            "tests": [{"id": t.id, "category": t.category, "title": t.title} for t in TESTS],
            "reports": [r.to_dict() for r in self.reports],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformanceResult:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not a conformance result file (schema {data.get('schema')!r})")
        reports = [
            ProviderReport(
                provider=r["provider"],
                results=[TestResult.from_dict(x) for x in r["results"]],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
                headers_seen=list(r.get("headers_seen") or []),
                stopped=r.get("stopped"),
            )
            for r in data["reports"]
        ]
        return cls(reports, dict(data.get("environment") or {}), bool(data.get("synthetic")))

    @property
    def failed(self) -> bool:
        return any(r.outcome == FAIL for rep in self.reports for r in rep.results)


# --------------------------------------------------------------------------- exchanges


@dataclass
class Exchange:
    """One request and what came back, kept raw so tests can judge it."""

    status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    data: Any = None
    json_error: str | None = None
    content_type: str = ""
    events: list[Any] = field(default_factory=list)  # parsed data events (stream)
    bad_events: int = 0
    done: bool = False
    comments: int = 0
    transport_error: str | None = None
    elapsed_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.transport_error is None

    def text(self, limit: int = 200) -> str:
        return redact(" ".join(self.body[:limit].decode("utf-8", "replace").split()))


class StopSuite(Exception):
    """A permanent problem (bad key, no plan): the rest of this endpoint's tests are not run."""


class Context:
    """What a test can do: send requests to one endpoint, paced and bounded."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        provider: ProviderConfig,
        *,
        timeout_s: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.provider = provider
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self.requests = 0
        self.headers_seen: set[str] = set()
        self.shared: dict[str, Exchange] = {}
        if provider.is_synthetic:
            self._key: str | None = REFERENCE_KEY
        else:
            self._key = provider.api_key() if provider.needs_key else None
            REDACTOR.add(self._key)
        self._extra: dict[str, str] = {}
        for header, env in provider.headers_env.items():
            value = os.environ.get(env, "").strip()
            if value:
                REDACTOR.add(value)
                self._extra[header] = value

    @property
    def has_auth(self) -> bool:
        return _has_auth(self.provider)

    def headers(self, auth: str) -> dict[str, str]:
        p = self.provider
        out = {"Content-Type": "application/json", "Accept": "application/json"}
        out.update(p.headers)
        out.update(self._extra)
        key = {"valid": self._key, "invalid": INVALID_KEY, "none": None}[auth]
        if key and (p.auth_scheme == "bearer" or p.is_synthetic):
            out["Authorization" if p.is_synthetic else p.auth_header] = f"Bearer {key}"
        elif key and p.auth_scheme == "header":
            out[p.auth_header] = key
        return out

    async def _pace(self) -> None:
        rpm = self.provider.rate_limit_rpm
        if rpm and self._last is not None:
            wait = 60.0 / rpm - (self._clock() - self._last)
            if wait > 0:
                await self._sleep(wait)
        self._last = self._clock()

    async def send(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        raw: bytes | None = None,
        auth: str = "valid",
        stream: bool = False,
    ) -> Exchange:
        for attempt in range(2):
            ex = await self._send_once(method, path, body=body, raw=raw, auth=auth, stream=stream)
            retry_after = _retry_after(ex)
            if ex.status == 429 and attempt == 0 and retry_after is not None and retry_after <= 60:
                await self._sleep(retry_after)
                continue
            break
        if auth == "valid" and ex.status in (401, 402, 403):
            raise StopSuite(
                f"HTTP {ex.status} with the configured key ({error_message(ex)}); "
                "check the key, plan and base_url"
            )
        return ex

    async def _send_once(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None,
        raw: bytes | None,
        auth: str,
        stream: bool,
    ) -> Exchange:
        await self._pace()
        self.requests += 1
        url = f"{self.provider.base_url}{path}"
        headers = self.headers(auth)
        if stream:
            headers["Accept"] = "text/event-stream"
        content = (
            raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        )
        ex = Exchange()
        t0 = self._clock()
        try:
            async with asyncio.timeout(self.timeout_s):
                async with self.client.stream(
                    method, url, headers=headers, params=dict(self.provider.query), content=content
                ) as resp:
                    ex.status = resp.status_code
                    ex.headers = {k.lower(): v for k, v in resp.headers.items()}
                    ex.content_type = resp.headers.get("content-type", "")
                    if (
                        stream
                        and resp.status_code == 200
                        and "text/event-stream" in ex.content_type
                    ):
                        await _read_events(resp, ex)
                    else:
                        buf = bytearray()
                        async for part in resp.aiter_bytes():
                            buf.extend(part)
                            if len(buf) > _MAX_BODY:
                                break
                        ex.body = bytes(buf)
                        if ex.body.strip():
                            try:
                                ex.data = json.loads(ex.body)
                            except ValueError as err:
                                ex.json_error = str(err)
        except TimeoutError:
            ex.transport_error = f"no complete response within {self.timeout_s:g} s"
        except httpx.HTTPError as err:
            ex.transport_error = redact(f"{type(err).__name__}: {err}")[:200]
        ex.elapsed_s = self._clock() - t0
        self.headers_seen.update(h for h in ex.headers if _HEADER_HINTS.match(h))
        return ex

    def chat_body(self, **fields: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.provider.model}
        for key, value in self.provider.params.items():
            if value is not None:
                body[key] = value
        body["messages"] = [{"role": "user", "content": "Reply with the single word: ready"}]
        body.update(fields)
        return body

    async def chat(self, *, stream: bool = False, auth: str = "valid", **fields: Any) -> Exchange:
        body = self.chat_body(**fields)
        if stream:
            body["stream"] = True
        return await self.send("POST", "/chat/completions", body=body, auth=auth, stream=stream)


async def _read_events(resp: httpx.Response, ex: Exchange) -> None:
    parser = SSEParser()
    raw = bytearray()

    def take(items: list[Any]) -> None:
        for item in items:
            if isinstance(item, Comment):
                ex.comments += 1
            elif item.data.strip() == "[DONE]":
                ex.done = True
            else:
                try:
                    ex.events.append(json.loads(item.data))
                except ValueError:
                    ex.bad_events += 1

    async for line in resp.aiter_lines():
        raw.extend(line.encode()[:1000] + b"\n")
        take(parser.feed(line))
    take(parser.flush())
    ex.body = bytes(raw[:_MAX_BODY])


def _has_auth(provider: ProviderConfig) -> bool:
    return provider.is_synthetic or provider.auth_scheme != "none"


def _retry_after(ex: Exchange) -> float | None:
    value = ex.headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value else None
    except ValueError:
        return None


# --------------------------------------------------------------------------- helpers for tests


def error_shape(ex: Exchange) -> tuple[str, str]:
    """("openai" | "other-json" | "not-json" | "empty", message) for an error response."""
    if not ex.body.strip():
        return "empty", "(empty body)"
    data = ex.data
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return "openai", redact(err["message"])[:200]
        message = data.get("message") or data.get("detail") or data.get("error")
        return "other-json", redact(str(message))[:200] if message else json.dumps(data)[:200]
    title = re.search(r"(?is)<title>(.*?)</title>", ex.body.decode("utf-8", "replace"))
    return "not-json", (f"(HTML) {' '.join(title.group(1).split())}" if title else ex.text())


def error_message(ex: Exchange) -> str:
    if ex.transport_error:
        return ex.transport_error
    return error_shape(ex)[1]


def answer(choice: Any) -> str:
    """The answer text of a plain reply's choice, without <think> blocks."""
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return _THINK.sub("", content).strip() if isinstance(content, str) else ""


def first_choice(ex: Exchange) -> dict[str, Any] | None:
    if not isinstance(ex.data, dict):
        return None
    choices = ex.data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return None


def stream_text(ex: Exchange) -> tuple[str, str]:
    """(answer text, reasoning text) joined from a stream's deltas."""
    content, reasoning = [], []
    for ev in ex.events:
        for choice in (ev.get("choices") or []) if isinstance(ev, dict) else []:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for key in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str):
                    reasoning.append(delta[key])
    return _THINK.sub("", "".join(content)).strip(), "".join(reasoning)


def _usage_ints(usage: Any) -> tuple[int, int, int] | None:
    if not isinstance(usage, dict):
        return None
    values = [usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
    ints = [v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
    if len(ints) == 3:
        return ints[0], ints[1], ints[2]
    return None


def _not_ok(r: TestResult, ex: Exchange, what: str = "request") -> TestResult:
    """Judge a failed request that was expected to succeed."""
    r.http_status, r.latency_s = ex.status, ex.elapsed_s
    if ex.transport_error:
        r.outcome, r.summary = ERROR, f"{what} failed: {ex.transport_error}"
    elif ex.status is not None and ex.status >= 500:
        r.outcome, r.summary = ERROR, f"HTTP {ex.status}: {error_message(ex)}"
    elif ex.status == 429:
        r.outcome, r.summary = ERROR, f"rate limited (HTTP 429): {error_message(ex)}"
    else:
        r.outcome, r.summary = FAIL, f"HTTP {ex.status}: {error_message(ex)}"
    return r


def _unsupported(r: TestResult, ex: Exchange, param: str) -> TestResult:
    """A parameter request that did not return 200: an explicit 4xx is honest, not a failure."""
    if ex.status in (400, 404, 422) and not ex.transport_error:
        r.http_status, r.latency_s = ex.status, ex.elapsed_s
        r.outcome = WARN
        r.summary = f"not supported: rejected with HTTP {ex.status} ({error_message(ex)})"
        r.details.append(f"An explicit error is safer than silently ignoring {param}.")
        return r
    return _not_ok(r, ex)


def _done(r: TestResult, ex: Exchange, outcome: str, summary: str, *details: str) -> TestResult:
    r.http_status, r.latency_s = ex.status, ex.elapsed_s
    r.outcome, r.summary = outcome, summary
    r.details.extend(d for d in details if d)
    return r


# --------------------------------------------------------------------------- tests


@dataclass(frozen=True)
class TestSpec:
    id: str
    category: str
    title: str
    fn: Callable[[Context, TestResult], Awaitable[TestResult]]
    requests: int = 1
    needs_auth: bool = False


async def t_models_list(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.send("GET", "/models")
    if not ex.ok:
        return _not_ok(r, ex, "GET /models")
    data = ex.data
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return _done(r, ex, FAIL, "the body has no data list", f"got: {ex.text()}")
    ids = [m.get("id") for m in items if isinstance(m, dict)]
    if not ids or not all(isinstance(i, str) and i for i in ids):
        return _done(r, ex, FAIL, "data[] entries need a string id")
    notes: list[str] = []
    outcome = PASS
    if data.get("object") != "list":
        outcome, notes = WARN, [*notes, f'object is {data.get("object")!r}, expected "list"']
    if any(isinstance(m, dict) and m.get("object") != "model" for m in items):
        outcome = WARN
        notes.append('some entries lack object: "model"')
    if ctx.provider.model not in ids:
        outcome = WARN
        notes.append(f"the configured model {ctx.provider.model!r} is not in the list")
    return _done(r, ex, outcome, f"{len(ids)} model(s) listed", *notes)


async def t_chat_basic(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(
        messages=[
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Reply with the single word: ready"},
        ]
    )
    ctx.shared["chat.basic"] = ex
    if not ex.ok:
        return _not_ok(r, ex)
    data = ex.data
    if not isinstance(data, dict):
        return _done(r, ex, FAIL, "the body is not a JSON object", f"got: {ex.text()}")
    problems, notes = [], []
    choice = first_choice(ex)
    if choice is None:
        problems.append("choices is missing or empty")
    else:
        message = choice.get("message")
        if not isinstance(message, dict):
            problems.append("choices[0].message is missing")
        else:
            if message.get("role") != "assistant":
                problems.append(f"message.role is {message.get('role')!r}, expected assistant")
            if not answer(choice):
                problems.append("message.content is empty")
        if not isinstance(choice.get("finish_reason"), str):
            problems.append("finish_reason is missing")
    if not isinstance(data.get("id"), str):
        problems.append("id is missing")
    if data.get("object") != "chat.completion":
        notes.append(f'object is {data.get("object")!r}, expected "chat.completion"')
    if not isinstance(data.get("created"), int):
        notes.append("created is missing or not an integer")
    if not isinstance(data.get("model"), str):
        notes.append("model is missing")
    if problems:
        return _done(r, ex, FAIL, "; ".join(problems), *notes)
    reply = answer(choice)[:60]
    return _done(r, ex, WARN if notes else PASS, f"answered {reply!r}", *notes)


async def t_chat_usage(ctx: Context, r: TestResult) -> TestResult:
    ex = ctx.shared.get("chat.basic")
    if ex is None or not ex.ok or not isinstance(ex.data, dict):
        r.outcome, r.summary = SKIP, "needs a successful chat.basic reply"
        return r
    usage = ex.data.get("usage")
    counts = _usage_ints(usage)
    if counts is None:
        return _done(
            r, ex, FAIL, "usage with integer prompt, completion and total tokens is missing"
        )
    prompt, completion, total = counts
    summary = f"prompt {prompt}, completion {completion}, total {total}"
    if total != prompt + completion:
        return _done(r, ex, WARN, summary, "total_tokens is not prompt_tokens + completion_tokens")
    return _done(r, ex, PASS, summary)


async def t_stream_basic(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(stream=True)
    ctx.shared["stream.basic"] = ex
    if ex.status != 200 or ex.transport_error:
        return _not_ok(r, ex)
    if "text/event-stream" not in ex.content_type:
        return _done(
            r, ex, FAIL, f"content-type is {ex.content_type!r}, expected text/event-stream"
        )
    problems, notes = [], []
    if ex.bad_events:
        problems.append(f"{ex.bad_events} data event(s) are not JSON")
    content, reasoning = stream_text(ex)
    if not content:
        problems.append("no content delta")
    finishes = [
        c.get("finish_reason")
        for ev in ex.events
        if isinstance(ev, dict)
        for c in (ev.get("choices") or [])
        if isinstance(c, dict) and c.get("finish_reason")
    ]
    if not finishes:
        problems.append("no chunk carries a finish_reason")
    if not ex.done:
        problems.append("the stream did not end with data: [DONE]")
    errors = [ev for ev in ex.events if isinstance(ev, dict) and ev.get("error")]
    if errors:
        problems.append(f"error event in the stream: {redact(json.dumps(errors[0]))[:160]}")
    objects = {ev.get("object") for ev in ex.events if isinstance(ev, dict) and ev.get("choices")}
    if objects - {"chat.completion.chunk"}:
        notes.append(f"chunk object is {sorted(map(str, objects))}, expected chat.completion.chunk")
    if ex.comments:
        notes.append(
            f"{ex.comments} SSE comment line(s): clients must ignore lines starting with ':'"
        )
    if reasoning:
        notes.append("reasoning arrives in a separate delta field (reasoning_content or reasoning)")
    if problems:
        return _done(r, ex, FAIL, "; ".join(problems), *notes)
    return _done(r, ex, PASS, f"{len(ex.events)} chunks, answered {content[:40]!r}", *notes)


async def t_stream_usage(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(stream=True, stream_options={"include_usage": True})
    if ex.status != 200 or ex.transport_error:
        return _unsupported(r, ex, "stream_options")
    usage_events = [
        ev for ev in ex.events if isinstance(ev, dict) and _usage_ints(ev.get("usage")) is not None
    ]
    basic = ctx.shared.get("stream.basic")
    always = basic is not None and any(
        isinstance(ev, dict) and _usage_ints(ev.get("usage")) for ev in basic.events
    )
    note = "usage is also sent without stream_options" if always else ""
    if not usage_events:
        return _done(r, ex, FAIL, "include_usage was requested but no chunk carries usage")
    last = usage_events[-1]
    if last.get("choices"):
        return _done(
            r,
            ex,
            WARN,
            "usage arrives on a chunk that also has choices",
            "OpenAI sends usage on a final chunk with an empty choices list",
            note,
        )
    p, c, t = _usage_ints(last.get("usage")) or (0, 0, 0)
    return _done(r, ex, PASS, f"final usage chunk: prompt {p}, completion {c}, total {t}", note)


async def _max_tokens(ctx: Context, r: TestResult, param: str) -> TestResult:
    limit = 8
    fields: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Count from 1 to 60, separated by spaces."}],
        param: limit,
    }
    ex = await ctx.chat(**fields)
    if not ex.ok:
        return _unsupported(r, ex, param)
    choice = first_choice(ex) or {}
    usage = _usage_ints((ex.data or {}).get("usage") if isinstance(ex.data, dict) else None)
    finish = choice.get("finish_reason")
    if usage is not None:
        completion = usage[1]
        if completion <= limit + 1:
            outcome = PASS if finish == "length" else WARN
            note = "" if finish == "length" else f"finish_reason is {finish!r}, expected length"
            return _done(r, ex, outcome, f"{completion} completion tokens (limit {limit})", note)
        return _done(
            r,
            ex,
            FAIL,
            f"ignored: {completion} completion tokens with {param}={limit}",
            f"finish_reason {finish!r}",
        )
    words = len(answer(choice).split())
    if words > limit * 2:
        return _done(
            r, ex, FAIL, f"ignored: {words} words with {param}={limit} (no usage reported)"
        )
    return _done(r, ex, WARN, f"{words} words; no usage reported to confirm the limit")


async def t_max_tokens(ctx: Context, r: TestResult) -> TestResult:
    return await _max_tokens(ctx, r, "max_tokens")


async def t_max_completion_tokens(ctx: Context, r: TestResult) -> TestResult:
    return await _max_tokens(ctx, r, "max_completion_tokens")


async def t_stop(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(
        messages=[
            {
                "role": "user",
                "content": "Write the numbers from 1 to 12 in order, separated by single spaces, "
                "and nothing else.",
            }
        ],
        stop=["7"],
    )
    if not ex.ok:
        return _unsupported(r, ex, "stop")
    text = answer(first_choice(ex))
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if any(n >= 8 for n in numbers):
        return _done(r, ex, FAIL, f"ignored: the reply runs past the stop sequence ({text[:50]!r})")
    if 6 in numbers and 7 not in numbers:
        return _done(r, ex, PASS, f"stopped before '7' ({text[:40]!r})")
    return _done(r, ex, WARN, f"inconclusive reply {text[:50]!r}")


async def t_n(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(messages=[{"role": "user", "content": "Say hello in one word."}], n=2)
    if not ex.ok:
        return _unsupported(r, ex, "n")
    raw = ex.data.get("choices") if isinstance(ex.data, dict) else None
    choices: list[Any] = raw if isinstance(raw, list) else []
    count = len(choices)
    if count == 2:
        indexes = sorted(
            c["index"] for c in choices if isinstance(c, dict) and isinstance(c.get("index"), int)
        )
        note = "" if indexes == [0, 1] else f"choice indexes are {indexes}, expected [0, 1]"
        return _done(r, ex, WARN if note else PASS, "two choices returned", note)
    return _done(r, ex, FAIL, f"ignored: {count} choice(s) returned for n=2")


async def t_json_object(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(
        messages=[
            {
                "role": "user",
                "content": "Describe a typical house cat as a JSON object with the keys "
                "color and legs.",
            }
        ],
        response_format={"type": "json_object"},
    )
    if not ex.ok:
        return _unsupported(r, ex, "response_format json_object")
    text = answer(first_choice(ex))
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return _done(r, ex, PASS, "the reply is a bare JSON object")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return _done(
            r,
            ex,
            WARN,
            "JSON only inside a code fence: JSON mode was likely not applied",
            "The prompt asks for JSON, so a reply can look right without JSON mode.",
        )
    return _done(r, ex, FAIL, f"not a JSON object: {text[:60]!r}")


async def t_json_schema(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(
        messages=[{"role": "user", "content": "Describe a typical house cat in one sentence."}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "cat", "strict": True, "schema": CAT_SCHEMA},
        },
    )
    if not ex.ok:
        return _unsupported(r, ex, "response_format json_schema")
    text = answer(first_choice(ex))
    try:
        value = json.loads(text)
    except ValueError:
        return _done(
            r,
            ex,
            FAIL,
            f"ignored: the reply is not JSON ({text[:50]!r})",
            "The prompt does not mention JSON, so only the schema could make the reply JSON.",
        )
    errors = validate(value, CAT_SCHEMA)
    if errors:
        return _done(r, ex, FAIL, f"JSON, but the schema is not enforced: {errors[0]}")
    return _done(r, ex, PASS, "the reply matches the schema")


async def t_logprobs(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(logprobs=True, top_logprobs=1)
    if not ex.ok:
        return _unsupported(r, ex, "logprobs")
    lp = (first_choice(ex) or {}).get("logprobs")
    content = lp.get("content") if isinstance(lp, dict) else None
    if isinstance(content, list) and content and isinstance(content[0], dict):
        if "logprob" in content[0] and "token" in content[0]:
            return _done(r, ex, PASS, f"logprobs for {len(content)} token(s)")
        return _done(r, ex, WARN, "logprobs.content entries lack token or logprob")
    return _done(r, ex, FAIL, "ignored: choices[0].logprobs is missing")


async def t_temperature_range(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(temperature=3.5)
    if ex.status in (400, 422):
        return _done(r, ex, PASS, f"rejected with HTTP {ex.status} ({error_shape(ex)[1]})")
    if ex.ok:
        return _done(
            r,
            ex,
            WARN,
            "accepted temperature=3.5 (the OpenAI range is 0 to 2)",
            "Out-of-range values are clamped or dropped silently.",
        )
    return _not_ok(r, ex)


def _check_call(call: Any, r: TestResult) -> tuple[list[str], list[str], Any]:
    problems: list[str] = []
    notes: list[str] = []
    if not isinstance(call, dict):
        return ["a tool_calls entry is not an object"], notes, None
    fn = call.get("function") or {}
    if not isinstance(call.get("id"), str) or not call.get("id"):
        problems.append("tool call id is missing")
    if call.get("type") != "function":
        notes.append(f'type is {call.get("type")!r}, expected "function"')
    if fn.get("name") != "get_weather":
        problems.append(f"function.name is {fn.get('name')!r}, expected get_weather")
    args = fn.get("arguments")
    if not isinstance(args, str):
        problems.append(f"function.arguments is {type(args).__name__}, the spec says a JSON string")
        return problems, notes, args
    try:
        value = json.loads(args)
    except ValueError:
        problems.append(f"function.arguments is not valid JSON: {args[:60]!r}")
        return problems, notes, None
    errors = validate(value, WEATHER_TOOL["function"]["parameters"])
    if errors:
        notes.append(f"arguments do not match the schema: {errors[0]} (model quality, not format)")
    return problems, notes, value


FORCED = {"type": "function", "function": {"name": "get_weather"}}
WEATHER_PROMPT = [{"role": "user", "content": "What is the weather in Paris? Use celsius."}]


async def t_tools_call(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(messages=WEATHER_PROMPT, tools=[WEATHER_TOOL], tool_choice=FORCED)
    if not ex.ok:
        return _unsupported(r, ex, "tools")
    choice = first_choice(ex) or {}
    message = choice.get("message") or {}
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or not calls:
        return _done(r, ex, FAIL, "no tool_calls in the reply although tool_choice forced one")
    problems, notes, value = _check_call(calls[0], r)
    finish = choice.get("finish_reason")
    if finish not in ("tool_calls", "stop"):
        notes.append(f"finish_reason is {finish!r}")
    if problems:
        return _done(r, ex, FAIL, "; ".join(problems), *notes)
    return _done(r, ex, WARN if notes else PASS, f"get_weather({json.dumps(value)})", *notes)


async def t_tools_stream(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(
        stream=True, messages=WEATHER_PROMPT, tools=[WEATHER_TOOL], tool_choice=FORCED
    )
    if ex.status != 200 or ex.transport_error:
        return _unsupported(r, ex, "tools in a stream")
    calls: dict[int, dict[str, Any]] = {}
    fragments = 0
    problems: list[str] = []
    for ev in ex.events:
        for choice in (ev.get("choices") or []) if isinstance(ev, dict) else []:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            for d in (delta or {}).get("tool_calls") or [] if isinstance(delta, dict) else []:
                if not isinstance(d, dict):
                    continue
                if not isinstance(d.get("index"), int):
                    problems.append("a tool_calls delta has no integer index")
                    continue
                call = calls.setdefault(d["index"], {"id": None, "name": None, "args": []})
                fn = d.get("function") or {}
                call["id"] = call["id"] or d.get("id")
                call["name"] = call["name"] or fn.get("name")
                if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                    call["args"].append(fn["arguments"])
                    fragments += 1
    if not calls:
        return _done(r, ex, FAIL, "no tool_calls deltas in the stream")
    first = calls[min(calls)]
    rebuilt = {
        "id": first["id"],
        "type": "function",
        "function": {"name": first["name"], "arguments": "".join(first["args"])},
    }
    found, notes, value = _check_call(rebuilt, r)
    problems += found
    if not ex.done:
        problems.append("the stream did not end with data: [DONE]")
    shape = "in one piece" if fragments <= 1 else f"in {fragments} fragments"
    notes.insert(0, f"arguments arrived {shape}")
    if problems:
        return _done(r, ex, FAIL, "; ".join(dict.fromkeys(problems)), *notes)
    return _done(r, ex, PASS, f"get_weather({json.dumps(value)}) rebuilt from deltas", *notes)


async def t_tools_result(ctx: Context, r: TestResult) -> TestResult:
    messages = [
        *WEATHER_PROMPT,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_conformance_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris", "unit": "celsius"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_conformance_1",
            "content": '{"temperature": 18, "unit": "celsius", "conditions": "light rain"}',
        },
    ]
    ex = await ctx.chat(messages=messages, tools=[WEATHER_TOOL])
    if not ex.ok:
        return _not_ok(r, ex)
    choice = first_choice(ex) or {}
    text = answer(choice)
    calls = (choice.get("message") or {}).get("tool_calls") if isinstance(choice, dict) else None
    if not text and calls:
        return _done(r, ex, WARN, "called a tool again instead of answering from the result")
    if not text:
        return _done(r, ex, FAIL, "empty reply after the tool result")
    if "18" in text:
        return _done(r, ex, PASS, f"answered from the tool result ({text[:50]!r})")
    return _done(r, ex, WARN, f"answered without the result's value ({text[:50]!r})")


async def t_tools_none(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(messages=WEATHER_PROMPT, tools=[WEATHER_TOOL], tool_choice="none")
    if not ex.ok:
        return _unsupported(r, ex, 'tool_choice "none"')
    message = (first_choice(ex) or {}).get("message") or {}
    if isinstance(message, dict) and message.get("tool_calls"):
        return _done(r, ex, FAIL, 'ignored: a tool was called with tool_choice "none"')
    return _done(r, ex, PASS, "answered in text")


def _judge_error(r: TestResult, ex: Exchange, expected: tuple[int, ...], what: str) -> TestResult:
    if ex.transport_error:
        return _not_ok(r, ex)
    if ex.status == 200:
        return _done(r, ex, FAIL, f"HTTP 200 for {what}")
    shape, message = error_shape(ex)
    if ex.status is not None and ex.status >= 500:
        return _done(
            r, ex, FAIL, f"HTTP {ex.status} for {what}: a server error for a client mistake"
        )
    if ex.status not in expected:
        return _done(
            r,
            ex,
            WARN,
            f"HTTP {ex.status} ({message}); expected {' or '.join(map(str, expected))}",
            f"error body: {shape}",
        )
    if shape != "openai":
        return _done(
            r,
            ex,
            WARN,
            f"HTTP {ex.status}, but the body is {shape}: {message}",
            'OpenAI clients read {"error": {"message", "type", "code"}}; other shapes lose detail',
        )
    return _done(r, ex, PASS, f"HTTP {ex.status} with an OpenAI error body ({message})")


async def t_errors_invalid_key(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(auth="invalid")
    return _judge_error(r, ex, (401,), "an invalid key")


async def t_errors_missing_key(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(auth="none")
    return _judge_error(r, ex, (401,), "a request without a key")


async def t_errors_unknown_model(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(model=UNKNOWN_MODEL)
    if ex.ok:
        served = ex.data.get("model") if isinstance(ex.data, dict) else None
        return _done(
            r,
            ex,
            FAIL,
            f"HTTP 200: an unknown model was served (reply says model {served!r})",
            "A typo in a model id should fail, not run on another model.",
        )
    return _judge_error(r, ex, (404, 400, 422), "an unknown model")


async def t_errors_invalid_request(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.chat(messages=[])
    return _judge_error(r, ex, (400, 422), "an empty messages list")


async def t_errors_malformed_json(ctx: Context, r: TestResult) -> TestResult:
    ex = await ctx.send("POST", "/chat/completions", raw=b'{"model": "x", "messages": [')
    return _judge_error(r, ex, (400, 422), "a malformed JSON body")


TESTS: tuple[TestSpec, ...] = (
    TestSpec("models.list", "discovery", "GET /models lists models", t_models_list),
    TestSpec("chat.basic", "chat", "A plain chat completion has the standard shape", t_chat_basic),
    TestSpec(
        "chat.usage", "chat", "usage has prompt, completion and total tokens", t_chat_usage, 0
    ),
    TestSpec("stream.basic", "streaming", "SSE chunks, finish_reason and [DONE]", t_stream_basic),
    TestSpec(
        "stream.usage", "streaming", "stream_options.include_usage sends usage", t_stream_usage
    ),
    TestSpec("params.max_tokens", "parameters", "max_tokens limits the reply", t_max_tokens),
    TestSpec(
        "params.max_completion_tokens",
        "parameters",
        "max_completion_tokens limits the reply",
        t_max_completion_tokens,
    ),
    TestSpec("params.stop", "parameters", "stop ends the reply at the sequence", t_stop),
    TestSpec("params.n", "parameters", "n=2 returns two choices", t_n),
    TestSpec(
        "params.response_format.json_object",
        "parameters",
        "JSON mode returns a bare JSON object",
        t_json_object,
    ),
    TestSpec(
        "params.response_format.json_schema",
        "parameters",
        "Structured output follows a JSON schema",
        t_json_schema,
    ),
    TestSpec("params.logprobs", "parameters", "logprobs are returned", t_logprobs),
    TestSpec(
        "params.temperature_range",
        "parameters",
        "An out-of-range temperature is rejected",
        t_temperature_range,
    ),
    TestSpec("tools.call", "tools", "A forced tool call has the standard shape", t_tools_call),
    TestSpec(
        "tools.stream", "tools", "Streamed tool-call deltas rebuild a valid call", t_tools_stream
    ),
    TestSpec("tools.result", "tools", "A tool result message is accepted and used", t_tools_result),
    TestSpec("tools.choice_none", "tools", 'tool_choice "none" prevents calls', t_tools_none),
    TestSpec(
        "errors.invalid_key",
        "errors",
        "An invalid key gets 401 with an error body",
        t_errors_invalid_key,
        needs_auth=True,
    ),
    TestSpec(
        "errors.missing_key",
        "errors",
        "A missing key gets 401 with an error body",
        t_errors_missing_key,
        needs_auth=True,
    ),
    TestSpec(
        "errors.unknown_model",
        "errors",
        "An unknown model is an error, not a substitute",
        t_errors_unknown_model,
    ),
    TestSpec(
        "errors.invalid_request",
        "errors",
        "An empty messages list gets 400 or 422",
        t_errors_invalid_request,
    ),
    TestSpec(
        "errors.malformed_json",
        "errors",
        "A malformed JSON body gets 400, not a 5xx",
        t_errors_malformed_json,
    ),
)
CATEGORIES = tuple(dict.fromkeys(t.category for t in TESTS))


def select_tests(only: list[str] | None, skip: list[str] | None) -> list[TestSpec]:
    """Tests whose id or category matches ``only`` (all when empty), minus ``skip``."""

    def match(t: TestSpec, names: list[str]) -> bool:
        return any(t.id == n or t.category == n or t.id.startswith(n + ".") for n in names)

    known = {t.id for t in TESTS} | set(CATEGORIES)
    unknown = [
        n
        for n in [*(only or []), *(skip or [])]
        if n not in known and not any(t.id.startswith(n + ".") for t in TESTS)
    ]
    if unknown:
        raise ValueError(
            f"unknown test or category: {', '.join(unknown)} "
            f"(categories: {', '.join(CATEGORIES)}; `llm-bench conformance --list` shows tests)"
        )
    chosen = [t for t in TESTS if not only or match(t, only)]
    return [t for t in chosen if not skip or not match(t, skip)]


def planned_requests(provider: ProviderConfig, tests: list[TestSpec]) -> int:
    return sum(t.requests for t in tests if _has_auth(provider) or not t.needs_auth)


async def run_provider(
    provider: ProviderConfig,
    tests: list[TestSpec],
    *,
    timeout_s: float,
    connect_timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    progress: Callable[[TestResult], None] | None = None,
) -> ProviderReport:
    if transport is None and provider.is_synthetic:
        from .providers.reference import reference_transport

        transport = reference_transport(provider.synthetic or SyntheticProfile(), provider.model)
    started = datetime.now(UTC)
    results: list[TestResult] = []
    stopped: str | None = None
    async with httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
        follow_redirects=False,
        headers={"User-Agent": f"llm-api-benchmark/{__version__} (conformance)"},
        trust_env=True,
    ) as client:
        ctx = Context(client, provider, timeout_s=timeout_s, sleep=sleep)
        for spec in tests:
            r = TestResult(spec.id, spec.category, spec.title)
            if stopped:
                r.outcome, r.summary = ERROR, f"not run: {stopped}"
            elif spec.needs_auth and not ctx.has_auth:
                r.outcome, r.summary = SKIP, "the endpoint has no authentication configured"
            else:
                before = ctx.requests
                try:
                    r = await spec.fn(ctx, r)
                except StopSuite as stop:
                    stopped = str(stop)
                    r.outcome, r.summary = ERROR, stopped
                r.requests = ctx.requests - before
            results.append(r)
            if progress:
                progress(r)
        headers_seen = sorted(ctx.headers_seen)
    return ProviderReport(
        provider=provider.to_public_dict(),
        results=results,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        headers_seen=headers_seen,
        stopped=stopped,
    )
