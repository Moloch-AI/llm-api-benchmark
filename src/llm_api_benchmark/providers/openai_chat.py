"""OpenAI Chat Completions wire format (``POST {base_url}/chat/completions``)."""

from __future__ import annotations

import json
import re
from typing import Any

from ..redact import redact
from ..tools import ToolCall, ToolCallDelta
from .base import Adapter, ChatRequest, ChunkInfo, CompletionInfo, ErrorInfo, InvalidResponse, Usage

_TITLE = re.compile(r"(?is)<title>(.*?)</title>")
_MAX_MESSAGE = 300


class OpenAIChatAdapter(Adapter):
    def build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        max_tokens: int | None,
        temperature: float | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> ChatRequest:
        p = self.provider
        body: dict[str, Any] = {"model": p.model, "messages": messages}
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if stream:
            body["stream"] = True
            if p.stream_usage:
                body["stream_options"] = {"include_usage": True}
        if max_tokens is not None and p.max_tokens_param != "none":
            body[p.max_tokens_param] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        # Provider params win; a null value removes a field the benchmark would otherwise send.
        for key, value in p.params.items():
            if value is None:
                body.pop(key, None)
            else:
                body[key] = value

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        headers.update(p.headers)
        headers.update(self._extra_headers)
        if self._api_key and p.auth_scheme == "bearer":
            headers[p.auth_header] = f"Bearer {self._api_key}"
        elif self._api_key and p.auth_scheme == "header":
            headers[p.auth_header] = self._api_key
        return ChatRequest(
            url=f"{p.base_url}/chat/completions", headers=headers, query=dict(p.query), body=body
        )

    def parse_chunk(self, data: Any) -> ChunkInfo:
        if not isinstance(data, dict):
            raise InvalidResponse("a stream event is not a JSON object")
        if data.get("error"):
            return ChunkInfo(error=_error_from_obj(data["error"]))
        usage = _usage(data.get("usage"))
        model = data.get("model") if isinstance(data.get("model"), str) else None
        choices = data.get("choices")
        if choices is None and usage is None:
            raise InvalidResponse("a stream event has neither choices nor usage")
        if not choices:
            return ChunkInfo(usage=usage, model=model)
        if not isinstance(choices, list) or not isinstance(choices[0], dict):
            raise InvalidResponse("choices is not a list of objects")
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise InvalidResponse("delta is not an object")
        return ChunkInfo(
            content=_text(delta.get("content")),
            # reasoning_content (vLLM, SGLang, MolochAI, DeepSeek) or reasoning (OpenRouter, Groq)
            reasoning=_text(delta.get("reasoning_content")) or _text(delta.get("reasoning")),
            finish_reason=_finish(choice.get("finish_reason")),
            usage=usage,
            model=model,
            tool_calls=_tool_deltas(delta),
        )

    def parse_completion(self, data: Any) -> CompletionInfo:
        if not isinstance(data, dict):
            raise InvalidResponse("the response is not a JSON object")
        if "error" in data and data["error"] and not data.get("choices"):
            err = _error_from_obj(data["error"])
            raise InvalidResponse(
                f"status 200 with an error body: {err.type or ''} {err.message}".strip()
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InvalidResponse("the response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise InvalidResponse("choices[0].message is missing")
        return CompletionInfo(
            content=_text(message.get("content")),
            reasoning=_text(message.get("reasoning_content")) or _text(message.get("reasoning")),
            finish_reason=_finish(choices[0].get("finish_reason")),
            usage=_usage(data.get("usage")),
            model=data.get("model") if isinstance(data.get("model"), str) else None,
            tool_calls=_tool_calls(message),
        )

    def parse_error(self, status: int, body: bytes, content_type: str) -> ErrorInfo:
        return parse_error_body(body, content_type)


def parse_error_body(body: bytes, content_type: str = "") -> ErrorInfo:
    """Read the common error shapes without assuming JSON.

    {"error": {"type"|"code", "message"}}, {"message", "errors"}, {"message", "code"},
    a bare string, or HTML from a proxy (Cloudflare 524 and similar).
    """
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return ErrorInfo(None, "(empty body)")
    try:
        data = json.loads(text)
    except ValueError:
        if "<html" in text[:500].lower() or "html" in content_type:
            title = _TITLE.search(text)
            label = " ".join(title.group(1).split()) if title else "no title"
            return ErrorInfo(None, _clip(f"(HTML) {label}"))
        return ErrorInfo(None, _clip(" ".join(text.split())))
    if isinstance(data, dict):
        if isinstance(data.get("error"), dict | str):
            return _error_from_obj(data["error"])
        code = data.get("code") or data.get("type")
        message = data.get("message") or data.get("detail") or json.dumps(data)[:_MAX_MESSAGE]
        return ErrorInfo(str(code) if code else None, _clip(str(message)))
    return ErrorInfo(None, _clip(str(data)))


def _error_from_obj(err: Any) -> ErrorInfo:
    if isinstance(err, str):
        return ErrorInfo(None, _clip(err))
    if not isinstance(err, dict):
        return ErrorInfo(None, _clip(str(err)))
    kind = err.get("type") or err.get("code")
    return ErrorInfo(str(kind) if kind else None, _clip(str(err.get("message") or "")))


def _clip(text: str) -> str:
    text = redact(text)
    return text if len(text) <= _MAX_MESSAGE else text[: _MAX_MESSAGE - 1] + "…"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # content parts
        return "".join(
            p.get("text", "")
            for p in value
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _tool_deltas(delta: dict[str, Any]) -> tuple[ToolCallDelta, ...]:
    raw = delta.get("tool_calls")
    if raw is None and isinstance(delta.get("function_call"), dict):  # legacy single call
        raw = [{"index": 0, "function": delta["function_call"]}]
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise InvalidResponse("delta.tool_calls is not a list")
    out = []
    for pos, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InvalidResponse("delta.tool_calls has an entry that is not an object")
        fn = item.get("function") or {}
        if not isinstance(fn, dict):
            raise InvalidResponse("delta.tool_calls[].function is not an object")
        index = item.get("index")
        args = fn.get("arguments")
        out.append(
            ToolCallDelta(
                index=index if isinstance(index, int) and not isinstance(index, bool) else pos,
                id=item.get("id") if isinstance(item.get("id"), str) else None,
                name=fn.get("name") if isinstance(fn.get("name"), str) else None,
                arguments=args if isinstance(args, str) else (json.dumps(args) if args else ""),
            )
        )
    return tuple(out)


def _tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    raw = message.get("tool_calls")
    if raw is None and isinstance(message.get("function_call"), dict):  # legacy single call
        raw = [{"function": message["function_call"]}]
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise InvalidResponse("message.tool_calls is not a list")
    out = []
    for item in raw:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise InvalidResponse("message.tool_calls has an entry without function.name")
        args = fn.get("arguments")
        out.append(
            ToolCall(
                name=fn["name"],
                # The spec says a JSON string; some servers send an object. Keep it comparable.
                arguments=args if isinstance(args, str) else (json.dumps(args) if args else ""),
                id=item.get("id") if isinstance(item.get("id"), str) else None,
            )
        )
    return tuple(out)


def _finish(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    details = value.get("completion_tokens_details")
    reasoning = _int(details.get("reasoning_tokens")) if isinstance(details, dict) else None
    usage = Usage(
        prompt_tokens=_int(value.get("prompt_tokens")),
        completion_tokens=_int(value.get("completion_tokens")),
        total_tokens=_int(value.get("total_tokens")),
        reasoning_tokens=reasoning,
    )
    if usage.prompt_tokens is None and usage.completion_tokens is None:
        return None
    return usage
