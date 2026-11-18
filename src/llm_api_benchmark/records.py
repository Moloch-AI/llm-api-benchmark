"""The per-request measurement record."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

# Outcome of one request. Only "ok" counts as a success; "skipped" and "cancelled" requests
# were never completed and are excluded from error rates.
OK = "ok"
HTTP_ERROR = "http_error"
TIMEOUT = "timeout"
CONNECTION_ERROR = "connection_error"
INVALID_RESPONSE = "invalid_response"
STREAM_ERROR = "stream_error"
CANCELLED = "cancelled"
SKIPPED = "skipped"
STATUSES = (
    OK,
    HTTP_ERROR,
    TIMEOUT,
    CONNECTION_ERROR,
    INVALID_RESPONSE,
    STREAM_ERROR,
    CANCELLED,
    SKIPPED,
)
NOT_EXECUTED = (CANCELLED, SKIPPED)


@dataclass
class RequestRecord:
    index: int
    provider: str
    model: str
    workload: str
    prompt_id: str
    mode: str
    repetition: int
    warmup: bool = False
    started_at: str | None = None
    status: str = SKIPPED
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    # Seconds from the moment the request is sent.
    headers_s: float | None = None
    ttfb_s: float | None = None
    ttft_s: float | None = None
    ttfc_s: float | None = None
    last_token_s: float | None = None
    latency_s: float | None = None
    text_chunks: int = 0
    comment_lines: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage_reported: bool = False
    tokens_estimated: bool = False
    output_chars: int = 0
    reasoning_chars: int = 0
    finish_reason: str | None = None
    response_model: str | None = None
    done_marker: bool | None = None
    retry_after_s: float | None = None
    should_retry: str | None = None
    cost: float | None = None
    # Tool workloads only: the names called ([] = answered without a call) and a fingerprint
    # of names plus parsed arguments, used to measure agreement across repetitions.
    tool_call_names: list[str] | None = None
    tool_signature: str | None = None
    checks: list[dict[str, Any]] | None = None
    checks_passed: bool | None = None
    output: str | None = None
    tool_calls: list[dict[str, str]] | None = None  # with save_outputs only
    skip_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def executed(self) -> bool:
        return self.status not in NOT_EXECUTED

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def error_kind(self) -> str | None:
        """A stable label for grouping failures: http_429, timeout, stream_error:preempted..."""
        if self.status in (OK, *NOT_EXECUTED):
            return None
        if self.status == HTTP_ERROR:
            return f"http_{self.http_status}"
        if self.status == STREAM_ERROR and self.error_type:
            return f"stream_error:{self.error_type}"
        return self.status

    @property
    def output_tps(self) -> float | None:
        """Output tokens per second over the whole request (includes time to first token)."""
        if not self.ok or not self.completion_tokens or not self.latency_s:
            return None
        return self.completion_tokens / self.latency_s

    @property
    def decode_tps(self) -> float | None:
        """Tokens per second after the first output chunk (streaming only).

        (completion_tokens - 1) / (last text chunk - first text chunk). Chunks can carry several
        tokens, so this approximates the generation rate rather than timing each token.
        """
        if not self.ok or self.mode != "stream" or not self.completion_tokens:
            return None
        if self.ttft_s is None or self.last_token_s is None or self.completion_tokens < 2:
            return None
        window = self.last_token_s - self.ttft_s
        if window <= 0:
            return None
        return (self.completion_tokens - 1) / window

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["error_kind"] = self.error_kind
        out["output_tps"] = _round(self.output_tps)
        out["decode_tps"] = _round(self.decode_tps)
        if not out["extra"]:
            out.pop("extra")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRecord:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _round(v: float | None, digits: int = 3) -> float | None:
    return None if v is None else round(v, digits)


CSV_COLUMNS = [
    "index",
    "provider",
    "model",
    "response_model",
    "workload",
    "prompt_id",
    "mode",
    "repetition",
    "warmup",
    "started_at",
    "status",
    "error_kind",
    "http_status",
    "error_type",
    "error_message",
    "headers_s",
    "ttfb_s",
    "ttft_s",
    "ttfc_s",
    "latency_s",
    "output_tps",
    "decode_tps",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "usage_reported",
    "tokens_estimated",
    "output_chars",
    "reasoning_chars",
    "text_chunks",
    "comment_lines",
    "finish_reason",
    "done_marker",
    "tool_call_names",
    "tool_signature",
    "cost",
    "checks_passed",
    "skip_reason",
]
