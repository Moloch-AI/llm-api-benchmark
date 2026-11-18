"""Shared formatting for reports."""

from __future__ import annotations

from ..metrics import MIN_SAMPLES_FOR_TAIL, GroupSummary, Stat

DASH = "-"


def secs(v: float | None) -> str:
    if v is None:
        return DASH
    if v < 1:
        return f"{v:.3f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:.0f}"


def rate(v: float | None) -> str:
    return DASH if v is None else f"{v:.1f}"


def pct(v: float | None) -> str:
    return DASH if v is None else f"{v * 100:.1f}%"


def money(v: float | None, currency: str = "USD") -> str:
    if v is None:
        return DASH
    return f"{v:.4f} {currency}" if v < 1 else f"{v:.2f} {currency}"


def stat(s: Stat | None, attr: str, fmt: str = "secs") -> str:
    if s is None:
        return DASH
    value = getattr(s, attr)
    text = secs(value) if fmt == "secs" else rate(value)
    if attr in ("p90", "p95") and s.n < MIN_SAMPLES_FOR_TAIL:
        text += "*"
    return text


def tps(s: GroupSummary, attr: str = "median", which: str = "output") -> str:
    st = s.output_tps if which == "output" else s.decode_tps
    if st is None:
        return DASH
    text = rate(getattr(st, attr))
    return ("~" + text) if s.tokens_estimated else text


def checks(s: GroupSummary) -> str:
    if s.checks_run == 0:
        return DASH
    return f"{s.checks_passed}/{s.checks_run}"


def check_type(s: GroupSummary, kind: str) -> str:
    passed, run = s.check_types.get(kind, (0, 0))
    return DASH if run == 0 else f"{passed}/{run}"


def agreement(s: GroupSummary) -> str:
    if s.tool_agreement is None:
        return DASH
    return f"{s.tool_agreement * 100:.0f}% ({s.tool_agreement_prompts} prompts)"


def has_detail(s: GroupSummary) -> bool:
    """Whether a group has per-check or tool-call figures worth a second table."""
    return s.tool_call_rate is not None or len(s.check_types) > 1


TOOL_LEGEND = (
    "Called a tool: share of successful replies with at least one tool call. Same calls: for "
    "prompts run at least twice, the share of replies that match the most common set of calls "
    "(same functions, same parsed arguments), averaged over prompts. Check columns: replies that "
    "passed that check / replies it ran on."
)


def errors(s: GroupSummary) -> str:
    if not s.errors:
        return ""
    return ", ".join(f"{k} x{v}" for k, v in s.errors.items())


LEGEND = (
    "Times in seconds. TTFT: first streamed output (reasoning, answer or tool call). "
    "TTFC: first answer output (text or tool call). "
    "Latency: request sent to last byte. Out tok/s: completion tokens / latency. "
    "Decode tok/s: tokens after the first chunk / time after the first chunk. "
    "n: successful samples. * fewer than 20 samples (tail percentiles unstable). "
    "~ token counts estimated from text length."
)
