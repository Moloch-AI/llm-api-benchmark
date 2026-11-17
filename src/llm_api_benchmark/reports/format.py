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


def errors(s: GroupSummary) -> str:
    if not s.errors:
        return ""
    return ", ".join(f"{k} x{v}" for k, v in s.errors.items())


LEGEND = (
    "Times in seconds. TTFT: first streamed text (reasoning or answer). TTFC: first answer text. "
    "Latency: request sent to last byte. Out tok/s: completion tokens / latency. "
    "Decode tok/s: tokens after the first chunk / time after the first chunk. "
    "n: successful samples. * fewer than 20 samples (tail percentiles unstable). "
    "~ token counts estimated from text length."
)
