"""Cost from user-supplied prices. The tool ships no prices of its own."""

from __future__ import annotations

from datetime import date

from .config import Pricing

STALE_AFTER_DAYS = 90


def request_cost(
    pricing: Pricing, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """Cost of one request, or None when token counts are unknown."""
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (
        prompt_tokens * pricing.input_per_mtok + completion_tokens * pricing.output_per_mtok
    ) / 1_000_000


def worst_case_cost(pricing: Pricing, prompt_tokens: int, max_output_tokens: int) -> float:
    return (
        prompt_tokens * pricing.input_per_mtok + max_output_tokens * pricing.output_per_mtok
    ) / 1_000_000


def staleness_warning(pricing: Pricing, provider: str, today: date | None = None) -> str | None:
    today = today or date.today()
    if pricing.checked_on is None:
        return f"{provider}: pricing has no checked_on date; confirm it against {pricing.source}"
    age = (today - pricing.checked_on).days
    if age > STALE_AFTER_DAYS:
        return (
            f"{provider}: pricing was checked {age} days ago; confirm it against {pricing.source}"
        )
    return None
