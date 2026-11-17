from __future__ import annotations

from datetime import date

import pytest

from llm_api_benchmark.config import Pricing
from llm_api_benchmark.pricing import request_cost, staleness_warning, worst_case_cost

P = Pricing(
    input_per_mtok=0.5,
    output_per_mtok=2.0,
    currency="USD",
    source="https://example.com/pricing",
    checked_on=date(2026, 11, 1),
)


def test_request_cost() -> None:
    assert request_cost(P, 1_000_000, 0) == pytest.approx(0.5)
    assert request_cost(P, 1000, 500) == pytest.approx((1000 * 0.5 + 500 * 2.0) / 1e6)
    assert request_cost(P, None, 10) is None
    assert request_cost(P, 10, None) is None


def test_worst_case_cost() -> None:
    assert worst_case_cost(P, 100, 1000) == pytest.approx((100 * 0.5 + 1000 * 2.0) / 1e6)


def test_staleness() -> None:
    assert staleness_warning(P, "x", today=date(2026, 11, 17)) is None
    old = staleness_warning(P, "x", today=date(2027, 6, 1))
    assert old is not None
    assert "days ago" in old
    undated = Pricing(1, 1, "USD", "src", None)
    assert "no checked_on" in (staleness_warning(undated, "x") or "")
