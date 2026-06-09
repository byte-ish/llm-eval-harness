"""Tests for `harness.budget` — cost estimate + cost_fn factory."""

import pytest

from harness.budget import estimate_cost, make_cost_fn
from harness.config import PriceEntry
from harness.models import EvalCase, EvalSuite


def _suite(n: int = 1) -> EvalSuite:
    cases = tuple(
        EvalCase(
            id=f"c{i}",
            category="cat",
            scorer="exact_match",
            user="u",
            system="s",
            expected_contains=["x"],
        )
        for i in range(n)
    )
    return EvalSuite(
        name="s",
        dataset_version="1.0.0",
        default_system="s",
        default_user="u",
        cases=cases,
    )


class TestEstimateCost:
    def test_positive_for_priced_model(self) -> None:
        prices = {"m1": PriceEntry(input_per_million=10.0, output_per_million=20.0)}
        est = estimate_cost(_suite(5), prices, "m1")
        assert est.estimated_usd > 0
        assert est.model_id == "m1"
        assert est.case_count == 5

    def test_scales_linearly_with_case_count(self) -> None:
        prices = {"m1": PriceEntry(input_per_million=10.0, output_per_million=20.0)}
        one = estimate_cost(_suite(1), prices, "m1").estimated_usd
        ten = estimate_cost(_suite(10), prices, "m1").estimated_usd
        assert ten == pytest.approx(one * 10)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="price map"):
            estimate_cost(_suite(1), {}, "unknown-model")

    def test_default_token_assumptions(self) -> None:
        # 1 case @ 1000 input * $10/M + 500 output * $20/M = 0.01 + 0.01 = 0.02
        prices = {"m1": PriceEntry(input_per_million=10.0, output_per_million=20.0)}
        est = estimate_cost(_suite(1), prices, "m1")
        assert est.estimated_usd == pytest.approx(0.02)


class TestMakeCostFn:
    def test_cost_calculation(self) -> None:
        price = PriceEntry(input_per_million=15.0, output_per_million=75.0)
        cost_fn = make_cost_fn(price)
        # 1000 input * $15/M + 500 output * $75/M = 0.015 + 0.0375 = 0.0525
        assert cost_fn(1000, 500) == pytest.approx(0.0525)

    def test_zero_tokens_costs_zero(self) -> None:
        price = PriceEntry(input_per_million=15.0, output_per_million=75.0)
        cost_fn = make_cost_fn(price)
        assert cost_fn(0, 0) == 0.0
