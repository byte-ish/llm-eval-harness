"""Tests for the core data contracts in `harness.models`."""

import dataclasses
import math

import pytest

from harness.models import (
    EvalCase,
    EvalResult,
    EvalSuite,
    RunReport,
    ScorerResult,
)


def _make_result(
    *,
    category: str = "earnings_summary",
    passed: bool = True,
    score: float = 1.0,
    latency_ms: float = 100.0,
    cost_usd: float = 0.001,
    case_id: str = "case-1",
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        category=category,
        response="...",
        scorer_result=ScorerResult(passed=passed, score=score, reason=""),
        latency_ms=latency_ms,
        input_tokens=10,
        output_tokens=20,
        cost_usd=cost_usd,
        model="claude-opus-4-7",
        temperature=0.0,
    )


def _make_report(results: list[EvalResult]) -> RunReport:
    return RunReport(
        run_id="20260608-120000",
        model="claude-opus-4-7",
        suite="summarisation",
        dataset_version="1.0.0",
        results=results,
        started_at="2026-06-08T12:00:00Z",
        finished_at="2026-06-08T12:01:00Z",
        harness_version="0.1.0",
    )


class TestEvalSuiteImmutability:
    def test_cases_is_a_tuple(self) -> None:
        suite = EvalSuite(
            name="x",
            dataset_version="1.0.0",
            default_system=None,
            default_user=None,
            cases=(EvalCase(id="c1", category="cat", scorer="exact_match"),),
        )
        assert isinstance(suite.cases, tuple)

    def test_suite_is_frozen(self) -> None:
        suite = EvalSuite(
            name="x",
            dataset_version="1.0.0",
            default_system=None,
            default_user=None,
            cases=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            suite.name = "y"  # type: ignore[misc]

    def test_case_is_frozen(self) -> None:
        case = EvalCase(id="c1", category="cat", scorer="exact_match")
        with pytest.raises(dataclasses.FrozenInstanceError):
            case.id = "c2"  # type: ignore[misc]


class TestPassRate:
    def test_all_pass(self) -> None:
        report = _make_report([_make_result(passed=True) for _ in range(5)])
        assert report.pass_rate == 1.0

    def test_all_fail(self) -> None:
        report = _make_report([_make_result(passed=False) for _ in range(5)])
        assert report.pass_rate == 0.0

    def test_mixed(self) -> None:
        results = [_make_result(passed=True) for _ in range(3)] + [
            _make_result(passed=False) for _ in range(2)
        ]
        report = _make_report(results)
        assert report.pass_rate == 0.6

    def test_empty_returns_zero(self) -> None:
        report = _make_report([])
        assert report.pass_rate == 0.0


class TestPassRateByCategory:
    def test_groups_by_category(self) -> None:
        results = [
            _make_result(category="a", passed=True),
            _make_result(category="a", passed=False),
            _make_result(category="b", passed=True),
            _make_result(category="b", passed=True),
        ]
        report = _make_report(results)
        assert report.pass_rate_by_category == {"a": 0.5, "b": 1.0}

    def test_empty_returns_empty_dict(self) -> None:
        report = _make_report([])
        assert report.pass_rate_by_category == {}

    def test_single_category(self) -> None:
        results = [_make_result(category="only", passed=True) for _ in range(3)]
        report = _make_report(results)
        assert report.pass_rate_by_category == {"only": 1.0}


class TestTotalCost:
    def test_sums_costs(self) -> None:
        results = [_make_result(cost_usd=0.001) for _ in range(3)]
        report = _make_report(results)
        assert math.isclose(report.total_cost_usd, 0.003)

    def test_empty_returns_zero(self) -> None:
        report = _make_report([])
        assert report.total_cost_usd == 0.0


class TestLatencyPercentiles:
    def test_single_value(self) -> None:
        report = _make_report([_make_result(latency_ms=500.0)])
        assert report.p50_latency_ms == 500.0
        assert report.p95_latency_ms == 500.0

    def test_p50_median_of_odd(self) -> None:
        results = [_make_result(latency_ms=v) for v in (100.0, 200.0, 300.0)]
        report = _make_report(results)
        assert report.p50_latency_ms == 200.0

    def test_p95_linear_interpolation_over_100_values(self) -> None:
        # latencies 1..100; linear interp at p=95 gives 1 + 0.95 * 99 = 95.05
        results = [_make_result(latency_ms=float(v)) for v in range(1, 101)]
        report = _make_report(results)
        assert math.isclose(report.p95_latency_ms, 95.05)

    def test_empty_returns_zero(self) -> None:
        report = _make_report([])
        assert report.p50_latency_ms == 0.0
        assert report.p95_latency_ms == 0.0
