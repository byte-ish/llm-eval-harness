"""Tests for `harness.compare.compare_reports`."""

import pytest

from harness.compare import ComparisonError, compare_reports
from harness.models import EvalResult, RunReport, ScorerResult


def _result(case_id: str, category: str, passed: bool, cost: float = 0.001) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        category=category,
        response="text",
        scorer_result=ScorerResult(passed=passed, score=1.0 if passed else 0.0, reason=""),
        latency_ms=100.0,
        input_tokens=50,
        output_tokens=50,
        cost_usd=cost,
        model="resolved-model",
        temperature=0.0,
    )


def _report(model: str, results: list[EvalResult], dataset_version: str = "1.0.0") -> RunReport:
    return RunReport(
        run_id=f"run-{model}",
        model=model,
        suite="demo",
        dataset_version=dataset_version,
        results=results,
        started_at="2026-06-09T10:00:00+00:00",
        finished_at="2026-06-09T10:01:00+00:00",
        harness_version="0.1.0",
    )


def test_compare_two_reports_overall_delta() -> None:
    ref = _report(
        "claude-haiku",
        [_result("c1", "cat_a", True), _result("c2", "cat_a", False)],
    )
    cand = _report(
        "gpt-4o-mini",
        [_result("c1", "cat_a", True), _result("c2", "cat_a", True)],
    )
    comparison = compare_reports([ref, cand], ["anthropic:claude-haiku", "openai:gpt-4o-mini"])
    assert comparison.reference_label == "anthropic:claude-haiku"
    assert comparison.overall_pass_rate_delta(1) == pytest.approx(0.5)
    assert comparison.pass_rate_delta_by_category(1) == {"cat_a": 0.5}


def test_compare_cost_delta_signs() -> None:
    ref = _report("a", [_result("c1", "x", True, cost=0.01)])
    cand = _report("b", [_result("c1", "x", True, cost=0.05)])
    comparison = compare_reports([ref, cand], ["a", "b"])
    assert comparison.cost_delta_usd(1) == pytest.approx(0.04)


def test_compare_case_outcomes_preserve_reference_order() -> None:
    ref = _report(
        "ref",
        [_result("c1", "cat", True), _result("c2", "cat", False), _result("c3", "cat", True)],
    )
    cand = _report(
        "cand",
        [_result("c3", "cat", True), _result("c2", "cat", True), _result("c1", "cat", False)],
    )
    comparison = compare_reports([ref, cand], ["ref", "cand"])
    assert [c.case_id for c in comparison.case_outcomes] == ["c1", "c2", "c3"]
    assert comparison.case_outcomes[0].passed_by_model == [True, False]  # c1
    assert comparison.case_outcomes[1].passed_by_model == [False, True]  # c2


def test_compare_rejects_dataset_version_mismatch() -> None:
    ref = _report("ref", [_result("c1", "x", True)], dataset_version="1.0.0")
    cand = _report("cand", [_result("c1", "x", True)], dataset_version="1.1.0")
    with pytest.raises(ComparisonError, match="dataset_version mismatch"):
        compare_reports([ref, cand], ["ref", "cand"])


def test_compare_allows_dataset_mismatch_when_flagged() -> None:
    ref = _report("ref", [_result("c1", "x", True)], dataset_version="1.0.0")
    cand = _report("cand", [_result("c1", "x", True)], dataset_version="1.1.0")
    comparison = compare_reports([ref, cand], ["ref", "cand"], allow_dataset_mismatch=True)
    assert comparison.dataset_version == "1.0.0"  # picks reference's version


def test_compare_rejects_mixed_suites() -> None:
    ref = _report("ref", [_result("c1", "x", True)])
    object.__setattr__(ref, "suite", "demo")
    cand = _report("cand", [_result("c1", "x", True)])
    object.__setattr__(cand, "suite", "other")
    with pytest.raises(ComparisonError, match="same suite"):
        compare_reports([ref, cand], ["ref", "cand"])


def test_compare_needs_two_reports() -> None:
    ref = _report("ref", [_result("c1", "x", True)])
    with pytest.raises(ComparisonError, match="at least two"):
        compare_reports([ref], ["ref"])


def test_compare_rejects_length_mismatch() -> None:
    ref = _report("ref", [_result("c1", "x", True)])
    cand = _report("cand", [_result("c1", "x", True)])
    with pytest.raises(ComparisonError, match="length mismatch"):
        compare_reports([ref, cand], ["only-one"])


def test_compare_handles_three_models() -> None:
    ref = _report("a", [_result("c1", "cat", True), _result("c2", "cat", True)])
    b = _report("b", [_result("c1", "cat", True), _result("c2", "cat", False)])
    c = _report("c", [_result("c1", "cat", False), _result("c2", "cat", False)])
    comparison = compare_reports([ref, b, c], ["a", "b", "c"])
    assert comparison.overall_pass_rate_delta(1) == pytest.approx(-0.5)
    assert comparison.overall_pass_rate_delta(2) == pytest.approx(-1.0)
    assert comparison.case_outcomes[0].passed_by_model == [True, True, False]
