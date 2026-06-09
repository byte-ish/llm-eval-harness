"""Tests for `harness.regression` — baseline comparison + dataset-version guard."""

import pytest

from harness.models import EvalResult, RunReport, ScorerResult
from harness.regression import (
    DEFAULT_DROP_THRESHOLD,
    DatasetVersionMismatch,
    compare_to_baseline,
)


def _result(case_id: str, category: str, passed: bool) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        category=category,
        response="r",
        scorer_result=ScorerResult(passed=passed, score=1.0 if passed else 0.0, reason=""),
        latency_ms=10.0,
        input_tokens=10,
        output_tokens=10,
        cost_usd=0.0,
        model="m",
        temperature=0.0,
    )


def _report(results: list[EvalResult], *, dataset_version: str = "1.0.0") -> RunReport:
    return RunReport(
        run_id="r",
        model="m",
        suite="s",
        dataset_version=dataset_version,
        results=results,
        started_at="t",
        finished_at="t",
        harness_version="v",
    )


class TestImprovement:
    def test_improvement_no_regression(self) -> None:
        baseline = _report([_result("c1", "a", False), _result("c2", "a", True)])
        current = _report([_result("c1", "a", True), _result("c2", "a", True)])
        diff = compare_to_baseline(current, baseline)
        assert diff.has_regression is False
        assert diff.overall_pass_rate_delta == 0.5
        assert diff.per_category_delta["a"] == 0.5


class TestRegression:
    def test_drop_above_threshold_flags(self) -> None:
        # baseline 100%, current 0% in "a" → drop of 1.0
        baseline = _report([_result("c1", "a", True), _result("c2", "a", True)])
        current = _report([_result("c1", "a", False), _result("c2", "a", False)])
        diff = compare_to_baseline(current, baseline)
        assert diff.has_regression is True
        assert "a" in diff.flagged_categories
        assert diff.per_category_delta["a"] == pytest.approx(-1.0)

    def test_newly_failing_case_ids_listed(self) -> None:
        baseline = _report([_result("c1", "a", True), _result("c2", "a", True)])
        current = _report([_result("c1", "a", False), _result("c2", "a", True)])
        diff = compare_to_baseline(current, baseline)
        # c1 passed in baseline, fails in current → newly failing
        assert "c1" in diff.newly_failing_case_ids
        assert "c2" not in diff.newly_failing_case_ids


class TestThresholdBoundary:
    def test_drop_exactly_at_threshold_does_not_flag(self) -> None:
        # 4 cases per category; baseline 4/4 pass; current 3/4 pass → drop = 0.25
        # Use 0.25 threshold so drop == threshold (not >)
        baseline = _report([_result(f"c{i}", "a", True) for i in range(4)])
        current = _report(
            [_result("c0", "a", False)] + [_result(f"c{i}", "a", True) for i in range(1, 4)]
        )
        diff = compare_to_baseline(current, baseline, drop_threshold=0.25)
        assert diff.has_regression is False, "drop == threshold should not flag"

    def test_drop_just_above_threshold_flags(self) -> None:
        # drop = 0.25 → drop_threshold=0.20 should flag (0.25 > 0.20)
        baseline = _report([_result(f"c{i}", "a", True) for i in range(4)])
        current = _report(
            [_result("c0", "a", False)] + [_result(f"c{i}", "a", True) for i in range(1, 4)]
        )
        diff = compare_to_baseline(current, baseline, drop_threshold=0.20)
        assert diff.has_regression is True

    def test_default_threshold_is_0_05(self) -> None:
        assert DEFAULT_DROP_THRESHOLD == 0.05


class TestNewCategory:
    def test_new_category_in_current_listed(self) -> None:
        baseline = _report([_result("c1", "a", True)])
        current = _report([_result("c1", "a", True), _result("c2", "b", True)])
        diff = compare_to_baseline(current, baseline)
        assert "b" in diff.new_categories
        assert "a" not in diff.new_categories

    def test_removed_category_in_current_listed(self) -> None:
        baseline = _report([_result("c1", "a", True), _result("c2", "b", True)])
        current = _report([_result("c1", "a", True)])
        diff = compare_to_baseline(current, baseline)
        assert "b" in diff.removed_categories

    def test_new_category_does_not_count_as_regression(self) -> None:
        baseline = _report([_result("c1", "a", True)])
        current = _report([_result("c1", "a", True), _result("c2", "b", False)])
        diff = compare_to_baseline(current, baseline)
        assert diff.has_regression is False


class TestDatasetVersionMismatch:
    def test_mismatch_raises_by_default(self) -> None:
        baseline = _report([_result("c1", "a", True)], dataset_version="1.0.0")
        current = _report([_result("c1", "a", True)], dataset_version="1.1.0")
        with pytest.raises(DatasetVersionMismatch, match="dataset_version mismatch"):
            compare_to_baseline(current, baseline)

    def test_mismatch_bypassed_with_allow_flag(self) -> None:
        baseline = _report([_result("c1", "a", True)], dataset_version="1.0.0")
        current = _report([_result("c1", "a", False)], dataset_version="1.1.0")
        diff = compare_to_baseline(current, baseline, allow_dataset_mismatch=True)
        assert diff.has_regression is True  # regression still flagged in shared category

    def test_matching_versions_succeed(self) -> None:
        baseline = _report([_result("c1", "a", True)], dataset_version="2.0.0")
        current = _report([_result("c1", "a", True)], dataset_version="2.0.0")
        diff = compare_to_baseline(current, baseline)
        assert diff.has_regression is False


class TestOverallDelta:
    def test_overall_delta_computed(self) -> None:
        # baseline: 2/4 pass → 0.5. current: 4/4 → 1.0. delta = +0.5
        baseline = _report(
            [
                _result("c1", "a", True),
                _result("c2", "a", True),
                _result("c3", "a", False),
                _result("c4", "a", False),
            ]
        )
        current = _report([_result(f"c{i}", "a", True) for i in range(1, 5)])
        diff = compare_to_baseline(current, baseline)
        assert diff.overall_pass_rate_delta == 0.5
