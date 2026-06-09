"""Regression detection — compare a current `RunReport` to a stored baseline.

`compare_to_baseline()` returns a `RegressionDiff` describing per-category
pass-rate deltas, flagged categories (drop greater than `drop_threshold`,
default 0.05), newly failing case IDs, and the overall pass-rate delta.

`DatasetVersionMismatch` is raised when the baseline and current report carry
different `dataset_version` strings unless `allow_dataset_mismatch=True`. This
prevents the most common false-positive failure mode: "did the model regress?"
when in fact the dataset changed.
"""

from dataclasses import dataclass, field

from harness.models import RunReport

DEFAULT_DROP_THRESHOLD = 0.05


class DatasetVersionMismatch(ValueError):
    """Baseline and current report have different `dataset_version`."""


@dataclass(frozen=True)
class RegressionDiff:
    """Per-category and overall deltas, plus the regression decision."""

    has_regression: bool
    drop_threshold: float
    per_category_delta: dict[str, float]  # current - baseline (negative = worse)
    flagged_categories: list[str]  # categories whose drop exceeds `drop_threshold`
    newly_failing_case_ids: list[str]  # cases that passed in baseline but fail now
    overall_pass_rate_delta: float
    new_categories: list[str] = field(default_factory=list)
    removed_categories: list[str] = field(default_factory=list)


def compare_to_baseline(
    current: RunReport,
    baseline: RunReport,
    *,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    allow_dataset_mismatch: bool = False,
) -> RegressionDiff:
    """Compare `current` to `baseline` and return a structured diff.

    Raises `DatasetVersionMismatch` if the two reports have different
    `dataset_version` strings and `allow_dataset_mismatch` is False.
    """
    if current.dataset_version != baseline.dataset_version and not allow_dataset_mismatch:
        raise DatasetVersionMismatch(
            f"dataset_version mismatch: baseline={baseline.dataset_version!r}, "
            f"current={current.dataset_version!r}. Refresh the baseline "
            "(`--update-baseline`) or pass `--allow-dataset-mismatch` to override."
        )

    current_by_cat = current.pass_rate_by_category
    baseline_by_cat = baseline.pass_rate_by_category

    shared_categories = set(current_by_cat) & set(baseline_by_cat)
    new_categories = sorted(set(current_by_cat) - set(baseline_by_cat))
    removed_categories = sorted(set(baseline_by_cat) - set(current_by_cat))

    per_category_delta: dict[str, float] = {
        cat: current_by_cat[cat] - baseline_by_cat[cat] for cat in shared_categories
    }
    flagged_categories = sorted(
        cat for cat, delta in per_category_delta.items() if delta < -drop_threshold
    )
    newly_failing = sorted(_newly_failing_case_ids(current, baseline))

    return RegressionDiff(
        has_regression=bool(flagged_categories),
        drop_threshold=drop_threshold,
        per_category_delta=per_category_delta,
        flagged_categories=flagged_categories,
        newly_failing_case_ids=newly_failing,
        overall_pass_rate_delta=current.pass_rate - baseline.pass_rate,
        new_categories=new_categories,
        removed_categories=removed_categories,
    )


def _newly_failing_case_ids(current: RunReport, baseline: RunReport) -> list[str]:
    """Case IDs that passed in the baseline run but fail in the current run."""
    baseline_passing = {r.case_id for r in baseline.results if r.scorer_result.passed}
    current_failing = {r.case_id for r in current.results if not r.scorer_result.passed}
    return list(baseline_passing & current_failing)
