"""Multi-model comparison — run the same suite across N models and diff results.

`compare_reports(reports)` takes a list of `RunReport`s (each from the same
suite + dataset_version) and produces a `ComparisonReport` with per-model
summary stats, per-category pass-rate deltas, and per-case wins/losses.

The comparison treats the first report as the reference ("baseline") for
delta direction: positive delta = candidate is better than reference, negative
= worse. This matches how a reviewer reads "should I switch from A to B?".

Dataset-version safety: refuses to compare across mismatched `dataset_version`
strings unless explicitly allowed — same posture as `regression.compare_to_baseline`.
"""

from dataclasses import dataclass

from harness.models import RunReport


class ComparisonError(ValueError):
    """Raised when comparison preconditions aren't met."""


@dataclass(frozen=True)
class ModelSummary:
    """Aggregate stats for one model in the comparison."""

    model: str  # the resolved provider model ID from the RunReport
    label: str  # the user-supplied spec like "openai:gpt-4o-mini"
    pass_rate: float
    pass_rate_by_category: dict[str, float]
    total_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    case_count: int


@dataclass(frozen=True)
class CaseOutcome:
    """One case's pass/fail across all compared models, in display order."""

    case_id: str
    category: str
    passed_by_model: list[bool]  # parallel to the comparison's model list


@dataclass(frozen=True)
class ComparisonReport:
    """The full comparison: per-model summaries plus case-level outcomes.

    `labels`, `summaries`, and the parallel arrays inside `case_outcomes`
    all share the same ordering — index 0 is the reference model.
    """

    suite: str
    dataset_version: str
    labels: list[str]
    summaries: list[ModelSummary]
    case_outcomes: list[CaseOutcome]
    started_at: str
    finished_at: str

    @property
    def reference_label(self) -> str:
        return self.labels[0]

    def pass_rate_delta_by_category(self, candidate_index: int) -> dict[str, float]:
        """Per-category delta of `candidate` vs. the reference model.

        Positive = candidate is better than reference in that category.
        Categories missing from one model are skipped.
        """
        reference = self.summaries[0].pass_rate_by_category
        candidate = self.summaries[candidate_index].pass_rate_by_category
        shared = set(reference) & set(candidate)
        return {cat: candidate[cat] - reference[cat] for cat in sorted(shared)}

    def overall_pass_rate_delta(self, candidate_index: int) -> float:
        return self.summaries[candidate_index].pass_rate - self.summaries[0].pass_rate

    def cost_delta_usd(self, candidate_index: int) -> float:
        """Candidate cost minus reference cost. Positive = candidate is more expensive."""
        return self.summaries[candidate_index].total_cost_usd - self.summaries[0].total_cost_usd

    def p50_latency_delta_ms(self, candidate_index: int) -> float:
        return self.summaries[candidate_index].p50_latency_ms - self.summaries[0].p50_latency_ms


def compare_reports(
    reports: list[RunReport],
    labels: list[str],
    *,
    allow_dataset_mismatch: bool = False,
) -> ComparisonReport:
    """Build a `ComparisonReport` from `len(reports) == len(labels) >= 2` runs."""
    if len(reports) != len(labels):
        raise ComparisonError(
            f"reports/labels length mismatch: {len(reports)} reports vs {len(labels)} labels"
        )
    if len(reports) < 2:
        raise ComparisonError("comparison needs at least two reports")

    suites = {r.suite for r in reports}
    if len(suites) != 1:
        raise ComparisonError(f"reports must come from the same suite; got {sorted(suites)}")

    versions = {r.dataset_version for r in reports}
    if len(versions) != 1 and not allow_dataset_mismatch:
        raise ComparisonError(
            f"dataset_version mismatch across reports: {sorted(versions)}. "
            "Re-run with a refreshed baseline or pass --allow-dataset-mismatch."
        )

    summaries = [
        ModelSummary(
            model=r.model,
            label=label,
            pass_rate=r.pass_rate,
            pass_rate_by_category=r.pass_rate_by_category,
            total_cost_usd=r.total_cost_usd,
            p50_latency_ms=r.p50_latency_ms,
            p95_latency_ms=r.p95_latency_ms,
            case_count=len(r.results),
        )
        for r, label in zip(reports, labels, strict=True)
    ]

    case_outcomes = _case_outcomes(reports)

    started_at = min(r.started_at for r in reports)
    finished_at = max(r.finished_at for r in reports)

    return ComparisonReport(
        suite=reports[0].suite,
        dataset_version=reports[0].dataset_version,
        labels=labels,
        summaries=summaries,
        case_outcomes=case_outcomes,
        started_at=started_at,
        finished_at=finished_at,
    )


def _case_outcomes(reports: list[RunReport]) -> list[CaseOutcome]:
    """Build the per-case pass/fail matrix in the reference report's case order."""
    by_case: list[dict[str, bool]] = []
    categories: dict[str, str] = {}
    for r in reports:
        outcomes: dict[str, bool] = {}
        for result in r.results:
            outcomes[result.case_id] = result.scorer_result.passed
            categories.setdefault(result.case_id, result.category)
        by_case.append(outcomes)

    reference_ids = [r.case_id for r in reports[0].results]
    return [
        CaseOutcome(
            case_id=cid,
            category=categories[cid],
            passed_by_model=[per_model.get(cid, False) for per_model in by_case],
        )
        for cid in reference_ids
    ]
