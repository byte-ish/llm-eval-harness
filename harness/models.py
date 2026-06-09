"""Core data contracts for the eval harness.

These dataclasses are the backbone of the system. Every layer (config loader,
runner, scorers, store, regression detector, reporter) talks through them.
Keep them frozen, keep them small, and do not add behaviour that does not
belong on a value object.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    """One test case as loaded from a suite YAML.

    `system` and `user` are optional in YAML, but the config loader resolves
    them from the suite-level defaults at load time. By the time the runner
    sees a case, both prompts are filled in or load failed loudly.
    """

    id: str
    category: str
    scorer: str
    user: str | None = None
    system: str | None = None
    expected_contains: list[str] = field(default_factory=list)
    expected_regex: str | None = None
    judge_rubric: str | None = None
    temperature: float | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalSuite:
    """A loaded YAML suite plus its suite-level prompt defaults.

    `cases` is a tuple so the suite is structurally immutable — the runner
    iterates `suite.cases` and never mutates it.
    """

    name: str
    dataset_version: str
    default_system: str | None
    default_user: str | None
    cases: tuple[EvalCase, ...]


@dataclass(frozen=True)
class ScorerResult:
    """The output of any scorer for one case."""

    passed: bool
    score: float  # 0.0 to 1.0, even for binary scorers
    reason: str
    cost_usd: float = 0.0  # cost incurred by the scorer itself (e.g. judge model calls)


@dataclass(frozen=True)
class EvalResult:
    """One case after it has been run against the model and scored."""

    case_id: str
    category: str
    response: str
    scorer_result: ScorerResult
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str  # resolved provider model ID (post-alias)
    temperature: float
    error: str | None = None


@dataclass(frozen=True)
class RunReport:
    """The full result of one run, suitable for persistence and comparison."""

    run_id: str
    model: str
    suite: str
    dataset_version: str
    results: list[EvalResult]
    started_at: str
    finished_at: str
    harness_version: str

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.scorer_result.passed)
        return passed / len(self.results)

    @property
    def pass_rate_by_category(self) -> dict[str, float]:
        if not self.results:
            return {}
        totals: Counter[str] = Counter()
        passed: Counter[str] = Counter()
        for r in self.results:
            totals[r.category] += 1
            if r.scorer_result.passed:
                passed[r.category] += 1
        return {cat: passed[cat] / totals[cat] for cat in totals}

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def p50_latency_ms(self) -> float:
        return _percentile([r.latency_ms for r in self.results], 50.0)

    @property
    def p95_latency_ms(self) -> float:
        return _percentile([r.latency_ms for r in self.results], 95.0)


def _percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy's default `linear` method)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
