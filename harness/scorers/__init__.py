"""Scorers — pluggable strategies for judging a model response.

Every scorer implements `Scorer` from `harness.scorers.base`. The runner
dispatches via a registry (`build_default_registry()`); never special-case a
scorer in the runner. `DEFAULT_SCORER_NAMES` is the full set of registered
scorer names for use as `known_scorers` at suite load time.
"""

from harness.scorers.base import Scorer
from harness.scorers.exact_match import ExactMatchScorer
from harness.scorers.llm_judge import LLMJudgeScorer
from harness.scorers.regex_match import RegexMatchScorer
from harness.scorers.registry import DEFAULT_SCORER_NAMES, build_default_registry

__all__ = [
    "DEFAULT_SCORER_NAMES",
    "ExactMatchScorer",
    "LLMJudgeScorer",
    "RegexMatchScorer",
    "Scorer",
    "build_default_registry",
]
