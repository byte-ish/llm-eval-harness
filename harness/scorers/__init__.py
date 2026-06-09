"""Scorers — pluggable strategies for judging a model response.

Every scorer implements `Scorer` from `harness.scorers.base`. The runner
dispatches via a registry (`build_default_registry()`); never special-case a
scorer in the runner.
"""

from harness.scorers.base import Scorer
from harness.scorers.exact_match import ExactMatchScorer
from harness.scorers.regex_match import RegexMatchScorer
from harness.scorers.registry import build_default_registry

__all__ = [
    "ExactMatchScorer",
    "RegexMatchScorer",
    "Scorer",
    "build_default_registry",
]
