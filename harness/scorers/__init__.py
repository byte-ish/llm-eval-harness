"""Scorers — pluggable strategies for judging a model response.

Every scorer implements `Scorer` from `harness.scorers.base`. The runner
looks them up in a registry (Phase 3); never special-case a scorer.
"""

from harness.scorers.base import Scorer
from harness.scorers.exact_match import ExactMatchScorer

__all__ = ["ExactMatchScorer", "Scorer"]
