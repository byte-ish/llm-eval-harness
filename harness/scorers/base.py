"""Scorer Protocol — every scorer implements exactly this."""

from typing import Protocol

from harness.models import EvalCase, ScorerResult


class Scorer(Protocol):
    """Pluggable scoring strategy.

    The runner looks scorers up by `name` in a registry (Phase 3) and calls
    `score(response, case)` once per case. Implementations must be
    side-effect free and deterministic — they can be re-run on a stored
    response and produce the same `ScorerResult`.
    """

    name: str

    def score(self, response: str, case: EvalCase) -> ScorerResult: ...
