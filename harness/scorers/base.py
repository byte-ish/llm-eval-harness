"""Scorer Protocol — every scorer implements exactly this."""

from typing import Protocol

from harness.models import EvalCase, ScorerResult


class Scorer(Protocol):
    """Pluggable scoring strategy.

    The runner looks scorers up by `name` in a registry and `await`s
    `score(response, case)` once per case. `score` is async so a scorer can
    legitimately make a network call (e.g. `LLMJudgeScorer`) without blocking
    the runner's event loop; cpu-bound scorers (`exact_match`, `regex_match`)
    are async too, they just do not await anything.
    """

    name: str

    async def score(self, response: str, case: EvalCase) -> ScorerResult: ...
