"""LLM-as-judge scorer — separate model scores subjective quality vs. a rubric.

The judge model is independent of the candidate model under test. It receives
the case's `judge_rubric` and the candidate's response, and is asked to return
a single-line JSON object `{"score": 0..1, "reason": "..."}`. The score is
clamped to `[0, 1]` and compared against `pass_threshold`.

Failure modes are designed to never crash the run:
  - judge adapter error  → ScorerResult(passed=False, reason="judge adapter error: ...")
  - judge JSON malformed → ScorerResult(passed=False, reason="judge parse error: ...")
  - missing `score`/`reason` keys → ScorerResult(passed=False, reason="judge parse error: ...")

The cost of the judge call is reported on `ScorerResult.cost_usd` so the runner
can fold it into `EvalResult.cost_usd`.
"""

import json
from typing import Any

from harness.adapters.base import ModelAdapter
from harness.budget import CostFn
from harness.models import EvalCase, ScorerResult

_JUDGE_SYSTEM = (
    "You are an evaluation judge. Given a rubric and a model response, output "
    'ONLY a single-line JSON object with two fields: "score" (a number from '
    "0.0 to 1.0 indicating how well the response satisfies the rubric) and "
    '"reason" (one sentence under 200 characters). Output only the JSON object '
    "— no prose, no markdown fences."
)

DEFAULT_PASS_THRESHOLD = 0.5


class LLMJudgeScorer:
    """Subjective-quality scorer driven by a separate judge model."""

    name = "llm_judge"

    def __init__(
        self,
        judge_adapter: ModelAdapter,
        judge_cost_fn: CostFn,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 60.0,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ) -> None:
        self._adapter = judge_adapter
        self._cost_fn = judge_cost_fn
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._pass_threshold = pass_threshold

    async def score(self, response: str, case: EvalCase) -> ScorerResult:
        if not case.judge_rubric:
            raise ValueError(
                f"llm_judge case '{case.id}' has no judge_rubric; "
                "this should have been caught at config load time."
            )

        judge_user = f"Rubric:\n{case.judge_rubric}\n\nResponse to evaluate:\n{response}"

        try:
            judge_response = await self._adapter.complete(
                system=_JUDGE_SYSTEM,
                user=judge_user,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
            )
        except Exception as exc:
            return ScorerResult(
                passed=False,
                score=0.0,
                reason=f"judge adapter error: {exc!r}",
                cost_usd=0.0,
            )

        cost = self._cost_fn(judge_response.input_tokens, judge_response.output_tokens)

        try:
            parsed = _parse_judge_response(judge_response.text)
        except _JudgeParseError as exc:
            return ScorerResult(
                passed=False,
                score=0.0,
                reason=f"judge parse error: {exc}",
                cost_usd=cost,
            )

        clamped = max(0.0, min(1.0, parsed["score"]))
        passed = clamped >= self._pass_threshold
        return ScorerResult(
            passed=passed,
            score=clamped,
            reason=parsed["reason"],
            cost_usd=cost,
        )


class _JudgeParseError(ValueError):
    """Raised when the judge's response can't be parsed into a valid score+reason."""


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the judge's JSON reply. Tolerates whitespace and ``` fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            tail = -1 if lines[-1].startswith("```") else len(lines)
            cleaned = "\n".join(lines[1:tail])

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise _JudgeParseError(f"not valid JSON ({exc.msg})") from exc

    if not isinstance(obj, dict):
        raise _JudgeParseError("expected a JSON object at top level")

    if "score" not in obj:
        raise _JudgeParseError("missing 'score' field")
    if "reason" not in obj:
        raise _JudgeParseError("missing 'reason' field")

    try:
        score = float(obj["score"])
    except (TypeError, ValueError) as exc:
        raise _JudgeParseError(f"'score' is not a number: {obj['score']!r}") from exc

    return {"score": score, "reason": str(obj["reason"])}
