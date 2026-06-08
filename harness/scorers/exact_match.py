"""Exact-match scorer: case-insensitive substring check.

For each string in `case.expected_contains`, check whether it appears in the
model response (case-insensitive). `score = matched / total`, `passed = all
matched`. Empty `expected_contains` is rejected at config load time; this
scorer raises if it gets called with an empty list anyway, as a defensive
check against programmatic misuse.
"""

from harness.models import EvalCase, ScorerResult


class ExactMatchScorer:
    """Substring-presence scorer. See module docstring for semantics."""

    name = "exact_match"

    def score(self, response: str, case: EvalCase) -> ScorerResult:
        expected = case.expected_contains
        if not expected:
            raise ValueError(
                f"exact_match case '{case.id}' has empty expected_contains; "
                "this should have been caught at config load time."
            )

        response_lower = response.lower()
        matched = [p for p in expected if p.lower() in response_lower]
        missed = [p for p in expected if p.lower() not in response_lower]

        score = len(matched) / len(expected)
        passed = not missed

        if passed:
            reason = f"all {len(expected)} expected phrase(s) present"
        else:
            reason = f"{len(matched)}/{len(expected)} matched; missing: {missed!r}"

        return ScorerResult(passed=passed, score=score, reason=reason)
