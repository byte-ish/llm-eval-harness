"""Regex-match scorer: `re.search` over the response against `expected_regex`.

Binary score: `passed=True / score=1.0` on match, else `passed=False / score=0.0`.
Empty / missing `expected_regex` is rejected at config load time. The scorer
also guards defensively in case it is constructed programmatically.
"""

import re

from harness.models import EvalCase, ScorerResult


class RegexMatchScorer:
    """Single-pattern regex scorer. See module docstring for semantics."""

    name = "regex_match"

    def score(self, response: str, case: EvalCase) -> ScorerResult:
        if not case.expected_regex:
            raise ValueError(
                f"regex_match case '{case.id}' has no expected_regex; "
                "this should have been caught at config load time."
            )
        try:
            pattern = re.compile(case.expected_regex)
        except re.error as exc:
            raise ValueError(f"regex_match case '{case.id}' has invalid pattern: {exc}") from exc

        match = pattern.search(response)
        if match:
            return ScorerResult(
                passed=True,
                score=1.0,
                reason=f"matched: {match.group(0)[:80]}",
            )
        return ScorerResult(
            passed=False,
            score=0.0,
            reason=f"no match for: {case.expected_regex}",
        )
