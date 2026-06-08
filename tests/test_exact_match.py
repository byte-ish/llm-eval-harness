"""Tests for `harness.scorers.exact_match`."""

import pytest

from harness.models import EvalCase
from harness.scorers.exact_match import ExactMatchScorer


def _case(expected: list[str]) -> EvalCase:
    return EvalCase(
        id="c1",
        category="cat",
        scorer="exact_match",
        user="u",
        system="s",
        expected_contains=expected,
    )


class TestExactMatch:
    def test_all_phrases_present_passes(self) -> None:
        result = ExactMatchScorer().score(
            "alpha beta gamma delta", _case(["alpha", "beta", "gamma"])
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_partial_match_does_not_pass_but_scores(self) -> None:
        result = ExactMatchScorer().score("only alpha here", _case(["alpha", "beta", "gamma"]))
        assert result.passed is False
        assert result.score == pytest.approx(1 / 3)

    def test_no_match_scores_zero(self) -> None:
        result = ExactMatchScorer().score("totally unrelated", _case(["alpha", "beta"]))
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive(self) -> None:
        result = ExactMatchScorer().score("alpha BETA included", _case(["ALPHA", "Beta"]))
        assert result.passed is True

    def test_missing_phrases_named_in_reason(self) -> None:
        result = ExactMatchScorer().score("alpha only", _case(["alpha", "beta"]))
        assert "beta" in result.reason

    def test_empty_expected_raises(self) -> None:
        # Config-level validation catches this earlier; the scorer guards too
        # so programmatic construction can never silently succeed.
        with pytest.raises(ValueError, match="empty expected_contains"):
            ExactMatchScorer().score("anything", _case([]))

    def test_name_attribute(self) -> None:
        assert ExactMatchScorer.name == "exact_match"
