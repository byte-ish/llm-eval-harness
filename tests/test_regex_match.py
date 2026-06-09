"""Tests for `harness.scorers.regex_match`."""

import pytest

from harness.models import EvalCase
from harness.scorers.regex_match import RegexMatchScorer


def _case(pattern: str | None) -> EvalCase:
    return EvalCase(
        id="c1",
        category="cat",
        scorer="regex_match",
        user="u",
        system="s",
        expected_regex=pattern,
    )


class TestRegexMatch:
    def test_match_found_passes(self) -> None:
        result = RegexMatchScorer().score("call 555-1234 now", _case(r"\d{3}-\d{4}"))
        assert result.passed is True
        assert result.score == 1.0

    def test_no_match_fails_with_zero_score(self) -> None:
        result = RegexMatchScorer().score("no numbers here", _case(r"\d{3}-\d{4}"))
        assert result.passed is False
        assert result.score == 0.0

    def test_anchored_pattern_matches_exact_string(self) -> None:
        result = RegexMatchScorer().score("2026-06-09", _case(r"^\d{4}-\d{2}-\d{2}$"))
        assert result.passed is True

    def test_anchored_pattern_rejects_extra_content(self) -> None:
        result = RegexMatchScorer().score("today is 2026-06-09", _case(r"^\d{4}-\d{2}-\d{2}$"))
        assert result.passed is False

    def test_matched_text_quoted_in_reason(self) -> None:
        result = RegexMatchScorer().score("abc 555-1234 def", _case(r"\d{3}-\d{4}"))
        assert "555-1234" in result.reason

    def test_unmatched_pattern_quoted_in_reason(self) -> None:
        result = RegexMatchScorer().score("nope", _case(r"\d{3}-\d{4}"))
        assert r"\d{3}-\d{4}" in result.reason

    def test_empty_pattern_raises(self) -> None:
        # Config-level validation catches this earlier; the scorer guards too.
        with pytest.raises(ValueError, match="no expected_regex"):
            RegexMatchScorer().score("anything", _case(None))

    def test_invalid_pattern_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="invalid pattern"):
            RegexMatchScorer().score("anything", _case("[unclosed"))

    def test_name_attribute(self) -> None:
        assert RegexMatchScorer.name == "regex_match"
