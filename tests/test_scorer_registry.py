"""Tests for `harness.scorers.registry` — built-in scorer set."""

from harness.scorers import (
    ExactMatchScorer,
    RegexMatchScorer,
    build_default_registry,
)


class TestDefaultRegistry:
    def test_has_exact_match(self) -> None:
        registry = build_default_registry()
        assert "exact_match" in registry
        assert isinstance(registry["exact_match"], ExactMatchScorer)

    def test_has_regex_match(self) -> None:
        registry = build_default_registry()
        assert "regex_match" in registry
        assert isinstance(registry["regex_match"], RegexMatchScorer)

    def test_keys_match_scorer_name_attributes(self) -> None:
        # The runner does `registry[case.scorer]` — the registry key must equal
        # the scorer's own `name` attribute.
        for key, scorer in build_default_registry().items():
            assert key == scorer.name

    def test_returns_fresh_dict_each_call(self) -> None:
        a = build_default_registry()
        b = build_default_registry()
        assert a is not b

    def test_no_extra_scorers(self) -> None:
        # Lock the ship-set for Phase 3. Phase 4 will add llm_judge.
        assert set(build_default_registry().keys()) == {"exact_match", "regex_match"}
