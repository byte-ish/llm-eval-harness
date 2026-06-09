"""Tests for `harness.config` — YAML suite loading and price map."""

from pathlib import Path

import pytest

from harness.config import (
    PriceEntry,
    SuiteValidationError,
    load_price_map,
    load_suite,
)


def _write_suite(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "suite.yaml"
    p.write_text(body, encoding="utf-8")
    return p


class TestLoadSuite:
    def test_round_trips_dataset_version(self, tmp_path: Path) -> None:
        suite = load_suite(
            _write_suite(
                tmp_path,
                """
name: test_suite
dataset_version: "2.5.0"
default_system: helper
cases:
  - id: c1
    category: cat
    scorer: exact_match
    user: hello
    expected_contains: ["world"]
""",
            )
        )
        assert suite.dataset_version == "2.5.0"
        assert suite.name == "test_suite"

    def test_name_defaults_to_file_stem(self, tmp_path: Path) -> None:
        suite = load_suite(
            _write_suite(
                tmp_path,
                """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: exact_match
    expected_contains: ["x"]
""",
            )
        )
        assert suite.name == "suite"  # file stem is `suite.yaml`

    def test_missing_dataset_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteValidationError, match="dataset_version"):
            load_suite(
                _write_suite(
                    tmp_path,
                    """
name: x
cases: []
""",
                )
            )

    def test_suite_default_applied_to_case(self, tmp_path: Path) -> None:
        suite = load_suite(
            _write_suite(
                tmp_path,
                """
dataset_version: "1.0.0"
default_system: SHARED_SYSTEM
default_user: SHARED_USER
cases:
  - id: c1
    category: cat
    scorer: exact_match
    expected_contains: ["x"]
""",
            )
        )
        assert suite.cases[0].system == "SHARED_SYSTEM"
        assert suite.cases[0].user == "SHARED_USER"

    def test_case_override_wins(self, tmp_path: Path) -> None:
        suite = load_suite(
            _write_suite(
                tmp_path,
                """
dataset_version: "1.0.0"
default_system: SHARED_SYSTEM
default_user: SHARED_USER
cases:
  - id: c1
    category: cat
    scorer: exact_match
    user: PER_CASE_USER
    expected_contains: ["x"]
""",
            )
        )
        assert suite.cases[0].user == "PER_CASE_USER"
        assert suite.cases[0].system == "SHARED_SYSTEM"

    def test_missing_user_and_no_default_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteValidationError, match="no `user`"):
            load_suite(
                _write_suite(
                    tmp_path,
                    """
dataset_version: "1.0.0"
default_system: shared
cases:
  - id: c1
    category: cat
    scorer: exact_match
    expected_contains: ["x"]
""",
                )
            )

    def test_missing_system_and_no_default_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteValidationError, match="no `system`"):
            load_suite(
                _write_suite(
                    tmp_path,
                    """
dataset_version: "1.0.0"
default_user: shared
cases:
  - id: c1
    category: cat
    scorer: exact_match
    expected_contains: ["x"]
""",
                )
            )

    def test_exact_match_with_empty_expected_contains_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteValidationError, match="empty"):
            load_suite(
                _write_suite(
                    tmp_path,
                    """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: exact_match
""",
                )
            )

    def test_cases_is_tuple_of_evalcases(self, tmp_path: Path) -> None:
        suite = load_suite(
            _write_suite(
                tmp_path,
                """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: exact_match
    expected_contains: ["x"]
  - id: c2
    category: cat
    scorer: exact_match
    expected_contains: ["y"]
""",
            )
        )
        assert isinstance(suite.cases, tuple)
        assert len(suite.cases) == 2
        assert suite.cases[0].id == "c1"
        assert suite.cases[1].id == "c2"

    def test_per_case_temperature_parsed(self, tmp_path: Path) -> None:
        suite = load_suite(
            _write_suite(
                tmp_path,
                """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: exact_match
    temperature: 0.4
    expected_contains: ["x"]
""",
            )
        )
        assert suite.cases[0].temperature == 0.4

    def test_summarisation_suite_loads(self) -> None:
        # The shipped suite must load cleanly — it's what the runner consumes.
        suite = load_suite(Path("evals/summarisation.yaml"))
        assert suite.dataset_version == "1.2.0"
        assert len(suite.cases) >= 16
        assert all(c.system for c in suite.cases)
        assert all(c.user for c in suite.cases)

    def test_summarisation_suite_loads_with_known_scorers(self) -> None:
        suite = load_suite(
            Path("evals/summarisation.yaml"),
            known_scorers={"exact_match", "regex_match", "llm_judge"},
        )
        assert any(c.scorer == "regex_match" for c in suite.cases)
        assert any(c.scorer == "exact_match" for c in suite.cases)
        assert any(c.scorer == "llm_judge" for c in suite.cases)

    def test_llm_judge_with_no_judge_rubric_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteValidationError, match="llm_judge"):
            load_suite(
                _write_suite(
                    tmp_path,
                    """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: llm_judge
""",
                )
            )

    def test_regex_match_with_no_expected_regex_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteValidationError, match="regex_match"):
            load_suite(
                _write_suite(
                    tmp_path,
                    """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: regex_match
""",
                )
            )

    def test_unknown_scorer_raises_when_known_scorers_provided(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteValidationError, match="unknown scorer"):
            load_suite(
                _write_suite(
                    tmp_path,
                    """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: not_a_real_scorer
    expected_contains: ["x"]
""",
                ),
                known_scorers={"exact_match", "regex_match"},
            )

    def test_unknown_scorer_allowed_when_known_scorers_is_none(self, tmp_path: Path) -> None:
        # Backward compat: existing callers omit known_scorers and validation skips.
        suite = load_suite(
            _write_suite(
                tmp_path,
                """
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: not_a_real_scorer
    expected_contains: ["x"]
""",
            ),
        )
        assert suite.cases[0].scorer == "not_a_real_scorer"


class TestLoadPriceMap:
    def test_loads_known_models(self) -> None:
        prices = load_price_map()
        assert "claude-opus-4-7" in prices
        entry = prices["claude-opus-4-7"]
        assert isinstance(entry, PriceEntry)
        assert entry.input_per_million > 0
        assert entry.output_per_million > 0

    def test_all_entries_have_positive_rates(self) -> None:
        for model_id, entry in load_price_map().items():
            assert entry.input_per_million > 0, model_id
            assert entry.output_per_million > 0, model_id
