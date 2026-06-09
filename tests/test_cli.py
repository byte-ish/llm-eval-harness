"""Tests for `run_evals.py` CLI — budget guard + happy-path dispatch."""

from pathlib import Path

import pytest

import run_evals
from harness.config import PriceEntry
from tests.fixtures.mock_responses import MockAdapter, MockResponse

_PRICES = {"test-model": PriceEntry(input_per_million=10.0, output_per_million=20.0)}


def _write_minimal_suite(tmp_path: Path) -> Path:
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir()
    (evals_dir / "tiny.yaml").write_text(
        """
name: tiny
dataset_version: "1.0.0"
default_system: s
default_user: u
cases:
  - id: c1
    category: cat
    scorer: exact_match
    expected_contains: ["hello"]
""",
        encoding="utf-8",
    )
    return evals_dir


def test_budget_refused_returns_exit_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)

    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--model",
            "test-model",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-cost",
            "0.0000001",
            "--no-progress",
        ]
    )
    assert exit_code == 3


def test_force_bypasses_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    monkeypatch.setattr(
        run_evals,
        "_build_adapter",
        lambda *a, **kw: MockAdapter(responses=[MockResponse(text="hello world")]),
    )

    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--model",
            "test-model",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-cost",
            "0.0000001",
            "--force",
            "--no-progress",
        ]
    )
    assert exit_code == 0
    results = list((tmp_path / "results").glob("*.json"))
    assert len(results) == 1


def test_happy_path_writes_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    monkeypatch.setattr(
        run_evals,
        "_build_adapter",
        lambda *a, **kw: MockAdapter(responses=[MockResponse(text="hello there")]),
    )

    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--model",
            "test-model",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--no-progress",
        ]
    )
    assert exit_code == 0


def test_unknown_model_exits_with_systemexit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    with pytest.raises(SystemExit, match="price map"):
        run_evals.main(
            [
                "--suite",
                "tiny",
                "--model",
                "not-in-price-map",
                "--evals-dir",
                str(evals_dir),
                "--results-dir",
                str(tmp_path / "results"),
                "--no-progress",
            ]
        )
