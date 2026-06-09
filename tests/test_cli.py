"""Tests for `run_evals.py` CLI — budget guard + happy-path dispatch + --compare."""

from pathlib import Path

import pytest

import run_evals
from harness.adapters.factory import ModelSpec
from harness.config import PriceEntry
from tests.fixtures.mock_responses import MockAdapter, MockResponse

_PRICES = {
    "test-model": PriceEntry(input_per_million=10.0, output_per_million=20.0),
    "model-a": PriceEntry(input_per_million=1.0, output_per_million=2.0),
    "model-b": PriceEntry(input_per_million=2.0, output_per_million=4.0),
}


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


def _patch_adapter_factory(
    monkeypatch: pytest.MonkeyPatch, text_by_model: dict[str, str] | None = None
) -> None:
    """Replace `run_evals.build_adapter` with a `MockAdapter` factory.

    `text_by_model` lets the caller script different responses per model_id —
    useful for testing comparison flows.
    """
    text_by_model = text_by_model or {}

    def _factory(spec: ModelSpec) -> MockAdapter:
        text = text_by_model.get(spec.model_id, "hello there")
        return MockAdapter(responses=[MockResponse(text=text, resolved_model_id=spec.model_id)])

    monkeypatch.setattr(run_evals, "build_adapter", _factory)


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
            "--reports-dir",
            str(tmp_path / "reports"),
            "--max-cost",
            "0.0000001",
            "--no-progress",
        ]
    )
    assert exit_code == 3


def test_force_bypasses_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    _patch_adapter_factory(monkeypatch, {"test-model": "hello world"})

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
            "--reports-dir",
            str(tmp_path / "reports"),
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
    _patch_adapter_factory(monkeypatch, {"test-model": "hello there"})

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
            "--reports-dir",
            str(tmp_path / "reports"),
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
                "--reports-dir",
                str(tmp_path / "reports"),
                "--no-progress",
            ]
        )


def test_model_and_compare_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    with pytest.raises(SystemExit):
        run_evals.main(
            [
                "--suite",
                "tiny",
                "--model",
                "test-model",
                "--compare",
                "test-model,model-a",
                "--evals-dir",
                str(evals_dir),
                "--results-dir",
                str(tmp_path / "results"),
                "--reports-dir",
                str(tmp_path / "reports"),
                "--no-progress",
            ]
        )


def test_one_of_model_or_compare_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    with pytest.raises(SystemExit):
        run_evals.main(
            [
                "--suite",
                "tiny",
                "--evals-dir",
                str(evals_dir),
                "--results-dir",
                str(tmp_path / "results"),
                "--reports-dir",
                str(tmp_path / "reports"),
                "--no-progress",
            ]
        )


def test_compare_runs_each_model_and_writes_comparison_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    # model-a's response contains "hello" → passes; model-b's doesn't → fails.
    _patch_adapter_factory(monkeypatch, {"model-a": "hello world", "model-b": "different text"})

    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--compare",
            "model-a,model-b",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-progress",
        ]
    )
    assert exit_code == 0

    # Two run JSONs written (one per model).
    results = list((tmp_path / "results").glob("*.json"))
    assert len(results) == 2

    # One *_compare.html written.
    compares = list((tmp_path / "reports").glob("*_compare.html"))
    assert len(compares) == 1
    html = compares[0].read_text(encoding="utf-8")
    assert "model-a" in html
    assert "model-b" in html
    assert "reference" in html  # the ref-tag


def test_compare_needs_at_least_two_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    with pytest.raises(SystemExit, match="at least two"):
        run_evals.main(
            [
                "--suite",
                "tiny",
                "--compare",
                "model-a",
                "--evals-dir",
                str(evals_dir),
                "--results-dir",
                str(tmp_path / "results"),
                "--reports-dir",
                str(tmp_path / "reports"),
                "--no-progress",
            ]
        )


def test_compare_budget_refused_returns_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--compare",
            "model-a,model-b",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--max-cost",
            "0.0000001",
            "--no-progress",
        ]
    )
    assert exit_code == 3


def test_ollama_runs_without_price_map_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ollama is a zero-cost provider — no prices.yaml entry should be required."""
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    _patch_adapter_factory(monkeypatch, {"llama3.2": "hello world"})

    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--model",
            "ollama:llama3.2",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-progress",
        ]
    )
    assert exit_code == 0


def test_ollama_passes_tight_budget_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-cost providers should sail through even an aggressive --max-cost ceiling."""
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)
    _patch_adapter_factory(monkeypatch, {"llama3.2": "hello world"})

    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--model",
            "ollama:llama3.2",
            "--max-cost",
            "0.0000001",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-progress",
        ]
    )
    assert exit_code == 0


def test_provider_prefix_routes_via_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the CLI parses `anthropic:test-model` and passes a ModelSpec to the factory."""
    evals_dir = _write_minimal_suite(tmp_path)
    monkeypatch.setattr(run_evals, "load_price_map", lambda: _PRICES)

    seen: list[ModelSpec] = []

    def _factory(spec: ModelSpec) -> MockAdapter:
        seen.append(spec)
        return MockAdapter(responses=[MockResponse(text="hello world")])

    monkeypatch.setattr(run_evals, "build_adapter", _factory)

    exit_code = run_evals.main(
        [
            "--suite",
            "tiny",
            "--model",
            "anthropic:test-model",
            "--evals-dir",
            str(evals_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-progress",
        ]
    )
    assert exit_code == 0
    assert len(seen) == 1
    assert seen[0].provider == "anthropic"
    assert seen[0].model_id == "test-model"
