"""Tests for `harness.store` — RunReport JSON round-trip."""

from pathlib import Path

from harness.models import EvalResult, RunReport, ScorerResult
from harness.store import load, save


def _make_report() -> RunReport:
    results = [
        EvalResult(
            case_id="c1",
            category="cat_a",
            response="hello",
            scorer_result=ScorerResult(passed=True, score=1.0, reason="ok"),
            latency_ms=120.5,
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            model="claude-opus-4-7-20260101",
            temperature=0.0,
        ),
        EvalResult(
            case_id="c2",
            category="cat_b",
            response="",
            scorer_result=ScorerResult(passed=False, score=0.0, reason="error"),
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model="",
            temperature=0.4,
            error="rate limit",
        ),
    ]
    return RunReport(
        run_id="20260609-120000123456",
        model="claude-opus-4-7-20260101",
        suite="summarisation",
        dataset_version="1.0.0",
        results=results,
        started_at="2026-06-09T12:00:00Z",
        finished_at="2026-06-09T12:01:00Z",
        harness_version="0.1.0",
    )


def test_round_trip(tmp_path: Path) -> None:
    original = _make_report()
    path = save(original, tmp_path)
    assert path.exists()
    loaded = load(path)
    assert loaded == original


def test_save_creates_results_dir(tmp_path: Path) -> None:
    sub = tmp_path / "deep" / "dir"
    save(_make_report(), sub)
    assert sub.is_dir()


def test_filename_uses_run_id(tmp_path: Path) -> None:
    report = _make_report()
    path = save(report, tmp_path)
    assert path.name == f"{report.run_id}.json"


def test_error_field_round_trips(tmp_path: Path) -> None:
    report = _make_report()
    loaded = load(save(report, tmp_path))
    c2 = next(r for r in loaded.results if r.case_id == "c2")
    assert c2.error == "rate limit"
