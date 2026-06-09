"""Tests for `harness.report` — HTML rendering (single + comparison)."""

from pathlib import Path

from harness.compare import compare_reports
from harness.models import EvalResult, RunReport, ScorerResult
from harness.regression import compare_to_baseline
from harness.report import render, render_comparison


def _result(case_id: str, category: str, passed: bool, response: str = "ok") -> EvalResult:
    return EvalResult(
        case_id=case_id,
        category=category,
        response=response,
        scorer_result=ScorerResult(passed=passed, score=1.0 if passed else 0.0, reason="r"),
        latency_ms=100.0,
        input_tokens=20,
        output_tokens=10,
        cost_usd=0.001,
        model="m",
        temperature=0.0,
    )


def _report(results: list[EvalResult]) -> RunReport:
    return RunReport(
        run_id="20260609-120000000000",
        model="claude-opus-4-7-20260101",
        suite="summarisation",
        dataset_version="1.2.0",
        results=results,
        started_at="2026-06-09T12:00:00Z",
        finished_at="2026-06-09T12:01:00Z",
        harness_version="0.1.0",
    )


class TestRenderWithoutDiff:
    def test_writes_html_file(self, tmp_path: Path) -> None:
        report = _report([_result("c1", "a", True)])
        out = tmp_path / "report.html"
        path = render(report, None, out)
        assert path == out
        assert out.exists()

    def test_contains_run_metadata(self, tmp_path: Path) -> None:
        report = _report([_result("c1", "a", True)])
        out = render(report, None, tmp_path / "r.html")
        html = out.read_text(encoding="utf-8")
        assert report.run_id in html
        assert report.model in html
        assert report.dataset_version in html
        assert report.suite in html

    def test_contains_per_category_table(self, tmp_path: Path) -> None:
        report = _report([_result("c1", "a", True), _result("c2", "b", False)])
        out = render(report, None, tmp_path / "r.html")
        html = out.read_text(encoding="utf-8")
        assert "<table" in html
        assert ">a<" in html
        assert ">b<" in html

    def test_contains_per_case_details(self, tmp_path: Path) -> None:
        report = _report([_result("c1", "earnings_summary", True, response="hello world")])
        out = render(report, None, tmp_path / "r.html")
        html = out.read_text(encoding="utf-8")
        assert "c1" in html
        assert "earnings_summary" in html
        assert "hello world" in html
        assert "PASS" in html


class TestRenderWithDiff:
    def test_shows_regression_banner(self, tmp_path: Path) -> None:
        baseline = _report([_result("c1", "a", True), _result("c2", "a", True)])
        current = _report([_result("c1", "a", False), _result("c2", "a", False)])
        diff = compare_to_baseline(current, baseline)
        out = render(current, diff, tmp_path / "r.html")
        html = out.read_text(encoding="utf-8")
        assert "Regression detected" in html
        assert "a" in html

    def test_shows_no_regression_banner(self, tmp_path: Path) -> None:
        baseline = _report([_result("c1", "a", True)])
        current = _report([_result("c1", "a", True)])
        diff = compare_to_baseline(current, baseline)
        out = render(current, diff, tmp_path / "r.html")
        html = out.read_text(encoding="utf-8")
        assert "No regression" in html

    def test_delta_column_present(self, tmp_path: Path) -> None:
        baseline = _report([_result("c1", "a", False)])
        current = _report([_result("c1", "a", True)])
        diff = compare_to_baseline(current, baseline)
        out = render(current, diff, tmp_path / "r.html")
        html = out.read_text(encoding="utf-8")
        assert "Δ vs baseline" in html


class TestEscaping:
    def test_response_is_escaped(self, tmp_path: Path) -> None:
        # Jinja2 autoescape should turn <script> into &lt;script&gt;
        bad_response = "<script>alert('xss')</script>"
        report = _report([_result("c1", "a", True, response=bad_response)])
        out = render(report, None, tmp_path / "r.html")
        html = out.read_text(encoding="utf-8")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestRenderComparison:
    def test_writes_comparison_html(self, tmp_path: Path) -> None:
        ref = _report([_result("c1", "a", True), _result("c2", "b", False)])
        cand = _report([_result("c1", "a", True), _result("c2", "b", True)])
        comparison = compare_reports([ref, cand], ["anthropic:ref", "openai:cand"])
        out = render_comparison(comparison, tmp_path / "compare.html")
        html = out.read_text(encoding="utf-8")
        assert "anthropic:ref" in html
        assert "openai:cand" in html
        assert "reference" in html  # ref-tag
        assert "Per-category pass rate" in html
        assert "Per-case outcomes" in html

    def test_comparison_shows_pass_rate_deltas(self, tmp_path: Path) -> None:
        ref = _report([_result("c1", "a", True), _result("c2", "a", False)])
        cand = _report([_result("c1", "a", True), _result("c2", "a", True)])
        comparison = compare_reports([ref, cand], ["ref", "cand"])
        out = render_comparison(comparison, tmp_path / "compare.html")
        html = out.read_text(encoding="utf-8")
        # cand gained one case → +50% overall delta and +50% on category "a"
        assert "+50.0%" in html

    def test_comparison_escapes_user_supplied_labels(self, tmp_path: Path) -> None:
        ref = _report([_result("c1", "a", True)])
        cand = _report([_result("c1", "a", True)])
        evil_label = "<script>x</script>"
        comparison = compare_reports([ref, cand], [evil_label, "safe"])
        out = render_comparison(comparison, tmp_path / "compare.html")
        html = out.read_text(encoding="utf-8")
        assert "<script>x</script>" not in html
        assert "&lt;script&gt;" in html
