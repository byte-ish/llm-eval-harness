"""Generate the committed sample artifacts.

Outputs:
  - results/baseline.json   — synthetic RunReport at dataset_version 1.2.0
  - reports/sample_report.html — Jinja2-rendered HTML over that report

Re-run whenever the dataset bumps or the report template changes.
Usage:
  uv run python scripts/make_sample_artifacts.py
"""

import json
from dataclasses import asdict
from pathlib import Path

from harness.models import EvalResult, RunReport, ScorerResult
from harness.report import render

REPO_ROOT = Path(__file__).resolve().parent.parent


def _result(
    case_id: str,
    category: str,
    *,
    passed: bool,
    response: str,
    reason: str,
    score: float | None = None,
    latency_ms: float = 920.0,
    input_tokens: int = 180,
    output_tokens: int = 120,
    cost_usd: float = 0.0042,
    error: str | None = None,
) -> EvalResult:
    if score is None:
        score = 1.0 if passed else 0.0
    return EvalResult(
        case_id=case_id,
        category=category,
        response=response,
        scorer_result=ScorerResult(passed=passed, score=score, reason=reason),
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        model="claude-opus-4-7-20260101",
        temperature=0.0,
        error=error,
    )


def build_baseline() -> RunReport:
    """Construct a plausible synthetic run for the shipped summarisation suite."""
    results = [
        # earnings_summary — all pass
        _result(
            "earnings_001",
            "earnings_summary",
            passed=True,
            response=(
                "Acme Corp posted Q3 2026 revenue of $4.2 billion, up 18% "
                "year-over-year, beating the $3.9 billion analyst consensus. "
                "Net income reached $620 million."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        _result(
            "earnings_002",
            "earnings_summary",
            passed=True,
            response=(
                "Northwind Logistics narrowed its Q2 loss to $44 million from "
                "$61 million a year earlier, on revenue up 7% to $1.1 billion."
            ),
            reason="all 5 expected phrase(s) present",
        ),
        _result(
            "earnings_003",
            "earnings_summary",
            passed=True,
            response=(
                "Helios Semiconductor reported full-year 2025 revenue of "
                "$12.6 billion, down 4% from 2024. Operating margin contracted "
                "from 17% to 11%."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        # guidance_change — all pass
        _result(
            "guidance_001",
            "guidance_change",
            passed=True,
            response=(
                "Brightline Software raised its FY26 revenue guidance to "
                "$2.8-2.9 billion (prior: $2.6-2.7 billion) on stronger "
                "enterprise renewals."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        _result(
            "guidance_002",
            "guidance_change",
            passed=True,
            response=(
                "Quanta Energy cut its FY26 EBITDA outlook to $900 million "
                "from a prior midpoint of $1.05 billion, citing weaker LNG "
                "prices and a delayed Texas pipeline."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        _result(
            "guidance_003",
            "guidance_change",
            passed=True,
            response=(
                "Vertex Pharmaceuticals reaffirmed its 2026 product revenue "
                "guidance of $11.0-$11.5 billion despite a softer Q1."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        # ma_activity — 2 pass, 1 fail (illustrative)
        _result(
            "ma_001",
            "ma_activity",
            passed=True,
            response=(
                "Apex Industrial will acquire Caldera Pipeline for "
                "$7.4 billion in cash, expected to close Q4 2026 subject to "
                "regulatory approval."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        _result(
            "ma_002",
            "ma_activity",
            passed=False,
            response=(
                "Solstice Bank Group walked away from a planned tie-up with "
                "Riverstone Financial after EU competition pushback."
            ),
            reason=("3/4 matched; missing: ['terminated']"),
            score=0.75,
        ),
        _result(
            "ma_003",
            "ma_activity",
            passed=True,
            response=(
                "Glendale Capital completed its $5.8 billion take-private of "
                "TerraSat Communications, with TerraSat shareholders receiving "
                "$42 per share in cash."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        # regulatory_notice — all pass
        _result(
            "reg_001",
            "regulatory_notice",
            passed=True,
            response=(
                "The SEC charged Beacon Capital Advisors over undisclosed "
                "conflicts of interest in its proprietary trading desk; "
                "Beacon agreed to pay $185 million to settle."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        _result(
            "reg_002",
            "regulatory_notice",
            passed=True,
            response=(
                "The European Commission opened a formal antitrust "
                "investigation into Lumina Technologies' bundling of cloud "
                "storage and AI inference, sending a Statement of Objections."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        _result(
            "reg_003",
            "regulatory_notice",
            passed=True,
            response=(
                "The FDA granted accelerated approval to NovaBio's Nuvarex "
                "for refractory multiple myeloma, based on Phase II data with "
                "a 41% objective response rate."
            ),
            reason="all 4 expected phrase(s) present",
        ),
        # formatting (regex) — 2 pass, 1 fail
        _result(
            "format_001",
            "formatting",
            passed=True,
            response="$4.20 billion",
            reason="matched: $4.20 billion",
            latency_ms=420.0,
            output_tokens=8,
            cost_usd=0.0008,
        ),
        _result(
            "format_002",
            "formatting",
            passed=True,
            response="VRTX",
            reason="matched: VRTX",
            latency_ms=380.0,
            output_tokens=3,
            cost_usd=0.0006,
        ),
        _result(
            "format_003",
            "formatting",
            passed=False,
            response="March 15, 2026",
            reason=r"no match for: \d{4}-\d{2}-\d{2}",
            latency_ms=410.0,
            output_tokens=6,
            cost_usd=0.0007,
        ),
        # subjective_quality (judge) — pass
        _result(
            "judge_001",
            "subjective_quality",
            passed=True,
            response=(
                "Acme Corp's Q3 2026 results showcased durable demand: "
                "revenue of $4.2 billion (up 18% YoY) outpaced the "
                "$3.9 billion consensus, while net income of $620 million "
                "and a 200-bps operating-margin expansion to 22% point to "
                "operating leverage in the model."
            ),
            reason=(
                "Accurate on every number and entity; covers headline metrics "
                "(revenue, growth, beat); tone is appropriately analytical."
            ),
            score=0.88,
            latency_ms=1850.0,
            input_tokens=320,
            output_tokens=240,
            cost_usd=0.0218,  # candidate + judge call
        ),
    ]

    return RunReport(
        run_id="20260609-090000000000",
        model="claude-opus-4-7-20260101",
        suite="summarisation",
        dataset_version="1.2.0",
        results=results,
        started_at="2026-06-09T09:00:00+00:00",
        finished_at="2026-06-09T09:01:34+00:00",
        harness_version="0.1.0",
    )


def main() -> None:
    report = build_baseline()

    results_dir = REPO_ROOT / "results"
    reports_dir = REPO_ROOT / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = results_dir / "baseline.json"
    with baseline_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)
    print(f"wrote {baseline_path}")

    sample_html = reports_dir / "sample_report.html"
    render(report, None, sample_html)
    print(f"wrote {sample_html}")


if __name__ == "__main__":
    main()
