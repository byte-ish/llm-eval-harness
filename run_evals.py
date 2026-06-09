"""CLI entrypoint for the eval harness.

Usage (single model):
  python run_evals.py --suite summarisation --model claude-opus-4-7
  python run_evals.py --suite summarisation --model openai:gpt-4o-mini

Usage (multi-model comparison, Phase 7):
  python run_evals.py --suite summarisation \\
      --compare anthropic:claude-haiku-4-5-20251001,openai:gpt-4o-mini

Provider-prefix syntax: `<provider>:<model_id>`. Bare IDs default to
`anthropic` for backward compatibility.

Exit codes:
  0 - success (no regression in single-model mode; comparison ran)
  1 - regression detected vs --baseline (single-model mode only)
  2 - dataset_version mismatch between --baseline and current run
  3 - pre-flight budget refused (use --force to bypass)
"""

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from harness import __version__
from harness.adapters.base import ModelAdapter
from harness.adapters.factory import (
    ZERO_COST_PROVIDERS,
    ModelSpec,
    build_adapter,
    parse_model_spec,
)
from harness.budget import estimate_cost, make_cost_fn
from harness.compare import ComparisonReport, compare_reports
from harness.config import PriceEntry, load_price_map, load_suite
from harness.models import EvalSuite, RunReport
from harness.regression import (
    DatasetVersionMismatch,
    RegressionDiff,
    compare_to_baseline,
)
from harness.report import render as render_report
from harness.report import render_comparison
from harness.runner import run_suite
from harness.scorers import DEFAULT_SCORER_NAMES, build_default_registry
from harness.store import load as load_report
from harness.store import save as save_report

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_DATASET_MISMATCH = 2
EXIT_BUDGET_REFUSED = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_evals",
        description="Run an eval suite against an LLM and report results.",
    )
    parser.add_argument(
        "--suite", required=True, help="Suite name under --evals-dir (e.g. summarisation)"
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        help=(
            "Provider:model_id spec (e.g. anthropic:claude-opus-4-7, openai:gpt-4o-mini). "
            "Bare IDs default to anthropic."
        ),
    )
    model_group.add_argument(
        "--compare",
        help=(
            "Comma-separated provider:model_id specs to run side-by-side "
            "(e.g. anthropic:claude-haiku-4-5-20251001,openai:gpt-4o-mini). "
            "Emits a comparison HTML report; --baseline/--update-baseline are ignored."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the default temperature=0.0",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=(
            "Refuse to run if the pre-flight estimate exceeds this USD amount. "
            "In --compare mode, the ceiling applies per model."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if --max-cost would refuse",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model spec for llm_judge cases. Defaults to the model under test.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to a baseline RunReport JSON (single-model mode only)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Save the current run as the new baseline (single-model mode only)",
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Override the dataset_version safety check",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"))
    parser.add_argument("--no-progress", action="store_true", help="Hide the progress bar")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def _print_summary(report: RunReport, diff: RegressionDiff | None, console: Console) -> None:
    table = Table(title=f"Results: {report.suite} (dataset {report.dataset_version})")
    table.add_column("Category")
    table.add_column("Pass rate", justify="right")
    table.add_column("Cases", justify="right")
    if diff is not None:
        table.add_column("Δ baseline", justify="right")

    counts: dict[str, int] = {}
    for r in report.results:
        counts[r.category] = counts.get(r.category, 0) + 1

    for category, rate in sorted(report.pass_rate_by_category.items()):
        row = [category, f"{rate * 100:.1f}%", str(counts[category])]
        if diff is not None:
            delta = diff.per_category_delta.get(category)
            if delta is None:
                row.append("[dim]new[/dim]")
            else:
                colour = "red" if delta < 0 else "green" if delta > 0 else ""
                cell = f"{delta * 100:+.1f}%"
                row.append(f"[{colour}]{cell}[/{colour}]" if colour else cell)
        table.add_row(*row)

    table.add_section()
    overall_row = [
        "[bold]Overall",
        f"[bold]{report.pass_rate * 100:.1f}%",
        f"[bold]{len(report.results)}",
    ]
    if diff is not None:
        d = diff.overall_pass_rate_delta
        colour = "red" if d < 0 else "green" if d > 0 else ""
        cell = f"{d * 100:+.1f}%"
        overall_row.append(
            f"[bold][{colour}]{cell}[/{colour}][/bold]" if colour else f"[bold]{cell}[/bold]"
        )
    table.add_row(*overall_row)
    console.print(table)

    summary = Table(show_header=False, box=None)
    summary.add_row("Total cost USD", f"${report.total_cost_usd:.4f}")
    summary.add_row("p50 latency ms", f"{report.p50_latency_ms:.0f}")
    summary.add_row("p95 latency ms", f"{report.p95_latency_ms:.0f}")
    summary.add_row("Resolved model", report.model)
    summary.add_row("Run ID", report.run_id)
    console.print(summary)


def _print_comparison_summary(comparison: ComparisonReport, console: Console) -> None:
    table = Table(title=f"Comparison: {comparison.suite} (dataset {comparison.dataset_version})")
    table.add_column("Model")
    table.add_column("Pass rate", justify="right")
    table.add_column("Total cost", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("Δ pass vs ref", justify="right")
    table.add_column("Δ cost vs ref", justify="right")

    for i, s in enumerate(comparison.summaries):
        label = f"[bold]{s.label}[/bold]"
        if i == 0:
            label += " [dim](ref)[/dim]"
        if i == 0:
            row = [
                label,
                f"{s.pass_rate * 100:.1f}%",
                f"${s.total_cost_usd:.4f}",
                f"{s.p50_latency_ms:.0f}",
                "-",
                "-",
            ]
        else:
            pr_delta = comparison.overall_pass_rate_delta(i)
            cost_delta = comparison.cost_delta_usd(i)
            pr_colour = "green" if pr_delta > 0 else "red" if pr_delta < 0 else ""
            cost_colour = "green" if cost_delta < 0 else "red" if cost_delta > 0 else ""
            pr_cell = f"{pr_delta * 100:+.1f}%"
            cost_cell = f"${cost_delta:+.4f}"
            row = [
                label,
                f"{s.pass_rate * 100:.1f}%",
                f"${s.total_cost_usd:.4f}",
                f"{s.p50_latency_ms:.0f}",
                f"[{pr_colour}]{pr_cell}[/{pr_colour}]" if pr_colour else pr_cell,
                f"[{cost_colour}]{cost_cell}[/{cost_colour}]" if cost_colour else cost_cell,
            ]
        table.add_row(*row)
    console.print(table)


_ZERO_COST_PRICE = PriceEntry(input_per_million=0.0, output_per_million=0.0)


def _resolve_price(spec: ModelSpec, price_map: dict[str, PriceEntry]) -> PriceEntry:
    """Look up the price for a spec. Zero-cost providers bypass the price map."""
    if spec.provider in ZERO_COST_PROVIDERS:
        return _ZERO_COST_PRICE
    if spec.model_id not in price_map:
        raise SystemExit(
            f"model_id '{spec.model_id}' has no entry in price map; "
            "add it to harness/prices.yaml before running."
        )
    return price_map[spec.model_id]


def _print_regression_verdict(diff: RegressionDiff, console: Console) -> None:
    if diff.has_regression:
        console.print(
            f"[bold red]REGRESSION:[/bold red] {len(diff.flagged_categories)} "
            f"category(s) dropped > {diff.drop_threshold * 100:.0f}%: "
            f"{', '.join(diff.flagged_categories)}"
        )
        if diff.newly_failing_case_ids:
            console.print(f"Newly failing case(s): {', '.join(diff.newly_failing_case_ids)}")
    else:
        console.print(
            f"[bold green]OK:[/bold green] no regression "
            f"(overall {diff.overall_pass_rate_delta * 100:+.1f}% vs baseline)"
        )


def _baseline_path(args: argparse.Namespace) -> Path:
    if args.baseline is not None:
        return Path(args.baseline)
    return Path(args.results_dir) / "baseline.json"


def _parse_compare_specs(raw: str) -> list[ModelSpec]:
    """Split `--compare` value on commas at the **top level**, not inside provider:model.

    A bare model_id can't contain commas, so a simple `split(",")` is safe.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) < 2:
        raise SystemExit("--compare needs at least two comma-separated model specs")
    return [parse_model_spec(p) for p in parts]


def _preflight_budget(
    suite: EvalSuite,
    spec: ModelSpec,
    price_map: dict[str, PriceEntry],
    max_cost: float | None,
    force: bool,
    console: Console,
) -> int | None:
    """Return EXIT_BUDGET_REFUSED if the estimate is over budget; None otherwise."""
    if max_cost is None:
        return None
    if spec.provider in ZERO_COST_PROVIDERS:
        console.print(
            f"Pre-flight estimate: $0.0000 for {len(suite.cases)} case(s) on "
            f"{spec.display} (local model, zero-cost provider)"
        )
        return None
    estimate = estimate_cost(suite, price_map, spec.model_id)
    if estimate.estimated_usd > max_cost and not force:
        console.print(
            f"[red]{spec.display}: pre-flight estimate ${estimate.estimated_usd:.4f} exceeds "
            f"--max-cost ${max_cost:.4f}. Use --force to run anyway.[/red]"
        )
        return EXIT_BUDGET_REFUSED
    console.print(
        f"Pre-flight estimate: ${estimate.estimated_usd:.4f} for "
        f"{estimate.case_count} case(s) on {spec.display}"
    )
    return None


async def _run_one_model(
    spec: ModelSpec,
    suite: EvalSuite,
    args: argparse.Namespace,
    price_map: dict[str, PriceEntry],
    console: Console,
) -> RunReport:
    """Run the suite once against one model and return the report."""
    price = _resolve_price(spec, price_map)
    adapter: ModelAdapter = build_adapter(spec)
    cost_fn = make_cost_fn(price)

    needs_judge = any(c.scorer == "llm_judge" for c in suite.cases)
    if needs_judge:
        judge_spec = parse_model_spec(args.judge_model) if args.judge_model else spec
        judge_price = _resolve_price(judge_spec, price_map)
        judge_adapter = build_adapter(judge_spec)
        judge_cost_fn = make_cost_fn(judge_price)
        console.print(f"Judge model: [cyan]{judge_spec.display}[/cyan]")
        scorers = build_default_registry(judge_adapter, judge_cost_fn)
    else:
        scorers = build_default_registry()

    console.print(f"\n[bold]Running[/bold] [cyan]{spec.display}[/cyan]")
    return await run_suite(
        suite=suite,
        adapter=adapter,
        scorers=scorers,
        cost_fn=cost_fn,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        temperature_override=args.temperature,
        show_progress=not args.no_progress,
    )


async def _amain_single(
    args: argparse.Namespace,
    suite: EvalSuite,
    price_map: dict[str, PriceEntry],
    console: Console,
) -> int:
    spec = parse_model_spec(args.model)

    refused = _preflight_budget(suite, spec, price_map, args.max_cost, args.force, console)
    if refused is not None:
        return refused

    report = await _run_one_model(spec, suite, args, price_map, console)

    results_path = save_report(report, args.results_dir)
    console.print(f"\nSaved results: [cyan]{results_path}[/cyan]")

    diff: RegressionDiff | None = None
    baseline_path = _baseline_path(args)
    if args.baseline is not None or (args.update_baseline and baseline_path.exists()):
        if baseline_path.exists():
            baseline = load_report(baseline_path)
            try:
                diff = compare_to_baseline(
                    report,
                    baseline,
                    allow_dataset_mismatch=args.allow_dataset_mismatch,
                )
            except DatasetVersionMismatch as exc:
                console.print(f"[bold red]{exc}[/bold red]")
                return EXIT_DATASET_MISMATCH
        else:
            console.print(
                f"[yellow]Baseline {baseline_path} not found; skipping comparison.[/yellow]"
            )

    report_path = args.reports_dir / f"{report.run_id}.html"
    render_report(report, diff, report_path)
    console.print(f"Saved report:  [cyan]{report_path}[/cyan]")

    _print_summary(report, diff, console)
    if diff is not None:
        _print_regression_verdict(diff, console)

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(results_path.read_text(encoding="utf-8"), encoding="utf-8")
        console.print(f"[bold]Updated baseline:[/bold] [cyan]{baseline_path}[/cyan]")

    if diff is not None and diff.has_regression:
        return EXIT_REGRESSION
    return EXIT_OK


async def _amain_compare(
    args: argparse.Namespace,
    suite: EvalSuite,
    price_map: dict[str, PriceEntry],
    console: Console,
) -> int:
    specs = _parse_compare_specs(args.compare)
    console.print(f"Comparing [bold]{len(specs)}[/bold] models on suite [cyan]{suite.name}[/cyan]")

    for spec in specs:
        refused = _preflight_budget(suite, spec, price_map, args.max_cost, args.force, console)
        if refused is not None:
            return refused

    if args.baseline is not None or args.update_baseline:
        console.print(
            "[yellow]Note: --baseline / --update-baseline are ignored in --compare mode.[/yellow]"
        )

    reports: list[RunReport] = []
    for spec in specs:
        report = await _run_one_model(spec, suite, args, price_map, console)
        save_report(report, args.results_dir)
        reports.append(report)

    comparison = compare_reports(
        reports,
        labels=[s.display for s in specs],
        allow_dataset_mismatch=args.allow_dataset_mismatch,
    )

    comparison_id = reports[0].run_id
    report_path = args.reports_dir / f"{comparison_id}_compare.html"
    render_comparison(comparison, report_path)
    console.print(f"\nSaved comparison report: [cyan]{report_path}[/cyan]")

    _print_comparison_summary(comparison, console)
    return EXIT_OK


async def _amain(args: argparse.Namespace) -> int:
    console = Console()
    suite_path = args.evals_dir / f"{args.suite}.yaml"
    suite = load_suite(suite_path, known_scorers=set(DEFAULT_SCORER_NAMES))
    price_map = load_price_map()

    if args.compare:
        return await _amain_compare(args, suite, price_map, console)
    return await _amain_single(args, suite, price_map, console)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
