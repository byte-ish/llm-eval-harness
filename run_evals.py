"""CLI entrypoint for the eval harness.

Usage:
  python run_evals.py --suite summarisation --model claude-opus-4-7

Exit codes:
  0 - success
  3 - pre-flight budget refused (use --force to bypass)
"""

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from harness import __version__
from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.base import ModelAdapter
from harness.budget import estimate_cost, make_cost_fn
from harness.config import PriceEntry, load_price_map, load_suite
from harness.models import RunReport
from harness.runner import run_suite
from harness.scorers import ExactMatchScorer
from harness.scorers.base import Scorer
from harness.store import save

EXIT_OK = 0
EXIT_BUDGET_REFUSED = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_evals",
        description="Run an eval suite against an LLM and report results.",
    )
    parser.add_argument(
        "--suite", required=True, help="Suite name under --evals-dir (e.g. summarisation)"
    )
    parser.add_argument(
        "--model", required=True, help="Resolved provider model ID (e.g. claude-opus-4-7)"
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
        help="Refuse to run if the pre-flight estimate exceeds this USD amount",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if --max-cost would refuse",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"))
    parser.add_argument("--no-progress", action="store_true", help="Hide the progress bar")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def _build_scorer_registry() -> dict[str, Scorer]:
    return {"exact_match": ExactMatchScorer()}


def _build_adapter(model_id: str) -> ModelAdapter:
    return AnthropicAdapter(model_id=model_id)


def _print_summary(report: RunReport, console: Console) -> None:
    table = Table(title=f"Results: {report.suite} (dataset {report.dataset_version})")
    table.add_column("Category")
    table.add_column("Pass rate", justify="right")
    table.add_column("Cases", justify="right")

    counts: dict[str, int] = {}
    for r in report.results:
        counts[r.category] = counts.get(r.category, 0) + 1
    for category, rate in sorted(report.pass_rate_by_category.items()):
        table.add_row(category, f"{rate * 100:.1f}%", str(counts[category]))

    table.add_section()
    table.add_row(
        "[bold]Overall",
        f"[bold]{report.pass_rate * 100:.1f}%",
        f"[bold]{len(report.results)}",
    )
    console.print(table)

    summary = Table(show_header=False, box=None)
    summary.add_row("Total cost USD", f"${report.total_cost_usd:.4f}")
    summary.add_row("p50 latency ms", f"{report.p50_latency_ms:.0f}")
    summary.add_row("p95 latency ms", f"{report.p95_latency_ms:.0f}")
    summary.add_row("Resolved model", report.model)
    summary.add_row("Run ID", report.run_id)
    console.print(summary)


def _validate_model_in_price_map(model_id: str, price_map: dict[str, PriceEntry]) -> PriceEntry:
    if model_id not in price_map:
        raise SystemExit(
            f"model_id '{model_id}' has no entry in price map; "
            "add it to harness/prices.yaml before running."
        )
    return price_map[model_id]


async def _amain(args: argparse.Namespace) -> int:
    console = Console()
    suite_path = args.evals_dir / f"{args.suite}.yaml"
    suite = load_suite(suite_path)
    price_map = load_price_map()
    price = _validate_model_in_price_map(args.model, price_map)

    if args.max_cost is not None:
        estimate = estimate_cost(suite, price_map, args.model)
        if estimate.estimated_usd > args.max_cost and not args.force:
            console.print(
                f"[red]Pre-flight estimate ${estimate.estimated_usd:.4f} exceeds "
                f"--max-cost ${args.max_cost:.4f}. Use --force to run anyway.[/red]"
            )
            return EXIT_BUDGET_REFUSED
        console.print(
            f"Pre-flight estimate: ${estimate.estimated_usd:.4f} for "
            f"{estimate.case_count} case(s) on {estimate.model_id}"
        )

    adapter = _build_adapter(args.model)
    scorers = _build_scorer_registry()
    cost_fn = make_cost_fn(price)

    report = await run_suite(
        suite=suite,
        adapter=adapter,
        scorers=scorers,
        cost_fn=cost_fn,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        temperature_override=args.temperature,
        show_progress=not args.no_progress,
    )

    path = save(report, args.results_dir)
    console.print(f"\nSaved: [cyan]{path}[/cyan]")
    _print_summary(report, console)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
