"""Async runner — dispatches cases through the adapter and scorer registry.

Concurrency is bounded by an `asyncio.Semaphore`. Each case runs inside a
per-case `try/except Exception` so one failure (adapter error, scorer crash,
unknown scorer name) records on `EvalResult.error` and the rest of the suite
runs to completion.
"""

import asyncio
from datetime import UTC, datetime

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from harness import __version__
from harness.adapters.base import ModelAdapter
from harness.budget import CostFn
from harness.models import (
    EvalCase,
    EvalResult,
    EvalSuite,
    RunReport,
    ScorerResult,
)
from harness.scorers.base import Scorer


async def run_suite(
    suite: EvalSuite,
    adapter: ModelAdapter,
    scorers: dict[str, Scorer],
    cost_fn: CostFn,
    *,
    concurrency: int = 4,
    timeout_seconds: float = 60.0,
    temperature_override: float | None = None,
    show_progress: bool = True,
) -> RunReport:
    """Run an `EvalSuite` end-to-end and return a `RunReport`.

    `cost_fn` lets the runner stay provider-agnostic — the caller passes in
    a pricing closure (`harness.budget.make_cost_fn`) keyed to the model.
    """
    started_at = datetime.now(UTC).isoformat()
    semaphore = asyncio.Semaphore(concurrency)
    resolved_model_ids: list[str] = []

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
    )

    async def run_one(case: EvalCase, task_id: TaskID) -> EvalResult:
        async with semaphore:
            temperature = _resolve_temperature(case, temperature_override)
            assert case.system is not None and case.user is not None  # guaranteed by config
            try:
                response = await adapter.complete(
                    system=case.system,
                    user=case.user,
                    temperature=temperature,
                    timeout=timeout_seconds,
                )
            except Exception as exc:
                progress.advance(task_id)
                return EvalResult(
                    case_id=case.id,
                    category=case.category,
                    response="",
                    scorer_result=ScorerResult(
                        passed=False,
                        score=0.0,
                        reason=f"adapter error: {exc!r}",
                    ),
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    model="",
                    temperature=temperature,
                    error=str(exc),
                )

            resolved_model_ids.append(response.resolved_model_id)
            scorer_result, error = await _score(case, response.text, scorers)
            adapter_cost = cost_fn(response.input_tokens, response.output_tokens)
            total_cost = adapter_cost + scorer_result.cost_usd

            progress.advance(task_id)
            return EvalResult(
                case_id=case.id,
                category=case.category,
                response=response.text,
                scorer_result=scorer_result,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=total_cost,
                model=response.resolved_model_id,
                temperature=temperature,
                error=error,
            )

    with progress:
        task_id = progress.add_task("Running cases", total=len(suite.cases))
        results = list(await asyncio.gather(*(run_one(c, task_id) for c in suite.cases)))

    finished_at = datetime.now(UTC).isoformat()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S%f")
    final_model = resolved_model_ids[0] if resolved_model_ids else ""

    return RunReport(
        run_id=run_id,
        model=final_model,
        suite=suite.name,
        dataset_version=suite.dataset_version,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
        harness_version=__version__,
    )


def _resolve_temperature(case: EvalCase, override: float | None) -> float:
    if override is not None:
        return override
    if case.temperature is not None:
        return case.temperature
    return 0.0


async def _score(
    case: EvalCase,
    response_text: str,
    scorers: dict[str, Scorer],
) -> tuple[ScorerResult, str | None]:
    scorer = scorers.get(case.scorer)
    if scorer is None:
        msg = f"unknown scorer '{case.scorer}'"
        return ScorerResult(passed=False, score=0.0, reason=msg), msg
    try:
        return await scorer.score(response_text, case), None
    except Exception as exc:
        return (
            ScorerResult(passed=False, score=0.0, reason=f"scorer error: {exc!r}"),
            str(exc),
        )
