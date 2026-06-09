"""Filesystem store for `RunReport` JSON persistence.

`save(report, dir)` writes `results/<run_id>.json`. `load(path)` reconstructs
a `RunReport`. Frozen dataclasses + nested types make raw `asdict`/`**dict`
round-trip awkward, so reconstruction is explicit.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from harness.models import EvalResult, RunReport, ScorerResult


def save(report: RunReport, results_dir: Path) -> Path:
    """Serialise a `RunReport` to `results_dir/<run_id>.json`. Returns the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{report.run_id}.json"
    payload = asdict(report)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def load(path: Path) -> RunReport:
    """Read a serialised `RunReport` back from disk."""
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    raw: dict[str, Any] = loaded

    results = [_result_from_dict(r) for r in raw["results"]]

    return RunReport(
        run_id=str(raw["run_id"]),
        model=str(raw["model"]),
        suite=str(raw["suite"]),
        dataset_version=str(raw["dataset_version"]),
        results=results,
        started_at=str(raw["started_at"]),
        finished_at=str(raw["finished_at"]),
        harness_version=str(raw["harness_version"]),
    )


def _result_from_dict(raw: dict[str, Any]) -> EvalResult:
    scorer_raw: dict[str, Any] = raw["scorer_result"]
    return EvalResult(
        case_id=str(raw["case_id"]),
        category=str(raw["category"]),
        response=str(raw["response"]),
        scorer_result=ScorerResult(
            passed=bool(scorer_raw["passed"]),
            score=float(scorer_raw["score"]),
            reason=str(scorer_raw["reason"]),
        ),
        latency_ms=float(raw["latency_ms"]),
        input_tokens=int(raw["input_tokens"]),
        output_tokens=int(raw["output_tokens"]),
        cost_usd=float(raw["cost_usd"]),
        model=str(raw["model"]),
        temperature=float(raw["temperature"]),
        error=str(raw["error"]) if raw.get("error") is not None else None,
    )
