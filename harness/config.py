"""Load YAML eval suites and the model price map into validated objects.

`load_suite(path)` returns a fully-resolved `EvalSuite`: every case has its
`system` and `user` prompts filled in (case-level override beats suite
default), `dataset_version` is required, and per-scorer invariants are
checked. The runner can trust whatever it receives.
"""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from harness.models import EvalCase, EvalSuite


@dataclass(frozen=True)
class PriceEntry:
    """Cost per million tokens for one resolved provider model ID."""

    input_per_million: float
    output_per_million: float


class SuiteValidationError(ValueError):
    """Raised when a YAML suite fails validation at load time."""


def load_suite(path: Path, known_scorers: set[str] | None = None) -> EvalSuite:
    """Load a YAML suite and return a fully-validated `EvalSuite`.

    Validates:
      - top-level `dataset_version` is present
      - every case has a resolvable `system` and `user` (case or suite default)
      - every case has `id`, `category`, and `scorer`
      - if `known_scorers` is provided, every `scorer` value is in that set
      - every `exact_match` case has non-empty `expected_contains`
      - every `regex_match` case has a non-empty `expected_regex`
    """
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if not isinstance(loaded, dict):
        raise SuiteValidationError(f"{path}: top level must be a YAML mapping")
    raw: dict[str, Any] = loaded

    if "dataset_version" not in raw:
        raise SuiteValidationError(f"{path}: missing required top-level `dataset_version`")
    dataset_version = str(raw["dataset_version"])

    name = str(raw.get("name") or path.stem)
    default_system_value = raw.get("default_system")
    default_user_value = raw.get("default_user")
    default_system = str(default_system_value) if default_system_value is not None else None
    default_user = str(default_user_value) if default_user_value is not None else None

    raw_cases = raw.get("cases", [])
    if not isinstance(raw_cases, list):
        raise SuiteValidationError(f"{path}: `cases` must be a list")

    cases: list[EvalCase] = []
    for i, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise SuiteValidationError(f"{path}: case at index {i} must be a YAML mapping")
        cases.append(_resolve_case(raw_case, default_system, default_user, path, i, known_scorers))

    return EvalSuite(
        name=name,
        dataset_version=dataset_version,
        default_system=default_system,
        default_user=default_user,
        cases=tuple(cases),
    )


def _resolve_case(
    raw: dict[str, Any],
    default_system: str | None,
    default_user: str | None,
    path: Path,
    index: int,
    known_scorers: set[str] | None,
) -> EvalCase:
    case_id_value = raw.get("id")
    if not case_id_value:
        raise SuiteValidationError(f"{path}: case at index {index} missing required `id`")
    case_id = str(case_id_value)

    user_value = raw.get("user", default_user)
    if user_value is None:
        raise SuiteValidationError(
            f"{path}: case '{case_id}' has no `user` and no suite-level `default_user` is set"
        )
    system_value = raw.get("system", default_system)
    if system_value is None:
        raise SuiteValidationError(
            f"{path}: case '{case_id}' has no `system` and no suite-level `default_system` is set"
        )

    category_value = raw.get("category")
    if not category_value:
        raise SuiteValidationError(f"{path}: case '{case_id}' missing required `category`")
    scorer_value = raw.get("scorer")
    if not scorer_value:
        raise SuiteValidationError(f"{path}: case '{case_id}' missing required `scorer`")
    scorer_name = str(scorer_value)

    if known_scorers is not None and scorer_name not in known_scorers:
        raise SuiteValidationError(
            f"{path}: case '{case_id}' uses unknown scorer '{scorer_name}'; "
            f"known scorers: {sorted(known_scorers)}"
        )

    expected_contains = [str(p) for p in raw.get("expected_contains") or []]
    if scorer_name == "exact_match" and not expected_contains:
        raise SuiteValidationError(
            f"{path}: case '{case_id}' uses `exact_match` but has empty "
            "`expected_contains`; a scorer with nothing to check is a "
            "misconfigured test."
        )

    expected_regex_value = raw.get("expected_regex")
    if scorer_name == "regex_match" and not expected_regex_value:
        raise SuiteValidationError(
            f"{path}: case '{case_id}' uses `regex_match` but has no `expected_regex`"
        )
    judge_rubric_value = raw.get("judge_rubric")
    temperature_value = raw.get("temperature")

    temperature: float | None = float(temperature_value) if temperature_value is not None else None

    return EvalCase(
        id=case_id,
        category=str(category_value),
        scorer=scorer_name,
        user=str(user_value),
        system=str(system_value),
        expected_contains=expected_contains,
        expected_regex=str(expected_regex_value) if expected_regex_value is not None else None,
        judge_rubric=str(judge_rubric_value) if judge_rubric_value is not None else None,
        temperature=temperature,
        tags=[str(t) for t in raw.get("tags") or []],
    )


def load_price_map() -> dict[str, PriceEntry]:
    """Load the model price map shipped inside the `harness` package."""
    ref = resources.files("harness").joinpath("prices.yaml")
    with ref.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError("prices.yaml: top level must be a YAML mapping")
    raw: dict[str, Any] = loaded
    out: dict[str, PriceEntry] = {}
    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"prices.yaml: entry for '{model_id}' must be a mapping")
        out[str(model_id)] = PriceEntry(
            input_per_million=float(entry["input_per_million"]),
            output_per_million=float(entry["output_per_million"]),
        )
    return out
