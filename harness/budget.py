"""Pre-flight cost estimate for the CLI `--max-cost` guardrail.

Also exposes `make_cost_fn(price)` which the runner uses to record per-case
cost. Keeping pricing here (rather than on the adapter) means adapters stay
provider-specific without leaking pricing concerns into the Protocol.
"""

from collections.abc import Callable
from dataclasses import dataclass

from harness.config import PriceEntry
from harness.models import EvalSuite

# Conservative defaults — actual token counts vary. The guardrail is the point,
# not the accuracy of the estimate.
DEFAULT_INPUT_TOKENS_PER_CASE = 1000
DEFAULT_OUTPUT_TOKENS_PER_CASE = 500


CostFn = Callable[[int, int], float]


@dataclass(frozen=True)
class CostEstimate:
    """Pre-flight estimate of total suite cost in USD."""

    estimated_usd: float
    model_id: str
    case_count: int


def estimate_cost(
    suite: EvalSuite,
    price_map: dict[str, PriceEntry],
    model_id: str,
    input_tokens_per_case: int = DEFAULT_INPUT_TOKENS_PER_CASE,
    output_tokens_per_case: int = DEFAULT_OUTPUT_TOKENS_PER_CASE,
) -> CostEstimate:
    """Estimate total cost if every case used the default token budget."""
    if model_id not in price_map:
        raise ValueError(
            f"model_id '{model_id}' has no entry in price map; "
            "add it to harness/prices.yaml before running."
        )
    entry = price_map[model_id]
    cases = len(suite.cases)
    input_cost = (input_tokens_per_case * cases / 1_000_000.0) * entry.input_per_million
    output_cost = (output_tokens_per_case * cases / 1_000_000.0) * entry.output_per_million
    return CostEstimate(
        estimated_usd=input_cost + output_cost,
        model_id=model_id,
        case_count=cases,
    )


def make_cost_fn(price: PriceEntry) -> CostFn:
    """Closure that computes USD cost from `(input_tokens, output_tokens)`."""

    def cost(input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * price.input_per_million / 1_000_000.0
            + output_tokens * price.output_per_million / 1_000_000.0
        )

    return cost
