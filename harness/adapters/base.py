"""ModelAdapter Protocol + AdapterResponse value object.

Every provider (Anthropic — Phase 2, OpenAI/Bedrock — Phase 7) implements
`ModelAdapter`. The runner only ever holds a `ModelAdapter`, never a concrete
type, so swapping providers is a constructor argument.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AdapterResponse:
    """One model call's worth of measurable output.

    `resolved_model_id` is what the provider actually executed against, after
    any alias resolution (e.g. `claude-opus-4-latest` → a dated ID). This flows
    through to `EvalResult.model` so a stored result names the exact model.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    resolved_model_id: str


class ModelAdapter(Protocol):
    """Pluggable provider client. Implementations live in `harness/adapters/`.

    `complete` is the entire surface area. Adapters own their own retry,
    timeout, and cost-instrumentation policy; the runner stays provider-agnostic.
    """

    name: str

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> AdapterResponse: ...
