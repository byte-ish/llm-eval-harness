"""Provider-prefix routing — turn a `provider:model_id` string into a `ModelAdapter`.

Phase 7 introduces multi-provider support. To name a model unambiguously,
the CLI accepts a `provider:model_id` syntax:

  - ``anthropic:claude-haiku-4-5-20251001``  → ``AnthropicAdapter``
  - ``openai:gpt-4o-mini``                   → ``OpenAIAdapter``
  - ``bedrock:anthropic.claude-haiku-4-5-v1:0`` → ``BedrockAdapter``

A bare model ID (no prefix) defaults to anthropic — preserves backward
compatibility with Phase 6 invocations like ``--model claude-haiku-...``.

Note: the model ID itself can contain colons (Bedrock IDs are
``provider.family-version:revision``). We split on the **first** colon only.
"""

from dataclasses import dataclass

from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.base import ModelAdapter
from harness.adapters.bedrock import BedrockAdapter
from harness.adapters.openai import OpenAIAdapter

DEFAULT_PROVIDER = "anthropic"
KNOWN_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai", "bedrock"})


@dataclass(frozen=True)
class ModelSpec:
    """A parsed model specifier.

    `provider` selects the adapter class; `model_id` is the bare provider
    model ID used by both the SDK call and the price-map lookup.
    """

    provider: str
    model_id: str

    @property
    def display(self) -> str:
        return f"{self.provider}:{self.model_id}"


def parse_model_spec(spec: str) -> ModelSpec:
    """Parse a ``provider:model_id`` or bare-model-id string.

    Splits on the first colon. Unknown providers raise — a typo in a
    provider name should fail loudly, not silently fall through to the
    default.
    """
    if ":" not in spec:
        return ModelSpec(provider=DEFAULT_PROVIDER, model_id=spec)
    provider, _, model_id = spec.partition(":")
    if provider not in KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r} in model spec {spec!r}; "
            f"known providers: {sorted(KNOWN_PROVIDERS)}"
        )
    if not model_id:
        raise ValueError(f"model spec {spec!r} has an empty model_id after {provider!r}:")
    return ModelSpec(provider=provider, model_id=model_id)


def build_adapter(spec: ModelSpec) -> ModelAdapter:
    """Construct the right `ModelAdapter` for the spec's provider."""
    if spec.provider == "anthropic":
        return AnthropicAdapter(model_id=spec.model_id)
    if spec.provider == "openai":
        return OpenAIAdapter(model_id=spec.model_id)
    if spec.provider == "bedrock":
        return BedrockAdapter(model_id=spec.model_id)
    raise ValueError(f"no adapter registered for provider {spec.provider!r}")
