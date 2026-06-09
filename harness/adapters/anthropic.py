"""Anthropic provider adapter — async, with timeout, retry, and metadata capture.

Defaults `temperature=0`. Per-request timeout. Exponential backoff with jitter
on `RateLimitError` / `APIConnectionError` / `APITimeoutError` /
`InternalServerError`; max 3 attempts. Captures `resolved_model_id` from the
response so a stored result names the exact model the provider executed.

Pricing is intentionally NOT here — `harness.budget.make_cost_fn(price)`
returns the cost callable that the runner wires in. Adapters stay
provider-specific without leaking pricing.
"""

import asyncio
import os
import random
import time

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message

from harness.adapters.base import AdapterResponse

_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


class AnthropicAdapter:
    """Async Anthropic client wrapper with retry/timeout/metadata capture."""

    name = "anthropic"

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Pass api_key= or export it.")
        self._client = AsyncAnthropic(api_key=resolved_key)
        self._model_id = model_id
        self._max_attempts = max_attempts

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> AdapterResponse:
        max_tokens_resolved = max_tokens if max_tokens is not None else 1024
        last_exc: Exception | None = None

        for attempt in range(self._max_attempts):
            try:
                start = time.perf_counter()
                response = await self._client.messages.create(
                    model=self._model_id,
                    max_tokens=max_tokens_resolved,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    timeout=timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                return AdapterResponse(
                    text=_extract_text(response),
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=latency_ms,
                    resolved_model_id=response.model,
                )
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt + 1 >= self._max_attempts:
                    break
                await asyncio.sleep(_backoff_delay(attempt))

        assert last_exc is not None
        raise last_exc


def _extract_text(response: Message) -> str:
    """Concatenate text from a response's content blocks."""
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter. `attempt` is 0-based."""
    base: float = 0.5 * (2**attempt)
    jitter: float = random.uniform(0.0, base / 2.0)
    return base + jitter
