"""OpenAI provider adapter — async, with timeout, retry, and metadata capture.

Mirrors the shape of `AnthropicAdapter` so the runner stays provider-agnostic.
Defaults `temperature=0`. Per-request timeout. Exponential backoff with jitter
on `RateLimitError` / `APIConnectionError` / `APITimeoutError` /
`InternalServerError`; max 3 attempts. Captures `response.model` as
`resolved_model_id` so a stored result names the exact model the provider
executed.
"""

import asyncio
import os
import random
import time

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from harness.adapters.base import AdapterResponse

_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class OpenAIAdapter:
    """Async OpenAI client wrapper with retry/timeout/metadata capture."""

    name = "openai"

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY not set. Pass api_key= or export it.")
        self._client = AsyncOpenAI(api_key=resolved_key)
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
                response = await self._client.chat.completions.create(
                    model=self._model_id,
                    max_completion_tokens=max_tokens_resolved,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    timeout=timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                return AdapterResponse(
                    text=_extract_text(response),
                    input_tokens=_input_tokens(response),
                    output_tokens=_output_tokens(response),
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


def _extract_text(response: ChatCompletion) -> str:
    """Take the first choice's message content. Empty string if absent."""
    if not response.choices:
        return ""
    content = response.choices[0].message.content
    return content if content is not None else ""


def _input_tokens(response: ChatCompletion) -> int:
    return response.usage.prompt_tokens if response.usage is not None else 0


def _output_tokens(response: ChatCompletion) -> int:
    return response.usage.completion_tokens if response.usage is not None else 0


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter. `attempt` is 0-based."""
    base: float = 0.5 * (2**attempt)
    jitter: float = random.uniform(0.0, base / 2.0)
    return base + jitter
