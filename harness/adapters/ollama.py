"""Ollama provider adapter — async, talks to a local Ollama server over HTTP.

Targets locally-hosted open-weights models (Llama, Qwen, Mistral, Gemma,
gpt-oss, etc.) served by Ollama on ``http://localhost:11434`` by default.
Implements the same ``ModelAdapter`` Protocol as the cloud adapters so the
runner, regression detector, and ``--compare`` flow are provider-agnostic.

We talk to ``POST /api/chat`` with ``stream: false``; a single JSON response
carries ``message.content`` plus ``prompt_eval_count`` / ``eval_count`` for
token accounting. Wall-clock latency is measured locally because Ollama
doesn't expose it directly.

Retries on connection / read errors and 5xx with exponential backoff and
jitter; max 3 attempts. Captures ``response.model`` as ``resolved_model_id``
— Ollama echoes back the exact tag it served (e.g. ``llama3.2:latest``).

No external SDK: we use ``httpx`` (already a transitive dep via anthropic /
openai), which keeps the dependency footprint minimal.
"""

import asyncio
import os
import random
import time
from typing import Any

import httpx

from harness.adapters.base import AdapterResponse

DEFAULT_BASE_URL = "http://localhost:11434"

_RETRYABLE_HTTPX_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


class OllamaAdapter:
    """Async Ollama HTTP-client wrapper with retry/timeout/metadata capture."""

    name = "ollama"

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        resolved_base = base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_BASE_URL
        self._base_url = resolved_base.rstrip("/")
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
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens if max_tokens is not None else 1024,
            },
        }

        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                start = time.perf_counter()
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/api/chat",
                        json=payload,
                    )
                if response.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"ollama returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                else:
                    response.raise_for_status()
                    body: dict[str, Any] = response.json()
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    return AdapterResponse(
                        text=_extract_text(body),
                        input_tokens=int(body.get("prompt_eval_count", 0)),
                        output_tokens=int(body.get("eval_count", 0)),
                        latency_ms=latency_ms,
                        resolved_model_id=str(body.get("model", self._model_id)),
                    )
            except _RETRYABLE_HTTPX_ERRORS as exc:
                last_exc = exc

            if attempt + 1 >= self._max_attempts:
                break
            await asyncio.sleep(_backoff_delay(attempt))

        assert last_exc is not None
        raise last_exc


def _extract_text(body: dict[str, Any]) -> str:
    """Pull message.content out of an Ollama /api/chat response. Empty if absent."""
    message = body.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return str(content) if content is not None else ""


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter. `attempt` is 0-based."""
    base: float = 0.5 * (2**attempt)
    jitter: float = random.uniform(0.0, base / 2.0)
    return base + jitter
