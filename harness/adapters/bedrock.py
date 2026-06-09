"""AWS Bedrock provider adapter — async wrapper around boto3's sync client.

Targets Claude models hosted on Bedrock, which use the Anthropic messages
schema (`anthropic_version: bedrock-2023-05-31`). Other Bedrock model
families have different request/response shapes; supporting them is out of
scope for Phase 7.

`boto3.client("bedrock-runtime")` is synchronous, so each `invoke_model` call
is wrapped in `asyncio.to_thread()` and bounded by `asyncio.wait_for()` for
the per-request timeout. Retries on throttle / 5xx / connection / timeout
errors with exponential backoff and jitter; max 3 attempts. Captures
`resolved_model_id` from the response payload (Claude-on-Bedrock echoes its
model ID).
"""

import asyncio
import json
import os
import random
import time
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from harness.adapters.base import AdapterResponse

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient

# Bedrock error codes that are worth retrying.
_RETRYABLE_BOTO_CODES = frozenset(
    {
        "ThrottlingException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "InternalServerException",
    }
)


class BedrockAdapter:
    """Async wrapper over `bedrock-runtime` for Claude-on-Bedrock models."""

    name = "bedrock"

    def __init__(
        self,
        model_id: str,
        region: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        resolved_region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self._client: BedrockRuntimeClient = boto3.client(
            "bedrock-runtime", region_name=resolved_region
        )
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
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens if max_tokens is not None else 1024,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                start = time.perf_counter()
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._client.invoke_model,
                        modelId=self._model_id,
                        body=body,
                        contentType="application/json",
                        accept="application/json",
                    ),
                    timeout=timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                payload = json.loads(response["body"].read())
                return AdapterResponse(
                    text=_extract_text(payload),
                    input_tokens=int(payload["usage"]["input_tokens"]),
                    output_tokens=int(payload["usage"]["output_tokens"]),
                    latency_ms=latency_ms,
                    resolved_model_id=str(payload.get("model", self._model_id)),
                )
            except ClientError as exc:
                if not _is_retryable_client_error(exc):
                    raise
                last_exc = exc
            except (BotoCoreError, TimeoutError) as exc:
                last_exc = exc

            if attempt + 1 >= self._max_attempts:
                break
            await asyncio.sleep(_backoff_delay(attempt))

        assert last_exc is not None
        raise last_exc


def _extract_text(payload: dict[str, Any]) -> str:
    """Concatenate text from a Claude-on-Bedrock response's `content` blocks."""
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _is_retryable_client_error(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code", "")
    return code in _RETRYABLE_BOTO_CODES


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter. `attempt` is 0-based."""
    base: float = 0.5 * (2**attempt)
    jitter: float = random.uniform(0.0, base / 2.0)
    return base + jitter
