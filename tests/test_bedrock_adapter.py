"""Tests for `harness.adapters.bedrock` — retry/timeout/metadata capture.

The boto3 client is fully mocked: no AWS calls, no real network. We replace
`adapter._client` with a MagicMock so `invoke_model` returns scripted
payloads or raises scripted exceptions.
"""

import json
from collections.abc import Callable
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from harness.adapters.bedrock import BedrockAdapter


def _payload(text: str = "ok", model: str = "anthropic.claude-haiku-4-5-v1:0") -> dict[str, Any]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "stop_reason": "end_turn",
    }


def _invoke_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Shape `invoke_model` returns: a dict with a `body` StreamingBody."""
    return {"body": BytesIO(json.dumps(payload).encode("utf-8"))}


def _throttling_error() -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        operation_name="InvokeModel",
    )


def _validation_error() -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": "bad input"}},
        operation_name="InvokeModel",
    )


@pytest.fixture
def make_adapter(monkeypatch: pytest.MonkeyPatch) -> Callable[..., BedrockAdapter]:
    """Construct a BedrockAdapter without ever creating a real boto3 client."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    def _make(max_attempts: int = 3) -> BedrockAdapter:
        with patch("harness.adapters.bedrock.boto3.client", return_value=MagicMock()):
            return BedrockAdapter(
                model_id="anthropic.claude-haiku-4-5-v1:0", max_attempts=max_attempts
            )

    return _make


async def test_complete_returns_text_and_metadata(
    make_adapter: Callable[..., BedrockAdapter],
) -> None:
    adapter = make_adapter()
    adapter._client.invoke_model.return_value = _invoke_response(_payload(text="hello world"))
    resp = await adapter.complete(system="s", user="u")
    assert resp.text == "hello world"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.resolved_model_id == "anthropic.claude-haiku-4-5-v1:0"
    assert resp.latency_ms >= 0.0


async def test_invoke_model_called_with_anthropic_schema(
    make_adapter: Callable[..., BedrockAdapter],
) -> None:
    """Bedrock-Claude requires `anthropic_version` and a messages array."""
    adapter = make_adapter()
    adapter._client.invoke_model.return_value = _invoke_response(_payload())
    await adapter.complete(system="be terse", user="hi", temperature=0.0, max_tokens=42)
    kwargs = adapter._client.invoke_model.call_args.kwargs
    assert kwargs["modelId"] == "anthropic.claude-haiku-4-5-v1:0"
    body = json.loads(kwargs["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 42
    assert body["temperature"] == 0.0
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_retries_on_throttling_then_succeeds(
    make_adapter: Callable[..., BedrockAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.bedrock._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter()
    adapter._client.invoke_model.side_effect = [
        _throttling_error(),
        _invoke_response(_payload(text="ok-after-retry")),
    ]
    resp = await adapter.complete(system="s", user="u")
    assert resp.text == "ok-after-retry"
    assert adapter._client.invoke_model.call_count == 2


async def test_non_retryable_client_error_raises_immediately(
    make_adapter: Callable[..., BedrockAdapter],
) -> None:
    """ValidationException isn't a transient — it should not be retried."""
    adapter = make_adapter()
    adapter._client.invoke_model.side_effect = _validation_error()
    with pytest.raises(ClientError):
        await adapter.complete(system="s", user="u")
    assert adapter._client.invoke_model.call_count == 1


async def test_retry_exhaustion_raises_terminal_error(
    make_adapter: Callable[..., BedrockAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.bedrock._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter(max_attempts=3)
    adapter._client.invoke_model.side_effect = [_throttling_error()] * 3
    with pytest.raises(ClientError):
        await adapter.complete(system="s", user="u")
    assert adapter._client.invoke_model.call_count == 3


async def test_timeout_is_retried(
    make_adapter: Callable[..., BedrockAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """asyncio.wait_for raises TimeoutError on overrun — runner should retry."""
    monkeypatch.setattr("harness.adapters.bedrock._backoff_delay", lambda _i: 0.0)

    async def fail_then_succeed() -> dict[str, Any]:
        if fail_then_succeed.calls == 0:  # type: ignore[attr-defined]
            fail_then_succeed.calls += 1  # type: ignore[attr-defined]
            raise TimeoutError("timed out")
        return _invoke_response(_payload(text="ok"))

    fail_then_succeed.calls = 0  # type: ignore[attr-defined]
    adapter = make_adapter()

    async def fake_wait_for(coro: Any, timeout: float) -> Any:
        # Close the to_thread coroutine without awaiting it so it doesn't warn.
        coro.close()
        return await fail_then_succeed()

    monkeypatch.setattr("harness.adapters.bedrock.asyncio.wait_for", fake_wait_for)
    resp = await adapter.complete(system="s", user="u", timeout=0.001)
    assert resp.text == "ok"


def test_init_uses_default_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no env var is set, default to us-east-1."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    with patch("harness.adapters.bedrock.boto3.client") as mock_client:
        BedrockAdapter(model_id="anthropic.claude-haiku-4-5-v1:0")
    mock_client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")


def test_init_respects_explicit_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    with patch("harness.adapters.bedrock.boto3.client") as mock_client:
        BedrockAdapter(model_id="anthropic.claude-haiku-4-5-v1:0", region="eu-west-2")
    mock_client.assert_called_once_with("bedrock-runtime", region_name="eu-west-2")
