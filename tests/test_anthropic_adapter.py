"""Tests for `harness.adapters.anthropic` — retry/timeout/metadata capture."""

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from harness.adapters.anthropic import AnthropicAdapter


def _fake_message(text: str = "ok", model: str = "claude-opus-4-7-20260101") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    msg.model = model
    msg.usage = MagicMock(input_tokens=100, output_tokens=50)
    return msg


def _rate_limit_error() -> anthropic.RateLimitError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req)
    return anthropic.RateLimitError("rate limited", response=resp, body=None)


@pytest.fixture
def make_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., AnthropicAdapter]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def _make(max_attempts: int = 3) -> AnthropicAdapter:
        return AnthropicAdapter(model_id="claude-opus-4-7", max_attempts=max_attempts)

    return _make


async def test_complete_returns_text_and_metadata(
    make_adapter: Callable[..., AnthropicAdapter],
) -> None:
    adapter = make_adapter()
    fake = _fake_message(text="hello world", model="claude-opus-4-7-20260101")
    with patch.object(adapter._client.messages, "create", new=AsyncMock(return_value=fake)):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == "hello world"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.resolved_model_id == "claude-opus-4-7-20260101"
    assert resp.latency_ms >= 0.0


async def test_retries_then_succeeds(
    make_adapter: Callable[..., AnthropicAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.anthropic._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter()
    mock = AsyncMock(side_effect=[_rate_limit_error(), _fake_message(text="ok")])
    with patch.object(adapter._client.messages, "create", new=mock):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == "ok"
    assert mock.call_count == 2


async def test_retry_exhaustion_raises_terminal_error(
    make_adapter: Callable[..., AnthropicAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.anthropic._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter(max_attempts=3)
    mock = AsyncMock(side_effect=[_rate_limit_error()] * 3)
    with (
        patch.object(adapter._client.messages, "create", new=mock),
        pytest.raises(anthropic.RateLimitError),
    ):
        await adapter.complete(system="s", user="u")
    assert mock.call_count == 3


def test_init_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicAdapter(model_id="claude-opus-4-7")


def test_init_takes_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = AnthropicAdapter(model_id="claude-opus-4-7", api_key="explicit-key")
    assert adapter.name == "anthropic"
