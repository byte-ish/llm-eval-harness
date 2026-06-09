"""Tests for `harness.adapters.openai` — retry/timeout/metadata capture."""

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from harness.adapters.openai import OpenAIAdapter


def _fake_completion(text: str = "ok", model: str = "gpt-4o-mini-2024-07-18") -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.model = model
    completion.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return completion


def _rate_limit_error() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return openai.RateLimitError("rate limited", response=resp, body=None)


@pytest.fixture
def make_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., OpenAIAdapter]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _make(max_attempts: int = 3) -> OpenAIAdapter:
        return OpenAIAdapter(model_id="gpt-4o-mini", max_attempts=max_attempts)

    return _make


async def test_complete_returns_text_and_metadata(
    make_adapter: Callable[..., OpenAIAdapter],
) -> None:
    adapter = make_adapter()
    fake = _fake_completion(text="hello world", model="gpt-4o-mini-2024-07-18")
    with patch.object(adapter._client.chat.completions, "create", new=AsyncMock(return_value=fake)):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == "hello world"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.resolved_model_id == "gpt-4o-mini-2024-07-18"
    assert resp.latency_ms >= 0.0


async def test_retries_then_succeeds(
    make_adapter: Callable[..., OpenAIAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.openai._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter()
    mock = AsyncMock(side_effect=[_rate_limit_error(), _fake_completion(text="ok")])
    with patch.object(adapter._client.chat.completions, "create", new=mock):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == "ok"
    assert mock.call_count == 2


async def test_retry_exhaustion_raises_terminal_error(
    make_adapter: Callable[..., OpenAIAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.openai._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter(max_attempts=3)
    mock = AsyncMock(side_effect=[_rate_limit_error()] * 3)
    with (
        patch.object(adapter._client.chat.completions, "create", new=mock),
        pytest.raises(openai.RateLimitError),
    ):
        await adapter.complete(system="s", user="u")
    assert mock.call_count == 3


async def test_empty_choices_returns_empty_text(
    make_adapter: Callable[..., OpenAIAdapter],
) -> None:
    """A response with no choices should not crash — return empty text."""
    adapter = make_adapter()
    completion = MagicMock()
    completion.choices = []
    completion.model = "gpt-4o-mini-2024-07-18"
    completion.usage = MagicMock(prompt_tokens=10, completion_tokens=0)
    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(return_value=completion)
    ):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == ""
    assert resp.input_tokens == 10


async def test_none_content_returns_empty_text(
    make_adapter: Callable[..., OpenAIAdapter],
) -> None:
    """A choice with None content (e.g. tool call response) should return empty string."""
    adapter = make_adapter()
    completion = _fake_completion()
    completion.choices[0].message.content = None
    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(return_value=completion)
    ):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == ""


def test_init_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIAdapter(model_id="gpt-4o-mini")


def test_init_takes_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIAdapter(model_id="gpt-4o-mini", api_key="explicit-key")
    assert adapter.name == "openai"
