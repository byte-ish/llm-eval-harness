"""Tests for `harness.adapters.ollama` — retry/timeout/metadata capture.

All tests are offline: we monkey-patch `httpx.AsyncClient` so no real socket
ever opens. Same convention as the other adapter tests.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from harness.adapters.ollama import DEFAULT_BASE_URL, OllamaAdapter


def _fake_response(
    body: dict[str, Any] | None = None,
    status_code: int = 200,
) -> MagicMock:
    """Build a fake httpx.Response-shaped MagicMock."""
    resolved_body: dict[str, Any] = (
        body
        if body is not None
        else {
            "model": "llama3.2:latest",
            "message": {"role": "assistant", "content": "hello world"},
            "prompt_eval_count": 42,
            "eval_count": 17,
            "done": True,
        }
    )
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json = MagicMock(return_value=resolved_body)
    response.raise_for_status = MagicMock()
    response.request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return response


def _patch_async_client(post_mock: AsyncMock) -> Any:
    """Build a context-manager patch over `httpx.AsyncClient` whose `post` is the given mock."""
    client_instance = MagicMock()
    client_instance.post = post_mock
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=None)
    return patch("harness.adapters.ollama.httpx.AsyncClient", return_value=client_instance)


@pytest.fixture
def make_adapter() -> Callable[..., OllamaAdapter]:
    def _make(max_attempts: int = 3, base_url: str | None = None) -> OllamaAdapter:
        return OllamaAdapter(
            model_id="llama3.2",
            base_url=base_url,
            max_attempts=max_attempts,
        )

    return _make


async def test_complete_returns_text_and_metadata(
    make_adapter: Callable[..., OllamaAdapter],
) -> None:
    adapter = make_adapter()
    post = AsyncMock(return_value=_fake_response())
    with _patch_async_client(post):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == "hello world"
    assert resp.input_tokens == 42
    assert resp.output_tokens == 17
    assert resp.resolved_model_id == "llama3.2:latest"
    assert resp.latency_ms >= 0.0


async def test_post_body_shape(make_adapter: Callable[..., OllamaAdapter]) -> None:
    """The request must use /api/chat, stream=False, and pass temperature + num_predict."""
    adapter = make_adapter()
    post = AsyncMock(return_value=_fake_response())
    with _patch_async_client(post):
        await adapter.complete(system="sys", user="usr", temperature=0.3, max_tokens=256)

    assert post.call_count == 1
    args, kwargs = post.call_args
    assert args[0] == "http://localhost:11434/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert payload["options"] == {"temperature": 0.3, "num_predict": 256}


async def test_retries_then_succeeds(
    make_adapter: Callable[..., OllamaAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.ollama._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter()
    post = AsyncMock(
        side_effect=[
            httpx.ConnectError("boom"),
            _fake_response(),
        ]
    )
    with _patch_async_client(post):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == "hello world"
    assert post.call_count == 2


async def test_retries_on_5xx(
    make_adapter: Callable[..., OllamaAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama returning 500 should be retried as a transient error."""
    monkeypatch.setattr("harness.adapters.ollama._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter()
    post = AsyncMock(
        side_effect=[
            _fake_response(status_code=503),
            _fake_response(),
        ]
    )
    with _patch_async_client(post):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == "hello world"
    assert post.call_count == 2


async def test_retry_exhaustion_raises_terminal_error(
    make_adapter: Callable[..., OllamaAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.adapters.ollama._backoff_delay", lambda _i: 0.0)
    adapter = make_adapter(max_attempts=3)
    post = AsyncMock(side_effect=[httpx.ConnectError("boom")] * 3)
    with _patch_async_client(post), pytest.raises(httpx.ConnectError):
        await adapter.complete(system="s", user="u")
    assert post.call_count == 3


async def test_missing_message_returns_empty_text(
    make_adapter: Callable[..., OllamaAdapter],
) -> None:
    """A response without a `message` key shouldn't crash — return empty text."""
    adapter = make_adapter()
    body = {
        "model": "llama3.2:latest",
        "prompt_eval_count": 5,
        "eval_count": 0,
        "done": True,
    }
    post = AsyncMock(return_value=_fake_response(body=body))
    with _patch_async_client(post):
        resp = await adapter.complete(system="s", user="u")
    assert resp.text == ""
    assert resp.input_tokens == 5
    assert resp.output_tokens == 0


async def test_missing_token_counts_default_to_zero(
    make_adapter: Callable[..., OllamaAdapter],
) -> None:
    """An older Ollama version may omit eval counts — degrade gracefully."""
    adapter = make_adapter()
    body = {
        "model": "llama3.2:latest",
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
    }
    post = AsyncMock(return_value=_fake_response(body=body))
    with _patch_async_client(post):
        resp = await adapter.complete(system="s", user="u")
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0


def test_default_base_url() -> None:
    adapter = OllamaAdapter(model_id="llama3.2")
    assert adapter._base_url == DEFAULT_BASE_URL


def test_explicit_base_url_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:9999")
    adapter = OllamaAdapter(model_id="llama3.2", base_url="http://explicit:7777")
    assert adapter._base_url == "http://explicit:7777"


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://my-remote-ollama:11434")
    adapter = OllamaAdapter(model_id="llama3.2")
    assert adapter._base_url == "http://my-remote-ollama:11434"


def test_trailing_slash_in_base_url_stripped() -> None:
    adapter = OllamaAdapter(model_id="llama3.2", base_url="http://host:11434/")
    assert adapter._base_url == "http://host:11434"


def test_adapter_name() -> None:
    assert OllamaAdapter.name == "ollama"
