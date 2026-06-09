"""Tests for `harness.adapters.factory` — provider-prefix routing."""

from unittest.mock import patch

import pytest

from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.bedrock import BedrockAdapter
from harness.adapters.factory import (
    DEFAULT_PROVIDER,
    KNOWN_PROVIDERS,
    ModelSpec,
    build_adapter,
    parse_model_spec,
)
from harness.adapters.openai import OpenAIAdapter


def test_bare_model_id_defaults_to_anthropic() -> None:
    spec = parse_model_spec("claude-haiku-4-5-20251001")
    assert spec == ModelSpec(provider="anthropic", model_id="claude-haiku-4-5-20251001")
    assert spec.display == "anthropic:claude-haiku-4-5-20251001"


def test_explicit_anthropic_prefix() -> None:
    spec = parse_model_spec("anthropic:claude-opus-4-7")
    assert spec.provider == "anthropic"
    assert spec.model_id == "claude-opus-4-7"


def test_openai_prefix() -> None:
    spec = parse_model_spec("openai:gpt-4o-mini")
    assert spec.provider == "openai"
    assert spec.model_id == "gpt-4o-mini"


def test_bedrock_model_id_can_contain_colons() -> None:
    """Bedrock IDs look like 'anthropic.claude-haiku-4-5-v1:0' — colons matter."""
    spec = parse_model_spec("bedrock:anthropic.claude-haiku-4-5-v1:0")
    assert spec.provider == "bedrock"
    assert spec.model_id == "anthropic.claude-haiku-4-5-v1:0"


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown provider 'cohere'"):
        parse_model_spec("cohere:command-r-plus")


def test_empty_model_id_raises() -> None:
    with pytest.raises(ValueError, match="empty model_id"):
        parse_model_spec("openai:")


def test_known_providers_constant_is_frozen() -> None:
    assert isinstance(KNOWN_PROVIDERS, frozenset)
    assert "anthropic" in KNOWN_PROVIDERS
    assert "openai" in KNOWN_PROVIDERS
    assert "bedrock" in KNOWN_PROVIDERS
    assert DEFAULT_PROVIDER == "anthropic"


def test_build_adapter_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    spec = ModelSpec(provider="anthropic", model_id="claude-opus-4-7")
    adapter = build_adapter(spec)
    assert isinstance(adapter, AnthropicAdapter)


def test_build_adapter_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    spec = ModelSpec(provider="openai", model_id="gpt-4o-mini")
    adapter = build_adapter(spec)
    assert isinstance(adapter, OpenAIAdapter)


def test_build_adapter_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    spec = ModelSpec(provider="bedrock", model_id="anthropic.claude-haiku-4-5-v1:0")
    with patch("harness.adapters.bedrock.boto3.client"):
        adapter = build_adapter(spec)
    assert isinstance(adapter, BedrockAdapter)
