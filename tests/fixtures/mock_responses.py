"""Mock ModelAdapter for offline runner/CLI tests.

`MockAdapter` implements the `ModelAdapter` Protocol and returns scripted
responses in call order. Inject failures via `raise_on_call` to test the
runner's per-case error isolation.
"""

from dataclasses import dataclass, field

from harness.adapters.base import AdapterResponse


@dataclass
class MockResponse:
    """A scripted single response for one adapter call."""

    text: str
    input_tokens: int = 50
    output_tokens: int = 50
    latency_ms: float = 100.0
    resolved_model_id: str = "mock-model-1"


@dataclass
class MockAdapter:
    """In-memory `ModelAdapter` for tests.

    Returns `responses[i]` on the i-th call. If `raise_on_call[i]` is non-None,
    raises that exception instead. Reuses the last response for further calls.
    """

    name: str = "mock"
    responses: list[MockResponse] = field(default_factory=list)
    raise_on_call: list[Exception | None] = field(default_factory=list)
    _call_index: int = 0

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> AdapterResponse:
        i = self._call_index
        self._call_index += 1

        if i < len(self.raise_on_call):
            exc = self.raise_on_call[i]
            if exc is not None:
                raise exc

        if i < len(self.responses):
            r = self.responses[i]
        elif self.responses:
            r = self.responses[-1]
        else:
            r = MockResponse(text="default")

        return AdapterResponse(
            text=r.text,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            latency_ms=r.latency_ms,
            resolved_model_id=r.resolved_model_id,
        )
