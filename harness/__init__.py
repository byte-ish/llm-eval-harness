"""llm-eval-harness — versioned, CI-gated regression testing for LLM features."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-eval-harness")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
