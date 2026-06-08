"""Model adapters — pluggable provider clients.

Every adapter implements `ModelAdapter` from `harness.adapters.base`. The
runner imports the Protocol, never a concrete adapter, so swapping providers
is a constructor-arg change.
"""

from harness.adapters.base import AdapterResponse, ModelAdapter

__all__ = ["AdapterResponse", "ModelAdapter"]
