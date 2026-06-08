"""Phase 0 smoke tests: the package is importable and exposes a version string."""

import harness


def test_harness_importable() -> None:
    assert harness is not None


def test_harness_exposes_version() -> None:
    assert isinstance(harness.__version__, str)
    assert harness.__version__, "version string must be non-empty"
