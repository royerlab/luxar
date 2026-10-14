"""Release-gate requirements for the Metal parity suite.

The whole suite runs against whichever kernel backend is selected; pin it with
``LUXAR_METAL_BACKEND=aot`` or ``LUXAR_METAL_BACKEND=runtime`` to run it once
per backend (a forced backend that cannot run fails collection loudly).
"""

from __future__ import annotations

import os

import pytest

from luxar.gsplats.models.gsplats.metal import get_metal_status, is_metal_available


def pytest_configure() -> None:
    """Fail instead of skipping when a release requires real Metal coverage."""
    if os.environ.get("LUXAR_REQUIRE_METAL") == "1" and not is_metal_available():
        raise pytest.UsageError(
            f"LUXAR_REQUIRE_METAL=1 but the Metal backend is unavailable: "
            f"{get_metal_status()}"
        )
