"""
Root pytest configuration for the luxar package.

Adds an autouse fixture that seeds numpy's global RNG to a fixed value
before every test in the package. Tests using `np.random.*` (without an
explicit `default_rng(seed)`) inherit determinism for free. Tests that
already create their own seeded `default_rng` are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_numpy_global_rng() -> None:
    """Seed numpy's legacy global RNG before each test (P7 determinism).

    Use `np.random.default_rng(seed)` in new tests for a per-test
    generator. This fixture only protects existing call sites that rely
    on the legacy module-level functions (`np.random.uniform`,
    `np.random.randn`, etc.).
    """
    np.random.seed(0xC0FFEE)
