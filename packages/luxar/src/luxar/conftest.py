"""
Root pytest configuration for the luxar package.

Adds an autouse fixture that seeds numpy's global RNG to a fixed value
before every test in the package. Tests using `np.random.*` (without an
explicit `default_rng(seed)`) inherit determinism for free. Tests that
already create their own seeded `default_rng` are unaffected.

Also holds a couple of small cross-language-constant-lock helpers
(``find_repo_relative_file`` / ``read_ts_number_const``) shared by the
handful of Python tests that read a numeric constant straight out of a
TypeScript source file rather than trust a prose comment to stay in sync.
"""

from __future__ import annotations

import re
from pathlib import Path

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


def find_repo_relative_file(rel_path: Path, start: Path) -> Path | None:
    """Walk up from ``start``'s parents looking for ``parent / rel_path``.

    Used by cross-language constant-lock tests to locate a file elsewhere in
    the repo (e.g. a TypeScript source under ``packages/luxar-viewer``)
    without hard-coding a parent-directory depth — a package move would
    silently break a fixed ``../../..`` chain, whereas this keeps searching
    until it finds the file or runs out of ancestors.
    """
    return next((p / rel_path for p in start.parents if (p / rel_path).is_file()), None)


def read_ts_number_const(source: str, name: str) -> float:
    """Parse ``const NAME = <number>;`` out of TypeScript source text.

    Tolerates ``export const``, a plain ``const``, and an optional ``: number``
    type annotation. Raises ``AssertionError`` (not a parse exception) so a
    failure reads as a normal, informative test failure.
    """
    m = re.search(
        rf"^\s*(?:export\s+)?const\s+{name}\s*(?::\s*number\s*)?=\s*"
        r"([0-9]*\.?[0-9]+)\s*;",
        source,
        re.MULTILINE,
    )
    assert m is not None, (
        f"no `const {name} = <number>;` declaration found in the given source. "
        "If it was renamed or computed, update the caller — do NOT delete it."
    )
    return float(m.group(1))
