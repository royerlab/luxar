"""Cross-rule structural invariants for the BSP partitioners.

B2/[P2][P12]: every partition rule (midpoint / median / SAH) must satisfy
two contracts on *any* non-degenerate input, regardless of seed:

1. **Disjoint cover** — the concatenation of all parts is a permutation of
   ``range(N)``: every input element lands in exactly one part, with no
   loss and no duplication.
2. **Cap respected** — every returned part has ``size <= max_elements``
   (guaranteed for non-coincident data, where a split can always make
   progress).

These are seeded example sweeps (the project does not depend on Hypothesis);
running the same assertion across many seeds and sizes is a deterministic
stand-in for a property test and kills mutants that the fixed-input unit
tests in ``test_bsp.py`` / ``test_median.py`` / ``test_sah.py`` miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.group.partition import (
    median_bsp_partition,
    midpoint_bsp_partition,
    sah_bsp_partition,
)

PARTITIONERS = [
    pytest.param(midpoint_bsp_partition, id="midpoint"),
    pytest.param(median_bsp_partition, id="median"),
    pytest.param(sah_bsp_partition, id="sah"),
]


@pytest.mark.parametrize("partition_fn", PARTITIONERS)
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "n,max_elements",
    [(200, 50), (500, 64), (1000, 120), (333, 100)],
)
def test_disjoint_cover_and_cap(partition_fn, seed, n, max_elements) -> None:
    """Every element appears exactly once; every part honors the cap."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-10.0, 10.0, (n, 3)).astype(np.float32)
    parts = partition_fn(pos, max_elements=max_elements)

    # Disjoint cover: concatenation is a permutation of range(n).
    concat = np.concatenate(parts)
    assert concat.size == n, "element count changed (loss or duplication)"
    assert len(set(concat.tolist())) == n, "duplicate indices across parts"
    np.testing.assert_array_equal(np.sort(concat), np.arange(n))

    # Cap respected on non-coincident data.
    assert all(p.size <= max_elements for p in parts)
    # And a non-degenerate input over the cap must actually split.
    assert len(parts) >= 2


@pytest.mark.parametrize("partition_fn", PARTITIONERS)
@pytest.mark.parametrize("seed", range(4))
def test_anisotropic_input_preserves_cover(partition_fn, seed) -> None:
    """Strongly anisotropic data (long X, thin Y/Z) still yields a clean
    disjoint cover under every rule — the axis-selection branches don't drop
    or duplicate elements."""
    rng = np.random.RandomState(100 + seed)
    pos = np.stack(
        [
            rng.uniform(-100.0, 100.0, 400),
            rng.uniform(-0.5, 0.5, 400),
            rng.uniform(-0.5, 0.5, 400),
        ],
        axis=1,
    ).astype(np.float32)
    parts = partition_fn(pos, max_elements=120)
    concat = np.concatenate(parts)
    assert len(set(concat.tolist())) == 400
    np.testing.assert_array_equal(np.sort(concat), np.arange(400))
    assert all(p.size <= 120 for p in parts)
