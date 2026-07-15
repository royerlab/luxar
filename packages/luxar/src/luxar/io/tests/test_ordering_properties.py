"""Property-based tests for the spatial-ordering curve encoders.

Complement the example-based ordering tests with the invariant that the
Morton/Hilbert encoders are pure, permutation-*equivariant* functions of the
coordinates: reordering the input rows reorders the output codes identically,
and the multiset of codes is invariant. This is what lets the ordering be a
stable spatial sort regardless of the order splats/points arrive in.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from luxar.io.ordering import hilbert_encode_nd, morton_encode_nd


@given(
    n=st.integers(min_value=1, max_value=64),
    d=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_morton_is_permutation_equivariant(n: int, d: int, data: st.DataObject) -> None:
    """morton(coords[perm]) == morton(coords)[perm] — a pure function of the
    coordinates, so a spatial sort is independent of input row order."""
    coords = data.draw(
        hnp.arrays(np.int64, (n, d), elements=st.integers(0, 2**16 - 1))
    )
    perm = data.draw(st.permutations(list(range(n))))
    perm_arr = np.asarray(perm, dtype=np.intp)
    base = morton_encode_nd(coords, bits_per_dim=16)
    permuted = morton_encode_nd(coords[perm_arr], bits_per_dim=16)
    np.testing.assert_array_equal(permuted, base[perm_arr])


@given(
    n=st.integers(min_value=1, max_value=64),
    d=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_hilbert_is_permutation_equivariant(
    n: int, d: int, data: st.DataObject
) -> None:
    """Same equivariance for the Hilbert encoder."""
    coords = data.draw(
        hnp.arrays(np.int64, (n, d), elements=st.integers(0, 2**16 - 1))
    )
    perm = data.draw(st.permutations(list(range(n))))
    perm_arr = np.asarray(perm, dtype=np.intp)
    base = hilbert_encode_nd(coords, bits_per_dim=16)
    permuted = hilbert_encode_nd(coords[perm_arr], bits_per_dim=16)
    np.testing.assert_array_equal(permuted, base[perm_arr])


# --- numba path vs numpy-fallback parity -------------------------------------
# The encoders have a Numba-JIT fast path and a pure-NumPy fallback (used when
# Numba is unavailable). With Numba installed the property tests above only
# exercise the JIT path; these deterministic tests force the fallback and assert
# it produces byte-identical codes — so the fallback is verified AND kept in
# parity with the JIT kernel.

# The lazy-compiled numba kernels are cached as module globals inside the
# curve encoders (io/_ordering/curves/{morton,hilbert}.py); patch them there.
import luxar.io._ordering.curves.hilbert as _hilbert_mod  # noqa: E402
import luxar.io._ordering.curves.morton as _morton_mod  # noqa: E402


def _encode_forcing_numpy_fallback(fn, coords: np.ndarray) -> np.ndarray:
    """Run ``fn`` with both Numba kernels forced OFF (pure-NumPy path)."""
    saved = (_morton_mod._morton_numba_kernel, _hilbert_mod._hilbert_numba_kernel)
    try:
        _morton_mod._morton_numba_kernel = False
        _hilbert_mod._hilbert_numba_kernel = False
        return fn(coords, bits_per_dim=16)
    finally:
        _morton_mod._morton_numba_kernel = saved[0]
        _hilbert_mod._hilbert_numba_kernel = saved[1]


def _encode_forcing_numba(fn, coords: np.ndarray) -> np.ndarray:
    """Run ``fn`` with the kernels reset to None so the JIT path lazy-loads
    (falls back to NumPy only if Numba is genuinely unavailable)."""
    saved = (_morton_mod._morton_numba_kernel, _hilbert_mod._hilbert_numba_kernel)
    try:
        _morton_mod._morton_numba_kernel = None
        _hilbert_mod._hilbert_numba_kernel = None
        return fn(coords, bits_per_dim=16)
    finally:
        _morton_mod._morton_numba_kernel = saved[0]
        _hilbert_mod._hilbert_numba_kernel = saved[1]


def test_morton_numba_numpy_parity() -> None:
    """morton: JIT path and NumPy fallback yield byte-identical codes."""
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 2**16, size=(256, 3), dtype=np.int64)
    np.testing.assert_array_equal(
        _encode_forcing_numpy_fallback(morton_encode_nd, coords),
        _encode_forcing_numba(morton_encode_nd, coords),
    )


def test_hilbert_numba_numpy_parity() -> None:
    """hilbert: JIT path and NumPy fallback yield byte-identical codes."""
    rng = np.random.default_rng(1)
    coords = rng.integers(0, 2**16, size=(256, 3), dtype=np.int64)
    np.testing.assert_array_equal(
        _encode_forcing_numpy_fallback(hilbert_encode_nd, coords),
        _encode_forcing_numba(hilbert_encode_nd, coords),
    )
