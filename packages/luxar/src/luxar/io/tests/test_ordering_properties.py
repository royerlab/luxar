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
