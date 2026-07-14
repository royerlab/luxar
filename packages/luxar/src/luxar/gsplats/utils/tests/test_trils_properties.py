"""Property-based tests for the triangular-packing utilities.

Complement the example-based tests in ``test_trils.py`` with algebraic
invariants checked over many machine-generated inputs (Hypothesis): the
pack/unpack and diag/offdiag split round-trips must hold for *every* batch of
Cholesky factors, not just the hand-picked shapes.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from luxar.gsplats.utils.trils import (
    merge_tril,
    pack_tril,
    split_tril,
    unpack_tril,
)

_finite = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=32
)


@given(
    d=st.integers(min_value=1, max_value=6),
    n=st.integers(min_value=1, max_value=8),
    data=st.data(),
)
def test_pack_unpack_recovers_lower_triangle(
    d: int, n: int, data: st.DataObject
) -> None:
    """unpack_tril(pack_tril(L)) == tril(L) for any batch — the packing is a
    lossless round-trip that keeps the lower triangle and zeroes the upper
    (the on-disk Cholesky storage contract)."""
    matrices = data.draw(hnp.arrays(np.float32, (n, d, d), elements=_finite))
    recovered = unpack_tril(pack_tril(matrices), d)
    np.testing.assert_array_equal(recovered, np.tril(matrices))


@given(
    d=st.integers(min_value=1, max_value=6),
    n=st.integers(min_value=1, max_value=8),
    data=st.data(),
)
def test_split_merge_roundtrip(d: int, n: int, data: st.DataObject) -> None:
    """merge_tril(*split_tril(packed)) == packed for any packed batch — the
    diag/offdiag split (the geolog vs signed-log encoding boundary) is a
    lossless partition of the packed vector."""
    k = d * (d + 1) // 2
    packed = data.draw(hnp.arrays(np.float32, (n, k), elements=_finite))
    diag, offdiag = split_tril(packed, d)
    np.testing.assert_array_equal(merge_tril(diag, offdiag, d), packed)
