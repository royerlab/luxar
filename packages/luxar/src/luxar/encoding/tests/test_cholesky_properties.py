"""Property-based round-trip tests for the Cholesky encodings.

Complement ``test_cholesky_split_quant.py`` with the encode->decode identity
checked over many machine-generated factor batches (Hypothesis). In PRECISION
mode the encoding is float32-lossless, so the diagonal (positive, per-column
log) and off-diagonal (signed, per-column signed-log) families must round-trip
to within float32 tolerance for *any* input.
"""

from __future__ import annotations

import numpy as np
import zarr
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from luxar.encoding.decoder import ArrayDecoder
from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


def _roundtrip(data: np.ndarray, semantic_type: SemanticType) -> np.ndarray:
    enc = ArrayEncoder()
    g = zarr.group(store=zarr.MemoryStore())
    enc.encode(
        data=data,
        zarr_group=g,
        name="a",
        semantic_type=semantic_type,
        mode=EncodingMode.PRECISION,
    )
    return ArrayDecoder().decode(g["a"], g)


@given(
    n=st.integers(min_value=1, max_value=64),
    d=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
def test_cholesky_diag_precision_roundtrip(
    n: int, d: int, data: st.DataObject
) -> None:
    """Positive per-column diagonals round-trip losslessly in PRECISION mode."""
    diag = data.draw(
        hnp.arrays(
            np.float32,
            (n, d),
            elements=st.floats(0.05, 1e3, allow_nan=False, allow_infinity=False),
        )
    )
    decoded = _roundtrip(diag, SemanticType.CHOLESKY_DIAG)
    np.testing.assert_allclose(decoded, diag, rtol=1e-5, atol=1e-5)


@given(
    n=st.integers(min_value=1, max_value=64),
    d=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
def test_cholesky_offdiag_precision_roundtrip(
    n: int, d: int, data: st.DataObject
) -> None:
    """Signed off-diagonals round-trip losslessly in PRECISION mode."""
    off = data.draw(
        hnp.arrays(
            np.float32,
            (n, d),
            elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        )
    )
    decoded = _roundtrip(off, SemanticType.CHOLESKY_OFFDIAG)
    np.testing.assert_allclose(decoded, off, rtol=1e-5, atol=1e-5)
