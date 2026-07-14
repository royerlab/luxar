"""Morton (Z-order) space-filling curve encoders.

Bit-interleaving encoders that map nD integer grid coordinates to a scalar
code whose sort order is spatially local. A Numba JIT kernel is used when
available, with a vectorized NumPy fallback (kept byte-for-byte identical and
exercised by the parity tests in ``io/tests/test_ordering_properties.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_morton_numba_kernel():  # type: ignore[no-untyped-def]
    """Lazy-compile the Numba Morton encoding kernel on first use."""
    import numba

    @numba.njit(cache=True)
    def _morton_kernel(coords: np.ndarray, bits_per_dim: int, out: np.ndarray) -> None:
        n_points = coords.shape[0]
        n_dims = coords.shape[1]
        for idx in range(n_points):
            h = np.uint64(0)
            for bit in range(bits_per_dim):
                for d in range(n_dims):
                    h |= np.uint64((coords[idx, d] >> bit) & 1) << np.uint64(
                        bit * n_dims + d
                    )
            out[idx] = h

    return _morton_kernel


# None = not tried, False = tried and failed, callable = compiled kernel
_morton_numba_kernel: Any = None


def morton_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Morton codes via bit interleaving.

    Uses a Numba JIT-compiled kernel when available, falling back to
    vectorized NumPy.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension (default 16)

    Returns:
        Morton codes, shape (N,), dtype uint64
    """
    global _morton_numba_kernel  # noqa: PLW0603

    n_points, n_dims = coords.shape

    if _morton_numba_kernel is None:
        try:
            _morton_numba_kernel = _get_morton_numba_kernel()  # type: ignore[no-untyped-call]
        except (ImportError, Exception):
            _morton_numba_kernel = False

    if _morton_numba_kernel:
        out = np.empty(n_points, dtype=np.uint64)
        coords_i64 = np.ascontiguousarray(coords, dtype=np.int64)
        _morton_numba_kernel(coords_i64, bits_per_dim, out)
        return out

    # Fallback: vectorized NumPy
    morton = np.zeros(n_points, dtype=np.uint64)
    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            morton |= coord_bit.astype(np.uint64) << (bit * n_dims + dim)
    return morton


def morton_encode_128bit(
    coords: np.ndarray, bits_per_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Encode nD integer coordinates to 128-bit Morton codes as (high, low) pairs.

    For high-dimensional data (> 6 dims), 64-bit Morton codes have insufficient
    precision. This function produces 128-bit codes as paired uint64 values.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension

    Returns:
        high: Upper 64 bits of Morton codes, shape (N,), dtype uint64
        low: Lower 64 bits of Morton codes, shape (N,), dtype uint64
    """
    n_points, n_dims = coords.shape
    high = np.zeros(n_points, dtype=np.uint64)
    low = np.zeros(n_points, dtype=np.uint64)

    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            bit_pos = bit * n_dims + dim
            if bit_pos < 64:
                low |= coord_bit.astype(np.uint64) << bit_pos
            else:
                high |= coord_bit.astype(np.uint64) << (bit_pos - 64)

    return high, low
