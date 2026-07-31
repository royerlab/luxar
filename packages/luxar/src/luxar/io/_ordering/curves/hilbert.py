"""Hilbert space-filling curve encoder.

Maps nD integer grid coordinates to Hilbert-curve indices (better locality
than Morton at the cost of a more involved transform). A Numba JIT kernel is
used when available, falling back to the ``hilbertcurve`` library.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_hilbert_numba_kernel():  # type: ignore[no-untyped-def]
    """Lazy-compile the Numba Hilbert encoding kernel on first use."""
    import numba

    @numba.njit(cache=True)  # type: ignore[misc]
    def _hilbert_kernel(coords: np.ndarray, bits_per_dim: int, out: np.ndarray) -> None:
        """Numba-accelerated Hilbert curve encoding.

        Implements the same algorithm as the hilbertcurve library
        (Skilling's "Programming the Hilbert curve") in compiled native code.
        """
        n_points = coords.shape[0]
        n_dims = coords.shape[1]
        m = np.int64(1) << np.int64(bits_per_dim - 1)

        for idx in range(n_points):
            # Copy point to local mutable array
            pt = np.empty(n_dims, dtype=np.int64)
            for d in range(n_dims):
                pt[d] = np.int64(coords[idx, d])

            # --- Inverse undo excess work ---
            q = m
            while q > 1:
                p = q - 1
                for i in range(n_dims):
                    if pt[i] & q:
                        pt[0] ^= p
                    else:
                        t = (pt[0] ^ pt[i]) & p
                        pt[0] ^= t
                        pt[i] ^= t
                q >>= 1

            # --- Gray encode ---
            for i in range(1, n_dims):
                pt[i] ^= pt[i - 1]

            t2 = np.int64(0)
            q = m
            while q > 1:
                if pt[n_dims - 1] & q:
                    t2 ^= q - 1
                q >>= 1

            for i in range(n_dims):
                pt[i] ^= t2

            # --- Transpose to Hilbert integer (MSB-first bit interleave) ---
            # Matches hilbertcurve library convention: MSB of dim 0 first.
            h = np.uint64(0)
            for bit in range(bits_per_dim - 1, -1, -1):
                for d in range(n_dims):
                    h = (h << np.uint64(1)) | np.uint64((pt[d] >> bit) & 1)

            out[idx] = h

    return _hilbert_kernel


# None = not tried, False = tried and failed, callable = compiled kernel
_hilbert_numba_kernel: Any = None


def hilbert_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Hilbert curve indices.

    Uses a Numba JIT-compiled kernel to avoid Python-loop overhead.
    Falls back to the hilbertcurve library if Numba is unavailable.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension (default 16)

    Returns:
        Hilbert indices, shape (N,), dtype uint64
    """
    global _hilbert_numba_kernel  # noqa: PLW0603

    n_points, n_dims = coords.shape

    # Try Numba first (compiled, no Python-loop overhead)
    if _hilbert_numba_kernel is None:
        try:
            _hilbert_numba_kernel = _get_hilbert_numba_kernel()  # type: ignore[no-untyped-call]
        except (ImportError, Exception):
            _hilbert_numba_kernel = False

    if _hilbert_numba_kernel:
        out = np.empty(n_points, dtype=np.uint64)
        _hilbert_numba_kernel(coords.astype(np.int64), bits_per_dim, out)
        return out

    # Fallback: hilbertcurve library (pure Python, slow for large N)
    try:
        from hilbertcurve.hilbertcurve import (  # type: ignore[import-untyped]
            HilbertCurve,
        )
    except ImportError:
        raise ImportError(
            "Either numba or hilbertcurve package is required for Hilbert ordering. "
            "Install with: pip install numba  (or: pip install hilbertcurve)"
        )

    hilbert = HilbertCurve(bits_per_dim, n_dims)
    hilbert_indices = np.array(
        [hilbert.distance_from_point(coords[i]) for i in range(n_points)],
        dtype=np.uint64,
    )
    return hilbert_indices
