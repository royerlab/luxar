"""
Volume downscaling utilities for Gaussian splat fitting.

Provides anti-aliased integer downscaling of volumes before fitting,
and coordinate rescaling of fitted splat parameters back to original resolution.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_downscale(
    downscale: int | Sequence[int] | None,
    ndim: int,
) -> tuple[int, ...] | None:
    """Normalize downscale parameter to a per-axis integer tuple.

    Parameters
    ----------
    downscale : int, sequence of int, or None
        Downscale factor(s). A single int is broadcast to all axes.
        A sequence must have one element per dimension.
    ndim : int
        Number of volume dimensions.

    Returns
    -------
    tuple[int, ...] or None
        Per-axis factors, or None if no downscaling is needed.

    Raises
    ------
    ValueError
        If factors are invalid (non-positive, wrong length).
    """
    if downscale is None:
        return None
    if isinstance(downscale, (int, np.integer)):
        factors = (int(downscale),) * ndim
    else:
        factors = tuple(int(f) for f in downscale)
        if len(factors) != ndim:
            raise ValueError(
                f"downscale has {len(factors)} elements but volume has {ndim} dimensions"
            )
    for f in factors:
        if f < 1:
            raise ValueError(f"downscale factors must be >= 1, got {f}")
    if all(f == 1 for f in factors):
        return None  # No-op
    return factors


def downscale_volume(
    V: np.ndarray,
    factors: tuple[int, ...],
) -> np.ndarray:
    """Anti-alias and decimate a volume by integer factors per axis.

    Applies a Gaussian blur (sigma = factor/2 per axis) for anti-aliasing,
    then decimates via strided slicing.

    Parameters
    ----------
    V : np.ndarray
        Input volume of any dimensionality.
    factors : tuple[int, ...]
        Per-axis integer downscale factors (length must equal V.ndim).

    Returns
    -------
    np.ndarray
        Downscaled volume.
    """
    from scipy.ndimage import gaussian_filter

    # sigma = factor/2 per axis; no blur for axes with factor=1
    sigma = tuple(f / 2.0 if f > 1 else 0.0 for f in factors)
    blurred = gaussian_filter(V.astype(np.float32), sigma=sigma)
    slices = tuple(slice(None, None, f) for f in factors)
    return blurred[slices]


def rescale_centers(
    centers: np.ndarray,
    factors: tuple[int, ...],
) -> np.ndarray:
    """Scale splat centers from downscaled coordinates back to original coordinates.

    Parameters
    ----------
    centers : np.ndarray, shape (N, d)
        Splat centers in downscaled voxel coordinates.
    factors : tuple[int, ...]
        Per-axis downscale factors.

    Returns
    -------
    np.ndarray, shape (N, d)
        Splat centers in original voxel coordinates.
    """
    return centers * np.array(factors, dtype=np.float32)


def rescale_cholesky_packed(
    cholesky_packed: np.ndarray,
    factors: tuple[int, ...],
) -> np.ndarray:
    """Scale packed Cholesky factors from downscaled coordinates to original coordinates.

    For a lower-triangular Cholesky factor L, row i is scaled by factors[i]:
        L_orig[i, j] = L_down[i, j] * factors[i]

    The packed format stores row i with (i+1) elements, so the scale vector
    repeats factors[i] for (i+1) positions. This is identical to the voxel_size
    scaling logic in finalize_results().

    Parameters
    ----------
    cholesky_packed : np.ndarray, shape (N, d*(d+1)//2)
        Packed lower-triangular Cholesky factors.
    factors : tuple[int, ...]
        Per-axis downscale factors.

    Returns
    -------
    np.ndarray, shape (N, d*(d+1)//2)
        Rescaled packed Cholesky factors.
    """
    d = len(factors)
    fv = np.array(factors, dtype=np.float32)
    tril_scales = np.concatenate([[fv[i]] * (i + 1) for i in range(d)])
    return cholesky_packed * tril_scales
