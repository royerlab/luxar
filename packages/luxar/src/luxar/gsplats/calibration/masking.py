"""Blind-spot cross-validation masking (Noise2Self self-supervision).

The two primitives the calibration protocol builds on: a deterministic
Bernoulli hold-out mask, and a donut-median fill that hides the held-out
values from the fitter (Batson & Royer 2019). The fill uses only unmasked
neighbours, so the fitted volume is independent of every held-out value.
"""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

_MAX_DONUT_BYTES = 16 * 1024 * 1024


def cv_mask(
    shape: Tuple[int, ...],
    fraction: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic Bernoulli boolean mask for blind-spot cross-validation.

    Defaults match Batson & Royer (2019) and the Luxar manuscript: 5% of
    voxels are held out with seed 42.

    Parameters
    ----------
    shape : tuple of int
        Output array shape.
    fraction : float, default=0.05
        Probability of any voxel being marked True (held out).
    seed : int, default=42
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray of bool, shape ``shape``
        ``True`` at held-out positions, ``False`` elsewhere.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    rng = np.random.RandomState(seed)
    return rng.rand(*shape) < fraction


def donut_median_fill(
    V: np.ndarray,
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Replace masked voxels with the median of their *unmasked* donut neighbours.

    The donut is the ``(2r+1)^D`` cube around each masked voxel with the
    centre excluded — 26 neighbours in 3D when ``r=1``. Neighbours that are
    themselves held out are **excluded from the median**, so the filled
    volume is a function of the unmasked voxels only: perturbing the values
    at masked positions leaves the output unchanged everywhere. This is what
    makes the blind-spot argument hold — the fitter never sees a held-out
    value, directly or through a neighbour's fill. (At a 5 % Bernoulli mask
    with a 26-neighbour donut, ~74 % of masked voxels have at least one
    masked neighbour, so the exclusion is not a corner case.)

    Operates on arrays of arbitrary dimension (2D, 3D, 4D, ...). Edge voxels
    use ``mode='reflect'`` padding (applied to ``V`` and ``mask`` alike, so a
    reflected masked voxel stays excluded). If every donor at ``radius`` is
    masked, the neighbourhood expands one shell at a time until an unmasked
    donor is found. Genuine NaN donors remain distinct from held-out donors
    and propagate through the median as they did before masked-neighbour
    exclusion was added.

    Donors are gathered and sorted in bounded chunks. The working donor
    buffer targets 16 MiB (or one donor column when that alone is larger), in
    the input dtype, in addition to the reflected copies of ``V`` and ``mask``
    for the current radius.

    Parameters
    ----------
    V : np.ndarray
        Volume to fill.
    mask : np.ndarray of bool, same shape as ``V``
        ``True`` at positions to replace (held out).
    radius : int, default=1
        Donut half-width. Default ``1`` → ``3^D`` neighbourhood, matching
        the manuscript.

    Returns
    -------
    np.ndarray, same shape and dtype as ``V``
        Copy of ``V`` with masked voxels replaced by donut medians of their
        unmasked neighbours. Unmasked voxels are unchanged.

    Raises
    ------
    ValueError
        If the shapes differ, ``radius`` is less than one, or every voxel is
        held out so no fill donor exists.
    """
    if V.shape != mask.shape:
        raise ValueError(f"V shape {V.shape} != mask shape {mask.shape}")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    masked_flat = np.flatnonzero(mask)
    n_masked = masked_flat.size

    if n_masked == 0:
        out_empty: np.ndarray = V.copy()
        return out_empty
    if n_masked == mask.size:
        raise ValueError("mask holds out every voxel; nothing to fill from")

    V_filled: np.ndarray = V.copy()
    pending = masked_flat
    current_radius = radius
    max_radius = max(radius, max(V.shape) - 1)
    while pending.size:
        median_values, has_donor = _masked_donor_medians(
            V, mask, pending, current_radius, shell_only=current_radius > radius
        )
        V_filled.flat[pending[has_donor]] = median_values[has_donor].astype(
            V.dtype, copy=False
        )
        pending = pending[~has_donor]
        if pending.size == 0:
            break
        if current_radius >= max_radius:
            raise RuntimeError("failed to find an unmasked fill donor")
        current_radius += 1
    return V_filled


def _masked_donor_medians(
    V: np.ndarray,
    mask: np.ndarray,
    masked_flat: np.ndarray,
    radius: int,
    shell_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute donor medians and report which masked voxels found a donor."""
    offsets = _donut_offsets(V.ndim, radius, shell_only)
    V_pad = np.pad(V, radius, mode="reflect")
    mask_pad = np.pad(mask, radius, mode="reflect")
    medians = np.empty(masked_flat.size, dtype=np.float64)
    has_donor = np.zeros(masked_flat.size, dtype=bool)
    bytes_per_column = max(1, len(offsets) * V.dtype.itemsize)
    chunk_size = max(1, _MAX_DONUT_BYTES // bytes_per_column)
    sentinel = _invalid_donor_sentinel(V.dtype)
    tracks_nan = np.issubdtype(V.dtype, np.floating)

    for start in range(0, masked_flat.size, chunk_size):
        stop = min(start + chunk_size, masked_flat.size)
        chunk_flat = masked_flat[start:stop]
        masked_idx = np.unravel_index(chunk_flat, V.shape)
        donor_values = np.empty((len(offsets), stop - start), dtype=V.dtype)
        donor_counts = np.zeros(stop - start, dtype=np.intp)
        has_nan = np.zeros(stop - start, dtype=bool)
        for row, offset in enumerate(offsets):
            shifted = tuple(
                masked_idx[dim] + radius + offset[dim] for dim in range(V.ndim)
            )
            donor_mask = mask_pad[shifted]
            donor_values[row] = V_pad[shifted]
            donor_counts += ~donor_mask
            if tracks_nan:
                has_nan |= ~donor_mask & np.isnan(donor_values[row])
            donor_values[row, donor_mask] = sentinel

        donor_values.sort(axis=0)
        found = donor_counts > 0
        columns = np.flatnonzero(found)
        lower = (donor_counts[found] - 1) // 2
        upper = donor_counts[found] // 2
        low_values = donor_values[lower, columns].astype(np.float64)
        high_values = donor_values[upper, columns].astype(np.float64)
        chunk_medians = (low_values + high_values) / 2.0
        chunk_medians[has_nan[found]] = np.nan
        medians[start:stop][found] = chunk_medians
        has_donor[start:stop] = found

    return medians, has_donor


def _donut_offsets(ndim: int, radius: int, shell_only: bool) -> list[tuple[int, ...]]:
    """Return offsets for a full donut or only its outer Chebyshev shell."""
    offsets = itertools.product(range(-radius, radius + 1), repeat=ndim)
    if shell_only:
        return [offset for offset in offsets if max(map(abs, offset)) == radius]
    return [offset for offset in offsets if any(component != 0 for component in offset)]


def _invalid_donor_sentinel(dtype: np.dtype) -> float | int | bool:
    """Return a sortable high value used beyond each column's valid donors."""
    if np.issubdtype(dtype, np.floating):
        return np.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    if np.issubdtype(dtype, np.bool_):
        return True
    return np.inf
