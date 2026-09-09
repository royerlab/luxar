"""Blind-spot cross-validation masking (Noise2Self self-supervision).

The two primitives the calibration protocol builds on: a deterministic
Bernoulli hold-out mask, and a donut-median fill that hides the held-out
values from the fitter (Batson & Royer 2019). The fill uses only unmasked
neighbours, so the fitted volume is independent of every held-out value.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Tuple

import numpy as np


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
    reflected masked voxel stays excluded). In the practically unreachable
    event that every donor of a voxel is masked (probability ``f^26`` at
    ``r=1`` in 3D), the fill falls back to the median of all unmasked
    voxels in the volume.

    Vectorised: gathers donut values for all masked positions at once via
    stacked shifted-index lookups against a single padded copy of ``V``.
    Memory cost: ``(2r+1)^D - 1`` float64 values per masked voxel.

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
    """
    if V.shape != mask.shape:
        raise ValueError(f"V shape {V.shape} != mask shape {mask.shape}")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    D = V.ndim
    # Donut footprint: all (2r+1)^D offsets except (0,...,0)
    offsets = [
        o
        for o in itertools.product(range(-radius, radius + 1), repeat=D)
        if any(c != 0 for c in o)
    ]

    masked_idx = np.nonzero(mask)
    n_masked = masked_idx[0].size if len(masked_idx) > 0 else 0

    if n_masked == 0:
        out_empty: np.ndarray = V.copy()
        return out_empty
    if n_masked == mask.size:
        raise ValueError("mask holds out every voxel; nothing to fill from")

    # Pad with reflect so edge voxels have full neighbourhoods. The mask is
    # padded the same way so reflected donors carry their held-out status.
    V_pad = np.pad(V, radius, mode="reflect")
    mask_pad = np.pad(mask, radius, mode="reflect")

    n_donut = len(offsets)
    # float64 working array: NaN marks a held-out donor (V may be integer).
    donut_values = np.empty((n_donut, n_masked), dtype=np.float64)

    for k, offset in enumerate(offsets):
        shifted = tuple(masked_idx[d] + radius + offset[d] for d in range(D))
        vals = V_pad[shifted].astype(np.float64, copy=False)
        vals[mask_pad[shifted]] = np.nan
        donut_values[k] = vals

    with warnings.catch_warnings():
        # An all-NaN column (every donor held out) is handled just below.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_values = np.nanmedian(donut_values, axis=0)

    all_masked = np.isnan(median_values)
    if all_masked.any():
        median_values[all_masked] = np.median(V[~mask].astype(np.float64))

    V_filled: np.ndarray = V.copy()
    V_filled[masked_idx] = median_values.astype(V.dtype, copy=False)
    return V_filled
