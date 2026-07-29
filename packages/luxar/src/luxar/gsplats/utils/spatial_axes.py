"""Spatial vs categorical axis discrimination — one shared definition.

Several operations must tell **spatial** axes (real covariance extent) apart from
**categorical / degenerate** axes (a zero-variance stacked-time or channel axis):
the scale / eccentricity / isolation filters, and spatial-only centering. Both
the flat (``GSplatData``) and node-tree code paths reduce to a per-axis
max-marginal-sigma vector and then apply the SAME threshold + fallback here, so
the rule lives in exactly one place.
"""

from __future__ import annotations

import numpy as np

#: An axis counts as spatial when its maximum marginal sigma across the splats
#: exceeds this (world units). A per-timepoint time axis (sigma ≈ 0) falls below.
SPATIAL_SIGMA_EPS = 1e-6


def spatial_axes_from_max_sigma(
    max_sigma: np.ndarray, eps: float = SPATIAL_SIGMA_EPS, fallback: bool = True
) -> np.ndarray:
    """Indices of axes whose max marginal sigma exceeds ``eps``.

    With ``fallback=True`` (the default) falls back to ALL axes when none qualify
    (empty or all-degenerate input), so callers never receive an empty selection
    (which would, e.g., disable centering entirely). Pass ``fallback=False`` when
    the caller needs to distinguish "every axis is spatial" from "no axis is" —
    the fallback makes both return every index — and prefers an empty selection.
    """
    max_sigma = np.asarray(max_sigma)
    keep = np.flatnonzero(max_sigma > eps)
    if keep.size > 0 or not fallback:
        return keep
    return np.arange(max_sigma.shape[0])


def spatial_only_shift(centroid: np.ndarray, spatial_axes: np.ndarray) -> np.ndarray:
    """Translation vector that moves only ``spatial_axes`` by ``centroid`` and
    leaves every other (categorical) axis at zero — the shared basis for
    spatial-only re-centering.
    """
    centroid = np.asarray(centroid, dtype=np.float64)
    shift = np.zeros(centroid.shape[0], dtype=np.float64)
    shift[spatial_axes] = centroid[spatial_axes]
    return shift
