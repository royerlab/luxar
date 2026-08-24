"""Infer a :class:`~luxar.core.dimensions.Dimensions` object from data extent.

Domain helper (reused by the gsplat ``convert`` and ``view`` CLI commands) that
builds display dimensions whose ranges match a splat center bounding box.
Previously lived in ``luxar.cli.gsplat_config``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from arbol import aprint

__all__ = ["build_dimensions_from_data", "infer_discrete_step"]

_MAX_EXACT_FLOAT32_INTEGER = 1 << 24


def infer_discrete_step(coordinates: np.ndarray) -> float:
    """Infer an integer coordinate stride for a range-min-anchored grid.

    Non-integer coordinates retain the historical unit step. Invalid or
    inexact float32 coordinates warn and retain that fallback rather than
    making conversion fail.
    """
    values = np.unique(np.asarray(coordinates))
    if not np.all(np.isfinite(values)):
        aprint("⚠️  Discrete dimension coordinates must be finite; using step 1.0")
        return 1.0
    if values.size < 2:
        return 1.0
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        return 1.0
    if np.max(np.abs(rounded)) > _MAX_EXACT_FLOAT32_INTEGER:
        aprint(
            "⚠️  Discrete dimension coordinates must be within ±2^24 for exact "
            "float32 representation; using step 1.0"
        )
        return 1.0
    differences = np.diff(rounded.astype(np.int64))
    return float(np.gcd.reduce(differences))


def build_dimensions_from_data(centers: np.ndarray) -> Any:
    """Build a ``Dimensions`` object from a gsplat center bounding box.

    Args:
        centers: Splat center positions (N, D)

    Returns:
        Dimensions with ranges matching the data extent
    """
    from luxar.core.dimensions import Dimension, Dimensions

    ndim = centers.shape[1]
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)

    # Ensure range is valid (min < max) — add epsilon for degenerate dims
    for i in range(ndim):
        if maxs[i] <= mins[i]:
            maxs[i] = mins[i] + 1.0

    if ndim == 2:
        dims = Dimensions.default_2d()
        for i, dim in enumerate(dims.dimensions):
            dim.range = (float(mins[i]), float(maxs[i]))
        return dims

    if ndim == 3:
        dims = Dimensions.default_3d()
        for i, dim in enumerate(dims.dimensions):
            dim.range = (float(mins[i]), float(maxs[i]))
        return dims

    # nD: first 3 displayed, rest non-displayed
    dim_list = []
    for i in range(ndim):
        dim_list.append(
            Dimension(
                name=f"dim{i}",
                unit="voxel",
                range=(float(mins[i]), float(maxs[i])),
                step=infer_discrete_step(centers[:, i]) if i >= 3 else 1.0,
                display=(i < 3),
            )
        )
    return Dimensions(dimensions=dim_list)
