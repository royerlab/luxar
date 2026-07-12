"""Infer a :class:`~luxar.core.dimensions.Dimensions` object from data extent.

Domain helper (reused by the gsplat ``convert`` and ``view`` CLI commands) that
builds display dimensions whose ranges match a splat center bounding box.
Previously lived in ``luxar.cli.gsplat_config``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["build_dimensions_from_data"]


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
                step=1.0,
                display=(i < 3),
            )
        )
    return Dimensions(dimensions=dim_list)
