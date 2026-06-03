"""Finalize-time validation: discrete-dimension range vs. data-extent checks."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import zarr


def validate_discrete_dimension_ranges(
    store: zarr.Group,
    scene_bounds: Optional[Dict[str, List[float]]],
) -> None:
    """Validate that discrete dimension ranges align with actual data.

    For discrete dimensions (like time frames), the declared range should
    correspond to actual data positions. If range.min < data_min or
    range.max > data_max, the viewer may initialize to a position with no data.

    This validation helps catch cases where:
    - Range starts at 0 but data starts at frame 1
    - Range extends beyond actual data extent

    Args:
        store: The opened zarr store to read dimensions from
        scene_bounds: Scene-level position bounds, or None.
    """
    # Check if we have the necessary data
    if scene_bounds is None:
        return
    if "scene_dimensions" not in store.attrs:
        return

    # Read dimensions from the store
    from ....core.dimensions import Dimensions

    scene_dims_dict = store.attrs["scene_dimensions"]
    dimensions = Dimensions.from_dict(scene_dims_dict)
    dims = dimensions.dimensions
    ndim = len(dims)

    # Only check dimensions that we have bounds for
    bounds_ndim = len(scene_bounds["min"])
    check_ndim = min(ndim, bounds_ndim)

    for i in range(check_ndim):
        dim = dims[i]

        # Only validate discrete, non-displayed dimensions with defined ranges
        if not dim.discrete or dim.display or dim.range is None:
            continue

        declared_min, declared_max = dim.range
        data_min = scene_bounds["min"][i]
        data_max = scene_bounds["max"][i]

        # Check for range/data misalignment
        tolerance = (dim.step / 2) if dim.step else 0.5

        if declared_min < data_min - tolerance:
            warnings.warn(
                f"Dimension '{dim.name}' has range starting at {declared_min}, "
                f"but actual data starts at {data_min:.4f}. "
                f"The viewer will initialize at {declared_min} where no data exists. "
                f"Consider setting range=({data_min}, {declared_max}) to match data extent.",
                UserWarning,
                stacklevel=3,
            )

        if declared_max > data_max + tolerance:
            warnings.warn(
                f"Dimension '{dim.name}' has range ending at {declared_max}, "
                f"but actual data ends at {data_max:.4f}. "
                f"Navigation beyond {data_max} will show no data. "
                f"Consider setting range=({declared_min}, {data_max}) to match data extent.",
                UserWarning,
                stacklevel=3,
            )
