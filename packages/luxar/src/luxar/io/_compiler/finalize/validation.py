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

        # Check for range/data misalignment. Mirrors the viewer's discrete
        # query tolerance (DISCRETE_TOLERANCE_FRACTION = 0.25 in
        # tolerance-computer.ts): a quarter-step. If the declared range
        # extends further than this beyond the data, the viewer's on-grid
        # query at the range edge would select nothing — warn.
        tolerance = (dim.step / 4) if dim.step else 0.25

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

        # On-grid check: the viewer snaps discrete navigation TARGETS to the
        # absolute grid k*step (SceneDimsManager.setDimensionValue) and its
        # chunk query reaches only a quarter-step around them, while the
        # write side pads discrete chunk bounds by a tiny epsilon only
        # (ordering._BARRIER_BOUND_EPS). Discrete DATA sitting more than a
        # quarter-step off that grid can therefore pass the viewer's
        # half-step visibility gate yet never have its chunks fetched —
        # silently disappearing. Warn on off-grid data so the "discrete data
        # is on-grid" contract the query design assumes is checked, not
        # hoped. (min/max are proxies for the value set — a uniformly
        # offset grid, the archetypal mistake, is always caught.)
        #
        # A dimension WITHOUT a declared step is NOT exempt: the viewer
        # normalizes a missing step to 1.0 (scene-dims-manager.ts
        # `step: dim.step || 1.0`) and snaps/queries on that integer grid,
        # so step-less discrete data at non-integer values disappears just
        # the same. Check against the viewer's effective grid.
        effective_step = dim.step if dim.step else 1.0
        for edge_name, value in (("starts", data_min), ("ends", data_max)):
            grid_offset = abs(value - round(value / effective_step) * effective_step)
            if grid_offset > tolerance:
                step_note = (
                    f"multiples of {dim.step}"
                    if dim.step
                    else "integers — no step is declared, and the viewer "
                    "defaults a missing step to 1.0"
                )
                warnings.warn(
                    f"Dimension '{dim.name}' has discrete data that "
                    f"{edge_name} at {value:.4f}, which is "
                    f"{grid_offset:.4f} off the step grid "
                    f"({step_note}). The viewer navigates "
                    f"discrete dimensions on that grid and only fetches "
                    f"chunks within a quarter-step of it, so off-grid "
                    f"values may silently not display. Shift the "
                    f"coordinates onto multiples of the step (or adjust "
                    f"the step) so data lies on-grid.",
                    UserWarning,
                    stacklevel=3,
                )
