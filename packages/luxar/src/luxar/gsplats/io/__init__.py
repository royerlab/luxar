"""I/O operations for Gaussian splat persistence.

This package provides functions for saving and loading fitted Gaussian splat
data in a dedicated zarr format (.gsplats.zarr).

Main functions:
- save_gsplats() - Save GSplatData to .gsplats.zarr
- load_gsplats() - Load GSplatData from .gsplats.zarr
- load_default_gsplats() - Materialize the tree's default-rendered selection
- inspect_gsplats_zarr() - Inspect .gsplats.zarr metadata
- format_gsplats_info() - Format inspection info as string

Spatial ordering utilities:
- sort_splats_spatial() - Sort splats using Morton or Hilbert curves
- compute_chunk_bounds_gsplats() - Compute chunk bounding boxes with extent
"""

from luxar.gsplats.io.inspect_gsplats import (
    format_gsplats_info,
    inspect_gsplats_zarr,
)
from luxar.gsplats.io.load_gsplats import load_default_gsplats, load_gsplats
from luxar.gsplats.io.save_gsplats import save_gsplats

# Re-export ordering functions from luxar.io for convenience
from luxar.io.ordering import (
    compute_chunk_bounds_gsplats,
    sort_splats_spatial,
)

__all__ = [
    "save_gsplats",
    "load_gsplats",
    "load_default_gsplats",
    "inspect_gsplats_zarr",
    "format_gsplats_info",
    "sort_splats_spatial",
    "compute_chunk_bounds_gsplats",
]
