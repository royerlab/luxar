"""I/O operations for Gaussian splat persistence.

This package provides functions for saving and loading fitted Gaussian splat
results in a dedicated zarr format (.gsplats.zarr).

Main functions:
- save_gsplats() - Save GaussianSplatResult to .gsplats.zarr
- load_gsplats() - Load GaussianSplatResult from .gsplats.zarr
- inspect_gsplats_zarr() - Inspect .gsplats.zarr metadata
- format_gsplats_info() - Format inspection info as string

Spatial ordering utilities:
- sort_splats_spatially() - Sort splats using Morton or Hilbert curves
- compute_chunk_bounds() - Compute chunk bounding boxes with extent
"""

from luxar.gsplats.io.inspect_gsplats import (
    format_gsplats_info,
    inspect_gsplats_zarr,
)
from luxar.gsplats.io.load_gsplats import load_gsplats
from luxar.gsplats.io.ordering import (
    compute_chunk_bounds,
    sort_splats_spatially,
)
from luxar.gsplats.io.save_gsplats import save_gsplats

__all__ = [
    "save_gsplats",
    "load_gsplats",
    "inspect_gsplats_zarr",
    "format_gsplats_info",
    "sort_splats_spatially",
    "compute_chunk_bounds",
]
