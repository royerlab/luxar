"""I/O operations for Gaussian splat persistence.

This package provides functions for saving and loading fitted Gaussian splat
results in a dedicated zarr format (.gsplats.zarr).

Future implementation will include:
- save_gsplats() - Save GaussianSplatResult to .gsplats.zarr
- load_gsplats() - Load GaussianSplatResult from .gsplats.zarr
- inspect_gsplats_zarr() - Inspect .gsplats.zarr metadata
- Spatial ordering utilities (Morton/Hilbert)
"""

# Future exports (when implemented):
# from .save import save_gsplats
# from .load import load_gsplats
# from .inspect import inspect_gsplats_zarr

__all__ = []
