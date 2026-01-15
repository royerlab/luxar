# seeds/__init__.py
"""
Seed generation methods for Gaussian splatting.

This package provides unified seeding for Gaussian splat fitting. All methods
return GSplatData with scale-informed Gaussian shapes.

**Seeding Methods:**

1. **seed_from_decomposition**: Scale-hierarchical detection via image
   decomposition. Each seed's sigma equals the decomposition scale factor.
   Best for blob-like features.

2. **seed_from_grid**: Uniform grid seeding for spatial coverage.
   Isotropic Gaussians with user-defined or auto-computed sigma.

3. **seed_from_edges**: Edge-based seeding with anisotropic shapes.
   Uses structure tensor for Gaussian orientation along edges.

**Unified Entry Point:**

Use `generate_seeds()` for a unified interface to all methods. The default
method is "auto" which uses fast edges + grid combination (decomposition
excluded for speed). Use method="decomposition,edges,grid" to include all.

Examples
--------
>>> from luxar.gsplats.seeds import generate_seeds
>>>
>>> # Automatic method selection (recommended) - fast edges + grid
>>> seeds = generate_seeds(image)
>>>
>>> # Single method
>>> seeds = generate_seeds(image, method="decomposition")
>>>
>>> # Use with fitter
>>> from luxar.gsplats import fit_gaussian_splats
>>> result = fit_gaussian_splats(image, seeds=seeds)
"""

from luxar.gsplats.seeds.edges import seed_from_edges
from luxar.gsplats.seeds.generate import generate_seeds
from luxar.gsplats.seeds.grid import seed_from_grid
from luxar.gsplats.seeds.multiscale_decomposition import seed_from_decomposition
from luxar.gsplats.seeds.utils import (
    combine_seeds,
    dedupe_farthest_first,
    local_maxima,
    sigmas_to_cholesky_isotropic,
)

__all__ = [
    # Primary public API
    "generate_seeds",  # Unified entry point (recommended)
    # Individual seeding methods
    "seed_from_decomposition",
    "seed_from_grid",
    "seed_from_edges",
    # Utility functions (for advanced usage)
    "sigmas_to_cholesky_isotropic",
    "local_maxima",
    "dedupe_farthest_first",
    "combine_seeds",
]
