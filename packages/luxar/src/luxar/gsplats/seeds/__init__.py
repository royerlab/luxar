# seeds/__init__.py
"""
Seed generation methods for Gaussian splatting.

This package provides unified seeding for Gaussian splat fitting. All methods
return GSplatData with scale-informed Gaussian shapes.

**Seeding Methods:**

1. **seed_from_gaussian**: Multiscale Gaussian-blurred peak detection.
   Each seed's sigma equals the blur scale at which it was detected.

2. **seed_from_decomposition**: Scale-hierarchical detection via image
   decomposition. Each seed's sigma equals the decomposition scale factor.

3. **seed_from_moments**: Moment-based seeding with full covariance estimation.
   Captures anisotropic (elliptical) features. Slower but most accurate shapes.

**Unified Entry Point:**

Use `generate_seeds()` for a unified interface to all methods.

Examples
--------
>>> from luxar.gsplats.seeds import generate_seeds, seed_from_gaussian
>>>
>>> # Unified entry point (recommended)
>>> seeds = generate_seeds(image, method="decomposition")
>>>
>>> # Direct method access
>>> seeds = seed_from_gaussian(image, scales=[2, 4, 8])
>>>
>>> # Use with fitter
>>> from luxar.gsplats import fit_gaussian_splats
>>> result = fit_gaussian_splats(image, seeds=seeds)
"""

from luxar.gsplats.seeds.generate import generate_seeds
from luxar.gsplats.seeds.moment_seeding import seed_from_moments
from luxar.gsplats.seeds.multiscale_decomposition import seed_from_decomposition
from luxar.gsplats.seeds.multiscale_gaussian import seed_from_gaussian
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
    "seed_from_gaussian",
    "seed_from_decomposition",
    "seed_from_moments",
    # Utility functions (for advanced usage)
    "sigmas_to_cholesky_isotropic",
    "local_maxima",
    "dedupe_farthest_first",
    "combine_seeds",
]
