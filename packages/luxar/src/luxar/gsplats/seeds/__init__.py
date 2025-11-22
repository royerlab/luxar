# seeds/__init__.py
"""
Seed generation methods for Gaussian splatting.

This package provides two complementary approaches for generating seed
splat locations:

1. **Multiscale Gaussian** (`find_seeds_multiscale_gaussian`): Multiscale
   Gaussian-blurred peak detection with optional CLAHE preprocessing.

2. **Decomposition** (`find_seeds_multiscale_decomposition`): Scale-hierarchical
   detection via image decomposition for principled scale separation.

Examples
--------
>>> from luxar.gsplats.seeds import (
...     find_seeds_multiscale_gaussian,
...     find_seeds_multiscale_decomposition,
... )
>>>
>>> # Multiscale Gaussian method (standard)
>>> seeds1 = find_seeds_multiscale_gaussian(image)
>>>
>>> # Decomposition method (new, noise-robust)
>>> seeds2 = find_seeds_multiscale_decomposition(
...     image,
...     scales=[1, 2, 4, 8],
...     ignore_finest_k=1,
... )
"""

from luxar.gsplats.seeds.generate import generate_seeds
from luxar.gsplats.seeds.multiscale_decomposition import (
    find_seeds_multiscale_decomposition,
)
from luxar.gsplats.seeds.multiscale_gaussian import find_seeds_multiscale_gaussian
from luxar.gsplats.seeds.utils import (
    combine_seeds,
    dedupe_farthest_first,
    local_maxima,
)

__all__ = [
    # Primary public API
    "generate_seeds",  # Unified entry point (recommended)
    # Individual methods
    "find_seeds_multiscale_gaussian",
    "find_seeds_multiscale_decomposition",
    # Utility functions (for advanced usage and testing)
    "local_maxima",
    "dedupe_farthest_first",
    "combine_seeds",
]
