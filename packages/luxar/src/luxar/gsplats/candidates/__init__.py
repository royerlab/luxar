# candidates/__init__.py
"""
Candidate generation methods for Gaussian splatting.

This package provides two complementary approaches for generating candidate
splat locations:

1. **Multiscale Gaussian** (`find_candidates_multiscale_gaussian`): Multiscale
   Gaussian-blurred peak detection with optional CLAHE preprocessing.

2. **Decomposition** (`find_candidates_from_decomposition`): Scale-hierarchical
   detection via image decomposition for principled scale separation.

Examples
--------
>>> from luxar.gsplats.candidates import (
...     find_candidates_multiscale_gaussian,
...     find_candidates_from_decomposition,
... )
>>>
>>> # Multiscale Gaussian method (standard)
>>> candidates1 = find_candidates_multiscale_gaussian(image)
>>>
>>> # Decomposition method (new, noise-robust)
>>> candidates2 = find_candidates_from_decomposition(
...     image,
...     scales=[1, 2, 4, 8],
...     ignore_finest_k=1,
... )
"""

from luxar.gsplats.candidates.decomposition import find_candidates_from_decomposition
from luxar.gsplats.candidates.multiscale_gaussian import (
    find_candidates_multiscale_gaussian,
)
from luxar.gsplats.candidates.utils import (
    combine_candidates,
    dedupe_farthest_first,
    local_maxima,
)

__all__ = [
    # Primary public APIs
    "find_candidates_multiscale_gaussian",
    "find_candidates_from_decomposition",
    # Utility functions (for advanced usage and testing)
    "local_maxima",
    "dedupe_farthest_first",
    "combine_candidates",
]
