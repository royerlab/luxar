"""Private support subpackage for :mod:`luxar.gsplats.lod.substitutive`.

The substitutive-LOD orchestrator (``substitutive.py``) delegates the
partition algorithms and the post-merge refinement to sibling modules here:

- :mod:`.warm_start` — Morton (Z-order) space-filling-curve partition.
- :mod:`.kmeans_lloyd` — vectorised cost-increment Lloyd refinement.
- :mod:`.greedy` — bottom-up Runnalls hierarchical merge (lazy-heap).
- :mod:`.refine` — L2 mixture-to-mixture refit of a merged level
  (``refine="l2"``; trusted-checkpoint Adam on the closed-form mixture L²).

Named with a leading underscore because a module ``substitutive.py`` and
a package ``substitutive/`` cannot coexist in one directory in CPython;
the public import path ``luxar.gsplats.lod.substitutive`` stays a file.
"""

from __future__ import annotations

import torch

# Numerical floor for divisions by per-bin mass / template norm.
_TINY = 1e-30


def chromatic_affinity(distance_sq: torch.Tensor, color_weight: float) -> torch.Tensor:
    """Return the shared exponential affinity for squared chromatic distance."""
    return torch.exp(-color_weight * distance_sq)
