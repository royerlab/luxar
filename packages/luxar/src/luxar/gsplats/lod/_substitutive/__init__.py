"""Private support subpackage for :mod:`luxar.gsplats.lod.substitutive`.

The substitutive-LOD orchestrator (``substitutive.py``) delegates the
three partition algorithms to sibling modules here:

- :mod:`.warm_start` — Morton (Z-order) space-filling-curve partition.
- :mod:`.kmeans_lloyd` — vectorised cost-increment Lloyd refinement.
- :mod:`.greedy` — bottom-up Runnalls hierarchical merge (lazy-heap).

Named with a leading underscore because a module ``substitutive.py`` and
a package ``substitutive/`` cannot coexist in one directory in CPython;
the public import path ``luxar.gsplats.lod.substitutive`` stays a file.
"""

from __future__ import annotations

# Numerical floor for divisions by per-bin mass / template norm.
_TINY = 1e-30
