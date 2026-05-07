"""Levels-of-Detail (LOD) post-processing for fitted Gaussian-splat datasets.

Two LOD axes are implemented:

- **Additive** — same N splats, ordered so that the prefix sum at any
  k splats is the best L^2 approximation of the full scene.  Implemented
  in :mod:`luxar.gsplats.lod.additive`. Entry point:
  :func:`make_additive_lod` returns a multi-LOD :class:`GSplatData`
  where ``up_to_lod(k)`` is a valid additive prefix.

- **Substitutive** — synthesise M < N representative splats per
  coarser level via mixture reduction (supp doc
  ``substitutive_lod.tex``). Implemented in
  :mod:`luxar.gsplats.lod.substitutive`. Entry point:
  :func:`make_substitutive_lod` returns a list of flat
  :class:`GSplatData` (one per level), since each level *replaces*
  the previous one rather than extending it.

Both operators are pure post-processes on a fitted ``GSplatData``;
fitting (single-pass or progressive) returns a single flattened
dataset, and an LOD hierarchy is built only on demand.
"""

from luxar.gsplats.lod.additive import (
    compute_additive_order,
    make_additive_lod,
)
from luxar.gsplats.lod.substitutive import (
    make_substitutive_lod,
)

__all__ = [
    "compute_additive_order",
    "make_additive_lod",
    "make_substitutive_lod",
]
