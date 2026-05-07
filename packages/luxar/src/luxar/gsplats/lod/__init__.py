"""Levels-of-Detail (LOD) post-processing for fitted Gaussian-splat datasets.

Two LOD axes are envisioned (see ``additive_lod`` and ``substitutive_lod``
supplementary documents):

- **Additive** — same N splats, ordered so that the prefix sum at any
  k splats is the best L^2 approximation of the full scene.  Implemented
  in :mod:`luxar.gsplats.lod.additive`.
- **Substitutive** — synthesise M < N representative splats per coarser
  level via mixture reduction.  Not yet implemented; will live in
  ``luxar.gsplats.lod.substitutive``.

The additive operator is a pure post-process on a fitted ``GSplatData``;
fitting (single-pass or progressive) returns a single flattened dataset,
and an LOD ladder is built only on demand via
:func:`make_additive_lod`.
"""

from luxar.gsplats.lod.additive import (
    compute_additive_order,
    make_additive_lod,
)

__all__ = [
    "compute_additive_order",
    "make_additive_lod",
]
