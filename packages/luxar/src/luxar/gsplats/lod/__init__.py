"""Levels-of-Detail (LOD) post-processing for fitted Gaussian-splat datasets.

Two LOD axes are implemented:

- **Additive** — same N splats, ordered so that the prefix sum at any
  k splats is the best L^2 approximation of the full scene.  Implemented
  in :mod:`luxar.gsplats.lod.additive`. Entry point:
  :func:`make_additive_lod` returns a v2.0 :class:`GSplatData` where the
  selected substitutive level's ``additive_prefix(k)`` is a valid
  additive prefix.

- **Substitutive** — synthesise M < N representative splats per
  coarser level via mixture reduction (supp doc
  ``substitutive_lod.tex``). Implemented in
  :mod:`luxar.gsplats.lod.substitutive`. Entry point:
  :func:`make_substitutive_lod` returns a v2.0 :class:`GSplatData` with
  ``n_substitutive = levels + 1`` and a single additive sub-LOD per
  substitutive level.

Both operators are pure post-processes on a fitted ``GSplatData``;
fitting (single-pass or progressive) returns a single flattened
dataset, and an LOD hierarchy is built only on demand.

The convenience function :func:`make_lod_pyramid` (in
:mod:`luxar.gsplats.lod.pyramid`) chains the two: substitutive reduction
first (outer axis), then an additive ladder inside each substitutive
level (inner axis). The result is a single v2.0 ``.gsplats.zarr``
carrying the full 2-D pyramid.
"""

from __future__ import annotations

from luxar.gsplats.lod.additive import compute_additive_order, make_additive_lod
from luxar.gsplats.lod.pyramid import make_lod_pyramid
from luxar.gsplats.lod.substitutive import make_substitutive_lod

__all__ = [
    "compute_additive_order",
    "make_additive_lod",
    "make_lod_pyramid",
    "make_substitutive_lod",
]
