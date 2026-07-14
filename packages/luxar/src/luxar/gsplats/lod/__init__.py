"""Levels-of-Detail (LOD) post-processing for fitted Gaussian-splat datasets.

Two LOD axes are implemented:

- **Additive** — same N splats, ordered so that the prefix sum at any
  k splats is the best L^2 approximation of the full scene.  Implemented
  in :mod:`luxar.gsplats.lod.additive`. Entry point:
  :func:`make_additive_lod` returns a matrix-shaped :class:`GSplatData`
  where the selected substitutive level's ``additive_prefix(k)`` is a
  valid additive prefix.

- **Substitutive** — synthesise M < N representative splats per
  coarser level via mixture reduction (supp doc
  ``substitutive_lod.tex``). Implemented in
  :mod:`luxar.gsplats.lod.substitutive`. Entry point:
  :func:`make_substitutive_lod` returns a matrix-shaped
  :class:`GSplatData` with ``n_substitutive = levels + 1`` and a single
  additive sub-LOD per substitutive level.

Both operators are pure post-processes on a fitted ``GSplatData``;
fitting (single-pass or progressive) returns a single flattened
dataset, and an LOD hierarchy is built only on demand.

The convenience function :func:`make_lod_pyramid` (in
:mod:`luxar.gsplats.lod.pyramid`) chains the two: substitutive reduction
first (outer axis), then an additive ladder inside each substitutive
level (inner axis). Saved to disk, the result is a single v3.2
node-tree ``.gsplats.zarr`` carrying the full 2-D pyramid.

The :mod:`luxar.gsplats.lod.recipes` module composes these builders into
named, scale-ordered **representation topologies** (``flat`` / ``stream`` /
``levels`` / ``tiles`` / ``overview`` / ``adaptive``) — the
``luxar gsplat lod --recipe`` CLI is a thin wrapper over :func:`build_recipe`.
"""

from __future__ import annotations

from luxar.gsplats.lod.additive import compute_additive_order, make_additive_lod
from luxar.gsplats.lod.pyramid import make_lod_pyramid
from luxar.gsplats.lod.recipes import (
    COMPOSED_RECIPES,
    MATRIX_RECIPES,
    RECIPE_NAMES,
    RecipeName,
    RecipeParams,
    RecipeResult,
    build_recipe,
)
from luxar.gsplats.lod.substitutive import make_substitutive_lod

__all__ = [
    "COMPOSED_RECIPES",
    "MATRIX_RECIPES",
    "RECIPE_NAMES",
    "RecipeName",
    "RecipeParams",
    "RecipeResult",
    "build_recipe",
    "compute_additive_order",
    "make_additive_lod",
    "make_lod_pyramid",
    "make_substitutive_lod",
]
