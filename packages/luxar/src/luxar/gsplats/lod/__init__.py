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

The convenience function :func:`make_lod_pyramid` chains the two:
substitutive reduction first (outer axis), then an additive ladder
inside each substitutive level (inner axis). The result is a single v2.0
``.gsplats.zarr`` carrying the full 2-D pyramid.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.additive import (
    BreakpointSpec,
    compute_additive_order,
    make_additive_lod,
)
from luxar.gsplats.lod.additive import (
    MethodName as AdditiveMethodName,
)
from luxar.gsplats.lod.substitutive import (
    AutoOrMethod as SubstitutiveMethodName,
)
from luxar.gsplats.lod.substitutive import (
    make_substitutive_lod,
)

__all__ = [
    "compute_additive_order",
    "make_additive_lod",
    "make_lod_pyramid",
    "make_substitutive_lod",
]


def make_lod_pyramid(
    data: GSplatData,
    *,
    compression_factor: int = 4,
    levels: int = 3,
    substitutive_method: SubstitutiveMethodName = "auto",
    lloyd_iterations: int = 5,
    candidate_bins_k: int = 12,
    device: Union[str, torch.device, None] = "auto",
    n_additive_lods: int = 4,
    additive_method: AdditiveMethodName = "greedy",
    breakpoints: BreakpointSpec = "equal-count",
    truncation_sigmas: float = 3.0,
    max_n_dense: int = 2_000,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> GSplatData:
    """Build the full 2-D LOD pyramid (substitutive × additive) in one call.

    The pipeline runs :func:`make_substitutive_lod` first (outer axis)
    and then calls :func:`make_additive_lod` on each substitutive level
    (inner axis). The result is a single v2.0 :class:`GSplatData` with
    ``n_substitutive = levels + 1`` and ``M_i = n_additive_lods`` (or as
    resolved by ``breakpoints``) per level.

    Parameters
    ----------
    data
        Source fitted gsplat dataset.
    compression_factor, levels
        Substitutive axis parameters (passed to
        :func:`make_substitutive_lod`).
    substitutive_method, lloyd_iterations, candidate_bins_k, device
        Substitutive axis algorithm parameters.
    n_additive_lods, additive_method, breakpoints
        Additive axis parameters (passed to :func:`make_additive_lod`).
    truncation_sigmas, max_n_dense
        Additive axis algorithmic knobs.
    seed
        Optional shared seed (per-axis offsets are added internally).
    verbose
        Per-step Arbol logging from substitutive reduction.

    Returns
    -------
    GSplatData
        A v2.0 dataset with the full ``[levels+1, n_additive_lods]``
        pyramid.
    """
    pyramid = make_substitutive_lod(
        data,
        compression_factor=compression_factor,
        levels=levels,
        method=substitutive_method,
        lloyd_iterations=lloyd_iterations,
        candidate_bins_k=candidate_bins_k,
        device=device,
        seed=seed,
        verbose=verbose,
    )

    # Build an additive ladder on each substitutive level.
    out = pyramid
    for s in range(out.n_substitutive):
        out = make_additive_lod(
            out,
            n_lods=n_additive_lods,
            method=additive_method,
            breakpoints=breakpoints,
            truncation_sigmas=truncation_sigmas,
            max_n_dense=max_n_dense,
            seed=None if seed is None else seed + s,
            substitutive_level=s,
        )
    return out
