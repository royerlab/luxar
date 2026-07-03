"""Full 2-D LOD pyramid (substitutive × additive) for Gaussian splats.

The convenience entry point :func:`make_lod_pyramid` chains the two LOD
axes: substitutive reduction first (outer axis, see
:mod:`luxar.gsplats.lod.substitutive`), then an additive ladder inside
each substitutive level (inner axis, see
:mod:`luxar.gsplats.lod.additive`). The result is a single v2.0
``.gsplats.zarr`` carrying the full 2-D pyramid.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.additive import (
    AutoOrMethod as AdditiveMethodName,
)
from luxar.gsplats.lod.additive import (
    BreakpointSpec,
    clamp_counts_breakpoints,
    make_additive_lod,
    validate_counts_breakpoints,
)
from luxar.gsplats.lod.substitutive import (
    AutoOrMethod as SubstitutiveMethodName,
)
from luxar.gsplats.lod.substitutive import (
    make_substitutive_lod,
)


def make_lod_pyramid(
    data: GSplatData,
    *,
    compression_factor: int = 4,
    levels: int = 3,
    substitutive_method: SubstitutiveMethodName = "auto",
    lloyd_iterations: int = 5,
    candidate_bins_k: int = 12,
    coverage_inflation: float = 3.0,
    device: Union[str, torch.device, None] = "auto",
    coarsen_dims: Optional[Sequence[int]] = None,
    n_additive_lods: int = 4,
    additive_method: AdditiveMethodName = "auto",
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
    substitutive_method, lloyd_iterations, candidate_bins_k, coverage_inflation, device
        Substitutive axis algorithm parameters (``coverage_inflation``
        is the anti-grid inter-spread widening; see
        :func:`make_substitutive_lod`).
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
    # Explicit `counts:` breakpoints must still be sane for the FULL dataset:
    # the finest pyramid level IS the input's (flattened) default substitutive
    # level, so a largest count exceeding that N is a whole-dataset-scale typo
    # and aborts loudly here — BEFORE the expensive substitutive reduction.
    # Only the coarser (smaller-by-K^s) levels clamp, in the loop below.
    validate_counts_breakpoints(
        breakpoints,
        data.substitutive_levels[data.default_substitutive].n_splats_total,
    )

    pyramid = make_substitutive_lod(
        data,
        compression_factor=compression_factor,
        levels=levels,
        method=substitutive_method,
        lloyd_iterations=lloyd_iterations,
        candidate_bins_k=candidate_bins_k,
        coverage_inflation=coverage_inflation,
        device=device,
        seed=seed,
        coarsen_dims=coarsen_dims,
        verbose=verbose,
    )

    # Build an additive ladder on each substitutive level. Explicit `counts:`
    # breakpoints (strictly validated against the full dataset above) are
    # clamped to each level's own splat count — coarser levels are smaller by
    # K^s, so a fixed counts list sized for the finest level would otherwise
    # abort the build ("largest breakpoint exceeds N"). String/energy specs
    # pass through (already size-adaptive).
    out = pyramid
    for s in range(out.n_substitutive):
        level_n = out.at_substitutive(s).n_splats
        out = make_additive_lod(
            out,
            n_lods=n_additive_lods,
            method=additive_method,
            breakpoints=clamp_counts_breakpoints(breakpoints, level_n),
            truncation_sigmas=truncation_sigmas,
            max_n_dense=max_n_dense,
            seed=None if seed is None else seed + s,
            substitutive_level=s,
        )
    return out
