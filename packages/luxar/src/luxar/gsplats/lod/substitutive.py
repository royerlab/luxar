"""Substitutive Levels-of-Detail for Gaussian splat datasets.

Substitutive LOD is the second of the two LOD axes for Gaussian splats
(complementing the *additive* axis in :mod:`luxar.gsplats.lod.additive`).
Each level synthesises $\\Mlev = N/\\KK$ representative splats that
**replace** the finer level — real geometry / memory compression rather
than just a streaming order. The math derives from the supplementary
document ``luxar-paper/supp_doc/substitutive_lod/substitutive_lod.tex``;
in particular Algorithm 4.3 (cost-increment Lloyd) and the
:math:`L^2`-optimal $K$-wise merge (Prop. 2.2).

Public API
----------
:func:`make_substitutive_lod`
    Build a level-by-level hierarchy ``[level_0=data, level_1, ..., level_L]``
    by iterating the partition-and-merge operator $\\mathcal{R}_K$.

Algorithms (selected via the ``method`` argument):

- ``"auto"`` (default): resolved per level from the level's input count
  — ``"greedy"`` at or below ``5000`` splats (highest quality and fast
  there), ``"kmeans_lloyd"`` above (greedy is ~50-100x slower at large
  N). For a large dataset the coarse early levels use ``kmeans_lloyd``
  and the small later levels switch to ``greedy``.
- ``"kmeans_lloyd"`` (large-N workhorse): a Morton
  (Z-order) space-filling-curve warm start → cost-increment Lloyd
  refinement. The warm start sorts splats along the curve and chunks
  the sorted sequence into ``M = N/K`` contiguous, balanced, spatially
  coherent bins in ``O(N log N)``; Lloyd then reassigns splats to the
  template they best project onto. Per supp doc Experiment C, the
  refined ladder dominates amplitude culling on real anisotropic data.
- ``"kmeans"``: warm start only, no Lloyd refinement — the raw
  Morton-chunk partition. Fast and already high quality; the ``_lloyd``
  variant typically adds a few dB of PSNR.
- ``"greedy"``: bottom-up Runnalls-style merging using closed-form
  pairwise merge cost. Quality-leading at small $N$ and small $K$.
  Implemented as a lazy-deletion priority queue with incremental
  neighbour updates and batched pair-cost evaluation —
  ~$\\mathcal{O}(N k \\log(N k))$, a constant-factor heavier than the
  Morton warm start but usable well beyond the former
  $\\mathcal{O}(N^2)$ full re-scan.
- ``"greedy_lloyd"``: greedy warm start + Lloyd refinement.

The method names retain their ``kmeans`` prefix for API stability; the
warm start is now the ``O(N log N)`` Morton partition rather than a
global k-means++ (whose ``O(M·N) = O(N²/K)`` initialisation was
intractable once ``M = N/K`` reached tens of thousands — the
substitutive regime). Both the warm start and the vectorised Lloyd pass
avoid Python per-splat / per-bin loops: every per-bin quantity is a
segment reduction (:meth:`torch.Tensor.index_add_`) keyed by the bin
assignment, running on PyTorch (CUDA / MPS / CPU; ``device='auto'``).
Per-bin merge math (moment matching, $L^2$-optimal amplitude, residual
energy) lives in :mod:`luxar.gsplats.lod._kernels` and is shared with
the additive axis.

The returned value is a single :class:`GSplatData` with
``n_substitutive = levels + 1`` and ``M_i = 1`` per substitutive level
(one additive sub-LOD each). On disk this is a single v2.0
``.gsplats.zarr`` file with ``splats/substitutive_<s>/additive_0/``
nested groups; the legacy directory + manifest.json layout is replaced
by the unified format.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional, Union

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.gsplat_data import (
    AdditiveSubLOD,
    GSplatData,
    SubstitutiveLevel,
)
from luxar.gsplats.lod._substitutive.greedy import _greedy_partition
from luxar.gsplats.lod._substitutive.kmeans_lloyd import (
    _build_representatives_vectorized,
    _cost_increment_lloyd_vectorized,
)
from luxar.gsplats.lod._substitutive.warm_start import _morton_partition
from luxar.gsplats.utils.device import resolve_torch_device
from luxar.gsplats.utils.trils import pack_tril, unpack_tril

MethodName = Literal["kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"]
_VALID_METHODS: tuple[MethodName, ...] = (
    "kmeans",
    "kmeans_lloyd",
    "greedy",
    "greedy_lloyd",
)

# ``method="auto"`` resolves per level: greedy (highest quality, and fastest at
# small N) when a level's input count is at or below this threshold, otherwise
# kmeans_lloyd (greedy is ~50-100x slower above ~25K splats). At the threshold
# greedy takes only a few seconds.
AutoOrMethod = Literal["auto", "kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"]
_VALID_CHOICES: tuple[str, ...] = ("auto", *_VALID_METHODS)
_AUTO_GREEDY_MAX_N = 5000


def _resolve_method(method: AutoOrMethod, n_in: int) -> MethodName:
    """Resolve ``method`` for a level of ``n_in`` splats (handles ``"auto"``)."""
    if method != "auto":
        return method
    return "greedy" if n_in <= _AUTO_GREEDY_MAX_N else "kmeans_lloyd"


def _pack_level(
    data: GSplatData,
    *,
    compression_factor: int,
    parent_method: Optional[MethodName],
    level_index: int,
    stats: dict,
) -> SubstitutiveLevel:
    """Wrap one splat set as a single-additive-sub-LOD :class:`SubstitutiveLevel`.

    Substitutive reduction emits exactly one additive sub-LOD per level
    (``lod_method="none"``); this collapses the identical
    ``SubstitutiveLevel(additive_sublods=[AdditiveSubLOD(...)])`` packing
    used for the finest level, the normal reduced levels, and the
    ``input_too_small`` early-stop level.
    """
    return SubstitutiveLevel(
        additive_sublods=[
            AdditiveSubLOD(
                # np.array (copy) not np.asarray: ``data`` may be a flattened()
                # view whose arrays are read-only; a packed level must own
                # writable arrays.
                centers=np.array(data.centers, dtype=np.float32),
                amplitudes=np.array(data.amplitudes, dtype=np.float32),
                cholesky_factors=np.array(data.cholesky_factors, dtype=np.float32),
                colors=(np.array(data.colors) if data.colors is not None else None),
                stats={"lod_method": "none", "lod_level": 0},
                truncation_radius=data.truncation_radius,
            )
        ],
        compression_factor=compression_factor,
        parent_method=parent_method,
        level_index=level_index,
        stats=stats,
    )


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def make_substitutive_lod(
    data: GSplatData,
    *,
    compression_factor: int = 4,
    levels: int = 3,
    method: AutoOrMethod = "auto",
    lloyd_iterations: int = 5,
    candidate_bins_k: int = 12,
    device: Union[str, torch.device, None] = "auto",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> GSplatData:
    """Build a substitutive-LOD hierarchy.

    Parameters
    ----------
    data
        Source dataset. If multi-substitutive, only its default
        substitutive level is reduced (additive sub-LODs at that
        level are flattened first).
    compression_factor
        Per-level branching factor $K$. Each level ``ℓ`` has
        ``ceil(N / K^ℓ)`` splats.
    levels
        Number of *coarser* levels to produce. The returned object has
        ``levels + 1`` substitutive levels (the original at index 0).
    method
        Partition algorithm, or ``"auto"`` (default). ``"auto"`` resolves
        per level: ``"greedy"`` when the level's input has ``<= 5000``
        splats (highest quality, and fast there) and ``"kmeans_lloyd"``
        above (greedy is ~50-100x slower at large N). See module docstring
        for the individual methods.
    lloyd_iterations
        Maximum number of cost-increment Lloyd passes per level (only
        for ``"kmeans_lloyd"`` / ``"greedy_lloyd"``). The loop exits
        early as soon as a pass fails to improve the projection energy.
    candidate_bins_k
        Number of Morton-curve neighbours whose current bins are the
        move candidates for each splat during Lloyd refinement. Tighter
        k → faster, slightly worse quality.
    device
        ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"mps"``, or a
        :class:`torch.device`.
    seed
        Accepted for API stability; the Morton warm start and the
        synchronous Lloyd pass are deterministic, so it has no effect on
        the ``kmeans*`` methods.
    verbose
        Per-level Arbol logging.

    Returns
    -------
    GSplatData
        A v2.0 dataset with ``n_substitutive = levels + 1`` and a
        single additive sub-LOD per substitutive level (the finest at
        ``substitutive_levels[0]``).

    Raises
    ------
    ValueError
        If ``compression_factor < 2``, ``levels < 1``, or ``method``
        is not recognised.
    """
    if compression_factor < 2:
        raise ValueError(f"compression_factor must be >= 2, got {compression_factor}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if method not in _VALID_CHOICES:
        raise ValueError(
            f"method must be one of {list(_VALID_CHOICES)}, got {method!r}"
        )
    K = int(compression_factor)
    L_levels = int(levels)

    # Flatten the input's default substitutive level (and its additive
    # sub-LODs) down to a single splat set. Substitutive reduction always
    # operates on the finest level; non-default substitutive levels of the
    # input are discarded by design (substitutive composes with itself by
    # taking the finest as the new finest).
    src = data.flattened()
    if src.n_substitutive > 1:
        src = src.at_substitutive(src.default_substitutive)
    target_device = resolve_torch_device(
        device if not isinstance(device, str) or device != "auto" else None
    )
    # Lloyd's move-acceptance test (1e-12 tolerance on a residual-energy
    # delta) requires float64, which MPS does not support. CPU + float64 is
    # the honest fallback; the algorithm already round-trips through CPU
    # for the spatial-hash and knn queries, so MPS speedup was partial.
    if target_device.type == "mps":
        warnings.warn(
            "make_substitutive_lod: MPS backend lacks float64 support; "
            "falling back to CPU. Pass device='cpu' explicitly to silence.",
            RuntimeWarning,
            stacklevel=2,
        )
        target_device = torch.device("cpu")

    # Collect per-level outputs and pack them as SubstitutiveLevels.
    sub_levels: list[SubstitutiveLevel] = [
        _pack_level(
            src,
            compression_factor=1,
            parent_method=None,
            level_index=0,
            stats={"n_splats_total": int(src.n_splats)},
        )
    ]

    current = src
    for level_idx in range(1, L_levels + 1):
        N_in = current.n_splats
        if N_in <= 1:
            # Cannot reduce further; emit the unchanged dataset and stop.
            sub_levels.append(
                _pack_level(
                    current,
                    compression_factor=K**level_idx,
                    parent_method=_resolve_method(method, N_in),
                    level_index=level_idx,
                    stats={
                        "n_splats_total": int(N_in),
                        "stop_reason": "input_too_small",
                    },
                )
            )
            break
        M_target = max(1, math.ceil(N_in / K))
        level_method = _resolve_method(method, N_in)
        if verbose:
            label = (
                f"{level_method} (auto)" if method == "auto" else level_method
            )
            with asection(
                f"Substitutive level {level_idx}: {N_in} -> {M_target} splats"
            ):
                aprint(f"method={label}")
                new_data = _reduce_one_level(
                    current,
                    M_target=M_target,
                    method=level_method,
                    lloyd_iterations=lloyd_iterations,
                    candidate_bins_k=candidate_bins_k,
                    device=target_device,
                )
        else:
            new_data = _reduce_one_level(
                current,
                M_target=M_target,
                method=level_method,
                lloyd_iterations=lloyd_iterations,
                candidate_bins_k=candidate_bins_k,
                device=target_device,
            )
        sub_levels.append(
            _pack_level(
                new_data,
                compression_factor=K**level_idx,
                parent_method=level_method,
                level_index=level_idx,
                stats={"n_splats_total": int(new_data.n_splats)},
            )
        )
        current = new_data

    out_stats = dict(src.stats)
    out_stats.update(
        {
            "lod_kind": "substitutive",
            "compression_factor": K,
            "method": method,
            "n_substitutive_levels": len(sub_levels),
        }
    )
    return GSplatData.from_substitutive_levels(sub_levels, stats=out_stats)


# ─────────────────────────────────────────────────────────────────────
# Per-level reduction
# ─────────────────────────────────────────────────────────────────────


def _reduce_one_level(
    data: GSplatData,
    *,
    M_target: int,
    method: MethodName,
    lloyd_iterations: int,
    candidate_bins_k: int,
    device: torch.device,
) -> GSplatData:
    """Run one application of the partition-and-merge operator $\\mathcal{R}_K$."""
    D = data.ndim
    # np.array (copy) not np.asarray: ``data`` may be a flattened() view with
    # read-only arrays, which torch.from_numpy rejects (non-writable). The
    # copy is immediately recast to float64 on-device, so it is near-free.
    centres_t = torch.from_numpy(np.array(data.centers, dtype=np.float32)).to(
        device=device, dtype=torch.float64
    )
    L_t = torch.from_numpy(
        unpack_tril(np.asarray(data.cholesky_factors, dtype=np.float32), D).astype(
            np.float64
        )
    ).to(device=device)
    amps_t = torch.from_numpy(np.array(data.amplitudes, dtype=np.float32)).to(
        device=device, dtype=torch.float64
    )
    if data.colors is not None:
        colors_t: Optional[torch.Tensor] = torch.from_numpy(np.array(data.colors)).to(
            device=device
        )
    else:
        colors_t = None

    # 1) Warm-start partition.
    #
    # ``kmeans*`` methods use an O(N log N) Morton-order space-filling-curve
    # partition: sort splats along the curve and chunk into M contiguous bins
    # of ~K spatially-coherent splats. This replaces a global k-means++ warm
    # start whose init was O(M·N) = O(N²/K) — intractable once M = N/K reaches
    # tens of thousands (the substitutive regime). Shape-incompatible splats
    # that happen to be spatial neighbours are separated by the cost-aware
    # Lloyd pass below, which is shape-aware via the Gaussian inner product.
    if method.startswith("kmeans"):
        assignments = _morton_partition(centres_t, M=M_target)
    else:
        assignments = _greedy_partition(centres_t, L_t, amps_t, M_target=M_target)

    # 2) Optional Lloyd cost-increment refinement (vectorised, monotone).
    if method.endswith("_lloyd"):
        assignments = _cost_increment_lloyd_vectorized(
            centres_t,
            L_t,
            amps_t,
            assignments,
            M=M_target,
            iterations=lloyd_iterations,
            candidate_bins_k=candidate_bins_k,
            device=device,
        )

    # 3) Bin merge: produce M representative splats (vectorised segment ops).
    new_centres, new_L, new_amps, new_colors = _build_representatives_vectorized(
        centres_t, L_t, amps_t, colors_t, assignments, M=M_target
    )

    # Cull empty / degenerate bins (zero optimal amplitude).
    keep_mask = new_amps > 0
    if not bool(torch.all(keep_mask)):
        new_centres = new_centres[keep_mask]
        new_L = new_L[keep_mask]
        new_amps = new_amps[keep_mask]
        if new_colors is not None:
            new_colors = new_colors[keep_mask]

    # Pack back to the GSplatData format.
    centres_np = new_centres.detach().cpu().numpy().astype(np.float32)
    chol_np = pack_tril(new_L.detach().cpu().numpy()).astype(np.float32)
    amps_np = new_amps.detach().cpu().numpy().astype(np.float32)
    colors_np = new_colors.detach().cpu().numpy() if new_colors is not None else None

    return GSplatData(
        centers=centres_np,
        amplitudes=amps_np,
        cholesky_factors=chol_np,
        colors=colors_np,
        truncation_radius=data.truncation_radius,
    )
