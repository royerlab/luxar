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

**Coverage inflation** (``coverage_inflation``, default 3.0): every method
finishes with a merge whose covariance is the bin's moment match
(intra + inter spread). For balanced spatial bins of pitch ``d`` the
moment-matched σ is ≈ ``d/√12`` ≈ 0.29 d — well below the σ ≳ d/2 a
lattice of Gaussians needs to sum flat — and, because the Morton warm
start quantises bin boundaries onto a *global dyadic grid*, the coverage
dips align into coherent axis-aligned planes: a very visible grid
pattern at every coarse level. The fix widens the inter-center term only
(``Σ_out = intra + β·inter``; β=3 turns d²/12 into (d/2)²) with a
mass-preserving amplitude rescale, and is the exact fixed point of the
level recurrence so it stays calibrated at every depth. Set
``coverage_inflation=1.0`` for the historical pure moment match.

**L2 refinement** (``refine="l2"``, opt-in): after each merge, the level is
Adam-optimized against its fine input under the closed-form mixture L²
(:mod:`._substitutive.refine`) — the merge (with its β=3 inflation) becomes
the optimizer *seed*, and the refit takes over the exact calibration. The
refit is never worse than the merge in its trusted metric, keeps total mass
pinned to the fine mixture's (no brightness pop across levels), and freezes
barrier dims under ``coarsen_dims`` grouping.

The returned value is a single :class:`GSplatData` with
``n_substitutive = levels + 1`` and ``M_i = 1`` per substitutive level
(one additive sub-LOD each). Saved to disk, this becomes a single v3.4
node-tree ``.gsplats.zarr`` (a ``kind=lod`` group with one child per
level — see :mod:`luxar.gsplats.tree`).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union, cast

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
from luxar.gsplats.lod._substitutive.refine import (
    L2RefineConfig,
    l2_refine_mixture,
)
from luxar.gsplats.lod._substitutive.warm_start import _morton_partition
from luxar.gsplats.utils.device import resolve_torch_device
from luxar.gsplats.utils.trils import pack_tril, unpack_tril

if TYPE_CHECKING:
    from luxar.gsplats.lod.volume_refit import VolumeRefitConfig

MethodName = Literal["kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"]
_VALID_METHODS: tuple[MethodName, ...] = (
    "kmeans",
    "kmeans_lloyd",
    "greedy",
    "greedy_lloyd",
)

#: Post-merge per-level refinement of the substitutive reduction. ``"l2"``
#: Adam-optimizes each merged level against its fine input under the
#: closed-form mixture L² (see :mod:`._substitutive.refine`). ``"volume"``
#: warm-start re-fits each merged level against the source volume itself
#: (see :mod:`.volume_refit`; requires the ``volume`` argument).
RefineName = Literal["none", "l2", "volume"]
_VALID_REFINE: tuple[RefineName, ...] = ("none", "l2", "volume")


def _merge_refine_stats(sink: dict, rstats: dict) -> None:
    """Aggregate per-group refine stats into a per-level sink.

    Counters and wall time sum; trusted E values sum (the objective is additive
    over disjoint barrier groups, so the summed improvement fraction is the
    level's); booleans OR; the mass ratios are averaged over groups.
    """
    for key in ("iters_run", "rebuilds", "nan_grad_skips", "wall_s"):
        sink[key] = sink.get(key, 0) + rstats.get(key, 0)
    for key in ("trusted_E_seed", "trusted_E_best"):
        if key in rstats:
            sink[key] = sink.get(key, 0.0) + rstats[key]
    for key in ("minibatched", "mass_drift_warning"):
        sink[key] = bool(sink.get(key, False) or rstats.get(key, False))
    for key in ("mass_vs_seed", "mass_vs_fine"):
        if key in rstats:
            n = sink.get("_mass_n", 0)
            sink[key] = (sink.get(key, 0.0) * n + rstats[key]) / (n + 1)
    if "mass_vs_fine" in rstats:
        sink["_mass_n"] = sink.get("_mass_n", 0) + 1
    if "trusted_E_seed" in sink:
        seed_e = float(sink["trusted_E_seed"])
        best_e = float(sink.get("trusted_E_best", seed_e))
        # Presence-gated (not truthiness: an exactly-0.0 summed seed objective
        # is valid and must still report the improvement). Normalize by
        # |seed E|; in the degenerate zero-seed case fall back to |best E| so
        # a real improvement yields a finite, meaningful fraction.
        denom = abs(seed_e) if seed_e != 0.0 else abs(best_e)
        sink["improvement_frac"] = (seed_e - best_e) / max(denom, 1e-30)


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


def _finest_content(data: GSplatData) -> GSplatData:
    """Flatten *data* down to the single splat set a reduction operates on.

    The input's default substitutive level (and its additive sub-LODs) collapse
    to one set: substitutive reduction always operates on the finest level, and
    non-default substitutive levels of the input are discarded by design
    (substitutive composes with itself by taking the finest as the new finest).
    """
    src = data.flattened()
    if src.n_substitutive > 1:
        src = src.at_substitutive(src.default_substitutive)
    return src


def _refuse_categorical_coarsening(data: GSplatData) -> None:
    if data.label_ids is not None:
        raise ValueError(
            "cannot coarsen: input carries categorical channel 'label_ids'; "
            "merging would have to combine class ids, and there is no meaningful "
            "combination of two class ids. Call without_label_ids() first if a "
            "coarse level is what you want."
        )


def _resolve_reduction_device(
    device: Union[str, torch.device, None], *, caller: str
) -> torch.device:
    """Resolve the reduction device, downgrading MPS to CPU.

    Lloyd's move-acceptance test (1e-12 tolerance on a residual-energy delta)
    requires float64, which MPS does not support. CPU + float64 is the honest
    fallback; the algorithm already round-trips through CPU for the spatial-hash
    and knn queries, so the MPS speedup was partial anyway.
    """
    target_device = resolve_torch_device(device)
    if target_device.type == "mps":
        warnings.warn(
            f"{caller}: MPS backend lacks float64 support; falling back to CPU. "
            "Pass device='cpu' explicitly to silence.",
            RuntimeWarning,
            stacklevel=3,
        )
        target_device = torch.device("cpu")
    return target_device


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
# Barrier-dim grouping (coarsen_dims)
# ─────────────────────────────────────────────────────────────────────


def _normalise_coarsen_dims(
    coarsen_dims: Optional[Sequence[int]], src: GSplatData
) -> Optional[tuple[int, ...]]:
    """Validate ``coarsen_dims`` against ``src.ndim``.

    Returns the sorted unique tuple of allowed-coarsen dims, or ``None`` to mean
    "coarsen over all dims" (the historical path). Passing every dim normalises
    to ``None`` (no barrier). Warns if the implied barrier dims look continuous
    (so many distinct values that grouping would disable coarsening).
    """
    if coarsen_dims is None:
        return None
    d_total = src.ndim
    cd = sorted({int(d) for d in coarsen_dims})
    if not cd:
        raise ValueError("coarsen_dims must be non-empty")
    for d in cd:
        if d < 0 or d >= d_total:
            raise ValueError(f"coarsen_dims index {d} out of range for ndim={d_total}")
    if len(cd) == d_total:
        return None  # no barrier dims -> identical to the all-dims path
    barrier = [d for d in range(d_total) if d not in set(cd)]
    keys = np.asarray(src.centers)[:, barrier]
    n_groups = len(np.unique(keys, axis=0))
    if n_groups > 0.5 * max(int(src.n_splats), 1):
        warnings.warn(
            f"coarsen_dims barrier {barrier} yields {n_groups} groups for "
            f"{src.n_splats} splats: the barrier dims look continuous, so "
            "coarsening will be negligible. Barrier dims should be discrete "
            "(categorical / timepoint / channel).",
            RuntimeWarning,
            stacklevel=3,
        )
    return tuple(cd)


def resolved_merge_coarsen_dims(
    coarsen_dims: Optional[Sequence[int]], ndim: Optional[int]
) -> list[int]:
    """The dims a substitutive reduction ACTUALLY coarsens over, spelled EXPLICITLY.

    The single resolution shared by every path that WRITES the ``coarsen_dims``
    stamp (:func:`make_substitutive_lod` here, :func:`~luxar.gsplats.lod.decimate
    .decimate`'s ``merge`` family, and the ``batch-fit merge`` per-part record)
    — one function so a fourth one cannot quietly publish the same choice a
    second way.

    Three substitutive producers write NO stamp at all and are therefore not
    reached by this: ``lod --recipe adaptive`` / ``--recipe overview`` and
    ``fit --recipe levels`` build their ``pipeline/`` group out of their INPUT's
    stats rather than out of the recipe they ran, so the reduction's own choice
    — an explicit ``--coarsen-dims`` included — never lands on disk, and an
    absent key reads exactly like the ``null`` below. Routing those through here
    means plumbing a composed recipe's parameters into its record, which is a
    separate change tracked on #1600.

    Always a non-empty literal list, never ``None`` — including for the
    coarsen-everything case (the ``coarsen_dims=None`` default, and a request
    naming every dim, which :func:`_normalise_coarsen_dims` collapses to the
    same thing). An empty explicit request is invalid: its empty complement
    would claim every axis as a barrier. The two valid coarsen-everything
    spellings are NOT interchangeable on disk:
    :func:`~luxar.gsplats.io.save_gsplats._barrier_from_coarsen_dims` cannot tell
    a written ``null`` from an absent key, so both read as "no provenance" and
    fall through to ``detect_barrier_dims`` auto-detection — a GUESS about the
    result's coordinates, not "no barrier".

    What that guess costs depends on the data, and it was measured rather than
    asserted (#1600 review). Auto-detection re-imposes the very barrier this
    merge blended over exactly when the reduction leaves the stacked axis' grid
    INTACT: on 200 4D splats over three timepoints spaced 1000 apart against a
    spatial extent of 100, no cluster ever spans two timepoints, the coordinates
    stay integral, and the fallback hands back ``[3]``. On a fine grid (step 1)
    the merge averages those coordinates away, the axis stops looking integral,
    and the fallback finds nothing — but only on the levels it actually merged,
    so a ladder came out with a per-level MIXTURE (``[[], [], [3]]``: the finest
    level is the unreduced input and keeps its integral grid). ``[0, …, d-1]``
    asserts the empty complement outright on either grid and on every level,
    i.e. the no-barrier layout the reduction actually earned.

    ``ndim`` is only read to EXPAND a ``None`` request, so a caller that always
    names its dims may pass ``None`` for it rather than a stand-in width — a
    made-up width is the one thing this must not appear to assert. The two
    ``None``\\ s together are a caller bug, not a coarsen-everything answer, and
    raise instead of returning the empty list (whose complement is every axis a
    barrier — the splat-dropping direction).
    """
    if coarsen_dims is None:
        if ndim is None:
            raise ValueError(
                "resolved_merge_coarsen_dims: a coarsen-everything request "
                "(coarsen_dims=None) needs the width to expand it over, but "
                "ndim is None too."
            )
        return list(range(int(ndim)))
    resolved = sorted({int(d) for d in coarsen_dims})
    if not resolved:
        raise ValueError("coarsen_dims must be non-empty")
    return resolved


def _subset_gsplatdata(data: GSplatData, mask: np.ndarray) -> GSplatData:
    """Boolean-index a flat ``GSplatData`` (preserves truncation_radius)."""
    colors = np.asarray(data.colors)[mask] if data.colors is not None else None
    return GSplatData(
        centers=np.asarray(data.centers)[mask],
        amplitudes=np.asarray(data.amplitudes)[mask],
        cholesky_factors=np.asarray(data.cholesky_factors)[mask],
        colors=colors,
        truncation_radius=data.truncation_radius,
    )


def _drop_nonpositive(data: GSplatData) -> GSplatData:
    """Drop non-positive-amplitude splats (parity with _reduce_one_level's cull
    for the kept-as-is small-group branch). No-op for the usual all-positive
    lifted input."""
    amps = np.asarray(data.amplitudes)
    if amps.size == 0 or bool((amps > 0).all()):
        return data
    return _subset_gsplatdata(data, amps > 0)


def _concat_gsplatdata(parts: list[GSplatData]) -> GSplatData:
    """Concatenate flat ``GSplatData`` parts (colors kept iff all present)."""
    keep_colors = all(p.colors is not None for p in parts)
    return GSplatData(
        centers=np.concatenate([np.asarray(p.centers) for p in parts], axis=0),
        amplitudes=np.concatenate([np.asarray(p.amplitudes) for p in parts], axis=0),
        cholesky_factors=np.concatenate(
            [np.asarray(p.cholesky_factors) for p in parts], axis=0
        ),
        colors=(
            np.concatenate([np.asarray(p.colors) for p in parts], axis=0)
            if keep_colors
            else None
        ),
        truncation_radius=parts[0].truncation_radius,
    )


def _allocate_group_M(sizes: np.ndarray, M_target: int) -> np.ndarray:
    """Split ``M_target`` representatives across groups: floor 1 each, the rest
    proportional to ``size - 1`` (largest-remainder), clamped to group size with
    deficit water-filled into groups that still have slack."""
    g_count = len(sizes)
    total = max(int(M_target), g_count)  # >= 1 per group
    alloc = np.ones(g_count, dtype=np.int64)
    rem = total - g_count
    if rem > 0:
        weights = np.maximum(sizes - 1, 0).astype(np.float64)
        if weights.sum() > 0:
            ideal = rem * weights / weights.sum()
            floor = np.floor(ideal).astype(np.int64)
            leftover = int(rem - floor.sum())
            if leftover > 0:
                # Stable sort so ties (common with equal-size groups) break by
                # group index -> reproducible allocation across numpy versions.
                order = np.argsort(-(ideal - floor), kind="stable")
                floor[order[:leftover]] += 1
            alloc = alloc + floor
    alloc = np.minimum(alloc, sizes)
    # Water-fill any deficit (from the size clamp) into groups with slack.
    deficit = total - int(alloc.sum())
    while deficit > 0:
        slack = sizes - alloc
        idx = np.where(slack > 0)[0]
        if idx.size == 0:
            break
        order = idx[np.argsort(-slack[idx], kind="stable")]
        take = order[: min(deficit, idx.size)]
        alloc[take] += 1
        deficit = total - int(alloc.sum())
    return cast(np.ndarray, alloc)


def _reduce_one_level_grouped(
    data: GSplatData,
    *,
    M_target: int,
    coarsen_dims: tuple[int, ...],
    method: MethodName,
    lloyd_iterations: int,
    candidate_bins_k: int,
    coverage_inflation: float,
    device: torch.device,
    conserve_mass: bool = True,
    amplitude: str = "l2",
    refine_config: "Optional[L2RefineConfig]" = None,
    generator: "Optional[torch.Generator]" = None,
    refine_stats: Optional[dict] = None,
) -> GSplatData:
    """One reduction level that never merges across the barrier dims.

    Splats are partitioned by their exact coordinate in the barrier dims (all
    dims except ``coarsen_dims``); each group is reduced independently with the
    **unchanged** :func:`_reduce_one_level` and a proportional share of
    ``M_target``, then concatenated. Within a group every barrier coordinate is
    identical, so the merged representatives stay on that value. When an L2
    refine is configured, each per-group refit receives the barrier dims as
    ``frozen_dims`` so refined centers/covariances never move or widen across a
    barrier; kept-as-is small groups are not refined (nothing was merged).

    Note: with ``refine="l2"`` and many barrier groups (e.g. a long
    time/channel axis), a full independent refit — standardize + spatial-hash
    build + Adam loop — runs *per group*, so wall time scales with the group
    count. This is the correctness-first choice (groups must not blend); the
    per-group cost is why refine at whole-timelapse scale is validated
    separately before being exposed on ``batch-fit merge``.
    """
    d_total = data.ndim
    coarsen_set = set(coarsen_dims)
    barrier = [d for d in range(d_total) if d not in coarsen_set]
    if not barrier:
        return _reduce_one_level(
            data,
            M_target=M_target,
            method=method,
            lloyd_iterations=lloyd_iterations,
            candidate_bins_k=candidate_bins_k,
            coverage_inflation=coverage_inflation,
            device=device,
            conserve_mass=conserve_mass,
            mass_dims=None,
            amplitude=amplitude,
            refine_config=refine_config,
            generator=generator,
            refine_stats=refine_stats,
        )
    keys = np.asarray(data.centers)[:, barrier]
    _group_ids, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    g_count = int(inverse.max()) + 1 if inverse.size else 0
    if g_count <= 1:
        # A single group: every barrier coordinate is shared, so the refit
        # still freezes the barrier dims (centers/Σ must stay on the value).
        return _reduce_one_level(
            data,
            M_target=M_target,
            method=method,
            lloyd_iterations=lloyd_iterations,
            candidate_bins_k=candidate_bins_k,
            coverage_inflation=coverage_inflation,
            device=device,
            conserve_mass=conserve_mass,
            mass_dims=tuple(coarsen_dims),
            amplitude=amplitude,
            refine_config=refine_config,
            refine_frozen_dims=tuple(barrier),
            generator=generator,
            refine_stats=refine_stats,
        )
    sizes = np.bincount(inverse, minlength=g_count)
    if M_target < g_count:
        aprint(
            f"  substitutive: M_target={M_target} < {g_count} barrier groups; "
            f"level clamped to {g_count} splats (>=1 per group)."
        )
    alloc = _allocate_group_M(sizes, M_target)
    # Group row indices in one O(N log N) pass (avoids an O(N*G) mask scan when
    # the barrier has many groups). Stable sort preserves within-group order.
    order = np.argsort(inverse, kind="stable")
    ends = np.cumsum(sizes)
    starts = ends - sizes
    parts: list[GSplatData] = []
    for g in range(g_count):
        idx = order[starts[g] : ends[g]]
        sub = _subset_gsplatdata(data, idx)
        if int(alloc[g]) >= int(sizes[g]):
            # Group already small enough; keep as-is, but still drop any
            # non-positive-amplitude splats that a reduction would have culled
            # (fidelity with the reduced branch).
            parts.append(_drop_nonpositive(sub))
        else:
            parts.append(
                _reduce_one_level(
                    sub,
                    M_target=int(alloc[g]),
                    method=method,
                    lloyd_iterations=lloyd_iterations,
                    candidate_bins_k=candidate_bins_k,
                    coverage_inflation=coverage_inflation,
                    device=device,
                    conserve_mass=conserve_mass,
                    mass_dims=tuple(coarsen_dims),
                    amplitude=amplitude,
                    refine_config=refine_config,
                    refine_frozen_dims=tuple(barrier),
                    generator=generator,
                    refine_stats=refine_stats,
                )
            )
    return _concat_gsplatdata(parts)


def _within_box(
    data: GSplatData,
    dims: Sequence[int],
    box: Sequence[tuple[float, float]],
    *,
    tol: Optional[float] = None,
) -> bool:
    """Are every splat's centers still inside their tile, on the boxed dims?

    Centers only — a Gaussian's tails always cross a tile boundary, and the
    partition has never claimed otherwise (its parts are split by centre, and
    ``chunk_bounds`` widens for extent separately). What must not happen is a
    centre migrating into a neighbour's cell.

    ``tol`` defaults to the level's own median splat sigma — the same slack
    :func:`~.volume_refit._relocated` grants, for the same stated reason: a
    coarse splat correcting within its own footprint is optimization, not
    migration. A hard tolerance instead discards a whole tile's re-fit over
    sub-voxel drift (measured: half the tiles, each overshooting by well under
    one sigma), and the crop's own outward rounding already reaches that far past
    the boundary. Sub-sigma overlap also sits well inside what a uniform
    (apodized) tiling deliberately carries, and part bounds are recomputed from
    the actual centers at write time, so viewer culling follows the splats.
    """
    from luxar.gsplats.lod.volume_refit import _median_splat_sigma

    if tol is None:
        tol = max(1e-3, _median_splat_sigma(data)) if data.n_splats else 1e-3
    centers = np.asarray(data.centers)
    for k, d in enumerate(dims):
        low, high = box[k]
        col = centers[:, int(d)]
        if np.any(col < low - tol) or np.any(col > high + tol):
            return False
    return True


def _with_stored_mse(stats: dict) -> dict:
    """Record which candidate's error was actually KEPT, on a single-piece refit.

    The aggregated (barrier) path reports this too. Without it here, a consumer
    reading ``mse_refit`` on a tile whose re-fit was rejected for leaving its
    tile sees the discarded candidate's error — which can be the LOWER of the two
    — and concludes the level improved when the merge was kept.
    """
    if "mse_seed" in stats and "mse_refit" in stats:
        kept_seed = bool(stats.get("seed_won")) or bool(stats.get("tile_escape"))
        stats["mse_stored"] = float(
            stats["mse_seed"] if kept_seed else stats["mse_refit"]
        )
    return stats


def _volume_refine_level(
    level: GSplatData,
    # Array-LIKE (see select_sub_volume): a lazy zarr store stays lazy.
    volume: Any,
    *,
    coarsen_dims: Optional[tuple[int, ...]],
    config: "VolumeRefitConfig",
    device: Optional[str],
    volume_axes: Optional[Sequence[int]] = None,
    box: Optional[Sequence[tuple[float, float]]] = None,
) -> tuple[GSplatData, dict]:
    """Volume re-fit one already-merged level, piecewise over its barrier groups.

    Without barrier dims this is the historical single call. With them, the level
    is split by barrier coordinate and each group is re-fitted against its OWN
    slice of ``volume``, projected to the coarsened dims so the fit cannot move a
    splat along a barrier axis (see :mod:`.volume_regions`); the barrier
    coordinate and covariance rows come back verbatim from the seed.

    Runs at the same altitude as the historical call — on the concatenated level,
    after :func:`_reduce` — so no reduction signature changes. Mass is pinned per
    group by :func:`~.volume_refit.volume_refine_splats`, matching the merge's
    per-group ``mass_dims`` and keeping per-timepoint brightness intact.

    ``box`` restricts the coarsened dims to one spatial tile, for the per-part
    (adaptive) caller; ``None`` spans the whole volume.
    """
    from luxar.gsplats.lod.volume_refit import volume_refine_splats
    from luxar.gsplats.lod.volume_regions import (
        finalize_volume_refit_stats,
        merge_volume_refit_stats,
        project_to_dims,
        restore_dims,
        select_sub_volume,
    )

    d_total = level.ndim
    coarsen = tuple(range(d_total)) if coarsen_dims is None else tuple(coarsen_dims)
    barrier = tuple(d for d in range(d_total) if d not in set(coarsen))

    def _one(
        piece: GSplatData, coords: Sequence[float]
    ) -> tuple[GSplatData, dict, int]:
        sub = select_sub_volume(
            volume,
            ndim=d_total,
            barrier_dims=barrier,
            barrier_coords=coords,
            box=box,
            volume_axes=volume_axes,
        )
        seed = project_to_dims(piece, sub.dims)
        shifted = bool(np.any(sub.origin != 0.0))
        if shifted:
            seed = seed.translate(-sub.origin.astype(np.float32))
        refit, st = volume_refine_splats(seed, sub.array, config=config, device=device)
        if shifted:
            refit = refit.translate(sub.origin.astype(np.float32))
        out = restore_dims(refit, piece, sub.dims)
        if box is not None and not _within_box(out, sub.dims, box):
            # A partition part's splats must stay inside their tile: the viewer
            # frustum-culls by part bounds, so a splat that wandered out would
            # simply stop being drawn from most viewpoints. The never-worse MSE
            # guard is structurally blind to this — a splat that left the tile can
            # still lower the crop's MSE — so the containment verdict is separate
            # and, like a frame mismatch, resolves by keeping the merge.
            st = dict(st)
            st["tile_escape"] = True
            st["improved"] = False
            st["seed_won"] = True
            return piece, _with_stored_mse(st), int(sub.array.size)
        st = dict(st)
        st["tile_escape"] = False
        return out, _with_stored_mse(st), int(sub.array.size)

    if not barrier:
        refined_all, st_all, _ = _one(level, ())
        return refined_all, st_all

    keys = np.asarray(level.centers)[:, list(barrier)]
    group_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    sizes = np.bincount(inverse, minlength=len(group_keys))
    # One O(N log N) bucketing pass; stable so within-group order is preserved.
    order = np.argsort(inverse, kind="stable")
    ends = np.cumsum(sizes)
    starts = ends - sizes

    pieces: list[GSplatData] = []
    sink: dict = {}
    for g in range(len(group_keys)):
        piece = _subset_gsplatdata(level, order[starts[g] : ends[g]])
        if piece.n_splats == 0:
            continue
        refined, st, n_voxels = _one(piece, group_keys[g])
        pieces.append(refined)
        merge_volume_refit_stats(sink, st, weight=n_voxels)
    if not pieces:
        return level, finalize_volume_refit_stats(sink)
    return _concat_gsplatdata(pieces), finalize_volume_refit_stats(sink)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def _resolve_refit_image_min(
    data: GSplatData, image_min: Optional[float], verbose: bool
) -> Optional[float]:
    from luxar.gsplats.fit_basis import MISSING_BASIS_HINT, fit_image_min

    resolved = fit_image_min(
        {"image_min": image_min} if image_min is not None else data.stats
    )
    if resolved is None and verbose:
        aprint(f"refine=volume: {MISSING_BASIS_HINT}")
    return resolved


def make_substitutive_lod(
    data: GSplatData,
    *,
    compression_factor: int = 4,
    levels: int = 3,
    method: AutoOrMethod = "auto",
    lloyd_iterations: int = 5,
    candidate_bins_k: int = 12,
    coverage_inflation: float = 3.0,
    conserve_mass: bool = True,
    amplitude: Literal["l2", "mass"] = "l2",
    refine: RefineName = "none",
    refine_iters: Optional[int] = None,
    volume: Optional[np.ndarray] = None,
    volume_axes: Optional[Sequence[int]] = None,
    image_min: Optional[float] = None,
    volume_box: Optional[Sequence[tuple[float, float]]] = None,
    device: Union[str, torch.device, None] = "auto",
    seed: Optional[int] = None,
    coarsen_dims: Optional[Sequence[int]] = None,
    verbose: bool = False,
    quality_stamps: bool = False,
    quality_max_pair_splats: int = 2_000_000,
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
    coverage_inflation
        Inflation factor β >= 1 applied to each representative's
        *inter-center* spread (``Σ_out = intra + β·inter``) with a
        mass-preserving amplitude rescale. Pure moment matching gives
        the balanced bins σ ≈ pitch/√12 — too narrow for neighbouring
        representatives to sum flat, which renders as a strong periodic
        grid ripple along the shared Morton-cell boundaries. The default
        β=3 widens exactly the inter term to σ ≈ pitch/2 (flat-sum
        threshold) and is the exact fixed point of the level recurrence,
        so the calibration holds at every level. ``1.0`` disables
        (historical pure-moment-matching behaviour). Trade-off: coarse
        levels look slightly smoother; each splat's integral (X-ray
        projection) is preserved exactly. With ``refine="l2"`` the
        inflation is demoted from final answer to *optimizer seed*: the
        refit takes over the exact flat-sum calibration.
    conserve_mass
        Rescale each reduced level's amplitudes by one global factor so its
        total mass over the coarsened dims equals its fine input's (per
        barrier group under ``coarsen_dims``). The per-bin L²-optimal
        amplitude is not mass-preserving (3–17 % loss per level measured,
        content-dependent), and that mass is the DC an additive render
        integrates — uncorrected it shows as a brightness pop at every LOD
        switch. Default True; ``False`` restores the raw per-bin amplitudes.
        The rescale is skipped (with a warning) when the implied factor
        falls outside ``[0.1, 10]`` — a numerically degenerate coarsened-dim
        mass, where "conserving" it would blow the amplitudes up instead.
    amplitude
        Per-bin merged-amplitude rule. ``"l2"`` (default) is the L²-optimal
        projection amplitude — the right choice for fitted volumetric
        gsplats. ``"mass"`` makes every bin exactly mass-preserving
        (``a = Σ member a·|det L| / |det L_out|``, on the final inflated
        covariance): per-bin colored light is then conserved together with
        the bin-mass-weighted mean colors, which is what the lifted
        points/lines LOD path uses to keep brightness/hue coherent across
        levels (the beads are a stroke stand-in, not a density to L²-fit).
        Under ``"mass"`` the global ``conserve_mass`` rescale is a no-op by
        construction (kept as a safety net). Exactness note: with barrier
        groups the conserved per-bin quantity is the full-determinant mass;
        the sliced (coarsened-dims-only) mass coincides when member barrier
        widths are equal within a bin — true for lifted isotropic beads.
    refine
        Post-merge per-level refinement. ``"l2"`` Adam-optimizes each
        merged level's ``(mu, Σ, a)`` against that level's fine input
        under the closed-form mixture L² (sparse pair lists, trusted
        checkpoints, total mass pinned to the fine mixture's — see
        :mod:`._substitutive.refine`). Never worse than the merge in the
        trusted metric; substantially higher fidelity (prototype: rel-L²
        0.089 vs 0.151 on flat fields, peak preservation 0.99 vs 0.91 on
        isolated blobs). ``"volume"`` warm-start re-fits each merged level
        against the source ``volume`` itself (a full
        :func:`~luxar.gsplats.fit_gsplats.fit_gaussian_splats` pass seeded
        by the merge) — the highest-fidelity option (+5–12 dB over the
        merge on real microscopy, see :mod:`.volume_refit`); requires
        ``volume``. With barrier dims (``coarsen_dims`` set) each barrier
        group is re-fitted against its OWN slice of the volume, in the
        coarsened dims only — see :mod:`.volume_regions` for why the barrier
        axis is sliced away rather than held still. Each level keeps
        whichever of {merge seed, re-fit} renders closer to the volume, so it
        is never worse than the merge. ``"none"`` (default) keeps the merge
        output.
    refine_iters
        Adam steps per refined level (``refine="l2"``) / fit iterations per
        re-fitted level (``refine="volume"``). ``None`` (default) resolves to
        the engine's own config default — 120 for ``l2``
        (:class:`~._substitutive.refine.L2RefineConfig`), 300 for ``volume``
        (:class:`~.volume_refit.VolumeRefitConfig`).
    volume
        The source volume (full resolution, same voxel coordinate frame as
        the splats) that ``refine="volume"`` fits against. Required for —
        and only meaningful with — that mode. Only ever *sliced*, never
        coerced whole, so a lazy store (a zarr array) stays lazy: a
        253-timepoint 407x2048x2048 uint16 timelapse is 431 GB while one
        timepoint is 3.4 GB.
    volume_axes
        ``volume_axes[i]`` is the ``volume`` axis holding center dim ``i``.
        ``None`` (default) means the identity, which is what a whole-volume
        3D re-fit has always assumed. A stacked timelapse needs it: Luxar
        puts spatial dims first and the stacked axis LAST, while the source
        array is typically ``(t, z, y, x)`` with time FIRST.
    image_min
        Normalization level removed by the input fit. When omitted, it is read
        from ``data.stats``; per-part recipe callers pass it explicitly because
        converting a bare tree node to ``GSplatData`` has no top-level stats.
    volume_box
        Per-coarsened-dim ``(low, high)`` bounds restricting the re-fit to one
        spatial tile, for the per-part (``adaptive``) caller. The re-fit then
        sees only that tile's crop, and a re-fit that moves a centre out of the
        tile is rejected in favour of the merge — the viewer frustum-culls by
        part bounds, so an escapee would silently stop being drawn.
    device
        ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"mps"``, or a
        :class:`torch.device`.
    seed
        Seeds the L2-refine minibatch pair sampler when ``refine="l2"``
        (a local :class:`torch.Generator`; global torch RNG untouched).
        Otherwise accepted for API stability only — the Morton warm start
        and the synchronous Lloyd pass are deterministic.
    coarsen_dims
        Center-column indices that coarsening is *allowed* to cluster/merge
        over. The complementary dims become hard grouping boundaries: splats
        are partitioned by their exact coordinate in those barrier dims and
        each group is reduced independently, so a coarse splat never blends
        across a barrier value (e.g. a categorical ``coloring`` axis, time, or
        channel). ``None`` (default) coarsens over all dims (the historical
        behavior). Passing all dims is equivalent to ``None``. Because every
        non-empty group keeps >= 1 representative, the coarsest level has at
        least as many splats as there are barrier groups.
    verbose
        Per-level Arbol logging.
    quality_stamps
        Measure each level's approximation quality against the finest
        content (closed-form mixture L², ``lod/quality.py``) and stamp
        ``quality`` + ``reference_energy`` into every level's stats — the
        Q of the viewer's committed quality ``Q·e(k)``. ``reference_energy``
        is the FINEST content's total self-energy (constant across the
        group), so partition-of-lod aggregation weighs every tile by its
        region's content regardless of which level the tile displays.
        Default False at this primitive layer (the measurement costs
        seconds per level); the RECIPE/CLI pipeline enables it by default —
        stamped artifacts are its product, speed-sensitive library callers
        opt in.
    quality_max_pair_splats
        Pair-term subsampling threshold for the quality measurement
        (see :func:`~luxar.gsplats.lod.quality.mixture_quality`).

    Returns
    -------
    GSplatData
        A matrix-shaped dataset with ``n_substitutive = levels + 1`` and a
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
    if coverage_inflation < 1.0:
        raise ValueError(f"coverage_inflation must be >= 1.0, got {coverage_inflation}")
    if amplitude not in ("l2", "mass"):
        raise ValueError(f"amplitude must be 'l2' or 'mass', got {amplitude!r}")
    if refine not in _VALID_REFINE:
        raise ValueError(f"refine must be one of {list(_VALID_REFINE)}, got {refine!r}")
    if refine_iters is not None and refine_iters < 1:
        raise ValueError(f"refine_iters must be >= 1, got {refine_iters}")
    if refine == "volume" and volume is None:
        raise ValueError("refine='volume' requires the `volume` argument")
    if volume is not None and refine != "volume":
        raise ValueError(
            "`volume` is only consumed by refine='volume'; "
            f"got volume with refine={refine!r}"
        )
    if volume_axes is not None and volume is None:
        # An axis map describes a volume. Silently ignoring it would let a
        # typo'd or misplaced map vanish without trace, and the map is exactly
        # what decides whether a stacked re-fit targets the right axis.
        raise ValueError(
            "`volume_axes` describes the layout of `volume`, but no volume was "
            "given; pass volume=... (with refine='volume') or drop volume_axes"
        )
    K = int(compression_factor)
    L_levels = int(levels)

    src = _finest_content(data)
    _refuse_categorical_coarsening(src)
    target_device = _resolve_reduction_device(device, caller="make_substitutive_lod")

    # Normalise coarsen_dims -> a sorted barrier set (or None == coarsen all dims).
    norm_coarsen = _normalise_coarsen_dims(coarsen_dims, src)

    # Refine setup. ``refine_iters=None`` resolves to each engine's own config
    # default (L2RefineConfig 120 / VolumeRefitConfig 300 — the single source
    # of truth; the CLI passes None through, so API and CLI defaults agree).
    refine_cfg: Optional[L2RefineConfig] = None
    refine_gen: Optional[torch.Generator] = None
    eff_refine_iters: Optional[int] = None
    if refine == "l2":
        eff_refine_iters = (
            int(refine_iters) if refine_iters is not None else L2RefineConfig().iters
        )
        refine_cfg = replace(L2RefineConfig(), iters=eff_refine_iters)
        # A LOCAL torch.Generator for the minibatch pair sampler (global RNG
        # untouched).
        if seed is not None:
            refine_gen = torch.Generator()
            refine_gen.manual_seed(int(seed))

    # Volume re-fit setup. The fit runs on the *requested* device (not
    # ``target_device``, which may have been downgraded to CPU for Lloyd's
    # float64 requirement — the fitting stack is float32 and MPS/CUDA-happy).
    volume_cfg: Optional["VolumeRefitConfig"] = None
    refit_device: Optional[str] = None
    if refine == "volume":
        from luxar.gsplats.lod.volume_refit import VolumeRefitConfig

        eff_refine_iters = (
            int(refine_iters) if refine_iters is not None else VolumeRefitConfig().iters
        )
        # The ladder's basis comes from the INPUT fit, which is the only thing
        # that knows what background was already removed. Without it the inner
        # re-fit re-estimates one from the raw volume and the refined level can
        # end up on a different basis from its siblings (#1177).
        refit_image_min = _resolve_refit_image_min(data, image_min, verbose)
        volume_cfg = replace(
            VolumeRefitConfig(),
            iters=eff_refine_iters,
            conserve_mass=bool(conserve_mass),
            image_min=refit_image_min,
        )
        if device is not None and not (isinstance(device, str) and device == "auto"):
            refit_device = str(device)

    def _reduce(
        cur: GSplatData, m_target: int, meth: MethodName, level_refine_stats: dict
    ) -> GSplatData:
        sink = level_refine_stats if refine_cfg is not None else None
        if norm_coarsen is None:
            return _reduce_one_level(
                cur,
                M_target=m_target,
                method=meth,
                lloyd_iterations=lloyd_iterations,
                candidate_bins_k=candidate_bins_k,
                coverage_inflation=coverage_inflation,
                device=target_device,
                conserve_mass=conserve_mass,
                mass_dims=None,
                amplitude=amplitude,
                refine_config=refine_cfg,
                generator=refine_gen,
                refine_stats=sink,
            )
        return _reduce_one_level_grouped(
            cur,
            M_target=m_target,
            coarsen_dims=norm_coarsen,
            method=meth,
            lloyd_iterations=lloyd_iterations,
            candidate_bins_k=candidate_bins_k,
            coverage_inflation=coverage_inflation,
            device=target_device,
            conserve_mass=conserve_mass,
            amplitude=amplitude,
            refine_config=refine_cfg,
            generator=refine_gen,
            refine_stats=sink,
        )

    # Quality stamps (the Q of the viewer's Q·e(k)): every level is measured
    # against the SAME reference — the group's finest content — so the values
    # order correctly across levels. reference_energy is likewise the finest
    # content's total (self-energy is quadratic in amplitude, so unlike mass
    # it is NOT conserved across levels — per-level energies would skew
    # partition-of-lod weighting by whichever level a tile happens to show).
    ref_energy: Optional[float] = None
    if quality_stamps:
        from luxar.gsplats.lod.quality import mixture_quality, total_self_energy

        ref_energy = total_self_energy(src)

    def _stamp_quality(level_stats: dict, level_data: GSplatData) -> None:
        if not quality_stamps:
            return
        try:
            result = mixture_quality(
                level_data.flattened(),
                src,
                max_pair_splats=quality_max_pair_splats,
                device=device if isinstance(device, str) else str(device),
            )
            level_stats["quality"] = result.quality
        except Exception as exc:  # measurement must never fail the build
            aprint(f"quality stamp skipped for level: {exc}")
        if ref_energy is not None and np.isfinite(ref_energy):
            level_stats["reference_energy"] = float(ref_energy)

    def _stamp_footprint(level_stats: dict, level_data: GSplatData) -> None:
        if level_data.n_splats == 0:
            return
        sigma_geo = np.sqrt(level_data.volumes())
        finite = sigma_geo[np.isfinite(sigma_geo) & (sigma_geo > 0)]
        if finite.size:
            level_stats["median_footprint"] = float(np.median(finite))

    # Collect per-level outputs and pack them as SubstitutiveLevels.
    finest_stats: dict = {"n_splats_total": int(src.n_splats)}
    _stamp_footprint(finest_stats, src)
    if quality_stamps:
        # The finest level IS the reference: quality 1.0 by construction.
        finest_stats["quality"] = 1.0
        if ref_energy is not None and np.isfinite(ref_energy):
            finest_stats["reference_energy"] = float(ref_energy)
    sub_levels: list[SubstitutiveLevel] = [
        _pack_level(
            src,
            compression_factor=1,
            parent_method=None,
            level_index=0,
            stats=finest_stats,
        )
    ]

    current = src
    for level_idx in range(1, L_levels + 1):
        N_in = current.n_splats
        if N_in <= 1:
            # Cannot reduce further; emit the unchanged dataset and stop.
            stop_stats: dict = {
                "n_splats_total": int(N_in),
                "stop_reason": "input_too_small",
            }
            _stamp_quality(stop_stats, current)
            sub_levels.append(
                _pack_level(
                    current,
                    compression_factor=K**level_idx,
                    parent_method=_resolve_method(method, N_in),
                    level_index=level_idx,
                    stats=stop_stats,
                )
            )
            break
        M_target = max(1, math.ceil(N_in / K))
        level_method = _resolve_method(method, N_in)
        level_refine_stats: dict = {}
        if verbose:
            label = f"{level_method} (auto)" if method == "auto" else level_method
            with asection(
                f"Substitutive level {level_idx}: {N_in} -> {M_target} splats"
            ):
                aprint(f"method={label}")
                new_data = _reduce(current, M_target, level_method, level_refine_stats)
                if level_refine_stats:
                    aprint(
                        "refine=l2: trusted-E improvement "
                        f"{level_refine_stats.get('improvement_frac', 0.0):.1%} "
                        f"({level_refine_stats.get('iters_run', 0)} steps, "
                        f"{level_refine_stats.get('wall_s', 0.0):.1f}s)"
                    )
        else:
            new_data = _reduce(current, M_target, level_method, level_refine_stats)
        # Volume re-fit: replace the STORED level with the warm-start re-fit
        # (or keep the merge if it renders closer — the engine's never-worse
        # guard). The merge chain continues from the unrefined merge output.
        stored = new_data
        volume_refit_stats: Optional[dict] = None
        if volume_cfg is not None and new_data.n_splats > 0:
            assert volume is not None  # validated above
            stored, volume_refit_stats = _volume_refine_level(
                new_data,
                volume,
                coarsen_dims=norm_coarsen,
                config=volume_cfg,
                device=refit_device,
                volume_axes=volume_axes,
                box=volume_box,
            )
            if verbose:
                pieces = int(volume_refit_stats.get("n_pieces", 1))
                # With barrier dims a level is refined in pieces and any of them
                # may have gone either way, so report the fraction rather than a
                # single verdict that would hide 252 of 253 outcomes.
                where = "" if pieces <= 1 else f" across {pieces} barrier groups"
                aprint(
                    "refine=volume: "
                    + (
                        "re-fit won "
                        f"{volume_refit_stats.get('improved_frac', 1.0):.0%}"
                        f"{where} (MSE {volume_refit_stats['mse_seed']:.3e} -> "
                        f"{volume_refit_stats['mse_refit']:.3e})"
                        if volume_refit_stats.get("improved")
                        else (
                            "merge seed kept (coordinate-frame mismatch — "
                            "re-fit rejected)"
                            if volume_refit_stats.get("frame_mismatch")
                            else "merge seed kept (re-fit did not improve)"
                        )
                    )
                    + f", {volume_refit_stats['wall_s']:.1f}s"
                )
        level_stats: dict = {"n_splats_total": int(stored.n_splats)}
        _stamp_footprint(level_stats, stored)
        if level_refine_stats:
            level_refine_stats.pop("_mass_n", None)
            level_stats["refine"] = "l2"
            level_stats["refine_stats"] = level_refine_stats
        if volume_refit_stats is not None:
            level_stats["refine"] = "volume"
            level_stats["refine_stats"] = volume_refit_stats
        _stamp_quality(level_stats, stored)
        if verbose and "quality" in level_stats:
            aprint(f"quality vs finest: {level_stats['quality']:.4f}")
        sub_levels.append(
            _pack_level(
                stored,
                compression_factor=K**level_idx,
                parent_method=level_method,
                level_index=level_idx,
                stats=level_stats,
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
            "coverage_inflation": float(coverage_inflation),
            "conserve_mass": bool(conserve_mass),
            "refine": refine,
            "refine_iters": eff_refine_iters,
            # ALWAYS the explicit dim list, coarsen-everything included. The
            # writer derives the chunk-ordering barrier from this key's
            # COMPLEMENT, and it cannot tell a written `null` from an absent
            # one — so the `None` this used to publish for the coarsen-all case
            # landed back on auto-detection and re-imposed a barrier on the very
            # axis every level had just been blended over (#1600). See
            # `resolved_merge_coarsen_dims` for the measurement. `src.ndim` is
            # the right width: every level is a reduction of `src` in the same
            # center columns, so the node the stamp is read back against has
            # exactly these dims.
            "coarsen_dims": resolved_merge_coarsen_dims(norm_coarsen, src.ndim),
        }
    )
    return GSplatData.from_substitutive_levels(sub_levels, stats=out_stats)


def merge_to_count(
    data: GSplatData,
    *,
    n_target: int,
    method: AutoOrMethod = "auto",
    lloyd_iterations: int = 5,
    candidate_bins_k: int = 12,
    coverage_inflation: float = 3.0,
    device: Union[str, torch.device, None] = "auto",
    coarsen_dims: Optional[Sequence[int]] = None,
) -> GSplatData:
    """Merge *data* into ``n_target`` representatives — ONE flat level.

    A single application of the partition-and-merge operator that
    :func:`make_substitutive_lod` iterates, exposed for callers who want a
    SIZE rather than a ladder. ``make_substitutive_lod`` reduces by an INTEGER
    per-level factor, so the counts it can land on are quantised (N/2, N/3, …)
    and an arbitrary request falls between two of them; here the count is the
    input. Everything else is shared with the ladder path — the same merge math,
    the same barrier-dim grouping, and the same per-group mass conservation, so
    the result keeps the input's brightness instead of dimming it.

    Args:
        data: Source dataset (reduced from its finest content).
        n_target: Number of representatives to produce. A request at or above
            the input count returns the finest content unreduced.
        method: Partition algorithm or ``"auto"`` — see
            :func:`make_substitutive_lod`.
        lloyd_iterations: Lloyd refinement passes.
        candidate_bins_k: Lloyd move-candidate neighbours per splat.
        coverage_inflation: Inter-center spread inflation β (see
            :func:`make_substitutive_lod`).
        device: Torch device (``"auto"`` resolves; MPS downgrades to CPU).
        coarsen_dims: Center-column indices merging may combine over; the rest
            are hard barriers. Default: all dims.

    Returns:
        A flat :class:`GSplatData` with at most ``n_target`` splats. It can land
        slightly under: the merge culls degenerate (empty / non-positive-mass)
        clusters, and the barrier grouping keeps at least one representative per
        group, which can push the count up instead.

    Raises:
        ValueError: If ``n_target < 1`` or ``method`` is not recognised.
            Also raised when the input carries categorical labels, because
            merging has no defined label-combination rule.
    """
    if n_target < 1:
        raise ValueError(f"n_target must be >= 1, got {n_target}")
    if method not in _VALID_CHOICES:
        raise ValueError(
            f"method must be one of {list(_VALID_CHOICES)}, got {method!r}"
        )
    src = _finest_content(data)
    if src.n_splats <= n_target:
        return src
    _refuse_categorical_coarsening(src)
    target_device = _resolve_reduction_device(device, caller="merge_to_count")
    norm_coarsen = _normalise_coarsen_dims(coarsen_dims, src)
    level_method = _resolve_method(method, src.n_splats)
    if norm_coarsen is None:
        return _reduce_one_level(
            src,
            M_target=n_target,
            method=level_method,
            lloyd_iterations=lloyd_iterations,
            candidate_bins_k=candidate_bins_k,
            coverage_inflation=coverage_inflation,
            device=target_device,
        )
    return _reduce_one_level_grouped(
        src,
        M_target=n_target,
        coarsen_dims=norm_coarsen,
        method=level_method,
        lloyd_iterations=lloyd_iterations,
        candidate_bins_k=candidate_bins_k,
        coverage_inflation=coverage_inflation,
        device=target_device,
    )


# ─────────────────────────────────────────────────────────────────────
# Per-level reduction
# ─────────────────────────────────────────────────────────────────────


#: ``conserve_mass`` guard: the global amplitude rescale is skipped (with a
#: warning) when the implied factor leaves ``[1/x, x]`` for this bound. The
#: per-bin L²-optimal amplitude loses only 3–17 % mass per level (the measured
#: drift this feature corrects), so a legitimate correction is a few tens of
#: percent; a factor beyond 10x / below 0.1x means the coarsened-dim mass is
#: numerically degenerate (e.g. most representatives' submatrices collapsed to
#: ~zero determinant, leaving ``mass_out`` tiny-but-positive) and rescaling
#: would blow amplitudes up (white-out) rather than fix a drift.
_MASS_SCALE_BOUND = 10.0


def _subset_mass(
    L: torch.Tensor,
    amps: torch.Tensor,
    dims: Optional[tuple[int, ...]],
    chunk: int = 2_000_000,
) -> float:
    """Total mass ``Σ a·|Σ[dims,dims]|^{1/2}`` over a dim subset (chunked).

    ``dims=None`` uses the full covariance (``|Σ|^{1/2} = Π diag(L)``). For a
    subset — the *coarsened* dims of a barrier-grouped reduction — the
    submatrix determinant is the mass a viewer slicing over the barrier dims
    actually integrates, immune to any barrier-width numerics. Degenerate
    (non-PD) submatrices contribute zero.
    """
    if dims is None:
        diag = torch.diagonal(L, dim1=-2, dim2=-1).abs()
        return float((amps * torch.prod(diag, dim=-1)).sum())
    idx = torch.as_tensor(dims, dtype=torch.int64, device=L.device)
    total = 0.0
    for s in range(0, L.shape[0], chunk):
        Ls = L[s : s + chunk]
        sub = (Ls @ Ls.transpose(-1, -2))[:, idx][:, :, idx]
        Lc, info = torch.linalg.cholesky_ex(sub)
        det = torch.prod(torch.diagonal(Lc, dim1=-2, dim2=-1).abs(), dim=-1)
        det = torch.where(info == 0, det, torch.zeros_like(det))
        total += float((amps[s : s + chunk] * det).sum())
    return total


def _reduce_one_level(
    data: GSplatData,
    *,
    M_target: int,
    method: MethodName,
    lloyd_iterations: int,
    candidate_bins_k: int,
    coverage_inflation: float,
    device: torch.device,
    conserve_mass: bool = True,
    mass_dims: Optional[tuple[int, ...]] = None,
    amplitude: str = "l2",
    refine_config: Optional[L2RefineConfig] = None,
    refine_frozen_dims: tuple[int, ...] = (),
    generator: Optional[torch.Generator] = None,
    refine_stats: Optional[dict] = None,
) -> GSplatData:
    """Run one application of the partition-and-merge operator $\\mathcal{R}_K$.

    When ``refine_config`` is given, the merged level is post-optimized against
    this level's fine input via :func:`l2_refine_mixture` (the merge acts as
    the seed / trust region; the refit is never worse than it in the trusted
    metric). ``refine_frozen_dims`` freezes barrier coordinates of a
    ``coarsen_dims`` group; per-call stats aggregate into ``refine_stats``.

    ``conserve_mass`` (default True) rescales the merged amplitudes by one
    global factor so the level's total mass over ``mass_dims`` (the coarsened
    dims; ``None`` = all) equals the fine input's — the per-bin L²-optimal
    amplitude is NOT mass-preserving (measured 3–17 % loss per level,
    content-dependent), and total mass over the displayed dims is the DC an
    additive render integrates, so uncorrected drift shows as a visible
    brightness pop at every LOD switch. Because the grouped path calls this
    once per barrier group, conservation holds PER GROUP (e.g. per timepoint).
    The rescale is skipped when the factor leaves ``[1/_MASS_SCALE_BOUND,
    _MASS_SCALE_BOUND]`` — see the constant's rationale.
    """
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
        centres_t,
        L_t,
        amps_t,
        colors_t,
        assignments,
        M=M_target,
        coverage_inflation=coverage_inflation,
        amplitude=amplitude,
    )

    # Cull empty / degenerate bins (zero optimal amplitude).
    keep_mask = new_amps > 0
    if not bool(torch.all(keep_mask)):
        new_centres = new_centres[keep_mask]
        new_L = new_L[keep_mask]
        new_amps = new_amps[keep_mask]
        if new_colors is not None:
            new_colors = new_colors[keep_mask]

    # 4) Mass conservation: one global amplitude factor so this level's total
    #    mass over the coarsened dims equals the fine input's (per barrier
    #    group, since the grouped path calls this per group). Runs BEFORE the
    #    optional refit, whose own mass manifold then sees a consistent seed.
    if conserve_mass and int(new_amps.numel()) > 0:
        mass_in = _subset_mass(L_t, amps_t, mass_dims)
        mass_out = _subset_mass(new_L, new_amps, mass_dims)
        if mass_in > 0.0 and mass_out > 0.0:
            scale = mass_in / mass_out
            if 1.0 / _MASS_SCALE_BOUND <= scale <= _MASS_SCALE_BOUND:
                new_amps = new_amps * scale
            else:
                aprint(
                    f"  substitutive: conserve_mass rescale skipped — factor "
                    f"{scale:.3g} outside [{1.0 / _MASS_SCALE_BOUND:g}, "
                    f"{_MASS_SCALE_BOUND:g}] (degenerate coarsened-dim mass; "
                    "raw per-bin amplitudes kept)."
                )

    # 5) Optional L2 refit of the merged level against this level's fine input.
    #    Colors are untouched: the refit changes no splat count or order, so
    #    the merge's bin-mass-weighted colors stay aligned.
    if refine_config is not None and int(new_amps.numel()) > 0:
        new_centres, new_L, new_amps, rstats = l2_refine_mixture(
            centres_t,
            L_t,
            amps_t,
            new_centres,
            new_L,
            new_amps,
            config=refine_config,
            frozen_dims=refine_frozen_dims,
            generator=generator,
        )
        if refine_stats is not None:
            _merge_refine_stats(refine_stats, rstats)

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
