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

import heapq
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
from luxar.gsplats.lod._kernels import (
    bin_inner_product_with_template_torch,
    bin_residual_energy_torch,
    gaussian_pair_inner_product_torch,
    gaussian_self_energy_torch,
    kwise_moment_match_torch,
    l2_optimal_amplitude_torch,
    sqrt_det_from_cholesky,
    template_squared_norm_torch,
)
from luxar.gsplats.lod._substitutive import _TINY
from luxar.gsplats.lod._substitutive.warm_start import _morton_order, _morton_partition
from luxar.gsplats.utils.device import resolve_torch_device
from luxar.gsplats.utils.trils import pack_tril, unpack_tril
from luxar.utils.spatial_hash import BatchedSpatialHashGrid

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
        SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=np.asarray(src.centers, dtype=np.float32),
                    amplitudes=np.asarray(src.amplitudes, dtype=np.float32),
                    cholesky_factors=np.asarray(src.cholesky_factors, dtype=np.float32),
                    colors=(np.asarray(src.colors) if src.colors is not None else None),
                    stats={"lod_method": "none", "lod_level": 0},
                    truncation_radius=src.truncation_radius,
                )
            ],
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
                SubstitutiveLevel(
                    additive_sublods=[
                        AdditiveSubLOD(
                            centers=np.asarray(current.centers, dtype=np.float32),
                            amplitudes=np.asarray(current.amplitudes, dtype=np.float32),
                            cholesky_factors=np.asarray(
                                current.cholesky_factors, dtype=np.float32
                            ),
                            colors=(
                                np.asarray(current.colors)
                                if current.colors is not None
                                else None
                            ),
                            stats={"lod_method": "none", "lod_level": 0},
                            truncation_radius=current.truncation_radius,
                        )
                    ],
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
            SubstitutiveLevel(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=np.asarray(new_data.centers, dtype=np.float32),
                        amplitudes=np.asarray(new_data.amplitudes, dtype=np.float32),
                        cholesky_factors=np.asarray(
                            new_data.cholesky_factors, dtype=np.float32
                        ),
                        colors=(
                            np.asarray(new_data.colors)
                            if new_data.colors is not None
                            else None
                        ),
                        stats={"lod_method": "none", "lod_level": 0},
                        truncation_radius=new_data.truncation_radius,
                    )
                ],
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
    centres_t = torch.from_numpy(np.asarray(data.centers, dtype=np.float32)).to(
        device=device, dtype=torch.float64
    )
    L_t = torch.from_numpy(
        unpack_tril(np.asarray(data.cholesky_factors, dtype=np.float32), D).astype(
            np.float64
        )
    ).to(device=device)
    amps_t = torch.from_numpy(np.asarray(data.amplitudes, dtype=np.float32)).to(
        device=device, dtype=torch.float64
    )
    if data.colors is not None:
        colors_t: Optional[torch.Tensor] = torch.from_numpy(np.asarray(data.colors)).to(
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


# ─────────────────────────────────────────────────────────────────────
# Vectorised partition + merge (O(N log N); no Python per-splat/-bin loops)
# ─────────────────────────────────────────────────────────────────────

def _segment_templates(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    assignments: torch.Tensor,
    M: int,
    *,
    Sigma: Optional[torch.Tensor] = None,
    sqrt_det: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-bin moment-matched template + L²-amplitude ingredients (vectorised).

    Every quantity is a segment reduction keyed by ``assignments`` (via
    :meth:`torch.Tensor.index_add_`) — no Python loop over bins. Mirrors the
    closed forms in :func:`kwise_moment_match_torch` /
    :func:`bin_inner_product_with_template_torch` exactly, but batched over
    all ``M`` bins at once.

    Returns ``(mu_bar (M,D), Sigma_bar (M,D,D), inner (M,), norm_sq (M,),
    weights (N,))`` where ``inner`` = ⟨f_Sb, Ḡb⟩ and ``norm_sq`` = ‖Ḡb‖². The
    per-bin L²-optimal amplitude is ``inner / norm_sq`` and the projection
    energy is ``inner² / norm_sq``.
    """
    N, D = centres.shape
    device = centres.device
    dt = centres.dtype
    if Sigma is None:
        Sigma = L @ L.transpose(-1, -2)  # (N, D, D)
    if sqrt_det is None:
        sqrt_det = sqrt_det_from_cholesky(L)  # (N,)

    masses = amps * sqrt_det  # (N,)
    bin_mass = torch.zeros(M, dtype=dt, device=device)
    bin_mass.index_add_(0, assignments, masses)
    w = masses / bin_mass[assignments].clamp_min(_TINY)  # (N,) normalised in-bin

    mu_bar = torch.zeros(M, D, dtype=dt, device=device)
    mu_bar.index_add_(0, assignments, w.unsqueeze(-1) * centres)

    # Intra-bin spread Σ_i w_i Σ_i  +  inter-bin spread Σ_i w_i δδᵀ.
    Sigma_bar = torch.zeros(M, D, D, dtype=dt, device=device)
    Sigma_bar.index_add_(0, assignments, w.view(N, 1, 1) * Sigma)
    delta = centres - mu_bar[assignments]  # (N, D)
    outer = w.view(N, 1, 1) * (delta.unsqueeze(2) * delta.unsqueeze(1))
    Sigma_bar.index_add_(0, assignments, outer)

    # ⟨f_Sb, Ḡb⟩ = Σ_i K(splat_i, template_{bin_i}); ‖Ḡb‖² = π^{D/2}|Σ̄_b|^{1/2}.
    ones = torch.ones(N, dtype=dt, device=device)
    K_i = gaussian_pair_inner_product_torch(
        centres, Sigma, amps, mu_bar[assignments], Sigma_bar[assignments], ones
    )  # (N,)
    inner = torch.zeros(M, dtype=dt, device=device)
    inner.index_add_(0, assignments, K_i)
    norm_sq = template_squared_norm_torch(Sigma_bar)  # (M,)
    return mu_bar, Sigma_bar, inner, norm_sq, w


def _build_representatives_vectorized(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    colors: Optional[torch.Tensor],
    assignments: torch.Tensor,
    *,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Vectorised replacement for the per-bin :func:`_build_representatives`."""
    N, D = centres.shape
    device = centres.device
    dt = centres.dtype
    Sigma = L @ L.transpose(-1, -2)
    sqrt_det = sqrt_det_from_cholesky(L)
    mu_bar, Sigma_bar, inner, norm_sq, w = _segment_templates(
        centres, L, amps, assignments, M, Sigma=Sigma, sqrt_det=sqrt_det
    )
    a_star = l2_optimal_amplitude_torch(inner, norm_sq).clamp_min(0.0)  # (M,)

    # Σ̄ → lower-triangular Cholesky; non-PD / empty bins → identity (then culled
    # by their zero amplitude). A small ridge mirrors the per-bin fallback.
    ridge = 1e-6 * torch.eye(D, dtype=dt, device=device)
    L_bar, info = torch.linalg.cholesky_ex(Sigma_bar + ridge)
    bad = info != 0
    if bool(bad.any()):
        eye = torch.eye(D, dtype=dt, device=device).expand(M, D, D)
        L_bar = torch.where(bad.view(M, 1, 1), eye, L_bar)
        a_star = torch.where(bad, torch.zeros_like(a_star), a_star)

    if colors is not None:
        cdt = colors.dtype
        col = colors.to(dtype=dt)
        extra = tuple(col.shape[1:])
        new_colors = torch.zeros((M,) + extra, dtype=dt, device=device)
        wshape = (N,) + (1,) * len(extra)
        new_colors.index_add_(0, assignments, w.view(wshape) * col)
        new_colors = new_colors.to(dtype=cdt)
    else:
        new_colors = None
    return mu_bar, L_bar, a_star, new_colors


def _score_and_pick(
    centres: torch.Tensor,
    Sigma: torch.Tensor,
    amps: torch.Tensor,
    mu_bar: torch.Tensor,
    Sigma_bar: torch.Tensor,
    cand: torch.Tensor,
    *,
    chunk: int = 50_000,
) -> torch.Tensor:
    """Pick, per splat, the candidate bin maximising the normalised splat→
    template Gaussian inner product ``K(i, Ḡ_b) / ‖Ḡ_b‖``. Chunked over splats
    to bound the ``(chunk, C, D, D)`` working set. Returns ``(N,)`` int64."""
    N = centres.shape[0]
    C = cand.shape[1]
    D = centres.shape[1]
    device = centres.device
    dt = centres.dtype
    norm = template_squared_norm_torch(Sigma_bar).clamp_min(_TINY).sqrt()  # (M,)
    out = torch.empty(N, dtype=torch.int64, device=device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = e - s
        cb = cand[s:e]  # (b, C)
        mu_c = mu_bar[cb]  # (b, C, D)
        Sig_c = Sigma_bar[cb]  # (b, C, D, D)
        mu_i = centres[s:e].unsqueeze(1).expand(b, C, D)
        Sig_i = Sigma[s:e].unsqueeze(1).expand(b, C, D, D)
        a_i = amps[s:e].unsqueeze(1).expand(b, C)
        ones = torch.ones(b, C, dtype=dt, device=device)
        Kij = gaussian_pair_inner_product_torch(mu_i, Sig_i, a_i, mu_c, Sig_c, ones)
        score = Kij / norm[cb]  # (b, C)
        pick = score.argmax(dim=1)  # (b,)
        out[s:e] = cb[torch.arange(b, device=device), pick]
    return out


def _cost_increment_lloyd_vectorized(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    assignments: torch.Tensor,
    *,
    M: int,
    iterations: int,
    candidate_bins_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Vectorised, monotone cost-aware Lloyd refinement.

    Each pass reassigns every splat (synchronously) to the candidate bin whose
    representative template it best projects onto, then rebuilds templates and
    recomputes the global projection energy ``P = Σ_b ⟨f,Ḡ⟩² / ‖Ḡ‖²``. A pass
    is committed only if it strictly *increases* ``P`` (within a numerical
    tolerance; equivalently, strictly decreases the L² residual ‖f‖² − P), so
    the refinement is monotone and never worse than the warm start. An
    equal-energy pass is rejected and iteration stops at the first pass that
    fails to improve ``P``.

    Candidate bins for a splat are the *current* bins of its Morton-order
    neighbours — its spatial neighbours, since splats are Morton-sorted. This
    is an ``O(N·k)`` gather that replaces a per-iteration spatial-hash kNN
    (which dominated runtime at ~18 s/pass on 256K splats); robustness to bin
    drift comes from re-gathering against live assignments each pass.
    """
    if iterations <= 0:
        return assignments
    N = centres.shape[0]
    Sigma = L @ L.transpose(-1, -2)
    sqrt_det = sqrt_det_from_cholesky(L)

    # Precompute Morton-order neighbour splat indices once (fixed geometry).
    order = _morton_order(centres)  # (N,) splat indices in curve order
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N, dtype=np.int64)  # splat → position on the curve
    w = max(1, int(candidate_bins_k) // 2)
    offs = np.arange(-w, w + 1, dtype=np.int64)
    nbr_pos = np.clip(rank[:, None] + offs[None, :], 0, N - 1)  # (N, 2w+1)
    nbr_splats = torch.from_numpy(order[nbr_pos]).to(
        device=device, dtype=torch.int64
    )  # (N, 2w+1) splat indices of each splat's curve neighbours

    def projection_energy(
        assign: torch.Tensor,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        mu_bar, Sigma_bar, inner, norm_sq, _ = _segment_templates(
            centres, L, amps, assign, M, Sigma=Sigma, sqrt_det=sqrt_det
        )
        P = (inner * inner / norm_sq.clamp_min(_TINY)).sum()
        return float(P), mu_bar, Sigma_bar

    best_P, mu_bar, Sigma_bar = projection_energy(assignments)

    for _ in range(int(iterations)):
        # Candidate bins = current bins of this splat's curve neighbours
        # (+ its own bin, which is included since a splat neighbours itself).
        cand = assignments[nbr_splats]  # (N, 2w+1)
        new_assign = _score_and_pick(centres, Sigma, amps, mu_bar, Sigma_bar, cand)
        P_new, mu_bar_new, Sigma_bar_new = projection_energy(new_assign)
        if P_new > best_P + abs(best_P) * 1e-9:
            best_P = P_new
            assignments = new_assign
            mu_bar, Sigma_bar = mu_bar_new, Sigma_bar_new
        else:
            break
    return assignments


# ─────────────────────────────────────────────────────────────────────
# Greedy hierarchical (supp doc Algorithm 4.2)
# ─────────────────────────────────────────────────────────────────────


def _pair_merge_costs(
    mu0: torch.Tensor,
    L0: torch.Tensor,
    a0: torch.Tensor,
    mu1: torch.Tensor,
    L1: torch.Tensor,
    a1: torch.Tensor,
) -> torch.Tensor:
    """Batched residual energy E* after merging aligned cluster pairs.

    All inputs are leading-batched over P pairs (``mu*`` (P, D), ``L*``
    (P, D, D) lower-triangular Cholesky, ``a*`` (P,)). Returns ``(P,)`` —
    ``E* = ‖f‖² − ⟨f, Ḡ⟩²/‖Ḡ‖²`` (supp doc Eq. 2.5), the vectorised
    equivalent of the scalar per-pair cost it replaces.
    """
    D = mu0.shape[-1]
    S0 = L0 @ L0.transpose(-1, -2)
    S1 = L1 @ L1.transpose(-1, -2)
    sd0 = sqrt_det_from_cholesky(L0)
    sd1 = sqrt_det_from_cholesky(L1)
    m0 = a0 * sd0
    m1 = a1 * sd1
    tot = (m0 + m1).clamp_min(_TINY)
    w0 = m0 / tot
    w1 = m1 / tot
    mu_bar = w0.unsqueeze(-1) * mu0 + w1.unsqueeze(-1) * mu1
    d0 = mu0 - mu_bar
    d1 = mu1 - mu_bar
    Sigma_bar = (
        w0.view(-1, 1, 1) * S0
        + w1.view(-1, 1, 1) * S1
        + w0.view(-1, 1, 1) * (d0.unsqueeze(-1) * d0.unsqueeze(-2))
        + w1.view(-1, 1, 1) * (d1.unsqueeze(-1) * d1.unsqueeze(-2))
    )
    ones = torch.ones_like(a0)
    inner = gaussian_pair_inner_product_torch(
        mu0, S0, a0, mu_bar, Sigma_bar, ones
    ) + gaussian_pair_inner_product_torch(mu1, S1, a1, mu_bar, Sigma_bar, ones)
    norm_sq = template_squared_norm_torch(Sigma_bar)
    k01 = gaussian_pair_inner_product_torch(mu0, S0, a0, mu1, S1, a1)
    f_norm_sq = (
        gaussian_self_energy_torch(a0, sd0, D)
        + gaussian_self_energy_torch(a1, sd1, D)
        + 2.0 * k01
    )
    return bin_residual_energy_torch(f_norm_sq, inner, norm_sq)


def _merge_two_clusters(
    cen: torch.Tensor,
    Lc: torch.Tensor,
    am: torch.Tensor,
    a: int,
    b: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moment-match clusters ``a`` and ``b`` into one representative splat.

    Returns ``(mu_bar (D,), L_bar (D, D), a_star scalar)`` — identical math
    to the per-merge step of the original greedy partition.
    """
    D = cen.shape[1]
    idx = torch.tensor([a, b], dtype=torch.int64, device=cen.device)
    mu = cen[idx]
    Lp = Lc[idx]
    ap = am[idx]
    mu_bar, Sigma_bar, _ = kwise_moment_match_torch(mu, Lp, ap)
    inner = bin_inner_product_with_template_torch(mu, Lp, ap, mu_bar, Sigma_bar)
    norm_sq = template_squared_norm_torch(Sigma_bar)
    a_star = l2_optimal_amplitude_torch(inner, norm_sq).clamp_min(0.0)
    try:
        L_bar = torch.linalg.cholesky(Sigma_bar)
    except RuntimeError:
        ridge = 1e-6 * torch.eye(D, dtype=Sigma_bar.dtype, device=cen.device)
        L_bar = torch.linalg.cholesky(Sigma_bar + ridge)
    return mu_bar, L_bar, a_star


def _greedy_partition(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    M_target: int,
) -> torch.Tensor:
    """Bottom-up greedy hierarchical merge (Runnalls), lazy-heap variant.

    Repeatedly merges the active cluster pair with the smallest merge
    residual energy until ``M_target`` clusters remain — the same
    sequential cheapest-pair decisions as the classic Runnalls reduction.

    Efficient incremental implementation: each candidate pair's cost is
    computed once (in batch) instead of re-scanning every pair on every
    iteration. A lazy-deletion min-heap holds candidate edges tagged with
    per-cluster version counters; stale entries (an endpoint was changed or
    consumed) are discarded on pop. After a merge, only the O(k) edges from
    the new cluster to its neighbours (the union of the two parents'
    adjacency) are recomputed and pushed. Clusters live in fixed slots with
    an ``active`` free-list, so no array is spliced. Candidate neighbours
    come from a spatial-hash kNN graph, rebuilt only if the heap drains
    before the target is reached. Roughly ``O(N·k·log(N·k))`` vs. the former
    ``O(N²·k)`` full re-scan (which took ~57 s at N=200).
    """
    device = centres.device
    N = centres.shape[0]
    if N <= M_target or N < 2:
        return torch.arange(max(N, 0), dtype=torch.int64, device=device)

    # Cluster state in fixed slots [0, N); slot i starts as splat i.
    cen = centres.clone()
    Lc = L.clone()
    am = amps.clone()
    active = np.ones(N, dtype=bool)
    version = np.zeros(N, dtype=np.int64)
    members: list[list[int]] = [[i] for i in range(N)]
    adj: list[set[int]] = [set() for _ in range(N)]
    heap: list[tuple[float, int, int, int, int]] = []
    n_active = N

    def push_edges(a_idx: np.ndarray, b_idx: np.ndarray) -> None:
        """Batch-compute costs for aligned pairs and push them to the heap."""
        if len(a_idx) == 0:
            return
        ai = torch.from_numpy(a_idx).to(device=device, dtype=torch.int64)
        bi = torch.from_numpy(b_idx).to(device=device, dtype=torch.int64)
        costs = (
            _pair_merge_costs(cen[ai], Lc[ai], am[ai], cen[bi], Lc[bi], am[bi])
            .detach()
            .cpu()
            .numpy()
        )
        for c, a, b in zip(costs, a_idx.tolist(), b_idx.tolist()):
            lo, hi = (a, b) if a < b else (b, a)
            heapq.heappush(
                heap, (float(c), lo, hi, int(version[lo]), int(version[hi]))
            )
            adj[lo].add(hi)
            adj[hi].add(lo)

    def rebuild(slots: np.ndarray) -> None:
        """Rebuild the candidate kNN graph + heap over the given active slots."""
        heap.clear()
        if len(slots) < 2:
            return
        coords = cen[slots].detach().cpu().numpy()
        grid = BatchedSpatialHashGrid.from_points(
            coords, cell_size=_estimate_cell_size(cen[slots]), device="cpu"
        )
        k = min(8, len(slots))
        _, nb = grid.query_knn(coords, k=k)
        a_list: list[int] = []
        b_list: list[int] = []
        seen: set[tuple[int, int]] = set()
        for ii in range(len(slots)):
            u = int(slots[ii])
            for jj in nb[ii]:
                jj = int(jj)
                if jj < 0 or jj == ii:
                    continue
                v = int(slots[jj])
                lo, hi = (u, v) if u < v else (v, u)
                if (lo, hi) in seen:
                    continue
                seen.add((lo, hi))
                a_list.append(lo)
                b_list.append(hi)
        push_edges(
            np.asarray(a_list, dtype=np.int64), np.asarray(b_list, dtype=np.int64)
        )

    rebuild(np.arange(N, dtype=np.int64))

    while n_active > M_target:
        if not heap:
            slots = np.flatnonzero(active)
            rebuild(slots)
            if not heap:
                # Disconnected residue: force-merge the two nearest active
                # clusters by Euclidean distance to guarantee progress.
                if len(slots) < 2:
                    break
                sub = cen[torch.from_numpy(slots).to(device)]
                d = torch.cdist(sub, sub)
                d.fill_diagonal_(float("inf"))
                fa = int(d.argmin())
                ia, ib = divmod(fa, len(slots))
                push_edges(
                    np.asarray([int(slots[ia])], dtype=np.int64),
                    np.asarray([int(slots[ib])], dtype=np.int64),
                )
        cost, a, b, va, vb = heapq.heappop(heap)
        if not active[a] or not active[b] or version[a] != va or version[b] != vb:
            continue  # stale entry — an endpoint changed or was consumed

        # Merge b into a's slot.
        mu_bar, L_bar, a_star = _merge_two_clusters(cen, Lc, am, a, b)
        cen[a] = mu_bar
        Lc[a] = L_bar
        am[a] = a_star
        members[a].extend(members[b])
        active[b] = False
        version[a] += 1
        n_active -= 1

        # New candidate edges: a vs (neighbours(a) ∪ neighbours(b)) ∩ active.
        cand = (adj[a] | adj[b]) - {a, b}
        adj[a] = set()
        adj[b] = set()
        fresh = [c for c in cand if active[c]]
        if fresh:
            push_edges(
                np.full(len(fresh), a, dtype=np.int64),
                np.asarray(fresh, dtype=np.int64),
            )

    # Build assignments: original splat index → compacted bin index.
    assignments = torch.empty(N, dtype=torch.int64, device=device)
    bin_idx = 0
    for slot in range(N):
        if not active[slot]:
            continue
        for m in members[slot]:
            assignments[m] = bin_idx
        bin_idx += 1
    return assignments


def _estimate_cell_size(centres: torch.Tensor) -> float:
    """Estimate a reasonable spatial-hash cell size from cluster centres.

    Uses a uniform-density approximation: ``(bbox_volume / N)^(1/D) * 2``
    so that the typical NN distance fits comfortably in shell 1.
    """
    N, D = centres.shape
    if N <= 1:
        return 1.0
    coords = centres.detach().cpu().numpy()
    bbox = (coords.max(axis=0) - coords.min(axis=0)).astype(np.float64)
    bbox = np.maximum(bbox, 1e-9)
    typical = float((np.prod(bbox) / max(N, 1)) ** (1.0 / D))
    return max(typical * 2.0, 1e-6)
