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

- ``"kmeans_lloyd"`` (recommended workhorse, default): k-means++ warm
  start on splat means → cost-increment Lloyd refinement. Per supp doc
  Experiment C, dominates amplitude culling on real anisotropic data
  and matches/beats greedy at large compression while running ~8×
  faster.
- ``"kmeans"``: warm-start only, no Lloyd refinement. The supp doc
  shows this is *worse* than amplitude culling on real anisotropic
  data — exposed for contrast / ablation, not as a recommended option.
- ``"greedy"``: bottom-up Runnalls-style merging using closed-form
  pairwise merge cost. Quality-leading at small $N$ and small $K$;
  $\\mathcal{O}(N^2)$ even with spatial-hash candidate pruning.
- ``"greedy_lloyd"``: greedy warm start + Lloyd refinement.

Performance is critical for substitutive Lloyd: the warm-start k-means
and the per-iteration top-k bin lookup both run on PyTorch (CUDA / MPS /
CPU; ``device='auto'``) using
:class:`luxar.utils.spatial_hash.BatchedSpatialHashGrid`. Per-bin merge
math (moment matching, $L^2$-optimal amplitude, residual energy) lives
in :mod:`luxar.gsplats.lod._kernels` and is shared with the additive
axis.

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
from luxar.gsplats.lod._kernels import (
    bin_inner_product_with_template_torch,
    bin_residual_energy_torch,
    bin_squared_norm_torch,
    kwise_moment_match_torch,
    l2_optimal_amplitude_torch,
    template_squared_norm_torch,
)
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


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def make_substitutive_lod(
    data: GSplatData,
    *,
    compression_factor: int = 4,
    levels: int = 3,
    method: MethodName = "kmeans_lloyd",
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
        Partition algorithm. See module docstring.
    lloyd_iterations
        Maximum number of cost-increment Lloyd passes per level (only
        for ``"kmeans_lloyd"`` / ``"greedy_lloyd"``). The loop exits
        early once no splat moves.
    candidate_bins_k
        Top-k spatial-hash candidates considered for each splat during
        Lloyd refinement. Tighter k → faster, slightly worse quality.
    device
        ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"mps"``, or a
        :class:`torch.device`.
    seed
        Optional RNG seed for k-means++ init and Lloyd's randomised
        scan order.
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
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {list(_VALID_METHODS)}, got {method!r}"
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
                    parent_method=method,
                    level_index=level_idx,
                    stats={
                        "n_splats_total": int(N_in),
                        "stop_reason": "input_too_small",
                    },
                )
            )
            break
        M_target = max(1, math.ceil(N_in / K))
        if verbose:
            with asection(
                f"Substitutive level {level_idx}: {N_in} -> {M_target} splats"
            ):
                aprint(f"method={method}")
                new_data = _reduce_one_level(
                    current,
                    M_target=M_target,
                    method=method,
                    lloyd_iterations=lloyd_iterations,
                    candidate_bins_k=candidate_bins_k,
                    device=target_device,
                    seed=None if seed is None else seed + level_idx,
                )
        else:
            new_data = _reduce_one_level(
                current,
                M_target=M_target,
                method=method,
                lloyd_iterations=lloyd_iterations,
                candidate_bins_k=candidate_bins_k,
                device=target_device,
                seed=None if seed is None else seed + level_idx,
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
                parent_method=method,
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
    seed: Optional[int],
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

    # 1) Warm-start partition
    if method.startswith("kmeans"):
        # Shape-aware seeding for the refined ``kmeans_lloyd`` workhorse:
        # cluster on centres augmented with covariance (marginal-σ) features
        # so spatially-near but shape-incompatible splats don't share a bin
        # — the very anisotropy the substitutive format exists to represent.
        # Plain ``kmeans`` stays centres-only as a reproducible baseline.
        if method.endswith("_lloyd"):
            kmeans_input = _augment_with_shape(centres_t, L_t)
        else:
            kmeans_input = centres_t
        assignments = _kmeans_partition(kmeans_input, M=M_target, seed=seed, max_iter=20)
    else:
        assignments = _greedy_partition(centres_t, L_t, amps_t, M_target=M_target)

    # 2) Optional Lloyd cost-increment refinement
    if method.endswith("_lloyd"):
        assignments = _cost_increment_lloyd(
            centres_t,
            L_t,
            amps_t,
            assignments,
            M=M_target,
            iterations=lloyd_iterations,
            candidate_bins_k=candidate_bins_k,
            device=device,
            seed=seed,
        )

    # 3) Bin merge: produce M representative splats
    new_centres, new_L, new_amps, new_colors = _build_representatives(
        centres_t, L_t, amps_t, colors_t, assignments, M=M_target
    )

    # Cull empty bins (if k-means produced any).
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
# k-means warm start (PyTorch)
# ─────────────────────────────────────────────────────────────────────


def _augment_with_shape(
    centres: torch.Tensor,
    L: torch.Tensor,
    *,
    weight: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Augment centres with covariance (shape) features for k-means seeding.

    Each splat contributes its per-dimension marginal standard deviations
    ``σ_i = sqrt(Σ_ii) = sqrt(Σ_j L[i,j]²)``, log-compressed to tame the
    dynamic range, and rescaled so the shape block carries roughly
    ``weight`` × the spatial block's spread. The result is the centres
    concatenated with the scaled log-σ features — clustering on it keeps
    spatially-near but differently-shaped (anisotropic) splats apart.

    Args:
        centres: ``(N, D)`` splat centres.
        L: ``(N, D, D)`` lower-triangular Cholesky factors (Σ = L Lᵀ).
        weight: Relative influence of the shape block vs. the spatial block
            (0 → centres-only; larger → shape dominates). Default 0.5.
        eps: Numerical floor for the log and the scale ratio.

    Returns:
        ``(N, 2D)`` augmented feature tensor (same dtype/device as ``centres``).
    """
    if weight <= 0:
        return centres
    # Marginal σ per dimension: sqrt of the row sum of squares of L.
    sigma = L.pow(2).sum(dim=2).clamp_min(eps).sqrt()  # (N, D)
    log_sigma = torch.log(sigma.clamp_min(eps))
    # Scale the shape block to be commensurate with the spatial block: match
    # the mean per-dim spread, then apply the relative ``weight``.
    centre_scale = centres.std(dim=0).mean()
    shape_scale = log_sigma.std(dim=0).mean().clamp_min(eps)
    beta = weight * (centre_scale / shape_scale)
    return torch.cat([centres, beta * log_sigma], dim=1)


def _kmeans_partition(
    points: torch.Tensor,
    M: int,
    *,
    seed: Optional[int] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> torch.Tensor:
    """k-means with k-means++ initialisation.

    Returns a length-N int64 tensor of cluster assignments in [0, M).
    Uses chunked all-pairs distance to keep peak memory bounded for
    moderate ``M``.
    """
    N, D = points.shape
    device = points.device
    M = min(M, N)

    rng = np.random.RandomState(seed)
    centres = _kmeanspp_init(points, M, rng)

    for _ in range(max_iter):
        assignments = _assign_to_nearest(points, centres)
        new_centres = _scatter_mean(points, assignments, M)
        # Empty cluster: keep old centroid.
        empty_mask = torch.bincount(assignments, minlength=M).to(device=device) == 0
        new_centres = torch.where(empty_mask.unsqueeze(-1), centres, new_centres)
        shift = (new_centres - centres).abs().max()
        centres = new_centres
        if float(shift) < tol:
            break

    return _assign_to_nearest(points, centres)


def _kmeanspp_init(
    points: torch.Tensor, M: int, rng: np.random.RandomState
) -> torch.Tensor:
    """k-means++ seeding (Arthur & Vassilvitskii 2007)."""
    N, D = points.shape
    device = points.device
    centres = torch.empty(M, D, dtype=points.dtype, device=device)
    first_idx = int(rng.randint(0, N))
    centres[0] = points[first_idx]
    min_d2 = ((points - centres[0]) ** 2).sum(dim=1)
    for k in range(1, M):
        probs = min_d2.detach().cpu().numpy()
        if probs.sum() <= 0:
            # Degenerate: all remaining points coincide. Pick any.
            idx = int(rng.randint(0, N))
        else:
            probs = probs / probs.sum()
            idx = int(rng.choice(N, p=probs))
        centres[k] = points[idx]
        new_d2 = ((points - centres[k]) ** 2).sum(dim=1)
        min_d2 = torch.minimum(min_d2, new_d2)
    return centres


def _assign_to_nearest(
    points: torch.Tensor, centres: torch.Tensor, *, chunk: int = 4096
) -> torch.Tensor:
    """Assign each point to the closest centre by Euclidean distance.

    Memory-bounded: processes ``points`` in chunks of ``chunk`` rows so
    the peak ``(chunk, M, D)`` tensor stays small.
    """
    N = points.shape[0]
    out = torch.empty(N, dtype=torch.int64, device=points.device)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        diff = points[start:end].unsqueeze(1) - centres.unsqueeze(0)
        d2 = (diff * diff).sum(dim=-1)  # (chunk, M)
        out[start:end] = d2.argmin(dim=1)
    return out


def _scatter_mean(
    points: torch.Tensor, assignments: torch.Tensor, M: int
) -> torch.Tensor:
    """Per-cluster mean of ``points`` grouped by ``assignments``."""
    D = points.shape[1]
    device = points.device
    summed = torch.zeros(M, D, dtype=points.dtype, device=device)
    counts = torch.zeros(M, dtype=points.dtype, device=device)
    summed.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), points)
    counts.scatter_add_(
        0, assignments, torch.ones(points.shape[0], dtype=points.dtype, device=device)
    )
    counts_safe = torch.clamp(counts, min=1.0)
    return summed / counts_safe.unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────
# Greedy hierarchical (supp doc Algorithm 4.2)
# ─────────────────────────────────────────────────────────────────────


def _greedy_partition(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    M_target: int,
) -> torch.Tensor:
    """Bottom-up greedy hierarchical merge.

    At each step: among all candidate cluster pairs (spatially close,
    via :class:`BatchedSpatialHashGrid`), pick the pair whose union
    has the smallest residual energy and merge. Repeat until
    ``M_target`` clusters remain.

    Naive implementation: rebuilds the spatial-hash index of cluster
    centres after each merge. $\\mathcal{O}(N^2 \\log N)$ in the worst
    case; intended for small/medium $N$ (< few thousand) where greedy
    is competitive on quality. Use ``"kmeans_lloyd"`` at large $N$.
    """
    device = centres.device
    N = centres.shape[0]
    members: list[list[int]] = [[i] for i in range(N)]
    active_centres = centres.clone()
    active_L = L.clone()
    active_amps = amps.clone()

    while len(members) > M_target:
        # Candidate pairs via spatial hash on cluster centres.
        n_active = len(members)
        if n_active < 2:
            break
        # Generous cell_size: median pairwise distance (cheap to estimate).
        cell_size = _estimate_cell_size(active_centres)
        grid = BatchedSpatialHashGrid.from_points(
            active_centres.detach().cpu().numpy(),
            cell_size=cell_size,
            device="cpu",  # small N; CPU faster than GPU launch overhead
        )
        # k-NN over cluster centres; exclude self-pair via [:, 1:].
        k = min(8, n_active)
        _, neighbour_idx = grid.query_knn(active_centres.detach().cpu().numpy(), k=k)
        # Find best pair to merge.
        best_pair: Optional[tuple[int, int]] = None
        best_delta = float("inf")
        seen_pairs: set[tuple[int, int]] = set()
        for i in range(n_active):
            for j_raw in neighbour_idx[i]:
                j = int(j_raw)
                if j == i or j < 0:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                delta = _merge_cost_pair(
                    active_centres[a],
                    active_centres[b],
                    active_L[a],
                    active_L[b],
                    active_amps[a],
                    active_amps[b],
                )
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (a, b)
        if best_pair is None:
            # No spatial neighbours at this scale — fall back to the
            # closest pair by Euclidean distance to make progress.
            d = torch.cdist(active_centres, active_centres)
            d.fill_diagonal_(float("inf"))
            flat_argmin = int(d.argmin())
            a, b = divmod(flat_argmin, n_active)
            if a > b:
                a, b = b, a
            best_pair = (a, b)
        a, b = best_pair
        # Merge a and b into a new cluster (replace a's slot, drop b's).
        merged_members = members[a] + members[b]
        merged_centres = torch.stack([active_centres[a], active_centres[b]], dim=0)
        merged_L = torch.stack([active_L[a], active_L[b]], dim=0)
        merged_amps = torch.stack([active_amps[a], active_amps[b]], dim=0)
        mu_bar, Sigma_bar, _ = kwise_moment_match_torch(
            merged_centres, merged_L, merged_amps
        )
        merged_chol = torch.linalg.cholesky(Sigma_bar)

        # Compute representative amplitude from L²-optimal formula.
        merged_amp_t = _compute_l2_optimal_amplitude_for_bin(
            merged_centres, merged_L, merged_amps, mu_bar, Sigma_bar
        )

        # Splice the merged result back into the active arrays.
        members[a] = merged_members
        members.pop(b)
        active_centres = torch.cat(
            [
                active_centres[:a],
                mu_bar.unsqueeze(0),
                active_centres[a + 1 : b],
                active_centres[b + 1 :],
            ],
            dim=0,
        )
        active_L = torch.cat(
            [
                active_L[:a],
                merged_chol.unsqueeze(0),
                active_L[a + 1 : b],
                active_L[b + 1 :],
            ],
            dim=0,
        )
        active_amps = torch.cat(
            [
                active_amps[:a],
                merged_amp_t.unsqueeze(0),
                active_amps[a + 1 : b],
                active_amps[b + 1 :],
            ],
            dim=0,
        )

    # Build assignments tensor: original_idx -> bin_idx.
    assignments = torch.empty(N, dtype=torch.int64, device=device)
    for bin_idx, member_list in enumerate(members):
        for m in member_list:
            assignments[m] = bin_idx
    return assignments


def _merge_cost_pair(
    mu_a: torch.Tensor,
    mu_b: torch.Tensor,
    L_a: torch.Tensor,
    L_b: torch.Tensor,
    amp_a: torch.Tensor,
    amp_b: torch.Tensor,
) -> float:
    """Residual energy after merging two splats into a single template."""
    centres = torch.stack([mu_a, mu_b], dim=0)
    L_pair = torch.stack([L_a, L_b], dim=0)
    amps_pair = torch.stack([amp_a, amp_b], dim=0)
    mu_bar, Sigma_bar, _ = kwise_moment_match_torch(centres, L_pair, amps_pair)
    template_inner = bin_inner_product_with_template_torch(
        centres, L_pair, amps_pair, mu_bar, Sigma_bar
    )
    template_norm_sq = template_squared_norm_torch(Sigma_bar)
    bin_norm_sq = bin_squared_norm_torch(centres, L_pair, amps_pair)
    residual = bin_residual_energy_torch(bin_norm_sq, template_inner, template_norm_sq)
    return float(residual)


def _compute_l2_optimal_amplitude_for_bin(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    mu_bar: torch.Tensor,
    Sigma_bar: torch.Tensor,
) -> torch.Tensor:
    template_inner = bin_inner_product_with_template_torch(
        centres, L, amps, mu_bar, Sigma_bar
    )
    template_norm_sq = template_squared_norm_torch(Sigma_bar)
    return l2_optimal_amplitude_torch(template_inner, template_norm_sq)


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


# ─────────────────────────────────────────────────────────────────────
# Cost-increment Lloyd refinement (supp doc Algorithm 4.3)
# ─────────────────────────────────────────────────────────────────────


def _cost_increment_lloyd(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    assignments: torch.Tensor,
    *,
    M: int,
    iterations: int,
    candidate_bins_k: int,
    device: torch.device,
    seed: Optional[int],
) -> torch.Tensor:
    """Iteratively move splats to bins where they reduce per-bin residual.

    Each pass:

    1. Compute current per-bin residual energy $E_b^\\star$.
    2. Build a :class:`BatchedSpatialHashGrid` on bin mass-weighted
       means; query top-``candidate_bins_k`` candidate bins per splat.
    3. For each splat ``i`` (random order): for each candidate ``b``
       different from its current bin ``a``, compute
       $\\Delta_b(i) = [E_a^\\star(\\mathcal{S}_a\\setminus\\{i\\}) +
       E_b^\\star(\\mathcal{S}_b\\cup\\{i\\})] - [E_a^\\star + E_b^\\star]$.
       Move ``i`` to the candidate with the most negative $\\Delta$.

    Stops early when no splat moves in a full pass. The per-bin
    residual is recomputed from scratch on each candidate evaluation
    (the bin-merge math is closed-form and inexpensive for typical
    bin sizes ``K = N/M``); we deliberately keep the per-iteration
    cost honest rather than maintaining incrementally-updated state
    that would complicate correctness.
    """
    N = centres.shape[0]
    rng = np.random.RandomState(seed)

    for it in range(iterations):
        bin_members = _bin_membership_lists(assignments, M)
        bin_mu = _per_bin_mu_bar(centres, L, amps, bin_members, M, device)

        cell_size = _estimate_cell_size(bin_mu)
        grid = BatchedSpatialHashGrid.from_points(
            bin_mu.detach().cpu().numpy(),
            cell_size=cell_size,
            device="auto",
        )
        # Top-k candidate bins per splat — query against current bin centres.
        k = min(candidate_bins_k + 1, M)  # +1 to allow excluding current bin
        _, candidate_bins_np = grid.query_knn(centres.detach().cpu().numpy(), k=k)
        candidate_bins = torch.from_numpy(candidate_bins_np).to(device=device)

        # Cache per-bin residual energy for the unchanged baseline.
        bin_residuals = torch.zeros(M, dtype=torch.float64, device=device)
        for b, members in enumerate(bin_members):
            if not members:
                continue
            idx = torch.tensor(members, dtype=torch.int64, device=device)
            bin_residuals[b] = _bin_residual_for_indices(
                centres[idx], L[idx], amps[idx]
            )

        order = rng.permutation(N)
        any_moved = False
        for i in order:
            i = int(i)
            a = int(assignments[i].item())
            cands_for_i = candidate_bins[i].tolist()
            best_delta = 0.0
            best_b = -1
            E_a = float(bin_residuals[a].item())
            members_a = [j for j in bin_members[a] if j != i]
            E_a_minus_i = (
                _bin_residual_for_indices(
                    centres[members_a] if members_a else centres[:0],
                    L[members_a] if members_a else L[:0],
                    amps[members_a] if members_a else amps[:0],
                )
                if members_a
                else torch.zeros((), dtype=torch.float64, device=device)
            )
            for b in cands_for_i:
                if b == a or b < 0:
                    continue
                E_b = float(bin_residuals[b].item())
                members_b_plus = bin_members[b] + [i]
                idx_b_plus = torch.tensor(
                    members_b_plus, dtype=torch.int64, device=device
                )
                E_b_plus_i = _bin_residual_for_indices(
                    centres[idx_b_plus], L[idx_b_plus], amps[idx_b_plus]
                )
                delta = (float(E_a_minus_i.item()) + float(E_b_plus_i.item())) - (
                    E_a + E_b
                )
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_b = b
            if best_b >= 0:
                # Commit the move.
                assignments[i] = best_b
                bin_members[a].remove(i)
                bin_members[best_b].append(i)
                # Recompute residuals for the two affected bins.
                if bin_members[a]:
                    idx_a = torch.tensor(
                        bin_members[a], dtype=torch.int64, device=device
                    )
                    bin_residuals[a] = _bin_residual_for_indices(
                        centres[idx_a], L[idx_a], amps[idx_a]
                    )
                else:
                    bin_residuals[a] = torch.zeros(
                        (), dtype=torch.float64, device=device
                    )
                idx_b = torch.tensor(
                    bin_members[best_b], dtype=torch.int64, device=device
                )
                bin_residuals[best_b] = _bin_residual_for_indices(
                    centres[idx_b], L[idx_b], amps[idx_b]
                )
                any_moved = True
        if not any_moved:
            break
    return assignments


def _bin_membership_lists(assignments: torch.Tensor, M: int) -> list[list[int]]:
    """Convert per-splat assignments to per-bin member-index lists."""
    out: list[list[int]] = [[] for _ in range(M)]
    for i, b in enumerate(assignments.tolist()):
        if 0 <= b < M:
            out[b].append(i)
    return out


def _per_bin_mu_bar(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    bin_members: list[list[int]],
    M: int,
    device: torch.device,
) -> torch.Tensor:
    """Mass-weighted bin centres (for the spatial-hash on bin centres)."""
    D = centres.shape[1]
    out = torch.zeros(M, D, dtype=centres.dtype, device=device)
    for b, members in enumerate(bin_members):
        if not members:
            continue
        idx = torch.tensor(members, dtype=torch.int64, device=device)
        mu_bar, _, _ = kwise_moment_match_torch(centres[idx], L[idx], amps[idx])
        out[b] = mu_bar
    return out


def _bin_residual_for_indices(
    centres_b: torch.Tensor,
    L_b: torch.Tensor,
    amps_b: torch.Tensor,
) -> torch.Tensor:
    """$E_j^\\star$ for one bin's member splats."""
    K = centres_b.shape[0]
    if K == 0:
        return torch.zeros((), dtype=centres_b.dtype, device=centres_b.device)
    mu_bar, Sigma_bar, _ = kwise_moment_match_torch(centres_b, L_b, amps_b)
    template_inner = bin_inner_product_with_template_torch(
        centres_b, L_b, amps_b, mu_bar, Sigma_bar
    )
    template_norm_sq = template_squared_norm_torch(Sigma_bar)
    bin_norm_sq = bin_squared_norm_torch(centres_b, L_b, amps_b)
    return bin_residual_energy_torch(bin_norm_sq, template_inner, template_norm_sq)


# ─────────────────────────────────────────────────────────────────────
# Final bin merge: build per-bin representative splats
# ─────────────────────────────────────────────────────────────────────


def _build_representatives(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    colors: Optional[torch.Tensor],
    assignments: torch.Tensor,
    *,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Produce one representative splat per bin (supp doc Algorithm 4.1)."""
    D = centres.shape[1]
    device = centres.device
    new_centres = torch.zeros(M, D, dtype=centres.dtype, device=device)
    new_L = (
        torch.eye(D, dtype=centres.dtype, device=device).unsqueeze(0).repeat(M, 1, 1)
    )
    new_amps = torch.zeros(M, dtype=centres.dtype, device=device)
    if colors is not None:
        new_colors = torch.zeros(
            (M,) + colors.shape[1:], dtype=colors.dtype, device=device
        )
    else:
        new_colors = None

    bin_members = _bin_membership_lists(assignments, M)
    for b, members in enumerate(bin_members):
        if not members:
            continue
        idx = torch.tensor(members, dtype=torch.int64, device=device)
        c_b = centres[idx]
        L_b = L[idx]
        a_b = amps[idx]
        mu_bar, Sigma_bar, weights = kwise_moment_match_torch(c_b, L_b, a_b)
        template_inner = bin_inner_product_with_template_torch(
            c_b, L_b, a_b, mu_bar, Sigma_bar
        )
        template_norm_sq = template_squared_norm_torch(Sigma_bar)
        a_star = l2_optimal_amplitude_torch(template_inner, template_norm_sq)
        # Sigma_bar -> Cholesky (lower triangular).
        try:
            L_bar = torch.linalg.cholesky(Sigma_bar)
        except RuntimeError:
            # Numerically degenerate (non-PD). Add a small ridge.
            ridge = 1e-6 * torch.eye(D, dtype=Sigma_bar.dtype, device=device)
            L_bar = torch.linalg.cholesky(Sigma_bar + ridge)
        new_centres[b] = mu_bar
        new_L[b] = L_bar
        new_amps[b] = torch.clamp(a_star, min=0.0)
        if new_colors is not None and colors is not None:
            # Mass-weighted colour mean using the bin weights.
            color_b = colors[idx].to(dtype=weights.dtype)
            new_colors[b] = (
                (weights.unsqueeze(-1) * color_b).sum(dim=0).to(dtype=colors.dtype)
            )

    return new_centres, new_L, new_amps, new_colors
