"""Greedy hierarchical (Runnalls) merge for substitutive LOD.

The ``greedy*`` methods of
:func:`luxar.gsplats.lod.substitutive.make_substitutive_lod` reduce the
splat set bottom-up: repeatedly merge the active cluster pair with the
smallest closed-form merge residual energy (supp doc Algorithm 4.2) until
``M_target`` clusters remain. Implemented as a lazy-deletion priority
queue with incremental neighbour updates and batched pair-cost evaluation
— ~``O(N k log(N k))``, quality-leading at small ``N`` / ``K``.
"""

from __future__ import annotations

import heapq

import numpy as np
import torch

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
from luxar.gsplats.spatial_hash import BatchedSpatialHashGrid


def _pair_merge_costs(
    mu0: torch.Tensor,
    L0: torch.Tensor,
    a0: torch.Tensor,
    mu1: torch.Tensor,
    L1: torch.Tensor,
    a1: torch.Tensor,
    color0: torch.Tensor | None = None,
    color1: torch.Tensor | None = None,
    color_weight: float = 0.0,
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
    cost = bin_residual_energy_torch(f_norm_sq, inner, norm_sq)
    if color0 is not None and color1 is not None and color_weight > 0.0:
        color_distance_sq = ((color0 - color1) ** 2).sum(dim=-1)
        cost = cost + color_weight * color_distance_sq * f_norm_sq
    return cost


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
    color_features: torch.Tensor | None = None,
    color_weight: float = 0.0,
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
    cluster_colors = color_features.clone() if color_features is not None else None
    cluster_weights = amps * sqrt_det_from_cholesky(L)
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
            _pair_merge_costs(
                cen[ai],
                Lc[ai],
                am[ai],
                cen[bi],
                Lc[bi],
                am[bi],
                cluster_colors[ai] if cluster_colors is not None else None,
                cluster_colors[bi] if cluster_colors is not None else None,
                color_weight,
            )
            .detach()
            .cpu()
            .numpy()
        )
        for c, a, b in zip(costs, a_idx.tolist(), b_idx.tolist()):
            lo, hi = (a, b) if a < b else (b, a)
            heapq.heappush(heap, (float(c), lo, hi, int(version[lo]), int(version[hi])))
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
        if cluster_colors is not None:
            total_weight = (cluster_weights[a] + cluster_weights[b]).clamp_min(_TINY)
            cluster_colors[a] = (
                cluster_weights[a] * cluster_colors[a]
                + cluster_weights[b] * cluster_colors[b]
            ) / total_weight
            cluster_weights[a] = total_weight
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
