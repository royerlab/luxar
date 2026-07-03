"""L2 mixture-to-mixture refinement for substitutive LOD (silent tensor kernel).

The ``refine="l2"`` mode of
:func:`luxar.gsplats.lod.substitutive.make_substitutive_lod` post-optimizes each
merged level: starting from the coverage-inflated moment-matched seed, Adam
optimizes the coarse splats' ``(mu, L, a)`` to minimize the closed-form
mixture-to-mixture L² residual

.. math::

    \\|f - g\\|^2 = \\|f\\|^2 - 2\\langle f, g\\rangle + \\|g\\|^2

where every term expands into pairwise Gaussian inner products
(:mod:`luxar.gsplats.lod._kernels`) over sparse neighbour pair lists. Unlike the
per-bin merge (whose objective is structurally blind to cross-bin overlap), this
objective *contains* the coverage gaps and the over-blur, so the refit widens
splats exactly where the field is flat and keeps isolated structure tight
(measured: rel-L² 0.089 vs 0.151 for the β=3 merge on flat fields, 0.141 vs
0.233 on isolated blobs with peak preservation 0.99 vs 0.91).

Hard-won stability discipline (each violation was observed to corrupt results):

1. **Trusted checkpoints.** A truncated / stale sparse pair list is *exploitable*:
   the optimizer inflates amplitudes/widths to harvest cross terms a capped
   ``\\|g\\|^2`` pair list cannot see (+41 % mass observed). Pair lists are
   therefore rebuilt from the *current* geometry every ``rebuild_every`` steps,
   and the objective is only **trusted** — compared, snapshotted — immediately
   after a rebuild. The seed is evaluated first, so the returned best iterate is
   provably never worse than the seed in the trusted metric.
2. **PD by construction.** Coarse Cholesky diagonals are softplus-floored, so
   every covariance sum entering
   :func:`~luxar.gsplats.lod._kernels.gaussian_pair_inner_product_torch` is PD
   and its non-PD ``torch.where`` masking (a NaN-gradient hazard) is
   structurally unreachable through grad-carrying tensors. A defensive NaN/Inf
   gradient step-skip remains as a second layer.
3. **Dimensionless learning rates.** Coordinates are standardized internally by
   a single isotropic bbox scale (triangularity of ``L`` preserved), so the lr
   constants in :class:`L2RefineConfig` are scale-free; the result is
   un-standardized on return (the objective is scale-equivariant, so the
   minimizer is invariant).

Two further structural safeguards:

4. **Mass manifold.** Amplitudes are renormalized inside the forward pass so the
   coarse mixture's total mass ``Σ a·|Σ|^{1/2}`` equals the *fine* mixture's
   exactly at every iterate. Total mass is the DC every additive (X-ray)
   projection integrates to, so this pins brightness across LOD levels, kills
   the mass-inflation exploit direction outright, and removes a transient Adam
   overshoot (~+30 % mass early in optimization) observed with unconstrained
   amplitudes. The *raw* (un-normalized) seed is still the first trusted
   candidate, so the never-worse-than-seed guarantee is unconditional.
5. **Unbiased minibatching (both terms).** When a sparse pair list exceeds
   ``step_pair_budget``, each gradient step samples that many pairs uniformly
   from the full per-rebuild list (sum scaled by ``P/B`` — unbiased, since each
   term is a plain sum over pairs). This applies to BOTH the fine→coarse
   ``\\langle f,g\\rangle`` term and the coarse-coarse cross term of
   ``\\|g\\|^2``; otherwise a large first-level reduction (M coarse splats →
   ~M·cc_k/2 grad-carrying pairs) would OOM. The no-grad trusted evaluation
   instead *chunks* the full lists (bounded working set without subsampling).
   Per-window *fine-subset* sampling was rejected: coarse splats not covered by
   a window's subset would see only the ``\\|g\\|^2`` shrink force and decay
   systematically. Large lists are parked on the CPU and gathered per step.

Layering contract (same as ``warm_start.py`` / ``kmeans_lloyd.py`` /
``greedy.py``): this module is a silent tensor-level kernel — it knows nothing
about ``GSplatData``, bins, or LOD levels, and never logs. The orchestrator in
``substitutive.py`` owns data conversion, logging, and stats plumbing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from luxar.gsplats.lod._kernels import (
    gaussian_pair_inner_product_torch,
    gaussian_self_energy_torch,
    sqrt_det_from_cholesky,
)
from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus_torch
from luxar.utils.spatial_hash import BatchedSpatialHashGrid

__all__ = ["L2RefineConfig", "l2_refine_mixture"]

_EVAL_CHUNK_PAIRS = 2_000_000  # no-grad kernel evals are chunked to bound memory


@dataclass(frozen=True)
class L2RefineConfig:
    """Hyperparameters of the L2 refit (stability-critical constants, not a
    tuning surface — only ``iters`` is exposed on the public builders)."""

    iters: int = 120  # Adam steps
    rebuild_every: int = 20  # pair-list rebuild + trusted-E cadence
    cross_k: int = 16  # kNN coarse neighbours per fine splat
    cc_k: int = 64  # kNN coarse neighbours per coarse splat
    cross_radius_sigmas: float = 3.0  # prune beyond this * (med σ_c + med σ_f)
    cc_radius_sigmas: float = 5.0  # prune beyond this * med σ_c
    lr_shape: float = 1e-2  # Adam lr for raw_diag / strict_lower / raw_a
    lr_center_scale: float = 0.05  # center lr = this * median σ_c (standardized)
    step_pair_budget: int = 2_000_000  # cross pairs per gradient step (minibatch)
    early_stop_patience: int = 3  # trusted checkpoints w/o improvement → stop
    diag_floor_scale: float = 1e-4  # softplus diag floor = this * median seed σ
    mass_drift_warn: float = 0.10  # |mass_vs_fine − 1| beyond this → stats warning


# ─────────────────────────────────────────────────────────────────────
# Pair lists
# ─────────────────────────────────────────────────────────────────────

#: Dims whose extent is below this fraction of the max extent are treated as
#: degenerate when sizing the spatial-hash cell. A (near-)constant dim — e.g.
#: the barrier coordinate of a ``coarsen_dims`` group, identical across the
#: whole group — contributes nothing to neighbour distances, but including its
#: ~zero extent in the bbox product collapses the cell orders of magnitude
#: below the true neighbour spacing; the shell search (``_KNN_MAX_SHELL``)
#: then finds nothing and every kNN query degenerates to a brute-force scan of
#: the whole set (measured 14x slowdown at 20k/5k; effectively a hang at
#: realistic sizes). 1e-3 is deliberately generous: a dim thinner than 0.1 %
#: of the max extent perturbs squared neighbour distances by < 1e-6
#: relatively, while excluding it merely rounds the cell UP (more candidates
#: per cell — still exact, mildly slower), never down into the brute-force
#: cliff. This also catches "nearly constant" barrier dims (float jitter on a
#: shared value), not just exactly-constant ones.
_DEGENERATE_EXTENT_FRAC = 1e-3


def _hash_cell_size(coarse_mu_np: np.ndarray) -> float:
    """Spatial-hash cell ≈ 2× the typical coarse NN spacing (greedy.py
    convention), computed over the EFFECTIVE dims only.

    Only dims with materially nonzero extent (``> _DEGENERATE_EXTENT_FRAC ×
    max extent``) enter the bbox product and the root — degenerate dims add
    nothing to neighbour distance, so they must not shrink the cell. kNN
    expands shells automatically; this only tunes performance, but an
    underestimate of orders of magnitude defeats the shell search entirely.
    """
    M = coarse_mu_np.shape[0]
    ext = (coarse_mu_np.max(0) - coarse_mu_np.min(0)).astype(np.float64)
    live = ext > _DEGENERATE_EXTENT_FRAC * float(ext.max())
    if bool(live.any()):
        spacing = float((np.prod(ext[live]) / max(M, 1)) ** (1.0 / int(live.sum())))
    else:
        spacing = 0.0  # all points coincide; any positive cell works
    return max(2.0 * spacing, 1e-6)


def _knn_pairs(
    query_np: np.ndarray,
    grid: BatchedSpatialHashGrid,
    k: int,
    radius: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """kNN pairs ``(query_idx, point_idx)`` pruned to ``dist < radius``.

    Generous-k kNN plus a radius prune emulates a radius query using the
    GPU-batched ``query_knn`` path (``query_radius`` is a per-query NumPy loop
    with jagged output — unusable at millions of queries).
    """
    k = min(k, len(grid))
    dist, idx = grid.query_knn(query_np, k=k)
    keep = (idx >= 0) & (dist < radius)
    qi, col = np.nonzero(keep)
    return (
        torch.from_numpy(qi.astype(np.int64)).to(device),
        torch.from_numpy(idx[qi, col].astype(np.int64)).to(device),
    )


def _build_pair_lists(
    fine_mu_np: np.ndarray,
    coarse_mu_np: np.ndarray,
    *,
    config: L2RefineConfig,
    sigma_c: float,
    sigma_f: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse pair lists against the CURRENT coarse geometry.

    Returns ``(ci, cb, bi, bj)``: fine→coarse cross pairs ``(fine ci, coarse
    cb)`` and coarse-coarse pairs ``(bi < bj)`` (each unordered pair once; the
    ×2 factor is applied in the objective). The cc list is OR-symmetrized: an
    unordered pair {a,b} is kept if ``a`` is in ``b``'s kNN OR ``b`` is in
    ``a``'s — the per-query kNN truncation (``cc_k``) is asymmetric in dense
    regions, so keeping only directed ``a→b`` edges (the naive ``bi<bj``) would
    silently drop genuine near-pairs and under-count ‖g‖². Kept standalone — a
    future tile-merge stitcher is the most likely second consumer of this piece.
    """
    M = coarse_mu_np.shape[0]
    cell = _hash_cell_size(coarse_mu_np)
    grid = BatchedSpatialHashGrid.from_points(
        coarse_mu_np, cell_size=cell, device=str(device)
    )
    r_cross = config.cross_radius_sigmas * (sigma_c + sigma_f)
    r_cc = config.cc_radius_sigmas * sigma_c
    ci, cb = _knn_pairs(fine_mu_np, grid, config.cross_k, r_cross, device)
    bi, bj = _knn_pairs(coarse_mu_np, grid, config.cc_k + 1, r_cc, device)
    # OR-symmetrize + dedup: canonicalize each directed edge to (min, max),
    # drop self-pairs, and keep each unordered pair once. torch.unique on the
    # flattened key recovers {a,b} whenever EITHER endpoint saw the other.
    lo = torch.minimum(bi, bj)
    hi = torch.maximum(bi, bj)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    if lo.numel():
        key = torch.unique(lo * M + hi)
        bi2, bj2 = key // M, key % M
    else:
        bi2 = bj2 = lo  # empty
    return ci, cb, bi2, bj2


# ─────────────────────────────────────────────────────────────────────
# Objective pieces
# ─────────────────────────────────────────────────────────────────────


def _pair_K(
    mu_i: torch.Tensor,
    L_i: torch.Tensor,
    a_i: torch.Tensor,
    mu_j: torch.Tensor,
    L_j: torch.Tensor,
    a_j: torch.Tensor,
) -> torch.Tensor:
    S_i = L_i @ L_i.transpose(-1, -2)
    S_j = L_j @ L_j.transpose(-1, -2)
    return gaussian_pair_inner_product_torch(mu_i, S_i, a_i, mu_j, S_j, a_j)


def _pair_K_sum_chunked(
    mu_i: torch.Tensor,
    L_i: torch.Tensor,
    a_i: torch.Tensor,
    mu_j: torch.Tensor,
    L_j: torch.Tensor,
    a_j: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
) -> torch.Tensor:
    """No-grad pair-kernel sum, chunked to bound the (P, D, D) working set."""
    total = torch.zeros((), dtype=mu_i.dtype, device=mu_i.device)
    for s in range(0, idx_i.numel(), _EVAL_CHUNK_PAIRS):
        e = min(s + _EVAL_CHUNK_PAIRS, idx_i.numel())
        ii, jj = idx_i[s:e], idx_j[s:e]
        total = (
            total
            + _pair_K(mu_i[ii], L_i[ii], a_i[ii], mu_j[jj], L_j[jj], a_j[jj]).sum()
        )
    return total


class _PairMinibatch:
    """A sparse pair list that bounds the per-step grad working set.

    Small lists stay on-device and are used whole every step (scale 1). Lists
    above ``budget`` pairs are parked on the CPU and a fresh uniform ``budget``
    sample is drawn per step (scale = n/budget — unbiased, since the summed
    kernel is a plain sum over pairs). Shared by the fine→coarse (fg) and
    coarse-coarse (cc) terms so both are memory-bounded identically.
    """

    def __init__(
        self,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        budget: int,
        device: torch.device,
    ) -> None:
        self.n = int(idx_i.numel())
        self.budget = budget
        self.device = device
        self.minibatched = self.n > budget
        # Park large lists on CPU (the per-step sample is gathered to device).
        self._i = idx_i.cpu() if self.minibatched else idx_i
        self._j = idx_j.cpu() if self.minibatched else idx_j

    def sample(
        self, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if not self.minibatched:
            return self._i, self._j, 1.0
        pos = torch.randint(self.n, (self.budget,), generator=generator)
        return (
            self._i[pos].to(self.device),
            self._j[pos].to(self.device),
            (self.n / self.budget),
        )


def _g_self_energy(L: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Diagonal part of ‖g‖²: Σ_b a²π^{D/2}|Σ_b|^{1/2} (per-splat, O(M), exact)."""
    return gaussian_self_energy_torch(a, sqrt_det_from_cholesky(L), L.shape[-1]).sum()


def _g_norm_sq(
    mu: torch.Tensor,
    L: torch.Tensor,
    a: torch.Tensor,
    bi: torch.Tensor,
    bj: torch.Tensor,
    cc_scale: float = 1.0,
) -> torch.Tensor:
    """‖g‖² = Σ_b a²π^{D/2}|Σ_b|^{1/2} + 2·cc_scale·Σ_{(b,c)∈pairs} K_bc.

    Differentiable (grad-step path). ``bi``/``bj`` may be a *minibatch* of the
    coarse-coarse pairs, in which case ``cc_scale = n_cc / |batch|`` keeps the
    cross sum unbiased — mirroring the fine→coarse ``fg`` term, so the grad
    working set is bounded on both terms (the no-grad trusted path chunks the
    full cc list instead; see ``trusted_eval``)."""
    self_e = _g_self_energy(L, a)
    if bi.numel() == 0:
        return self_e
    cross = _pair_K(mu[bi], L[bi], a[bi], mu[bj], L[bj], a[bj]).sum()
    return self_e + 2.0 * cc_scale * cross


def _total_mass(L: torch.Tensor, a: torch.Tensor) -> float:
    return float((a * sqrt_det_from_cholesky(L)).sum())


# ─────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────


class _RefineState:
    """Mutable optimization state (parameterization + best-iterate tracking).

    Groups what the trusted-checkpoint evaluation and the step loop share, so
    :func:`l2_refine_mixture` stays a readable orchestration of phases.
    """

    def __init__(
        self,
        smu0: torch.Tensor,
        sL0: torch.Tensor,
        sa0: torch.Tensor,
        frozen_dims: tuple[int, ...],
        config: L2RefineConfig,
        sigma_c: float,
        mass_target: float,
    ) -> None:
        M, D = smu0.shape
        f32 = smu0.dtype
        device = smu0.device
        self.smu0 = smu0
        self.mass_target = mass_target
        free = torch.ones(D, dtype=f32, device=device)
        for d in frozen_dims:
            free[d] = 0.0
        # center_free zeroes barrier coordinates of the center delta; L_free
        # keeps only the free×free lower-triangular block learnable (any entry
        # whose row OR column is a barrier dim stays at the seed value, so Σ's
        # barrier rows/cols are bit-identical to the seed's).
        self.center_free = free.view(1, D)
        tril = torch.tril(torch.ones(D, D, dtype=f32, device=device))
        L_free = tril * free.view(D, 1) * free.view(1, D)  # incl. free diagonal
        self.strict_free = L_free * (1.0 - torch.eye(D, dtype=f32, device=device))
        self.diag_free = free  # (D,)
        self.L_frozen = (sL0 * (1.0 - L_free)).detach()  # constant part

        # PD by construction: softplus-floored free diagonal.
        self.floor = config.diag_floor_scale * max(sigma_c, 1e-12)
        seed_diag = torch.diagonal(sL0, dim1=-2, dim2=-1).abs()
        self.raw_diag = (
            stable_inverse_softplus_torch((seed_diag - self.floor).clamp_min(1e-12))
            .detach()
            .clone()
            .requires_grad_(True)
        )
        self.strict_lower = (
            torch.tril(sL0, diagonal=-1).detach().clone().requires_grad_(True)
        )
        self.raw_a = (
            stable_inverse_softplus_torch(sa0).detach().clone().requires_grad_(True)
        )
        self.delta = torch.zeros(M, D, dtype=f32, device=device, requires_grad=True)
        self.params = [self.delta, self.raw_diag, self.strict_lower, self.raw_a]
        self.opt = torch.optim.Adam(
            [
                {
                    "params": [self.delta],
                    "lr": config.lr_center_scale * max(sigma_c, 1e-12),
                },
                {"params": [self.raw_diag, self.strict_lower], "lr": config.lr_shape},
                {"params": [self.raw_a], "lr": config.lr_shape},
            ]
        )
        self.best: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.best_E = float("inf")
        self.trusted_E_seed: Optional[float] = None
        self.stall = 0

    def realize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Current (mu, L, a), with amplitudes renormalized onto the constant-
        total-mass manifold (mass == the FINE mixture's — pins the additive-
        render DC and closes the mass-inflation exploit direction)."""
        mu = self.smu0 + self.delta * self.center_free
        diag = (
            torch.nn.functional.softplus(self.raw_diag) + self.floor
        ) * self.diag_free
        L = (
            self.L_frozen
            + self.strict_lower * self.strict_free
            + torch.diag_embed(diag)
        )
        a_hat = torch.nn.functional.softplus(self.raw_a)
        mass = (a_hat * sqrt_det_from_cholesky(L)).sum().clamp_min(1e-30)
        a = a_hat * (self.mass_target / mass)
        return mu, L, a


def l2_refine_mixture(
    fine_mu: torch.Tensor,
    fine_L: torch.Tensor,
    fine_a: torch.Tensor,
    seed_mu: torch.Tensor,
    seed_L: torch.Tensor,
    seed_a: torch.Tensor,
    *,
    config: L2RefineConfig,
    frozen_dims: tuple[int, ...] = (),
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Refine a coarse Gaussian mixture against a fine one under closed-form L².

    Parameters
    ----------
    fine_mu, fine_L, fine_a
        The target mixture ``f``: centers ``(N, D)``, lower-triangular Cholesky
        factors ``(N, D, D)``, amplitudes ``(N,)``. Constants (no gradients).
    seed_mu, seed_L, seed_a
        The coarse mixture ``g`` to refine (the coverage-inflated merge output,
        post-cull, all amplitudes strictly positive): shapes ``(M, ...)``.
    config
        See :class:`L2RefineConfig`.
    frozen_dims
        Center-column indices that must not change (barrier dims of a
        ``coarsen_dims`` group): the corresponding center coordinates and every
        Cholesky entry in those rows/columns stay at the seed values, so the
        barrier block of ``Σ`` is exactly preserved (no sliced-dim bleed).
        Lower-triangular with positive diagonal ⇒ PD for arbitrary frozen sets.
    generator
        Optional CPU :class:`torch.Generator` seeding the minibatch pair
        sampling (local determinism; global torch RNG untouched).

    Returns
    -------
    (mu, L, a, stats)
        Refined tensors in float32 on the inputs' device — the best iterate
        over trusted checkpoints, the raw seed included (never worse than the
        seed in the trusted metric). ``stats`` is a dict of plain Python
        scalars (``trusted_E_seed``, ``trusted_E_best``, ``improvement_frac``,
        ``iters_run``, ``rebuilds``, ``nan_grad_skips``, ``minibatched``,
        ``seed_won``, ``mass_vs_seed``, ``mass_vs_fine``,
        ``mass_drift_warning``, ``n_cross_pairs``, ``n_cc_pairs``, ``wall_s``).
    """
    t0 = time.perf_counter()
    device = seed_mu.device
    N = fine_mu.shape[0]
    M = seed_mu.shape[0]
    if M == 0 or N == 0 or config.iters <= 0:
        return (
            seed_mu.to(torch.float32),
            seed_L.to(torch.float32),
            seed_a.to(torch.float32),
            {"iters_run": 0, "rebuilds": 0, "minibatched": False},
        )
    if generator is None:
        generator = torch.Generator()
        generator.seed()

    # ── Standardize: shift by the combined bbox min, scale by ONE scalar (the
    # max extent) — isotropic, so L/scale stays lower-triangular with the same
    # shape. Amplitudes untouched (objective scale-equivariant).
    f32 = torch.float32
    with torch.no_grad():
        lo = torch.minimum(fine_mu.amin(0), seed_mu.amin(0)).to(f32)
        hi = torch.maximum(fine_mu.amax(0), seed_mu.amax(0)).to(f32)
        scale = float((hi - lo).max().clamp_min(1e-12))
    fmu = ((fine_mu.to(f32) - lo) / scale).contiguous()
    fL = (fine_L.to(f32) / scale).contiguous()
    fa = fine_a.to(f32).contiguous()
    smu0 = ((seed_mu.to(f32) - lo) / scale).contiguous()
    sL0 = (seed_L.to(f32) / scale).contiguous()
    sa0 = seed_a.to(f32).clamp_min(1e-12).contiguous()

    sigma_c = float(torch.diagonal(sL0, dim1=-2, dim2=-1).abs().median())
    sigma_f = float(torch.diagonal(fL, dim1=-2, dim2=-1).abs().median())
    mass_target = _total_mass(fL, fa)
    state = _RefineState(smu0, sL0, sa0, frozen_dims, config, sigma_c, mass_target)
    fmu_np = fmu.detach().cpu().numpy()

    stats: dict = {
        "nan_grad_skips": 0,
        "rebuilds": 0,
        "minibatched": False,
        "n_cross_pairs": 0,
        "n_cc_pairs": 0,
    }

    def trusted_eval(
        mu: torch.Tensor,
        L: torch.Tensor,
        a: torch.Tensor,
        pairs: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> tuple[float, tuple]:
        """Full trusted objective on FRESH pair lists for this geometry.

        ``pairs`` may supply pair lists already built for this exact geometry
        (only the it==0 seed reuse below qualifies — see there); otherwise the
        lists are rebuilt from the CURRENT coarse centers, preserving the
        trusted-checkpoint discipline.
        """
        with torch.no_grad():
            if pairs is None:
                pairs = _build_pair_lists(
                    fmu_np,
                    mu.detach().cpu().numpy(),
                    config=config,
                    sigma_c=float(torch.diagonal(L, dim1=-2, dim2=-1).abs().median()),
                    sigma_f=sigma_f,
                    device=device,
                )
                stats["rebuilds"] += 1
            ci, cb, bi, bj = pairs
            # Both the cross (fg) and coarse-coarse (gg) sums are chunked here,
            # so the trusted eval's working set is bounded regardless of M / N.
            fg = _pair_K_sum_chunked(fmu, fL, fa, mu, L, a, ci, cb)
            gg = _g_self_energy(L, a) + 2.0 * _pair_K_sum_chunked(
                mu, L, a, mu, L, a, bi, bj
            )
        return float(gg - 2.0 * fg), pairs

    def consider(E: float, mu: torch.Tensor, L: torch.Tensor, a: torch.Tensor) -> bool:
        """Track the best trusted iterate; returns True when patience runs out."""
        if state.best is None or E < state.best_E - abs(state.best_E) * 1e-7:
            state.best_E = E
            state.best = (mu.detach().clone(), L.detach().clone(), a.detach().clone())
            state.stall = 0
            return False
        state.stall += 1
        return state.stall >= config.early_stop_patience

    # ── The raw (un-normalized) seed is the first trusted candidate: the
    # returned best can therefore never be worse than the seed. Its pair
    # lists are kept for the it==0 window below.
    E_seed, seed_pairs = trusted_eval(smu0, sL0, sa0)
    state.trusted_E_seed = E_seed
    consider(E_seed, smu0, sL0, sa0)

    # ── Trusted-checkpoint Adam loop. Pair minibatches are (re)built at every
    # rebuild point; the first iteration always rebuilds (it == 0).
    empty_idx = torch.empty(0, dtype=torch.int64, device=device)
    cross_mb = _PairMinibatch(empty_idx, empty_idx, config.step_pair_budget, device)
    cc_mb = _PairMinibatch(empty_idx, empty_idx, config.step_pair_budget, device)
    stopped_early = False
    iters_run = 0

    for it in range(config.iters):
        if it % config.rebuild_every == 0:
            with torch.no_grad():
                mu, L, a = state.realize()
            # it == 0: the geometry is still the seed's — delta is zero so the
            # centers are bit-identical to smu0, and realize() only projects
            # the amplitudes onto the mass manifold (plus a ~1-ulp softplus
            # round-trip of L). Pair lists depend on centers and radii only,
            # so the seed evaluation's lists are already fresh for this window;
            # reuse them instead of a byte-identical rebuild (startup was
            # otherwise paying two identical builds + gathers).
            E_tr, pairs = trusted_eval(mu, L, a, pairs=seed_pairs if it == 0 else None)
            if consider(E_tr, mu, L, a) and it > 0:
                stopped_early = True
                break
            ci, cb, bi, bj = pairs
            cross_mb = _PairMinibatch(ci, cb, config.step_pair_budget, device)
            cc_mb = _PairMinibatch(bi, bj, config.step_pair_budget, device)
            stats["n_cross_pairs"] = cross_mb.n
            stats["n_cc_pairs"] = cc_mb.n
            stats["minibatched"] = cross_mb.minibatched or cc_mb.minibatched
        mu, L, a = state.realize()
        ii, jj, fg_scale = cross_mb.sample(generator)
        if ii.numel():
            fg = _pair_K(fmu[ii], fL[ii], fa[ii], mu[jj], L[jj], a[jj]).sum()
        else:
            fg = torch.zeros((), dtype=f32, device=device)
        # gg's coarse-coarse cross term is minibatched too (same budget/scale as
        # fg) so a large first-level reduction does not OOM on the grad path.
        bii, bjj, cc_scale = cc_mb.sample(generator)
        gg = _g_norm_sq(mu, L, a, bii, bjj, cc_scale=cc_scale)
        E = gg - 2.0 * fg_scale * fg
        state.opt.zero_grad()
        E.backward()  # type: ignore[no-untyped-call]
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in state.params
        ):
            # Defensive second layer behind PD-by-construction: skip the step.
            state.opt.zero_grad()
            stats["nan_grad_skips"] += 1
            continue
        state.opt.step()
        iters_run += 1

    if not stopped_early:
        # Loop exhausted without early stop: let the last window's progress win.
        with torch.no_grad():
            mu, L, a = state.realize()
        E_tr, _ = trusted_eval(mu, L, a)
        consider(E_tr, mu, L, a)

    if state.best is None or state.trusted_E_seed is None:  # pragma: no cover
        raise RuntimeError("l2_refine_mixture: no trusted checkpoint was evaluated")
    mu_b, L_b, a_b = state.best
    # Two mass ratios (all standardized; scale cancels):
    #   mass_vs_seed — how much the winning iterate changed total mass vs the
    #                  merge seed (informational).
    #   mass_vs_fine — the WINNING iterate's total mass over the fine target.
    #                  Optimized iterates are pinned to the fine mass in
    #                  realize(), so this is exactly 1 whenever the refit beats
    #                  the merge (the normal case); it departs from 1 only when
    #                  the raw seed wins (rare degenerate cases) and that seed's
    #                  mass was itself off — the meaningful DC-correctness check
    #                  (the old pair-list mass-inflation exploit is structurally
    #                  impossible now that amplitudes live on the mass manifold).
    mass_best = _total_mass(L_b, a_b)
    mass_vs_seed = mass_best / max(_total_mass(sL0, sa0), 1e-30)
    mass_vs_fine = mass_best / max(mass_target, 1e-30)
    stats.update(
        {
            "trusted_E_seed": state.trusted_E_seed,
            "trusted_E_best": state.best_E,
            "improvement_frac": (state.trusted_E_seed - state.best_E)
            / max(abs(state.trusted_E_seed), 1e-30),
            "iters_run": iters_run,
            "seed_won": mu_b.shape == smu0.shape and bool(torch.equal(a_b, sa0)),
            "mass_vs_seed": mass_vs_seed,
            "mass_vs_fine": mass_vs_fine,
            "mass_drift_warning": bool(
                abs(mass_vs_fine - 1.0) > config.mass_drift_warn
            ),
            "wall_s": time.perf_counter() - t0,
        }
    )
    # Un-standardize.
    out_mu = (mu_b * scale + lo).to(torch.float32)
    out_L = (L_b * scale).to(torch.float32)
    out_a = a_b.to(torch.float32)
    return out_mu, out_L, out_a, stats
