"""Vectorised cost-increment Lloyd refinement for substitutive LOD.

After the Morton warm start (:mod:`.warm_start`), the ``*_lloyd`` methods
of :func:`luxar.gsplats.lod.substitutive.make_substitutive_lod` refine the
partition with a monotone, cost-aware Lloyd pass. Every per-bin quantity
is a segment reduction (``torch.Tensor.index_add_``) keyed by the bin
assignment — no Python per-splat / per-bin loops. The per-bin merge math
(moment matching, L²-optimal amplitude, residual energy) lives in
:mod:`luxar.gsplats.lod._kernels` and is shared with the additive axis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from luxar.gsplats.lod._kernels import (
    gaussian_pair_inner_product_torch,
    l2_optimal_amplitude_torch,
    sqrt_det_from_cholesky,
    template_squared_norm_torch,
)
from luxar.gsplats.lod._substitutive import _TINY
from luxar.gsplats.lod._substitutive.warm_start import _morton_order


def _segment_templates(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    assignments: torch.Tensor,
    M: int,
    *,
    Sigma: Optional[torch.Tensor] = None,
    sqrt_det: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Per-bin moment-matched template + L²-amplitude ingredients (vectorised).

    Every quantity is a segment reduction keyed by ``assignments`` (via
    :meth:`torch.Tensor.index_add_`) — no Python loop over bins. Mirrors the
    closed forms in :func:`kwise_moment_match_torch` /
    :func:`bin_inner_product_with_template_torch` exactly, but batched over
    all ``M`` bins at once.

    Returns ``(mu_bar (M,D), Sigma_bar (M,D,D), inter (M,D,D), inner (M,),
    norm_sq (M,), weights (N,))`` where ``Sigma_bar = intra + inter`` is the
    moment-matched template covariance and ``inter`` its inter-center spread
    term alone (Σ_i w_i δδᵀ — needed by the coverage-inflation correction in
    :func:`_build_representatives_vectorized`), ``inner`` = ⟨f_Sb, Ḡb⟩ and
    ``norm_sq`` = ‖Ḡb‖². The per-bin L²-optimal amplitude is
    ``inner / norm_sq`` and the projection energy is ``inner² / norm_sq``.
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

    # Intra-bin spread Σ_i w_i Σ_i  +  inter-bin spread Σ_i w_i δδᵀ
    # (accumulated separately so the inter term is available on its own).
    intra = torch.zeros(M, D, D, dtype=dt, device=device)
    intra.index_add_(0, assignments, w.view(N, 1, 1) * Sigma)
    delta = centres - mu_bar[assignments]  # (N, D)
    outer = w.view(N, 1, 1) * (delta.unsqueeze(2) * delta.unsqueeze(1))
    inter = torch.zeros(M, D, D, dtype=dt, device=device)
    inter.index_add_(0, assignments, outer)
    Sigma_bar = intra + inter

    # ⟨f_Sb, Ḡb⟩ = Σ_i K(splat_i, template_{bin_i}); ‖Ḡb‖² = π^{D/2}|Σ̄_b|^{1/2}.
    ones = torch.ones(N, dtype=dt, device=device)
    K_i = gaussian_pair_inner_product_torch(
        centres, Sigma, amps, mu_bar[assignments], Sigma_bar[assignments], ones
    )  # (N,)
    inner = torch.zeros(M, dtype=dt, device=device)
    inner.index_add_(0, assignments, K_i)
    norm_sq = template_squared_norm_torch(Sigma_bar)  # (M,)
    return mu_bar, Sigma_bar, inter, inner, norm_sq, w


def _build_representatives_vectorized(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    colors: Optional[torch.Tensor],
    assignments: torch.Tensor,
    *,
    M: int,
    coverage_inflation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Vectorised replacement for the per-bin :func:`_build_representatives`.

    ``coverage_inflation`` (β >= 1) widens each representative's *inter-center*
    spread term: ``Σ_out = intra + β·inter``, with a mass-preserving amplitude
    rescale (``a·|Σ|^{1/2}`` kept constant, so the splat's integral — hence the
    additive-blend X-ray projection — is unchanged). Rationale: the balanced
    warm-start bins tile space with pitch ``d``, and pure moment matching gives
    them σ ≈ d/√12 ≈ 0.29 d — far below the σ ≳ d/2 a lattice of Gaussians
    needs to sum flat, so the un-inflated levels render with a strong periodic
    (grid) intensity ripple along the shared Morton-cell boundaries. β·inter
    fixes exactly that term: β=3 turns d²/12 into (d/2)², and the recurrence
    across levels (σ_ℓ² = σ_{ℓ-1}² + β(d_ℓ² − d_{ℓ-1}²)/12) has σ = √(β/12)·d
    as its exact fixed point, so the calibration holds at every level with no
    compounding. Single-member bins have ``inter = 0`` and are untouched.
    """
    N, D = centres.shape
    device = centres.device
    dt = centres.dtype
    Sigma = L @ L.transpose(-1, -2)
    sqrt_det = sqrt_det_from_cholesky(L)
    mu_bar, Sigma_bar, inter, inner, norm_sq, w = _segment_templates(
        centres, L, amps, assignments, M, Sigma=Sigma, sqrt_det=sqrt_det
    )
    a_star = l2_optimal_amplitude_torch(inner, norm_sq).clamp_min(0.0)  # (M,)

    if coverage_inflation != 1.0:
        Sigma_infl = Sigma_bar + (coverage_inflation - 1.0) * inter
        norm_infl = template_squared_norm_torch(Sigma_infl)  # π^{D/2}|Σ_out|^{1/2}
        # Mass preservation: a_out·|Σ_out|^{1/2} = a*·|Σ̄|^{1/2}, i.e. scale by
        # norm_sq/norm_infl (the π^{D/2} factors cancel).
        a_star = torch.where(
            norm_infl > 0,
            a_star * norm_sq / norm_infl,
            torch.zeros_like(a_star),
        )
        Sigma_bar = Sigma_infl

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
        mu_bar, Sigma_bar, _inter, inner, norm_sq, _ = _segment_templates(
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
        # Accept only a strict improvement (monotone refinement). The relative
        # 1e-9 margin is calibrated for the deterministic float64 CPU path
        # (the supported/verified path — see make_substitutive_lod, which
        # forces CPU+float64). NOTE: ``projection_energy`` is built from
        # ``index_add_`` segment reductions, which PyTorch documents as
        # NON-deterministic on CUDA (atomic adds). On a CUDA device a genuine
        # improvement smaller than the atomic-add noise floor could therefore
        # be rejected here, stopping iteration early; the only consequence is a
        # marginally coarser refinement at the noise floor (never a wrong or
        # NaN result). For bit-exact CUDA behaviour, run with
        # ``torch.use_deterministic_algorithms(True)`` set by the caller.
        if P_new > best_P + abs(best_P) * 1e-9:
            best_P = P_new
            assignments = new_assign
            mu_bar, Sigma_bar = mu_bar_new, Sigma_bar_new
        else:
            break
    return assignments
