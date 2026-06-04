"""Closed-form Gaussian-mixture kernels shared by additive + substitutive LOD.

The two LOD axes share a small set of primitives derived from the
Gaussian inner product

.. math::

    K_{ij} = a_i a_j (2\\pi)^{D/2}\\,
        \\sqrt{\\frac{|\\Sigma_i||\\Sigma_j|}{|\\Sigma_i + \\Sigma_j|}}\\,
        \\exp\\!\\Bigl(-\\tfrac{1}{2}(\\bm{\\mu}_i-\\bm{\\mu}_j)^\\top
        (\\Sigma_i+\\Sigma_j)^{-1}(\\bm{\\mu}_i-\\bm{\\mu}_j)\\Bigr)

(supp doc :file:`additive_lod.tex` Eq. (2.1) and
:file:`substitutive_lod.tex` Eq. (2.1)).

This module exposes both NumPy and PyTorch variants. The NumPy variants
support the existing additive sparse-Gram path (inline closed form
moved out of :mod:`luxar.gsplats.lod.additive`). The PyTorch variants
are vectorised over a bin's splats and run on CUDA / MPS / CPU; they
are the building blocks of :mod:`luxar.gsplats.lod.substitutive`.

Functions
---------
gaussian_pair_inner_product_numpy
    Single-pair $K_{ij}$ in NumPy.
gaussian_self_energy_numpy
    Per-splat $K_{ii} = a_i^2 \\pi^{D/2} |\\Sigma_i|^{1/2}$ (vectorised).
truncation_radii_numpy
    Per-splat $r_i = \\sigma\\sqrt{\\lambda_{\\max}(\\Sigma_i)}$.
kwise_moment_match_torch
    Mass-weighted moment-matched single Gaussian for one bin
    (supp doc Prop. 2.1).
l2_optimal_amplitude_torch
    L²-optimal bin amplitude (supp doc Prop. 2.2 / Eq. (2.4)).
bin_squared_norm_torch
    $\\|f_{\\mathcal{S}_j}\\|_{L^2}^2 = \\sum_{i,k} K_{ik}$ over the bin.
bin_residual_energy_torch
    Closed-form per-bin residual $E_j^\\star$ (supp doc Eq. (2.5)).
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────
# NumPy primitives (used by additive sparse Gram)
# ─────────────────────────────────────────────────────────────────────


def gaussian_pair_inner_product_numpy(
    mu_i: np.ndarray,
    Sigma_i: np.ndarray,
    a_i: float,
    mu_j: np.ndarray,
    Sigma_j: np.ndarray,
    a_j: float,
    *,
    sqrt_det_Sigma_i: float | None = None,
    sqrt_det_Sigma_j: float | None = None,
) -> float:
    """Closed-form $K_{ij}$ for a single pair of Gaussians (NumPy).

    ``Sigma_i`` and ``Sigma_j`` are the full $D\\times D$ covariance
    matrices. If ``sqrt_det_Sigma_*`` are precomputed (e.g. via the
    Cholesky diagonal product) they are reused; otherwise computed
    via :func:`numpy.linalg.slogdet` on the full covariance.

    Returns 0.0 when the covariance sum $S = \\Sigma_i + \\Sigma_j$ is
    not positive-definite (numerically degenerate); the caller can
    treat such pairs as structurally zero.
    """
    D = mu_i.shape[0]
    diff = mu_i - mu_j
    S = Sigma_i + Sigma_j
    try:
        sign_S, logdet_S = np.linalg.slogdet(S)
        if sign_S <= 0:
            return 0.0
        sol = np.linalg.solve(S, diff)
    except np.linalg.LinAlgError:
        return 0.0
    quad = float(diff @ sol)
    if sqrt_det_Sigma_i is None:
        sign_i, logdet_i = np.linalg.slogdet(Sigma_i)
        if sign_i <= 0:
            return 0.0
        sqrt_det_Sigma_i = float(np.exp(0.5 * logdet_i))
    if sqrt_det_Sigma_j is None:
        sign_j, logdet_j = np.linalg.slogdet(Sigma_j)
        if sign_j <= 0:
            return 0.0
        sqrt_det_Sigma_j = float(np.exp(0.5 * logdet_j))
    inv_sqrt_det_S = float(np.exp(-0.5 * logdet_S))
    two_pi_half_D = (2.0 * np.pi) ** (D / 2.0)
    K_ij = float(
        a_i
        * a_j
        * two_pi_half_D
        * sqrt_det_Sigma_i
        * sqrt_det_Sigma_j
        * inv_sqrt_det_S
        * np.exp(-0.5 * quad)
    )
    if not np.isfinite(K_ij) or K_ij <= 0.0:
        return 0.0
    return K_ij


def gaussian_self_energy_numpy(
    amps: np.ndarray,
    sqrt_det_Sigma: np.ndarray,
    ndim: int,
) -> np.ndarray:
    """Per-splat self-energy $K_{ii} = a_i^2 \\pi^{D/2} |\\Sigma_i|^{1/2}$.

    Vectorised over an array of splats.
    """
    pi_half_D = math.pi ** (ndim / 2.0)
    out: np.ndarray = (
        np.asarray(amps, dtype=np.float64) ** 2
        * pi_half_D
        * np.asarray(sqrt_det_Sigma, dtype=np.float64)
    )
    return out


def truncation_radii_numpy(L: np.ndarray, sigmas: float) -> np.ndarray:
    """Per-splat $r_i = \\sigma\\sqrt{\\lambda_{\\max}(\\Sigma_i)}$.

    ``L`` has shape ``(N, D, D)`` (lower-triangular Cholesky). Returns
    a length-N float64 array.
    """
    if L.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    Sigma = L @ L.transpose(0, 2, 1)
    eigs = np.linalg.eigvalsh(Sigma)  # ascending eigenvalues, shape (N, D)
    lam_max = np.maximum(eigs[:, -1], 0.0)
    out: np.ndarray = float(sigmas) * np.sqrt(lam_max)
    return out


# ─────────────────────────────────────────────────────────────────────
# PyTorch primitives (used by substitutive bin work)
# ─────────────────────────────────────────────────────────────────────


def sqrt_det_from_cholesky(L: torch.Tensor) -> torch.Tensor:
    """Public alias of :func:`_sqrt_det_from_cholesky` for cross-module reuse."""
    return _sqrt_det_from_cholesky(L)


def _sqrt_det_from_cholesky(L: torch.Tensor) -> torch.Tensor:
    """Per-splat $|\\Sigma|^{1/2} = |\\det L|$ from lower-triangular ``L``.

    Accepts ``L`` of shape ``(..., D, D)``; returns shape ``(...)``.
    """
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    return torch.abs(torch.prod(diag, dim=-1))


def kwise_moment_match_torch(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mass-weighted moment match of a bin (supp doc Prop. 2.1).

    Parameters
    ----------
    centres
        Bin members' centres, shape ``(K, D)``.
    L
        Bin members' lower-triangular Cholesky factors, shape ``(K, D, D)``.
    amps
        Bin members' amplitudes, shape ``(K,)``.

    Returns
    -------
    (mu_bar, Sigma_bar, weights)
        - ``mu_bar``: shape ``(D,)``, mass-weighted mean.
        - ``Sigma_bar``: shape ``(D, D)``, intra + inter spread.
        - ``weights``: shape ``(K,)``, mass-weighted bin weights
          (returned for downstream use to avoid recomputation).

    Empty bin (K=0) returns ``(zeros(D), I_D, zeros(0))``.
    """
    K = centres.shape[0]
    D = centres.shape[1]
    if K == 0:
        return (
            torch.zeros(D, dtype=centres.dtype, device=centres.device),
            torch.eye(D, dtype=centres.dtype, device=centres.device),
            torch.zeros(0, dtype=centres.dtype, device=centres.device),
        )
    # Mass weights: m_i = a_i * (2π)^(D/2) * |Σ_i|^{1/2}.
    # Constants drop out of the normalised weights; use the proportional form.
    sqrt_det_Sigma = _sqrt_det_from_cholesky(L)  # (K,)
    masses = amps * sqrt_det_Sigma
    total = masses.sum()
    if total <= 0:
        # Pathological bin (all-zero amplitudes / det). Fall back to
        # uniform weights so downstream code doesn't divide by zero.
        weights = torch.full((K,), 1.0 / K, dtype=centres.dtype, device=centres.device)
    else:
        weights = masses / total

    mu_bar = (weights.unsqueeze(-1) * centres).sum(dim=0)  # (D,)

    # Intra-bin spread: Σ_i w_i Σ_i.
    Sigma = L @ L.transpose(-1, -2)  # (K, D, D)
    intra = (weights.view(K, 1, 1) * Sigma).sum(dim=0)  # (D, D)

    # Inter-bin spread: Σ_i w_i (μ_i - μ̄)(μ_i - μ̄)^T.
    delta = centres - mu_bar.unsqueeze(0)  # (K, D)
    inter = (weights.view(K, 1, 1) * (delta.unsqueeze(2) @ delta.unsqueeze(1))).sum(
        dim=0
    )  # (D, D)

    Sigma_bar = intra + inter
    return mu_bar, Sigma_bar, weights


def gaussian_pair_inner_product_torch(
    mu_i: torch.Tensor,
    Sigma_i: torch.Tensor,
    a_i: torch.Tensor,
    mu_j: torch.Tensor,
    Sigma_j: torch.Tensor,
    a_j: torch.Tensor,
) -> torch.Tensor:
    """Vectorised closed-form $K_{ij}$ between aligned splat pairs.

    Each input may be batched on a leading axis; broadcasting follows
    standard PyTorch rules.

    - ``mu_i, mu_j``: ``(..., D)``.
    - ``Sigma_i, Sigma_j``: ``(..., D, D)`` (full covariances, not Cholesky).
    - ``a_i, a_j``: ``(...)``.

    Returns ``(...)``, the inner product per pair. Numerically
    degenerate covariance sums (non-PD) yield 0.
    """
    D = mu_i.shape[-1]
    diff = mu_i - mu_j  # (..., D)
    S = Sigma_i + Sigma_j  # (..., D, D)
    try:
        # cholesky_ex returns info != 0 for non-PD; we handle that.
        L_S, info = torch.linalg.cholesky_ex(S, check_errors=False)
    except RuntimeError:
        return torch.zeros_like(a_i * a_j)
    valid = info == 0  # (...,) — True where S is PD.
    # Solve S x = diff via the Cholesky factor; mask invalid entries.
    sol = torch.linalg.solve_triangular(L_S, diff.unsqueeze(-1), upper=False)
    sol = torch.linalg.solve_triangular(L_S.transpose(-1, -2), sol, upper=True).squeeze(
        -1
    )
    quad = (diff * sol).sum(dim=-1)  # (...,)
    # |Σ_i|^{1/2}, |Σ_j|^{1/2}, |S|^{1/2} via Cholesky.
    L_i, info_i = torch.linalg.cholesky_ex(Sigma_i, check_errors=False)
    L_j, info_j = torch.linalg.cholesky_ex(Sigma_j, check_errors=False)
    sqrt_det_i = torch.abs(torch.prod(torch.diagonal(L_i, dim1=-2, dim2=-1), dim=-1))
    sqrt_det_j = torch.abs(torch.prod(torch.diagonal(L_j, dim1=-2, dim2=-1), dim=-1))
    sqrt_det_S = torch.abs(torch.prod(torch.diagonal(L_S, dim1=-2, dim2=-1), dim=-1))
    two_pi_half_D = (2.0 * math.pi) ** (D / 2.0)
    K = (
        a_i
        * a_j
        * two_pi_half_D
        * sqrt_det_i
        * sqrt_det_j
        / sqrt_det_S
        * torch.exp(-0.5 * quad)
    )
    K = torch.where(
        valid & (info_i == 0) & (info_j == 0),
        K,
        torch.zeros_like(K),
    )
    return K


def gaussian_self_energy_torch(
    amps: torch.Tensor,
    sqrt_det_Sigma: torch.Tensor,
    ndim: int,
) -> torch.Tensor:
    """Vectorised $K_{ii} = a_i^2 \\pi^{D/2} |\\Sigma_i|^{1/2}$ in PyTorch."""
    pi_half_D = math.pi ** (ndim / 2.0)
    out: torch.Tensor = amps * amps * pi_half_D * sqrt_det_Sigma
    return out


def bin_squared_norm_torch(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
) -> torch.Tensor:
    """$\\|f_{\\mathcal{S}}\\|_{L^2}^2 = \\sum_{i,k\\in\\mathcal{S}} K_{ik}$.

    Computes the dense $K\\times K$ Gram block for one bin and sums.
    Used to evaluate per-bin residual energies in substitutive Lloyd.
    """
    K = centres.shape[0]
    D = centres.shape[1]
    if K == 0:
        return torch.zeros((), dtype=centres.dtype, device=centres.device)
    Sigma = L @ L.transpose(-1, -2)  # (K, D, D)
    sqrt_det = _sqrt_det_from_cholesky(L)  # (K,)
    pi_half_D = math.pi ** (D / 2.0)
    two_pi_half_D = (2.0 * math.pi) ** (D / 2.0)

    # Diagonal contribution: Σ_i a_i^2 π^(D/2) |Σ_i|^{1/2}.
    diag_sum: torch.Tensor = (amps * amps * pi_half_D * sqrt_det).sum()

    if K == 1:
        return diag_sum

    # Off-diagonal contributions: 2 * Σ_{i<k} K_{ik}.
    # Build pairwise differences and covariance sums.
    mu_diff = centres.unsqueeze(0) - centres.unsqueeze(1)  # (K, K, D)
    S = Sigma.unsqueeze(0) + Sigma.unsqueeze(1)  # (K, K, D, D)
    L_S, info_S = torch.linalg.cholesky_ex(S, check_errors=False)
    valid = info_S == 0
    sol = torch.linalg.solve_triangular(L_S, mu_diff.unsqueeze(-1), upper=False)
    sol = torch.linalg.solve_triangular(L_S.transpose(-1, -2), sol, upper=True).squeeze(
        -1
    )
    quad = (mu_diff * sol).sum(dim=-1)  # (K, K)
    sqrt_det_S = torch.abs(
        torch.prod(torch.diagonal(L_S, dim1=-2, dim2=-1), dim=-1)
    )  # (K, K)
    sqrt_det_outer = sqrt_det.unsqueeze(0) * sqrt_det.unsqueeze(1)  # (K, K)
    amp_outer = amps.unsqueeze(0) * amps.unsqueeze(1)  # (K, K)
    K_ij = (
        amp_outer * two_pi_half_D * sqrt_det_outer / sqrt_det_S * torch.exp(-0.5 * quad)
    )
    K_ij = torch.where(valid, K_ij, torch.zeros_like(K_ij))
    # Off-diagonal sum = total - diagonal = sum of K_ij - sum_i K_ii.
    diag = torch.diagonal(K_ij)
    off_diag_sum = K_ij.sum() - diag.sum()
    out: torch.Tensor = diag_sum + off_diag_sum
    return out


def bin_inner_product_with_template_torch(
    centres: torch.Tensor,
    L: torch.Tensor,
    amps: torch.Tensor,
    mu_bar: torch.Tensor,
    Sigma_bar: torch.Tensor,
) -> torch.Tensor:
    """$\\langle f_{\\mathcal{S}}, \\bar G\\rangle$ for one bin against unit template.

    The template $\\bar G$ has unit amplitude, mean ``mu_bar``,
    covariance ``Sigma_bar``. Returns a scalar.
    """
    K = centres.shape[0]
    if K == 0:
        return torch.zeros((), dtype=centres.dtype, device=centres.device)
    Sigma = L @ L.transpose(-1, -2)  # (K, D, D)
    Sigma_bar_b = Sigma_bar.unsqueeze(0).expand(K, -1, -1)  # (K, D, D)
    mu_bar_b = mu_bar.unsqueeze(0).expand(K, -1)  # (K, D)
    a_template = torch.ones(K, dtype=centres.dtype, device=centres.device)
    # inner product per bin member with the template
    K_i_template = gaussian_pair_inner_product_torch(
        centres, Sigma, amps, mu_bar_b, Sigma_bar_b, a_template
    )  # (K,)
    return K_i_template.sum()


def template_squared_norm_torch(
    Sigma_bar: torch.Tensor,
) -> torch.Tensor:
    """$\\|\\bar G\\|_{L^2}^2 = \\pi^{D/2} |\\bar\\Sigma|^{1/2}$ for the unit-amplitude template."""
    D = Sigma_bar.shape[-1]
    sign, logabsdet = torch.linalg.slogdet(Sigma_bar)
    sqrt_det = torch.where(
        sign > 0,
        torch.exp(0.5 * logabsdet),
        torch.zeros_like(logabsdet),
    )
    out: torch.Tensor = math.pi ** (D / 2.0) * sqrt_det
    return out


def l2_optimal_amplitude_torch(
    bin_template_inner: torch.Tensor,
    template_norm_sq: torch.Tensor,
) -> torch.Tensor:
    """L²-optimal bin amplitude (supp doc Prop. 2.2 / Eq. (2.4)).

    $\\bar a^\\star = \\langle f_{\\mathcal{S}}, \\bar G\\rangle / \\|\\bar G\\|_{L^2}^2$.
    Returns 0 when the template norm is degenerate.
    """
    safe_norm = torch.where(
        template_norm_sq > 0,
        template_norm_sq,
        torch.ones_like(template_norm_sq),
    )
    a_star = bin_template_inner / safe_norm
    return torch.where(
        template_norm_sq > 0,
        a_star,
        torch.zeros_like(a_star),
    )


def bin_residual_energy_torch(
    bin_norm_sq: torch.Tensor,
    bin_template_inner: torch.Tensor,
    template_norm_sq: torch.Tensor,
) -> torch.Tensor:
    """Closed-form per-bin residual $E_j^\\star$ (supp doc Eq. (2.5)).

    $E_j^\\star = \\|f_{\\mathcal{S}_j}\\|^2 - |\\langle f_{\\mathcal{S}_j}, \\bar G_j\\rangle|^2 / \\|\\bar G_j\\|^2$.
    Numerically clamped to be non-negative (the closed form can dip
    slightly below zero on near-degenerate bins).
    """
    safe_norm = torch.where(
        template_norm_sq > 0,
        template_norm_sq,
        torch.ones_like(template_norm_sq),
    )
    proj = (bin_template_inner * bin_template_inner) / safe_norm
    proj = torch.where(
        template_norm_sq > 0,
        proj,
        torch.zeros_like(proj),
    )
    return torch.clamp(bin_norm_sq - proj, min=0.0)
