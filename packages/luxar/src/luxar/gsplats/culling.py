"""Contribution-based Gaussian splat culling.

This module provides principled removal of splats that contribute negligibly
to the reconstruction, going beyond simple amplitude thresholds by evaluating
each splat's *actual* impact on the rendered volume.

Two modes are available
------------------------

**Error-budget mode** (``target`` provided):
    Uses the original volume that was fitted.  Computes the residual
    ``R = target - V_pred`` and derives an error budget *tau* from the
    existing reconstruction error.  For each splat, measures the maximum
    error *increase* from removal: ``max(0, |R+g_j| - |R|)``.  A splat
    is safe to remove when this increase is below *tau*.  This formulation
    is robust to pre-existing high-error voxels — a splat near a noisy
    region can still be culled if it contributes negligibly.
    This is the most principled mode, but requires the target volume.

**Redundancy mode** (no ``target``):
    Works from the splats alone — no target volume needed.  For each splat,
    measures the maximum *fractional contribution*: ``g_j(x) / V_pred(x)``
    within the splat's support.  If a splat never contributes more than a
    small fraction of the total signal at any point, it is redundant — other
    splats already cover its region.  A ``redundancy_threshold`` (e.g. 0.01)
    means "remove splats that contribute less than 1% of the local signal
    everywhere."

Both modes include a **joint compounding check** that verifies the joint
removal of all candidates does not exceed the budget.  If it does, a binary
search tightens the per-splat threshold until the joint constraint holds.

Why two modes?
--------------
Error-budget mode is strictly more powerful: it accounts for the actual
fitting error and can detect splats in regions where the reconstruction is
already poor (removing them doesn't make things worse).  Redundancy mode
cannot make this distinction because it has no reference to compare against.

However, the target volume is often unavailable — splats may have been
pre-computed, transferred, or the original data discarded.  Redundancy mode
provides a useful fallback that still captures the key idea: spatially
redundant splats can be removed without degrading the reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.models.gsplats.rendering_core import (
    cached_base_and_offsets,
    calculate_optimal_chunk_size,
    compute_aabb_with_intensity_floor,
    fwd_norm2_2d,
    fwd_norm2_3d,
    group_by_box_gpu,
    linear_strides,
    render_gaussians,
)


@dataclass(frozen=True)
class CullResult:
    """Result of contribution-based culling.

    Attributes
    ----------
    keep_mask : np.ndarray, shape (N,)
        Boolean mask — True for splats to keep.
    n_culled : int
        Number of splats removed.
    error_budget : float
        The error budget *tau* used for the final decision.  In error-budget
        mode this is ``percentile(|R|) * tolerance``; in redundancy mode it
        equals the ``redundancy_threshold``.
    phase1_candidates : int
        Number of individually safe candidates before the joint compounding check.
    phase2_iterations : int
        Number of binary-search iterations in the joint compounding check.
    max_joint_error : float
        The max ``|R_joint|`` (error-budget) or max fractional contribution
        (redundancy) after removing the final set of culled splats.
    mode : str
        ``"error_budget"`` or ``"redundancy"`` — which mode was used.
    """

    keep_mask: np.ndarray
    n_culled: int
    error_budget: float
    phase1_candidates: int
    phase2_iterations: int
    max_joint_error: float
    mode: str = "error_budget"


# =====================================================================
# Per-splat deletion error (error-budget mode)
# =====================================================================
#
# For each splat j, within its AABB, compute:
#     max_x |R(x) + g_j(x)|
# where R = V_target - V_pred is the current residual.
# This is the worst-case local error that would result from removing
# splat j.  The rendering is additive, so R_new = R + g_j exactly.
# =====================================================================


def _deletion_error_2d(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    reference_flat: torch.Tensor,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float,
    chunk_size: Optional[int],
    fractional: bool = False,
) -> torch.Tensor:
    """Per-splat error for 2D.  If fractional, computes g_j/V_pred instead."""
    device = centers.device
    N = centers.shape[0]
    max_errors = torch.zeros(N, device=device)
    strides = linear_strides(shape, device)

    # Shifted Gaussian constants (must match render_gaussians)
    shift_C = math.exp(-0.5 * truncate * truncate)
    shift_scale = 1.0 / (1.0 - shift_C)

    lo, hi, valid = compute_aabb_with_intensity_floor(
        centers, Ls, amps, shape, truncate, intensity_floor, device
    )

    if not torch.all(valid):
        valid_idx = valid.nonzero(as_tuple=True)[0]
        centers = centers[valid_idx]
        Ls = Ls[valid_idx]
        amps = amps[valid_idx]
        lo, hi = lo[valid_idx], hi[valid_idx]
    else:
        valid_idx = torch.arange(N, device=device)

    if centers.numel() == 0:
        return max_errors

    uniq, inv = group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]
        L = Ls[idx]
        a = amps[idx]
        lo_sel = lo[idx]

        base, lin_offsets = cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)

        P = base.shape[1]
        K = len(idx)
        P_chunk = chunk_size or calculate_optimal_chunk_size(
            K=K, d=2, device=device, dtype=torch.float32
        )

        group_max = torch.zeros(K, device=device)

        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]

            dist_sq = fwd_norm2_2d(L, d0, d1)
            # Shifted Gaussian: must match render_gaussians formula
            g_vals = (
                a[:, None]
                * shift_scale
                * torch.clamp(torch.exp(-0.5 * dist_sq) - shift_C, min=0.0)
            )

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            ref_vals = reference_flat[idx_flat].reshape(K, -1)

            if fractional:
                chunk_err = (g_vals / torch.clamp(ref_vals, min=1e-12)).amax(dim=1)
            else:
                # Error INCREASE from removing splat j:
                # damage(x) = max(0, |R(x)+g_j(x)| - |R(x)|)
                # This measures how much worse removal makes each voxel,
                # ignoring pre-existing errors that have nothing to do
                # with splat j.  Without this, a splat in a region with
                # ANY high-error voxel (|R|>=tau) can never be culled,
                # even if the splat itself is negligible.
                error_before = torch.abs(ref_vals)
                error_after = torch.abs(ref_vals + g_vals)
                damage = torch.clamp(error_after - error_before, min=0.0)
                chunk_err = damage.amax(dim=1)
            group_max = torch.maximum(group_max, chunk_err)

        max_errors[valid_idx[idx]] = group_max

    return max_errors


def _deletion_error_3d(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    reference_flat: torch.Tensor,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float,
    chunk_size: Optional[int],
    fractional: bool = False,
) -> torch.Tensor:
    """Per-splat error for 3D.  If fractional, computes g_j/V_pred instead."""
    device = centers.device
    N = centers.shape[0]
    max_errors = torch.zeros(N, device=device)
    strides = linear_strides(shape, device)

    # Shifted Gaussian constants (must match render_gaussians)
    shift_C = math.exp(-0.5 * truncate * truncate)
    shift_scale = 1.0 / (1.0 - shift_C)

    lo, hi, valid = compute_aabb_with_intensity_floor(
        centers, Ls, amps, shape, truncate, intensity_floor, device
    )

    if not torch.all(valid):
        valid_idx = valid.nonzero(as_tuple=True)[0]
        centers = centers[valid_idx]
        Ls = Ls[valid_idx]
        amps = amps[valid_idx]
        lo, hi = lo[valid_idx], hi[valid_idx]
    else:
        valid_idx = torch.arange(N, device=device)

    if centers.numel() == 0:
        return max_errors

    uniq, inv = group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]
        L = Ls[idx]
        a = amps[idx]
        lo_sel = lo[idx]

        base, lin_offsets = cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)

        P = base.shape[1]
        K = len(idx)
        P_chunk = chunk_size or calculate_optimal_chunk_size(
            K=K, d=3, device=device, dtype=torch.float32
        )

        group_max = torch.zeros(K, device=device)

        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]
            d2 = base[2, p0:p1][None, :] + lo_sel[:, 2:3] - mu[:, 2:3]

            dist_sq = fwd_norm2_3d(L, d0, d1, d2)
            # Shifted Gaussian: must match render_gaussians formula
            g_vals = (
                a[:, None]
                * shift_scale
                * torch.clamp(torch.exp(-0.5 * dist_sq) - shift_C, min=0.0)
            )

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            ref_vals = reference_flat[idx_flat].reshape(K, -1)

            if fractional:
                chunk_err = (g_vals / torch.clamp(ref_vals, min=1e-12)).amax(dim=1)
            else:
                # Error INCREASE from removing splat j:
                # damage(x) = max(0, |R(x)+g_j(x)| - |R(x)|)
                # This measures how much worse removal makes each voxel,
                # ignoring pre-existing errors that have nothing to do
                # with splat j.  Without this, a splat in a region with
                # ANY high-error voxel (|R|>=tau) can never be culled,
                # even if the splat itself is negligible.
                error_before = torch.abs(ref_vals)
                error_after = torch.abs(ref_vals + g_vals)
                damage = torch.clamp(error_after - error_before, min=0.0)
                chunk_err = damage.amax(dim=1)
            group_max = torch.maximum(group_max, chunk_err)

        max_errors[valid_idx[idx]] = group_max

    return max_errors


def _deletion_error_nd(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    reference_flat: torch.Tensor,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float,
    chunk_size: Optional[int],
    fractional: bool = False,
) -> torch.Tensor:
    """Per-splat error for nD.  If fractional, computes g_j/V_pred instead."""
    device = centers.device
    d = len(shape)
    N = centers.shape[0]
    max_errors = torch.zeros(N, device=device)
    strides = linear_strides(shape, device)

    # Shifted Gaussian constants (must match render_gaussians)
    shift_C = math.exp(-0.5 * truncate * truncate)
    shift_scale = 1.0 / (1.0 - shift_C)

    lo, hi, valid = compute_aabb_with_intensity_floor(
        centers, Ls, amps, shape, truncate, intensity_floor, device
    )

    if not torch.all(valid):
        valid_idx = valid.nonzero(as_tuple=True)[0]
        centers = centers[valid_idx]
        Ls = Ls[valid_idx]
        amps = amps[valid_idx]
        lo, hi = lo[valid_idx], hi[valid_idx]
    else:
        valid_idx = torch.arange(N, device=device)

    if centers.numel() == 0:
        return max_errors

    sizes = (hi - lo).to(torch.int32)
    if sizes.device.type == "mps":
        sizes_cpu = sizes.cpu()
        uniq, inv = torch.unique(sizes_cpu, dim=0, return_inverse=True)
        uniq, inv = uniq.to(device), inv.to(device)
    else:
        uniq, inv = torch.unique(sizes, dim=0, return_inverse=True)

    uniq_cpu = uniq.cpu().tolist()

    for g_idx, box_shape in enumerate(map(tuple, uniq_cpu)):
        idx = torch.nonzero(inv == g_idx, as_tuple=False).squeeze(1)

        mu = centers[idx]
        L = Ls[idx]
        a = amps[idx]

        ranges = [
            torch.arange(s, device=device, dtype=torch.float32) for s in box_shape
        ]
        grids = torch.meshgrid(*ranges, indexing="ij")
        P = int(np.prod(box_shape))
        base = torch.stack([g.reshape(-1) for g in grids], dim=0)

        lo_f = lo[idx].to(torch.float32)
        strides_f = strides.to(torch.float32)
        lin_offsets = (base.T @ strides_f).to(torch.long)
        base_idx = (lo[idx].to(torch.long) * strides).sum(dim=1)

        K = len(idx)
        P_chunk = chunk_size or calculate_optimal_chunk_size(
            K=K, d=d, device=device, dtype=torch.float32
        )

        group_max = torch.zeros(K, device=device)

        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            base_chunk = base[:, p0:p1]
            lin_offsets_chunk = lin_offsets[p0:p1]

            delta = base_chunk[None, :, :] + lo_f[:, :, None] - mu[:, :, None]

            try:
                y = torch.linalg.solve_triangular(L, delta, upper=False)
            except AttributeError:
                y, _ = torch.triangular_solve(delta, L, upper=False)

            dist_sq = torch.sum(y * y, dim=1)
            # Shifted Gaussian: must match render_gaussians formula
            g_vals = (
                a[:, None]
                * shift_scale
                * torch.clamp(torch.exp(-0.5 * dist_sq) - shift_C, min=0.0)
            )

            idx_flat = (base_idx[:, None] + lin_offsets_chunk[None, :]).reshape(-1)
            ref_vals = reference_flat[idx_flat].reshape(K, -1)

            if fractional:
                chunk_err = (g_vals / torch.clamp(ref_vals, min=1e-12)).amax(dim=1)
            else:
                # Error INCREASE from removing splat j:
                # damage(x) = max(0, |R(x)+g_j(x)| - |R(x)|)
                # This measures how much worse removal makes each voxel,
                # ignoring pre-existing errors that have nothing to do
                # with splat j.  Without this, a splat in a region with
                # ANY high-error voxel (|R|>=tau) can never be culled,
                # even if the splat itself is negligible.
                error_before = torch.abs(ref_vals)
                error_after = torch.abs(ref_vals + g_vals)
                damage = torch.clamp(error_after - error_before, min=0.0)
                chunk_err = damage.amax(dim=1)
            group_max = torch.maximum(group_max, chunk_err)

        max_errors[valid_idx[idx]] = group_max

    return max_errors


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _compute_per_splat_error(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    reference: torch.Tensor,
    shape: Sequence[int],
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
    chunk_size: Optional[int] = None,
    fractional: bool = False,
) -> torch.Tensor:
    """Compute per-splat error metric within each splat's AABB.

    Two metrics are supported, selected by *fractional*:

    * ``fractional=False`` (error-budget mode): for each splat j,
      ``max_x max(0, |R(x)+g_j(x)| - |R(x)|)`` — the worst-case error
      *increase* from removing splat j, where *reference* is the residual
      ``R``.  This ignores pre-existing errors and only measures how much
      worse removal makes each voxel.

    * ``fractional=True`` (redundancy mode): for each splat j,
      ``max_x  g_j(x) / V_pred(x)`` — the maximum fraction of the local
      signal contributed by splat j, where *reference* is ``V_pred``.

    Parameters
    ----------
    reference : torch.Tensor
        Either the residual ``R`` (error-budget) or the full
        reconstruction ``V_pred`` (redundancy).
    fractional : bool
        If True, compute fractional contribution instead of absolute error.

    Returns
    -------
    torch.Tensor, shape (N,)
        Per-splat error or fractional-contribution metric.
    """
    reference_flat = reference.reshape(-1)
    d = len(shape)

    if d == 2:
        return _deletion_error_2d(
            centers,
            Ls,
            amps,
            reference_flat,
            shape,
            truncate,
            intensity_floor,
            chunk_size,
            fractional,
        )
    if d == 3:
        return _deletion_error_3d(
            centers,
            Ls,
            amps,
            reference_flat,
            shape,
            truncate,
            intensity_floor,
            chunk_size,
            fractional,
        )
    return _deletion_error_nd(
        centers,
        Ls,
        amps,
        reference_flat,
        shape,
        truncate,
        intensity_floor,
        chunk_size,
        fractional,
    )


# Public error-budget entry point (the private worker also serves the
# fractional=True redundancy path).
def compute_per_splat_deletion_error(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    residual: torch.Tensor,
    shape: Sequence[int],
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute per-splat maximum deletion error (error-budget mode).

    Convenience wrapper around :func:`_compute_per_splat_error` with
    ``fractional=False``.  See that function for details.
    """
    return _compute_per_splat_error(
        centers,
        Ls,
        amps,
        residual,
        shape,
        truncate,
        intensity_floor,
        chunk_size,
        fractional=False,
    )


# =====================================================================
# Main culling function
# =====================================================================


def cull_by_contribution(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    target: Optional[torch.Tensor],
    shape: Sequence[int],
    truncate: float = 3.0,
    error_percentile: float = 99.0,
    error_tolerance: float = 1.0,
    redundancy_threshold: float = 0.01,
    max_binary_search_iters: int = 8,
    intensity_floor: float = 1e-5,
    chunk_size: Optional[int] = None,
    verbose: bool = False,
) -> CullResult:
    """Cull splats that contribute negligibly to the reconstruction.

    This function supports two modes, selected automatically based on
    whether a *target* volume is provided:

    **Error-budget mode** (``target`` is not None)
        Computes the residual ``R = target - V_pred`` and establishes an
        error budget ``tau = percentile(|R|, error_percentile) * error_tolerance``.
        A splat is safe to remove when the worst-case error *increase*
        ``max_x max(0, |R(x)+g_j(x)| - |R(x)|) <= tau`` — i.e., removing
        it does not degrade any voxel's error by more than the budget.
        This formulation is robust to pre-existing high-error voxels: a
        splat in a noisy region can still be culled if it contributes
        negligibly to the reconstruction there.  This is the most principled
        approach: it uses the actual reconstruction quality to set the
        threshold, and it can detect splats in high-error regions where
        removal is harmless.

    **Redundancy mode** (``target`` is None)
        Works from the splats alone — no target volume needed.  Renders
        the full reconstruction ``V_pred = sum g_i`` and measures each
        splat's maximum *fractional contribution*:
        ``max_x  g_j(x) / V_pred(x)``.  A splat is safe to remove when
        its fractional contribution is everywhere below
        ``redundancy_threshold`` — other splats already cover its region.
        This mode is useful when the original volume is unavailable (e.g.,
        pre-computed splat datasets), but it cannot account for fitting
        error and may be slightly more conservative.

    Both modes include a **joint compounding check**: after identifying
    individual candidates, the function verifies that their *joint*
    removal does not exceed the budget.  If it does (because overlapping
    candidates compound), the threshold is tightened via binary search.

    Parameters
    ----------
    centers : torch.Tensor, shape (N, d)
        Splat center positions (GPU tensor).
    Ls : torch.Tensor, shape (N, d, d)
        Lower-triangular Cholesky factors (GPU tensor).
    amps : torch.Tensor, shape (N,)
        Splat amplitudes (GPU tensor).
    target : torch.Tensor or None
        Target volume to compare against.  If provided, error-budget mode
        is used.  If ``None``, redundancy mode is used.
    shape : Sequence[int]
        Volume shape for rendering.
    truncate : float
        Truncation radius in standard deviations.
    error_percentile : float
        *Error-budget mode only.*  Percentile of ``|residual|`` used to set
        the budget (0--100).  Higher = more conservative.
    error_tolerance : float
        *Error-budget mode only.*  Multiplier on the budget.
    redundancy_threshold : float
        *Redundancy mode only.*  Maximum fractional contribution below
        which a splat is considered redundant (0--1).  E.g. 0.01 means
        "remove splats contributing < 1% of the local signal everywhere."
    max_binary_search_iters : int
        Maximum binary-search iterations for the joint compounding check.
    intensity_floor : float
        Minimum intensity threshold for AABB computation.
    chunk_size : int, optional
        Chunk size for memory management.
    verbose : bool
        Print progress information via arbol.

    Returns
    -------
    CullResult
        Culling result with keep_mask, diagnostics, and metadata.

    Examples
    --------
    Error-budget mode (target available):

    >>> result = cull_by_contribution(centers, Ls, amps, target, shape)

    Redundancy mode (no target):

    >>> result = cull_by_contribution(
    ...     centers, Ls, amps, None, shape,
    ...     redundancy_threshold=0.02,
    ... )
    """
    N = centers.shape[0]
    use_error_budget = target is not None
    mode = "error_budget" if use_error_budget else "redundancy"

    if N == 0:
        return CullResult(
            keep_mask=np.ones(0, dtype=bool),
            n_culled=0,
            error_budget=0.0,
            phase1_candidates=0,
            phase2_iterations=0,
            max_joint_error=0.0,
            mode=mode,
        )

    with torch.no_grad():
        # --- Step 1: Render full reconstruction ---
        V_pred = render_gaussians(
            shape, centers, Ls, amps, truncate, intensity_floor, chunk_size
        )

        if use_error_budget:
            # --- Error-budget mode ---
            assert target is not None  # guarded by use_error_budget
            R = target - V_pred
            abs_R = torch.abs(R)
            tau = (
                torch.quantile(abs_R.reshape(-1), error_percentile / 100.0).item()
                * error_tolerance
            )
            reference = R
            fractional = False
        else:
            # --- Redundancy mode ---
            tau = redundancy_threshold
            reference = V_pred
            fractional = True

        if verbose:
            label = "Error-budget culling" if use_error_budget else "Redundancy culling"
            with asection(label):
                aprint(f"Splats: {N}")
                aprint(f"Mode: {mode}")
                if use_error_budget:
                    aprint(f"Error budget (tau): {tau:.6f}")
                    aprint(
                        f"  percentile={error_percentile}, tolerance={error_tolerance}"
                    )
                    aprint(
                        f"  residual max={abs_R.max().item():.6f}, "
                        f"mean={abs_R.mean().item():.6f}"
                    )
                else:
                    aprint(
                        f"Redundancy threshold: {tau:.4f} "
                        f"({tau * 100:.1f}% max fractional contribution)"
                    )
                    aprint(
                        f"  V_pred max={V_pred.max().item():.6f}, "
                        f"mean={V_pred.mean().item():.6f}"
                    )

        # --- Candidate metric pass ---
        per_splat_errors = _compute_per_splat_error(
            centers,
            Ls,
            amps,
            reference,
            shape,
            truncate,
            intensity_floor,
            chunk_size,
            fractional,
        )

        safe_mask = per_splat_errors <= tau
        n_phase1 = int(safe_mask.sum().item())

        if verbose:
            aprint(f"Candidates: {n_phase1}/{N} individually safe to remove")

        if n_phase1 == 0:
            max_ref = (
                abs_R.max().item()
                if use_error_budget
                else per_splat_errors.max().item()
            )
            return CullResult(
                keep_mask=np.ones(N, dtype=bool),
                n_culled=0,
                error_budget=tau,
                phase1_candidates=0,
                phase2_iterations=0,
                max_joint_error=max_ref,
                mode=mode,
            )

        # --- Joint compounding check ---
        # Full binary search: find the MAXIMUM multiplier on tau whose
        # joint removal satisfies the constraint.  The old code stopped
        # at the first passing multiplier, which could overshoot (tighten
        # too much) and produce non-monotonic results across percentiles.
        best_safe_mask = torch.zeros(N, dtype=torch.bool, device=centers.device)
        lo_mult = 0.0
        hi_mult = 1.0
        final_max_joint_error = 0.0
        n_iterations = 0

        def _check_joint(mult: float) -> tuple[bool, float]:
            """Check if removing all splats with damage <= mult*tau is safe."""
            threshold = mult * tau
            mask = per_splat_errors <= threshold
            n_s = int(mask.sum().item())

            if n_s == 0:
                return True, 0.0

            indices = mask.nonzero(as_tuple=True)[0]
            G_s = render_gaussians(
                shape,
                centers[indices],
                Ls[indices],
                amps[indices],
                truncate,
                intensity_floor,
                chunk_size,
            )

            if use_error_budget:
                R_joint = R + G_s
                joint_damage = torch.clamp(torch.abs(R_joint) - torch.abs(R), min=0.0)
                max_err = joint_damage.max().item()
                ok = max_err <= tau
            else:
                frac_removed = G_s / torch.clamp(V_pred, min=1e-12)
                max_err = frac_removed.max().item()
                V_remaining = V_pred - G_s
                ok = (max_err <= tau) and (V_remaining.min().item() >= -1e-6)

            return ok, max_err

        # First try: full threshold (mult=1.0)
        ok, max_err = _check_joint(1.0)
        final_max_joint_error = max_err

        if ok:
            # Joint check passed at full threshold — no tightening needed
            best_safe_mask = per_splat_errors <= tau
            if verbose:
                n_s = int(best_safe_mask.sum().item())
                aprint(
                    f"Joint check: PASS at full threshold — "
                    f"max_joint={max_err:.6f} <= tau={tau:.6f}, "
                    f"removing {n_s} splats"
                )
        else:
            # Binary search to find maximum working multiplier
            lo_mult = 0.0
            hi_mult = 1.0

            for iteration in range(max_binary_search_iters):
                n_iterations = iteration + 1
                mid = (lo_mult + hi_mult) / 2.0
                ok, max_err = _check_joint(mid)

                n_at_mid = int((per_splat_errors <= mid * tau).sum().item())
                if verbose:
                    status = "PASS" if ok else "FAIL"
                    aprint(
                        f"Joint check (iter {iteration}): {status} — "
                        f"mult={mid:.4f}, {n_at_mid} splats, "
                        f"max_joint={max_err:.6f}"
                    )

                if ok:
                    lo_mult = mid
                    best_safe_mask = per_splat_errors <= (mid * tau)
                    final_max_joint_error = max_err
                else:
                    hi_mult = mid

        safe_mask = best_safe_mask
        n_culled = int(safe_mask.sum().item())
        keep_mask = ~safe_mask

        if verbose:
            aprint(
                f"Final: culled {n_culled}/{N} splats "
                f"({100.0 * n_culled / N:.1f}%), "
                f"keeping {N - n_culled}"
            )

    return CullResult(
        keep_mask=keep_mask.cpu().numpy(),
        n_culled=n_culled,
        error_budget=tau,
        phase1_candidates=n_phase1,
        phase2_iterations=n_iterations,
        max_joint_error=final_max_joint_error,
        mode=mode,
    )
