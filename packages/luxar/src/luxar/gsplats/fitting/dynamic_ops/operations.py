# operations.py
"""Fixed-pool splat relocation for adaptive Gaussian splatting.

Performance-optimized implementation using batched tensor operations.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from arbol import aprint, asection

from luxar.gsplats.fitting.dynamic_ops.config import DynamicOpsConfig
from luxar.gsplats.fitting.dynamic_ops.peak_finding import _find_residual_peaks
from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus


def apply_dynamic_operations(
    model,
    V_target: torch.Tensor,
    V_pred: torch.Tensor,
    cfg: DynamicOpsConfig,
    max_abs_error_threshold: float,
    verbose: bool = False,
) -> bool:
    """
    Apply fixed-pool splat relocation for adaptive Gaussian splatting.

    Instead of adding/removing splats, this relocates weak splats to high-residual
    regions. This preserves the total splat count and works with standard PyTorch
    Adam optimizer (no per-splat optimizer needed).

    Algorithm:
    1. Find residual peaks (high-error locations needing coverage)
    2. Identify weak splats (low importance = amplitude × volume)
    3. Match weak splats to peaks (avoiding already-covered locations)
    4. Relocate matched splats to their assigned peaks

    Args:
        model: GaussianSplatModel with current splat parameters
        V_target: Target tensor to reconstruct
        V_pred: Current prediction tensor from model
        cfg: Dynamic operations configuration
        max_abs_error_threshold: Convergence threshold
        verbose: Whether to print detailed progress information

    Returns:
        bool: True if any splats were relocated
    """
    with torch.no_grad():
        # === Cache model parameters once (avoid repeated current_params() calls) ===
        centers, Ls, amps, _ = model.current_params()

        # Compute residual image
        residual = V_target - V_pred

        # === STEP 1: Find Residual Peaks ===
        peak_locations = _find_residual_peaks(
            residual,
            cfg.k_max_residuals,
            cfg.nms_radius_vox,
            enable_tiled=cfg.enable_tiled_seeding,
            num_tiles_per_dim=cfg.num_tiles_per_dim,
        )

        if len(peak_locations) == 0:
            if verbose:
                aprint("No residual peaks found - skipping relocation")
            return False

        if verbose:
            mode = "tiled" if cfg.enable_tiled_seeding else "global"
            aprint(f"Found {len(peak_locations)} residual peaks (mode: {mode})")

        # Convergence guard: skip if strongest residual below threshold
        # Note: peak_locations only contains positive residual locations (undershoot),
        # so residual value is already positive
        strongest_peak_coords = peak_locations[0]  # Sorted by strength
        strongest_peak_residual = residual[strongest_peak_coords].item()

        if strongest_peak_residual < max_abs_error_threshold:
            if verbose:
                aprint(
                    f"Convergence guard: strongest residual {strongest_peak_residual:.5f} "
                    f"< threshold {max_abs_error_threshold:.5f}"
                )
                aprint("  → Skipping all dynamic operations")
            return False

        # === STEP 2: Identify Weak Splats ===
        importance = _calculate_splat_importance(Ls, amps)
        weak_splat_indices = _select_weak_splats(importance, cfg.relocation_percentile)

        if len(weak_splat_indices) == 0:
            if verbose:
                aprint("No weak splats found - skipping relocation")
            return False

        if verbose:
            aprint(
                f"Identified {len(weak_splat_indices)} weak splats "
                f"(bottom {cfg.relocation_percentile}% by importance)"
            )

        # === STEP 3: Match Weak Splats to Peaks (vectorized) ===
        matches = _match_weak_splats_to_peaks_batch(
            centers,
            Ls,
            amps,
            weak_splat_indices,
            peak_locations,
            cfg.min_contribution_threshold,
            cfg.max_relocations_per_step,
            verbose,
        )

        if len(matches) == 0:
            if verbose:
                aprint("No valid relocation matches found")
            return False

        # === STEP 4: Relocate Splats ===
        n_relocated = 0
        if verbose:
            with asection(f"Relocating {len(matches)} splats"):
                for splat_idx, peak_coords in matches:
                    _relocate_splat(model, splat_idx, peak_coords, residual, cfg)
                    n_relocated += 1
                    aprint(
                        f"Splat {splat_idx} → {peak_coords} "
                        f"(importance was {importance[splat_idx]:.6f})"
                    )
        else:
            for splat_idx, peak_coords in matches:
                _relocate_splat(model, splat_idx, peak_coords, residual, cfg)
                n_relocated += 1

        if verbose and n_relocated > 0:
            aprint(f"Dynamic ops: Relocated {n_relocated} splats")

        return n_relocated > 0


def _calculate_splat_importance(Ls: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
    """
    Calculate importance metric for all splats: amplitude × volume.

    Approximates splat "mass" using: importance = a_k × prod(diag(L_k))
    This is a computationally efficient approximation of ∫ f_k(x) dx.

    Args:
        Ls: Cholesky factors, shape (N, d, d)
        amps: Amplitudes, shape (N,)

    Returns:
        torch.Tensor: Importance values for all splats, shape (N,)
    """
    # Compute volume approximation: prod(diag(L_k)) for each splat
    # This approximates sqrt(det(Σ)) where Σ = L @ L^T
    diag_products = torch.prod(torch.diagonal(Ls, dim1=-2, dim2=-1), dim=-1)

    # Importance = amplitude × volume
    importance = amps * diag_products

    return importance


def _select_weak_splats(
    importance: torch.Tensor, relocation_percentile: float
) -> List[int]:
    """
    Select least important splats as relocation candidates.

    Args:
        importance: Importance values for all splats
        relocation_percentile: Percentage of least important splats to consider

    Returns:
        List of splat indices sorted by importance (least important first)
    """
    n_splats = len(importance)
    n_candidates = max(1, int(n_splats * relocation_percentile / 100.0))

    # Get indices sorted by importance (ascending order)
    sorted_indices = torch.argsort(importance).tolist()

    # Return least important splats
    return sorted_indices[:n_candidates]


def _compute_peak_coverage_batch(
    peak_locations: torch.Tensor,
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    exclude_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute max influence at each peak from non-excluded splats (vectorized).

    This replaces the old per-peak _find_splat_with_influence_at_location function
    with a single batched operation that processes all peaks at once.

    Args:
        peak_locations: Peak coordinates, shape (P, d)
        centers: Splat centers, shape (N, d)
        Ls: Cholesky factors, shape (N, d, d)
        amps: Amplitudes, shape (N,)
        exclude_mask: Boolean mask, True for splats to exclude (weak splats), shape (N,)

    Returns:
        torch.Tensor: Max influence at each peak from non-excluded splats, shape (P,)
    """
    P = peak_locations.shape[0]
    N, d, _ = Ls.shape
    device = peak_locations.device

    if P == 0 or N == 0:
        return torch.zeros(P, device=device)

    # Compute all pairwise differences: (P, N, d)
    # peak_locations: (P, d) -> (P, 1, d)
    # centers: (N, d) -> (1, N, d)
    diff = peak_locations[:, None, :] - centers[None, :, :]  # (P, N, d)

    # Early distance filtering to save memory/compute for distant splats
    # Compute max sigma per splat for reach estimation
    max_sigma_per_splat = (
        torch.diagonal(Ls, dim1=-2, dim2=-1).max(dim=-1).values
    )  # (N,)
    max_reach = 6.0 * max_sigma_per_splat  # (N,)

    # Euclidean distance from each peak to each splat center
    euclidean_dist = torch.norm(diff, dim=-1)  # (P, N)

    # Mask for splats within reach of each peak
    within_reach = euclidean_dist <= max_reach[None, :]  # (P, N)

    # Combined mask: within reach AND not excluded
    active_mask = within_reach & ~exclude_mask[None, :]  # (P, N)

    # If no active splats for any peak, return zeros
    if not active_mask.any():
        return torch.zeros(P, device=device)

    # For efficiency, we'll compute influences for all (P, N) pairs but mask later
    # Batched solve: For each (p, n) pair, solve L[n] @ y = diff[p, n]
    # Reshape for batched solve_triangular:
    # Ls: (N, d, d) -> expand to (P, N, d, d) -> reshape to (P*N, d, d)
    # diff: (P, N, d) -> reshape to (P*N, d, 1)

    Ls_expanded = Ls[None, :, :, :].expand(P, -1, -1, -1).reshape(P * N, d, d)
    diff_flat = diff.reshape(P * N, d, 1)

    # Batched triangular solve
    # y shape: (P*N, d, 1)
    y = torch.linalg.solve_triangular(Ls_expanded, diff_flat, upper=False)

    # Squared Mahalanobis distance: ||y||^2 for each (p, n) pair
    squared_dist = (y.squeeze(-1) ** 2).sum(dim=-1).reshape(P, N)  # (P, N)

    # Compute influences: a * exp(-0.5 * ||y||^2)
    influences = amps[None, :] * torch.exp(-0.5 * squared_dist)  # (P, N)

    # Zero out excluded splats and out-of-reach splats
    influences = influences * active_mask.float()

    # Max influence per peak
    max_influences, _ = influences.max(dim=1)  # (P,)

    return max_influences


def _find_closest_weak_splats_batch(
    peak_locations: torch.Tensor,
    centers: torch.Tensor,
    weak_indices: torch.Tensor,
    uncovered_mask: torch.Tensor,
) -> List[Tuple[int, int]]:
    """
    For each uncovered peak, find closest available weak splat (vectorized).

    Uses greedy matching: process peaks in order, assign closest available weak splat.

    Args:
        peak_locations: All peak coordinates, shape (P, d)
        centers: All splat centers, shape (N, d)
        weak_indices: Indices of weak splats, shape (W,)
        uncovered_mask: Boolean mask for uncovered peaks, shape (P,)

    Returns:
        List of (weak_splat_idx, peak_idx) pairs
    """
    device = peak_locations.device

    # Get uncovered peak indices
    uncovered_peak_indices = torch.where(uncovered_mask)[0]  # (U,)
    U = len(uncovered_peak_indices)

    if U == 0 or len(weak_indices) == 0:
        return []

    # Get weak splat centers
    weak_centers = centers[weak_indices]  # (W, d)

    # Get uncovered peak locations
    uncovered_peaks = peak_locations[uncovered_peak_indices]  # (U, d)

    # Compute pairwise distances: (U, W)
    dists = torch.cdist(uncovered_peaks, weak_centers)

    # Greedy matching: process peaks in order (they're already sorted by residual strength)
    matches = []
    available_mask = torch.ones(len(weak_indices), dtype=torch.bool, device=device)

    for i in range(U):
        if not available_mask.any():
            break

        # Get distances for this peak, mask unavailable weak splats
        peak_dists = dists[i].clone()
        peak_dists[~available_mask] = float("inf")

        # Find closest available weak splat
        closest_weak_local = peak_dists.argmin()

        # Check if any available (inf means none available)
        if peak_dists[closest_weak_local] == float("inf"):
            break

        # Record match (global indices)
        closest_weak_global = weak_indices[closest_weak_local].item()
        peak_global = uncovered_peak_indices[i].item()

        matches.append((closest_weak_global, peak_global))
        available_mask[closest_weak_local] = False

    return matches


def _match_weak_splats_to_peaks_batch(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    weak_splat_indices: List[int],
    peak_locations: List[Tuple[int, ...]],
    min_contribution_threshold: float,
    max_relocations: int,
    verbose: bool = False,
) -> List[Tuple[int, Tuple[int, ...]]]:
    """
    Match weak splats to residual peaks using vectorized operations.

    Algorithm:
    1. Batch compute coverage for all peaks at once
    2. Identify uncovered peaks (influence < threshold)
    3. Batch match uncovered peaks to closest weak splats

    Args:
        centers: Splat centers, shape (N, d)
        Ls: Cholesky factors, shape (N, d, d)
        amps: Amplitudes, shape (N,)
        weak_splat_indices: Indices of weak splats eligible for relocation
        peak_locations: Peak coordinates sorted by residual magnitude (descending)
        min_contribution_threshold: Minimum influence to consider a peak "covered"
        max_relocations: Maximum number of relocations to return
        verbose: Whether to print debug info

    Returns:
        List of (splat_idx, peak_coords) pairs
    """
    device = centers.device
    N = centers.shape[0]
    P = len(peak_locations)

    if P == 0 or len(weak_splat_indices) == 0:
        return []

    # Convert peaks to tensor
    peak_tensor = torch.tensor(peak_locations, dtype=torch.float32, device=device)

    # Convert weak indices to tensor
    weak_tensor = torch.tensor(weak_splat_indices, dtype=torch.long, device=device)

    # Create exclusion mask (True = weak splat, exclude from coverage check)
    exclude_mask = torch.zeros(N, dtype=torch.bool, device=device)
    exclude_mask[weak_tensor] = True

    # Step 1: Batch compute coverage for all peaks
    max_influences = _compute_peak_coverage_batch(
        peak_tensor, centers, Ls, amps, exclude_mask
    )

    # Step 2: Identify uncovered peaks
    uncovered_mask = max_influences < min_contribution_threshold

    if verbose:
        n_covered = (~uncovered_mask).sum().item()
        if n_covered > 0:
            aprint(f"  {n_covered} peaks already covered by existing splats")

    # Step 3: Match uncovered peaks to closest weak splats
    matches_indices = _find_closest_weak_splats_batch(
        peak_tensor, centers, weak_tensor, uncovered_mask
    )

    # Limit relocations
    matches_indices = matches_indices[:max_relocations]

    # Convert back to expected format: (weak_idx, peak_coords_tuple)
    return [
        (weak_idx, peak_locations[peak_idx]) for weak_idx, peak_idx in matches_indices
    ]


def _relocate_splat(
    model,
    splat_idx: int,
    new_center_coords: Tuple[int, ...],
    residual: torch.Tensor,
    cfg: DynamicOpsConfig,
) -> None:
    """
    Relocate a splat to a new location.

    Resets the splat's parameters:
    - Center: set to new location
    - Covariance: reset to isotropic (init_sigma_vox)
    - Amplitude: set to residual value at new location
    - Sharpness: kept unchanged (optimizer will adjust)

    Args:
        model: GaussianSplatModel
        splat_idx: Index of splat to relocate
        new_center_coords: New center coordinates (voxel coordinates)
        residual: Residual tensor (for amplitude initialization)
        cfg: Dynamic operations configuration
    """
    device = model.raw_mu.device
    d = len(model.shape)
    shape_arr = torch.tensor(model.shape, device=device, dtype=torch.float32)

    # New center in voxel coordinates
    new_center = torch.tensor(new_center_coords, dtype=torch.float32, device=device)

    # Convert center to normalized [0,1] coordinates then to raw (logit) space
    u = torch.clamp(
        new_center / torch.clamp(shape_arr - 1.0, min=1.0), 1e-6, 1.0 - 1e-6
    )
    raw_mu_new = torch.log(u) - torch.log(1.0 - u)

    # New amplitude from residual at new location
    # Note: We only relocate to positive residual locations (undershoot),
    # so residual value is already positive (target > prediction)
    new_amplitude = residual[new_center_coords].item()
    new_amplitude = max(new_amplitude, 1e-6)  # Defensive: avoid zero/negative
    raw_a_new = torch.tensor(
        stable_inverse_softplus(new_amplitude), device=device, dtype=torch.float32
    )

    # New isotropic covariance: L_diag = init_sigma_vox
    # raw_L_diag = inverse_softplus(init_sigma_vox - sigma_min_diag)
    # Since sigma_min_diag is applied in the model, we need to account for it
    sigma_min = (
        model.sigma_min_diag[0].item() if model.sigma_min_diag is not None else 0.0
    )
    effective_diag = max(cfg.init_sigma_vox - sigma_min, 1e-6)
    raw_L_diag_new = torch.tensor(
        stable_inverse_softplus(effective_diag), device=device, dtype=torch.float32
    )
    raw_L_diag_new = raw_L_diag_new.expand(d)

    # Zero off-diagonal elements (isotropic)
    n_off_diag = d * (d - 1) // 2
    L_off_new = torch.zeros(n_off_diag, device=device, dtype=torch.float32)

    # Update model parameters in-place
    model.raw_mu.data[splat_idx] = raw_mu_new
    model.raw_L_diag.data[splat_idx] = raw_L_diag_new
    model.L_off.data[splat_idx] = L_off_new
    model.raw_a.data[splat_idx] = raw_a_new
    # Keep sharpness unchanged - optimizer will adjust if needed
