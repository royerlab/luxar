# operations.py
"""Fixed-pool splat relocation for adaptive Gaussian splatting."""

from __future__ import annotations

from typing import List, Set, Tuple

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
    device = V_target.device

    with torch.no_grad():
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
        strongest_peak_coords = peak_locations[0]  # Sorted by strength
        strongest_peak_residual = torch.abs(residual[strongest_peak_coords]).item()

        if strongest_peak_residual < max_abs_error_threshold:
            if verbose:
                aprint(
                    f"Convergence guard: strongest residual {strongest_peak_residual:.5f} "
                    f"< threshold {max_abs_error_threshold:.5f}"
                )
                aprint("  → Skipping all dynamic operations")
            return False

        # === STEP 2: Identify Weak Splats ===
        importance = _calculate_splat_importance(model)
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

        # === STEP 3: Match Weak Splats to Peaks ===
        matches = _match_weak_splats_to_peaks(
            model,
            weak_splat_indices,
            peak_locations,
            residual,
            cfg.min_contribution_threshold,
            verbose,
        )

        if len(matches) == 0:
            if verbose:
                aprint("No valid relocation matches found")
            return False

        # Limit relocations per step to prevent destabilization
        matches = matches[: cfg.max_relocations_per_step]

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


def _calculate_splat_importance(model) -> torch.Tensor:
    """
    Calculate importance metric for all splats: amplitude × volume.

    Approximates splat "mass" using: importance = a_k × prod(diag(L_k))
    This is a computationally efficient approximation of ∫ f_k(x) dx.

    Returns:
        torch.Tensor: Importance values for all splats, shape (N,)
    """
    centers, Ls, amps, sharpness = model.current_params()

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


def _find_splat_with_influence_at_location(
    model,
    location: torch.Tensor,
    exclude_indices: Set[int],
    min_influence_threshold: float = 0.01,
) -> Tuple[int, float]:
    """
    Find splat with significant influence at the given location.

    Uses Gaussian evaluation to detect if any splat (except those in exclude_indices)
    already covers the location.

    Args:
        model: GaussianSplatModel
        location: Location to check (d,)
        exclude_indices: Set of splat indices to exclude from search (weak splats)
        min_influence_threshold: Minimum influence value to consider significant

    Returns:
        tuple: (splat_index, influence_value)
               Returns (-1, 0.0) if no splat has significant influence
    """
    if model.n_splats() == 0:
        return -1, 0.0

    centers, Ls, amps, sharpness = model.current_params()
    n_splats = model.n_splats()

    # Compute diff = location - centers for all splats: (N, d)
    diff = location.unsqueeze(0) - centers  # (1, d) - (N, d) = (N, d)

    # Early filtering: skip splats that are too far away
    max_sigma_per_splat = torch.diagonal(Ls, dim1=-2, dim2=-1).max(dim=-1).values
    euclidean_dist = torch.norm(diff, dim=-1)
    max_reach = 6.0 * max_sigma_per_splat  # Conservative: 6σ reach

    # Mask for candidate splats (within reach and not excluded)
    candidate_mask = euclidean_dist <= max_reach
    candidate_indices = torch.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return -1, 0.0

    # Compute full influence only for candidate splats
    influences = torch.zeros(n_splats, device=location.device)

    for idx in candidate_indices:
        i = idx.item()
        if i in exclude_indices:
            continue  # Skip weak splats

        L = Ls[i]
        amp = amps[i]

        # Compute Gaussian value at location
        try:
            y = torch.linalg.solve_triangular(L, diff[i], upper=False)
            squared_distance = torch.sum(y * y)
            influences[i] = amp * torch.exp(-0.5 * squared_distance)
        except RuntimeError:
            influences[i] = 0.0

    # Find splat with maximum influence
    max_influence, max_idx = torch.max(influences, dim=0)
    max_influence_val = max_influence.item()
    max_idx_val = max_idx.item()

    if max_influence_val >= min_influence_threshold:
        return max_idx_val, max_influence_val
    else:
        return -1, 0.0


def _match_weak_splats_to_peaks(
    model,
    weak_splat_indices: List[int],
    peak_locations: List[Tuple[int, ...]],
    residual: torch.Tensor,
    min_contribution_threshold: float,
    verbose: bool = False,
) -> List[Tuple[int, Tuple[int, ...]]]:
    """
    Match weak splats to residual peaks, avoiding crowding.

    Algorithm:
    1. Process peaks in order of residual magnitude (strongest first)
    2. For each peak:
       - Check if any non-weak splat already covers it
       - If covered, skip this peak
       - If not covered, assign closest unassigned weak splat

    Args:
        model: GaussianSplatModel
        weak_splat_indices: Indices of weak splats eligible for relocation
        peak_locations: Peak coordinates sorted by residual magnitude (descending)
        residual: Residual tensor
        min_contribution_threshold: Minimum influence to consider a peak "covered"
        verbose: Whether to print debug info

    Returns:
        List of (splat_idx, peak_coords) pairs
    """
    device = residual.device
    centers, _, _, _ = model.current_params()

    # Track which weak splats are still available
    available_weak = set(weak_splat_indices)
    exclude_set = set(weak_splat_indices)  # Exclude weak splats from coverage check

    matches = []

    for peak_coords in peak_locations:
        if not available_weak:
            break  # No more weak splats to relocate

        peak_location = torch.tensor(peak_coords, dtype=torch.float32, device=device)
        local_residual = torch.abs(residual[peak_coords]).item()

        # Check if any non-weak splat already covers this peak
        covering_splat, influence = _find_splat_with_influence_at_location(
            model, peak_location, exclude_set, min_contribution_threshold
        )

        if covering_splat != -1:
            if verbose:
                aprint(
                    f"  Peak {peak_coords}: already covered by splat {covering_splat} "
                    f"(influence={influence:.4f})"
                )
            continue  # Skip - already covered

        # Find closest available weak splat to this peak
        min_dist = float("inf")
        closest_weak = None
        for splat_idx in available_weak:
            dist = torch.norm(centers[splat_idx] - peak_location).item()
            if dist < min_dist:
                min_dist = dist
                closest_weak = splat_idx

        if closest_weak is not None:
            matches.append((closest_weak, peak_coords))
            available_weak.remove(closest_weak)
            if verbose:
                aprint(
                    f"  Peak {peak_coords}: assigned to weak splat {closest_weak} "
                    f"(residual={local_residual:.4f})"
                )

    return matches


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
    new_amplitude = torch.abs(residual[new_center_coords]).item()
    new_amplitude = max(new_amplitude, 1e-6)  # Avoid zero
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
