# operations.py
"""Main dynamic operations for adaptive Gaussian splatting."""

from __future__ import annotations

from typing import List, Tuple

import torch
from arbol import aprint, asection

from luxar.gsplats.fitting.dynamic_ops.config import DynamicOpsConfig
from luxar.gsplats.fitting.dynamic_ops.peak_finding import _find_residual_peaks
from luxar.gsplats.optim import ModelOptimizerCoordinator


def apply_dynamic_operations(
    model,
    optimizer,
    scheduler,
    V_target: torch.Tensor,
    V_pred: torch.Tensor,
    cfg: DynamicOpsConfig,
    current_lr: float,
    max_abs_error_threshold: float,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, bool]:
    """
    Apply convergence-driven dynamic operations for adaptive Gaussian splatting.

    Implements the three-step algorithm with convergence-based detection:
    1. Residual Peak Analysis: Find k strongest residual peaks with convergence guard
    2. Convergence-Based Operations: Seed/split only where coverage is insufficient
    3. Principled Pruning Analysis: Remove ineffective splats based on quality impact

    Key Features:
    - Convergence-based detection: only act where residual > convergence threshold
    - Adaptive thresholds: amplitude validation scales with local residual magnitude
    - Influence-based splitting: identify dominant splats for refinement
    - Adaptive covariance: estimate splat size from local residual structure

    Args:
        model: GaussianSplatModel with current splat parameters
        optimizer: PerSplatAdam optimizer with per-splat learning rates
        scheduler: Per-splat scheduler for learning rate adaptation
        V_target: Target tensor to reconstruct
        V_pred: Current prediction tensor from model
        cfg: Dynamic operations configuration with all algorithm parameters
        current_lr: Current global learning rate for new splats
        max_abs_error_threshold: Convergence threshold (inf if no convergence criterion)
        device: PyTorch device for tensor operations
        verbose: Whether to print detailed progress information

    Returns:
        tuple: (optimizer, scheduler, operations_occurred)
               operations_occurred=True if any splats were added/removed
    """
    # Create coordinator for seamless optimizer state management
    coordinator = ModelOptimizerCoordinator(model, optimizer, scheduler)

    with torch.no_grad():
        # Compute residual image
        residual = V_target - V_pred

        # Track if we modified model topology
        topology_changed = False
        operations_performed = []

        # === STEP 1: Residual Peak Analysis ===
        # Find residual peaks (tiled for spatial fairness or global)
        peak_locations = _find_residual_peaks(
            residual,
            cfg.k_max_residuals,
            cfg.nms_radius_vox,
            enable_tiled=cfg.enable_tiled_seeding,
            num_tiles_per_dim=cfg.num_tiles_per_dim,  # type: ignore[arg-type]
        )

        if verbose and len(peak_locations) > 0:
            mode = "tiled" if cfg.enable_tiled_seeding else "global"
            aprint(f"Found {len(peak_locations)} residual peaks (mode: {mode})")

        # Convergence guard: skip if strongest residual below threshold
        if len(peak_locations) > 0:
            strongest_peak_coords = peak_locations[0]  # Sorted by strength
            strongest_peak_residual = torch.abs(residual[strongest_peak_coords]).item()

            if strongest_peak_residual < max_abs_error_threshold:
                if verbose:
                    aprint(
                        f"Convergence guard: strongest residual {strongest_peak_residual:.5f} < threshold {max_abs_error_threshold:.5f}"
                    )
                    aprint("  → Skipping all dynamic operations")
                return optimizer, scheduler, False

        # === STEP 2: Adaptive Splat Operations ===
        for peak_coords in peak_locations:
            peak_location = torch.tensor(
                peak_coords, dtype=torch.float32, device=device
            )
            local_residual = torch.abs(residual[peak_coords]).item()

            # CONVERGENCE-BASED DETECTION: Is coverage sufficient?
            coverage_sufficient = _is_coverage_sufficient(
                local_residual, max_abs_error_threshold
            )

            if verbose:
                aprint(
                    f"    Peak {peak_coords}: residual {local_residual:.5f} vs threshold {max_abs_error_threshold:.5f} → coverage {'sufficient' if coverage_sufficient else 'insufficient'}"
                )

            if not coverage_sufficient:  # Coverage is insufficient - need action
                # Check if existing splat has significant influence at this location
                influential_splat_idx, influence_value = (
                    _find_splat_with_significant_influence_at_location(
                        model,
                        peak_location,
                        min_influence_threshold=cfg.min_contribution_threshold,
                    )
                )

                if verbose:
                    if influential_splat_idx != -1:
                        aprint(
                            f"      → Existing influence: splat {influential_splat_idx} has {influence_value:.6f}"
                        )
                    else:
                        aprint(
                            f"      → No existing influence above {cfg.min_contribution_threshold}"
                        )

                if influential_splat_idx != -1:
                    # Existing splat covers this area - boost its LR instead of adding new splat
                    # This prevents splat proliferation in already-covered regions
                    if influence_value >= cfg.boost_influence_threshold:
                        boosted_lr = _boost_splat_learning_rate(
                            optimizer, influential_splat_idx, current_lr, cfg
                        )
                        if verbose:
                            aprint(
                                f"      → Boosted LR for splat {influential_splat_idx}: {boosted_lr:.6f} (factor: {cfg.lr_boost_factor})"
                            )
                            aprint(
                                "      → Skipping seeding (existing splat will handle coverage)"
                            )
                    elif verbose:
                        aprint(
                            f"      → Existing splat influence {influence_value:.6f} below boost threshold {cfg.boost_influence_threshold}"
                        )
                        aprint("      → Skipping seeding (existing splat present)")
                else:
                    # No existing coverage - seed new splat
                    # Use adaptive threshold based on local residual magnitude
                    min_amp_threshold = (
                        local_residual * cfg.relative_contribution_factor
                    )
                    if verbose:
                        aprint(
                            f"      → Attempting seeding (min amplitude threshold = {min_amp_threshold:.6f})..."
                        )

                    success = _seed_new_splat(
                        coordinator,
                        model.shape,
                        peak_location,
                        residual,
                        cfg,
                        current_lr,
                        min_amplitude_threshold=min_amp_threshold,
                    )
                    if success:
                        topology_changed = True
                        operations_performed.append(f"Seeded splat at {peak_coords}")
                        if verbose:
                            aprint(
                                f"      ✓ Successfully seeded splat at {peak_coords}"
                            )
                    else:
                        if verbose:
                            aprint(
                                f"      ✗ Failed to seed splat at {peak_coords} - amplitude below threshold"
                            )
            elif verbose:
                aprint("    → Coverage sufficient, no action needed")

        # === STEP 3: Principled Splat Pruning Analysis ===
        if model.n_splats() > cfg.min_splats_to_keep:
            removable_splats = _principled_pruning_analysis(
                model, V_target, V_pred, cfg, max_abs_error_threshold, verbose
            )

            if len(removable_splats) > 0:
                # Ensure we don't remove too many splats
                n_to_remove = min(
                    len(removable_splats), model.n_splats() - cfg.min_splats_to_keep
                )
                if n_to_remove > 0:
                    # Create keep mask (True for splats to keep)
                    keep_mask = torch.ones(
                        model.n_splats(), dtype=torch.bool, device=device
                    )
                    for splat_idx in removable_splats[:n_to_remove]:
                        keep_mask[splat_idx] = False

                    coordinator.prune_splats(keep_mask)
                    topology_changed = True
                    operations_performed.append(
                        f"Pruned {n_to_remove} ineffective splats (importance-based)"
                    )

        # Report operations performed
        if verbose and operations_performed:
            if len(operations_performed) == 1:
                aprint(f"Dynamic ops: {operations_performed[0]}")
            else:
                with asection("Dynamic Operations"):
                    for op in operations_performed:
                        aprint(f"• {op}")
        elif verbose:
            aprint("Dynamic ops: No operations performed")

    return optimizer, scheduler, topology_changed


def _boost_splat_learning_rate(
    optimizer, splat_idx: int, current_lr: float, cfg: DynamicOpsConfig
) -> float:
    """
    Boost learning rate for a splat covering a problematic region.

    Args:
        optimizer: PerSplatAdam optimizer
        splat_idx: Index of splat to boost
        current_lr: Current base learning rate
        cfg: Dynamic operations configuration

    Returns:
        float: New boosted learning rate (capped at base rate)
    """
    # Get current learning rate for this splat
    current_splat_lr = optimizer.get_effective_learning_rates()[splat_idx]

    # Calculate boosted learning rate with safety cap
    boosted_lr = min(current_splat_lr * cfg.lr_boost_factor, current_lr)

    # Apply the boost
    optimizer.set_learning_rate(splat_idx, boosted_lr)

    return boosted_lr


def _is_coverage_sufficient(
    local_residual: float, max_abs_error_threshold: float
) -> bool:
    """
    Determine if coverage at a location is sufficient based on convergence criteria.

    Args:
        local_residual: Absolute residual value at the location
        max_abs_error_threshold: Convergence threshold (always finite with auto-threshold)

    Returns:
        bool: True if coverage is sufficient, False if more coverage is needed
    """
    # Coverage is sufficient if residual is below convergence threshold
    return local_residual <= max_abs_error_threshold


def _find_splat_with_significant_influence_at_location(
    model, location: torch.Tensor, min_influence_threshold: float = 0.01
) -> Tuple[int, float]:
    """
    Find splat with significant influence at the given location.

    This replaces position-based detection with influence-based detection to avoid
    the issue where splats migrate away from their seeded locations during optimization.

    Uses vectorized operations for O(N) computation but with efficient tensor ops.
    For very large N (>10000), could be further optimized with spatial indexing.

    Args:
        model: GaussianSplatModel
        location: Location to check (d,)
        min_influence_threshold: Minimum influence value to consider significant

    Returns:
        tuple: (splat_index, influence_value) where influence_value is the Gaussian value
               Returns (-1, 0.0) if no splat has significant influence at location
    """
    if model.n_splats() == 0:
        return -1, 0.0

    centers, Ls, amps, sharpness = model.current_params()
    n_splats = model.n_splats()

    # Vectorized computation of influences for all splats
    # location: (d,), centers: (N, d), Ls: (N, d, d), amps: (N,)

    # Compute diff = location - centers for all splats: (N, d)
    diff = location.unsqueeze(0) - centers  # (1, d) - (N, d) = (N, d)

    # Early filtering: skip splats that are too far away (Euclidean distance)
    # For a Gaussian with truncate=3, influence is negligible beyond 3σ
    # Use max diagonal of L as conservative sigma estimate
    max_sigma_per_splat = (
        torch.diagonal(Ls, dim1=-2, dim2=-1).max(dim=-1).values
    )  # (N,)
    euclidean_dist = torch.norm(diff, dim=-1)  # (N,)
    max_reach = 6.0 * max_sigma_per_splat  # Conservative: 6σ reach

    # Mask for splats that could potentially have significant influence
    candidate_mask = euclidean_dist <= max_reach  # (N,)
    candidate_indices = torch.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return -1, 0.0

    # Compute full influence only for candidate splats
    influences = torch.zeros(n_splats, device=location.device)

    for idx in candidate_indices:
        i = idx.item()
        L = Ls[i]
        amp = amps[i]

        # Compute Gaussian value at location: amp * exp(-0.5 * (x-mu)^T * Sigma^-1 * (x-mu))
        # Where Sigma^-1 = (L * L^T)^-1 = L^-T * L^-1
        try:
            y = torch.linalg.solve_triangular(L, diff[i], upper=False)
            squared_distance = torch.sum(y * y)
            influences[i] = amp * torch.exp(-0.5 * squared_distance)
        except RuntimeError:
            # Handle singular matrices gracefully
            influences[i] = 0.0

    # Find splat with maximum influence
    max_influence, max_idx = torch.max(influences, dim=0)
    max_influence_val = max_influence.item()
    max_idx_val = max_idx.item()

    # Check if influence is significant enough
    if max_influence_val >= min_influence_threshold:
        return max_idx_val, max_influence_val
    else:
        return -1, 0.0


def _seed_new_splat(
    coordinator,
    shape: Tuple[int, ...],
    center: torch.Tensor,
    residual: torch.Tensor,
    cfg: DynamicOpsConfig,
    lr: float,
    min_amplitude_threshold: float = 0.0,
) -> bool:
    """
    Seed a new Gaussian splat with ultra-simple amplitude and shape estimation.

    Uses direct residual value at center for amplitude and isotropic covariance.
    Optimization will evolve optimal shapes during training.

    Args:
        coordinator: ModelOptimizerCoordinator for adding splats
        shape: Image/volume shape
        center: Center position for new splat
        residual: Residual tensor (target - prediction)
        cfg: Dynamic operations configuration
        lr: Learning rate for new splat
        min_amplitude_threshold: Minimum amplitude to seed (skips tiny splats)

    Returns:
        bool: True if seeding was successful
    """
    try:
        # Ultra-simple amplitude: residual value at center
        center_coords = torch.round(center).long()

        # Clamp coordinates to valid range
        for i in range(len(center_coords)):
            center_coords[i] = torch.clamp(center_coords[i], 0, residual.shape[i] - 1)

        amplitude = torch.abs(residual[tuple(center_coords)])

        # Ultra-simple shape: isotropic splat
        d = len(shape)
        L = torch.eye(d, device=center.device) * cfg.init_sigma_vox

        # Validate amplitude is significant enough to be worth adding
        # Skip seeding if amplitude is below minimum threshold
        amplitude_value = amplitude.item()
        if amplitude_value < min_amplitude_threshold:
            return False

        # Also skip if amplitude is below absolute minimum contribution threshold
        if amplitude_value < cfg.min_contribution_threshold:
            return False

        # Initial sharpness: 2.0 (standard Gaussian profile)
        sharpness = torch.tensor([2.0], device=center.device)

        # Add the new splat
        coordinator.add_splats(
            center.unsqueeze(0),  # (1, d)
            L.unsqueeze(0),  # (1, d, d)
            amplitude.unsqueeze(0),  # (1,)
            sharpness,  # (1,)
            lr_new=lr,
        )

        return True

    except (RuntimeError, IndexError, ValueError):
        # Handle expected errors gracefully (device mismatches, invalid coordinates, etc.)
        return False
    except Exception as e:
        # Log unexpected errors for debugging
        aprint(f"⚠️ Unexpected seeding error: {type(e).__name__}: {e}")
        return False


# Note: Splat splitting functionality removed - rarely used and added unnecessary complexity


def _calculate_splat_importance(model) -> torch.Tensor:
    """
    Calculate importance metric for all splats: amplitude × volume.

    Approximates splat "mass" using: importance = a_k × prod(diag(L_k))
    This is computationally efficient approximation of ∫ f_k(x) dx.

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


def _select_pruning_candidates(
    importance: torch.Tensor, pruning_percentile: float
) -> List[int]:
    """
    Select least important splats as pruning candidates.

    Args:
        importance: Importance values for all splats
        pruning_percentile: Percentage of least important splats to consider

    Returns:
        List of splat indices sorted by importance (least important first)
    """
    n_splats = len(importance)
    n_candidates = max(1, int(n_splats * pruning_percentile / 100.0))

    # Get indices sorted by importance (ascending order)
    sorted_indices = torch.argsort(importance).tolist()

    # Return least important splats
    return sorted_indices[:n_candidates]


def _compute_splat_influence_region(
    center: torch.Tensor, L: torch.Tensor, shape: Tuple[int, ...], radius: float = 3.0
) -> torch.Tensor:
    """
    Compute boolean mask for splat's influence region (elliptical region within radius σ).

    Args:
        center: Splat center coordinates (d,)
        L: Lower triangular covariance matrix (d, d)
        shape: Image/volume shape
        radius: Radius in standard deviations (default 3.0 for 3σ region)

    Returns:
        torch.Tensor: Boolean mask of shape `shape` indicating influence region
    """
    device = center.device
    d = len(shape)

    # Create coordinate grids
    coords = []
    for i in range(d):
        coords.append(torch.arange(shape[i], dtype=torch.float32, device=device))

    if d == 2:
        y_coords, x_coords = torch.meshgrid(coords[0], coords[1], indexing="ij")
        coord_stack = torch.stack([y_coords, x_coords], dim=-1)  # (H, W, 2)
    elif d == 3:
        z_coords, y_coords, x_coords = torch.meshgrid(
            coords[0], coords[1], coords[2], indexing="ij"
        )
        coord_stack = torch.stack(
            [z_coords, y_coords, x_coords], dim=-1
        )  # (D, H, W, 3)
    else:
        # For higher dimensions, use approximate circular region
        flat_coords = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
        distances = torch.norm(flat_coords - center, dim=-1)
        return distances <= radius * 2.0  # Approximate with circular region

    # Compute Mahalanobis distance for each pixel: (x - μ)^T Σ^{-1} (x - μ)
    # Where Σ^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
    diff = coord_stack - center  # (..., d)
    flat_diff = diff.reshape(-1, d)  # (N, d)

    # Solve L @ y = diff for each point to get y = L^{-1} @ diff
    try:
        y = torch.linalg.solve_triangular(L, flat_diff.T, upper=False).T  # (N, d)
        mahalanobis_squared = torch.sum(y * y, dim=-1)  # (N,)
        influence_mask = mahalanobis_squared <= radius * radius  # (N,)
        return influence_mask.reshape(shape)
    except RuntimeError:
        # Fallback to circular region if matrix is singular
        distances = torch.norm(diff, dim=-1)
        return distances <= radius * 2.0


def _test_local_removal_impact(
    model,
    splat_idx: int,
    V_target: torch.Tensor,
    max_abs_error_threshold: float,
    V_pred: torch.Tensor | None = None,
) -> bool:
    """
    Test if removing a splat would cause local residual to exceed convergence threshold.

    Uses local convergence-based validation: tests impact only in splat's influence region
    and ensures removal doesn't violate convergence criteria locally.

    Optimized to avoid full re-renders by computing only the single splat's contribution
    and subtracting it from the current prediction in the local region.

    Args:
        model: GaussianSplatModel
        splat_idx: Index of splat to test for removal
        V_target: Target tensor
        max_abs_error_threshold: Convergence threshold (always finite with auto-threshold)
        V_pred: Current full prediction (optional, for optimization)

    Returns:
        bool: True if splat can be safely removed, False if it should be kept
    """
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

    with torch.no_grad():
        # Get current parameters
        centers, Ls, amps, sharpness = model.current_params()

        if model.n_splats() <= 1:
            return False  # Don't remove if it's the last splat

        # Define influence region for this splat (3σ ellipse)
        splat_center = centers[splat_idx]
        splat_L = Ls[splat_idx]
        splat_amp = amps[splat_idx]
        splat_sharpness = sharpness[splat_idx]

        influence_mask = _compute_splat_influence_region(
            splat_center, splat_L, V_target.shape, radius=3.0
        )

        # Skip if influence region is empty
        if not torch.any(influence_mask):
            return True  # Can safely remove if no influence

        # Optimization: Instead of re-rendering without the splat,
        # compute just this splat's contribution and subtract from V_pred
        if V_pred is not None:
            # Render only the single splat's contribution
            single_splat_contribution = render_gaussians(
                V_target.shape,
                splat_center.unsqueeze(0),  # (1, d)
                splat_L.unsqueeze(0),  # (1, d, d)
                splat_amp.unsqueeze(0),  # (1,)
                splat_sharpness.unsqueeze(0),  # (1,)
                truncate=model.truncate,  # Use model's truncate for consistency
            )

            # Subtract splat contribution from current prediction
            pred_without_splat_local = (
                V_pred[influence_mask] - single_splat_contribution[influence_mask]
            )
        else:
            # Fallback: full re-render without this splat (slower but always correct)
            keep_mask = torch.ones(
                model.n_splats(), dtype=torch.bool, device=centers.device
            )
            keep_mask[splat_idx] = False

            temp_centers = centers[keep_mask]
            temp_Ls = Ls[keep_mask]
            temp_amps = amps[keep_mask]
            temp_sharpness = sharpness[keep_mask]

            pred_without_splat = render_gaussians(
                V_target.shape,
                temp_centers,
                temp_Ls,
                temp_amps,
                temp_sharpness,
                truncate=model.truncate,  # Use model's truncate for consistency
            )
            pred_without_splat_local = pred_without_splat[influence_mask]

        # Extract target in local region
        target_local = V_target[influence_mask]

        # Compute local residual without the splat
        local_residual = torch.abs(pred_without_splat_local - target_local)
        max_local_residual = torch.max(local_residual).item()

        # Local convergence-based decision: ensure local convergence is maintained
        can_remove = max_local_residual <= max_abs_error_threshold

        return can_remove


def _principled_pruning_analysis(
    model,
    V_target: torch.Tensor,
    V_pred: torch.Tensor,
    cfg: DynamicOpsConfig,
    max_abs_error_threshold: float,
    verbose: bool = False,
) -> List[int]:
    """
    Perform principled pruning analysis using importance-based pre-filtering
    and local convergence-based validation.

    Uses local convergence testing: removes splats only if their removal doesn't
    cause local residual to exceed convergence threshold in their influence region.

    Optimized: passes V_pred to avoid full re-renders during removal impact testing.

    Returns:
        List of splat indices that can be safely removed without violating local convergence
    """
    if model.n_splats() <= cfg.min_splats_to_keep:
        return []

    # Step 1: Calculate importance for all splats
    importance = _calculate_splat_importance(model)

    # Step 2: Pre-filter to least important splats
    candidates = _select_pruning_candidates(importance, cfg.pruning_percentile)

    if verbose and len(candidates) > 0:
        aprint(
            f"    Pruning analysis: testing {len(candidates)} least important splats ({cfg.pruning_percentile}% of {model.n_splats()})"
        )

    # Step 3: Test each candidate for local removal impact
    # Pass V_pred to enable optimized single-splat subtraction instead of full re-render

    removable_splats = []
    for splat_idx in candidates:
        can_remove = _test_local_removal_impact(
            model, splat_idx, V_target, max_abs_error_threshold, V_pred=V_pred
        )
        if can_remove:
            removable_splats.append(splat_idx)
            if verbose:
                aprint(
                    f"      → Splat {splat_idx} marked for removal (importance={importance[splat_idx]:.6f})"
                )
        elif verbose:
            aprint(f"      → Splat {splat_idx} kept (would degrade quality)")

    return removable_splats
