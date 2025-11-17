# dynamic_ops.py

"""
Dynamic Gaussian Splat Operations

This module implements convergence-driven dynamic operations for adaptive Gaussian splatting.
Dynamic operations address reconstruction deficiencies by analyzing the residual image
(target - prediction) to identify where the current representation fails to meet convergence criteria.

## Core Philosophy

The approach focuses on two core operations driven by convergence requirements:
- **Seeding**: Create new splats where residual exceeds convergence thresholds
- **Pruning**: Remove splats that contribute minimally to reconstruction quality

## Implementation Approach

### Step 1: Residual Peak Analysis
- Compute residual image: residual = V_target - V_pred
- Find k strongest peaks with spatial exclusion (non-maximum suppression)
- Apply convergence guard: skip all operations if strongest residual < convergence threshold

### Step 2: Convergence-Based Splat Operations
- For each peak, determine if coverage is sufficient using convergence criteria
- **Seeding**: Coverage insufficient (residual > convergence threshold or no threshold set)
- **Adaptive learning rate boosting**: "Unfreeze" existing splats covering problematic regions
- **Adaptive thresholds**: Amplitude validation scales with local residual magnitude

### Step 3: Principled Pruning Analysis
- Independent of residual peaks, identify splats with minimal reconstruction impact
- Use importance-based pre-filtering and quality validation for principled removal
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from luxar.gsplats.clahe import apply_clahe
from luxar.gsplats.optim import ModelOptimizerCoordinator


class DynamicOpsConfig:
    """
    Configuration for convergence-driven dynamic Gaussian splat operations.

    This class contains all parameters for the three-step dynamic operations algorithm:
    1. Residual Peak Analysis: Find strongest error locations
    2. Convergence-Based Operations: Seed/split based on convergence criteria
    3. Global Pruning: Remove ineffective splats

    Key features:
    - Convergence-based detection aligns operations with optimization goals
    - Adaptive thresholds prevent plateau issues
    - Asymmetric loss awareness for additive Gaussian models
    - Hybrid seeding: combines error-driven (residual) and structure-driven (CLAHE) approaches
    """

    def __init__(self):
        # Scheduling
        self.step_every: int = 50  # Run operations every N iterations

        # Step 1: Residual Peak Analysis
        self.k_max_residuals: int = 10  # Total seed budget per cycle
        self.nms_radius_vox: float = 2.0  # Minimum distance between detected peaks

        # Hybrid Seeding Strategy: CLAHE-Based Coverage Seeding
        self.density_seeding_fraction: float = 0.0  # Fraction of seeds from CLAHE-based coverage (0.0 = disabled, 0.5 = 50-50)
        self.clahe_tile_size: int = 16  # Tile size for CLAHE in voxels (default: ~2× feature diameter)
        self.clahe_clip_limit: float = 2.0  # Contrast limiting factor for CLAHE (1.0-4.0, higher = more aggressive)
        self.clahe_nbins: int = 256  # Number of histogram bins for CLAHE equalization

        # Step 2: Adaptive Operations
        self.min_contribution_threshold: float = (
            0.05  # Legacy fixed threshold for influence detection
        )
        self.relative_contribution_factor: float = (
            0.1  # Adaptive threshold: fraction of local residual
        )

        # Adaptive Learning Rate Boosting
        self.lr_boost_factor: float = (
            1.5  # Multiplication factor for problematic regions
        )
        self.boost_influence_threshold: float = 0.05  # Minimum influence to boost LR

        # Step 3: Principled Pruning Parameters
        self.pruning_percentile: float = (
            5.0  # Percentage of least important splats to consider for removal
        )
        self.min_splats_to_keep: int = (
            10  # Minimum number of splats to retain regardless of importance
        )

        # Seeding parameters
        self.init_sigma_vox: float = 1.5  # Initial covariance for new splats


def _find_residual_peaks(
    residual: torch.Tensor, k_max_residuals: int, nms_radius_vox: float
) -> List[Tuple[int, ...]]:
    """
    Find k strongest residual peaks with spatial exclusion (non-maximum suppression).

    Args:
        residual: Residual image (target - prediction)
        k_max_residuals: Number of peaks to find
        nms_radius_vox: Minimum distance between peaks

    Returns:
        List of peak coordinates as tuples
    """
    residual_abs = torch.abs(residual)
    d = residual.ndim

    # Create kernel for non-maximum suppression
    kernel_size = int(2 * nms_radius_vox + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if d == 2:
        max_pooled = torch.nn.functional.max_pool2d(
            residual_abs[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    elif d == 3:
        max_pooled = torch.nn.functional.max_pool3d(
            residual_abs[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    else:
        # Fallback for other dimensions
        max_pooled = residual_abs

    # Find local maxima
    is_peak = (residual_abs >= max_pooled) & (residual_abs > 0)
    peak_indices = torch.nonzero(is_peak, as_tuple=False)

    if len(peak_indices) == 0:
        return []

    # Get values and sort by magnitude
    peak_values = residual_abs[tuple(peak_indices.T)]
    sorted_indices = torch.argsort(peak_values, descending=True)

    # Take top k
    top_k = min(k_max_residuals, len(sorted_indices))
    selected_peaks = peak_indices[sorted_indices[:top_k]]

    return [tuple(peak.tolist()) for peak in selected_peaks]


def _find_clahe_based_seed_locations(
    V_target: torch.Tensor,
    k_clahe_seeds: int,
    cfg: DynamicOpsConfig,
) -> List[Tuple[int, ...]]:
    """
    Find seed locations using CLAHE-equalized intensities as sampling probabilities.

    This approach samples from the target volume weighted by local perceptual importance
    (CLAHE-equalized intensity) rather than raw intensity, ensuring dim structures
    in dark regions receive fair sampling probability.

    Args:
        V_target: Target tensor to sample from
        k_clahe_seeds: Number of CLAHE-based seeds to generate
        cfg: Dynamic operations configuration

    Returns:
        List of seed location coordinates as tuples
    """
    if k_clahe_seeds <= 0:
        return []

    device = V_target.device
    shape = V_target.shape

    # Step 1: Apply CLAHE to target volume
    V_clahe = apply_clahe(
        V_target,
        tile_size=cfg.clahe_tile_size,
        clip_limit=cfg.clahe_clip_limit,
        nbins=cfg.clahe_nbins,
    )

    # Step 2: Normalize to [0, 1] for probability distribution
    V_min, V_max = V_clahe.min(), V_clahe.max()
    if V_max - V_min < 1e-12:
        # Uniform - use uniform sampling
        V_norm = torch.ones_like(V_clahe)
    else:
        V_norm = (V_clahe - V_min) / (V_max - V_min)

    # Step 3: Flatten and normalize to valid probability distribution
    V_flat = V_norm.reshape(-1)
    prob_sum = V_flat.sum()

    if prob_sum < 1e-12:
        return []  # No valid sampling distribution

    probabilities = V_flat / prob_sum

    # Step 4: Sample k_clahe_seeds locations with replacement
    try:
        sampled_indices = torch.multinomial(
            probabilities,
            num_samples=min(k_clahe_seeds, len(probabilities)),
            replacement=True,
        )
    except RuntimeError:
        # Handle edge case where probabilities are invalid
        return []

    # Step 5: Convert flat indices to nD coordinates
    # Use numpy's unravel_index for correct coordinate conversion
    flat_indices_np = sampled_indices.cpu().numpy()
    coords_np = np.unravel_index(flat_indices_np, shape)

    # Convert to list of tuples
    seed_locations = [
        tuple(int(coords_np[i][j]) for i in range(len(shape)))
        for j in range(len(flat_indices_np))
    ]

    return seed_locations


# Note: Complex covariance and amplitude estimation functions removed
# Replaced with ultra-simple approach: amplitude = residual[center], shape = isotropic


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
    from arbol import aprint, asection

    # Create coordinator for seamless optimizer state management
    coordinator = ModelOptimizerCoordinator(model, optimizer, scheduler)

    with torch.no_grad():
        # Compute residual image
        residual = V_target - V_pred

        # Track if we modified model topology
        topology_changed = False
        operations_performed = []

        # === STEP 1: Hybrid Seed Location Finding ===
        # Split budget between residual-based and CLAHE-based seeding
        k_residual = int(cfg.k_max_residuals * (1.0 - cfg.density_seeding_fraction))
        k_clahe = int(cfg.k_max_residuals * cfg.density_seeding_fraction)

        # Find residual-based peaks (error-driven)
        residual_peak_locations = _find_residual_peaks(
            residual, k_residual, cfg.nms_radius_vox
        )

        # Find CLAHE-based seeds (structure-driven)
        clahe_seed_locations = []
        if k_clahe > 0:
            clahe_seed_locations = _find_clahe_based_seed_locations(
                V_target, k_clahe, cfg
            )

        # Combine both types of seed locations
        peak_locations = residual_peak_locations + clahe_seed_locations

        if verbose and len(peak_locations) > 0:
            aprint(
                f"Found {len(residual_peak_locations)} residual peaks + "
                f"{len(clahe_seed_locations)} CLAHE seeds = {len(peak_locations)} total"
            )

        # Convergence guard: Apply only to residual peaks (not CLAHE seeds)
        # If strongest residual peak is below threshold, skip residual-based operations
        # but still allow CLAHE-based seeding for spatial coverage
        if len(residual_peak_locations) > 0:
            strongest_peak_coords = residual_peak_locations[
                0
            ]  # _find_residual_peaks returns sorted by strength
            strongest_peak_residual = torch.abs(residual[strongest_peak_coords]).item()

            if strongest_peak_residual < max_abs_error_threshold:
                if verbose:
                    aprint(
                        f"Convergence guard: strongest residual {strongest_peak_residual:.5f} < threshold {max_abs_error_threshold:.5f}"
                    )
                    if k_clahe > 0:
                        aprint(f"  → Skipping residual-based seeding, but allowing {len(clahe_seed_locations)} CLAHE seeds for coverage")
                    else:
                        aprint("  → Skipping all dynamic operations (no CLAHE seeding enabled)")
                # Remove residual peaks from consideration, keep only CLAHE seeds
                peak_locations = clahe_seed_locations
                if len(peak_locations) == 0:
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
                # Check if existing splat can be split vs needing new splat
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

                if (
                    influential_splat_idx != -1
                ):  # Existing Coverage - boost LR (splitting removed)
                    # Boost learning rate to "unfreeze" problematic splat
                    if influence_value >= cfg.boost_influence_threshold:
                        boosted_lr = _boost_splat_learning_rate(
                            optimizer, influential_splat_idx, current_lr, cfg
                        )
                        if verbose:
                            aprint(
                                f"      → Boosted LR for splat {influential_splat_idx}: {boosted_lr:.6f} (factor: {cfg.lr_boost_factor})"
                            )

                # No coverage or inadequate coverage - SEED new splat
                # Use new adaptive approach - always attempt seeding since convergence check passed
                if verbose:
                    aprint(
                        f"      → Attempting adaptive seeding (threshold = {local_residual * cfg.relative_contribution_factor:.6f})..."
                    )

                success = _seed_new_splat(
                    coordinator, model.shape, peak_location, residual, cfg, current_lr
                )
                if success:
                    topology_changed = True
                    operations_performed.append(f"Seeded splat at {peak_coords}")
                    if verbose:
                        aprint(f"      ✓ Successfully seeded splat at {peak_coords}")
                else:
                    if verbose:
                        aprint(
                            f"      ✗ Failed to seed splat at {peak_coords} - adaptive threshold check failed"
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

    # Compute influence of each splat at the given location
    influences = []
    for i in range(model.n_splats()):
        center = centers[i]
        L = Ls[i]
        amp = amps[i]

        # Compute Gaussian value at location: amp * exp(-0.5 * (x-mu)^T * Sigma^-1 * (x-mu))
        # Where Sigma^-1 = (L * L^T)^-1 = L^-T * L^-1
        diff = location - center

        # Solve L * y = diff to get y = L^-1 * diff (avoid explicit matrix inversion)
        try:
            y = torch.linalg.solve_triangular(L, diff, upper=False)
            squared_distance = torch.sum(y * y)
            influence = amp * torch.exp(-0.5 * squared_distance)
            influences.append(influence.item())
        except RuntimeError:
            # Handle singular matrices gracefully
            influences.append(0.0)

    if not influences:
        return -1, 0.0

    # Find splat with maximum influence
    max_influence = max(influences)
    max_idx = influences.index(max_influence)

    # Check if influence is significant enough
    if max_influence >= min_influence_threshold:
        return max_idx, max_influence
    else:
        return -1, 0.0


def _seed_new_splat(
    coordinator,
    shape: Tuple[int, ...],
    center: torch.Tensor,
    residual: torch.Tensor,
    cfg: DynamicOpsConfig,
    lr: float,
) -> bool:
    """
    Seed a new Gaussian splat with ultra-simple amplitude and shape estimation.

    Uses direct residual value at center for amplitude and isotropic covariance.
    Optimization will evolve optimal shapes during training.

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

        # ADAPTIVE THRESHOLD: Check amplitude relative to local residual
        amplitude_value = amplitude.item()
        adaptive_threshold = amplitude_value * cfg.relative_contribution_factor

        # Validate amplitude against adaptive threshold
        if amplitude_value < adaptive_threshold:
            return False

        # Add the new splat
        coordinator.add_splats(
            center.unsqueeze(0),  # (1, d)
            L.unsqueeze(0),  # (1, d, d)
            amplitude.unsqueeze(0),  # (1,)
            lr_new=lr,
        )

        return True

    except (RuntimeError, IndexError, ValueError):
        # Handle expected errors gracefully (device mismatches, invalid coordinates, etc.)
        return False
    except Exception as e:
        # Log unexpected errors for debugging
        print(f"Unexpected seeding error: {type(e).__name__}: {e}")
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
    model, splat_idx: int, V_target: torch.Tensor, max_abs_error_threshold: float
) -> bool:
    """
    Test if removing a splat would cause local residual to exceed convergence threshold.

    Uses local convergence-based validation: tests impact only in splat's influence region
    and ensures removal doesn't violate convergence criteria locally.

    Args:
        model: GaussianSplatModel
        splat_idx: Index of splat to test for removal
        V_target: Target tensor
        max_abs_error_threshold: Convergence threshold (always finite with auto-threshold)

    Returns:
        bool: True if splat can be safely removed, False if it should be kept
    """
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

    with torch.no_grad():
        # Get current parameters
        centers, Ls, amps, sharpness = model.current_params()

        if torch.sum(torch.ones(model.n_splats())) <= 1:
            return False  # Don't remove if it's the last splat

        # Define influence region for this splat (3σ ellipse)
        splat_center = centers[splat_idx]
        splat_L = Ls[splat_idx]
        influence_mask = _compute_splat_influence_region(
            splat_center, splat_L, V_target.shape, radius=3.0
        )

        # Skip if influence region is empty
        if not torch.any(influence_mask):
            return True  # Can safely remove if no influence

        # Create temporary parameters without the target splat
        keep_mask = torch.ones(
            model.n_splats(), dtype=torch.bool, device=centers.device
        )
        keep_mask[splat_idx] = False

        temp_centers = centers[keep_mask]
        temp_Ls = Ls[keep_mask]
        temp_amps = amps[keep_mask]
        temp_sharpness = sharpness[keep_mask]

        # Render full prediction without the target splat
        pred_without_splat = render_gaussians(
            V_target.shape,
            temp_centers,
            temp_Ls,
            temp_amps,
            temp_sharpness,
            truncate=3.0,
        )

        # Extract local regions for comparison
        pred_local_without = pred_without_splat[influence_mask]
        target_local = V_target[influence_mask]

        # Compute local residual without the splat
        local_residual = torch.abs(pred_local_without - target_local)
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
        from arbol import aprint

        aprint(
            f"    Pruning analysis: testing {len(candidates)} least important splats ({cfg.pruning_percentile}% of {model.n_splats()})"
        )

    # Step 3: Test each candidate for local removal impact
    # Note: No longer use global max error - each splat tested for local convergence impact

    removable_splats = []
    for splat_idx in candidates:
        can_remove = _test_local_removal_impact(
            model, splat_idx, V_target, max_abs_error_threshold
        )
        if can_remove:
            removable_splats.append(splat_idx)
            if verbose:
                from arbol import aprint

                aprint(
                    f"      → Splat {splat_idx} marked for removal (importance={importance[splat_idx]:.6f})"
                )
        elif verbose:
            from arbol import aprint

            aprint(f"      → Splat {splat_idx} kept (would degrade quality)")

    return removable_splats
