# dynamic_ops.py

"""
Dynamic Gaussian Splat Operations

This module implements convergence-driven dynamic operations for adaptive Gaussian splatting.
Dynamic operations address reconstruction deficiencies by analyzing the residual image
(target - prediction) to identify where the current representation fails to meet convergence criteria.

## Core Philosophy

The approach focuses on three core operations driven by convergence requirements:
- **Seeding**: Create new splats where residual exceeds convergence thresholds
- **Splitting**: Divide problematic splats that are too large or elongated
- **Pruning**: Remove splats that contribute minimally to reconstruction quality

## Implementation Approach

### Step 1: Residual Peak Analysis
- Compute residual image: residual = V_target - V_pred
- Find k strongest peaks with spatial exclusion (non-maximum suppression)
- Apply convergence guard: skip all operations if strongest residual < convergence threshold

### Step 2: Convergence-Based Splat Operations
- For each peak, determine if coverage is sufficient using convergence criteria
- **Seeding**: Coverage insufficient (residual > convergence threshold or no threshold set)
- **Splitting**: Existing splat has influence but geometric criteria suggest refinement needed
- **Adaptive thresholds**: Amplitude validation scales with local residual magnitude

### Step 3: Global Pruning Analysis
- Independent of residual peaks, analyze all splats for minimal contribution
- Remove stagnant splats with low learning rates and minimal impact
"""

from __future__ import annotations

from typing import List, Tuple

import torch

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
    """

    def __init__(self):
        # Scheduling
        self.step_every: int = 50  # Run operations every N iterations

        # Step 1: Residual Peak Analysis
        self.k_max_residuals: int = 10  # Number of strongest residual peaks to analyze
        self.nms_radius_vox: float = 2.0  # Minimum distance between detected peaks

        # Step 2: Adaptive Operations
        self.min_contribution_threshold: float = 0.05  # Legacy fixed threshold for influence detection
        self.relative_contribution_factor: float = 0.1  # Adaptive threshold: fraction of local residual

        # Step 3: Global Pruning
        self.learning_rate_threshold: float = 1e-6  # LR below which splats are "stagnant"

        # Seeding parameters
        self.init_sigma_vox: float = 1.5  # Initial covariance for new splats

        # Splitting parameters
        self.split_size_threshold: float = 3.0  # sqrt(λ_max) threshold for splitting
        self.split_elongation_threshold: float = 4.0  # λ_max/λ_min ratio for splitting


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
            padding=kernel_size // 2
        )[0, 0]
    elif d == 3:
        max_pooled = torch.nn.functional.max_pool3d(
            residual_abs[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
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








def _estimate_covariance_from_local_residual(
    residual: torch.Tensor, center: torch.Tensor, window_size: int = 7
) -> torch.Tensor:
    """
    Estimate covariance matrix by analyzing local residual structure.

    Args:
        residual: Full residual image
        center: Peak location (d,)
        window_size: Size of analysis window around peak

    Returns:
        L: Lower triangular matrix for covariance (d, d)
    """
    d = len(center)
    device = center.device

    # Extract local window around peak
    half_window = window_size // 2
    center_int = torch.round(center).long()

    # Get window bounds (clamped to image bounds)
    bounds = []
    for i in range(d):
        start = max(0, center_int[i] - half_window)
        end = min(residual.shape[i], center_int[i] + half_window + 1)
        bounds.append((start, end))

    # Extract local residual window
    if d == 2:
        local_residual = residual[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]
    elif d == 3:
        local_residual = residual[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]
    else:
        # For higher dimensions, use a more general approach
        local_residual = residual  # Fallback to full image

    # Create coordinate grids for the local window
    coords = []
    for i, (start, end) in enumerate(bounds):
        coords.append(torch.arange(start, end, dtype=torch.float32, device=device) - center[i])

    if d == 2:
        y_coords, x_coords = torch.meshgrid(coords[0], coords[1], indexing='ij')
        coord_stack = torch.stack([y_coords, x_coords], dim=-1)  # (H, W, 2)
    elif d == 3:
        z_coords, y_coords, x_coords = torch.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        coord_stack = torch.stack([z_coords, y_coords, x_coords], dim=-1)  # (D, H, W, 3)
    else:
        # Fallback for higher dimensions
        return torch.eye(d, device=device) * 2.0  # Default to reasonable size

    # Flatten for analysis - use ABSOLUTE residual to avoid sign issues
    flat_coords = coord_stack.reshape(-1, d)  # (N, d)
    flat_weights = torch.abs(local_residual).flatten()  # (N,) - ABSOLUTE values

    # Remove zero weights
    non_zero_mask = flat_weights > 1e-8
    if torch.sum(non_zero_mask) < 3:  # Need at least 3 points for covariance
        return torch.eye(d, device=device) * 1.5  # Fallback to default

    flat_coords = flat_coords[non_zero_mask]
    flat_weights = flat_weights[non_zero_mask]

    # Normalize weights
    flat_weights = flat_weights / torch.sum(flat_weights)

    # Compute weighted covariance matrix
    weighted_mean = torch.sum(flat_coords * flat_weights.unsqueeze(1), dim=0)  # (d,)
    centered_coords = flat_coords - weighted_mean.unsqueeze(0)  # (N, d)

    # Weighted covariance: C = sum(w_i * (x_i - mean) * (x_i - mean)^T)
    weighted_centered = centered_coords * flat_weights.unsqueeze(1).sqrt()  # (N, d)
    covariance = torch.mm(weighted_centered.T, weighted_centered)  # (d, d)

    # Add regularization to prevent singular matrices
    covariance += torch.eye(d, device=device) * 0.25

    # Ensure minimum size
    eigenvals, eigenvecs = torch.linalg.eigh(covariance)
    eigenvals = torch.clamp(eigenvals, min=0.8)  # Minimum std of ~0.9 voxels
    covariance = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T

    # Return Cholesky decomposition (lower triangular L such that LL^T = Σ)
    try:
        L = torch.linalg.cholesky(covariance)
        return L
    except RuntimeError:
        # Fallback if Cholesky fails
        return torch.eye(d, device=device) * 1.5


def _estimate_amplitude_from_residual(
    shape: Tuple[int, ...], center: torch.Tensor, L: torch.Tensor, residual: torch.Tensor, truncate: float = 3.0
) -> torch.Tensor:
    """
    Estimate amplitude for a Gaussian splat using least-squares fitting.

    Args:
        shape: Shape of the target volume/image
        center: Center of the Gaussian (d,)
        L: Lower triangular matrix for covariance (d, d)
        residual: Current residual image
        truncate: Truncation parameter for rendering

    Returns:
        Estimated amplitude as tensor scalar
    """
    from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians

    centers = center[None]  # (1, d)
    Ls = L[None]  # (1, d, d)
    amps = torch.ones((1,), device=center.device, dtype=torch.float32)

    # Render unit-amplitude Gaussian
    g = render_gaussians(shape, centers, Ls, amps, truncate=truncate)

    # Least squares: a = <|residual|, g> / <g, g> - use ABSOLUTE residual to avoid sign issues
    abs_residual = torch.abs(residual)
    numerator = torch.sum(abs_residual * g)
    denominator = torch.sum(g * g) + 1e-12
    amplitude = torch.clamp(numerator / denominator, min=0.0)

    return amplitude


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
    3. Global Pruning Analysis: Remove stagnant splats

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

        # === STEP 1: Residual Peak Analysis ===
        peak_locations = _find_residual_peaks(
            residual, cfg.k_max_residuals, cfg.nms_radius_vox
        )

        if verbose and len(peak_locations) > 0:
            aprint(f"Found {len(peak_locations)} residual peaks for analysis")

        # Convergence guard: If strongest residual peak is below convergence threshold, skip all operations
        # Only apply convergence guard if a finite threshold is set (not None/inf)
        if len(peak_locations) > 0 and max_abs_error_threshold != float('inf'):
            strongest_peak_coords = peak_locations[0]  # _find_residual_peaks returns sorted by strength
            strongest_peak_residual = torch.abs(residual[strongest_peak_coords]).item()

            if strongest_peak_residual < max_abs_error_threshold:
                if verbose:
                    aprint(f"Convergence guard: strongest residual {strongest_peak_residual:.5f} < threshold {max_abs_error_threshold:.5f}, skipping all dynamic operations")
                return optimizer, scheduler, False

        # === STEP 2: Adaptive Splat Operations ===
        for peak_coords in peak_locations:
            peak_location = torch.tensor(peak_coords, dtype=torch.float32, device=device)
            local_residual = torch.abs(residual[peak_coords]).item()

            # CONVERGENCE-BASED DETECTION: Is coverage sufficient?
            coverage_sufficient = _is_coverage_sufficient(local_residual, max_abs_error_threshold)

            if verbose:
                if max_abs_error_threshold != float('inf'):
                    aprint(f"    Peak {peak_coords}: residual {local_residual:.5f} vs threshold {max_abs_error_threshold:.5f} → coverage {'sufficient' if coverage_sufficient else 'insufficient'}")
                else:
                    aprint(f"    Peak {peak_coords}: residual {local_residual:.5f}, no threshold → always seed")

            if not coverage_sufficient:  # Coverage is insufficient - need action
                # Check if existing splat can be split vs needing new splat
                influential_splat_idx, influence_value = _find_splat_with_significant_influence_at_location(
                    model, peak_location, min_influence_threshold=cfg.min_contribution_threshold
                )

                if verbose:
                    if influential_splat_idx != -1:
                        aprint(f"      → Existing influence: splat {influential_splat_idx} has {influence_value:.6f}")
                    else:
                        aprint(f"      → No existing influence above {cfg.min_contribution_threshold}")

                if influential_splat_idx != -1:  # Existing Coverage - Check if SPLITTING is appropriate
                    if verbose:
                        aprint(f"      → Checking if splat {influential_splat_idx} should be split...")

                    if _should_split_splat(
                        model, influential_splat_idx, local_residual,
                        cfg, max_abs_error_threshold
                    ):
                        if verbose:
                            aprint(f"      → Attempting to split splat {influential_splat_idx}")
                        success = _split_problematic_splat(
                            coordinator, model, influential_splat_idx, cfg
                        )
                        if success:
                            topology_changed = True
                            operations_performed.append(f"Split splat {influential_splat_idx} at {peak_coords}")
                            if verbose:
                                aprint(f"      ✓ Successfully split splat {influential_splat_idx}")
                            continue
                        elif verbose:
                            aprint(f"      ✗ Failed to split splat {influential_splat_idx}")
                    elif verbose:
                        aprint(f"      → Splat {influential_splat_idx} does not meet splitting criteria")

                # Either no existing coverage OR splitting failed/inappropriate - SEED new splat
                # Use new adaptive approach - always attempt seeding since convergence check passed
                if verbose:
                    aprint(f"      → Attempting adaptive seeding (threshold = {local_residual * cfg.relative_contribution_factor:.6f})...")

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
                        aprint(f"      ✗ Failed to seed splat at {peak_coords} - adaptive threshold check failed")
            elif verbose:
                aprint("    → Coverage sufficient, no action needed")

        # === STEP 3: Global Pruning Analysis ===
        if model.n_splats() > 0:
            stagnant_splats = _find_stagnant_splats(
                model, optimizer, scheduler, residual, cfg
            )

            if len(stagnant_splats) > 0:
                # Create keep mask (True for splats to keep)
                keep_mask = torch.ones(model.n_splats(), dtype=torch.bool, device=device)
                for splat_idx in stagnant_splats:
                    keep_mask[splat_idx] = False

                coordinator.prune_splats(keep_mask)
                topology_changed = True
                operations_performed.append(f"Pruned {len(stagnant_splats)} stagnant splats")

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


def _is_coverage_sufficient(local_residual: float, max_abs_error_threshold: float) -> bool:
    """
    Determine if coverage at a location is sufficient based on convergence criteria.

    Args:
        local_residual: Absolute residual value at the location
        max_abs_error_threshold: Convergence threshold (inf if no threshold set)

    Returns:
        bool: True if coverage is sufficient, False if more coverage is needed
    """
    if max_abs_error_threshold == float('inf'):
        # No convergence threshold set - always try to minimize error further
        return False
    else:
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

    centers, Ls, amps = model.current_params()

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
    coordinator, shape: Tuple[int, ...], center: torch.Tensor,
    residual: torch.Tensor, cfg: DynamicOpsConfig, lr: float
) -> bool:
    """
    Seed a new Gaussian splat at the specified location with adaptive sizing.

    Returns:
        bool: True if seeding was successful
    """
    try:
        # Estimate covariance from local residual structure (ADAPTIVE SIZING)
        L = _estimate_covariance_from_local_residual(residual, center, window_size=7)

        # Estimate amplitude
        amplitude = _estimate_amplitude_from_residual(shape, center, L, residual)

        # ADAPTIVE THRESHOLD: Check amplitude relative to local residual
        amplitude_value = amplitude.item()
        local_residual_value = torch.abs(residual[tuple(torch.round(center).long())]).item()
        adaptive_threshold = local_residual_value * cfg.relative_contribution_factor

        # Validate amplitude against adaptive threshold
        if amplitude_value < adaptive_threshold:
            return False

        # Add the new splat
        coordinator.add_splats(
            center.unsqueeze(0),  # (1, d)
            L.unsqueeze(0),       # (1, d, d)
            amplitude.unsqueeze(0), # (1,)
            lr_new=lr
        )

        return True

    except (RuntimeError, IndexError, ValueError) as e:
        # Handle expected errors gracefully (device mismatches, invalid coordinates, etc.)
        return False
    except Exception as e:
        # Log unexpected errors for debugging
        print(f"Unexpected seeding error: {type(e).__name__}: {e}")
        return False


def _should_split_splat(
    model, splat_idx: int, local_residual: float,
    cfg: DynamicOpsConfig, max_abs_error_threshold: float
) -> bool:
    """
    Determine if a splat should be split based on size and residual criteria.
    """
    centers, Ls, amps = model.current_params()

    # Check size criterion (principal axis length)
    Sigma = Ls[splat_idx] @ Ls[splat_idx].T
    eigvals = torch.linalg.eigvals(Sigma).real
    lambda_max = torch.max(eigvals)
    lambda_min = torch.min(eigvals)

    principal_axis_length = torch.sqrt(lambda_max)
    elongation_ratio = lambda_max / (lambda_min + 1e-12)

    # Check criteria
    size_criterion = principal_axis_length > cfg.split_size_threshold
    elongation_criterion = elongation_ratio > cfg.split_elongation_threshold

    # Residual criterion: if no convergence threshold set, use a reasonable default
    if max_abs_error_threshold == float('inf'):
        residual_criterion = local_residual > cfg.min_contribution_threshold
    else:
        residual_criterion = local_residual > 0.5 * max_abs_error_threshold

    return size_criterion and elongation_criterion and residual_criterion


def _split_problematic_splat(
    coordinator, model, splat_idx: int, cfg: DynamicOpsConfig
) -> bool:
    """
    Split a problematic splat along its principal axis.

    Returns:
        bool: True if splitting was successful
    """
    try:
        centers, Ls, amps = model.current_params()

        parent_center = centers[splat_idx]
        parent_L = Ls[splat_idx]
        parent_amp = amps[splat_idx]

        # Compute principal eigenvector
        Sigma = parent_L @ parent_L.T
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        v1 = eigvecs[:, -1]  # Principal eigenvector
        lambda_max = eigvals[-1]

        # Compute offset along principal axis
        offset = 0.3 * torch.sqrt(lambda_max) * v1

        # Create two child splats
        child1_center = parent_center - offset
        child2_center = parent_center + offset

        # Scale down covariances
        child_L = 0.6 * parent_L

        # Distribute amplitude
        child_amp = 0.6 * parent_amp

        # Remove parent splat
        keep_mask = torch.ones(model.n_splats(), dtype=torch.bool, device=parent_center.device)
        keep_mask[splat_idx] = False
        coordinator.prune_splats(keep_mask)

        # Add child splats
        child_centers = torch.stack([child1_center, child2_center], dim=0)
        child_Ls = torch.stack([child_L, child_L], dim=0)
        child_amps = torch.stack([child_amp, child_amp], dim=0)

        coordinator.add_splats(child_centers, child_Ls, child_amps)

        return True

    except (RuntimeError, IndexError, ValueError):
        # Handle expected errors gracefully (invalid indices, device issues, etc.)
        return False
    except Exception as e:
        # Log unexpected errors for debugging
        print(f"Unexpected splitting error: {type(e).__name__}: {e}")
        return False


def _find_stagnant_splats(
    model, optimizer, scheduler, residual: torch.Tensor, cfg: DynamicOpsConfig
) -> List[int]:
    """
    Find stagnant splats that contribute minimally to reconstruction quality.

    Returns:
        List of splat indices to remove
    """
    if model.n_splats() == 0:
        return []

    # Get effective learning rates from optimizer
    effective_lrs = optimizer.get_effective_learning_rates()

    # Find splats with low learning rates (stagnant)
    stagnant_indices = []

    for splat_idx in range(model.n_splats()):
        current_lr = effective_lrs[splat_idx]

        # Learning rate criterion
        if current_lr < cfg.learning_rate_threshold:
            # Additional checks for contribution
            if _has_minimal_contribution(model, splat_idx, residual, cfg):
                stagnant_indices.append(splat_idx)

    return stagnant_indices


def _has_minimal_contribution(
    model, splat_idx: int, residual: torch.Tensor, cfg: DynamicOpsConfig
) -> bool:
    """
    Check if a splat has minimal contribution to reconstruction quality.
    """
    centers, Ls, amps = model.current_params()

    # Get splat parameters
    center = centers[splat_idx]
    amp = amps[splat_idx]

    # Compute maximum possible change this splat could make
    # This is a rough approximation
    max_gaussian_value = amp  # Maximum at center

    # Estimate local residual in splat's 2σ region
    # For simplicity, just sample at center
    center_coords = torch.round(center).long()

    # Clamp coordinates to valid range
    for d in range(len(center_coords)):
        center_coords[d] = torch.clamp(center_coords[d], 0, residual.shape[d] - 1)

    local_residual = torch.abs(residual[tuple(center_coords)]).item()

    # Check if maximum possible change is less than thresholds
    max_change = max_gaussian_value.item()

    minimal_contrib = max_change < cfg.min_contribution_threshold
    less_than_residual = max_change < 0.1 * local_residual

    return minimal_contrib and less_than_residual
