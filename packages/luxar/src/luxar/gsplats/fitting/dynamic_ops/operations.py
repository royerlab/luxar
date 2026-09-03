# operations.py
"""Fixed-pool splat relocation for adaptive Gaussian splatting.

Performance-optimized implementation using batched tensor operations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from arbol import aprint, asection

from luxar.gsplats.fitting.dynamic_ops.config import DynamicOpsConfig
from luxar.gsplats.fitting.dynamic_ops.peak_finding import _find_residual_peaks
from luxar.gsplats.models.utils.inverse_softplus import (
    stable_inverse_softplus_torch,
)


class RecentlyRelocatedTracker:
    """Track recently relocated splats to avoid immediate re-selection.

    This prevents the critical bug where the same weak splats get relocated
    repeatedly while many splats remain untouched.

    The cooldown mechanism ensures that after a splat is relocated, it won't
    be selected for relocation again until it has had time to be optimized
    at its new location.

    Performance: Uses GPU tensors for bulk filtering operations.
    """

    def __init__(self, n_splats: int, cooldown_steps: int = 3, device: str = "cpu"):
        """Initialize tracker.

        Args:
            n_splats: Total number of splats in the model
            cooldown_steps: Number of dynamic ops steps to wait before
                allowing a splat to be relocated again. Default is 3 steps.
            device: Device to store tensors on
        """
        self.cooldown_steps = cooldown_steps
        self.n_splats = n_splats
        self.device = device

        # Vectorized storage: last relocation step for each splat
        # -1 means never relocated
        self.last_relocation_step = torch.full(
            (n_splats,), -1, dtype=torch.long, device=device
        )

        self.current_step = 0
        self.total_relocations = 0
        self.unique_splats_relocated = 0

    def mark_relocated_batch(self, splat_indices: torch.Tensor) -> None:
        """Mark multiple splats as recently relocated (vectorized).

        Args:
            splat_indices: Tensor of splat indices that were relocated
        """
        if len(splat_indices) == 0:
            return

        # Track which splats are being relocated for the first time
        first_time_mask = self.last_relocation_step[splat_indices] == -1
        self.unique_splats_relocated += int(first_time_mask.sum().item())

        # Update last relocation step for all relocated splats
        self.last_relocation_step[splat_indices] = self.current_step
        self.total_relocations += len(splat_indices)

    def filter_eligible_splats(self, candidate_indices: torch.Tensor) -> torch.Tensor:
        """Filter candidates to only those eligible for relocation (vectorized).

        Args:
            candidate_indices: Tensor of candidate splat indices to check

        Returns:
            Tensor of indices that are eligible for relocation (on same device)
        """
        if len(candidate_indices) == 0:
            return candidate_indices

        # Vectorized eligibility check
        last_steps = self.last_relocation_step[candidate_indices]
        steps_since_relocation = self.current_step - last_steps

        # Never relocated (step=-1) OR cooldown expired
        eligible_mask = (last_steps == -1) | (
            steps_since_relocation >= self.cooldown_steps
        )

        return candidate_indices[eligible_mask]

    def advance_step(self) -> None:
        """Advance to the next dynamic ops step."""
        self.current_step += 1

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about relocations.

        Returns:
            Dictionary with statistics:
            - total_relocations: Total number of relocations performed
            - unique_splats: Number of unique splats that have been relocated
            - currently_on_cooldown: Number of splats currently in cooldown
        """
        # Only count splats that have been relocated (step != -1)
        relocated_mask = self.last_relocation_step >= 0
        steps_since = self.current_step - self.last_relocation_step

        # On cooldown: relocated AND within cooldown window
        currently_on_cooldown = int(
            (relocated_mask & (steps_since < self.cooldown_steps)).sum().item()
        )

        return {
            "total_relocations": self.total_relocations,
            "unique_splats": self.unique_splats_relocated,
            "currently_on_cooldown": currently_on_cooldown,
        }


def _resolve_operation_seed(
    seed: Optional[int], relocation_tracker: Optional[RecentlyRelocatedTracker]
) -> Optional[int]:
    if seed is None or relocation_tracker is None:
        return seed
    return seed + relocation_tracker.current_step


def apply_dynamic_operations(
    model: Any,
    V_target: torch.Tensor,
    V_pred: torch.Tensor,
    cfg: DynamicOpsConfig,
    max_abs_error_threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    relocation_tracker: Optional[RecentlyRelocatedTracker] = None,
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
    3. Filter out recently relocated splats (cooldown mechanism)
    4. Match weak splats to peaks (avoiding already-covered locations)
    5. Relocate matched splats and reset their optimizer state

    Args:
        model: GaussianSplatModel with current splat parameters
        V_target: Target tensor to reconstruct
        V_pred: Current prediction tensor from model
        cfg: Dynamic operations configuration
        max_abs_error_threshold: Convergence threshold
        optimizer: Optional optimizer (for state reset). If provided, optimizer
            state (momentum, variance) will be reset for relocated splats.
        relocation_tracker: Optional tracker for cooldown mechanism. If provided,
            prevents immediate re-relocation of recently moved splats.
        verbose: Whether to print detailed progress information

    Returns:
        bool: True if any splats were relocated
    """
    with torch.no_grad():
        operation_seed = _resolve_operation_seed(cfg.seed, relocation_tracker)

        # === Cache model parameters once (avoid repeated current_params() calls) ===
        centers, Ls, amps = model.current_params()

        # Compute residual image
        residual = V_target - V_pred

        # === STEP 1: Find Residual Peaks ===
        peak_locations = _find_residual_peaks(
            residual,
            cfg.k_max_residuals,
            cfg.nms_radius_vox,
            enable_tiled=cfg.enable_tiled_seeding,
            num_tiles_per_dim=cfg.num_tiles_per_dim
            if cfg.num_tiles_per_dim is not None
            else 8,
            seed=operation_seed,
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
        strongest_peak_residual = residual[tuple(strongest_peak_coords.long())].item()

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
        weak_splat_indices = _select_weak_splats(
            importance,
            centers,
            residual,
            cfg.relocation_percentile,
            relocation_tracker,
            seed=operation_seed,
        )

        if len(weak_splat_indices) == 0:
            if verbose:
                aprint("No weak splats found - skipping relocation")
            return False

        if verbose:
            aprint(
                f"Identified {len(weak_splat_indices)} weak splats "
                f"(bottom {cfg.relocation_percentile}% by importance)"
            )
            if relocation_tracker is not None:
                stats = relocation_tracker.get_statistics()
                aprint(
                    f"  Relocation history: {stats['unique_splats']} unique splats relocated "
                    f"({stats['total_relocations']} total), "
                    f"{stats['currently_on_cooldown']} currently on cooldown"
                )

        # === STEP 3: Match Weak Splats to Peaks (vectorized) ===
        # Returns tensors (weak_splat_indices, peak_indices)
        matched_splat_indices, matched_peak_indices = _match_weak_splats_to_peaks_batch(
            centers,
            Ls,
            amps,
            weak_splat_indices,
            peak_locations,
            cfg.min_contribution_threshold,
            cfg.max_relocations_per_step,
            cfg.enable_coverage_check,
            verbose,
        )

        n_relocated = len(matched_splat_indices)

        if n_relocated == 0:
            if verbose:
                aprint("No valid relocation matches found")
            return False

        # === STEP 4: Relocate Splats (BATCHED) ===
        # Index peak coordinates by matched indices (already a tensor)
        peak_coords_tensor = peak_locations.to(
            device=centers.device, dtype=torch.float32
        )[matched_peak_indices]

        # Perform all relocations in one batched operation (with optimizer state reset)
        _relocate_splats_batch(
            model, matched_splat_indices, peak_coords_tensor, residual, cfg, optimizer
        )

        # Mark splats as relocated in tracker (already a tensor!)
        if relocation_tracker is not None:
            relocation_tracker.mark_relocated_batch(matched_splat_indices)

        # Verbose logging (print details but operation already complete)
        if verbose and n_relocated > 0:
            with asection(f"Relocated {n_relocated} splats"):
                # Show sample of relocations (not all to avoid spam)
                sample_size = min(5, n_relocated)
                for i in range(sample_size):
                    splat_idx = int(matched_splat_indices[i].item())
                    peak_idx = int(matched_peak_indices[i].item())
                    peak_coords = tuple(peak_locations[peak_idx].long().tolist())
                    aprint(
                        f"Splat {splat_idx} → {peak_coords} "
                        f"(importance was {importance[splat_idx]:.6f})"
                    )
                if n_relocated > sample_size:
                    aprint(f"... and {n_relocated - sample_size} more")

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
    importance: torch.Tensor,
    centers: torch.Tensor,
    residual: torch.Tensor,
    relocation_percentile: float,
    relocation_tracker: Optional[RecentlyRelocatedTracker] = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Select weak splats that are safe to relocate.

    Strategy:
    1. Sort by importance (ascending)
    2. Filter OUT splats at high-residual locations (they're fighting, doing work)
    3. Filter OUT splats on cooldown (recently relocated)
    4. Return the weakest N that pass all filters

    This ensures we only relocate truly useless splats - those that are weak AND
    at locations with low residual (not contributing to high-error regions) AND
    not recently relocated (have had time to be optimized).

    Args:
        importance: Importance values for all splats
        centers: Splat center positions, shape (N, d)
        residual: Residual image (target - prediction)
        relocation_percentile: Percentage of least important splats to consider
        relocation_tracker: Optional tracker for cooldown filtering
        seed: RNG seed for the residual-percentile sample. None uses PyTorch's
            global RNG stream; a fixed seed uses a private generator and makes
            a re-fit bit-reproducible.

    Returns:
        Tensor of splat indices sorted by importance (weakest first)
    """
    n_splats = len(importance)
    n_candidates = max(1, int(n_splats * relocation_percentile / 100.0))

    # Get residual value at each splat's center location
    center_coords = torch.round(centers).long()

    # Vectorized clamping to valid range (all dimensions at once)
    residual_shape = torch.tensor(
        residual.shape, device=centers.device, dtype=torch.long
    )
    center_coords = torch.maximum(center_coords, torch.zeros_like(center_coords))
    center_coords = torch.minimum(center_coords, residual_shape - 1)

    residual_at_centers = residual[tuple(center_coords.T)]

    # Sort ALL splats by importance (ascending = weakest first)
    sorted_indices = torch.argsort(importance)

    # Filter strategy: Only relocate splats at LOW-residual locations
    # High residual = splat is fighting a tough region, leave it there!
    # Use 25th percentile of positive residuals as threshold
    # OPTIMIZED: Sample residual instead of using all voxels (10-100x faster)
    residual_positive = torch.clamp(residual, min=0)
    if residual_positive.max() > 0:
        # Sample up to 10K voxels instead of all voxels (e.g., 256^3 = 16M)
        # This is statistically sufficient for percentile estimation
        sample_size = min(10000, residual.numel())
        if sample_size < residual.numel():
            # Random sampling (optimized)
            # Use randint instead of randperm (2-3x faster: generates only sample_size random numbers)
            # Use reshape instead of flatten (avoids copy when tensor is contiguous)
            # Use a private CPU generator for explicit seeds so the whole
            # dynamic-ops path is reproducible without perturbing global state.
            # With seed=None, preserve the caller-controlled global RNG stream.
            generator = None
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(seed)
            indices = torch.randint(
                0, residual.numel(), (sample_size,), generator=generator
            )
            indices = indices.to(residual.device)
            sampled = torch.clamp(residual.reshape(-1)[indices], min=0)
            # Only compute quantile on positive values from sample
            positive_sample = sampled[sampled > 0]
            if len(positive_sample) > 0:
                residual_threshold = torch.quantile(positive_sample, 0.25).item()
            else:
                residual_threshold = 0.0
        else:
            # Small volume, use all voxels
            residual_threshold = torch.quantile(
                residual_positive[residual_positive > 0].flatten(), 0.25
            ).item()
    else:
        residual_threshold = 0.0

    # VECTORIZED: Filter candidates without Python loop
    # Get residual values at sorted splat locations (all on GPU, no sync)
    residual_at_sorted = residual_at_centers[sorted_indices]

    # Boolean mask: True for splats at low-residual locations (safe to relocate)
    low_residual_mask = residual_at_sorted <= residual_threshold

    # Use cumulative sum to find first n_candidates that pass filter
    # This replaces the Python loop with a single vectorized operation
    cumsum = torch.cumsum(low_residual_mask.int(), dim=0)
    selection_mask = low_residual_mask & (cumsum <= n_candidates)

    # Get selected indices (still on GPU, no sync yet)
    candidates_tensor = sorted_indices[selection_mask]

    # If we didn't get enough candidates (all weak splats are fighting errors),
    # fallback to just taking the weakest by importance
    if len(candidates_tensor) < max(1, n_candidates // 2):
        candidates_tensor = sorted_indices[:n_candidates]

    # === COOLDOWN FILTER: Remove recently relocated splats (vectorized) ===
    if relocation_tracker is not None:
        candidates_tensor = relocation_tracker.filter_eligible_splats(candidates_tensor)

        # If cooldown filtering removed too many, expand candidate pool
        if len(candidates_tensor) < max(1, n_candidates // 2):
            # Try expanding to 2x candidates before filtering
            expanded_candidates = sorted_indices[: n_candidates * 2]
            expanded_low_residual = low_residual_mask[: n_candidates * 2]
            expanded_filtered = expanded_candidates[expanded_low_residual]
            expanded_eligible = relocation_tracker.filter_eligible_splats(
                expanded_filtered
            )

            if len(expanded_eligible) > len(candidates_tensor):
                candidates_tensor = expanded_eligible[:n_candidates]

    return candidates_tensor


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

    # MEMORY OPTIMIZATION: Only process active (peak, splat) pairs
    # This avoids creating huge (P×N×d×d) tensors for distant splats
    # For P=100 peaks, N=10K splats, d=3: reduces ~100MB to ~10MB
    active_pairs = torch.nonzero(active_mask, as_tuple=False)  # (M, 2) where M << P*N
    peak_indices = active_pairs[:, 0]  # (M,)
    splat_indices = active_pairs[:, 1]  # (M,)

    # Gather only active pairs' data
    active_centers = centers[splat_indices]  # (M, d)
    active_Ls = Ls[splat_indices]  # (M, d, d)
    active_amps = amps[splat_indices]  # (M,)
    active_peaks = peak_locations[peak_indices]  # (M, d)

    # Compute for M pairs instead of P*N pairs
    diff_active = active_peaks - active_centers  # (M, d)
    y = torch.linalg.solve_triangular(
        active_Ls, diff_active.unsqueeze(-1), upper=False
    )  # (M, d, 1)
    squared_dist = (y.squeeze(-1) ** 2).sum(dim=-1)  # (M,)
    influences_active = active_amps * torch.exp(-0.5 * squared_dist)  # (M,)

    # Scatter max influences back to peaks (vectorized)
    # For each peak, take the maximum influence from all its active splats
    max_influences = torch.zeros(P, device=device)

    # Use scatter_reduce for efficient O(M) operation instead of O(P×M) loop
    # PyTorch >= 1.12 has scatter_reduce_, older versions fall back to loop
    if hasattr(torch, "scatter_reduce") or hasattr(max_influences, "scatter_reduce_"):
        # Efficient vectorized scatter
        max_influences.scatter_reduce_(
            0, peak_indices, influences_active, reduce="amax", include_self=False
        )
    else:
        # Fallback for older PyTorch versions
        for p in range(P):
            peak_mask = peak_indices == p
            if peak_mask.any():
                max_influences[p] = influences_active[peak_mask].max()

    return max_influences


def _match_weak_to_peaks_direct(
    weak_indices: torch.Tensor,
    uncovered_peak_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct pairing: weakest splat → highest error peak.

    Both inputs are already sorted (weak_indices by importance ascending,
    uncovered_peak_indices by residual descending), so we simply zip them
    together. This is simpler and more logical than distance-based matching
    since relocated splats are completely reset anyway.

    Args:
        weak_indices: Indices of weak splats, sorted by importance (ascending)
        uncovered_peak_indices: Indices of uncovered peaks, sorted by residual (descending)

    Returns:
        Tuple of (weak_splat_indices_tensor, peak_indices_tensor) - both on GPU
    """
    # Direct pairing: i-th weakest splat → i-th highest error peak
    n_matches = min(len(weak_indices), len(uncovered_peak_indices))

    # Keep as tensors (no Python loop!)
    return weak_indices[:n_matches], uncovered_peak_indices[:n_matches]


def _match_weak_splats_to_peaks_batch(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    weak_splat_indices: torch.Tensor,
    peak_locations: torch.Tensor,
    min_contribution_threshold: float,
    max_relocations: int | None,
    enable_coverage_check: bool,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match weak splats to residual peaks using vectorized operations.

    Algorithm:
    1. (Optional) Batch compute coverage for all peaks at once
    2. (Optional) Identify uncovered peaks (influence < threshold)
    3. Batch match uncovered peaks to weak splats

    Args:
        centers: Splat centers, shape (N, d)
        Ls: Cholesky factors, shape (N, d, d)
        amps: Amplitudes, shape (N,)
        weak_splat_indices: Indices of weak splats eligible for relocation
        peak_locations: Peak coordinates sorted by residual magnitude (descending)
        min_contribution_threshold: Minimum influence to consider a peak "covered"
        max_relocations: Maximum number of relocations to return (None = no limit)
        enable_coverage_check: If True, skip peaks already covered by non-weak splats.
            If False, relocate to all high-residual peaks regardless of existing coverage.
        verbose: Whether to print debug info

    Returns:
        Tuple of (splat_indices_tensor, peak_indices_tensor) - both on GPU
    """
    device = centers.device
    N = centers.shape[0]
    P = len(peak_locations)

    if P == 0 or len(weak_splat_indices) == 0:
        # Return empty tensors (not empty list)
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty

    # Ensure tensors are on correct device
    peak_tensor = peak_locations.to(device=device, dtype=torch.float32)
    weak_tensor = weak_splat_indices.to(device=device, dtype=torch.long)

    if enable_coverage_check:
        # Original behavior: Skip peaks already covered by non-weak splats

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

        # Step 3: Match uncovered peaks to weak splats (direct pairing)
        uncovered_peak_indices = torch.where(uncovered_mask)[0]
    else:
        # New behavior: Relocate to ALL peaks regardless of existing coverage
        # Rationale: If residual is high enough to be a peak, existing coverage is insufficient
        if verbose:
            aprint("  Coverage check disabled - considering all peaks")
        uncovered_peak_indices = torch.arange(P, device=device)

    # Returns (weak_splat_indices, peak_indices) as tensors
    weak_matches, peak_matches = _match_weak_to_peaks_direct(
        weak_tensor, uncovered_peak_indices
    )

    # Limit relocations (if max_relocations is not None)
    if max_relocations is not None:
        weak_matches = weak_matches[:max_relocations]
        peak_matches = peak_matches[:max_relocations]

    # Return tensor indices (caller will convert as needed)
    return weak_matches, peak_matches


def _reset_optimizer_state_batch(
    optimizer: torch.optim.Optimizer,
    model: "GaussianSplatModel",  # type: ignore[name-defined]  # noqa: F821
    splat_indices: torch.Tensor,
) -> None:
    """
    Reset optimizer state for relocated splats (vectorized).

    For Adam optimizer, this zeros out:
    - exp_avg: First moment estimates (momentum)
    - exp_avg_sq: Second moment estimates (variance)

    This prevents relocated splats from being influenced by gradients
    from their old locations, allowing them to be optimized fresh.

    Args:
        optimizer: PyTorch optimizer (typically Adam)
        model: GaussianSplatModel
        splat_indices: Tensor of splat indices to reset (shape: (n_relocations,))
    """
    if not isinstance(optimizer, torch.optim.Adam):
        # Only Adam has exp_avg/exp_avg_sq state
        # Other optimizers (SGD, etc.) don't need resetting
        return

    # Model parameters that need state reset
    params_to_reset = [
        model.raw_mu,
        model.raw_L_diag,
        model.L_off,
        model.raw_a,
    ]

    for param in params_to_reset:
        if param not in optimizer.state:
            # Parameter hasn't been optimized yet, skip
            continue

        state = optimizer.state[param]

        # Reset momentum (first moment)
        if "exp_avg" in state:
            state["exp_avg"][splat_indices] = 0.0

        # Reset variance (second moment)
        if "exp_avg_sq" in state:
            state["exp_avg_sq"][splat_indices] = 0.0


def _relocate_splats_batch(
    model: Any,
    splat_indices: torch.Tensor,
    peak_coords: torch.Tensor,
    residual: torch.Tensor,
    cfg: DynamicOpsConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Relocate multiple splats to new locations in a single batched operation.

    Uses fully vectorized tensor operations to process all relocations
    simultaneously, avoiding Python loops and repeated tensor allocations.

    Additionally resets the optimizer state (momentum, variance) for relocated
    splats to prevent them from being influenced by gradients from their old
    locations.

    Performance: Fully GPU-accelerated with no Python loops in critical path.

    Args:
        model: GaussianSplatModel (from luxar.gsplats.models)
        splat_indices: Tensor of splat indices to relocate, shape (n_relocations,)
        peak_coords: Tensor of new center coordinates (voxel), shape (n_relocations, d)
        residual: Residual tensor (for amplitude initialization)
        cfg: Dynamic operations configuration
        optimizer: Optional optimizer (for state reset). If provided, Adam state
            (momentum, variance) will be reset for relocated splats.
    """
    n_relocations = len(splat_indices)
    if n_relocations == 0:
        return

    device = model.raw_mu.device
    d = len(model.shape)

    # Ensure inputs are on correct device (no conversion if already there)
    splat_indices_t = splat_indices.to(device=device, dtype=torch.long)
    peak_coords_t = peak_coords.to(device=device, dtype=torch.float32)
    shape_arr = torch.tensor(model.shape, device=device, dtype=torch.float32)

    # === BATCH 1: Compute new centers (vectorized) ===
    # Convert centers to normalized [0,1] coordinates then to raw (logit) space
    u = torch.clamp(
        peak_coords_t / torch.clamp(shape_arr - 1.0, min=1.0), 1e-6, 1.0 - 1e-6
    )  # (n_relocations, d)
    raw_mu_new = torch.log(u) - torch.log(1.0 - u)  # (n_relocations, d)

    # === BATCH 2: Compute new amplitudes (vectorized indexing) ===
    # Get residual values at new locations using advanced indexing
    peak_coords_long = peak_coords_t.long()
    indices = tuple(peak_coords_long[:, i] for i in range(d))
    new_amplitudes = residual[indices]  # (n_relocations,)
    new_amplitudes = torch.clamp(new_amplitudes, min=1e-6)  # Defensive

    # Vectorized inverse softplus
    raw_a_new = stable_inverse_softplus_torch(new_amplitudes).to(
        device=device, dtype=torch.float32
    )  # (n_relocations,)

    # === BATCH 3: Compute new covariances (isotropic, same for all) ===
    if model.sigma_min_diag is not None:
        sigma_min = model.sigma_min_diag.to(device=device, dtype=torch.float32)
    else:
        sigma_min = torch.zeros(d, device=device, dtype=torch.float32)

    effective_diag = torch.clamp(cfg.init_sigma_vox - sigma_min, min=1e-6)
    raw_L_diag_new = stable_inverse_softplus_torch(effective_diag).to(
        device=device, dtype=torch.float32
    )  # (d,)

    # Broadcast to all relocations
    raw_L_diag_new = raw_L_diag_new.unsqueeze(0).expand(
        n_relocations, -1
    )  # (n_relocations, d)

    # === BATCH 4: Zero off-diagonal elements ===
    n_off_diag = d * (d - 1) // 2
    L_off_new = torch.zeros(
        (n_relocations, n_off_diag), device=device, dtype=torch.float32
    )

    # === BATCH UPDATE: Update all model parameters at once ===
    # Single indexing operation per parameter (much faster than loop)
    model.raw_mu.data[splat_indices_t] = raw_mu_new
    model.raw_L_diag.data[splat_indices_t] = raw_L_diag_new
    model.L_off.data[splat_indices_t] = L_off_new
    model.raw_a.data[splat_indices_t] = raw_a_new

    # === OPTIMIZER STATE RESET: Zero out Adam momentum/variance for relocated splats ===
    if optimizer is not None:
        _reset_optimizer_state_batch(optimizer, model, splat_indices_t)
