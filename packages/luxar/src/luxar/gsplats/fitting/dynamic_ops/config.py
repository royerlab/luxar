# config.py
"""Configuration for dynamic Gaussian splat operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DynamicOpsConfig:
    """
    Configuration for fixed-pool splat relocation operations.

    This class contains all parameters for the splat relocation algorithm:
    1. Residual Peak Analysis: Find strongest error locations
    2. Weak Splat Identification: Find splats with low importance (amplitude x volume)
    3. Relocation: Move weak splats to high-residual peaks

    Key features:
    - Fixed splat pool (no topology changes) enables fast standard optimizer
    - Relocation preserves total splat count while redistributing coverage
    - NMS ensures relocated splats don't crowd each other
    - Convergence-based guards prevent unnecessary operations
    """

    # Scheduling
    step_every: int = 50  # Run operations every N iterations

    # Step 1: Residual Peak Analysis
    k_max_residuals: int = 40  # Max peaks to find per step
    nms_radius_vox: float = 2.0  # Minimum distance between detected peaks

    # Tile-based peak finding for spatial fairness (enabled by default)
    # In tiled mode, k_per_tile is auto-calculated as k_max_residuals / num_tiles
    # If k_per_tile >= 1: deterministic (keep floor(k_per_tile) per tile)
    # If k_per_tile < 1: probabilistic (keep each peak with probability k_per_tile)
    enable_tiled_seeding: bool = True  # Use tiled seeding for spatial fairness
    num_tiles_per_dim: Optional[int] = (
        None  # Auto: 16 for 2D, 6 for 3D, 4 for 4D, 2 for 5D+
    )
    # RNG seed for the probabilistic per-tile keep decision used when
    # k_per_tile < 1 and for the weak-splat residual sample. Explicit seeds
    # advance with each dynamic-ops step, keeping a fit reproducible without
    # repeating the same samples. None leaves the residual sample on PyTorch's
    # global RNG stream and the per-tile keep decision unseeded.
    seed: Optional[int] = 42

    # Step 2: Weak Splat Identification
    relocation_percentile: float = (
        1.0  # Percentage of least important splats eligible for relocation
    )
    max_relocations_per_step: Optional[int] = (
        64  # Maximum splats to relocate per step (None = no limit, relocate all matches)
    )

    # Step 3: Relocation Parameters
    init_sigma_vox: float = (
        0.5  # Initial sigma for relocated splats (isotropic, single-voxel scale)
    )
    min_contribution_threshold: float = (
        0.01  # Minimum influence to consider a peak "covered" by existing splat
    )
    enable_coverage_check: bool = (
        False  # If True, skip peaks already covered by non-weak splats
    )
    # Default False: Relocate to ALL high-residual peaks regardless of coverage
    # Rationale: If a peak has high residual, existing coverage is clearly insufficient
    # The issue: "has influence" != "error is resolved"
    #
    # Set to True for conservative behavior (original): only relocate to uncovered peaks
    # This may leave persistent high-error regions unaddressed

    # Cooldown mechanism (prevents immediate re-relocation)
    relocation_cooldown_steps: int = (
        1  # Number of dynamic ops steps to wait before allowing re-relocation
    )
    # After a splat is relocated, it cannot be relocated again for N dynamic ops steps.
    # This ensures diverse splat coverage instead of repeatedly relocating the same splats.
    # Increase for more conservative relocation, decrease for more aggressive adaptation.

    # Safety parameters
    min_splats_to_keep: int = (
        10  # Minimum splats - never relocate below this count (backwards compat)
    )
