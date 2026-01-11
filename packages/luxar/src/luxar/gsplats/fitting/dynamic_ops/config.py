# config.py
"""Configuration for dynamic Gaussian splat operations."""


class DynamicOpsConfig:
    """
    Configuration for fixed-pool splat relocation operations.

    This class contains all parameters for the splat relocation algorithm:
    1. Residual Peak Analysis: Find strongest error locations
    2. Weak Splat Identification: Find splats with low importance (amplitude × volume)
    3. Relocation: Move weak splats to high-residual peaks

    Key features:
    - Fixed splat pool (no topology changes) enables fast standard optimizer
    - Relocation preserves total splat count while redistributing coverage
    - NMS ensures relocated splats don't crowd each other
    - Convergence-based guards prevent unnecessary operations
    """

    def __init__(self) -> None:
        # Scheduling
        self.step_every: int = 50  # Run operations every N iterations

        # Step 1: Residual Peak Analysis
        self.k_max_residuals: int = 20  # Max peaks to find per step
        self.nms_radius_vox: float = 2.0  # Minimum distance between detected peaks

        # Tile-based peak finding for spatial fairness (enabled by default)
        # In tiled mode, k_per_tile is auto-calculated as k_max_residuals / num_tiles
        # If k_per_tile >= 1: deterministic (keep floor(k_per_tile) per tile)
        # If k_per_tile < 1: probabilistic (keep each peak with probability k_per_tile)
        self.enable_tiled_seeding: bool = True  # Use tiled seeding for spatial fairness
        self.num_tiles_per_dim: int | None = (
            None  # Auto: 16 for 2D, 6 for 3D, 4 for 4D, 2 for 5D+
        )

        # Step 2: Weak Splat Identification
        self.relocation_percentile: float = (
            5.0  # Percentage of least important splats eligible for relocation
        )
        self.max_relocations_per_step: int = (
            10  # Maximum splats to relocate per step (prevents destabilization)
        )

        # Step 3: Relocation Parameters
        self.init_sigma_vox: float = (
            0.5  # Initial sigma for relocated splats (isotropic, single-voxel scale)
        )
        self.min_contribution_threshold: float = (
            0.01  # Minimum influence to consider a peak "covered" by existing splat
        )

        # Safety parameters
        self.min_splats_to_keep: int = (
            10  # Minimum splats - never relocate below this count (backwards compat)
        )
