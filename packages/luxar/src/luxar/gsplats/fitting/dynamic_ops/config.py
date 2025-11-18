# config.py
"""Configuration for dynamic Gaussian splat operations."""


class DynamicOpsConfig:
    """
    Configuration for convergence-driven dynamic Gaussian splat operations.

    This class contains all parameters for the three-step dynamic operations algorithm:
    1. Residual Peak Analysis: Find strongest error locations
    2. Convergence-Based Operations: Seed based on convergence criteria
    3. Global Pruning: Remove ineffective splats

    Key features:
    - Convergence-based detection aligns operations with optimization goals
    - Adaptive thresholds prevent plateau issues
    - Asymmetric loss awareness for additive Gaussian models
    - Residual-driven seeding targets reconstruction deficiencies
    """

    def __init__(self):
        # Scheduling
        self.step_every: int = 50  # Run operations every N iterations

        # Step 1: Residual Peak Analysis
        self.k_max_residuals: int = 10  # Number of residual peaks to analyze per cycle
        self.nms_radius_vox: float = 2.0  # Minimum distance between detected peaks

        # Tile-based seeding for spatial fairness
        self.enable_tiled_seeding: bool = (
            True  # Use tile-based seeding for fair coverage
        )
        self.num_tiles_per_dim: int | None = (
            None  # Auto: 16 for 2D, 6 for 3D, 4 for 4D, 2 for 5D+
        )
        self.k_per_tile: int = 1  # Number of peaks to find per tile

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
