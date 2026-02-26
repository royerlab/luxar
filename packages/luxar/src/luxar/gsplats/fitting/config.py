"""
Configuration dataclasses for Gaussian splat fitting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig


@dataclass(frozen=True)
class OptimConfig:
    """Optimization hyperparameters for fit_gaussian_splats().

    Example::

        from luxar.gsplats.fitting.config import OptimConfig
        cfg = OptimConfig(n_iters=2000, lr=0.01, early_stop_patience=500)
        result = fit_gaussian_splats(volume, optim=cfg)
    """

    n_iters: int = 1000
    lr: float = 0.05
    gradient_clip: Optional[float] = 1.0
    scheduler_type: str = "plateau"
    patience: int = 25
    lr_reduction_factor: float = 0.98
    early_stop_patience: Optional[int] = 300


@dataclass(frozen=True)
class LossConfig:
    """Loss function configuration for fit_gaussian_splats().

    Example::

        from luxar.gsplats.fitting.config import LossConfig
        cfg = LossConfig(loss_type="mse", asymmetric_penalty=5.0)
        result = fit_gaussian_splats(volume, loss=cfg)
    """

    loss_type: str = "l1"
    asymmetric_penalty: Optional[float] = 10.0
    l1_amp: Optional[float] = None
    l1_diag: Optional[float] = None
    l1_sharpness: Optional[float] = None
    boundary_penalty: Optional[float] = None


@dataclass(frozen=True)
class ConstraintConfig:
    """Constraint configuration for fit_gaussian_splats().

    Example::

        from luxar.gsplats.fitting.config import ConstraintConfig
        cfg = ConstraintConfig(amp_max=2.0, max_eccentricity=5.0)
        result = fit_gaussian_splats(volume, constraints=cfg)
    """

    sigma_min_diag: Optional[Sequence[float] | float] = None
    sigma_max_diag: Optional[Sequence[float]] = None
    amp_max: Optional[float] = None
    max_eccentricity: Optional[float] = 10.0
    sharpness_range: Optional[tuple[float, float] | float] = 2.0
    truncate: float = 3.0
    boundary_penalty: Optional[float] = None
    clip_to_bounds: bool = False


@dataclass
class FitConfig:
    """
    Configuration for Gaussian splat fitting.

    Contains all parameters and settings needed for the fitting process.
    """

    # Input data (required)
    V: np.ndarray
    seeds: Optional[np.ndarray | int | float | "GSplatData"]  # noqa: F821 - Array, int, compression ratio, or GSplatData

    # Normalization
    norm_percentile: float

    # Model parameters
    init_sigma_vox: Optional[float]
    sigma_min_diag: Optional[Sequence[float]]
    sigma_max_diag: Optional[Sequence[float]]
    truncate: float

    # Optimization parameters
    n_iters: int
    lr: float
    max_abs_error: Optional[float]
    gradient_clip: Optional[float]

    # Loss function
    loss_type: str
    asymmetric_penalty: Optional[float]
    l1_amp: Optional[float]
    l1_diag: Optional[float]
    l1_sharpness: Optional[float]  # L1 regularization on sharpness offsets (s')

    # Scheduler parameters
    scheduler_type: str
    patience: int  # LR reduction patience (iterations without loss improvement)
    lr_reduction_factor: float  # LR multiplier (e.g., 0.5 = halve, 0.1 = reduce to 10%)
    early_stop_patience: Optional[int]  # Stop if no improvement for N iters

    # Dynamic operations
    enable_dynamic_ops: bool
    dynamic_config: DynamicOpsConfig
    dynamic_ops_verbose: bool

    # Movie recording
    napari_movie: bool
    movie_every: int
    movie_max_frames: Optional[int]

    # Device and logging
    device: torch.device
    verbose: bool

    # Metal acceleration (with defaults - must come after required fields)
    use_metal: bool = True  # Enable Metal acceleration when available (macOS + MPS)
    metal_intensity_floor: float = 1e-5  # Early culling threshold for Metal kernels
    metal_tile_size: int = 4  # Tile size for 3D binning (4=64 threads, 8=512 threads)

    # CUDA acceleration (with defaults)
    use_cuda: bool = True  # Enable custom CUDA kernels when available (NVIDIA GPUs)
    cuda_intensity_floor: float = 1e-5  # Early culling threshold for CUDA kernels
    cuda_tile_size: Optional[int] = None  # Auto-select based on dimension if None

    # Seed generation (with defaults - must come after required fields)
    seed_method: str = (
        "auto"  # "decomposition", "grid", "edges", "auto", or comma-separated
    )
    seed_kwargs: Optional[Dict[str, Any]] = None  # Additional parameters for seed generation

    # Pre-initialized parameters (for GSplatData seeds or moment pursuit)
    # If set, these override the default initialization
    init_L: Optional[np.ndarray] = None  # Shape (N, d, d) - Cholesky factors
    init_amps: Optional[np.ndarray] = None  # Shape (N,) - amplitudes
    init_sharpness: Optional[np.ndarray] = None  # Shape (N,) - sharpness values

    # Amplitude constraint (prevents explosion with few splats)
    amp_max: Optional[float] = None  # Maximum amplitude value if specified

    # Constraint parameters
    max_eccentricity: Optional[float] = None  # Limit ratio of longest to shortest axis
    sharpness_range: Optional[tuple[float, float] | float] = (
        None  # (min, max) tuple or fixed value
    )

    # Post-fit culling ratio: threshold = cull_ratio * max_abs_error
    # Splats with amplitude below this threshold are removed after fitting.
    # 0.0 disables culling, 1.0 culls at the full convergence threshold.
    cull_ratio: float = 0.1

    # Voxel footprint correction (post-processing)
    # - False: Disabled (default)
    # - True: Enable with 1-voxel box footprint (sigma ≈ 0.289 voxels)
    # - float: Custom sigma in voxel units (e.g., 0.5 for half-voxel blur)
    voxel_footprint_correction: bool | float = False

    # Boundary containment (post-processing)
    # Clip Cholesky factors so no splat extends beyond the volume bounds.
    clip_to_bounds: bool = False

    # Boundary penalty weight (loss term during optimization)
    # Adds a differentiable penalty for splats whose effective support extends beyond bounds.
    boundary_penalty: Optional[float] = None


@dataclass
class PreprocessedData:
    """
    Data that has been preprocessed and is ready for optimization.

    Contains normalized data, seed centers, and preprocessing metadata.
    """

    # Preprocessed input data
    V_normalized: np.ndarray
    V_tensor: torch.Tensor

    # Seed centers for initialization
    seed_centers: np.ndarray

    # Normalization metadata
    image_min: float
    image_max: float
    intensity_range: float

    # Dimensions
    d: int
    N: int  # number of candidates

    # Convergence threshold
    max_abs_error: float

    # Computed L1 regularization values (set during preprocessing)
    # These are stored here instead of mutating FitConfig
    l1_amp: Optional[float] = None
    l1_diag: Optional[float] = None
    l1_sharpness: Optional[float] = None

    # Pre-initialized model parameters (set during preprocessing)
    # These are processed copies - FitConfig is not mutated
    init_L: Optional[np.ndarray] = None  # Shape (N, d, d) - Cholesky factors
    init_amps: Optional[np.ndarray] = None  # Shape (N,) - amplitudes
    init_sharpness: Optional[np.ndarray] = None  # Shape (N,) - sharpness values


@dataclass
class OptimizationResults:
    """
    Results from the optimization process.

    Contains final parameters, optimization statistics, and metadata.
    """

    # Model parameters (from best state)
    centers: torch.Tensor
    Ls: torch.Tensor
    amps: torch.Tensor
    sharpness: torch.Tensor

    # Optimization metadata
    converged_early: bool
    early_stopped: bool
    actual_iters: int
    best_iteration: int
    best_loss: float
    best_max_abs_error: float

    # Movie frames (if enabled)
    movie_frames: Optional[Dict[str, Any]]

    # Timing
    start_time: float
    end_time: float


@dataclass
class ModelComponents:
    """
    Components needed during optimization.

    Contains model, optimizer, and scheduler.
    """

    model: Any  # GaussianSplatModel
    optimizer: torch.optim.Optimizer
    scheduler: Any  # Learning rate scheduler
