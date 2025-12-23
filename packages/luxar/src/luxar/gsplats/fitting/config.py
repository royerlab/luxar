"""
Configuration dataclasses for Gaussian splat fitting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig


@dataclass
class FitConfig:
    """
    Configuration for Gaussian splat fitting.

    Contains all parameters and settings needed for the fitting process.
    """

    # Input data (required)
    V: np.ndarray
    seeds: Optional[np.ndarray | int | float]  # Array, int count, or proportion

    # Normalization
    norm_percentile: float

    # Model parameters
    init_sigma_vox: float
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

    # Seed generation (with defaults - must come after required fields)
    seed_method: str = "both"  # "gaussian", "decomposition", "both", etc.
    seed_kwargs: Dict[str, Any] = None  # Additional parameters for seed generation


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

    Contains model, optimizer, scheduler, and coordinator.
    """

    model: Any  # GaussianSplatModel
    optimizer: torch.optim.Optimizer
    scheduler: Any  # Learning rate scheduler
    coordinator: Any  # ModelOptimizerCoordinator
