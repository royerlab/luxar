"""
Configuration dataclasses for Gaussian splat fitting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


# Type alias for the per-iteration callback. The callback is invoked from
# within the optimisation loop alongside the existing periodic eval, so
# pred and target tensors are already up to date.
#
# Arguments:
#   iteration: int          1-based iteration index
#   pred:      torch.Tensor current model prediction (volume tensor; on the
#                            fitter's device, in [0, 1] normalised intensity).
#                            Detached from the graph; safe to pass to render
#                            utilities or compute metrics on.
#   info:      dict         current-iteration metrics:
#                              - "loss":          float (current iter loss)
#                              - "best_loss":     float (best loss seen so far)
#                              - "max_abs_error": float (training-volume max abs err)
#                              - "rel_l2":        float (relative L2)
#                              - "n_splats":      int   (current model splat count)
IterCallback = Callable[[int, torch.Tensor, Dict[str, Any]], None]


@dataclass(frozen=True)
class OptimConfig:
    """Optimization hyperparameters for fit_gaussian_splats().

    Example::

        from luxar.gsplats.fitting.config import OptimConfig
        cfg = OptimConfig(n_iters=2000, lr=0.01, early_stop_patience=500)
        result = fit_gaussian_splats(volume, optim=cfg)
    """

    n_iters: int = 1000
    lr: float = 0.01
    gradient_clip: Optional[float] = 1.0
    scheduler_type: str = "plateau"
    patience: int = 25
    lr_reduction_factor: float = 0.98
    early_stop_patience: Optional[int] = 300
    sort_splats_enabled: bool = True
    sort_splats_interval: int = 1000


@dataclass(frozen=True)
class LossConfig:
    """Loss function configuration for fit_gaussian_splats().

    Default loss is "l1": across the loss-comparison study (Supp. Doc. 5),
    L1 reaches equal-or-higher held-out PSNR than MSE on every microscopy
    dataset tested. Pass ``loss_type="mse"`` or ``loss_type="poisson"`` to
    override.

    Example::

        from luxar.gsplats.fitting.config import LossConfig
        cfg = LossConfig(loss_type="poisson", asymmetric_penalty=5.0)
        result = fit_gaussian_splats(volume, loss=cfg)
    """

    loss_type: str = "l1"
    asymmetric_penalty: Optional[float] = 1.0
    l1_amp: Optional[float] = None
    l1_diag: Optional[float] = None
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
    sigma_max_diag: Optional[Sequence[float] | float] = None
    amp_max: Optional[float] = None
    max_eccentricity: Optional[float] = 10.0
    truncate: float = 2.75
    voxel_size: Optional[Sequence[float] | float] = None
    output_space: str = "real"
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
    seeds: Optional[
        np.ndarray | int | float | "GSplatData"
    ]  # Array, int, compression ratio, or GSplatData

    # Normalization
    norm_percentile: float

    # Model parameters
    init_sigma_vox: Optional[float]
    sigma_min_diag: Optional[Sequence[float]]
    sigma_max_diag: Optional[Sequence[float] | float]
    truncate: float

    # Optimization parameters
    n_iters: int
    lr: float
    max_abs_error: Optional[float]
    rel_l2_target: Optional[float]
    gradient_clip: Optional[float]

    # Loss function
    loss_type: str
    asymmetric_penalty: Optional[float]
    l1_amp: Optional[float]
    l1_diag: Optional[float]

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

    # Background floor / DC-offset suppression (with default - see preprocessing)
    # Raises the effective image_min used in normalization so a constant
    # pedestal is clipped to 0 before fitting. "auto" | "none" | "pN" | float.
    floor: str | float | None = "auto"

    # Metal acceleration (with defaults - must come after required fields)
    use_metal: bool = True  # Enable Metal acceleration when available (macOS + MPS)
    metal_intensity_floor: float = 1e-5  # Early culling threshold for Metal kernels

    # CUDA acceleration (with defaults)
    use_cuda: bool = True  # Enable custom CUDA kernels when available (NVIDIA GPUs)
    cuda_intensity_floor: float = 1e-5  # Early culling threshold for CUDA kernels

    # Seed generation (with defaults - must come after required fields)
    seed_method: str = (
        "auto"  # "decomposition", "grid", "edges", "auto", or comma-separated
    )
    seed_kwargs: Optional[Dict[str, Any]] = (
        None  # Additional parameters for seed generation
    )

    # Pre-initialized parameters (for GSplatData seeds or moment pursuit)
    # If set, these override the default initialization
    init_L: Optional[np.ndarray] = None  # Shape (N, d, d) - Cholesky factors
    init_amps: Optional[np.ndarray] = None  # Shape (N,) - amplitudes

    # Amplitude constraint (prevents explosion with few splats)
    amp_max: Optional[float] = None  # Maximum amplitude value if specified

    # Constraint parameters
    max_eccentricity: Optional[float] = None  # Limit ratio of longest to shortest axis

    # Voxel footprint correction (post-processing)
    # - False: Disabled (default)
    # - True: Enable with 1-voxel box footprint (sigma ≈ 0.289 voxels)
    # - float: Custom sigma in voxel units (e.g., 0.5 for half-voxel blur)
    voxel_footprint_correction: bool | float = False

    # Boundary containment (post-processing)
    # Clip Cholesky factors so no splat extends beyond the volume bounds.
    clip_to_bounds: bool = False

    # Anisotropic voxel spacing (physical size per voxel along each axis)
    # None = isotropic (all 1s). Array of shape (d,) for anisotropic volumes.
    voxel_size: Optional[np.ndarray] = None

    # Output coordinate system: "real" (physical) or "voxel"
    # When "real" and voxel_size is set, output centers and Cholesky are scaled to physical coords.
    # When voxel_size is None, "real" and "voxel" produce identical results.
    output_space: str = "real"

    # Boundary penalty weight (loss term during optimization)
    # Adds a differentiable penalty for splats whose effective support extends beyond bounds.
    boundary_penalty: Optional[float] = None

    # Z-order (Morton) sorting for memory locality
    sort_splats_enabled: bool = True  # Enable periodic Morton-code sorting of splats
    sort_splats_interval: int = (
        1000  # Sort every N iterations (also sorts at iteration 0)
    )

    # Volume downscaling (preprocessing)
    # Per-axis integer factors, e.g. (1, 4, 4). None = no downscaling.
    # Volume is anti-alias filtered (Gaussian, sigma=factor/2) and decimated before fitting.
    # Splat parameters are automatically rescaled to original coordinates after fitting.
    downscale: Optional[tuple[int, ...]] = None

    # Per-iteration callback. Invoked alongside the existing periodic eval
    # (every iter_callback_every iters or every _EVAL_INTERVAL iters,
    # whichever is larger). When None, no callback is fired.
    # See ``IterCallback`` type alias above for the expected signature.
    # Use cases: validation-set scoring, custom snapshot logging, stop-on-
    # external-criterion. The callback runs inside ``torch.no_grad()`` and
    # must not raise.
    iter_callback: Optional[IterCallback] = None
    iter_callback_every: int = 25  # call cadence (capped to >= eval interval)


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

    # Convergence thresholds
    max_abs_error: float
    rel_l2_target: Optional[float] = None

    # Resolved background floor that was subtracted (None if disabled).
    # Equal to image_min when floor suppression is active. Recorded for
    # inspection/reproducibility; NOT added back to output amplitudes.
    floor: Optional[float] = None

    # Computed L1 regularization values (set during preprocessing)
    # These are stored here instead of mutating FitConfig
    l1_amp: Optional[float] = None
    l1_diag: Optional[float] = None

    # Pre-initialized model parameters (set during preprocessing)
    # These are processed copies - FitConfig is not mutated
    init_L: Optional[np.ndarray] = None  # Shape (N, d, d) - Cholesky factors
    init_amps: Optional[np.ndarray] = None  # Shape (N,) - amplitudes

    # Downscale factors applied during preprocessing (for rescaling in finalize_results)
    downscale_factors: Optional[tuple[int, ...]] = None


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

    # Optimization metadata
    converged_early: bool
    early_stopped: bool
    actual_iters: int
    best_iteration: int
    best_loss: float
    best_max_abs_error: float
    best_rel_l2: float

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
