"""
Configuration dataclasses for Gaussian splat fitting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.typing_utils.constants import (
    DEFAULT_SIGMA_MIN_DIAG,
    DEFAULT_TRUNCATION_RADIUS,
)

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
    """Optimization hyperparameters for :func:`fit_gaussian_splats`.

    A declarative bundle. ``fit_gaussian_splats`` takes these as FLAT keyword
    arguments — there is no ``optim=`` parameter — so a config is applied by
    unpacking it::

        from dataclasses import asdict

        from luxar.gsplats import fit_gaussian_splats
        from luxar.gsplats.fitting.config import OptimConfig

        cfg = OptimConfig(n_iters=2000, lr=0.01, early_stop_patience=500)
        result = fit_gaussian_splats(volume, **asdict(cfg))

    Every field defaults to exactly what ``fit_gaussian_splats`` defaults to, so
    unpacking a default-constructed config is a no-op rather than a silent
    change of behaviour. ``test_no_default_disagrees_with_the_entry_point``
    enforces that for every field of all three configs.
    """

    n_iters: int = 1000
    lr: float = 0.01
    # Disabled, matching the fitter: MSE gradients are well-scaled.
    gradient_clip: Optional[float] = None
    scheduler_type: str = "plateau"
    # 15/0.9, not the gentler 25/0.98 this once carried: the fitter's own
    # comment records that the tuned pair is 33% faster at the same PSNR, and a
    # config claiming to configure the fitter must not hand back the superseded
    # values.
    patience: int = 15
    lr_reduction_factor: float = 0.9
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

    Applied by unpacking — there is no ``loss=`` parameter::

        from dataclasses import asdict

        from luxar.gsplats import fit_gaussian_splats
        from luxar.gsplats.fitting.config import LossConfig

        cfg = LossConfig(loss_type="poisson", asymmetric_penalty=5.0)
        result = fit_gaussian_splats(volume, **asdict(cfg))
    """

    loss_type: str = "l1"
    asymmetric_penalty: Optional[float] = 1.0
    l1_amp: Optional[float] = None
    l1_diag: Optional[float] = None


@dataclass(frozen=True)
class ConstraintConfig:
    """Constraint configuration for :func:`fit_gaussian_splats`.

    Applied by unpacking — there is no ``constraints=`` parameter::

        from dataclasses import asdict

        from luxar.gsplats import fit_gaussian_splats
        from luxar.gsplats.fitting.config import ConstraintConfig

        cfg = ConstraintConfig(amp_max=2.0, max_eccentricity=5.0)
        result = fit_gaussian_splats(volume, **asdict(cfg))
    """

    # DEFAULT_SIGMA_MIN_DIAG, not None. `None` removes the lower bound on splat
    # width entirely, so unpacking a default config used to switch the floor OFF
    # while reading as "no change".
    sigma_min_diag: Optional[Sequence[float] | float] = DEFAULT_SIGMA_MIN_DIAG
    sigma_max_diag: Optional[Sequence[float] | float] = None
    amp_max: Optional[float] = None
    max_eccentricity: Optional[float] = 10.0
    truncate: float = DEFAULT_TRUNCATION_RADIUS
    voxel_size: Optional[Sequence[float] | float] = None
    output_space: str = "real"
    boundary_penalty: Optional[float] = None
    clip_to_bounds: bool = False


@dataclass(frozen=True)
class FitParameters:
    """Raw parameters threaded through the internal fitting pipeline.

    ``fit_gaussian_splats`` remains the explicit public API. This bundle removes
    the duplicate parameter signatures from ``GaussianSplatFitter.fit`` and
    ``prepare_fit_config`` while preserving their existing values until the
    validation boundary normalizes them into :class:`FitConfig`.
    """

    V: np.ndarray
    seeds: Optional[np.ndarray | int | float | "GSplatData"] = None
    norm_percentile: float = 0.0
    floor: str | float | None = "auto"
    norm_range: tuple[float, float] | None = None
    downscale: Optional[int | Sequence[int]] = None
    init_sigma_vox: Optional[float] = None
    n_iters: int = 1000
    lr: float = 0.01
    loss_type: str = "l1"
    asymmetric_penalty: Optional[float] = 1.0
    l1_amp: Optional[float] = None
    l1_diag: Optional[float] = None
    sigma_min_diag: Optional[Sequence[float] | float] = DEFAULT_SIGMA_MIN_DIAG
    sigma_max_diag: Optional[Sequence[float] | float] = None
    amp_max: Optional[float] = None
    max_eccentricity: Optional[float] = 10.0
    truncate: float = DEFAULT_TRUNCATION_RADIUS
    seed_method: str = "auto"
    verbose: bool = True
    max_abs_error: Optional[float] = None
    rel_l2_target: Optional[float] = None
    gradient_clip: Optional[float] = None
    napari_movie: bool = False
    movie_every: int = 1
    movie_max_frames: Optional[int] = None
    scheduler_type: str = "plateau"
    patience: int = 15
    lr_reduction_factor: float = 0.9
    early_stop_patience: Optional[int] = 300
    dynamic_ops_verbose: bool = False
    voxel_footprint_correction: bool | float = False
    boundary_penalty: Optional[float] = None
    clip_to_bounds: bool = False
    voxel_size: Optional[Sequence[float] | float] = None
    output_space: str = "real"
    sort_splats_enabled: bool = True
    sort_splats_interval: int = 1000
    iter_callback: Optional[Any] = None
    iter_callback_every: int = 25
    seed_amps_background_relative: bool = False
    source_dtype: Optional[str] = None
    source_shape: Optional[Sequence[int]] = None
    source_stored_bytes: Optional[int] = None
    seed_kwargs: Dict[str, Any] = field(default_factory=dict)


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

    # Explicit (image_min, image_max) for normalization, overriding the values
    # `norm_percentile` would derive from THIS array. Set by the tiled fitter to
    # a level resolved against the WHOLE volume, so every tile maps the same
    # physical intensity to the same normalized value — the amplitude-scale
    # counterpart of the globally-resolved `floor`. Leave None for a
    # whole-volume fit, where the array already IS the volume.
    norm_range: Optional[tuple[float, float]] = None

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

    # Amplitude convention of a ``seeds=GSplatData`` warm start (#1172). The
    # door carries both kinds and they differ by exactly the background floor,
    # so only the CALLER knows which one it is holding:
    #
    # * False (default) — RAW-IMAGE-SAMPLED, e.g. ``generate_seeds()`` output
    #   (the documented explicit-seeding workflow): amplitudes read off the
    #   original volume, pedestal included. Rescaled as
    #   ``(a - image_min) / intensity_range``.
    # * True — BACKGROUND-RELATIVE, e.g. a previous fit's output or a
    #   ``.gsplats.zarr`` WRITTEN BY a fit (``gsplat fit`` / ``gsplat lod``):
    #   ``finalize_results`` scales by ``intensity_range`` and never adds
    #   ``image_min`` back. Rescaled as ``a / intensity_range``; subtracting
    #   ``image_min`` again would remove the floor twice and zero every
    #   sub-floor seed.
    #
    # An IMPORTED store (``gsplat import``, opacity mapped into ~[0, 1]) or an
    # intensity-rescaled one (``gsplat transform --normalize-intensity`` /
    # ``--scale-intensity``) carries neither convention exactly, so its warm
    # start is approximate whichever value is declared.
    #
    # Only consulted when ``seeds`` is a GSplatData.
    seed_amps_background_relative: bool = False

    # Pre-initialized parameters (for GSplatData seeds or moment pursuit)
    # If set, these override the default initialization
    init_L: Optional[np.ndarray] = None  # Shape (N, d, d) - Cholesky factors
    # Shape (N,) - amplitudes, in RAW IMAGE INTENSITY units (the same scale as
    # ``V``, pedestal included), which is what the seeding methods produce.
    # preprocess_data rescales them to the optimizer's [0, 1] scale as
    # ``(a - image_min) / intensity_range``, so a background floor is removed
    # exactly once. Amplitudes coming out of a PREVIOUS fit are already
    # background-relative (see finalize_results) and must NOT be passed here —
    # pass the whole GSplatData as ``seeds=`` together with
    # ``seed_amps_background_relative=True``, which skips the ``- image_min``
    # term (#1172).
    init_amps: Optional[np.ndarray] = None

    # Amplitude constraint (prevents explosion with few splats)
    amp_max: Optional[float] = None  # Maximum amplitude value if specified

    # Constraint parameters
    # 10.0, matching `ConstraintConfig` and `fit_gaussian_splats`. It was
    # None here, so constructing a `FitConfig(...)` without explicitly passing
    # this optional field removed the eccentricity limit entirely while every
    # documented default says 10.
    max_eccentricity: Optional[float] = 10.0  # Limit ratio of longest to shortest axis

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
    # The caller's array as handed in, BEFORE `V` was cast to float32. Recorded
    # because it is the honest denominator of a compression ratio: a uint16
    # source cast to float32 doubles in size, and quoting the cast size would
    # overstate compression by 2x. `None` when unknown.
    source_dtype: Optional[str] = None
    source_itemsize: Optional[int] = None
    #: Declared grid of the ACQUISITION, when the caller preprocessed before
    #: fitting. `None` means the array handed in IS the source.
    source_shape: Optional[list[int]] = None
    #: Bytes the acquisition OCCUPIES on disk (compressed), as opposed to
    #: the decoded `source_bytes`. Enables the second, apples-to-apples ratio.
    source_stored_bytes: Optional[int] = None


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
    # Shape (N,) - amplitudes, already rescaled to the optimizer's normalized
    # [0, 1] scale (whichever input convention they arrived in - see
    # FitConfig.init_amps and preprocessing._InitContext).
    init_amps: Optional[np.ndarray] = None

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
