"""Fit Gaussian splats to a volume with PyTorch Adam and fixed-pool relocation.

Provides ``GaussianSplatFitter`` (the configurable fitter class) and the
``fit_gaussian_splats`` convenience function, which optimize a fixed pool of
oriented Gaussians against an nD volume, periodically relocating the least
informative splats to high-residual regions so optimizer tensor shapes stay
constant. Runs on CPU, CUDA, or MPS.
"""

from __future__ import annotations

import gc
import socket
import time
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from arbol import asection

from luxar.gsplats.fitting import (
    FitParameters,
    create_loss_function,
    finalize_results,
    initialize_optimization,
    prepare_fit_config,
    preprocess_data,
    run_optimization_loop,
)
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fitting.splat_hygiene import (
    DEFAULT_MIN_EXTENT_MAX_MASS_SHARE,
    DEFAULT_MIN_EXTENT_VOX,
)
from luxar.gsplats.fitting.validation import DEFAULT_SIGMA_MIN_DIAG
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils import resolve_torch_device
from luxar.gsplats.utils.trils import tril_size
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from luxar.typing_utils.json_safe import json_safe_value
from luxar.utils.arbol_warnings import arbol_warnings

_CPU_NOTE_PRINTED = False


class GaussianSplatFitter:
    """
    Gaussian splat fitter using standard PyTorch Adam with fixed-pool relocation.

    Parameters are optimized with a single ``torch.optim.Adam`` instance
    (fused on CUDA when available). Rather than dynamically adding and
    removing splats, the fitter periodically relocates the least
    informative splats to regions of high reconstruction residual, which
    keeps optimizer tensor shapes constant and avoids per-splat state
    management.

    Parameters
    ----------
    device : str, optional
        PyTorch device ('auto', 'cpu', 'cuda', 'mps'). Auto-detects for
        ``None`` or ``'auto'``.
    enable_dynamic_ops : bool, default=True
        Enable fixed-pool splat relocation during fitting.
    dynamic_config : DynamicOpsConfig, optional
        Configuration for dynamic operations.
    use_metal : bool, default=True
        Enable Metal acceleration on Apple Silicon (macOS + MPS device).
    use_cuda : bool, default=True
        Enable custom CUDA kernels on NVIDIA GPUs.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        enable_dynamic_ops: bool = True,
        dynamic_config: Optional[DynamicOpsConfig] = None,
        use_metal: bool = True,
        use_cuda: bool = True,
    ) -> None:
        global _CPU_NOTE_PRINTED  # noqa: PLW0603
        # Hardware acceleration flags (stored first for device detection)
        self.use_metal = use_metal
        self.use_cuda = use_cuda

        # Auto-detect best performing device based on platform and available hardware:
        # - macOS + Apple Silicon: MPS with Metal acceleration (substantial speedup, chip-dependent)
        # - Linux + NVIDIA GPU: CUDA with custom kernels (often orders of magnitude faster, GPU-dependent)
        # - Fallback: CPU
        self.device = resolve_torch_device(
            device,
            use_cuda=use_cuda,
            use_metal=use_metal,
        )

        if self.device.type == "cpu" and not _CPU_NOTE_PRINTED:
            from arbol import aprint

            # A note, not a UserWarning: whether CPU fitting is slow depends on
            # the dimensionality and numba, which initialization knows; it
            # warns (warn_cpu_fitting) only when the PyTorch renderer runs.
            aprint(
                "Fitting on the CPU; 3D fits use the native CPU kernels when "
                "numba is available; a CUDA or Apple Silicon GPU is still faster "
                "for large volumes."
            )
            _CPU_NOTE_PRINTED = True

        # Dynamic operations configuration
        self.enable_dynamic_ops = enable_dynamic_ops
        self.dynamic_config = dynamic_config or DynamicOpsConfig()

    def fit(self, parameters: FitParameters) -> GSplatData:
        """
        Fit Gaussian splats using the modular fitting pipeline.

        See fit_gaussian_splats() for full parameter documentation.

        Parameters
        ----------
        parameters : FitParameters
            Raw fit parameters collected by :func:`fit_gaussian_splats`.

        Returns
        -------
        GSplatData
            Dataclass containing centers, amplitudes, cholesky_factors, and stats.
        """
        # Step 1: Validate and prepare configuration
        started = time.perf_counter()
        config = prepare_fit_config(self, parameters)
        resolved_config = _configuration_snapshot(config)

        # Clear cached rendering grids from previous fitting sessions
        from luxar.gsplats.models.gsplats.rendering_core import clear_grid_cache

        clear_grid_cache()

        # Step 2: Preprocess data and generate candidates
        preprocessed_data = preprocess_data(config)
        prepared = time.perf_counter()
        resolved = {
            "seed_count": preprocessed_data.N,
            "floor": preprocessed_data.floor,
            "l1_amp": preprocessed_data.l1_amp,
            "l1_diag": preprocessed_data.l1_diag,
            "max_abs_error": preprocessed_data.max_abs_error,
            "rel_l2_target": preprocessed_data.rel_l2_target,
        }

        # Handle edge case of no candidates
        if preprocessed_data.N == 0:
            d = preprocessed_data.d
            empty = GSplatData(
                centers=np.zeros((0, d), dtype=np.float32),
                amplitudes=np.zeros((0,), dtype=np.float32),
                cholesky_factors=np.zeros((0, tril_size(d)), dtype=np.float32),
                stats={},
            )
            empty.stats["fit_timings"] = {
                "preprocessing": prepared - started - preprocessed_data.seed_seconds,
                "seeding": preprocessed_data.seed_seconds,
            }
            empty.stats["fit_resolved"] = resolved
            empty.stats["fit_resolved_config"] = resolved_config
            return empty

        # Step 3: Initialize model and optimizer
        components = initialize_optimization(config, preprocessed_data)

        # Step 4: Create loss function
        loss_fn = create_loss_function(config, preprocessed_data, components.model)

        # Step 5: Run optimization loop
        optimization_results = run_optimization_loop(
            components, loss_fn, config, preprocessed_data
        )
        optimized = time.perf_counter()

        # Step 6: Finalize and return results
        result = finalize_results(optimization_results, config, preprocessed_data)
        finalized = time.perf_counter()
        quality_seconds = result.stats.pop("quality_seconds", 0.0)
        result.stats["fit_timings"] = {
            "preprocessing": prepared - started - preprocessed_data.seed_seconds,
            "seeding": preprocessed_data.seed_seconds,
            "optimization": optimized - prepared,
            "finalization": finalized - optimized - quality_seconds,
            "quality_render": quality_seconds,
        }
        result.stats["fit_resolved"] = resolved
        result.stats["fit_resolved_config"] = resolved_config
        return result


def _collect_fit_parameters(values: dict[str, Any]) -> FitParameters:
    """Build the internal parameter bundle from the public call arguments."""
    return FitParameters(
        **{field.name: values[field.name] for field in fields(FitParameters)}
    )


def _configuration_snapshot(parameters: Any) -> dict[str, Any]:
    """Keep scalar/short-array settings, never volume or seed payloads."""
    config: dict[str, Any] = {}
    excluded = {"V", "seeds", "iter_callback", "init_L", "init_amps"}
    for field in fields(parameters):
        if field.name in excluded:
            continue
        value = getattr(parameters, field.name)
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        elif isinstance(value, torch.device):
            value = str(value)
        elif isinstance(value, np.ndarray):
            if value.size > 16:
                continue
            value = value.tolist()
        ok, safe = json_safe_value(value)
        if ok:
            config[field.name] = safe
    return config


def _fit_provenance(
    parameters: FitParameters, result: GSplatData, device: torch.device
) -> dict[str, Any]:
    """A bounded, JSON-safe record of the fit that produced this leaf."""
    from luxar import __version__
    from luxar.gsplats.io.save_gsplats import _FITTING_INFO_KEYS

    config = _configuration_snapshot(parameters)
    config.update(result.stats.pop("fit_resolved_config", {}))
    seeds = parameters.seeds
    config["seeds"] = (
        seeds if isinstance(seeds, (int, float)) or seeds is None else "explicit"
    )
    config["floor_spec"] = config.get("floor")
    config.update(result.stats.pop("fit_resolved", {}))
    config["floor"] = result.stats.get("floor", config.get("floor"))
    config["device"] = str(device)
    outcome: dict[str, Any] = {}
    for key in _FITTING_INFO_KEYS:
        if (
            key in {"fit_provenance", "part_provenance", "_fit_scale_population_bits"}
            or key not in result.stats
        ):
            continue
        ok, safe = json_safe_value(result.stats[key])
        if ok:
            outcome[key] = safe
    outcome["n_splats"] = result.n_splats
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)
    return {
        "schema_version": 1,
        "command": [],
        "config": config,
        "environment": {
            "luxar_version": __version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "host": socket.gethostname(),
            "device": str(device),
            "device_name": device_name,
        },
        "timings": result.stats.pop("fit_timings", {}),
        "outcome": outcome,
        "loss_curve": result.stats.pop("loss_curve", []),
        "relocation_curve": result.stats.pop("relocation_curve", []),
    }


@arbol_warnings()
def fit_gaussian_splats(
    V: np.ndarray,
    seeds: Optional[np.ndarray | int | float | GSplatData] = None,
    norm_percentile: float = 0.0,
    floor: "str | float | None" = "auto",
    norm_range: "tuple[float, float] | None" = None,
    downscale: Optional[int | Sequence[int]] = None,
    init_sigma_vox: Optional[float] = None,
    n_iters: int = 1000,
    lr: float = 0.01,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 1.0,
    l1_amp: Optional[float] = None,
    l1_diag: Optional[float] = None,
    sigma_min_diag: Optional[Sequence[float] | float] = DEFAULT_SIGMA_MIN_DIAG,
    sigma_max_diag: Optional[Sequence[float] | float] = None,
    amp_max: Optional[float] = None,
    max_eccentricity: Optional[float] = 10.0,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    device: Optional[str] = None,
    seed_method: str = "auto",
    verbose: bool = True,
    # Optimization parameters
    max_abs_error: Optional[float] = None,
    rel_l2_target: Optional[float] = None,
    gradient_clip: Optional[float] = None,  # Disabled: MSE gradients are well-scaled
    # Scheduler parameters (empirically tuned):
    # patience 15 + factor 0.9 gives faster LR decay than the gentler 25/0.98,
    # yielding 33% speed improvement with same PSNR.
    scheduler_type: str = "plateau",
    patience: int = 15,
    lr_reduction_factor: float = 0.9,
    early_stop_patience: Optional[int] = 300,
    early_stop_rel_tol: Optional[float] = None,
    # Dynamic operations parameters
    enable_dynamic_ops: bool = True,
    dynamic_config: Optional[DynamicOpsConfig] = None,
    dynamic_ops_verbose: bool = False,
    relocation_reseed_sigma_vox: Optional[float] = None,
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    # Hardware acceleration
    use_metal: bool = True,
    use_cuda: bool = True,
    # Post-processing
    cull_retention: float | None = 0.999,
    min_extent: float | None = DEFAULT_MIN_EXTENT_VOX,
    min_extent_max_mass_share: float = DEFAULT_MIN_EXTENT_MAX_MASS_SHARE,
    voxel_footprint_correction: bool | float = False,
    # Boundary containment
    boundary_penalty: Optional[float] = None,
    clip_to_bounds: bool = False,
    # Anisotropic voxel spacing
    voxel_size: Optional[Sequence[float] | float] = None,
    output_space: str = "real",
    # Z-order sorting for memory locality
    sort_splats_enabled: bool = True,
    sort_splats_interval: int = 1000,
    # Per-iteration callback (e.g. validation-set scoring during fitting)
    iter_callback: Optional[Any] = None,
    iter_callback_every: int = 25,
    seed_amps_background_relative: bool = False,
    seed_space: Optional[str] = None,
    # Element type of the volume as it was ACQUIRED / stored on disk
    source_dtype: Optional[str] = None,
    source_shape: Optional[Sequence[int]] = None,
    source_stored_bytes: Optional[int] = None,
    **seed_kwargs: Any,
) -> GSplatData:
    """
    Fit n-dimensional oriented Gaussian splats to reconstruct input image/volume.

    Uses standard PyTorch Adam (fused on CUDA when available) combined with
    fixed-pool splat relocation: the least informative splats are periodically
    moved to regions of high reconstruction residual, which keeps optimizer
    tensor shapes constant and avoids per-splat state management.

    The optimization uses:
    - Standard PyTorch Adam with gradient-dilution-compensated learning rates
    - Center position (bounded to image domain via sigmoid)
    - Non-negative amplitude (via softplus activation)
    - Covariance matrix Σ = L @ L^T where L is the Cholesky factor
    - Efficient rendering via batched triangular solve

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to reconstruct. Will be normalized to [0,1].
    seeds : np.ndarray (N, d), int, float, GSplatData, or None
        Initial seed center positions, count, compression ratio, or full
        warm-start dataset.

        - If np.ndarray: Explicit seed centers in voxel coordinates
        - If int: Exact number (keeps highest intensity if more detected)
        - If float (0 < seeds <= 1.0): Compression ratio - the ratio of floats
          used to represent Gaussian splats over total image floats. For example,
          seeds=0.1 targets a representation using 10% of the original storage.
          The number of splats is computed as: n = ratio × total_voxels / floats_per_splat
          where floats_per_splat = d + d×(d+1)/2 + 1 (center + Cholesky + amp).
        - If GSplatData: Explicit seeds or a warm start (centers + Cholesky +
          amplitudes carried over directly). Both ``generate_seeds()`` output and
          a previously fitted result go through this door — they differ in
          amplitude scale, so declare it with
          ``seed_amps_background_relative`` (below).
        - If None: Auto-generated using dimension-aware intelligent defaults:
          * Universal scales: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) for comprehensive detection
          * Volume-proportional density: ~1% of voxels as seeds
          * Inclusive threshold: percentile_thresh=70 for broad feature coverage
    norm_percentile : float, default=0.0
        Normalization method for handling outliers and noise:
        - 0.0: Full min-max range (maximum dynamic range, sensitive to outliers)
        - >0: Percentile clipping (e.g., 1.0 uses 1%-99% range, robust to outliers)
        Higher values provide more outlier robustness but may clip important data.
    floor : str, float, or None, default="auto"
        Background floor / DC-offset suppression, applied before normalization
        (subtracts a constant pedestal that a localized-Gaussian basis cannot
        represent efficiently). Raises the effective image_min so sub-floor
        intensity clips to 0.

        - "auto": histogram-mode estimate (capped at the median; a no-op on
          clean data with no pedestal).
        - "pN" (e.g. "p10"): the Nth percentile of non-zero intensities.
        - float: a fixed intensity value.
        - "none" / 0 / None: disabled (today's hard-min normalization).

        Orthogonal to ``norm_percentile`` unless the floor overtakes its
        percentile-derived high endpoint; then image_max expands to the data
        maximum and bright-outlier clipping is dropped to preserve signal.
    norm_range : tuple of float, or None, default=None
        Explicit ``(image_min, image_max)`` for normalization, replacing the
        pair ``norm_percentile`` would derive from ``V`` itself. Tiled fitting
        passes a range resolved against the WHOLE volume
        (:func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_norm_range`)
        so every tile maps a given physical intensity to the same normalized
        value, and is therefore held to the same absolute convergence tolerance
        and thresholds. Must be finite with ``image_max > image_min``. Because
        such a range is a bounded-sample estimate, a voxel above ``image_max``
        is left unclipped when ``norm_percentile == 0`` (and the auto ``amp_max``
        rises with it), rather than flattening the brightest structure. Leave
        None for a whole-volume fit — the array already IS the volume.
    downscale : int, sequence of int, or None, default=None
        Downsample the volume by integer factor(s) before fitting.
        Useful for band-limited data where high-frequency voxels contain only noise.
        A Gaussian anti-alias filter (sigma = factor/2) is applied before decimation.
        - If int: Isotropic downscale (e.g., ``downscale=4`` reduces all axes by 4x).
        - If sequence: Per-axis factors (e.g., ``downscale=(1, 4, 4)`` for anisotropic).
        - If None: No downscaling (default).
        Fitted splat parameters are automatically rescaled to original coordinates.
    init_sigma_vox : float or None, default=None
        Initial isotropic standard deviation for Gaussian splats (in voxels).
        If None, uses scale-informed initialization from seeding methods.
        If no scale info available, auto-computes based on image size (~5% of
        smallest dimension, min 1.5).
    n_iters : int, default=1000
        Maximum number of optimization iterations. Default is generous to allow
        max_abs_error convergence criterion to work effectively.
    lr : float, default=0.01
        Learning rate for Adam optimizer.
    loss_type : str, default="l1"
        Loss function: "l1" (default; robust to outliers, preserves sharp features;
        in the loss-comparison study, Supp. Doc. 5, L1 beats MSE on held-out PSNR
        on 11 of 17 microscopy volumes and never trails it by more than 0.28 dB),
        "mse" (the global unregularized MSE minimizer maximizes training PSNR, but
        finite-iteration regularized fits do not reach it), or "poisson" (natural
        for count/photon data; stops in the fewest iterations on most confocal
        volumes, 8-15x shorter wall time than MSE where MSE runs to the iteration
        cap, at a held-out cost vs L1 of up to ~2.9 dB).
    asymmetric_penalty : float, default=1.0
        Over-prediction penalty factor for asymmetric loss. Multiplies loss for regions
        where pred > target by this factor. Set to None to disable asymmetric loss.
        Default 1.0 (symmetric); progressive fitting uses 10.0 for residual passes to
        prevent locked-in overshoot.
    l1_amp : float, default=None (auto: 0.1 * lr)
        L1 regularization coefficient on splat amplitudes for sparsity.
        If None, automatically set to 10% of learning rate for consistent
        sparsity pressure that scales with optimization strength.
    l1_diag : float, default=None (auto: 0.01 * lr)
        L1 regularization coefficient on diagonal elements of Cholesky factors.
        Encourages smaller, more isotropic splats. If None, automatically set
        to 1% of learning rate for mild shape regularization.
    sigma_min_diag : Sequence[float] | float, optional
        Minimum diagonal values for Cholesky factor L along each axis. A single
        float is broadcast across all dimensions.
        Defaults to sqrt(1/12) ≈ 0.289 (1-voxel box footprint) to allow
        single-voxel splats while preventing degeneracy.
    sigma_max_diag : Sequence[float] | float, optional
        Maximum diagonal values for Cholesky factor L along each axis.

        - If Sequence[float]: Per-axis absolute bounds (one per dimension).
        - If float: Fraction of volume extent per axis. Each dimension gets
          ``shape[i] * fraction`` independently. E.g., ``sigma_max_diag=1/16``
          on a (50, 200, 300) volume gives ``[3.125, 12.5, 18.75]``.
    amp_max : float or None, default=None (auto: 1.0)
        Maximum amplitude constraint for splats. Prevents amplitude explosion
        during optimization, especially with aggressive compression (few splats).
        Since the image is normalized to [0, 1], a value of 1.0 matches the max
        possible intensity. If None, automatically set to 1.0. Set to higher
        values (e.g., 2.0) for more flexibility, or lower (e.g., 0.5) for tighter
        control.
    max_eccentricity : float or None, default=10.0
        Maximum ratio of longest to shortest axis for splat covariance. Limits
        anisotropy by constraining diagonal elements of Cholesky factor L so that
        max(diag)/min(diag) <= sqrt(max_eccentricity). For example, 2.0 means the
        longest axis can be at most sqrt(2) ≈ 1.41x the shortest axis.
    truncate : float, default=2.75
        Truncation radius in standard deviations for rendering efficiency.
    device : str, optional
        PyTorch device ("auto", "cpu", "cuda", "mps"). Auto-detects for
        ``None`` or ``"auto"``.
    seed_method : str, default="auto" (RECOMMENDED)
        Method for generating seeds when seeds=None:

        - **"auto"** (DEFAULT, RECOMMENDED): Fast edges + grid combination.
          Provides good convergence by capturing boundaries (edges) and spatial
          coverage (grid). Decomposition is excluded by default for speed.
          Budget allocation: ~60% edges, ~40% grid.

        - "decomposition": Multi-scale decomposition for blob-like features (slow).
          Captures global structure but may miss boundaries and fine details.

        - "grid": Uniform grid seeding for spatial coverage.
          Fast and simple, good for uniform textures.

        - "edges": Edge-based seeding with anisotropic shapes.
          Good for images with clear boundaries and structure.

        - Comma-separated combinations (e.g., "decomposition,edges,grid").

        This parameter is only used when seeds=None. If seeds are provided explicitly,
        this parameter is ignored.
    **seed_kwargs
        Additional keyword arguments for seed generation (e.g., num_scales,
        percentile_thresh, scale_voxels, etc.). Only used when seeds=None.
        See seed generation functions for available options.
    verbose : bool, default=True
        Whether to print optimization progress.
    max_abs_error : float or None, default=None (auto: 0.01)
        Maximum absolute error threshold for convergence. If specified,
        optimization stops when ``max(|prediction - target|)`` < max_abs_error.
        If None, automatically set to 0.01 (1% of normalized [0,1] range) for
        sensible convergence behavior. Auto-threshold usage is logged.
    rel_l2_target : float or None, default=None
        Relative L2 error threshold for convergence. If specified,
        optimization stops when ``||pred - target||₂ / ||target||₂`` < rel_l2_target.
        This is an additional (OR) criterion alongside max_abs_error — either
        being satisfied triggers convergence. Provides a smoother, more stable
        convergence signal than max_abs_error. If None, this criterion is disabled.
    gradient_clip : float or None, default=None
        Maximum gradient norm for clipping. None disables clipping (default for
        MSE loss where gradients are inherently well-scaled by error magnitude).
    scheduler_type : str, default="plateau"
        Type of learning rate scheduler ("plateau" or "exponential").
    patience : int, default=15
        Scheduler patience: iterations without loss improvement before LR reduction.
    lr_reduction_factor : float, default=0.9
        LR multiplier on plateau (new_lr = lr × lr_reduction_factor).
        Examples: 0.5=halve LR, 0.1=reduce to 10%, 0.9=moderate reduction.
    early_stop_patience : Optional[int], default=300
        Early stopping: stop if no loss improvement for N iterations.
        None disables early stopping (runs until convergence or iteration limit).
        Example: 300 stops if no improvement for 300 consecutive iterations.
    early_stop_rel_tol : Optional[float], default=None
        Relative early stop. When set, the fit also stops once the best loss
        has improved by less than this fraction (of the best loss one window
        earlier) over the last ``early_stop_patience`` iterations. None keeps
        the plain rule. Ignored when ``early_stop_patience`` is None.
    enable_dynamic_ops : bool, default=True
        Enable dynamic operations (seeding and pruning).
    dynamic_config : DynamicOpsConfig, optional
        Configuration for dynamic operations. Uses defaults if None.
    dynamic_ops_verbose : bool, default=False
        Enable detailed console logging for dynamic operations. Shows residual analysis,
        seeding attempts and pruning operations.
    relocation_reseed_sigma_vox : float or None, default=None
        Isotropic sigma (voxels) a relocated splat is reseeded at; overrides
        ``dynamic_config.relocation_reseed_sigma_vox``. ``None`` keeps the 0.5
        voxel default. Larger values (0.75-1.0) leave no sub-voxel specks and
        finish faster, but on sparse stacks they cost up to ~0.07 dB held-out
        and on dense ones they can gain ~0.2-0.4 dB; ``min_extent`` removes
        the specks of the default reseed at no measurable cost.
    napari_movie : bool, default=False
        Record optimization movie for napari visualization.
        Enable this to create a time-series visualization of optimization progress.
    movie_every : int, default=1
        Record movie frame every N iterations.
    movie_max_frames : int, default=None (infinite)
        Maximum number of movie frames to store in memory. Older frames are automatically
        removed when this limit is exceeded, preventing memory exhaustion during long optimizations.
    use_metal : bool, default=True
        Enable Metal acceleration on Apple Silicon (macOS + MPS device).
        Provides substantial speedup for 3D volumes (chip-dependent).
        Automatically disabled if not available.
    use_cuda : bool, default=True
        Enable custom CUDA kernels on NVIDIA GPUs.
        Provides substantial speedup for 2D-8D volumes (often orders of
        magnitude, GPU-dependent). Automatically disabled if not available.
    cull_retention : float or None, default=0.999
        Post-fit cumulative culling.  Keeps the top splats that account for
        this fraction of the total amplitude (0--1).  At 0.999, the discarded
        splats collectively contribute at most 0.1% of total amplitude.
        Set to ``None`` to disable.
    min_extent : float or None, default=0.6
        After the fit (before ``cull_retention``), drop splats whose
        geometric-mean SPATIAL sigma is below this many FITTED-grid voxels
        (``voxel_size`` and ``downscale`` are undone first; a zero-variance
        stacked axis is ignored). These are relocation "specks" stuck near the
        0.5-voxel reseed size, which render as glitter; no polish follows.
        Measured on nine fits: -8 to -70% splats, held-out foreground PSNR
        within -0.07 dB (+0.40 dB on a Tribolium light-sheet stack). The count
        and mass share removed are recorded as ``stats["min_extent_removed"]``
        / ``stats["min_extent_removed_mass_share"]``, and the stored quality is
        re-scored on the pruned set. ``0`` or ``None`` disables. In a native
        nD fit a fitted (non-stacked) time axis counts in the geometric mean.
    min_extent_max_mass_share : float, default=0.02
        Guard for ``min_extent``: when pruning would remove more than this
        share of the fitted brightness, the splats are kept and a note is
        printed (``stats["min_extent_skipped"] = True``). Under ``floor="auto"``
        the specks carry 0.5-0.75%; after a high floor (``p95``) the same
        threshold removed 4.3-15%, i.e. real structure fitted small.
    voxel_footprint_correction : bool | float, default=False
        Post-fit correction to inflate splat covariances by the voxel footprint.
        This ensures that upsampling doesn't invent detail beyond what the original
        discrete data can represent. The correction adds sigma^2 to covariance diagonals:
        Sigma_new = Sigma_original + sigma^2 * I_d
        - False: Disabled (default)
        - True: Enable with 1-voxel box footprint (sigma ≈ 0.289 voxels)
        - float: Custom sigma in voxel units (e.g., 0.5 for half-voxel blur, 1.0 for 1-voxel blur)
        Works for any dimension d.
    boundary_penalty : float or None, default=None
        Weight for boundary containment penalty during optimization.
        Adds a differentiable penalty for splats whose effective support
        (truncate * sqrt(Sigma_ii)) extends beyond the volume bounds.
        The penalty is: boundary_penalty * mean(overflow^2).
        - None: Disabled (default)
        - float > 0: Enable with this weight (e.g., 0.1 for mild, 1.0 for strong)
    clip_to_bounds : bool, default=False
        Post-fit hard clipping to guarantee no splat extends beyond the volume bounds.
        Scales down rows of the Cholesky factor L so that
        truncate * sqrt(Sigma_ii) <= distance_to_nearest_edge for each dimension.
        Preserves splat orientation but shrinks to fit within bounds.
    voxel_size : Sequence[float] | float, optional
        Physical voxel spacing per axis (e.g., ``(5.0, 1.0, 1.0)`` for Z-anisotropic
        microscopy). A scalar means isotropic spacing. Affects:
        - ``max_eccentricity``: evaluated in physical space
        - Auto ``init_sigma``: based on physical dimensions
        - Output coordinates: converted to physical space (see ``output_space``)
        If None (default), all voxels are treated as unit-spaced.
    output_space : str, default="real"
        Coordinate system for output Gaussians:

        - ``"real"``: Physical coordinates (centers and Cholesky scaled by voxel_size).
          When voxel_size is None, identical to ``"voxel"``.
        - ``"voxel"``: Raw voxel indices (no conversion).
    seed_amps_background_relative : bool, default=False
        Which intensity convention the amplitudes of a ``seeds=GSplatData``
        carry. Ignored for every other kind of ``seeds``.

        - False (default): RAW-IMAGE-SAMPLED — the amplitudes were sampled off the
          original volume (up to a fixed seeding scale factor), background
          pedestal included. This is what
          ``generate_seeds()`` returns, i.e. the explicit-seeding workflow
          (``fit_gaussian_splats(V, seeds=generate_seeds(V))``). The seeding
          factor ``s`` is divided out, the pedestal removed and ``s``
          re-applied (``(a / s - image_min) * s / intensity_range``), so an
          active ``floor`` is subtracted exactly once and ``s`` never pushes a
          seed on a pedestal below it. Hand-built amplitudes in raw image
          units should be multiplied by ``SEED_AMPLITUDE_SCALE`` first, or
          passed through ``init_amps`` instead.
        - True: BACKGROUND-RELATIVE — the amplitudes already have the pedestal
          removed. This is what a previous fit returns (the fit's output
          amplitudes are the normalized ones times ``intensity_range``, with
          ``image_min`` never added back), hence also what a ``.gsplats.zarr``
          WRITTEN BY a fit (``gsplat fit`` / ``gsplat lod``) carries. They are
          rescaled as ``a / intensity_range``.

        A store that was IMPORTED (``gsplat import`` maps PLY/SPZ opacity into
        roughly [0, 1]) or intensity-rescaled (``gsplat transform
        --normalize-intensity`` / ``--scale-intensity``) carries neither
        convention exactly, so its warm start is approximate either way.

        Getting this wrong is silent: declaring False on a fit's output makes an
        active ``floor`` be subtracted twice, initializing every seed dimmer than
        the floor to exactly 0; declaring True on raw amplitudes starts every
        seed too bright by ``floor / intensity_range`` (#1172).
    seed_space : {"voxel", "real"} or None, default=None
        Coordinate frame of a ``seeds=GSplatData``'s centers and Cholesky
        factors. Ignored for every other kind of ``seeds`` (an ndarray of
        centers is always voxel indices).

        - ``"voxel"``: voxel indices — what ``generate_seeds()`` returns and what
          a fit with ``output_space="voxel"`` (or no ``voxel_size``) writes.
        - ``"real"``: physical units — what a fit with ``voxel_size`` and the
          default ``output_space="real"`` writes. Divided back by this call's
          ``voxel_size`` before fitting (an identity when ``voxel_size`` is None).
        - None (default): a seed DECLARED to be a fit's output
          (``seed_amps_background_relative=True``) is taken to be in this
          call's ``output_space``, so a result fed straight back with the same
          settings round-trips exactly; any other seed is ``"voxel"``.

        Getting this wrong is silent and severe: physical-unit seeds read as
        voxel indices land at the wrong positions with the wrong shapes (a
        24^3 warm start fell from 47 dB to 18 dB).
    source_dtype : str or None, default=None
        Element type of the volume as it was ACQUIRED (e.g. ``"uint16"``), when
        that differs from ``V.dtype``. Recorded in ``stats["source_dtype"]`` /
        ``stats["source_bytes"]`` — the honest denominator of a compression
        ratio. Pass it when the volume has already been cast to float before
        reaching here (as ``luxar.io.volume.load_volume`` does), otherwise the
        recorded source size describes the float working copy and overstates
        compression by the cast's inflation factor. If None, ``V.dtype`` is used.
    source_shape : sequence of int or None, default=None
        Grid of the ACQUISITION this fit represents, when the caller
        preprocessed before fitting. Most producers do: a demo that pulls one
        channel out of a 5D OME-Zarr, downscales it and normalizes it hands over
        an array that is no longer the data anyone means by "the source", so
        measuring ``V`` would quote compression against the working copy. Pair it
        with ``source_dtype`` — a declared grid with the cast's dtype is still
        the wrong denominator. Recorded with ``stats["source_declared"] = True``
        so a reader can tell a stated grid from a measured one. If None, ``V``'s
        own shape is used, which is correct whenever nothing was preprocessed.

    Returns
    -------
    GSplatData
        Dataclass containing all fitting results:
        - centers: np.ndarray, shape (N, d) - Center positions (physical or voxel, see output_space)
        - amplitudes: np.ndarray, shape (N,) - Non-negative amplitudes rescaled to original intensity
        - cholesky_factors: np.ndarray, shape (N, d*(d+1)//2) - Packed lower-triangular Cholesky factors
        - stats: Dict[str, Any] - Optimization statistics (time, iterations, convergence, etc.)

        All arrays represent the BEST state encountered during optimization (lowest loss).
        Note: Gaussian splatting cannot represent uniform DC components - only variations.

    Notes
    -----
    The optimization uses standard PyTorch Adam combined with fixed-pool
    splat relocation:
    - Gradient-dilution-compensated learning rates for dimensional consistency
    - Periodic relocation of low-importance splats to high-residual regions
    - Optimizer state reset for relocated splats; all others untouched
    - Early stopping and adaptive learning-rate scheduling
    """

    # Internal tile/box handoff; never forward it to seed generation or expose it
    # as a fit-config field.
    carry_scale_population = bool(seed_kwargs.pop("_carry_scale_population", False))
    # Keep this first: the bundle must capture the public arguments before rebinding.
    parameters = _collect_fit_parameters(locals())
    min_extent_vox, dynamic_config = _resolve_hygiene_options(
        min_extent, relocation_reseed_sigma_vox, dynamic_config
    )
    from luxar.gsplats.fitting.splat_hygiene import validate_max_mass_share

    max_mass_share = validate_max_mass_share(min_extent_max_mass_share)
    _warn_reseed_without_relocation(relocation_reseed_sigma_vox, enable_dynamic_ops)

    with asection("Fitting Gaussian Splats"):
        fitter = GaussianSplatFitter(
            device=device,
            enable_dynamic_ops=enable_dynamic_ops,
            dynamic_config=dynamic_config,
            use_metal=use_metal,
            use_cuda=use_cuda,
        )

        # Fit and extract results
        result = fitter.fit(parameters)
        resolved_device = fitter.device

    # Record the spacing the centers were multiplied by, so `compare`/`render`
    # can put physical-unit splats back on the voxel grid they were fitted on.
    if result.stats:
        from luxar.gsplats.scoring_grid import physical_voxel_size_stamp

        result.stats.update(
            physical_voxel_size_stamp(voxel_size, output_space, int(V.ndim))
        )

    # Explicitly release the fitter and its GPU resources (optimizer state,
    # model weights, preprocessed V_tensor) before post-fit operations.
    # Also clear the rendering grid cache which accumulates GPU tensors
    # for each unique AABB box shape seen during optimization.
    from luxar.gsplats.models.gsplats.rendering_core import clear_grid_cache

    del fitter
    clear_grid_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # After fitting section closes, show summary and movie
    if verbose:
        from arbol import aprint

        with asection("Optimization Complete"):
            if "time_seconds" not in result.stats:
                # No seed candidates: GaussianSplatFitter.fit returns an empty
                # result with an empty stats dict before any optimization runs
                # (a signal-free tile, or an empty GSplatData warm start), so
                # there is no timing/iteration/convergence record to report.
                aprint("No seed candidates — returned an empty result")
            else:
                aprint(f"Time: {result.stats['time_seconds']:.2f} seconds")
                aprint(f"Iterations: {result.stats['iterations']}/{n_iters}")
                if result.stats["converged"]:
                    aprint(
                        f"✓ Converged (saved {n_iters - result.stats['iterations']} iterations)"
                    )

        # No compression figure here: the one definition Luxar quotes is the
        # decoded source over the SAVED store's bytes on disk, which exist only
        # once the (post-cull) result is written — the CLI reports it there,
        # and `luxar gsplat info` reports it for any store.

    # Show napari movie OUTSIDE the fitting section (so it doesn't affect timing)
    if result.stats.get("movie_frames") is not None:
        from luxar.gsplats.fitting.visualization import show_optimization_movie

        show_optimization_movie(
            result.stats["movie_frames"], result.stats["movie_shape"]
        )

    # Post-fit reductions: speck pruning, then the cumulative cull, then ONE
    # re-score so the stored quality describes the saved splat set.
    result = _post_fit_reductions(
        result,
        V,
        cull_retention=cull_retention,
        min_extent_vox=min_extent_vox,
        max_mass_share=max_mass_share,
        downscale=downscale,
        verbose=verbose,
        device=device,
        voxel_size=voxel_size,
        output_space=output_space,
    )

    if not carry_scale_population:
        from luxar.gsplats.fitting.scale_population import finish_population_stats

        finish_population_stats(result.stats)
    result.stats["fit_provenance"] = _fit_provenance(
        parameters, result, resolved_device
    )
    return result


def _warn_reseed_without_relocation(
    relocation_reseed_sigma_vox: Optional[float], enable_dynamic_ops: bool
) -> None:
    """A reseed sigma only acts on relocated splats; say so when nothing relocates."""
    if relocation_reseed_sigma_vox is not None and not enable_dynamic_ops:
        from arbol import aprint

        aprint(
            "⚠ relocation_reseed_sigma_vox / --reseed-sigma has no effect: "
            "relocation (dynamic ops) is off for this fit"
        )


def _resolve_hygiene_options(
    min_extent: "float | None",
    relocation_reseed_sigma_vox: Optional[float],
    dynamic_config: Optional[DynamicOpsConfig],
) -> "tuple[float, Optional[DynamicOpsConfig]]":
    """Validate ``min_extent`` and fold an explicit reseed sigma into the config."""
    from dataclasses import replace

    from luxar.gsplats.fitting.splat_hygiene import (
        validate_min_extent,
        validate_reseed_sigma,
    )

    min_extent_vox = validate_min_extent(min_extent)
    reseed = validate_reseed_sigma(relocation_reseed_sigma_vox)
    if reseed is not None:
        dynamic_config = replace(
            dynamic_config or DynamicOpsConfig(), relocation_reseed_sigma_vox=reseed
        )
    return min_extent_vox, dynamic_config


def _post_fit_reductions(
    result: GSplatData,
    V: np.ndarray,
    *,
    cull_retention: float | None,
    min_extent_vox: float,
    max_mass_share: float = DEFAULT_MIN_EXTENT_MAX_MASS_SHARE,
    downscale: Optional[int | Sequence[int]],
    verbose: bool,
    device: Optional[str],
    voxel_size: Optional[Sequence[float] | float],
    output_space: str,
) -> GSplatData:
    """Speck pruning + cumulative cull, then one re-score of the saved set."""
    timings = result.stats.get("fit_timings", {})
    if result.n_splats == 0:
        timings.update(pruning=0.0, culling=0.0, post_cull_quality=0.0)
        return result
    removed = 0
    stage_started = time.perf_counter()
    if min_extent_vox > 0:
        from luxar.gsplats.fitting.splat_hygiene import fitted_voxel_unit

        unit = fitted_voxel_unit(
            result.ndim,
            voxel_size=voxel_size,
            output_space=output_space,
            downscale=downscale,
        )
        result, n = prune_min_extent(
            result, min_extent_vox, unit, verbose=verbose, max_mass_share=max_mass_share
        )
        removed += n
    timings["pruning"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    if cull_retention is not None and 0 < cull_retention < 1.0 and result.n_splats > 0:
        result, n = _post_fit_cull(result, cull_retention, verbose=verbose)
        removed += n
    timings["culling"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    if removed > 0 and "psnr_db" in result.stats and result.n_splats > 0:
        _rescore_saved_model(
            result,
            V,
            verbose=verbose,
            device=device,
            voxel_size=voxel_size,
            output_space=output_space,
        )
    timings["post_cull_quality"] = time.perf_counter() - stage_started
    result.stats["fit_timings"] = timings
    return result


def prune_min_extent(
    result: GSplatData,
    min_extent_vox: float,
    unit: np.ndarray,
    *,
    verbose: bool = False,
    max_mass_share: float = DEFAULT_MIN_EXTENT_MAX_MASS_SHARE,
) -> "tuple[GSplatData, int]":
    """Drop splats whose spatial extent is below ``min_extent_vox`` (WP9).

    ``unit`` is one fitted-grid voxel in the result's output units
    (:func:`~luxar.gsplats.fitting.splat_hygiene.fitted_voxel_unit`). Like the
    closing cull this is part of producing the artifact, so the fit's measured
    scores ride across it (the caller re-scores) and the scale-population
    classification is reduced by the kept mask. Records the pruning in
    ``stats`` even when nothing was removed. When the splats to drop carry
    more than ``max_mass_share`` of the brightness, nothing is removed, a note
    is printed whatever ``verbose`` says, and ``stats["min_extent_skipped"]``
    is True. Returns ``(result, n_removed)``.
    """
    from arbol import aprint

    from luxar.gsplats._data.filtering import (
        measured_stats_snapshot,
        restore_measured_stats,
    )
    from luxar.gsplats.fitting.scale_population import reduce_population_stats
    from luxar.gsplats.fitting.splat_hygiene import (
        format_min_extent_line,
        integrated_mass,
        min_extent_keep_mask,
        skipped_note,
        stamp_min_extent,
    )

    keep = min_extent_keep_mask(result, min_extent_vox, unit)
    mass = integrated_mass(result)
    input_mass = float(mass.sum())
    n_removed = int(keep.size - np.count_nonzero(keep))
    removed_mass = float(mass[~keep].sum())
    share = removed_mass / input_mass if input_mass > 0 else 0.0
    if n_removed > 0 and share > max_mass_share:
        aprint(skipped_note(n_removed, share, max_mass_share))
        stamp_min_extent(
            result.stats,
            min_extent_vox=min_extent_vox,
            removed=0,
            removed_mass=0.0,
            input_mass=input_mass,
            skipped=True,
        )
        return result, 0
    if n_removed > 0:
        measured = measured_stats_snapshot(result)
        before_stats = result.stats
        result = result.filter(keep)
        restore_measured_stats(result, measured)
        reduce_population_stats(before_stats, result.stats, keep)
        result.stats["n_splats"] = result.n_splats
    stamp_min_extent(
        result.stats,
        min_extent_vox=min_extent_vox,
        removed=n_removed,
        removed_mass=removed_mass,
        input_mass=input_mass,
    )
    if verbose:
        line = format_min_extent_line(result.stats)
        if line:
            aprint(line)
    return result, n_removed


def _post_fit_cull(
    result: GSplatData,
    cull_retention: float,
    *,
    verbose: bool,
) -> "tuple[GSplatData, int]":
    """Cumulative post-fit cull; the caller re-scores the saved set.

    The fit's quality metrics were measured on the pre-cull splats. Carrying
    them over (#3107) keeps a store from publishing no PSNR at all, but it
    labels the saved model with a score for a different splat set. So when the
    closing reductions removed anything, :func:`_post_fit_reductions` renders
    the saved set once more and its scores replace the carried ones; only if
    that re-score fails do the pre-cull scores stand (and the console says so).
    Optimizer residuals (`final_loss`, ...) and the iteration counters still
    describe the run that produced them. Returns ``(result, n_removed)``.
    """
    from luxar.gsplats._data.filtering import (
        measured_stats_snapshot,
        restore_measured_stats,
    )
    from luxar.gsplats.fitting.scale_population import reduce_population_stats

    n_before = result.n_splats
    amp_before = float(np.sum(result.amplitudes))
    # `cull` drops the measured scores (they describe the splat set they were
    # taken on — see content_scoped_stats), but this trim is the last step of
    # the FIT: the alternative to carrying the measurement over is a fit that
    # publishes no PSNR at all.
    measured = measured_stats_snapshot(result)
    before_stats = result.stats
    kept: list[np.ndarray] = []
    result = result.cull(
        method="cumulative", retention=cull_retention, keep_mask_out=kept
    )
    restore_measured_stats(result, measured)
    reduce_population_stats(before_stats, result.stats, kept[0])
    result.stats["n_splats"] = result.n_splats
    n_removed = n_before - result.n_splats
    amp_after = float(np.sum(result.amplitudes))
    amp_retained_pct = 100.0 * amp_after / amp_before if amp_before > 0 else 100.0
    if verbose:
        from arbol import aprint

        aprint(
            f"Post-fit culling (cumulative, retention={cull_retention:.1%}): "
            f"{n_before} -> {result.n_splats} splats "
            f"(removed {n_removed}, {100.0 * n_removed / n_before:.1f}%; "
            f"amplitude retained: {amp_retained_pct:.1f}%)"
        )
    return result, n_removed


#: The ``compute_quality_metrics`` keys a post-cull re-score refreshes.
_RESCORED_QUALITY_KEYS = (
    "mse",
    "psnr_db",
    "psnr_robust_db",
    "ssim",
    "ssim_robust",
    "foreground_psnr_db",
    "foreground_psnr_robust_db",
    "foreground_threshold",
    "foreground_fraction",
)


def _voxel_space_view(
    result: GSplatData,
    voxel_size: Optional[Sequence[float] | float],
    output_space: str,
) -> GSplatData:
    """``result`` on the fitted volume's voxel grid (undo a real-space scale)."""
    if output_space != "real" or voxel_size is None:
        return result
    d = result.centers.shape[1]
    vs = np.broadcast_to(np.asarray(voxel_size, dtype=np.float64), (d,))
    tril_scales = np.concatenate([[vs[i]] * (i + 1) for i in range(d)])
    return GSplatData(
        centers=(result.centers / vs).astype(np.float32),
        amplitudes=result.amplitudes,
        cholesky_factors=(result.cholesky_factors / tril_scales).astype(np.float32),
        truncation_radius=result.truncation_radius,
    )


def _rescore_saved_model(
    result: GSplatData,
    V: np.ndarray,
    *,
    verbose: bool,
    device: Optional[str],
    voxel_size: Optional[Sequence[float] | float],
    output_space: str,
    scorer: "Optional[Callable[[torch.Tensor], dict[str, float]]]" = None,
) -> bool:
    """Re-measure ``result`` against ``V`` on the fit's basis, in place.

    Mirrors the scoring in ``fitting/results.py::finalize_results``: render on
    ``V``'s own voxel grid (undoing the physical-coordinate scale when the fit
    wrote real-space output) and, by default, compare against
    ``V - image_min`` with ``compute_quality_metrics``. A fitter with its own
    score definition (the progressive fitter's chunked PSNR) passes ``scorer``,
    which maps the render to the stats it refreshes.

    Never raises. On failure the carried (pre-cull) scores stand, one warning
    line is printed whatever ``verbose`` says, and the reason is recorded as
    ``stats["quality_error"]`` — the key ``finalize_results`` uses when its own
    scoring fails. Returns whether the re-score succeeded.
    """
    from arbol import aprint

    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    pre_stats = dict(result.stats)
    scored = _voxel_space_view(result, voxel_size, output_space)
    try:
        dev = str(resolve_torch_device(device))
        with torch.no_grad():
            rendered = render_to_volume_tensor(
                scored,
                shape=V.shape,
                device=dev,
                truncate=result.truncation_radius,
            )
            if scorer is None:
                quality = _default_fit_scorer(V, result.stats.get("image_min"))(
                    rendered
                )
            else:
                quality = scorer(rendered)
            del rendered
    except Exception as exc:  # noqa: BLE001 - scoring must never fail a fit
        reason = f"{type(exc).__name__}: {exc}"
        result.stats["quality_error"] = reason
        aprint(
            f"Warning: post-cull quality re-score skipped ({reason}); the stored "
            "quality is the PRE-cull (pre-pruning) measurement"
        )
        return False
    result.stats.update(quality)
    result.stats.pop("quality_error", None)
    if verbose:
        aprint(_post_cull_quality_line(quality, pre_stats))
    return True


def _post_cull_quality_line(quality: dict[str, float], pre: dict[str, Any]) -> str:
    """The post-cull summary, headlined like the fit's own (robust peak first).

    A scorer that reports no robust figure falls back to the full-range one.
    """
    key = "psnr_robust_db" if "psnr_robust_db" in quality else "psnr_db"
    new_psnr = quality[key]
    pre_psnr = pre.get(key)
    delta = (
        f" ({new_psnr - pre_psnr:+.2f} dB vs fitted)" if pre_psnr is not None else ""
    )
    fg_key = (
        "foreground_psnr_robust_db"
        if "foreground_psnr_robust_db" in quality
        else "foreground_psnr_db"
    )
    peak = (
        f" robust peak ({quality['psnr_db']:.1f} dB full range)"
        if key == "psnr_robust_db"
        else ""
    )
    extra = _ssim_mse_summary(quality)
    return (
        f"Quality (saved model, after post-fit pruning/cull): "
        f"PSNR={new_psnr:.1f} dB{peak}{delta}, "
        f"foreground PSNR={quality.get(fg_key, float('nan')):.1f} dB{extra}"
    )


def _ssim_mse_summary(quality: dict[str, float]) -> str:
    """The ``, SSIM=…, MSE=…`` tail of a summary line (whichever are present).

    SSIM is headlined on the robust peak like PSNR, the full range labelled.
    """
    parts = []
    if "ssim_robust" in quality and "ssim" in quality:
        parts.append(
            f"SSIM={quality['ssim_robust']:.4f} robust peak "
            f"({quality['ssim']:.4f} full range)"
        )
    elif "ssim" in quality:
        parts.append(f"SSIM={quality['ssim']:.4f}")
    if "mse" in quality:
        parts.append(f"MSE={quality['mse']:.2e}")
    return "".join(f", {part}" for part in parts)


def _default_fit_scorer(
    V: np.ndarray, image_min: Optional[float]
) -> "Callable[[torch.Tensor], dict[str, float]]":
    """``finalize_results``' own scoring: ``compute_quality_metrics`` vs ``V - image_min``."""
    from luxar.gsplats.fit_basis import reference_on_fit_basis
    from luxar.gsplats.metrics import compute_quality_metrics

    def score(rendered: torch.Tensor) -> dict[str, float]:
        ref = torch.from_numpy(
            reference_on_fit_basis(V, image_min).astype(np.float32, copy=False)
        ).to(rendered.device)
        quality = compute_quality_metrics(rendered, ref)
        del ref
        return {key: quality[key] for key in _RESCORED_QUALITY_KEYS}

    return score
