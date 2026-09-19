"""Fit Gaussian splats to a volume with PyTorch Adam and fixed-pool relocation.

Provides ``GaussianSplatFitter`` (the configurable fitter class) and the
``fit_gaussian_splats`` convenience function, which optimize a fixed pool of
oriented Gaussians against an nD volume, periodically relocating the least
informative splats to high-residual regions so optimizer tensor shapes stay
constant. Runs on CPU, CUDA, or MPS.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import fields
from typing import Any, Optional, Sequence

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
from luxar.gsplats.fitting.validation import DEFAULT_SIGMA_MIN_DIAG
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils import resolve_torch_device
from luxar.gsplats.utils.trils import tril_size
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from luxar.utils.arbol_warnings import arbol_warnings


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

        if self.device.type == "cpu":
            from arbol import aprint

            aprint(
                "WARNING: Gaussian splat fitting will run on CPU — "
                "this can be orders of magnitude SLOWER than GPU (hardware-dependent)! "
                "For serious work, use device='cuda' (NVIDIA) or device='mps' (Apple Silicon). "
                "To install CUDA support: make setup-cuda && make build-cuda"
            )
            warnings.warn(
                "Gaussian splat fitting will run on CPU. "
                "This can be orders of magnitude SLOWER than GPU (hardware-dependent). "
                "For serious work, use device='cuda' (NVIDIA) or device='mps' (Apple Silicon). "
                "To install CUDA support: make setup-cuda && make build-cuda",
                UserWarning,
                stacklevel=2,
            )

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
        config = prepare_fit_config(self, parameters)

        # Clear cached rendering grids from previous fitting sessions
        from luxar.gsplats.models.gsplats.rendering_core import clear_grid_cache

        clear_grid_cache()

        # Step 2: Preprocess data and generate candidates
        preprocessed_data = preprocess_data(config)

        # Handle edge case of no candidates
        if preprocessed_data.N == 0:
            d = preprocessed_data.d
            return GSplatData(
                centers=np.zeros((0, d), dtype=np.float32),
                amplitudes=np.zeros((0,), dtype=np.float32),
                cholesky_factors=np.zeros((0, tril_size(d)), dtype=np.float32),
                stats={},
            )

        # Step 3: Initialize model and optimizer
        components = initialize_optimization(config, preprocessed_data)

        # Step 4: Create loss function
        loss_fn = create_loss_function(config, preprocessed_data, components.model)

        # Step 5: Run optimization loop
        optimization_results = run_optimization_loop(
            components, loss_fn, config, preprocessed_data
        )

        # Step 6: Finalize and return results
        return finalize_results(optimization_results, config, preprocessed_data)


def _collect_fit_parameters(values: dict[str, Any]) -> FitParameters:
    """Build the internal parameter bundle from the public call arguments."""
    return FitParameters(
        **{field.name: values[field.name] for field in fields(FitParameters)}
    )


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
    # Dynamic operations parameters
    enable_dynamic_ops: bool = True,
    dynamic_config: Optional[DynamicOpsConfig] = None,
    dynamic_ops_verbose: bool = False,
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    # Hardware acceleration
    use_metal: bool = True,
    use_cuda: bool = True,
    # Post-processing
    cull_retention: float | None = 0.95,
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
        "mse" (the global unregularised MSE minimiser maximises training PSNR, but
        finite-iteration regularised fits do not reach it), or "poisson" (natural
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
    enable_dynamic_ops : bool, default=True
        Enable dynamic operations (seeding and pruning).
    dynamic_config : DynamicOpsConfig, optional
        Configuration for dynamic operations. Uses defaults if None.
    dynamic_ops_verbose : bool, default=False
        Enable detailed console logging for dynamic operations. Shows residual analysis,
        seeding attempts and pruning operations.
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
    cull_retention : float or None, default=0.95
        Post-fit cumulative culling.  Keeps the top splats that account for
        this fraction of the total amplitude (0--1).  At 0.95, roughly 5% of
        splats are removed — those that collectively contribute only 5% of
        the total signal.  Set to ``None`` to disable.
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
          (``fit_gaussian_splats(V, seeds=generate_seeds(V))``). They are
          rescaled as ``(a - image_min) / intensity_range``, so an active
          ``floor`` is subtracted exactly once.
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

    # Keep this first: the bundle must capture the public arguments before rebinding.
    parameters = _collect_fit_parameters(locals())

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
            if not result.stats:
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

        # Calculate and display compression ratio
        from luxar.gsplats.fitting.visualization import display_compression_analysis

        display_compression_analysis(V, result)

    # Show napari movie OUTSIDE the fitting section (so it doesn't affect timing)
    if result.stats.get("movie_frames") is not None:
        from luxar.gsplats.fitting.visualization import show_optimization_movie

        show_optimization_movie(
            result.stats["movie_frames"], result.stats["movie_shape"]
        )

    # Post-fit cumulative culling (keeps top cull_retention of amplitude)
    if cull_retention is not None and 0 < cull_retention < 1.0 and result.n_splats > 0:
        from luxar.gsplats._data.filtering import (
            measured_stats_snapshot,
            restore_measured_stats,
        )

        n_before = result.n_splats
        amp_before = float(np.sum(result.amplitudes))
        # `cull` drops the measured scores (they describe the splat set they were
        # taken on — see content_scoped_stats), but this trim is the last step of
        # the FIT: the alternative to carrying the measurement over is a fit that
        # publishes no PSNR at all, and re-scoring costs a second full render.
        measured = measured_stats_snapshot(result)
        result = result.cull(method="cumulative", retention=cull_retention)
        restore_measured_stats(result, measured)
        n_removed = n_before - result.n_splats
        amp_after = float(np.sum(result.amplitudes))
        amp_retained_pct = 100.0 * amp_after / amp_before if amp_before > 0 else 100.0
        if verbose:
            from arbol import aprint

            aprint(
                f"Post-fit culling (cumulative, retention={cull_retention:.0%}): "
                f"{n_before} -> {result.n_splats} splats "
                f"(removed {n_removed}, {100.0 * n_removed / n_before:.1f}%; "
                f"amplitude retained: {amp_retained_pct:.1f}%)"
            )

    return result
