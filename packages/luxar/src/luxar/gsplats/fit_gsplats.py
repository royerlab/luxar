# fit_gsplats.py

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np
import torch
from arbol import asection

from luxar.gsplats.fitting import (
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
from luxar.gsplats.utils.trils import tril_size


class GaussianSplatFitter:
    """
    Advanced Gaussian splat fitter with per-splat optimizer.

    This class uses the per-splat Adam optimizer to maintain momentum
    for individual splats during dynamic operations, providing smooth
    optimization without global disruption.

    Parameters
    ----------
    device : str, optional
        PyTorch device ('cpu', 'cuda', 'mps'). Auto-detects if None.
    enable_dynamic_ops : bool, default=True
        Enable dynamic operations (seeding, splitting, pruning).
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
        # - macOS + Apple Silicon: MPS with Metal acceleration (3-7x speedup)
        # - Linux + NVIDIA GPU: CUDA with custom kernels (10-50x speedup)
        # - Fallback: CPU
        if device is not None:
            self.device = torch.device(device)
        else:
            # Check for CUDA first (Linux with NVIDIA GPU)
            if use_cuda and torch.cuda.is_available():
                self.device = torch.device("cuda")
            # Check for MPS (macOS with Apple Silicon)
            elif use_metal and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        if self.device.type == "cpu":
            from arbol import aprint

            aprint(
                "WARNING: Gaussian splat fitting will run on CPU — "
                "this is 10-50x SLOWER than GPU! "
                "For serious work, use device='cuda' (NVIDIA) or device='mps' (Apple Silicon). "
                "To install CUDA support: make setup-cuda && make build-cuda"
            )
            warnings.warn(
                "Gaussian splat fitting will run on CPU. "
                "This is 10-50x SLOWER than GPU. "
                "For serious work, use device='cuda' (NVIDIA) or device='mps' (Apple Silicon). "
                "To install CUDA support: make setup-cuda && make build-cuda",
                UserWarning,
                stacklevel=2,
            )

        # Dynamic operations configuration
        self.enable_dynamic_ops = enable_dynamic_ops
        self.dynamic_config = dynamic_config or DynamicOpsConfig()

    def fit(
        self,
        V: np.ndarray,
        seeds: Optional[np.ndarray | int | float] = None,
        norm_percentile: float = 0.0,
        downscale: Optional[int | Sequence[int]] = None,
        init_sigma_vox: Optional[float] = None,
        n_iters: int = 1000,
        lr: float = 0.01,
        loss_type: str = "l1",
        asymmetric_penalty: Optional[float] = 10.0,
        l1_amp: Optional[float] = None,
        l1_diag: Optional[float] = None,
        sigma_min_diag: Optional[Sequence[float] | float] = DEFAULT_SIGMA_MIN_DIAG,
        sigma_max_diag: Optional[Sequence[float] | float] = None,
        amp_max: Optional[float] = None,  # Max amplitude (default auto: 1.0)
        max_eccentricity: Optional[float] = 10.0,
        truncate: float = 3.0,
        seed_method: str = "auto",
        verbose: bool = True,
        max_abs_error: Optional[float] = None,
        rel_l2_target: Optional[float] = None,
        gradient_clip: Optional[float] = 1.0,
        napari_movie: bool = False,
        movie_every: int = 1,
        movie_max_frames: Optional[int] = None,
        scheduler_type: str = "plateau",
        patience: int = 25,
        lr_reduction_factor: float = 0.98,
        early_stop_patience: Optional[int] = 300,
        dynamic_ops_verbose: bool = False,
        cull_ratio: float = 0.01,
        voxel_footprint_correction: bool | float = False,
        boundary_penalty: Optional[float] = None,
        clip_to_bounds: bool = False,
        voxel_size: Optional[Sequence[float] | float] = None,
        output_space: str = "real",
        sort_splats_enabled: bool = True,
        sort_splats_interval: int = 1000,
        **seed_kwargs: Any,
    ) -> GSplatData:
        """
        Fit Gaussian splats using per-splat Adam optimizer.

        This is the refactored version that uses the modular fitting pipeline.
        See fit_gaussian_splats() for full parameter documentation.

        Parameters
        ----------
        seed_method : str, default="auto" (RECOMMENDED)
            Method for generating seeds when seeds=None:

            - **"auto"** (DEFAULT, RECOMMENDED): Fast edges + grid combination.
              Provides good convergence by capturing boundaries (edges) and
              spatial coverage (grid). Decomposition excluded for speed.

            - "decomposition": Multi-scale decomposition for blob-like features (slow).

            - "grid": Uniform grid seeding for spatial coverage.

            - "edges": Edge-based seeding with anisotropic shapes.

            - Comma-separated combinations (e.g., "decomposition,edges,grid").

            This parameter is only used when seeds=None.
        **seed_kwargs
            Additional keyword arguments for seed generation (e.g., num_scales,
            percentile_thresh, etc.). Only used when seeds=None.

        Returns
        -------
        GSplatData
            Dataclass containing centers, amplitudes, cholesky_factors, and stats.
        """
        # Step 1: Validate and prepare configuration
        config = prepare_fit_config(
            self,
            V,
            seeds,
            norm_percentile,
            downscale=downscale,
            init_sigma_vox=init_sigma_vox,
            n_iters=n_iters,
            lr=lr,
            loss_type=loss_type,
            asymmetric_penalty=asymmetric_penalty,
            l1_amp=l1_amp,
            l1_diag=l1_diag,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            amp_max=amp_max,
            max_eccentricity=max_eccentricity,
            truncate=truncate,
            verbose=verbose,
            max_abs_error=max_abs_error,
            rel_l2_target=rel_l2_target,
            gradient_clip=gradient_clip,
            napari_movie=napari_movie,
            movie_every=movie_every,
            movie_max_frames=movie_max_frames,
            scheduler_type=scheduler_type,
            patience=patience,
            lr_reduction_factor=lr_reduction_factor,
            early_stop_patience=early_stop_patience,
            dynamic_ops_verbose=dynamic_ops_verbose,
            seed_method=seed_method,
            cull_ratio=cull_ratio,
            voxel_footprint_correction=voxel_footprint_correction,
            boundary_penalty=boundary_penalty,
            clip_to_bounds=clip_to_bounds,
            voxel_size=voxel_size,
            output_space=output_space,
            sort_splats_enabled=sort_splats_enabled,
            sort_splats_interval=sort_splats_interval,
            **seed_kwargs,
        )

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


def fit_gaussian_splats(
    V: np.ndarray,
    seeds: Optional[np.ndarray | int | float] = None,
    norm_percentile: float = 0.0,
    downscale: Optional[int | Sequence[int]] = None,
    init_sigma_vox: Optional[float] = None,
    n_iters: int = 1000,
    lr: float = 0.01,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
    l1_amp: Optional[float] = None,
    l1_diag: Optional[float] = None,
    sigma_min_diag: Optional[Sequence[float] | float] = DEFAULT_SIGMA_MIN_DIAG,
    sigma_max_diag: Optional[Sequence[float] | float] = None,
    amp_max: Optional[float] = None,
    max_eccentricity: Optional[float] = 10.0,
    truncate: float = 3.0,
    device: Optional[str] = None,
    seed_method: str = "auto",
    verbose: bool = True,
    # Optimization parameters
    max_abs_error: Optional[float] = None,
    rel_l2_target: Optional[float] = None,
    gradient_clip: Optional[float] = 1.0,
    # Per-splat optimizer parameters
    scheduler_type: str = "plateau",
    patience: int = 25,
    lr_reduction_factor: float = 0.98,
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
    cull_ratio: float = 0.01,
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
    **seed_kwargs: Any,
) -> GSplatData:
    """
    Fit n-dimensional oriented Gaussian splats to reconstruct input image/volume.

    This function uses the per-splat Adam optimizer to maintain momentum
    for individual splats during dynamic operations, providing smooth
    optimization without global disruption.

    The optimization uses:
    - Per-splat Adam optimizer with individual learning rates
    - Center position (bounded to image domain via sigmoid)
    - Non-negative amplitude (via softplus activation)
    - Covariance matrix Σ = L @ L^T where L is the Cholesky factor
    - Efficient rendering via batched triangular solve

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to reconstruct. Will be normalized to [0,1].
    seeds : np.ndarray, shape (N, d) or int or float, optional
        Initial seed center positions, count, or compression ratio.
        - If np.ndarray: Explicit seed centers in voxel coordinates
        - If int: Exact number (keeps highest intensity if more detected)
        - If float (0 < seeds <= 1.0): Compression ratio - the ratio of floats
          used to represent Gaussian splats over total image floats. For example,
          seeds=0.1 targets a representation using 10% of the original storage.
          The number of splats is computed as: n = ratio × total_voxels / floats_per_splat
          where floats_per_splat = d + d×(d+1)/2 + 1 (center + Cholesky + amp).
        - If None: Auto-generated using dimension-aware intelligent defaults:
          * Universal scales: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) for comprehensive detection
          * Volume-proportional density: ~1% of voxels as seeds
          * Inclusive threshold: percentile_thresh=70 for broad feature coverage
    norm_percentile : float, default=0.0
        Normalization method for handling outliers and noise:
        - 0.0: Full min-max range (maximum dynamic range, sensitive to outliers)
        - >0: Percentile clipping (e.g., 1.0 uses 1%-99% range, robust to outliers)
        Higher values provide more outlier robustness but may clip important data.
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
        Loss function: "mse", "poisson" (better for count/photon data), or "l1" (robust to outliers, preserves sharp features, default).
    asymmetric_penalty : float, default=10.0
        Over-prediction penalty factor for asymmetric loss. Multiplies loss for regions
        where pred > target by this factor. Set to None to disable asymmetric loss.
        Default 10.0 heavily penalizes over-prediction since non-negative Gaussian sums
        cannot easily reduce intensity, making under-prediction easier to correct.
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
    truncate : float, default=3.0
        Truncation radius in standard deviations for rendering efficiency.
    device : str, optional
        PyTorch device ("cpu", "cuda", "mps"). Auto-detects if None.
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
        optimization stops when max(|prediction - target|) < max_abs_error.
        If None, automatically set to 0.01 (1% of normalized [0,1] range) for
        sensible convergence behavior. Auto-threshold usage is logged.
    rel_l2_target : float or None, default=None
        Relative L2 error threshold for convergence. If specified,
        optimization stops when ||pred - target||₂ / ||target||₂ < rel_l2_target.
        This is an additional (OR) criterion alongside max_abs_error — either
        being satisfied triggers convergence. Provides a smoother, more stable
        convergence signal than max_abs_error. If None, this criterion is disabled.
    gradient_clip : float or None, default=1.0
        Maximum gradient norm for clipping. None disables clipping.
    scheduler_type : str, default="plateau"
        Type of learning rate scheduler ("plateau" or "exponential").
    patience : int, default=25
        Scheduler patience: iterations without loss improvement before LR reduction.
    lr_reduction_factor : float, default=0.98
        LR multiplier on plateau (new_lr = lr × lr_reduction_factor).
        Examples: 0.5=halve LR, 0.1=reduce to 10%, 0.98=gentle reduction.
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
        Provides 3-7x speedup for 3D volumes. Automatically disabled if not available.
    use_cuda : bool, default=True
        Enable custom CUDA kernels on NVIDIA GPUs.
        Provides 10-50x speedup for 2D-8D volumes. Automatically disabled if not available.
    cull_ratio : float, default=0.01
        Post-fit culling threshold as a fraction of max_abs_error. Splats with
        amplitude below ``cull_ratio * max_abs_error`` are removed after fitting.
        This completes the work L1 regularization started by removing near-zero
        splats that the softplus parameterization prevented from reaching exactly zero.
        - 0.0: Disable culling (keep all splats)
        - 0.01 (default): Cull splats below 1% of the convergence threshold
        - 1.0: Cull at the full convergence threshold (aggressive)
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
    The optimization uses per-splat Adam optimizer which provides:
    - Individual learning rates per splat
    - Momentum preservation during dynamic operations
    - Smooth optimization trajectory without global disruption
    - Early stopping for improved efficiency
    - Adaptive learning rate scheduling

    This approach is particularly beneficial when dynamic operations
    (prune, seed, merge, split) are enabled.
    """

    with asection("Fitting Gaussian Splats"):
        # Use per-splat optimizer
        fitter = GaussianSplatFitter(
            device=device,
            enable_dynamic_ops=enable_dynamic_ops,
            dynamic_config=dynamic_config,
            use_metal=use_metal,
            use_cuda=use_cuda,
        )

        # Fit and extract results
        result = fitter.fit(
            V=V,
            seeds=seeds,
            norm_percentile=norm_percentile,
            downscale=downscale,
            init_sigma_vox=init_sigma_vox,
            n_iters=n_iters,
            lr=lr,  # Note: gradient dilution compensation applied automatically in fit() method
            loss_type=loss_type,
            asymmetric_penalty=asymmetric_penalty,
            l1_amp=l1_amp,
            l1_diag=l1_diag,
            dynamic_ops_verbose=dynamic_ops_verbose,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            amp_max=amp_max,
            max_eccentricity=max_eccentricity,
            truncate=truncate,
            seed_method=seed_method,
            verbose=verbose,
            max_abs_error=max_abs_error,
            rel_l2_target=rel_l2_target,
            gradient_clip=gradient_clip,
            napari_movie=napari_movie,
            movie_every=movie_every,
            movie_max_frames=movie_max_frames,
            scheduler_type=scheduler_type,
            patience=patience,
            lr_reduction_factor=lr_reduction_factor,
            early_stop_patience=early_stop_patience,
            cull_ratio=cull_ratio,
            voxel_footprint_correction=voxel_footprint_correction,
            boundary_penalty=boundary_penalty,
            clip_to_bounds=clip_to_bounds,
            voxel_size=voxel_size,
            output_space=output_space,
            sort_splats_enabled=sort_splats_enabled,
            sort_splats_interval=sort_splats_interval,
            **seed_kwargs,
        )

    # After fitting section closes, show summary and movie
    if verbose:
        from arbol import aprint

        with asection("Optimization Complete"):
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

    return result
