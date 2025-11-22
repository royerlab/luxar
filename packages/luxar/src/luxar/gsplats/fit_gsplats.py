# fit_gsplats.py

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from arbol import asection

from luxar.gsplats.fit_result import GaussianSplatResult
from luxar.gsplats.fitting import (
    create_loss_function,
    finalize_results,
    initialize_optimization,
    prepare_fit_config,
    preprocess_data,
    run_optimization_loop,
)
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
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
    enable_dynamic_ops : bool, default=False
        Enable dynamic operations (seeding, splitting, pruning).
    dynamic_config : DynamicOpsConfig, optional
        Configuration for dynamic operations.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        enable_dynamic_ops: bool = False,
        dynamic_config: Optional[DynamicOpsConfig] = None,
    ):
        # Auto-detect best performing device: CUDA → CPU
        # Note: MPS is supported but currently slower than CPU for typical workloads
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Dynamic operations configuration
        self.enable_dynamic_ops = enable_dynamic_ops
        self.dynamic_config = dynamic_config or DynamicOpsConfig()

    def fit(
        self,
        V: np.ndarray,
        seeds: Optional[np.ndarray | float] = None,
        norm_percentile: float = 0.0,
        init_sigma_vox: float = 0.5,
        n_iters: int = 1000,
        lr: float = 0.01,
        loss_type: str = "l1",
        asymmetric_penalty: Optional[float] = 10.0,
        l1_amp: Optional[float] = None,
        l1_diag: Optional[float] = None,
        l1_sharpness: Optional[float] = None,
        sigma_min_diag: Optional[Sequence[float]] = None,
        sigma_max_diag: Optional[Sequence[float]] = None,
        truncate: float = 3.0,
        seed_method: str = "gaussian",
        verbose: bool = True,
        max_abs_error: Optional[float] = None,
        gradient_clip: Optional[float] = 1.0,
        napari_movie: bool = False,
        movie_every: int = 1,
        movie_max_frames: Optional[int] = None,
        scheduler_type: str = "plateau",
        patience: int = 10,
        factor: float = 0.5,
        dynamic_ops_verbose: bool = False,
        **seed_kwargs,
    ) -> GaussianSplatResult:
        """
        Fit Gaussian splats using per-splat Adam optimizer.

        This is the refactored version that uses the modular fitting pipeline.
        See fit_gaussian_splats() for full parameter documentation.

        Parameters
        ----------
        seed_method : str, default="gaussian"
            Method for generating seeds when seeds=None:
            - "gaussian": Gaussian multi-scale blob detection
            - "decomposition": Dictionary/PCA-based decomposition
            - "both": Hybrid approach combining both methods
            This parameter is only used when seeds=None.
        **seed_kwargs
            Additional keyword arguments for seed generation (e.g., num_scales,
            percentile_thresh, etc.). Only used when seeds=None.

        Returns
        -------
        GaussianSplatResult
            Dataclass containing centers, amplitudes, cholesky_factors, sharpnesses, and stats.
        """
        # Step 1: Validate and prepare configuration
        config = prepare_fit_config(
            self,
            V,
            seeds,
            norm_percentile,
            init_sigma_vox,
            n_iters,
            lr,
            loss_type,
            asymmetric_penalty,
            l1_amp,
            l1_diag,
            l1_sharpness,
            sigma_min_diag,
            sigma_max_diag,
            truncate,
            verbose,
            max_abs_error,
            gradient_clip,
            napari_movie,
            movie_every,
            movie_max_frames,
            scheduler_type,
            patience,
            factor,
            dynamic_ops_verbose,
            seed_method=seed_method,
            **seed_kwargs,
        )

        # Step 2: Preprocess data and generate candidates
        preprocessed_data = preprocess_data(config)

        # Handle edge case of no candidates
        if preprocessed_data.N == 0:
            d = preprocessed_data.d
            return GaussianSplatResult(
                centers=np.zeros((0, d), dtype=np.float32),
                amplitudes=np.zeros((0,), dtype=np.float32),
                cholesky_factors=np.zeros((0, tril_size(d)), dtype=np.float32),
                sharpnesses=np.zeros((0,), dtype=np.float32),
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
    seeds: Optional[np.ndarray | float] = None,
    norm_percentile: float = 0.0,
    init_sigma_vox: float = 0.5,
    n_iters: int = 1000,
    lr: float = 0.01,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
    l1_amp: Optional[float] = None,
    l1_diag: Optional[float] = None,
    l1_sharpness: Optional[float] = None,
    sigma_min_diag: Optional[Sequence[float]] = None,
    sigma_max_diag: Optional[Sequence[float]] = None,
    truncate: float = 3.0,
    device: Optional[str] = None,
    seed_method: str = "gaussian",
    verbose: bool = True,
    # Optimization parameters
    max_abs_error: Optional[float] = None,
    gradient_clip: Optional[float] = 1.0,
    # Per-splat optimizer parameters
    scheduler_type: str = "plateau",
    patience: int = 10,
    factor: float = 0.9,
    # Dynamic operations parameters
    enable_dynamic_ops: bool = True,
    dynamic_config: Optional[DynamicOpsConfig] = None,
    dynamic_ops_verbose: bool = False,
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    **seed_kwargs,
) -> GaussianSplatResult:
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
    seeds : np.ndarray, shape (N, d) or float, optional
        Initial seed center positions or proportion of voxels to use as seeds.
        - If np.ndarray: Explicit seed centers in voxel coordinates (float)
        - If float (0 < seeds <= 1.0): Proportion of voxels to use as seeds
        - If None: Auto-generated using dimension-aware intelligent defaults:
          * Universal scales: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) for comprehensive detection
          * Volume-proportional density: ~1% of voxels as seeds
          * Inclusive threshold: percentile_thresh=70 for broad feature coverage
    norm_percentile : float, default=0.0
        Normalization method for handling outliers and noise:
        - 0.0: Full min-max range (maximum dynamic range, sensitive to outliers)
        - >0: Percentile clipping (e.g., 1.0 uses 1%-99% range, robust to outliers)
        Higher values provide more outlier robustness but may clip important data.
    init_sigma_vox : float, default=0.5
        Initial isotropic standard deviation for Gaussian splats (in voxels).
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
    l1_sharpness : float, default=None (auto: 0.05 * lr)
        L1 regularization coefficient on sharpness offset parameters (s').
        Encourages splats to remain at standard Gaussian (s' = 0, s = 2) unless
        beneficial to deviate. If None, automatically set to 5% of base learning rate
        (which equals 10% of the sharpness learning rate due to the 0.5× multiplier).
        Higher values promote standard Gaussians, lower values allow more sharpness learning.
        Note: Sharpness learning rate is hard-coded to 0.5× the base effective learning rate.
    sigma_min_diag : Sequence[float], optional
        Minimum diagonal values for Cholesky factor L along each axis.
        Defaults to [0.5]*d to prevent degenerate splats.
    sigma_max_diag : Sequence[float], optional
        Maximum diagonal values for Cholesky factor L along each axis.
    truncate : float, default=3.0
        Truncation radius in standard deviations for rendering efficiency.
    device : str, optional
        PyTorch device ("cpu", "cuda", "mps"). Auto-detects if None.
    seed_method : str, default="gaussian"
        Method for generating seeds when seeds=None:
        - "gaussian": Gaussian multi-scale blob detection (recommended for general use)
        - "decomposition": Dictionary/PCA-based decomposition
        - "both": Hybrid approach combining both methods
        This parameter is only used when seeds=None. If seeds are provided,
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
    gradient_clip : float or None, default=1.0
        Maximum gradient norm for clipping. None disables clipping.
    scheduler_type : str, default="plateau"
        Type of learning rate scheduler ("plateau" or "exponential").
    patience : int, default=10
        Scheduler patience for plateau scheduler.
    factor : float, default=0.9
        Learning rate reduction factor for scheduler (new_lr = lr * factor).
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

    Returns
    -------
    GaussianSplatResult
        Dataclass containing all fitting results:
        - centers: np.ndarray, shape (N, d) - Center positions in voxel coordinates
        - amplitudes: np.ndarray, shape (N,) - Non-negative amplitudes rescaled to original intensity
        - cholesky_factors: np.ndarray, shape (N, d*(d+1)//2) - Packed lower-triangular Cholesky factors
        - sharpnesses: np.ndarray, shape (N,) - Per-splat sharpness values (s=2.0 is standard Gaussian)
        - stats: Dict[str, Any] - Optimization statistics (time, iterations, convergence, etc.)

        All arrays represent the BEST state encountered during optimization (lowest max_abs_error).
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
        )

        # Fit and extract results
        result = fitter.fit(
            V=V,
            seeds=seeds,
            norm_percentile=norm_percentile,
            init_sigma_vox=init_sigma_vox,
            n_iters=n_iters,
            lr=lr,  # Note: gradient dilution compensation applied automatically in fit() method
            loss_type=loss_type,
            asymmetric_penalty=asymmetric_penalty,
            l1_amp=l1_amp,
            l1_diag=l1_diag,
            l1_sharpness=l1_sharpness,
            dynamic_ops_verbose=dynamic_ops_verbose,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            truncate=truncate,
            seed_method=seed_method,
            verbose=verbose,
            max_abs_error=max_abs_error,
            gradient_clip=gradient_clip,
            napari_movie=napari_movie,
            movie_every=movie_every,
            movie_max_frames=movie_max_frames,
            scheduler_type=scheduler_type,
            patience=patience,
            factor=factor,
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

        # Display sharpness statistics
        with asection("Sharpness Statistics"):
            aprint(
                f"Range: [{result.stats['sharpness_min']:.2f}, {result.stats['sharpness_max']:.2f}]  "
                f"Mean: {result.stats['sharpness_mean']:.2f} ± {result.stats['sharpness_std']:.2f}  "
                f"Median: {result.stats['sharpness_median']:.2f}"
            )
            # Provide interpretation
            if result.stats["sharpness_mean"] < 1.5:
                aprint("→ Soft falloff (s < 2): Heavy-tailed Gaussians")
            elif result.stats["sharpness_mean"] < 2.5:
                aprint("→ Standard Gaussians (s ≈ 2): Classic Gaussian profiles")
            elif result.stats["sharpness_mean"] < 4.0:
                aprint("→ Sharp edges (2 < s < 4): Compact splats with faster decay")
            else:
                aprint("→ Very sharp (s ≥ 4): Near box-like splats with abrupt cutoff")

    # Show napari movie OUTSIDE the fitting section (so it doesn't affect timing)
    if result.stats.get("movie_frames") is not None:
        from luxar.gsplats.fitting.visualization import show_optimization_movie

        show_optimization_movie(result.stats["movie_frames"], result.stats["movie_shape"])

    return result
