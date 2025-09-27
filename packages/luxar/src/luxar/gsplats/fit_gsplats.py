# fit_gsplats.py

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from arbol import asection

from luxar.gsplats.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fitting import (
    create_loss_function,
    finalize_results,
    initialize_optimization,
    prepare_fit_config,
    preprocess_data,
    run_optimization_loop,
)
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
        centers_overcomplete: Optional[np.ndarray] = None,
        norm_percentile: float = 0.0,
        init_sigma_vox: float = 1.5,
        n_iters: int = 1000,
        lr: float = 0.01,
        loss_type: str = "l1",
        asymmetric_penalty: Optional[float] = 10.0,
        l1_amp: Optional[float] = None,
        sigma_min_diag: Optional[Sequence[float]] = None,
        sigma_max_diag: Optional[Sequence[float]] = None,
        truncate: float = 3.0,
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
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit Gaussian splats using per-splat Adam optimizer.

        This is the refactored version that uses the modular fitting pipeline.
        See fit_gaussian_splats() for full parameter documentation.

        Returns
        -------
        params_full : np.ndarray
            Splat parameters [centers, packed_L].
        amps : np.ndarray
            Splat amplitudes.
        stats : dict
            Optimization statistics (time, iterations, convergence).
        """
        # Step 1: Validate and prepare configuration
        config = prepare_fit_config(
            self, V, centers_overcomplete, norm_percentile, init_sigma_vox,
            n_iters, lr, loss_type, asymmetric_penalty, l1_amp,
            sigma_min_diag, sigma_max_diag, truncate, verbose,
            max_abs_error, gradient_clip, napari_movie, movie_every,
            movie_max_frames, scheduler_type, patience, factor, dynamic_ops_verbose
        )

        # Step 2: Preprocess data and generate candidates
        preprocessed_data = preprocess_data(config)

        # Handle edge case of no candidates
        if preprocessed_data.N == 0:
            return (
                np.zeros((0, preprocessed_data.d + tril_size(preprocessed_data.d)), np.float32),
                np.zeros((0,), np.float32),
                {},
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
    centers_overcomplete: Optional[np.ndarray] = None,
    norm_percentile: float = 0.0,
    init_sigma_vox: float = 0.5,
    n_iters: int = 1000,
    lr: float = 0.01,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
    l1_amp: Optional[float] = None,
    sigma_min_diag: Optional[Sequence[float]] = None,
    sigma_max_diag: Optional[Sequence[float]] = None,
    truncate: float = 3.0,
    device: Optional[str] = None,
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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
    centers_overcomplete : np.ndarray, shape (N, d), optional
        Initial candidate center positions in voxel coordinates (float).
        If None, automatically generated using dimension-aware intelligent defaults:
        - Universal scales: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) for comprehensive detection
        - Volume-proportional density: ~0.2% of pixels as candidates
        - Inclusive threshold: percentile_thresh=70 for broad feature coverage
    norm_percentile : float, default=0.0
        Normalization method for handling outliers and noise:
        - 0.0: Full min-max range (maximum dynamic range, sensitive to outliers)
        - >0: Percentile clipping (e.g., 1.0 uses 1%-99% range, robust to outliers)
        Higher values provide more outlier robustness but may clip important data.
    init_sigma_vox : float, default=1.5
        Initial isotropic standard deviation for Gaussian splats (in voxels).
    n_iters : int, default=1000
        Maximum number of optimization iterations. Default is generous to allow
        max_abs_error convergence criterion to work effectively.
    lr : float, default=0.2
        Learning rate for Adam optimizer.
    loss_type : str, default="mse"
        Loss function: "mse", "poisson" (better for count/photon data), or "l1" (robust to outliers, preserves sharp features).
    asymmetric_penalty : float, default=10.0
        Over-prediction penalty factor for asymmetric loss. Multiplies loss for regions
        where pred > target by this factor. Set to None to disable asymmetric loss.
        Default 10.0 heavily penalizes over-prediction since non-negative Gaussian sums
        cannot easily reduce intensity, making under-prediction easier to correct.
    l1_amp : float, default=None (auto: 0.1 * lr)
        L1 regularization coefficient on splat amplitudes for sparsity.
        If None, automatically set to 10% of learning rate for consistent
        sparsity pressure that scales with optimization strength.
    sigma_min_diag : Sequence[float], optional
        Minimum diagonal values for Cholesky factor L along each axis.
        Defaults to [0.5]*d to prevent degenerate splats.
    sigma_max_diag : Sequence[float], optional
        Maximum diagonal values for Cholesky factor L along each axis.
    truncate : float, default=3.0
        Truncation radius in standard deviations for rendering efficiency.
    device : str, optional
        PyTorch device ("cpu", "cuda", "mps"). Auto-detects if None.
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
    factor : float, default=0.5
        Learning rate reduction factor for scheduler.
    enable_dynamic_ops : bool, default=True
        Enable dynamic operations (seeding and pruning).
    dynamic_config : DynamicOpsConfig, optional
        Configuration for dynamic operations. Uses defaults if None.
    dynamic_ops_verbose : bool, default=False
        Enable detailed console logging for dynamic operations. Shows residual analysis,
        seeding attempts and pruning operations.
    napari_movie : bool, default=True
        Record optimization movie for napari visualization.
    movie_every : int, default=1
        Record movie frame every N iterations.
    movie_max_frames : int, default=None (infinite)
        Maximum number of movie frames to store in memory. Older frames are automatically
        removed when this limit is exceeded, preventing memory exhaustion during long optimizations.

    Returns
    -------
    params_full : np.ndarray, shape (N, d + d*(d+1)//2), dtype=float32
        Concatenated parameters for each splat: [center_coords, packed_cholesky_L].
        Centers remain in voxel coordinates, covariances in voxel units.
        Represents the BEST state encountered during optimization (lowest max_abs_error).
    amps : np.ndarray, shape (N,), dtype=float32
        Non-negative amplitude values for each splat, rescaled to original image
        intensity range. Can be used directly to reconstruct original image intensities.
        Represents the BEST state encountered during optimization.
        Note: Gaussian splatting cannot represent uniform DC components - only variations.
    stats : dict
        Optimization statistics including time, iterations, convergence status.
        Reflects the best state iteration, not the final iteration.

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
        params, amps, stats = fitter.fit(
            V=V,
            centers_overcomplete=centers_overcomplete,
            norm_percentile=norm_percentile,
            init_sigma_vox=init_sigma_vox,
            n_iters=n_iters,
            lr=lr,  # Note: gradient dilution compensation applied automatically in fit() method
            loss_type=loss_type,
            asymmetric_penalty=asymmetric_penalty,
            l1_amp=l1_amp,
            dynamic_ops_verbose=dynamic_ops_verbose,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            truncate=truncate,
            verbose=verbose,
            max_abs_error=max_abs_error,
            gradient_clip=gradient_clip,
            napari_movie=napari_movie,
            movie_every=movie_every,
            movie_max_frames=movie_max_frames,
            scheduler_type=scheduler_type,
            patience=patience,
            factor=factor,
        )

        if verbose:
            from arbol import aprint
            with asection("Optimization Complete"):
                aprint(f"Time: {stats['time_seconds']:.2f} seconds")
                aprint(f"Iterations: {stats['iterations']}/{n_iters}")
                if stats["converged"]:
                    aprint(
                        f"✓ Converged (saved {n_iters - stats['iterations']} iterations)"
                    )

        # Calculate and display compression ratio
        if verbose:
            from luxar.gsplats.fitting.visualization import display_compression_analysis
            display_compression_analysis(V, params, amps)

        return params, amps, stats


