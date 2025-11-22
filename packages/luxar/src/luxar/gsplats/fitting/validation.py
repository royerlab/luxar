"""
Input validation and configuration preparation for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from luxar.gsplats.fitting.config import FitConfig

if TYPE_CHECKING:
    from luxar.gsplats.fit_gsplats import GaussianSplatFitter


def prepare_fit_config(
    fitter: "GaussianSplatFitter",  # GaussianSplatFitter instance
    V: np.ndarray,
    seeds: Optional[np.ndarray | float] = None,
    norm_percentile: float = 0.0,
    init_sigma_vox: float = 1.5,
    n_iters: int = 1000,
    lr: float = 0.01,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
    l1_amp: Optional[float] = None,
    l1_diag: Optional[float] = None,
    l1_sharpness: Optional[float] = None,  # L1 regularization on sharpness offsets
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
    seed_method: str = "both",
    **seed_kwargs,
) -> FitConfig:
    """
    Validate input parameters and prepare configuration for fitting.

    Parameters
    ----------
    fitter : GaussianSplatFitter
        The fitter instance (for device and dynamic ops config)
    V : np.ndarray
        Input image/volume to reconstruct
    seed_method : str, default="both"
        Method for generating seeds when seeds=None:
        - "gaussian": Gaussian multi-scale blob detection
        - "decomposition": Dictionary/PCA-based decomposition
        - "both": Hybrid approach combining both methods
        This parameter is only used when seeds=None. If seeds are provided,
        this parameter is ignored.
    **seed_kwargs
        Additional keyword arguments for seed generation (e.g., num_scales,
        percentile_thresh, etc.). Only used when seeds=None.
    **kwargs
        All other fitting parameters

    Returns
    -------
    FitConfig
        Validated and prepared configuration

    Raises
    ------
    ValueError
        If any parameters are invalid
    """
    # Input validation
    V = np.asarray(V, dtype=np.float32)
    if V.size == 0:
        raise ValueError("Input image V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input image V must have at least 1 dimension")

    # Validate seeds if provided
    if seeds is not None:
        if isinstance(seeds, (int, float)):
            # Float proportion of voxels
            if seeds <= 0 or seeds > 1.0:
                raise ValueError("seeds as float must be in range (0, 1.0]")
        else:
            # Array of seed centers
            seeds = np.asarray(seeds, dtype=np.float32)
            if seeds.ndim != 2:
                raise ValueError("seeds must be a 2D array (N, ndim)")
            if seeds.shape[1] != V.ndim:
                raise ValueError(
                    f"seeds must have {V.ndim} columns to match image dimensions"
                )

    # L1 regularization defaults will be set in preprocessing.py after gradient dilution
    # is calculated, to ensure they scale properly with effective learning rates

    # Validate hyperparameters
    if init_sigma_vox <= 0:
        raise ValueError("init_sigma_vox must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if loss_type not in ["mse", "poisson", "l1"]:
        raise ValueError("loss_type must be 'mse', 'poisson', or 'l1'")
    if l1_amp is not None and l1_amp < 0:
        raise ValueError("l1_amp must be non-negative if specified")
    if l1_diag is not None and l1_diag < 0:
        raise ValueError("l1_diag must be non-negative if specified")
    if l1_sharpness is not None and l1_sharpness < 0:
        raise ValueError("l1_sharpness must be non-negative if specified")
    if asymmetric_penalty is not None and asymmetric_penalty < 1.0:
        raise ValueError(
            "asymmetric_penalty must be >= 1.0 (values < 1.0 would invert the penalty)"
        )
    if gradient_clip is not None and gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive if specified")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError("factor must be in range (0, 1)")
    if scheduler_type not in ["plateau", "exponential"]:
        raise ValueError("scheduler_type must be 'plateau' or 'exponential'")
    if truncate <= 0:
        raise ValueError("truncate must be positive")
    if max_abs_error is not None and max_abs_error <= 0:
        raise ValueError("max_abs_error must be positive if specified")
    elif movie_max_frames is not None and movie_max_frames <= 0:
        raise ValueError("movie_max_frames must be positive or None")

    # Movie frame limit
    if movie_max_frames is None:
        movie_max_frames = 10000  # Large but finite limit

    # Validate sigma constraints
    d = V.ndim
    if sigma_min_diag is None:
        sigma_min_diag = [0.01] * d
    else:
        if len(sigma_min_diag) != d:
            raise ValueError(f"sigma_min_diag must have length {d}")
        if any(s <= 0 for s in sigma_min_diag):
            raise ValueError("All sigma_min_diag values must be positive")

    if sigma_max_diag is not None:
        if len(sigma_max_diag) != d:
            raise ValueError(f"sigma_max_diag must have length {d}")
        if any(s <= 0 for s in sigma_max_diag):
            raise ValueError("All sigma_max_diag values must be positive")
        if any(s_max <= s_min for s_max, s_min in zip(sigma_max_diag, sigma_min_diag)):
            raise ValueError("sigma_max_diag must be greater than sigma_min_diag")

    return FitConfig(
        V=V,
        seeds=seeds,
        seed_method=seed_method,
        seed_kwargs=seed_kwargs,
        norm_percentile=norm_percentile,
        init_sigma_vox=init_sigma_vox,
        sigma_min_diag=sigma_min_diag,
        sigma_max_diag=sigma_max_diag,
        truncate=truncate,
        n_iters=n_iters,
        lr=lr,
        max_abs_error=max_abs_error,
        gradient_clip=gradient_clip,
        loss_type=loss_type,
        asymmetric_penalty=asymmetric_penalty,
        l1_amp=l1_amp,
        l1_diag=l1_diag,
        l1_sharpness=l1_sharpness,
        scheduler_type=scheduler_type,
        patience=patience,
        factor=factor,
        enable_dynamic_ops=fitter.enable_dynamic_ops,
        dynamic_config=fitter.dynamic_config,
        dynamic_ops_verbose=dynamic_ops_verbose,
        napari_movie=napari_movie,
        movie_every=movie_every,
        movie_max_frames=movie_max_frames,
        device=fitter.device,
        verbose=verbose,
    )
