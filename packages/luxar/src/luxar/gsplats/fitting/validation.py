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
) -> FitConfig:
    """
    Validate input parameters and prepare configuration for fitting.

    Parameters
    ----------
    fitter : GaussianSplatFitter
        The fitter instance (for device and dynamic ops config)
    V : np.ndarray
        Input image/volume to reconstruct
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

    # Validate candidates if provided
    if centers_overcomplete is not None:
        centers_overcomplete = np.asarray(centers_overcomplete, dtype=np.float32)
        if centers_overcomplete.ndim != 2:
            raise ValueError("centers_overcomplete must be a 2D array")
        if centers_overcomplete.shape[1] != V.ndim:
            raise ValueError(
                f"centers_overcomplete must have {V.ndim} columns to match image dimensions"
            )

    # Set proportional L1 regularization default before validation
    if l1_amp is None:
        l1_amp = 0.1 * lr  # 10% of learning rate

    # Validate hyperparameters
    if init_sigma_vox <= 0:
        raise ValueError("init_sigma_vox must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if loss_type not in ["mse", "poisson", "l1"]:
        raise ValueError("loss_type must be 'mse', 'poisson', or 'l1'")
    if l1_amp < 0:
        raise ValueError("l1_amp must be non-negative")
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
        sigma_min_diag = [0.1] * d
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
        if any(
            s_max <= s_min for s_max, s_min in zip(sigma_max_diag, sigma_min_diag)
        ):
            raise ValueError("sigma_max_diag must be greater than sigma_min_diag")

    return FitConfig(
        V=V,
        centers_overcomplete=centers_overcomplete,
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
