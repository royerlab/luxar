"""
Input validation and configuration preparation for Gaussian splat fitting.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from luxar.gsplats.fitting.config import FitConfig
from luxar.gsplats.gsplat_data import GSplatData
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

if TYPE_CHECKING:
    from luxar.gsplats.fit_gsplats import GaussianSplatFitter


DEFAULT_SIGMA_MIN_DIAG = float(np.sqrt(1.0 / 12.0))


def _validate_floor(floor: "str | float | None") -> None:
    """Validate a ``floor`` spec: ``auto`` / ``none`` / ``pN`` / float >= 0."""
    if floor is None:
        return
    if isinstance(floor, str):
        f = floor.strip().lower()
        if f in ("auto", "none", ""):
            return
        if f.startswith("p"):
            try:
                pct = float(f[1:])
            except ValueError as exc:
                raise ValueError(
                    f"floor percentile must be 'pN' (e.g. 'p10'), got {floor!r}"
                ) from exc
            if not 0.0 <= pct <= 100.0:
                raise ValueError(f"floor percentile must be in [0, 100], got {floor!r}")
            return
        try:
            value = float(f)
        except ValueError as exc:
            raise ValueError(
                f"floor must be 'auto'/'none'/'pN'/a number >= 0, got {floor!r}"
            ) from exc
    else:
        value = float(floor)
    if not math.isfinite(value):
        raise ValueError(f"floor must be a finite number, got {floor!r}")
    if value < 0.0:
        raise ValueError(f"floor must be >= 0, got {value}")


def prepare_fit_config(
    fitter: "GaussianSplatFitter",  # GaussianSplatFitter instance
    V: np.ndarray,
    seeds: Optional[np.ndarray | int | float | GSplatData] = None,
    norm_percentile: float = 0.0,
    floor: "str | float | None" = "auto",
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
    amp_max: Optional[float] = None,  # Maximum amplitude (prevents explosion)
    max_eccentricity: Optional[float] = None,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
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
    seed_method: str = "auto",
    voxel_footprint_correction: bool | float = False,
    boundary_penalty: Optional[float] = None,
    clip_to_bounds: bool = False,
    voxel_size: Optional[Sequence[float] | float] = None,
    output_space: str = "real",
    sort_splats_enabled: bool = True,
    sort_splats_interval: int = 1000,
    iter_callback: Optional[Any] = None,
    iter_callback_every: int = 25,
    **seed_kwargs: Any,
) -> FitConfig:
    """
    Validate input parameters and prepare configuration for fitting.

    Parameters
    ----------
    fitter : GaussianSplatFitter
        The fitter instance (for device and dynamic ops config)
    V : np.ndarray
        Input image/volume to reconstruct
    seed_method : str, default="auto"
        Method for generating seeds when seeds=None:
        - "decomposition": Scale-hierarchical detection via image decomposition
        - "grid": Uniform grid seeding for spatial coverage
        - "edges": Edge-based seeding with anisotropic shapes
        - "auto": Principled combination of all methods (recommended)
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

    # Normalize and validate downscale parameter
    from luxar.gsplats.fitting.downscale import normalize_downscale

    downscale_normalized = normalize_downscale(downscale, V.ndim)

    # Validate seeds if provided
    if seeds is not None:
        if isinstance(seeds, GSplatData):
            # GSplatData object - will be handled in preprocessing
            if seeds.centers is not None and len(seeds.centers) > 0:
                if seeds.centers.shape[1] != V.ndim:
                    raise ValueError(
                        f"GSplatData centers must have {V.ndim} columns to match image dimensions"
                    )
        elif isinstance(seeds, int):
            # Integer exact count
            if seeds <= 0:
                raise ValueError("seeds as int must be positive")
        elif isinstance(seeds, float):
            # Float compression ratio (splat floats / image floats)
            if seeds <= 0 or seeds > 1.0:
                raise ValueError(
                    "seeds as float (compression ratio) must be in range (0, 1.0]"
                )
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
    if init_sigma_vox is not None and init_sigma_vox <= 0:
        raise ValueError("init_sigma_vox must be positive if specified")
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
    if asymmetric_penalty is not None and asymmetric_penalty < 1.0:
        raise ValueError(
            "asymmetric_penalty must be >= 1.0 (values < 1.0 would invert the penalty)"
        )
    if gradient_clip is not None and gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive if specified")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if lr_reduction_factor <= 0.0 or lr_reduction_factor >= 1.0:
        raise ValueError("lr_reduction_factor must be in range (0, 1)")
    if early_stop_patience is not None and early_stop_patience < 1:
        raise ValueError("early_stop_patience must be >= 1 if specified")
    if scheduler_type not in ["plateau", "exponential"]:
        raise ValueError("scheduler_type must be 'plateau' or 'exponential'")
    if truncate <= 0:
        raise ValueError("truncate must be positive")
    if max_abs_error is not None and max_abs_error <= 0:
        raise ValueError("max_abs_error must be positive if specified")
    if rel_l2_target is not None and rel_l2_target <= 0:
        raise ValueError("rel_l2_target must be positive if specified")
    if movie_max_frames is not None and movie_max_frames <= 0:
        raise ValueError("movie_max_frames must be positive or None")
    if movie_every < 1:
        raise ValueError("movie_every must be >= 1")
    if not 0.0 <= norm_percentile < 50.0:
        raise ValueError(
            f"norm_percentile must be in range [0.0, 50.0), got {norm_percentile}"
        )
    _validate_floor(floor)

    # Movie frame limit
    if movie_max_frames is None:
        movie_max_frames = 10000  # Large but finite limit

    # Validate sigma constraints
    d = V.ndim
    if sigma_min_diag is None:
        sigma_min_diag = [DEFAULT_SIGMA_MIN_DIAG] * d
    elif isinstance(sigma_min_diag, (int, float)):
        if sigma_min_diag <= 0:
            raise ValueError("sigma_min_diag must be positive if specified")
        sigma_min_diag = [float(sigma_min_diag)] * d
    else:
        if len(sigma_min_diag) != d:
            raise ValueError(f"sigma_min_diag must have length {d}")
        if any(s <= 0 for s in sigma_min_diag):
            raise ValueError("All sigma_min_diag values must be positive")

    if sigma_max_diag is not None:
        if isinstance(sigma_max_diag, (int, float)):
            # Single scalar: interpret as fraction of volume extent per axis.
            # Each dimension gets shape[i] * fraction independently,
            # which correctly handles anisotropic volumes.
            fraction = float(sigma_max_diag)
            if fraction <= 0:
                raise ValueError("sigma_max_diag fraction must be positive")
            sigma_max_diag = [s * fraction for s in V.shape]
        else:
            if len(sigma_max_diag) != d:
                raise ValueError(f"sigma_max_diag must have length {d}")
            if any(s <= 0 for s in sigma_max_diag):
                raise ValueError("All sigma_max_diag values must be positive")
        if any(s_max <= s_min for s_max, s_min in zip(sigma_max_diag, sigma_min_diag)):
            raise ValueError("sigma_max_diag must be greater than sigma_min_diag")

    # Validate amp_max
    if amp_max is not None and amp_max <= 0:
        raise ValueError("amp_max must be positive if specified")

    # Validate max_eccentricity
    if max_eccentricity is not None and max_eccentricity < 1.0:
        raise ValueError(
            "max_eccentricity must be >= 1.0 (ratio of longest to shortest axis)"
        )

    # Validate voxel_footprint_correction
    # Check for numeric types (int/float) but exclude bool (which is a subclass of int)
    if (
        isinstance(voxel_footprint_correction, (int, float))
        and not isinstance(voxel_footprint_correction, bool)
        and voxel_footprint_correction <= 0
    ):
        raise ValueError("voxel_footprint_correction sigma must be positive")

    # Validate boundary_penalty
    if boundary_penalty is not None and boundary_penalty < 0:
        raise ValueError("boundary_penalty must be non-negative if specified")

    # Validate voxel_size
    voxel_size_arr = None
    if voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            if voxel_size <= 0:
                raise ValueError("voxel_size must be positive")
            voxel_size_arr = np.array([float(voxel_size)] * d, dtype=np.float32)
        else:
            voxel_size_arr = np.asarray(voxel_size, dtype=np.float32)
            if voxel_size_arr.shape != (d,):
                raise ValueError(
                    f"voxel_size must have length {d} to match image dimensions, "
                    f"got length {len(voxel_size_arr)}"
                )
            if np.any(voxel_size_arr <= 0):
                raise ValueError("All voxel_size values must be positive")

    # Validate output_space
    if output_space not in ("real", "voxel"):
        raise ValueError("output_space must be 'real' or 'voxel'")

    # Validate sort_splats_interval
    if sort_splats_interval < 1:
        raise ValueError("sort_splats_interval must be >= 1")

    # Validate iter_callback
    if iter_callback is not None and not callable(iter_callback):
        raise ValueError("iter_callback must be callable or None")
    if iter_callback_every < 1:
        raise ValueError("iter_callback_every must be >= 1")

    return FitConfig(
        V=V,
        seeds=seeds,
        seed_method=seed_method,
        seed_kwargs=seed_kwargs,
        norm_percentile=norm_percentile,
        floor=floor,
        init_sigma_vox=init_sigma_vox,
        sigma_min_diag=sigma_min_diag,
        sigma_max_diag=sigma_max_diag,
        truncate=truncate,
        n_iters=n_iters,
        lr=lr,
        max_abs_error=max_abs_error,
        rel_l2_target=rel_l2_target,
        gradient_clip=gradient_clip,
        loss_type=loss_type,
        asymmetric_penalty=asymmetric_penalty,
        l1_amp=l1_amp,
        l1_diag=l1_diag,
        scheduler_type=scheduler_type,
        patience=patience,
        lr_reduction_factor=lr_reduction_factor,
        early_stop_patience=early_stop_patience,
        enable_dynamic_ops=fitter.enable_dynamic_ops,
        dynamic_config=fitter.dynamic_config,
        dynamic_ops_verbose=dynamic_ops_verbose,
        napari_movie=napari_movie,
        movie_every=movie_every,
        movie_max_frames=movie_max_frames,
        device=fitter.device,
        verbose=verbose,
        # Hardware acceleration flags from fitter
        use_metal=fitter.use_metal,
        use_cuda=fitter.use_cuda,
        # Amplitude constraint
        amp_max=amp_max,
        # Constraint parameters
        max_eccentricity=max_eccentricity,
        # Post-processing
        voxel_footprint_correction=voxel_footprint_correction,
        # Boundary containment
        boundary_penalty=boundary_penalty,
        clip_to_bounds=clip_to_bounds,
        # Anisotropic voxel spacing
        voxel_size=voxel_size_arr,
        output_space=output_space,
        # Volume downscaling
        downscale=downscale_normalized,
        # Z-order sorting
        sort_splats_enabled=sort_splats_enabled,
        sort_splats_interval=sort_splats_interval,
        # Per-iteration callback
        iter_callback=iter_callback,
        iter_callback_every=iter_callback_every,
    )
