"""
Input validation and configuration preparation for Gaussian splat fitting.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from luxar.gsplats.fitting.config import FitConfig, FitParameters
from luxar.gsplats.gsplat_data import GSplatData
from luxar.typing_utils.constants import (
    DEFAULT_SIGMA_MIN_DIAG as _DEFAULT_SIGMA_MIN_DIAG,
)

if TYPE_CHECKING:
    from luxar.gsplats.fit_gsplats import GaussianSplatFitter


#: Re-exported from :mod:`luxar.typing_utils.constants`, where it now lives.
#:
#: It moved because this module imports ``FitConfig`` from
#: ``gsplats.fitting.config``, so the config module could not name its own
#: default from here without a circular import — which is how ``ConstraintConfig``
#: came to default to ``None`` while the fitter defaulted to this value
#: (audit A3-01). Kept bound here so the existing import sites do not move.
DEFAULT_SIGMA_MIN_DIAG = _DEFAULT_SIGMA_MIN_DIAG


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


def _validate_norm_range(norm_range: "tuple[float, float] | None") -> None:
    """Validate a supplied ``(image_min, image_max)`` normalization range.

    A degenerate or reversed range is otherwise silent: ``image_max ==
    image_min`` lands in ``_normalize_data``'s "nearly uniform" branch, which
    replaces the whole array with 0.5, and ``image_max < image_min`` normalizes
    every voxel negative and clips it to zero. Both fit successfully and return
    nonsense.
    """
    if norm_range is None:
        return
    if len(norm_range) != 2:
        raise ValueError(
            f"norm_range must be an (image_min, image_max) pair, got {norm_range!r}"
        )
    lo, hi = float(norm_range[0]), float(norm_range[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"norm_range values must be finite, got {norm_range!r}")
    if hi <= lo:
        raise ValueError(
            f"norm_range must satisfy image_max > image_min, got {norm_range!r}"
        )


def _explicit_dtype_name(source_dtype: Any) -> Optional[str]:
    """Normalize an EXPLICIT ``source_dtype`` argument to a dtype name, or None.

    A caller naturally passes a dtype OBJECT (``np.dtype("uint16")``) or a scalar
    type (``np.uint16``), not only a string, and the value is stored verbatim in
    ``stats`` — where a dtype object is not JSON-serializable, so the whole fit
    used to complete and then die in ``.save()``. Normalized through
    ``str(np.dtype(...))`` exactly like the derived path.

    ``None`` (absent) and a blank string both mean "no explicit value": storing
    ``""`` gave a ``None`` itemsize, which sends ``results.py`` to the post-cast
    float32 ``V.nbytes`` — the 2x-overstated source size this stamp exists to
    prevent. An unrecognizable dtype NAME is still tolerated and recorded as
    given (the itemsize resolver records it without a size).
    """
    if source_dtype is None:
        return None
    if isinstance(source_dtype, str):
        # Stripped for the lookup too, not only for the emptiness test above: a
        # quoted YAML `source_dtype: "uint16 "` is otherwise unsizable, and the
        # size then silently goes missing — the same degradation the blank check
        # exists to prevent.
        source_dtype = source_dtype.strip()
        if not source_dtype:
            return None
    try:
        return str(np.dtype(source_dtype))
    except TypeError:  # a name/object numpy cannot interpret — keep it verbatim
        return str(source_dtype)


def _exact_dim(value: Any) -> int:
    """One grid dimension as an exact integer, or raise.

    ``int()`` alone silently TRUNCATES, which is the wrong failure for a number
    that will be published as a denominator: a caller who computed a dimension
    (a downscale factor, a JSON round-trip) wants to hear about it, not to get a
    grid quietly off by a voxel. ``bool`` is rejected for the same reason — it is
    an ``int`` subclass, so ``True`` would pass as a one-voxel axis.
    """
    if isinstance(value, bool):
        raise TypeError(f"not an integer dimension: {value!r}")
    dim = int(value)  # raises TypeError/ValueError on anything uninterpretable
    if dim != value:
        raise ValueError(f"not an integer dimension: {value!r}")
    return dim


def _explicit_source_shape(source_shape: Any) -> Optional[list[int]]:
    """Normalize an EXPLICIT ``source_shape`` argument, or None.

    This declares the grid of the ACQUISITION the fit represents, for the very
    common case where the caller preprocessed before fitting -- a demo that
    downscales a 5D OME-Zarr channel to 128^3 and normalizes it hands the fitter
    an array that is no longer the data anyone means by "the source". Without a
    way to say so, the recorded grid is the working copy and every compression
    ratio quoted from it is against the wrong denominator.

    Validated rather than trusted: this number becomes the denominator of a
    published ratio, so a malformed one must fail here, loudly, and not surface
    later as a plausible-looking figure nobody can reproduce. That rules out two
    inputs a bare ``int(x)`` would have accepted with a straight face: a
    fractional dimension (``236 / 2`` truncates to a grid 0.2% off the one the
    caller meant) and a bare string (``"128"`` iterates into ``[1, 2, 8]``, a
    denominator four orders of magnitude wrong).
    """
    if source_shape is None:
        return None
    if isinstance(source_shape, (str, bytes)):
        raise ValueError(
            f"source_shape must be a sequence of integers, got {source_shape!r}"
        )
    try:
        dims = [_exact_dim(x) for x in source_shape]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source_shape must be a sequence of integers, got {source_shape!r}"
        ) from exc
    if not dims:
        raise ValueError("source_shape cannot be empty")
    if any(d <= 0 for d in dims):
        raise ValueError(f"source_shape dimensions must be positive, got {dims}")
    return dims


def _explicit_source_stored_bytes(value: Any) -> Optional[int]:
    """Normalize an EXPLICIT ``source_stored_bytes``, or None.

    The size the acquisition actually OCCUPIES -- the compressed file you
    download, not the array it decodes to. Both are wanted: a ratio against the
    decoded array says how much the splat representation beats raw voxels, and a
    ratio against the stored file says how much smaller the thing you download
    became. Quoting only the first invites reading it as the second, which
    flatters the splats by the source codec's own factor.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"source_stored_bytes must be an integer, got {value!r}")
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source_stored_bytes must be an integer, got {value!r}"
        ) from exc
    if size != value or size <= 0:
        raise ValueError(
            f"source_stored_bytes must be a positive integer, got {value!r}"
        )
    return size


def _resolve_source_dtype(V: Any, source_dtype: Any) -> tuple[str, Optional[int]]:
    """Resolve ``(source_dtype, source_itemsize)`` for the volume ``V``.

    Called BEFORE the fitter's float32 cast: the fitter works in float32, so
    after that cast the original element size is gone. It is the denominator of
    any compression ratio quoted about the result, and a uint16 volume recorded
    as float32 would overstate compression by 2x.

    An explicit ``source_dtype`` wins over what ``V`` reports, because a caller
    can be one cast further removed than we are: ``luxar gsplat fit`` loads
    through ``load_volume``, which already returns float32, so on that path --
    the one that produces essentially every stored dataset -- the on-disk
    element type is knowable ONLY from there. Reading ``V.dtype`` (rather than
    ``np.asarray(V).dtype``) keeps a lazy zarr/dask input lazy: materializing it
    twice would double both the peak memory and the read.
    """
    name = _explicit_dtype_name(source_dtype)
    if name is None:
        _dt = getattr(V, "dtype", None)
        try:
            name = str(np.dtype(_dt) if _dt is not None else np.asarray(V).dtype)
        except TypeError:  # a non-numpy dtype object (e.g. a torch dtype)
            name = str(np.asarray(V).dtype)
    try:
        source_itemsize: Optional[int] = int(np.dtype(name).itemsize)
    except TypeError:  # an unrecognized dtype name — record it without a size
        source_itemsize = None
    return name, source_itemsize


def prepare_fit_config(
    fitter: "GaussianSplatFitter",  # GaussianSplatFitter instance
    parameters: FitParameters,
) -> FitConfig:
    """
    Validate input parameters and prepare configuration for fitting.

    Parameters
    ----------
    fitter : GaussianSplatFitter
        The fitter instance (for device and dynamic ops config)
    parameters : FitParameters
        Raw fit parameters from the public entry point.

    Returns
    -------
    FitConfig
        Validated and prepared configuration

    Raises
    ------
    ValueError
        If any parameters are invalid
    """
    V = parameters.V
    seeds = parameters.seeds
    norm_percentile = parameters.norm_percentile
    floor = parameters.floor
    norm_range = parameters.norm_range
    downscale = parameters.downscale
    init_sigma_vox = parameters.init_sigma_vox
    n_iters = parameters.n_iters
    lr = parameters.lr
    loss_type = parameters.loss_type
    asymmetric_penalty = parameters.asymmetric_penalty
    l1_amp = parameters.l1_amp
    l1_diag = parameters.l1_diag
    sigma_min_diag = parameters.sigma_min_diag
    sigma_max_diag = parameters.sigma_max_diag
    amp_max = parameters.amp_max
    max_eccentricity = parameters.max_eccentricity
    truncate = parameters.truncate
    verbose = parameters.verbose
    max_abs_error = parameters.max_abs_error
    rel_l2_target = parameters.rel_l2_target
    gradient_clip = parameters.gradient_clip
    napari_movie = parameters.napari_movie
    movie_every = parameters.movie_every
    movie_max_frames = parameters.movie_max_frames
    scheduler_type = parameters.scheduler_type
    patience = parameters.patience
    lr_reduction_factor = parameters.lr_reduction_factor
    early_stop_patience = parameters.early_stop_patience
    dynamic_ops_verbose = parameters.dynamic_ops_verbose
    seed_method = parameters.seed_method
    voxel_footprint_correction = parameters.voxel_footprint_correction
    boundary_penalty = parameters.boundary_penalty
    clip_to_bounds = parameters.clip_to_bounds
    voxel_size = parameters.voxel_size
    output_space = parameters.output_space
    sort_splats_enabled = parameters.sort_splats_enabled
    sort_splats_interval = parameters.sort_splats_interval
    iter_callback = parameters.iter_callback
    iter_callback_every = parameters.iter_callback_every
    seed_amps_background_relative = parameters.seed_amps_background_relative
    source_dtype = parameters.source_dtype
    source_shape = parameters.source_shape
    source_stored_bytes = parameters.source_stored_bytes
    seed_kwargs = dict(parameters.seed_kwargs)

    # Input validation.
    # Capture the caller's dtype BEFORE the cast below (see the helper: after the
    # cast the original element size is gone).
    source_dtype, source_itemsize = _resolve_source_dtype(V, source_dtype)
    source_shape = _explicit_source_shape(source_shape)
    source_stored_bytes = _explicit_source_stored_bytes(source_stored_bytes)
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

    _validate_norm_range(norm_range)

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
        source_dtype=source_dtype,
        source_itemsize=source_itemsize,
        source_shape=source_shape,
        source_stored_bytes=source_stored_bytes,
        seeds=seeds,
        seed_amps_background_relative=seed_amps_background_relative,
        seed_method=seed_method,
        seed_kwargs=seed_kwargs,
        norm_percentile=norm_percentile,
        floor=floor,
        norm_range=norm_range,
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
