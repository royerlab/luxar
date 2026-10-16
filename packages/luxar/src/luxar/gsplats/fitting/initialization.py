"""
Model and optimizer initialization for Gaussian splat fitting.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np

from luxar.gsplats.fitting.config import FitConfig, ModelComponents, PreprocessedData
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import create_optimizer_and_scheduler

_SIGMA_MIN_DIAG_MARGIN = 0.1
# Smallest headroom above ``sigma_min_diag`` a SUPPLIED diagonal keeps. The model
# stores ``diag = sigma_min + softplus(raw)``, so a diagonal sitting exactly on
# the floor needs ``raw = -inf``; 1e-4 voxel keeps ``raw`` finite (about -9.2,
# still trainable under Adam) while changing the rendered splat by ~1e-4 relative.
_SUPPLIED_SIGMA_MIN_DIAG_EPS = 1e-4
# Relative tolerance below ``sigma_min_diag`` within which a supplied diagonal
# still counts as ON the floor (nudged by the eps above, not lifted by the full
# margin): a float32 round trip, a physical-unit division or a quantised store
# can land a fitted at-floor diagonal an ulp or so below it.
_SUPPLIED_SIGMA_MIN_DIAG_RTOL = 1e-3
# Relative margin below ``amp_max`` a warm-start amplitude is clipped to, so the
# parameter starts where the forward clamp still passes a gradient.
_WARM_AMP_MAX_RTOL = 1e-6

# Backend summaries already printed by quiet fits in this process.
_ANNOUNCED_BACKENDS: set[str] = set()


def apply_sigma_min_diag_floor(
    sigma_diag: np.ndarray, sigma_min_diag: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Apply the fitter's gradient-safety floor to Cholesky diagonals."""
    return np.asarray(
        np.maximum(
            np.asarray(sigma_diag, dtype=np.float32),
            np.asarray(sigma_min_diag, dtype=np.float32) + _SIGMA_MIN_DIAG_MARGIN,
        ),
        dtype=np.float32,
    )


def apply_supplied_sigma_min_diag_floor(
    sigma_diag: np.ndarray, sigma_min_diag: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Floor the Cholesky diagonals of a declared warm start so the model can
    represent them, changing as little as possible.

    A diagonal the model can already hold (``>= sigma_min_diag``, within a
    ``_SUPPLIED_SIGMA_MIN_DIAG_RTOL`` relative tolerance) is kept, up to a
    ``_SUPPLIED_SIGMA_MIN_DIAG_EPS`` nudge off the floor: a previous fit's
    output satisfies this by construction, so a warm start reproduces its seed.
    Lifting those by the full ``_SIGMA_MIN_DIAG_MARGIN`` (as
    :func:`apply_sigma_min_diag_floor` does for fresh seeds) widened every
    near-floor splat of a warm start before its first iteration. Only a diagonal
    clearly BELOW the floor, which the model would clamp anyway, gets the full
    margin.
    """
    diag = np.asarray(sigma_diag, dtype=np.float32)
    floor = np.asarray(sigma_min_diag, dtype=np.float32)
    representable = np.maximum(diag, floor + _SUPPLIED_SIGMA_MIN_DIAG_EPS)
    on_floor = diag >= floor * (1.0 - _SUPPLIED_SIGMA_MIN_DIAG_RTOL)
    return np.asarray(
        np.where(on_floor, representable, floor + _SIGMA_MIN_DIAG_MARGIN),
        dtype=np.float32,
    )


def _is_declared_warm_start(
    config: FitConfig, preprocessed_data: PreprocessedData
) -> bool:
    """True for a ``seeds=GSplatData`` declared to be a previous fit's output.

    The same declaration (``seed_amps_background_relative``) that selects the
    amplitude convention: a fit's output already satisfies the model's sigma
    floor, so only it is exempt from the fresh-seed margin. Generated seeds that
    carry Cholesky factors (``generate_seeds()`` output, scale-aware seeding)
    keep the margin they were tuned with.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    return (
        preprocessed_data.init_L is not None
        and isinstance(config.seeds, GSplatData)
        and config.seed_amps_background_relative
    )


def _clip_supplied_amps_under_cap(
    amps0: np.ndarray, amp_max: float, *, supplied: bool
) -> np.ndarray:
    """Clip SUPPLIED amplitudes just under ``amp_max``.

    The forward pass clamps at ``amp_max``, which passes no gradient to a raw
    amplitude above the cap: a supplied amplitude left there would start
    frozen. Clipping just under the cap renders the same at iteration 0.
    Image-sampled defaults are left alone.
    """
    if not supplied:
        return amps0
    cap = np.float32(amp_max * (1.0 - _WARM_AMP_MAX_RTOL))
    return np.asarray(np.minimum(amps0, cap), dtype=np.float32)


def _resolve_unfloored_fit_initial_sigma_diag(
    config: FitConfig, preprocessed_data: PreprocessedData
) -> np.ndarray:
    """Return the fresh-seed sigma vector before the gradient-safety floor."""
    opt_shape = tuple(preprocessed_data.V_tensor.shape)
    if config.init_sigma_vox is not None:
        return np.full(preprocessed_data.d, config.init_sigma_vox, dtype=np.float32)
    if config.voxel_size is not None:
        phys_dims = np.array(opt_shape, dtype=np.float32) * config.voxel_size
        init_sigma_phys = max(
            1.5 * float(config.voxel_size.min()), float(phys_dims.min()) * 0.05
        )
        return np.asarray(init_sigma_phys / config.voxel_size, dtype=np.float32)
    init_sigma = max(1.5, float(min(opt_shape)) * 0.05)
    return np.full(preprocessed_data.d, init_sigma, dtype=np.float32)


def resolve_fit_initial_sigma_diag(
    config: FitConfig, preprocessed_data: PreprocessedData
) -> np.ndarray:
    """Return the uniform voxel-space sigma vector used for fresh seeds.

    Callers use this only when ``preprocessed_data.init_L`` is absent; supplied
    covariances may contain a different initialization for every splat.
    """
    sigma_diag = _resolve_unfloored_fit_initial_sigma_diag(config, preprocessed_data)
    if config.sigma_min_diag is not None:
        sigma_diag = apply_sigma_min_diag_floor(sigma_diag, config.sigma_min_diag)
    return sigma_diag


def initialize_optimization(
    config: FitConfig, preprocessed_data: PreprocessedData
) -> ModelComponents:
    """
    Initialize model, optimizer, and scheduler.

    Parameters
    ----------
    config : FitConfig
        Configuration for the fitting process
    preprocessed_data : PreprocessedData
        Preprocessed data ready for optimization

    Returns
    -------
    ModelComponents
        Components needed for optimization
    """
    d = preprocessed_data.d
    N = preprocessed_data.N

    # Use the optimization volume shape (may differ from config.V.shape if downscaled)
    opt_shape = tuple(preprocessed_data.V_tensor.shape)

    # Handle edge case of no candidates
    if N == 0:
        # Return minimal components - this will be handled by caller
        return ModelComponents(
            model=None,
            optimizer=None,  # type: ignore[arg-type]
            scheduler=None,
        )

    # Initialize parameters - use pre-initialized values from preprocessed_data if provided
    from arbol import aprint

    if preprocessed_data.init_L is not None:
        # Use pre-computed Cholesky factors (from GSplatData or moment pursuit)
        L0 = preprocessed_data.init_L.astype(np.float32)
        if config.verbose:
            aprint(f"Using pre-initialized Cholesky factors: {L0.shape}")
    else:
        # Fallback: isotropic Gaussians with init_sigma_vox or auto-computed sigma
        sigma_diag = _resolve_unfloored_fit_initial_sigma_diag(
            config, preprocessed_data
        )
        L0 = np.zeros((N, d, d), dtype=np.float32)
        for i in range(d):
            L0[:, i, i] = sigma_diag[i]

        if config.verbose and config.init_sigma_vox is None:
            units = (
                "physical-derived voxel" if config.voxel_size is not None else "voxel"
            )
            aprint(f"Auto-computed init sigma diag={sigma_diag.tolist()} ({units})")

    # Ensure diagonal values are at least sigma_min_diag to prevent gradient death
    # (inverse_softplus of values near 0 causes gradients to vanish)
    # A DECLARED warm start (a previous fit's output, see
    # _is_declared_warm_start) keeps every diagonal the model can already
    # represent, so its iteration-0 model reproduces the seed. Fresh and
    # generated seeds keep the full margin.
    if config.sigma_min_diag is not None:
        floor_fn = (
            apply_supplied_sigma_min_diag_floor
            if _is_declared_warm_start(config, preprocessed_data)
            else apply_sigma_min_diag_floor
        )
        clamped_diagonal = floor_fn(
            np.diagonal(L0, axis1=1, axis2=2), config.sigma_min_diag
        )
        for i in range(d):
            L0[:, i, i] = clamped_diagonal[:, i]
        if config.verbose:
            aprint(
                f"Clamped L0 diagonal to >= sigma_min_diag + {_SIGMA_MIN_DIAG_MARGIN:g}"
                + (
                    " (below-floor supplied values only)"
                    if floor_fn is apply_supplied_sigma_min_diag_floor
                    else ""
                )
            )

    if preprocessed_data.init_amps is not None:
        # Use pre-computed amplitudes (already normalized to [0,1] in preprocessing)
        amps0 = preprocessed_data.init_amps.astype(np.float32)
        if config.verbose:
            aprint(
                f"Using pre-initialized amplitudes: range [{amps0.min():.4f}, {amps0.max():.4f}]"
            )
    else:
        # Default: extract amplitudes from image at seed locations
        idx = np.clip(
            np.round(preprocessed_data.seed_centers).astype(int),
            0,
            np.array(opt_shape) - 1,
        )
        amps0 = preprocessed_data.V_normalized[tuple(idx.T)]

    # Auto-determine amp_max if not specified
    # Default: the normalized image's own peak, which is exactly 1.0 whenever
    # the normalization derived its range from THIS array (that array's max IS
    # the ceiling by construction). A SUPPLIED whole-volume `norm_range` (tiled
    # fitting) is a bounded-sample estimate and deliberately leaves a brighter
    # voxel unclipped, so a hard 1.0 here would clamp — with zero gradient
    # above the cap — precisely the peak that unclipped scale exists to
    # preserve. Tracking the peak keeps the "no amplitude explosion" bound: an
    # amplitude still cannot exceed the data's own maximum.
    amp_max = config.amp_max
    if amp_max is None:
        peak = float(np.max(preprocessed_data.V_normalized))
        amp_max = max(1.0, peak) if np.isfinite(peak) else 1.0
        if config.verbose:
            aprint(f"Using auto amp_max={amp_max} (prevents amplitude explosion)")

    amps0 = _clip_supplied_amps_under_cap(
        amps0, amp_max, supplied=preprocessed_data.init_amps is not None
    )

    # Resolve sigma constraints: pass empty list when None (model uses defaults)
    _sigma_min: Sequence[float] = (
        config.sigma_min_diag if config.sigma_min_diag is not None else []
    )
    _sigma_max: Optional[Sequence[float]] = (
        list(config.sigma_max_diag)
        if isinstance(config.sigma_max_diag, (list, tuple))
        else [config.sigma_max_diag] * d
        if isinstance(config.sigma_max_diag, (int, float))
        else None
    )

    # Build model - use hardware acceleration when available
    model: Any = None

    # Try Metal acceleration (macOS + MPS)
    use_metal = (
        config.use_metal
        and d == 3  # Metal ONLY for 3D (overhead > benefit for 2D)
        and config.device.type == "mps"  # Requires MPS device
    )

    if use_metal:
        model = _try_metal_model(
            config,
            dict(
                shape=opt_shape,
                centers0=preprocessed_data.seed_centers,
                L0=L0,
                amps0=amps0,
                sigma_min_diag=_sigma_min,
                sigma_max_diag=_sigma_max,
                amp_max=amp_max,
                max_eccentricity=config.max_eccentricity,
                truncate=config.truncate,
                intensity_floor=config.metal_intensity_floor,
                voxel_size=config.voxel_size,
                device=config.device,
            ),
        )

    # Try CUDA acceleration (NVIDIA GPUs)
    use_cuda = (
        model is None  # Not already using Metal
        and config.use_cuda
        and 2 <= d <= 8  # CUDA supports 2D-8D
        and config.device.type == "cuda"  # Requires CUDA device
    )

    if use_cuda:
        try:
            from luxar.gsplats.models.gsplats.cuda import (
                CUDA_BACKEND_AVAILABLE,
                GaussianSplatModelCUDA,
            )

            if config.verbose:
                aprint(
                    f"CUDA custom kernels: {'compiled and available' if CUDA_BACKEND_AVAILABLE else 'NOT compiled (using PyTorch fallback)'}"
                )

            if CUDA_BACKEND_AVAILABLE:
                model = GaussianSplatModelCUDA(
                    shape=opt_shape,
                    centers0=preprocessed_data.seed_centers,
                    L0=L0,
                    amps0=amps0,
                    sigma_min_diag=_sigma_min,
                    sigma_max_diag=_sigma_max,
                    amp_max=amp_max,
                    max_eccentricity=config.max_eccentricity,
                    truncate=config.truncate,
                    intensity_floor=config.cuda_intensity_floor,
                    voxel_size=config.voxel_size,
                    device=config.device,
                )
                if config.verbose:
                    aprint(
                        "✓ Model class: GaussianSplatModelCUDA (often orders of magnitude faster on NVIDIA GPUs; depends on hardware)"
                    )
            else:
                aprint(
                    "WARNING: CUDA device detected but custom CUDA kernels are NOT compiled! "
                    "Fitting will use PyTorch fallback (significantly slower). "
                    "Build CUDA kernels with: make build-cuda"
                )
                warnings.warn(
                    "CUDA device detected but custom CUDA kernels are NOT compiled. "
                    "Fitting will use PyTorch fallback (significantly slower). "
                    "Build CUDA kernels with: make build-cuda",
                    UserWarning,
                    stacklevel=2,
                )
        except ImportError:
            if config.verbose:
                aprint(
                    "CUDA backend: NOT installed (cuda_splatting_backend module missing)"
                )
            pass  # CUDA backend not compiled

    # Fall back to standard PyTorch model
    if model is None:
        model = GaussianSplatModel(
            shape=opt_shape,
            centers0=preprocessed_data.seed_centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=_sigma_min,
            sigma_max_diag=_sigma_max,
            amp_max=amp_max,
            max_eccentricity=config.max_eccentricity,
            truncate=config.truncate,
            voxel_size=config.voxel_size,
            device=config.device,
        )
        device_type = config.device.type
        # No CPU warning here: the device is the fitter's own, and
        # GaussianSplatFitter.__init__ already warned (once) when it is CPU.
        if config.verbose:
            aprint(f"Model class: GaussianSplatModel (PyTorch, device={device_type})")

    _announce_model_backend(model, config)

    # Setup optimizer - always use standard PyTorch Adam (fast, vectorized)
    if config.verbose:
        aprint("Using standard PyTorch Adam optimizer")

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=config.lr,
        scheduler_type=config.scheduler_type,
        patience=config.patience,
        factor=config.lr_reduction_factor,
    )

    # Boost center position LR by 1.5x post-creation (empirically tuned).
    # Centers are the most critical parameters for PSNR — a mis-positioned
    # Gaussian produces large error regardless of shape/amplitude.  Applied
    # post-creation to preserve the test-verified defaults in integration.py.
    for pg in optimizer.param_groups:
        params = pg["params"]
        if len(params) == 1 and params[0] is model.raw_mu:
            pg["lr"] = pg["lr"] * 1.5

    return ModelComponents(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _warn_metal_kernels_unavailable(reason: str) -> None:
    """Loud, actionable warning that fitting fell back to the PyTorch renderer."""
    from arbol import aprint

    message = (
        "MPS device detected but the Metal splatting kernels are unavailable "
        f"({reason}). Fitting will use the PyTorch fallback (orders of magnitude "
        "slower). With torch >= 2.7 the kernels compile automatically at runtime "
        "(no Xcode needed); set LUXAR_METAL_BACKEND=torch to silence this warning."
    )
    aprint(f"WARNING: {message}")
    warnings.warn(message, UserWarning, stacklevel=3)


def _try_metal_model(config: FitConfig, model_kwargs: dict[str, Any]) -> Any | None:
    """Build a GaussianSplatModelMetal, or None (with a warning) when no kernels run.

    Kernel backend order (see ``models/gsplats/metal/backend.py``): compiled
    extension, then runtime compilation via ``torch.mps.compile_shader``, then
    the PyTorch model. A forced ``LUXAR_METAL_BACKEND=aot|runtime`` that
    cannot run raises out of :func:`get_metal_backend` rather than degrading.
    """
    from arbol import aprint

    from luxar.gsplats.models.gsplats.metal import (
        GaussianSplatModelMetal,
        get_metal_backend,
        metal_backend_unavailable_reason,
    )

    backend = get_metal_backend()
    if backend is None:
        reason = metal_backend_unavailable_reason() or "unknown reason"
        if reason != "LUXAR_METAL_BACKEND=torch":
            _warn_metal_kernels_unavailable(reason)
        return None
    model = GaussianSplatModelMetal(**model_kwargs)
    if config.verbose:
        aprint(
            f"✓ Model class: GaussianSplatModelMetal, {backend.name} kernels "
            "(substantially faster on Apple Silicon; speedup depends on chip)"
        )
    return model


def model_backend_summary(model: Any, device_type: str) -> str:
    """One line naming the model class and kernel backend a fit runs on."""
    name = type(model).__name__
    if type(model) is GaussianSplatModel:
        detail = "PyTorch renderer, no custom kernels"
    else:
        kernels = getattr(model, "kernel_backend", None)
        detail = f"{kernels} kernels" if kernels else "custom kernels"
    return f"Fitting backend: {name} ({detail}, device={device_type})"


def _announce_model_backend(model: Any, config: FitConfig) -> None:
    """Print the backend summary for a quiet fit, once per distinct backend.

    A verbose fit narrates its backend in detail; a ``--quiet`` one used to say
    nothing, hiding a silent fall back to the (much slower) PyTorch renderer.
    Deduplicated per process so a calibration sweep or a content-tiled fit —
    many quiet inner fits — states it once rather than once per fit.
    """
    if config.verbose:  # the verbose narration already names the backend
        return
    from luxar.utils.verbosity import aprint_heartbeat

    summary = model_backend_summary(model, config.device.type)
    if summary not in _ANNOUNCED_BACKENDS:
        _ANNOUNCED_BACKENDS.add(summary)
        # A heartbeat: survives a driver's `quiet_narration()` (content boxes).
        aprint_heartbeat(summary)
