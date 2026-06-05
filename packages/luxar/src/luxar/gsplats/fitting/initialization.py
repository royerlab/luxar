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
        init_sigma = config.init_sigma_vox
        init_sigma_phys = None  # Physical-space sigma (only used with voxel_size)
        if init_sigma is None:
            if config.voxel_size is not None:
                # Physical-space auto: use physical extents
                phys_dims = np.array(opt_shape, dtype=np.float32) * config.voxel_size
                min_phys_dim = float(phys_dims.min())
                min_vs = float(config.voxel_size.min())
                init_sigma_phys = max(1.5 * min_vs, min_phys_dim * 0.05)
                if config.verbose:
                    aprint(
                        f"Auto-computed init_sigma_phys={init_sigma_phys:.2f} "
                        f"(5% of min physical dim {min_phys_dim:.1f})"
                    )
            else:
                # Voxel-space auto: ~5% of smallest dimension, min 1.5
                min_dim = float(min(opt_shape))
                init_sigma = max(1.5, min_dim * 0.05)
                if config.verbose:
                    aprint(
                        f"Auto-computed init_sigma={init_sigma:.2f} (5% of min dim {min_dim})"
                    )

        L0 = np.zeros((N, d, d), dtype=np.float32)
        if init_sigma_phys is not None:
            # Physical sigma → per-axis voxel-space L_diag
            assert config.voxel_size is not None  # set when init_sigma_phys is set
            for i in range(d):
                L0[:, i, i] = init_sigma_phys / config.voxel_size[i]
        else:
            # Scalar voxel-space sigma (backward-compatible)
            for i in range(d):
                L0[:, i, i] = init_sigma

    # Ensure diagonal values are at least sigma_min_diag to prevent gradient death
    # (inverse_softplus of values near 0 causes gradients to vanish)
    if config.sigma_min_diag is not None:
        sigma_min = np.asarray(config.sigma_min_diag, dtype=np.float32)
        for i in range(d):
            L0[:, i, i] = np.maximum(L0[:, i, i], sigma_min[i] + 0.1)
        if config.verbose:
            aprint("Clamped L0 diagonal to >= sigma_min_diag + 0.1")

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
    # Default: 1.0 (matches max value in normalized [0, 1] image)
    # This prevents amplitude explosion during optimization
    amp_max = config.amp_max
    if amp_max is None:
        amp_max = 1.0
        if config.verbose:
            aprint(f"Using auto amp_max={amp_max} (prevents amplitude explosion)")

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
        try:
            from luxar.gsplats.models.gsplats.metal import (
                GaussianSplatModelMetal,
                is_metal_available,
            )

            metal_available = is_metal_available()
            if config.verbose:
                aprint(
                    f"Metal backend: {'available' if metal_available else 'NOT available'}"
                )

            if metal_available:
                model = GaussianSplatModelMetal(
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
                )
                if config.verbose:
                    aprint(
                        "✓ Model class: GaussianSplatModelMetal (substantially faster on Apple Silicon; speedup depends on chip)"
                    )
        except ImportError:
            if config.verbose:
                aprint("Metal backend: NOT installed")
            pass  # Metal backend not installed

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
        if device_type == "cpu":
            aprint(
                "WARNING: Gaussian splat fitting running on CPU — "
                "this can be orders of magnitude SLOWER than GPU (hardware-dependent)! "
                "For serious work, use device='cuda' or device='mps'."
            )
            warnings.warn(
                "Gaussian splat fitting running on CPU — this can be orders of magnitude SLOWER than GPU (hardware-dependent). "
                "For production use, install GPU support.",
                UserWarning,
                stacklevel=2,
            )
        if config.verbose:
            aprint(f"Model class: GaussianSplatModel (PyTorch, device={device_type})")

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
