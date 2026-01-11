"""
Model and optimizer initialization for Gaussian splat fitting.
"""

from __future__ import annotations

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

    # Handle edge case of no candidates
    if N == 0:
        # Return minimal components - this will be handled by caller
        return ModelComponents(
            model=None,
            optimizer=None,
            scheduler=None,
        )

    # Initialize parameters - use pre-initialized values if provided
    from arbol import aprint

    if config.init_L is not None:
        # Use pre-computed Cholesky factors (from GSplatData or moment pursuit)
        L0 = config.init_L.astype(np.float32)
        if config.verbose:
            aprint(f"Using pre-initialized Cholesky factors: {L0.shape}")
    else:
        # Fallback: isotropic Gaussians with init_sigma_vox or auto-computed sigma
        init_sigma = config.init_sigma_vox
        if init_sigma is None:
            # Auto-compute sigma based on image size: ~5% of smallest dimension, min 1.5
            min_dim = float(min(config.V.shape))
            init_sigma = max(1.5, min_dim * 0.05)
            if config.verbose:
                aprint(
                    f"Auto-computed init_sigma={init_sigma:.2f} (5% of min dim {min_dim})"
                )

        L0 = np.zeros((N, d, d), dtype=np.float32)
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

    if config.init_amps is not None:
        # Use pre-computed amplitudes
        amps0 = config.init_amps.astype(np.float32)
        if config.verbose:
            aprint(
                f"Using pre-initialized amplitudes: range [{amps0.min():.4f}, {amps0.max():.4f}]"
            )
    else:
        # Default: extract amplitudes from image at seed locations
        idx = np.clip(
            np.round(preprocessed_data.seed_centers).astype(int),
            0,
            np.array(config.V.shape) - 1,
        )
        amps0 = preprocessed_data.V_normalized[tuple(idx.T)]

    # Sharpness initialization (used if model supports it)
    sharpness0 = None
    if config.init_sharpness is not None:
        sharpness0 = config.init_sharpness.astype(np.float32)
        if config.verbose:
            aprint(
                f"Using pre-initialized sharpness: range [{sharpness0.min():.2f}, {sharpness0.max():.2f}]"
            )

    # Auto-determine amp_max if not specified
    # Default: 1.0 (matches max value in normalized [0, 1] image)
    # This prevents amplitude explosion during optimization
    amp_max = config.amp_max
    if amp_max is None:
        amp_max = 1.0
        if config.verbose:
            aprint(f"Using auto amp_max={amp_max} (prevents amplitude explosion)")

    # Build model - use hardware acceleration when available
    model = None

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
                    shape=config.V.shape,
                    centers0=preprocessed_data.seed_centers,
                    L0=L0,
                    amps0=amps0,
                    sigma_min_diag=config.sigma_min_diag,
                    sigma_max_diag=config.sigma_max_diag,
                    truncate=config.truncate,
                    intensity_floor=config.metal_intensity_floor,
                    tile_size=config.metal_tile_size,
                    device=config.device,
                )
                if config.verbose:
                    aprint(
                        "✓ Model class: GaussianSplatModelMetal (3-7x faster on Apple Silicon)"
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
                    shape=config.V.shape,
                    centers0=preprocessed_data.seed_centers,
                    L0=L0,
                    amps0=amps0,
                    sigma_min_diag=config.sigma_min_diag,
                    sigma_max_diag=config.sigma_max_diag,
                    amp_max=amp_max,
                    truncate=config.truncate,
                    intensity_floor=config.cuda_intensity_floor,
                    tile_size=config.cuda_tile_size,  # None = auto-select
                    device=config.device,
                )
                if config.verbose:
                    aprint(
                        "✓ Model class: GaussianSplatModelCUDA (10-50x faster on NVIDIA GPUs)"
                    )
            else:
                if config.verbose:
                    aprint(
                        "  → Will use PyTorch-based GaussianSplatModel on CUDA device"
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
            shape=config.V.shape,
            centers0=preprocessed_data.seed_centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=config.sigma_min_diag,
            sigma_max_diag=config.sigma_max_diag,
            amp_max=amp_max,
            truncate=config.truncate,
            device=config.device,
        )
        if config.verbose:
            device_type = config.device.type
            aprint(f"✓ Model class: GaussianSplatModel (PyTorch, device={device_type})")

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

    return ModelComponents(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
