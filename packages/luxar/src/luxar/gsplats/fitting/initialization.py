"""
Model and optimizer initialization for Gaussian splat fitting.
"""

from __future__ import annotations

import numpy as np

from luxar.gsplats.fitting.config import FitConfig, ModelComponents, PreprocessedData
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import create_per_splat_optimizer_setup


def initialize_optimization(
    config: FitConfig, preprocessed_data: PreprocessedData
) -> ModelComponents:
    """
    Initialize model, optimizer, scheduler, and coordinator.

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
            coordinator=None,
        )

    # Initialize parameters - use pre-initialized values if provided
    from arbol import aprint

    if config.init_L is not None:
        # Use pre-computed Cholesky factors (from GSplatData or moment pursuit)
        L0 = config.init_L.astype(np.float32)
        if config.verbose:
            aprint(f"Using pre-initialized Cholesky factors: {L0.shape}")
    else:
        # Default: isotropic Gaussians with init_sigma_vox
        L0 = np.zeros((N, d, d), dtype=np.float32)
        for i in range(d):
            L0[:, i, i] = config.init_sigma_vox

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

    # Build model - use Metal acceleration when available
    use_metal = (
        config.use_metal
        and d == 3  # Metal ONLY for 3D (overhead > benefit for 2D)
        and config.device.type == "mps"  # Requires MPS device
    )

    if use_metal:
        # Try to use Metal-accelerated model
        try:
            from luxar.gsplats.models.gsplats.metal import (
                GaussianSplatModelMetal,
                is_metal_available,
            )

            if is_metal_available():
                model = GaussianSplatModelMetal(
                    shape=config.V.shape,
                    centers0=preprocessed_data.seed_centers,
                    L0=L0,
                    amps0=amps0,
                    sigma_min_diag=config.sigma_min_diag,
                    sigma_max_diag=config.sigma_max_diag,
                    truncate=config.truncate,
                    intensity_floor=config.metal_intensity_floor,
                    tile_size=config.metal_tile_size,  # Configurable tile size
                    device=config.device,
                )
                if config.verbose:
                    from arbol import aprint

                    aprint(
                        "Using Metal-accelerated model (3-7x faster on Apple Silicon)"
                    )
            else:
                # Metal not available, fall back
                use_metal = False
        except ImportError:
            # Metal backend not installed
            use_metal = False

    if not use_metal:
        # Standard PyTorch model
        model = GaussianSplatModel(
            shape=config.V.shape,
            centers0=preprocessed_data.seed_centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=config.sigma_min_diag,
            sigma_max_diag=config.sigma_max_diag,
            truncate=config.truncate,
            device=config.device,
        )

    # Setup per-splat optimizer
    # Note: Gradient dilution compensation is handled internally by the optimizer
    optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
        model,
        lr=config.lr,  # Base learning rate (optimizer handles gradient dilution internally)
        scheduler_type=config.scheduler_type,
        patience=config.patience,
        lr_reduction_factor=config.lr_reduction_factor,
    )

    return ModelComponents(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        coordinator=coordinator,
    )
