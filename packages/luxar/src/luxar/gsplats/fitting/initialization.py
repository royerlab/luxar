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

    # Initialize parameters
    L0 = np.zeros((N, d, d), dtype=np.float32)
    for i in range(d):
        L0[:, i, i] = config.init_sigma_vox

    # Extract amplitudes from image at seed locations
    idx = np.clip(
        np.round(preprocessed_data.seed_centers).astype(int),
        0,
        np.array(config.V.shape) - 1,
    )
    amps0 = preprocessed_data.V_normalized[tuple(idx.T)]

    # Build model
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
        factor=config.factor,
    )

    return ModelComponents(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        coordinator=coordinator,
    )
