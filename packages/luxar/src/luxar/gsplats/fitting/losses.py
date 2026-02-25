"""
Loss function creation for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData

if TYPE_CHECKING:
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


def create_loss_function(
    config: FitConfig, preprocessed_data: PreprocessedData, model: "GaussianSplatModel"
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create loss function based on configuration.

    Parameters
    ----------
    config : FitConfig
        Configuration containing loss type and parameters
    preprocessed_data : PreprocessedData
        Preprocessed data containing target tensor and computed L1 values
    model : GaussianSplatModel
        Model for accessing parameters (needed for L1 regularization)

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Loss function that takes prediction tensor and returns loss
    """
    V_t = preprocessed_data.V_tensor
    loss_type = config.loss_type
    asymmetric_penalty = config.asymmetric_penalty
    # Use L1 values from preprocessed_data (computed during preprocessing)
    # This avoids mutating the input config object
    l1_amp = preprocessed_data.l1_amp
    l1_diag = preprocessed_data.l1_diag
    l1_sharpness = preprocessed_data.l1_sharpness

    def loss_fn(pred: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Model prediction

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        if loss_type.lower() == "poisson":
            data = _compute_poisson_loss(pred, V_t, asymmetric_penalty)
        elif loss_type.lower() == "l1":
            data = _compute_l1_loss(pred, V_t, asymmetric_penalty)
        else:
            # MSE loss (default)
            data = _compute_mse_loss(pred, V_t, asymmetric_penalty)

        # Add L1 regularization on amplitudes if specified
        if l1_amp is not None and l1_amp > 0:
            # Use raw parameters directly to avoid rebuilding L matrices and centers
            data = data + l1_amp * torch.mean(torch.abs(F.softplus(model.raw_a)))

        # Add L1 regularization on diagonal elements if specified
        if l1_diag is not None and l1_diag > 0:
            # Regularize the raw diagonal parameters (before softplus transformation)
            # This encourages smaller, more isotropic splats
            data = data + l1_diag * torch.mean(torch.abs(F.softplus(model.raw_L_diag)))

        # Add L1 regularization on sharpness offsets if specified
        if l1_sharpness is not None and l1_sharpness > 0:
            # Regularize sharpness offsets (s') directly
            # This encourages splats to stay at standard Gaussian (s' = 0, s = 2)
            # Only deviate from standard Gaussian when truly beneficial
            data = data + l1_sharpness * torch.mean(
                torch.abs(model.sharpness_offsets_raw)
            )

        return data

    return loss_fn


def _compute_poisson_loss(
    pred: torch.Tensor, target: torch.Tensor, asymmetric_penalty: float | None
) -> torch.Tensor:
    """Compute Poisson deviance loss."""
    eps = 1e-8
    Vc = torch.clamp(target, min=0.0)
    Pc = torch.clamp(pred, min=eps)
    # Use xlogy to safely handle Vc=0 (0 * log(0) = 0, with correct gradients)
    dev = 2.0 * torch.sum(Pc - Vc + torch.xlogy(Vc, torch.clamp(Vc / Pc, min=eps)))
    data = dev / target.numel()

    # Apply asymmetric penalty if specified
    if asymmetric_penalty is not None:
        over_prediction_mask = pred > target
        # Compute additional penalty for over-prediction regions only
        # This penalizes regions where we predict more intensity than target
        over_prediction_dev = 2.0 * torch.sum(
            over_prediction_mask
            * (Pc - Vc + torch.xlogy(Vc, torch.clamp(Vc / Pc, min=eps)))
        )
        # Add (F-1) times the over-prediction loss to get total F times penalty
        data = data + (asymmetric_penalty - 1.0) * over_prediction_dev / target.numel()

    return data


def _compute_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, asymmetric_penalty: float | None
) -> torch.Tensor:
    """Compute L1 (Mean Absolute Error) loss."""
    # L1 loss (Mean Absolute Error)
    l1_error = torch.abs(pred - target)
    if asymmetric_penalty is not None:
        # Asymmetric L1: heavily penalize over-prediction (pred > target)
        # L1 + asymmetric penalty provides excellent robustness and stability
        over_prediction_mask = pred > target
        data = torch.mean(
            torch.where(
                over_prediction_mask,
                asymmetric_penalty * l1_error,  # F times penalty for over-prediction
                l1_error,  # Normal penalty for under-prediction
            )
        )
    else:
        data = F.l1_loss(pred, target)

    return data


def _compute_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, asymmetric_penalty: float | None
) -> torch.Tensor:
    """Compute MSE (Mean Squared Error) loss."""
    # MSE loss (default)
    squared_error = (pred - target) ** 2
    if asymmetric_penalty is not None:
        # Asymmetric MSE: heavily penalize over-prediction (pred > target)
        # This addresses the fundamental asymmetry in additive Gaussian models:
        # - Under-prediction (pred < target): Easy to fix by adding more Gaussians
        # - Over-prediction (pred > target): Hard to fix, requires reducing/moving splats
        over_prediction_mask = pred > target
        data = torch.mean(
            torch.where(
                over_prediction_mask,
                asymmetric_penalty * squared_error,  # F times penalty
                squared_error,  # Normal penalty
            )
        )
    else:
        data = F.mse_loss(pred, target)

    return data
