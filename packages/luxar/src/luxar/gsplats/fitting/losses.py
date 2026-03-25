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
    boundary_penalty = config.boundary_penalty

    # Dilated signal mask for sparse targets.  The mask covers signal voxels
    # + a margin of truncate * sigma_max voxels (the maximum extent any
    # splat can reach).  Voxels beyond this margin cannot be influenced by
    # any splat, so their gradient is naturally zero — the mask just makes
    # this explicit for the CUDA backward kernel's early-exit check.
    # The FULL loss (incl asymmetric penalty) is computed within the mask.
    _dilated_mask: torch.Tensor | None = None
    if loss_type.lower() == "poisson":
        _zero_frac = float((V_t < 1e-6).sum()) / V_t.numel()
        if _zero_frac > 0.3:
            # Dilation radius: truncate * sigma_max (conservative estimate)
            # sigma_max_diag caps the maximum splat size; default ~4-8 voxels.
            # With truncate=3.0, margin = 3 * 8 = 24 voxels.
            # Use max_pool for fast GPU binary dilation.
            _margin = int(config.truncate * 8)  # conservative margin
            _kernel = 2 * _margin + 1
            _signal = (V_t > 1e-6).float()
            d = V_t.ndim
            if d == 3:
                _dilated_mask = torch.nn.functional.max_pool3d(
                    _signal.unsqueeze(0).unsqueeze(0),
                    kernel_size=_kernel, stride=1, padding=_margin,
                ).squeeze(0).squeeze(0)
            elif d == 2:
                _dilated_mask = torch.nn.functional.max_pool2d(
                    _signal.unsqueeze(0).unsqueeze(0),
                    kernel_size=_kernel, stride=1, padding=_margin,
                ).squeeze(0).squeeze(0)
            else:
                _dilated_mask = None  # skip for other dims
            if _dilated_mask is not None:
                _masked_frac = 1.0 - float(_dilated_mask.sum()) / V_t.numel()
                if _masked_frac < 0.1:
                    _dilated_mask = None  # not enough savings, skip

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
            data = _compute_poisson_loss(pred, V_t, asymmetric_penalty, _dilated_mask)
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

        # Add boundary containment penalty if specified
        # Penalizes splats whose effective support extends beyond the volume bounds
        if boundary_penalty is not None and boundary_penalty > 0:
            centers, L, _ = model.current_params()
            # Diagonal of covariance: Sigma_ii = sum_j(L[i,j]^2)
            sigma_diag = torch.sum(L * L, dim=2)  # (N, d)
            # Effective radius per dimension per splat
            radii = model.truncate * torch.sqrt(
                torch.clamp(sigma_diag, min=1e-8)
            )  # (N, d)
            shape_t = torch.tensor(
                model.shape, dtype=torch.float32, device=centers.device
            )
            # Overflow past lower bound (center too close to 0)
            overflow_lo = torch.relu(radii - centers)  # (N, d)
            # Overflow past upper bound (center too close to shape-1)
            overflow_hi = torch.relu(radii - (shape_t - 1.0 - centers))  # (N, d)
            boundary_loss = torch.mean(overflow_lo**2 + overflow_hi**2)
            data = data + boundary_penalty * boundary_loss

        return data

    return loss_fn


@torch.compile(fullgraph=False)
def _compute_poisson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    asymmetric_penalty: float | None,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Poisson deviance loss with optional spatial masking.

    When ``region_mask`` is provided (dilated signal mask), the FULL loss
    (including asymmetric penalty) is computed only within the mask.
    Voxels outside the mask get zero gradient, causing the CUDA backward
    kernel to skip them via its existing ``if (dL_dI == 0) continue`` check.
    """
    eps = 1e-8
    Vc = torch.clamp(target, min=0.0)
    Pc = torch.clamp(pred, min=eps)
    # Per-element deviance (computed once, reused below)
    per_elem = Pc - Vc + torch.xlogy(Vc, torch.clamp(Vc / Pc, min=eps))

    # Apply spatial mask: zero gradient outside dilated signal region
    if region_mask is not None:
        per_elem = per_elem * region_mask
        N = region_mask.sum().clamp(min=1)
    else:
        N = target.numel()

    data = 2.0 * torch.sum(per_elem) / N

    if asymmetric_penalty is not None:
        over_mask = (pred > target).float()
        if region_mask is not None:
            over_mask = over_mask * region_mask
        over_dev = 2.0 * torch.sum(over_mask * per_elem) / N
        data = data + (asymmetric_penalty - 1.0) * over_dev

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
