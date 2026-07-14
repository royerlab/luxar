"""
Loss function creation for Gaussian splat fitting.

Speed optimisations (empirically validated)
--------------------------------------------
1. **Poisson deviance dedup** (−4.9%): The per-element deviance
   ``Pc - Vc + xlogy(Vc, Vc/Pc)`` is computed once and reused for both
   the base loss and the asymmetric over-prediction penalty.  Previously
   it was computed twice (once for sum, once for the masked over-prediction sum).

2. **torch.compile on CUDA** (−14.4% in benchmarked cases): CUDA loss kernels
   are compiled opportunistically with a safe eager fallback. CPU and MPS use
   eager PyTorch to avoid runtime C++ toolchain requirements.
"""

from __future__ import annotations

import warnings
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

    poisson_loss = _compile_loss_kernel(_compute_poisson_loss, V_t.device)
    l1_loss = _compile_loss_kernel(_compute_l1_loss, V_t.device)
    mse_loss = _compile_loss_kernel(_compute_mse_loss, V_t.device)

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
            data = poisson_loss(pred, V_t, asymmetric_penalty)
        elif loss_type.lower() == "l1":
            data = l1_loss(pred, V_t, asymmetric_penalty)
        elif loss_type.lower() == "mse":
            data = mse_loss(pred, V_t, asymmetric_penalty)
        else:
            # An unknown loss_type must raise, not silently fall through to
            # MSE — a typo ("poisson_deviance", "mes", etc.) should fail
            # immediately so the caller knows.
            raise ValueError(
                f"Unknown loss_type: {loss_type!r}. "
                f"Expected one of: 'poisson', 'l1', 'mse'."
            )

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


def _compile_loss_kernel(
    fn: Callable[[torch.Tensor, torch.Tensor, float | None], torch.Tensor],
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor, float | None], torch.Tensor]:
    """Compile CUDA loss kernels with a runtime-safe eager fallback."""
    if device.type != "cuda":
        return fn

    try:
        compiled_fn = torch.compile(fn, fullgraph=False)
    except Exception as exc:
        warnings.warn(
            f"torch.compile unavailable for loss kernel {fn.__name__}; using eager PyTorch: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return fn

    compile_failed = False

    def wrapped(
        pred: torch.Tensor, target: torch.Tensor, asymmetric_penalty: float | None
    ) -> torch.Tensor:
        nonlocal compile_failed
        if compile_failed:
            return fn(pred, target, asymmetric_penalty)
        try:
            return compiled_fn(pred, target, asymmetric_penalty)
        except Exception as exc:
            compile_failed = True
            warnings.warn(
                f"torch.compile failed for loss kernel {fn.__name__}; using eager PyTorch: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return fn(pred, target, asymmetric_penalty)

    return wrapped


def _compute_poisson_loss(
    pred: torch.Tensor, target: torch.Tensor, asymmetric_penalty: float | None
) -> torch.Tensor:
    """Compute Poisson deviance loss.

    Optimized: per-element deviance is computed once, then reused for both
    the total deviance and the asymmetric over-prediction penalty.
    """
    eps = 1e-8
    Vc = torch.clamp(target, min=0.0)
    Pc = torch.clamp(pred, min=eps)
    # Per-element deviance (computed once, reused below)
    per_elem = Pc - Vc + torch.xlogy(Vc, torch.clamp(Vc / Pc, min=eps))
    N = target.numel()
    data = 2.0 * torch.sum(per_elem) / N

    if asymmetric_penalty is not None:
        # Additional penalty for over-prediction regions (reuses per_elem)
        over_mask = (pred > target).float()
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
