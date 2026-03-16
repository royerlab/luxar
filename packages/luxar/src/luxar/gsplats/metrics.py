"""Quality metrics for Gaussian splat reconstructions.

All functions operate on PyTorch tensors and stay on the input device,
avoiding unnecessary GPU-CPU transfers.  Only scalar results are moved
to CPU (via ``.item()``).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(
    window_size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    """Create a 1-D Gaussian kernel (normalised to sum=1)."""
    coords = (
        torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    )
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    return g / g.sum()


def _gaussian_kernel_nd(
    window_size: int, sigma: float, ndim: int, device: torch.device
) -> torch.Tensor:
    """Create an n-D Gaussian kernel via iterated outer products.

    Returns a tensor of shape ``(1, 1, *([window_size] * ndim))`` suitable
    for use as a convolution weight.
    """
    k = _gaussian_kernel_1d(window_size, sigma, device)
    kernel = k
    for _ in range(ndim - 1):
        kernel = kernel.unsqueeze(-1) * k
    # (1, 1, W, W, ...) for conv weight
    return kernel.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def _ssim_nd(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    data_range: float,
) -> float:
    """Compute SSIM for a 2-D or 3-D tensor pair using convolution.

    Uses valid (no-padding) convolution so the border region — where the
    kernel would overlap with implicit zeros — is excluded from the mean.
    This matches the standard scikit-image implementation.
    """
    ndim = pred.ndim
    if ndim not in (2, 3):
        raise ValueError(f"_ssim_nd expects 2D or 3D tensors, got {ndim}D")

    device = pred.device
    sigma = 1.5
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel_nd(window_size, sigma, ndim, device)

    # Add batch + channel dims: (B=1, C=1, *spatial)
    p = pred.unsqueeze(0).unsqueeze(0)
    t = target.unsqueeze(0).unsqueeze(0)

    conv_fn = F.conv2d if ndim == 2 else F.conv3d
    # padding=0 → valid convolution: output excludes border pixels
    # where the kernel would extend beyond the input.
    mu_p = conv_fn(p, kernel, padding=0)
    mu_t = conv_fn(t, kernel, padding=0)

    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p_sq = conv_fn(p * p, kernel, padding=0) - mu_p_sq
    sigma_t_sq = conv_fn(t * t, kernel, padding=0) - mu_t_sq
    sigma_pt = conv_fn(p * t, kernel, padding=0) - mu_pt

    ssim_map = ((2.0 * mu_pt + C1) * (2.0 * sigma_pt + C2)) / (
        (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
    )

    return ssim_map.mean().item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float | None = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) on the input device.

    For 2-D and 3-D tensors a true n-D SSIM is computed via ``F.conv{2,3}d``.
    For higher-dimensional tensors the SSIM is averaged over all 3-D
    sub-volumes along the leading dimensions.

    Parameters
    ----------
    pred, target : torch.Tensor
        Predicted and reference tensors (same shape, >= 2-D).
    window_size : int
        Side length of the Gaussian weighting window (must be odd).
    data_range : float, optional
        Dynamic range of the data.  If *None*, computed as
        ``target.max() - target.min()``.

    Returns
    -------
    float
        Mean SSIM in [−1, 1] (typically [0, 1] for non-negative data).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if data_range is None:
        data_range = float((target.max() - target.min()).item())
    if data_range == 0.0:
        return 1.0  # constant images are identical

    # Clamp window_size to the smallest spatial dimension (must be odd)
    min_dim = min(pred.shape[-min(pred.ndim, 3) :])
    if window_size > min_dim:
        window_size = min_dim if min_dim % 2 == 1 else max(min_dim - 1, 1)

    ndim = pred.ndim
    if ndim in (2, 3):
        return _ssim_nd(pred, target, window_size, data_range)

    # >3D: average SSIM over all 3D sub-volumes along leading dims
    leading = pred.shape[:-3]
    total = 0.0
    count = 0
    for idx in torch.cartesian_prod(*[torch.arange(s) for s in leading]):
        idx_tuple = tuple(idx.tolist()) if idx.ndim > 0 else (idx.item(),)
        total += _ssim_nd(pred[idx_tuple], target[idx_tuple], window_size, data_range)
        count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB.

    PSNR = 10 * log10(data_range² / MSE).

    Parameters
    ----------
    pred, target : torch.Tensor
        Same shape.
    data_range : float, optional
        If *None*, uses ``target.max() - target.min()``.

    Returns
    -------
    float
        PSNR in dB.  Returns ``float('inf')`` when MSE is zero.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0.0:
        return float("inf")

    if data_range is None:
        data_range = float((target.max() - target.min()).item())
    if data_range == 0.0:
        return float("inf")

    return 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item()


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    ssim_window_size: int = 11,
) -> Dict[str, float]:
    """Compute a suite of quality metrics between predicted and target volumes.

    All heavy computation stays on the input device; only scalar results are
    returned.

    Parameters
    ----------
    pred, target : torch.Tensor
        Predicted and reference tensors (same shape, >= 2-D).
    data_range : float, optional
        Dynamic range.  If *None*, computed from *target*.
    ssim_window_size : int
        SSIM window size.

    Returns
    -------
    dict
        Keys: ``mse``, ``psnr_db``, ``ssim``, ``rel_l2``, ``max_abs_error``.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    diff = pred - target

    mse = torch.mean(diff**2).item()
    max_abs_error = torch.max(torch.abs(diff)).item()
    target_norm = torch.linalg.norm(target.reshape(-1)).item()
    rel_l2 = float(torch.linalg.norm(diff.reshape(-1)).item() / (target_norm + 1e-12))

    if data_range is None:
        data_range = float((target.max() - target.min()).item())

    psnr_db = compute_psnr(pred, target, data_range=data_range)
    ssim = compute_ssim(
        pred, target, window_size=ssim_window_size, data_range=data_range
    )

    return {
        "mse": mse,
        "psnr_db": psnr_db,
        "ssim": ssim,
        "rel_l2": rel_l2,
        "max_abs_error": max_abs_error,
    }
