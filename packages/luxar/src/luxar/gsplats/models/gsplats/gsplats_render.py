# pytorch_splats_full.py
# Oriented (full-covariance) Gaussian splats in nD with PyTorch
# - Positive-definite Σ via Cholesky L (lower-triangular)
# - Centers inside volume via sigmoid parameterization
# - Amplitudes via softplus (>= 0)
# - Stable rendering using triangular solves (no cholesky_inverse required)
#
# Public API:
#   - fit_gaussian_splats_torch_full(...)
#   - render_gaussians_full_numpy(...)
#   - render_gaussians_full_torch(...)
#
# Author: (you)
# License: MIT (or your choice)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus
from luxar.gsplats.utils.trils import unpack_tril

# -------------------------------
# Rendering (no grads)
# -------------------------------


@torch.no_grad()
def render_gaussians_full_torch(
    shape: Sequence[int],
    params_full: np.ndarray,  # (N, d + d*(d+1)//2)
    amps: np.ndarray,  # (N,)
    truncate: float = 3.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Render oriented Gaussians (no grads), reusing the model's rasterizer.
    """
    shape = tuple(shape)
    d = len(shape)
    N = len(amps)
    device_t = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if N == 0:
        return torch.zeros(shape, dtype=torch.float32, device=device_t)

    centers0 = params_full[:, :d]
    L_packed = params_full[:, d:]
    L0 = unpack_tril(L_packed, d)  # (N, d, d)

    # Minimal diag floor just to satisfy parameterization during reconstruction
    sigma_min_diag = [1e-6] * d

    mdl = GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps,
        sigma_min_diag=sigma_min_diag,
        sigma_max_diag=None,
        truncate=truncate,
        device=device_t,
    )

    # Overwrite raw params so model renders *exact* values we passed in
    shape_arr = np.array(shape, dtype=np.float32)
    u = np.clip(centers0 / np.maximum(shape_arr - 1.0, 1.0), 1e-6, 1.0 - 1e-6)
    mdl.raw_mu.data = torch.tensor(
        np.log(u) - np.log(1 - u), dtype=torch.float32, device=device_t
    )

    diag = np.diagonal(L0, axis1=1, axis2=2)
    mdl.raw_L_diag.data = torch.tensor(
        stable_inverse_softplus(diag - np.asarray(sigma_min_diag, np.float32)),
        dtype=torch.float32,
        device=device_t,
    )

    # fill off-diagonals
    off_idx = [(i, j) for i in range(d) for j in range(i)]
    if len(off_idx):
        off = np.stack([L0[:, i, j] for (i, j) in off_idx], axis=1)
        mdl.L_off.data = torch.tensor(off, dtype=torch.float32, device=device_t)

    mdl.raw_a.data = torch.tensor(
        stable_inverse_softplus(np.maximum(amps, 1e-8)),
        dtype=torch.float32,
        device=device_t,
    )

    return mdl()


def render_gaussians_full_numpy(
    shape: Sequence[int],
    params_full: np.ndarray,
    amps: np.ndarray,
    truncate: float = 3.0,
) -> np.ndarray:
    """CPU NumPy output wrapper around torch renderer (no grads)."""
    return (
        render_gaussians_full_torch(shape, params_full, amps, truncate, device="cpu")
        .cpu()
        .numpy()
    )
