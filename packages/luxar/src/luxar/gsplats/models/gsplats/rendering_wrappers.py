# rendering_wrappers.py
"""
NumPy and PyTorch wrapper functions for Gaussian rendering.

This module provides user-friendly wrappers around the core rendering engine:
- render_gaussians_numpy: NumPy interface accepting GSplatData
- render_gaussians_pytorch: PyTorch interface accepting GSplatData

All wrappers work directly with GSplatData for clean, type-safe rendering.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.gsplats.utils.trils import unpack_tril
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS


def render_gaussians_numpy(
    shape: Sequence[int],
    result: GSplatData,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """
    CPU NumPy output wrapper around torch renderer (no grads).

    Takes a GSplatData and renders it to an image/volume.

    Parameters
    ----------
    shape : Sequence[int]
        Output image/volume shape.
    result : GSplatData
        Fitted Gaussian splat result containing centers, amplitudes,
        and cholesky_factors.
    truncate : float, default=DEFAULT_TRUNCATION_RADIUS
        Truncation radius in standard deviations.
    chunk_size : int, optional
        Chunk size for memory management.

    Returns
    -------
    np.ndarray
        Rendered image/volume.
    """
    # Input validation
    if len(result.amplitudes) == 0:
        return np.zeros(shape, dtype=np.float32)

    # Extract components from result
    centers: np.ndarray = result.centers.astype(np.float32)
    packed_L: np.ndarray = result.cholesky_factors.astype(np.float32)
    amps: np.ndarray = result.amplitudes.astype(np.float32)

    # Unpack Cholesky factors
    d = len(shape)
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors (CPU, no gradients needed)
    centers_torch = torch.from_numpy(centers)
    ls_torch = torch.from_numpy(ls)
    amps_torch = torch.from_numpy(amps)

    # Render using the PyTorch function
    with torch.no_grad():
        rendered = render_gaussians(
            shape,
            centers_torch,
            ls_torch,
            amps_torch,
            truncate=truncate,
            chunk_size=chunk_size,
        )

    return rendered.cpu().numpy()


def render_gaussians_pytorch(
    shape: Sequence[int],
    result: GSplatData,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    device: str = "cpu",
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    PyTorch wrapper for rendering gaussians.

    Takes a GSplatData and renders it to a tensor on specified device.

    Parameters
    ----------
    shape : Sequence[int]
        Output image/volume shape.
    result : GSplatData
        Fitted Gaussian splat result containing centers, amplitudes,
        and cholesky_factors.
    truncate : float, default=DEFAULT_TRUNCATION_RADIUS
        Truncation radius in standard deviations.
    device : str, default="cpu"
        PyTorch device for computation.
    chunk_size : int, optional
        Chunk size for memory management.

    Returns
    -------
    torch.Tensor
        Rendered image/volume on specified device.
    """
    # Input validation
    if len(result.amplitudes) == 0:
        return torch.zeros(shape, dtype=torch.float32, device=device)

    # Extract components from result
    centers: np.ndarray = result.centers.astype(np.float32)
    packed_L: np.ndarray = result.cholesky_factors.astype(np.float32)
    amps: np.ndarray = result.amplitudes.astype(np.float32)

    # Unpack Cholesky factors
    d = len(shape)
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors
    centers_torch = torch.from_numpy(centers).to(device=device)
    ls_torch = torch.from_numpy(ls).to(device=device)
    amps_torch = torch.from_numpy(amps).to(device=device)

    # Render using the PyTorch function (no gradients needed for inference)
    with torch.no_grad():
        rendered = render_gaussians(
            shape,
            centers_torch,
            ls_torch,
            amps_torch,
            truncate=truncate,
            chunk_size=chunk_size,
        )

    return rendered
