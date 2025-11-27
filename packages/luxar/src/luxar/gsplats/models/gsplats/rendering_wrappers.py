# rendering_wrappers.py
"""
NumPy and PyTorch wrapper functions for Gaussian rendering.

This module provides user-friendly wrappers around the core rendering engine:
- render_gaussians_numpy: NumPy interface accepting GaussianSplatResult
- render_gaussians_pytorch: PyTorch interface accepting GaussianSplatResult
- render_gaussians_batched: Batched rendering with required sharpness

All wrappers work directly with GaussianSplatResult for clean, type-safe rendering.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.fit_result import GaussianSplatResult
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.gsplats.utils.trils import unpack_tril


def render_gaussians_numpy(
    shape: Sequence[int],
    result: GaussianSplatResult,
    truncate: float = 3.0,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """
    CPU NumPy output wrapper around torch renderer (no grads).

    Takes a GaussianSplatResult and renders it to an image/volume.

    Parameters
    ----------
    shape : Sequence[int]
        Output image/volume shape.
    result : GaussianSplatResult
        Fitted Gaussian splat result containing centers, amplitudes,
        cholesky_factors, and sharpnesses.
    truncate : float, default=3.0
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
    sharpness: np.ndarray = result.sharpnesses.astype(np.float32)
    amps: np.ndarray = result.amplitudes.astype(np.float32)

    # Unpack Cholesky factors
    d = len(shape)
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors (CPU, no gradients needed)
    centers_torch = torch.tensor(centers, dtype=torch.float32, device="cpu")
    ls_torch = torch.tensor(ls, dtype=torch.float32, device="cpu")
    amps_torch = torch.tensor(amps, dtype=torch.float32, device="cpu")
    sharpness_torch = torch.tensor(sharpness, dtype=torch.float32, device="cpu")

    # Render using the PyTorch function
    with torch.no_grad():
        result = render_gaussians(
            shape,
            centers_torch,
            ls_torch,
            amps_torch,
            sharpness_torch,
            truncate=truncate,
            chunk_size=chunk_size,
        )

    return result.cpu().numpy()


def render_gaussians_pytorch(
    shape: Sequence[int],
    result: GaussianSplatResult,
    truncate: float = 3.0,
    device: str = "cpu",
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    PyTorch wrapper for rendering gaussians.

    Takes a GaussianSplatResult and renders it to a tensor on specified device.

    Parameters
    ----------
    shape : Sequence[int]
        Output image/volume shape.
    result : GaussianSplatResult
        Fitted Gaussian splat result containing centers, amplitudes,
        cholesky_factors, and sharpnesses.
    truncate : float, default=3.0
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
    sharpness: np.ndarray = result.sharpnesses.astype(np.float32)
    amps: np.ndarray = result.amplitudes.astype(np.float32)

    # Unpack Cholesky factors
    d = len(shape)
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors
    centers_torch = torch.tensor(centers, dtype=torch.float32, device=device)
    ls_torch = torch.tensor(ls, dtype=torch.float32, device=device)
    amps_torch = torch.tensor(amps, dtype=torch.float32, device=device)
    sharpness_torch = torch.tensor(sharpness, dtype=torch.float32, device=device)

    # Render using the PyTorch function
    rendered = render_gaussians(
        shape,
        centers_torch,
        ls_torch,
        amps_torch,
        sharpness_torch,
        truncate=truncate,
        chunk_size=chunk_size,
    )

    return rendered


def render_gaussians_batched(
    shape: Sequence[int],
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    sharpness: torch.Tensor,
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Batched wrapper for render_gaussians - identical functionality.

    Parameters
    ----------
    shape : Sequence[int]
        Output shape.
    centers : torch.Tensor, shape (N, d)
        Center positions.
    Ls : torch.Tensor, shape (N, d, d)
        Cholesky factors.
    amps : torch.Tensor, shape (N,)
        Amplitudes.
    sharpness : torch.Tensor, shape (N,)
        Per-splat sharpness values (required).
    truncate : float, default=3.0
        Truncation radius.
    intensity_floor : float, default=1e-5
        Minimum intensity for culling.
    chunk_size : int, optional
        Chunk size for memory management.

    Returns
    -------
    torch.Tensor
        Rendered output.
    """
    return render_gaussians(
        shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
    )
