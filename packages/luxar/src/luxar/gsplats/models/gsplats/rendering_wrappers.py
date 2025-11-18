# rendering_wrappers.py
"""
NumPy and PyTorch wrapper functions for Gaussian rendering.

This module provides user-friendly wrappers around the core rendering engine:
- render_gaussians_numpy: NumPy interface with auto-parameter extraction
- render_gaussians_pytorch: PyTorch interface with auto-parameter extraction
- render_gaussians_batched: Batched rendering with optional sharpness

All wrappers automatically extract centers, Cholesky factors, and sharpness
from packed parameter arrays, treating all geometric parameters consistently.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.gsplats.utils.trils import tril_size, unpack_tril


def render_gaussians_numpy(
    shape: Sequence[int],
    params_full: np.ndarray,
    amps: np.ndarray,
    truncate: float = 3.0,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """
    CPU NumPy output wrapper around torch renderer (no grads).

    Automatically extracts centers, Cholesky factors, and sharpness from params_full,
    treating all parameters consistently.

    Parameters
    ----------
    shape : Sequence[int]
        Output image/volume shape.
    params_full : np.ndarray, shape (N, d + d*(d+1)//2 + 1)
        Splat parameters: [centers, packed_cholesky, sharpness].
        Last column contains per-splat sharpness values.
        For backward compatibility, also accepts (N, d + d*(d+1)//2) without sharpness.
    amps : np.ndarray, shape (N,)
        Splat amplitudes.
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
    if params_full.size == 0:
        return np.zeros(shape, dtype=np.float32)

    # Determine dimensionality and check parameter format
    d = len(shape)
    tril_elements = tril_size(d)
    expected_without_sharpness = d + tril_elements
    expected_with_sharpness = d + tril_elements + 1

    # Auto-detect format and extract sharpness
    if params_full.shape[1] == expected_with_sharpness:
        # New format: includes sharpness in last column
        sharpness = params_full[:, -1].astype(np.float32)
        centers = params_full[:, :d].astype(np.float32)
        packed_L = params_full[:, d:-1].astype(np.float32)
    elif params_full.shape[1] == expected_without_sharpness:
        # Old format: no sharpness column (backward compatibility)
        sharpness = None
        centers = params_full[:, :d].astype(np.float32)
        packed_L = params_full[:, d:].astype(np.float32)
    else:
        raise ValueError(
            f"params_full should have either {expected_without_sharpness} columns (without sharpness) "
            f"or {expected_with_sharpness} columns (with sharpness) for {d}D data, "
            f"got {params_full.shape[1]}"
        )

    # Unpack Cholesky factors
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors (CPU, no gradients needed)
    centers_torch = torch.tensor(centers, dtype=torch.float32, device="cpu")
    ls_torch = torch.tensor(ls, dtype=torch.float32, device="cpu")
    amps_torch = torch.tensor(
        amps.astype(np.float32), dtype=torch.float32, device="cpu"
    )

    # Handle sharpness
    if sharpness is None:
        # Default to standard Gaussian sharpness (s = 2.0)
        sharpness_torch = torch.full(
            (centers_torch.shape[0],), 2.0, dtype=torch.float32, device="cpu"
        )
    else:
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
    params_full: np.ndarray,
    amps: np.ndarray,
    truncate: float = 3.0,
    device: str = "cpu",
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    PyTorch wrapper for rendering gaussians with packed parameters.

    Automatically extracts centers, Cholesky factors, and sharpness from params_full,
    treating all parameters consistently (same as render_gaussians_numpy).

    Parameters
    ----------
    shape : Sequence[int]
        Output image/volume shape.
    params_full : np.ndarray, shape (N, d + d*(d+1)//2 + 1)
        Splat parameters: [centers, packed_cholesky, sharpness].
        For backward compatibility, also accepts (N, d + d*(d+1)//2) without sharpness.
    amps : np.ndarray, shape (N,)
        Splat amplitudes.
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
    if params_full.size == 0:
        return torch.zeros(shape, dtype=torch.float32, device=device)

    # Determine dimensionality and check parameter format
    d = len(shape)
    tril_elements = tril_size(d)
    expected_without_sharpness = d + tril_elements
    expected_with_sharpness = d + tril_elements + 1

    # Auto-detect format and extract sharpness (same logic as render_gaussians_numpy)
    if params_full.shape[1] == expected_with_sharpness:
        # New format: includes sharpness in last column
        sharpness = params_full[:, -1].astype(np.float32)
        centers = params_full[:, :d].astype(np.float32)
        packed_L = params_full[:, d:-1].astype(np.float32)
    elif params_full.shape[1] == expected_without_sharpness:
        # Old format: no sharpness column (backward compatibility)
        sharpness = None
        centers = params_full[:, :d].astype(np.float32)
        packed_L = params_full[:, d:].astype(np.float32)
    else:
        raise ValueError(
            f"params_full should have either {expected_without_sharpness} columns (without sharpness) "
            f"or {expected_with_sharpness} columns (with sharpness) for {d}D data, "
            f"got {params_full.shape[1]}"
        )

    # Unpack Cholesky factors
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors
    centers_torch = torch.tensor(centers, dtype=torch.float32, device=device)
    ls_torch = torch.tensor(ls, dtype=torch.float32, device=device)
    amps_torch = torch.tensor(
        amps.astype(np.float32), dtype=torch.float32, device=device
    )

    # Handle sharpness
    if sharpness is None:
        # Default to standard Gaussian sharpness (s = 2.0)
        sharpness_torch = torch.full(
            (centers_torch.shape[0],), 2.0, dtype=torch.float32, device=device
        )
    else:
        sharpness_torch = torch.tensor(sharpness, dtype=torch.float32, device=device)

    # Render using the PyTorch function
    result = render_gaussians(
        shape,
        centers_torch,
        ls_torch,
        amps_torch,
        sharpness_torch,
        truncate=truncate,
        chunk_size=chunk_size,
    )

    return result


def render_gaussians_batched(
    shape: Sequence[int],
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    sharpness: Optional[torch.Tensor] = None,
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
    sharpness : torch.Tensor, shape (N,), optional
        Per-splat sharpness. If None, defaults to 2.0 (standard Gaussian).
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
    # Default to standard Gaussian (s = 2.0) if sharpness not provided
    if sharpness is None:
        sharpness = torch.full(
            (centers.shape[0],),
            2.0,
            dtype=torch.float32,
            device=centers.device,
        )
    return render_gaussians(
        shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
    )
