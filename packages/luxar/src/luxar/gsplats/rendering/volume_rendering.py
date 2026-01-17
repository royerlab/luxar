"""Volume rendering for Gaussian splats.

This module provides GPU-accelerated volume rendering for Gaussian splats,
with automatic backend selection (CUDA > MPS > CPU).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch

from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


def auto_detect_device() -> str:
    """Auto-detect the best available device for rendering.

    Priority: CUDA > MPS > CPU

    Returns
    -------
    str
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def render_to_volume(
    gsplat_data: GSplatData,
    shape: Tuple[int, ...],
    device: str | None = None,
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Render Gaussian splats to a volume using GPU-accelerated rendering.

    This function automatically selects the fastest available backend (CUDA, MPS, or CPU)
    and uses the optimized PyTorch renderer from the models package.

    Parameters
    ----------
    gsplat_data : GSplatData
        The Gaussian splat data to render, containing centers, Cholesky factors,
        amplitudes, and sharpnesses.
    shape : Tuple[int, ...]
        Output volume shape (e.g., (128, 128, 128) for 3D).
    device : str, optional
        Device to use for rendering. If None, auto-detects the best device.
        Options: "cuda", "mps", "cpu".
    truncate : float, default=3.0
        Truncation radius in standard deviations. Gaussians are evaluated within
        this radius from their centers.
    intensity_floor : float, default=1e-5
        Minimum intensity threshold for amplitude-aware culling. Splats with
        contributions below this threshold are culled early for performance.
    chunk_size : int, optional
        Chunk size for memory management when processing large volumes. If None,
        automatically calculated based on available memory.

    Returns
    -------
    np.ndarray
        Rendered volume with the same shape as specified, as a NumPy array.

    Examples
    --------
    >>> from luxar.gsplats.rendering import render_to_volume
    >>> volume = render_to_volume(gsplat_data, shape=(128, 128, 128))

    Notes
    -----
    - The rendering uses the fast PyTorch renderer with specialized 2D/3D fast paths
    - For 8K splats on 128³ volume: ~100-1000x faster than NumPy implementation
    - Supports nD rendering with automatic chunking to prevent OOM
    - Sharpness values control falloff: exp(-0.5 * ||y||^s) where s is sharpness
    """
    # Auto-detect device if not specified
    if device is None:
        device = auto_detect_device()

    # Convert to PyTorch tensors
    centers_t = torch.from_numpy(gsplat_data.centers).to(device)
    amps_t = torch.from_numpy(gsplat_data.amplitudes).to(device)
    sharpness_t = torch.from_numpy(gsplat_data.sharpnesses).to(device)

    # Unpack Cholesky factors from packed format (N, d*(d+1)/2) to (N, d, d) lower-triangular
    chol = gsplat_data.cholesky_factors
    ndim = gsplat_data.centers.shape[1]

    Ls_t = torch.zeros((len(chol), ndim, ndim), device=device, dtype=torch.float32)

    # Unpack based on dimensionality
    if ndim == 2:
        # 2D: [L00, L10, L11]
        Ls_t[:, 0, 0] = torch.from_numpy(chol[:, 0]).to(device)
        Ls_t[:, 1, 0] = torch.from_numpy(chol[:, 1]).to(device)
        Ls_t[:, 1, 1] = torch.from_numpy(chol[:, 2]).to(device)
    elif ndim == 3:
        # 3D: [L00, L10, L11, L20, L21, L22]
        Ls_t[:, 0, 0] = torch.from_numpy(chol[:, 0]).to(device)
        Ls_t[:, 1, 0] = torch.from_numpy(chol[:, 1]).to(device)
        Ls_t[:, 1, 1] = torch.from_numpy(chol[:, 2]).to(device)
        Ls_t[:, 2, 0] = torch.from_numpy(chol[:, 3]).to(device)
        Ls_t[:, 2, 1] = torch.from_numpy(chol[:, 4]).to(device)
        Ls_t[:, 2, 2] = torch.from_numpy(chol[:, 5]).to(device)
    else:
        # nD: Generic unpacking (row-major lower triangular)
        idx = 0
        for i in range(ndim):
            for j in range(i + 1):
                Ls_t[:, i, j] = torch.from_numpy(chol[:, idx]).to(device)
                idx += 1

    # Render using GPU-accelerated renderer
    rendered_t = render_gaussians(
        shape=shape,
        centers=centers_t,
        Ls=Ls_t,
        amps=amps_t,
        sharpness=sharpness_t,
        truncate=truncate,
        intensity_floor=intensity_floor,
        chunk_size=chunk_size,
    )

    # Convert back to NumPy
    return rendered_t.cpu().numpy()
