"""Volume rendering for Gaussian splats.

This module provides GPU-accelerated volume rendering for Gaussian splats,
with automatic backend selection (CUDA > MPS > CPU).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch

from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

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
    from luxar.gsplats.utils.device import resolve_torch_device

    return str(resolve_torch_device())


def render_to_volume_tensor(
    gsplat_data: GSplatData,
    shape: Tuple[int, ...],
    device: str | None = None,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    intensity_floor: float = 1e-5,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Render Gaussian splats to a volume, returning a GPU tensor.

    Same as :func:`render_to_volume` but returns a ``torch.Tensor`` on the
    rendering device instead of a NumPy array.  This avoids an unnecessary
    GPU → CPU copy when the result will be consumed by further GPU operations
    (e.g. quality-metric computation).

    Parameters
    ----------
    gsplat_data : GSplatData
        The Gaussian splat data to render.
    shape : Tuple[int, ...]
        Output volume shape (e.g., (128, 128, 128) for 3D).
    device : str, optional
        Device to use for rendering. If None, auto-detects the best device.
    truncate : float, default=DEFAULT_TRUNCATION_RADIUS
        Truncation radius in standard deviations.
    intensity_floor : float, default=1e-5
        Minimum intensity threshold for amplitude-aware culling.
    chunk_size : int, optional
        Chunk size for memory management when processing large volumes.

    Returns
    -------
    torch.Tensor
        Rendered volume on the rendering device.
    """
    # Auto-detect device if not specified
    if device is None:
        device = auto_detect_device()

    # Convert to PyTorch tensors
    centers_t = torch.from_numpy(gsplat_data.centers).to(device)
    amps_t = torch.from_numpy(gsplat_data.amplitudes).to(device)

    # Unpack Cholesky factors from packed format (N, d*(d+1)/2) to (N, d, d) lower-triangular
    chol = gsplat_data.cholesky_factors
    ndim = gsplat_data.centers.shape[1]
    chol_t = torch.from_numpy(chol).to(device)

    Ls_t = torch.zeros((len(chol), ndim, ndim), device=device, dtype=torch.float32)

    # Unpack based on dimensionality (slicing on-device avoids temporary GPU tensors)
    if ndim == 2:
        # 2D: [L00, L10, L11]
        Ls_t[:, 0, 0] = chol_t[:, 0]
        Ls_t[:, 1, 0] = chol_t[:, 1]
        Ls_t[:, 1, 1] = chol_t[:, 2]
    elif ndim == 3:
        # 3D: [L00, L10, L11, L20, L21, L22]
        Ls_t[:, 0, 0] = chol_t[:, 0]
        Ls_t[:, 1, 0] = chol_t[:, 1]
        Ls_t[:, 1, 1] = chol_t[:, 2]
        Ls_t[:, 2, 0] = chol_t[:, 3]
        Ls_t[:, 2, 1] = chol_t[:, 4]
        Ls_t[:, 2, 2] = chol_t[:, 5]
    else:
        # nD: Generic unpacking (row-major lower triangular)
        idx = 0
        for i in range(ndim):
            for j in range(i + 1):
                Ls_t[:, i, j] = chol_t[:, idx]
                idx += 1
    del chol_t

    # Use CUDA splatting backend if available — it's tiled, memory-efficient,
    # and much faster than the pure-PyTorch renderer (which creates massive
    # meshgrid intermediates that can OOM on large volumes).
    if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
        try:
            from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
                CUDA_BACKEND_AVAILABLE,
                CUDASplatFunction,
            )

            if CUDA_BACKEND_AVAILABLE:
                with torch.no_grad():
                    output: torch.Tensor = CUDASplatFunction.apply(  # type: ignore[no-untyped-call]
                        centers_t,
                        Ls_t,
                        amps_t,
                        tuple(shape),
                        truncate,
                        intensity_floor,
                        False,  # use_fp16
                    )
                    if output.shape != tuple(shape):
                        output = output.view(shape)
                    return output
        except (ImportError, Exception):
            pass  # Fall through to PyTorch renderer

    # Fallback: pure PyTorch renderer (works on all devices)
    return render_gaussians(
        shape=shape,
        centers=centers_t,
        Ls=Ls_t,
        amps=amps_t,
        truncate=truncate,
        intensity_floor=intensity_floor,
        chunk_size=chunk_size,
    )


def render_to_volume(
    gsplat_data: GSplatData,
    shape: Tuple[int, ...],
    device: str | None = None,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
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
        and amplitudes.
    shape : Tuple[int, ...]
        Output volume shape (e.g., (128, 128, 128) for 3D).
    device : str, optional
        Device to use for rendering. If None, auto-detects the best device.
        Options: "cuda", "mps", "cpu".
    truncate : float, default=DEFAULT_TRUNCATION_RADIUS
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
    - For 8K splats on 128³ volume: substantially faster than NumPy
      implementation (often orders of magnitude on GPU; varies by hardware)
    - Supports nD rendering with automatic chunking to prevent OOM
    - Uses standard Gaussian falloff: exp(-0.5 * ||y||^2)
    """
    return (
        render_to_volume_tensor(
            gsplat_data,
            shape=shape,
            device=device,
            truncate=truncate,
            intensity_floor=intensity_floor,
            chunk_size=chunk_size,
        )
        .cpu()
        .numpy()
    )
