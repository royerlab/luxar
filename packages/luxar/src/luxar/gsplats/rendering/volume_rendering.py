"""Volume rendering for Gaussian splats.

This module provides GPU-accelerated volume rendering for Gaussian splats,
with automatic backend selection (CUDA > MPS > CPU).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch

from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


_cuda_render_warning_emitted = False


def _try_cuda_render(
    centers: torch.Tensor,
    cholesky_factors: torch.Tensor,
    amplitudes: torch.Tensor,
    shape: Tuple[int, ...],
    truncate: float,
    intensity_floor: float,
) -> torch.Tensor | None:
    """Render with the CUDA extension when its supported fast path is available."""
    # Keep this in sync with cuda/src/cuda_splatting.h (MIN_DIMS/MAX_DIMS).
    if not 2 <= len(shape) <= 8:
        return None

    try:
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            CUDA_BACKEND_AVAILABLE,
            CUDASplatFunction,
        )
    except ImportError:
        return None

    if not CUDA_BACKEND_AVAILABLE:
        return None

    try:
        with torch.no_grad():
            output: torch.Tensor = CUDASplatFunction.apply(  # type: ignore[no-untyped-call, unused-ignore]
                centers,
                cholesky_factors,
                amplitudes,
                tuple(shape),
                truncate,
                intensity_floor,
                False,  # use_fp16
            )
            if output.shape != tuple(shape):
                output = output.view(shape)
            return output
    except Exception as exc:
        global _cuda_render_warning_emitted  # noqa: PLW0603
        if not _cuda_render_warning_emitted:
            warnings.warn(
                "CUDA volume renderer failed with "
                f"{type(exc).__name__}: {exc}; falling back to PyTorch",
                RuntimeWarning,
                stacklevel=2,
            )
            _cuda_render_warning_emitted = True
        return None


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
        Device to use for rendering. ``None`` and ``"auto"`` auto-detect the
        best device.
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
    from luxar.gsplats.utils.device import resolve_torch_device

    device = str(resolve_torch_device(device))

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
    if device.startswith("cuda"):
        output = _try_cuda_render(
            centers_t, Ls_t, amps_t, shape, truncate, intensity_floor
        )
        if output is not None:
            return output

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
        Device to use for rendering. ``None`` and ``"auto"`` auto-detect the
        best device. Options: "auto", "cuda", "mps", "cpu".
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
