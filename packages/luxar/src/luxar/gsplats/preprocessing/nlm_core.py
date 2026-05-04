"""
Non-Local Means denoising with multi-backend dispatch.

This module provides the public ``denoise_nlm`` function which automatically
selects the best available backend (CUDA kernel > PyTorch GPU > skimage CPU)
or allows explicit backend selection.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


def _resolve_backend(backend: str, device: torch.device) -> str:
    """Resolve ``'auto'`` to a concrete backend name."""
    if backend != "auto":
        return backend

    if device.type == "cuda":
        try:
            from luxar.gsplats.preprocessing.cuda import NLM_CUDA_AVAILABLE

            if NLM_CUDA_AVAILABLE:
                return "cuda"
        except ImportError:
            pass
        return "pytorch"

    if device.type == "mps":
        return "pytorch"

    # CPU: skimage is faster than PyTorch for NLM on CPU
    return "skimage"


def _validate_params(
    volume: torch.Tensor, patch_size: int, search_distance: int, h: float
) -> None:
    """Validate NLM parameters."""
    if volume.ndim not in (2, 3):
        raise ValueError(f"denoise_nlm supports 2D and 3D tensors, got {volume.ndim}D")
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}")
    if search_distance < 1:
        raise ValueError(f"search_distance must be >= 1, got {search_distance}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")


def _nlm_skimage(
    volume: torch.Tensor,
    h: float,
    patch_size: int,
    search_distance: int,
) -> torch.Tensor:
    """Reference implementation via scikit-image (CPU)."""
    try:
        from skimage.restoration import denoise_nl_means
    except ImportError as e:
        raise ImportError(
            "The 'skimage' backend requires scikit-image.\n"
            "Install with: pip install scikit-image"
        ) from e

    orig_device = volume.device
    orig_dtype = volume.dtype
    np_vol = volume.detach().cpu().float().numpy()

    denoised = denoise_nl_means(
        np_vol,
        h=h,
        patch_size=patch_size,
        patch_distance=search_distance,
        fast_mode=True,
    ).astype(np.float32)

    return torch.from_numpy(denoised).to(device=orig_device, dtype=orig_dtype)


def denoise_nlm(
    volume: torch.Tensor,
    h: float,
    patch_size: int = 3,
    search_distance: int = 5,
    *,
    backend: str = "auto",
    device: Union[str, torch.device, None] = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Non-Local Means denoising for 2D images and 3D volumes.

    For each pixel/voxel *x*, computes a weighted average over a local search
    neighbourhood.  Weights are derived from the similarity of small patches
    centred on *x* and each neighbour *y*:

        NLM(x) = sum_y  w(x,y) * I(y)  /  sum_y w(x,y)
        w(x,y) = exp( -||P(x) - P(y)||^2 / (patch_vol * h^2) )

    Three backends are available, selected automatically or via *backend*:

    * ``'cuda'``   — bare-metal CUDA kernel (fastest, requires compiled ext)
    * ``'pytorch'`` — pure-PyTorch GPU implementation
    * ``'skimage'`` — scikit-image CPU reference (slowest, always available)

    Parameters
    ----------
    volume : torch.Tensor
        2D ``(H, W)`` or 3D ``(D, H, W)`` input tensor, float32, ideally
        normalised to [0, 1].
    h : float
        Filtering strength.  Larger values smooth more aggressively.
    patch_size : int
        Side length of comparison patches (must be odd).  Default 3.
    search_distance : int
        Half-size of the search window around each pixel/voxel.  Default 5.
    backend : str
        ``'auto'`` (default), ``'cuda'``, ``'pytorch'``, or ``'skimage'``.
    device : str or torch.device, optional
        Target device.  If *None*, uses ``volume.device``.
    chunk_size : int, optional
        For large 3D volumes, process in overlapping chunks of this many
        slices along dim 0.  Only used by the ``'pytorch'`` backend.

    Returns
    -------
    torch.Tensor
        Denoised tensor, same shape, dtype, and device as *volume*.
    """
    if not isinstance(volume, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, got {type(volume).__name__}. "
            "Convert with torch.from_numpy(arr) first."
        )

    if device is not None:
        volume = volume.to(torch.device(device))

    _validate_params(volume, patch_size, search_distance, h)

    resolved = _resolve_backend(backend, volume.device)

    if resolved == "cuda":
        from .cuda.nlm_cuda_wrapper import nlm_cuda_denoise

        try:
            return nlm_cuda_denoise(volume, h, patch_size, search_distance)
        except (ValueError, RuntimeError) as exc:
            # Unsupported params (e.g., search_distance too large for GPU
            # shared memory) or runtime kernel failures (driver mismatch,
            # OOM, etc.) — fall back to PyTorch backend.
            import warnings

            warnings.warn(
                f"CUDA NLM: {exc} — falling back to PyTorch backend "
                f"(slower but supports any parameter combination).",
                UserWarning,
                stacklevel=2,
            )
            from .nlm_pytorch import nlm_pytorch_denoise

            return nlm_pytorch_denoise(
                volume, h, patch_size, search_distance, chunk_size=chunk_size
            )
    elif resolved == "pytorch":
        from .nlm_pytorch import nlm_pytorch_denoise

        return nlm_pytorch_denoise(
            volume, h, patch_size, search_distance, chunk_size=chunk_size
        )
    elif resolved == "skimage":
        return _nlm_skimage(volume, h, patch_size, search_distance)
    else:
        raise ValueError(
            f"Unknown backend: {resolved!r}. "
            "Choose from 'auto', 'cuda', 'pytorch', 'skimage'."
        )
