"""
Python wrapper for the NLM CUDA extension.

Handles parameter translation (patch_size → patch_half) and provides
a clean API matching the PyTorch backend interface.
"""

from __future__ import annotations

import torch


def nlm_cuda_denoise(
    volume: torch.Tensor,
    h: float,
    patch_size: int = 3,
    search_distance: int = 5,
) -> torch.Tensor:
    """Non-Local Means denoising via custom CUDA kernels.

    Parameters
    ----------
    volume : torch.Tensor
        2D ``(H, W)`` or 3D ``(D, H, W)`` float32 CUDA tensor.
    h : float
        Filtering strength.
    patch_size : int
        Comparison patch side length (must be 3 or 5).
    search_distance : int
        Search window half-size (must be 5 or 7).

    Returns
    -------
    torch.Tensor
        Denoised tensor, same shape/dtype/device as input.
    """
    import nlm_cuda_backend  # type: ignore[import-not-found]

    orig_dtype = volume.dtype
    vol = volume.float().contiguous()

    patch_half = patch_size // 2

    # Validate supported parameter combinations
    supported = {(1, 5), (1, 7), (2, 5), (2, 7)}
    if (patch_half, search_distance) not in supported:
        raise ValueError(
            f"CUDA NLM supports patch_size in {{3, 5}} and search_distance "
            f"in {{5, 7}}, got patch_size={patch_size} "
            f"(patch_half={patch_half}), search_distance={search_distance}. "
            f"Use backend='pytorch' for other parameter combinations."
        )

    if vol.ndim == 2:
        result: torch.Tensor = nlm_cuda_backend.nlm_denoise_2d(vol, h, patch_half, search_distance)
    elif vol.ndim == 3:
        result = nlm_cuda_backend.nlm_denoise_3d(vol, h, patch_half, search_distance)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {vol.ndim}D")

    return result.to(dtype=orig_dtype)
