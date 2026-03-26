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
        Search window half-size (must be 5, 7, 9, 11, 13, or 15).

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
    supported_patch = {1, 2}  # patch_half: 1 → patch_size=3, 2 → patch_size=5
    supported_search = {5, 7, 9, 11, 13, 15}
    if patch_half not in supported_patch or search_distance not in supported_search:
        raise ValueError(
            f"CUDA NLM supports patch_size in {{3, 5}} and search_distance "
            f"in {{5, 7, 9, 11, 13, 15}}, got patch_size={patch_size} "
            f"(patch_half={patch_half}), search_distance={search_distance}. "
            f"Use backend='pytorch' for other parameter combinations."
        )

    # Check shared memory for 3D with large search distances.
    # Ampere (RTX 3090) supports 100 KB, Ada/Hopper 164+ KB.
    if vol.ndim == 3 and search_distance >= 9:
        halo = search_distance + patch_half
        smem_z = 4 + 2 * halo  # TILE_3D_Z=4
        smem_y = 8 + 2 * halo  # TILE_3D_Y=8
        smem_x = 8 + 2 * halo  # TILE_3D_X=8
        smem_bytes = smem_z * smem_y * smem_x * 4  # float32

        # Query device shared memory limit
        device_idx = vol.device.index or 0
        props = torch.cuda.get_device_properties(device_idx)
        max_smem = getattr(
            props, "shared_memory_per_block_optin",
            getattr(props, "max_shared_memory_per_block_optin", 48 * 1024),
        )
        if smem_bytes > max_smem:
            raise ValueError(
                f"search_distance={search_distance} requires {smem_bytes // 1024} KB "
                f"shared memory, but {props.name} supports max "
                f"{max_smem // 1024} KB. Use a smaller search_distance or "
                f"backend='pytorch'."
            )

    if vol.ndim == 2:
        result: torch.Tensor = nlm_cuda_backend.nlm_denoise_2d(vol, h, patch_half, search_distance)
    elif vol.ndim == 3:
        result = nlm_cuda_backend.nlm_denoise_3d(vol, h, patch_half, search_distance)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {vol.ndim}D")

    return result.to(dtype=orig_dtype)
