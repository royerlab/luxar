"""
Pure-PyTorch Non-Local Means implementation.

Supports 2D images and 3D volumes with optional chunked processing for
large data.  All operations use the input tensor's device, so GPU
acceleration is automatic when the input resides on CUDA or MPS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _patch_distance_2d(
    padded: torch.Tensor,
    H: int,
    W: int,
    pad: int,
    half_patch: int,
    patch_size: int,
    dy: int,
    dx: int,
) -> torch.Tensor:
    """Compute mean squared patch distance for a single search offset (2D).

    Returns a ``(H, W)`` tensor of normalised squared distances between the
    patch centred on each output pixel and the patch at offset ``(dy, dx)``.
    """
    dist = torch.zeros(H, W, device=padded.device, dtype=padded.dtype)
    n_elements = patch_size * patch_size

    for py in range(-half_patch, half_patch + 1):
        for px in range(-half_patch, half_patch + 1):
            center = padded[
                pad + py : pad + py + H,
                pad + px : pad + px + W,
            ]
            shifted = padded[
                pad + dy + py : pad + dy + py + H,
                pad + dx + px : pad + dx + px + W,
            ]
            diff = center - shifted
            dist += diff * diff

    return dist / n_elements


def _nlm_2d(
    image: torch.Tensor,
    h: float,
    patch_size: int,
    search_distance: int,
) -> torch.Tensor:
    """Non-Local Means for a 2D image using direct slicing."""
    device = image.device
    dtype = image.dtype
    H, W = image.shape
    half_patch = patch_size // 2

    # Pad by search_distance + half_patch on each side (reflect for borders)
    pad = search_distance + half_patch
    padded = (
        F.pad(
            image.unsqueeze(0).unsqueeze(0),
            [pad, pad, pad, pad],
            mode="reflect",
        )
        .squeeze(0)
        .squeeze(0)
    )

    output = torch.zeros(H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(H, W, device=device, dtype=dtype)
    h_sq = h * h

    for dy in range(-search_distance, search_distance + 1):
        for dx in range(-search_distance, search_distance + 1):
            dist_sq = _patch_distance_2d(
                padded, H, W, pad, half_patch, patch_size, dy, dx
            )

            weights = torch.exp(-dist_sq / h_sq)

            # Pixel values at the shifted position
            shifted_vals = padded[
                pad + dy : pad + dy + H,
                pad + dx : pad + dx + W,
            ]

            output += weights * shifted_vals
            weight_sum += weights

    return output / weight_sum.clamp(min=1e-10)


def _patch_distance_3d(
    padded: torch.Tensor,
    D: int,
    H: int,
    W: int,
    pad: int,
    half_patch: int,
    patch_size: int,
    dz: int,
    dy: int,
    dx: int,
) -> torch.Tensor:
    """Compute mean squared patch distance for a single search offset (3D).

    Returns a ``(D, H, W)`` tensor of normalised squared distances.
    """
    dist = torch.zeros(D, H, W, device=padded.device, dtype=padded.dtype)
    n_elements = patch_size * patch_size * patch_size

    for pz in range(-half_patch, half_patch + 1):
        for py in range(-half_patch, half_patch + 1):
            for px in range(-half_patch, half_patch + 1):
                center = padded[
                    pad + pz : pad + pz + D,
                    pad + py : pad + py + H,
                    pad + px : pad + px + W,
                ]
                shifted = padded[
                    pad + dz + pz : pad + dz + pz + D,
                    pad + dy + py : pad + dy + py + H,
                    pad + dx + px : pad + dx + px + W,
                ]
                diff = center - shifted
                dist += diff * diff

    return dist / n_elements


def _nlm_3d(
    volume: torch.Tensor,
    h: float,
    patch_size: int,
    search_distance: int,
) -> torch.Tensor:
    """Non-Local Means for a 3D volume using direct slicing."""
    device = volume.device
    dtype = volume.dtype
    D, H, W = volume.shape
    half_patch = patch_size // 2

    pad = search_distance + half_patch
    padded = (
        F.pad(
            volume.unsqueeze(0).unsqueeze(0),
            [pad, pad, pad, pad, pad, pad],
            mode="reflect",
        )
        .squeeze(0)
        .squeeze(0)
    )

    output = torch.zeros(D, H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(D, H, W, device=device, dtype=dtype)
    h_sq = h * h

    for dz in range(-search_distance, search_distance + 1):
        for dy in range(-search_distance, search_distance + 1):
            for dx in range(-search_distance, search_distance + 1):
                dist_sq = _patch_distance_3d(
                    padded,
                    D,
                    H,
                    W,
                    pad,
                    half_patch,
                    patch_size,
                    dz,
                    dy,
                    dx,
                )

                weights = torch.exp(-dist_sq / h_sq)

                shifted_vals = padded[
                    pad + dz : pad + dz + D,
                    pad + dy : pad + dy + H,
                    pad + dx : pad + dx + W,
                ]

                output += weights * shifted_vals
                weight_sum += weights

    return output / weight_sum.clamp(min=1e-10)


def _nlm_3d_chunked(
    volume: torch.Tensor,
    h: float,
    patch_size: int,
    search_distance: int,
    chunk_size: int,
) -> torch.Tensor:
    """Non-Local Means for large 3D volumes, processed in overlapping chunks.

    Slices the volume along dim 0 into chunks of *chunk_size* slices with
    overlap of ``search_distance + patch_size // 2`` on each side.  Only the
    interior (non-halo) voxels are kept from each chunk.
    """
    D, H, W = volume.shape
    halo = search_distance + patch_size // 2
    output = torch.empty_like(volume)

    z = 0
    while z < D:
        z_end = min(z + chunk_size, D)

        # Expand chunk boundaries to include halo
        chunk_start = max(z - halo, 0)
        chunk_end = min(z_end + halo, D)

        chunk = volume[chunk_start:chunk_end]
        denoised_chunk = _nlm_3d(chunk, h, patch_size, search_distance)

        # Trim halo — keep only the interior corresponding to [z, z_end)
        trim_start = z - chunk_start
        trim_end = trim_start + (z_end - z)
        output[z:z_end] = denoised_chunk[trim_start:trim_end]

        z = z_end

    return output


def nlm_pytorch_denoise(
    volume: torch.Tensor,
    h: float,
    patch_size: int = 3,
    search_distance: int = 5,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Non-Local Means denoising via pure PyTorch operations.

    Parameters
    ----------
    volume : torch.Tensor
        2D ``(H, W)`` or 3D ``(D, H, W)`` float tensor.
    h : float
        Filtering strength.
    patch_size : int
        Comparison patch side length (odd).
    search_distance : int
        Search window half-size.
    chunk_size : int, optional
        If set and volume is 3D, process in overlapping chunks of this many
        slices along dim 0 to reduce peak GPU memory.

    Returns
    -------
    torch.Tensor
        Denoised tensor, same shape/dtype/device as input.
    """
    orig_dtype = volume.dtype
    vol = volume.float()

    if vol.ndim == 2:
        result = _nlm_2d(vol, h, patch_size, search_distance)
    elif vol.ndim == 3:
        if chunk_size is not None and chunk_size < vol.shape[0]:
            result = _nlm_3d_chunked(vol, h, patch_size, search_distance, chunk_size)
        else:
            result = _nlm_3d(vol, h, patch_size, search_distance)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {vol.ndim}D")

    return result.to(dtype=orig_dtype)
