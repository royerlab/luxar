"""NVRTC (runtime-compiled) backend for the NLM denoising kernels.

Compiles ``src/nlm_kernels.cuh`` -- the same device code the ``make
build-nlm-cuda`` extension uses -- with NVRTC on first use, caches the binary
on disk, and launches it on torch's current stream (see
``luxar/gsplats/_nvrtc.py``). :func:`nlm_denoise_2d` / :func:`nlm_denoise_3d`
take the same arguments and return the same tensor as the extension's
functions of the same name (a port of the host code in ``src/nlm_cuda.cu``).

Each ``(ndim, patch_half, search_dist)`` kernel compiles on its own the first
time it is used, so a run pays only for the one combination it needs.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F

from luxar.gsplats._nvrtc import Kernel, KernelProgram, get_runtime

SUPPORTED_PATCH_HALF = (1, 2)
SUPPORTED_SEARCH_DIST = (5, 7, 9, 11, 13, 15)

# Tile sizes: must match TILE_2D_* / TILE_3D_* in src/nlm_kernels.cuh.
TILE_2D_Y, TILE_2D_X = 16, 16
TILE_3D_Z, TILE_3D_Y, TILE_3D_X = 4, 8, 8

PROGRAM = KernelProgram(
    label="gsplat-nlm",
    # Angle brackets: resolve only through -I (never a same-named file in the CWD).
    source="#include <nlm_kernels.cuh>\n",
    include_dir=Path(__file__).parent / "src",
)


def _kernel(ndim: int, patch_half: int, search_dist: int, device_index: int) -> Kernel:
    """The loaded NLM kernel for one parameter combination, compiling on first use."""
    if (
        patch_half not in SUPPORTED_PATCH_HALF
        or search_dist not in SUPPORTED_SEARCH_DIST
    ):
        raise RuntimeError(
            f"Unsupported NLM params: patch_half={patch_half} search_dist={search_dist}. "
            "Supported: patch_size={3,5}, search_distance={5,7,9,11,13,15}"
        )
    name = f"nlm_{ndim}d_kernel<{patch_half}, {search_dist}>"
    return get_runtime().kernels(PROGRAM, [name], device_index)[name]


def _check_input(volume: torch.Tensor, ndim: int, h: float) -> None:
    """Port of the extension's TORCH_CHECKs on the input volume."""
    fn = f"nlm_denoise_{ndim}d"
    if volume.dim() != ndim:
        raise RuntimeError(f"{fn} expects {ndim}D tensor, got {volume.dim()}D")
    if not volume.is_cuda:
        raise RuntimeError("Input must be a CUDA tensor")
    if volume.dtype != torch.float32:
        raise RuntimeError("Input must be float32")
    if not h > 0:
        raise RuntimeError("h must be positive")


def nlm_denoise_2d(
    volume: torch.Tensor, h: float, patch_half: int, search_dist: int
) -> torch.Tensor:
    """2D NLM; same contract as the extension's ``nlm_denoise_2d``."""
    _check_input(volume, 2, h)
    device = volume.device
    with torch.cuda.device(device):
        kernel = _kernel(2, patch_half, search_dist, device.index)
        height, width = volume.shape
        pad = search_dist + patch_half
        padded = F.pad(volume[None, None], (pad, pad, pad, pad), mode="reflect")[0, 0]
        padded = padded.contiguous()
        output = torch.empty((height, width), dtype=torch.float32, device=device)
        ps = 2 * patch_half + 1
        exp_scale = -1.0 / (float(ps * ps) * h * h)
        grid = (
            (width + TILE_2D_X - 1) // TILE_2D_X,
            (height + TILE_2D_Y - 1) // TILE_2D_Y,
            1,
        )
        kernel.launch(
            grid,
            (TILE_2D_X, TILE_2D_Y, 1),
            [
                ctypes.c_void_p(padded.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_int(height),
                ctypes.c_int(width),
                ctypes.c_int(padded.shape[0]),
                ctypes.c_int(padded.shape[1]),
                ctypes.c_int(pad),
                ctypes.c_float(exp_scale),
            ],
            torch.cuda.current_stream(device).cuda_stream,
        )
    return output


def shared_bytes_3d(patch_half: int, search_dist: int) -> int:
    """Dynamic shared memory one 3D block needs (the halo-padded tile)."""
    halo = search_dist + patch_half
    return (TILE_3D_Z + 2 * halo) * (TILE_3D_Y + 2 * halo) * (TILE_3D_X + 2 * halo) * 4


def nlm_denoise_3d(
    volume: torch.Tensor, h: float, patch_half: int, search_dist: int
) -> torch.Tensor:
    """3D NLM; same contract as the extension's ``nlm_denoise_3d``."""
    _check_input(volume, 3, h)
    device = volume.device
    with torch.cuda.device(device):
        kernel = _kernel(3, patch_half, search_dist, device.index)
        vol = volume.contiguous()
        depth, height, width = vol.shape
        output = torch.empty((depth, height, width), dtype=torch.float32, device=device)
        ps = 2 * patch_half + 1
        exp_scale = -1.0 / (float(ps * ps * ps) * h * h)
        grid = (
            (width + TILE_3D_X - 1) // TILE_3D_X,
            (height + TILE_3D_Y - 1) // TILE_3D_Y,
            (depth + TILE_3D_Z - 1) // TILE_3D_Z,
        )
        kernel.launch(
            grid,
            (TILE_3D_X, TILE_3D_Y, TILE_3D_Z),
            [
                ctypes.c_void_p(vol.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_int(depth),
                ctypes.c_int(height),
                ctypes.c_int(width),
                ctypes.c_float(exp_scale),
            ],
            torch.cuda.current_stream(device).cuda_stream,
            shared_bytes=shared_bytes_3d(patch_half, search_dist),
        )
    return output
