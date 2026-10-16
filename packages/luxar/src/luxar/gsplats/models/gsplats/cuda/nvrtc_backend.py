"""NVRTC (runtime-compiled) backend for the CUDA splatting kernels.

The same device code as the ahead-of-time extension (``src/kernels_core.cuh``
and the headers it includes), compiled on first use by NVRTC for the GPU in
the machine and cached on disk -- so a plain ``pip install`` gets the fast
kernels with no ``nvcc``, no C++ compiler and no build step. See
``luxar/gsplats/_nvrtc.py`` for the compile/cache/launch machinery.

This module is the Python port of the extension's host glue
(``src/cuda_splatting.cu`` + ``src/bindings.cpp``): input validation, output
and gradient allocation, the device-resident shape tensor, the optional
pre-zeroed output buffer and backward's ``output_to_zero`` side effect, and
launching on torch's current stream. :func:`forward` and :func:`backward`
take the same arguments and return the same tuples as
``cuda_splatting_backend.forward`` / ``.backward``, so the autograd wrapper
cannot tell the two backends apart.

Kernels compile per dimensionality: the first 3D call compiles the four 3D
kernels (forward/backward x FP32/FP16), not all 28 instantiations.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Sequence

import torch

from luxar.gsplats._nvrtc import Kernel, KernelProgram, get_runtime

#: Mirrors MIN_DIM / MAX_SUPPORTED_DIM in src/cuda_splatting.h.
MIN_DIM = 2
MAX_SUPPORTED_DIM = 8

#: Threads per block -- the extension's launchers (src/kernel_launchers.cuh).
BLOCK_SIZE = 256

SRC_DIR = Path(__file__).parent / "src"

PROGRAM = KernelProgram(
    label="gsplat-splatting",
    # Angle brackets: resolve ONLY through -I<SRC_DIR>. A quoted top-level include
    # would first search the program's own directory, i.e. the CWD, so a stray
    # kernels_core.cuh there could be compiled (and cached under the real key).
    source="#include <kernels_core.cuh>\n",
    include_dir=SRC_DIR,
)

_C_TYPE = {torch.float32: "float", torch.float16: "__half"}


def kernel_names(dim: int) -> list[str]:
    """NVRTC name expressions for every kernel instantiation of ``dim``."""
    return [
        f"rasterize_{direction}_splat_centric_kernel<{dim}, {ctype}>"
        for direction in ("forward", "backward")
        for ctype in ("float", "__half")
    ]


def _kernels(dim: int, device_index: int) -> dict[str, Kernel]:
    """The loaded ``dim``-D kernels for a device, compiling on first use."""
    return get_runtime().kernels(PROGRAM, kernel_names(dim), device_index)


def prepare(dim: int, device: torch.device | int | str | None = None) -> None:
    """Compile (or load from cache) the ``dim``-D kernels for ``device`` now.

    Lets a caller pay the one-time compile up front -- and fall back cleanly
    if it fails -- instead of inside the first training step.
    """
    if not MIN_DIM <= dim <= MAX_SUPPORTED_DIM:
        raise RuntimeError(
            f"Dimension must be between {MIN_DIM} and {MAX_SUPPORTED_DIM}, got {dim}"
        )
    _kernels(dim, _device_index(device))


def _device_index(device: torch.device | int | str | None) -> int:
    """Resolve a device spec to a CUDA ordinal (None = current device)."""
    if device is None:
        return torch.cuda.current_device()
    dev = (
        torch.device(device)
        if not isinstance(device, int)
        else torch.device("cuda", device)
    )
    if dev.type != "cuda":
        raise RuntimeError(f"NVRTC splatting kernels need a CUDA device, got {dev}")
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _check(condition: bool, message: str) -> None:
    """Raise ``RuntimeError(message)`` unless ``condition`` (TORCH_CHECK semantics)."""
    # TORCH_CHECK raises RuntimeError; keep the same contract.
    if not condition:
        raise RuntimeError(message)


def validate_inputs(
    centers: torch.Tensor,
    conic: torch.Tensor,
    amps: torch.Tensor,
    shape: Sequence[int],
    expected_dtype: torch.dtype,
) -> None:
    """Port of ``validate_inputs`` in src/cuda_splatting.cu (same messages)."""
    _check(centers.is_cuda, "centers must be on CUDA device")
    _check(conic.is_cuda, "conic must be on CUDA device")
    _check(amps.is_cuda, "amps must be on CUDA device")
    _check(
        conic.device == centers.device and amps.device == centers.device,
        "centers, conic, and amps must be on the same CUDA device",
    )
    dim = len(shape)
    _check(
        MIN_DIM <= dim <= MAX_SUPPORTED_DIM,
        f"Dimension must be between {MIN_DIM} and {MAX_SUPPORTED_DIM}, got {dim}",
    )
    _check(
        all(int(s) > 0 for s in shape),
        f"shape entries must be positive, got {list(shape)}",
    )
    _check(
        centers.dim() == 2 and centers.size(1) == dim,
        f"centers must have shape (N, {dim})",
    )
    n = centers.size(0)
    conic_size = dim * (dim + 1) // 2
    _check(
        conic.dim() == 2 and conic.size(0) == n and conic.size(1) == conic_size,
        f"conic must have shape (N, {conic_size})",
    )
    _check(amps.dim() == 1 and amps.size(0) == n, f"amps must have shape ({n},)")
    name = "float16" if expected_dtype == torch.float16 else "float32"
    _check(centers.dtype == expected_dtype, f"centers must be {name}")
    _check(conic.dtype == expected_dtype, f"conic must be {name}")
    _check(amps.dtype == expected_dtype, f"amps must be {name}")
    _check(centers.is_contiguous(), "centers must be contiguous")
    _check(conic.is_contiguous(), "conic must be contiguous")
    _check(amps.is_contiguous(), "amps must be contiguous")
    _check(n < 2**31, f"at most 2**31 - 1 splats per launch, got {n}")


def _to_half(t: torch.Tensor) -> torch.Tensor:
    """``t`` as float16 (no copy when it already is)."""
    return t if t.dtype == torch.float16 else t.to(torch.float16)


def _ptr(t: torch.Tensor | None) -> ctypes.c_void_p:
    """A kernel pointer argument for ``t`` (NULL for None)."""
    return ctypes.c_void_p(t.data_ptr() if t is not None else None)


def _shape_tensor(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    """The volume shape as a device-resident int32 tensor (the kernels read it)."""
    return torch.tensor([int(s) for s in shape], dtype=torch.int32, device=device)


def _check_float_buffer(
    t: torch.Tensor, numel: int, device: torch.device, what: str
) -> None:
    """Validate a flat float32 volume-sized buffer handed to a kernel."""
    _check(
        t.is_cuda and t.device == device,
        f"{what} and inputs must be on the same CUDA device",
    )
    _check(t.dtype == torch.float32, f"{what} must be float32")
    _check(t.is_contiguous(), f"{what} must be contiguous")
    _check(
        t.numel() == numel,
        f"{what} must have prod(shape) = {numel} elements, got {t.numel()}",
    )


def forward(
    centers: torch.Tensor,
    conic: torch.Tensor,
    amps: torch.Tensor,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float,
    use_fp16: bool = False,
    output_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render Gaussians to a flattened FP32 volume.

    Same contract as ``cuda_splatting_backend.forward``: returns
    ``(output (prod(shape),) float32, shape_tensor (d,) int32)``; the shape
    tensor is meant to be handed back to :func:`backward`. A float32 CUDA
    ``output_buffer`` of the right size is reused (it must already be zero).
    """
    if use_fp16:
        centers, conic, amps = _to_half(centers), _to_half(conic), _to_half(amps)
    dtype = torch.float16 if use_fp16 else torch.float32
    validate_inputs(centers, conic, amps, shape, dtype)
    device = centers.device
    dim = len(shape)
    num_pixels = math.prod(int(s) for s in shape)

    with torch.cuda.device(device):
        if (
            output_buffer is not None
            and output_buffer.numel() == num_pixels
            and output_buffer.dtype == torch.float32
            and output_buffer.is_cuda
            and output_buffer.device == device
            and output_buffer.is_contiguous()
        ):
            output = output_buffer
        else:
            output = torch.zeros(num_pixels, dtype=torch.float32, device=device)
        shape_tensor = _shape_tensor(shape, device)
        n = centers.size(0)
        if n > 0:
            kernel = _kernels(dim, device.index)[
                f"rasterize_forward_splat_centric_kernel<{dim}, {_C_TYPE[dtype]}>"
            ]
            kernel.launch(
                (n, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [
                    _ptr(centers),
                    _ptr(conic),
                    _ptr(amps),
                    ctypes.c_int(n),
                    _ptr(shape_tensor),
                    ctypes.c_float(truncate),
                    ctypes.c_float(intensity_floor),
                    _ptr(output),
                ],
                torch.cuda.current_stream(device).cuda_stream,
            )
    return output, shape_tensor


def backward(
    grad_output: torch.Tensor,
    centers: torch.Tensor,
    conic: torch.Tensor,
    amps: torch.Tensor,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float,
    use_fp16: bool = False,
    shape_tensor_cached: torch.Tensor | None = None,
    output_to_zero: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients ``(d_centers, d_conic, d_amps)``, always FP32.

    Same contract as ``cuda_splatting_backend.backward``. ``output_to_zero``
    (a forward output) is zeroed by the kernel as a side effect, so it can be
    passed back to the next :func:`forward` as ``output_buffer``.
    """
    if use_fp16:
        centers, conic, amps = _to_half(centers), _to_half(conic), _to_half(amps)
    dtype = torch.float16 if use_fp16 else torch.float32
    validate_inputs(centers, conic, amps, shape, dtype)
    device = centers.device
    _check(grad_output.is_cuda, "grad_output must be on CUDA device")
    _check(
        grad_output.device == device,
        "grad_output and inputs must be on the same CUDA device",
    )
    dim = len(shape)
    num_pixels = math.prod(int(s) for s in shape)
    _check_float_buffer(grad_output, num_pixels, device, "grad_output")
    if output_to_zero is not None:
        _check_float_buffer(output_to_zero, num_pixels, device, "output_to_zero")

    n = centers.size(0)
    conic_size = dim * (dim + 1) // 2
    with torch.cuda.device(device):
        if shape_tensor_cached is not None:
            _check(
                shape_tensor_cached.is_cuda
                and shape_tensor_cached.device == device
                and shape_tensor_cached.dtype == torch.int32
                and shape_tensor_cached.numel() == dim,
                f"shape_tensor_cached must be a ({dim},) int32 tensor on {device}",
            )
            shape_tensor = shape_tensor_cached
        else:
            shape_tensor = _shape_tensor(shape, device)
        d_centers = torch.zeros((n, dim), dtype=torch.float32, device=device)
        d_conic = torch.zeros((n, conic_size), dtype=torch.float32, device=device)
        d_amps = torch.zeros(n, dtype=torch.float32, device=device)
        if n > 0:
            kernel = _kernels(dim, device.index)[
                f"rasterize_backward_splat_centric_kernel<{dim}, {_C_TYPE[dtype]}>"
            ]
            kernel.launch(
                (n, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [
                    _ptr(grad_output),
                    _ptr(centers),
                    _ptr(conic),
                    _ptr(amps),
                    ctypes.c_int(n),
                    _ptr(shape_tensor),
                    ctypes.c_float(truncate),
                    ctypes.c_float(intensity_floor),
                    _ptr(d_centers),
                    _ptr(d_conic),
                    _ptr(d_amps),
                    _ptr(output_to_zero),
                ],
                torch.cuda.current_stream(device).cuda_stream,
            )
    return d_centers, d_conic, d_amps
