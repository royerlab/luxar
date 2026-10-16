"""
CUDA Backend for Gaussian Splatting.

This module provides GPU-accelerated Gaussian splatting for NVIDIA GPUs
using custom CUDA kernels. It supports 2D-8D volumetric rendering with
splat-centric rasterization and optimized gradient computation.

The kernels reach the GPU one of two ways (see :mod:`.backend`): the
ahead-of-time extension built by ``make build-cuda``, or -- with no build
step at all -- runtime compilation through NVRTC on first use, cached on
disk. When neither is usable, fitting falls back to the PyTorch reference
implementation. ``LUXAR_CUDA_BACKEND=aot|nvrtc|torch`` forces a choice.
"""

from __future__ import annotations

# Backend availability check
CUDA_AVAILABLE = False

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

from luxar.gsplats.models.gsplats.cuda.backend import (  # noqa: E402
    AOT_BACKEND_AVAILABLE,
    SplattingBackend,
    get_splatting_backend,
    splatting_backend_unavailable_reason,
)

# Import public API
from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (  # noqa: E402
    CUDASplatFunction,
    GaussianSplatModelCUDA,
    cholesky_to_conic,
)

__all__ = [
    "AOT_BACKEND_AVAILABLE",
    "CUDA_AVAILABLE",
    "SplattingBackend",
    "get_splatting_backend",
    "splatting_backend_unavailable_reason",
    "GaussianSplatModelCUDA",
    "CUDASplatFunction",
    "cholesky_to_conic",
]


def run_benchmark(verbose: bool = True) -> dict:
    """
    Run the CUDA performance benchmark.

    Compares PyTorch CPU, PyTorch CUDA vanilla, and custom CUDA kernels.

    Args:
        verbose: If True, print results to stdout.

    Returns:
        Dictionary with benchmark results for each configuration.

    Example:
        >>> from luxar.gsplats.models.gsplats.cuda import run_benchmark
        >>> results = run_benchmark()
    """
    from luxar.gsplats.models.gsplats.cuda.benchmark import run_benchmark as _run

    return _run(verbose=verbose)
