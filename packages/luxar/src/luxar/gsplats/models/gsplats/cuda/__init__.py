"""
CUDA Backend for Gaussian Splatting.

This module provides GPU-accelerated Gaussian splatting for NVIDIA GPUs
using custom CUDA kernels. It supports 2D-8D volumetric rendering with
splat-centric rasterization and optimized gradient computation.

When the CUDA backend is not compiled, it falls back to the PyTorch
reference implementation transparently.
"""

from __future__ import annotations

# Backend availability check
CUDA_AVAILABLE = False
CUDA_BACKEND_AVAILABLE = False

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

# Import CUDA extension when available
# The extension may be built in-place in this directory
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_cuda_dir = Path(__file__).parent
if str(_cuda_dir) not in sys.path:
    sys.path.insert(0, str(_cuda_dir))

try:
    import cuda_splatting_backend  # type: ignore[import-not-found]  # noqa: F401

    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    if CUDA_AVAILABLE:
        import warnings

        from arbol import aprint

        aprint(
            "WARNING: CUDA GPU detected but splatting backend is not compiled. "
            "Custom CUDA kernels unavailable — GPU fitting will use slower PyTorch fallback. "
            "Build with: make build-cuda"
        )
        warnings.warn(
            "CUDA splatting backend not compiled. Custom CUDA kernels unavailable. "
            "GPU fitting will use slower PyTorch fallback. "
            "Build with: make build-cuda",
            UserWarning,
            stacklevel=2,
        )

# Import public API
from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (  # noqa: E402
    CUDASplatFunction,
    GaussianSplatModelCUDA,
    cholesky_to_conic,
)

__all__ = [
    "CUDA_AVAILABLE",
    "CUDA_BACKEND_AVAILABLE",
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
