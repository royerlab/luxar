"""
NLM CUDA Backend.

Provides GPU-accelerated Non-Local Means denoising via custom CUDA kernels.
When the extension is not compiled, NLM_CUDA_AVAILABLE is False and the
PyTorch fallback is used transparently.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Backend availability check
NLM_CUDA_AVAILABLE = False

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

# Try to import the compiled extension
_cuda_dir = Path(__file__).parent
if str(_cuda_dir) not in sys.path:
    sys.path.insert(0, str(_cuda_dir))

try:
    import nlm_cuda_backend  # type: ignore[import-not-found]  # noqa: F401

    NLM_CUDA_AVAILABLE = True
except ImportError:
    if _CUDA_AVAILABLE:
        import warnings

        warnings.warn(
            "CUDA GPU detected but NLM CUDA backend is not compiled. "
            "NLM denoising will use slower PyTorch fallback. "
            "Build with: make build-nlm-cuda",
            UserWarning,
            stacklevel=2,
        )

__all__ = ["NLM_CUDA_AVAILABLE"]
