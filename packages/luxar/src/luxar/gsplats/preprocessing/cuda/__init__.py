"""
NLM CUDA Backend.

Provides GPU-accelerated Non-Local Means denoising via custom CUDA kernels,
reached one of two ways -- the same order and ``LUXAR_CUDA_BACKEND`` override
as the splatting kernels (``models/gsplats/cuda/backend.py``):

1. ``aot``   -- the extension built by ``make build-nlm-cuda``;
2. ``nvrtc`` -- the same kernels compiled at runtime by NVRTC on first use and
   cached on disk (:mod:`.nlm_nvrtc`); needs only a CUDA build of torch.

When neither is usable, :func:`nlm_cuda_available` is False and the PyTorch
backend is used transparently.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_cuda_dir = Path(__file__).parent
if str(_cuda_dir) not in sys.path:
    sys.path.insert(0, str(_cuda_dir))

_aot_module: ModuleType | None
try:
    import nlm_cuda_backend as _aot_module  # type: ignore[import-not-found, no-redef]
except ImportError:
    _aot_module = None

#: True when the ahead-of-time extension (``make build-nlm-cuda``) is importable.
NLM_AOT_AVAILABLE = _aot_module is not None

_resolved: dict[str, ModuleType | None] = {}


def get_nlm_backend() -> ModuleType | None:
    """Module exposing ``nlm_denoise_2d`` / ``nlm_denoise_3d``, or None.

    Honors ``LUXAR_CUDA_BACKEND`` (``aot``/``nvrtc``/``torch``); a forced
    kernel backend that cannot run raises. NVRTC compilation is deferred to
    the first denoise call.
    """
    from luxar.gsplats.models.gsplats.cuda.backend import requested_backend

    choice = requested_backend()
    if choice not in _resolved:
        _resolved[choice] = _resolve(choice)
    return _resolved[choice]


def _resolve(choice: str) -> ModuleType | None:
    """Resolve one ``LUXAR_CUDA_BACKEND`` choice for NLM (uncached)."""
    import torch

    if choice == "torch":
        return None
    if not torch.cuda.is_available():
        if choice != "auto":
            raise RuntimeError(
                f"LUXAR_CUDA_BACKEND={choice} but no CUDA device is available"
            )
        return None
    if choice in ("auto", "aot") and _aot_module is not None:
        return _aot_module
    if choice == "aot":
        raise RuntimeError(
            "LUXAR_CUDA_BACKEND=aot but nlm_cuda_backend is not compiled "
            "(build it with `make build-nlm-cuda`)"
        )
    from luxar.gsplats._nvrtc import nvrtc_unavailable_reason

    reason = nvrtc_unavailable_reason()
    if reason is None:
        from luxar.gsplats.preprocessing.cuda import nlm_nvrtc

        return nlm_nvrtc
    if choice == "nvrtc":
        raise RuntimeError(
            f"LUXAR_CUDA_BACKEND=nvrtc but NVRTC is unavailable: {reason}"
        )
    return None


def nlm_cuda_available() -> bool:
    """True when a CUDA NLM kernel backend (AOT or NVRTC) can be used."""
    return get_nlm_backend() is not None


__all__ = ["NLM_AOT_AVAILABLE", "get_nlm_backend", "nlm_cuda_available"]
