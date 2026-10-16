"""Selection of the CUDA splatting kernel backend.

Three ways to run the splatting forward/backward on an NVIDIA GPU, tried in
this order:

1. ``aot``   -- the ahead-of-time torch extension (``cuda_splatting_backend``)
   built by ``make build-cuda``. Kept first so dev checkouts and HPC installs
   that built it behave exactly as before.
2. ``nvrtc`` -- the same kernels compiled at runtime by NVRTC on first use and
   cached on disk (:mod:`.nvrtc_backend`). Needs only a CUDA build of torch
   on Linux: no ``nvcc``, no C++ compiler.
3. ``torch`` -- no custom kernels; the pure-PyTorch renderer.

``LUXAR_CUDA_BACKEND=aot|nvrtc|torch`` forces one (``auto``/unset = the order
above). A forced kernel backend that cannot run raises instead of silently
degrading -- forcing exists for testing and benchmarking, where a quiet
fallback would report the wrong backend's numbers.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from arbol import aprint

BACKEND_ENV = "LUXAR_CUDA_BACKEND"
VALID_CHOICES = ("auto", "aot", "nvrtc", "torch")

_CUDA_DIR = Path(__file__).parent


@dataclass(frozen=True)
class SplattingBackend:
    """A usable kernel backend: same ``forward``/``backward`` contract for both."""

    name: str  # "aot" or "nvrtc"
    forward: Callable[..., Any]
    backward: Callable[..., Any]
    prepare: Callable[[int, Any], None]


def _no_prepare(dim: int, device: Any = None) -> None:
    """AOT kernels are already compiled; nothing to do."""


def _import_aot() -> ModuleType | None:
    """The compiled extension, or None when it is absent or does not load."""
    if str(_CUDA_DIR) not in sys.path:
        sys.path.insert(0, str(_CUDA_DIR))
    try:
        import cuda_splatting_backend  # type: ignore[import-not-found]
    except ImportError:
        return None
    module: ModuleType = cuda_splatting_backend
    return module


AOT_MODULE = _import_aot()
#: True when the ahead-of-time extension (``make build-cuda``) is importable.
AOT_BACKEND_AVAILABLE = AOT_MODULE is not None


def requested_backend(environ: dict[str, str] | None = None) -> str:
    """The ``LUXAR_CUDA_BACKEND`` choice, validated (``auto`` when unset)."""
    env = os.environ if environ is None else environ
    choice = env.get(BACKEND_ENV, "").strip().lower() or "auto"
    if choice not in VALID_CHOICES:
        raise ValueError(
            f"{BACKEND_ENV}={env.get(BACKEND_ENV)!r} is not one of {', '.join(VALID_CHOICES)}"
        )
    return choice


def _aot_backend() -> SplattingBackend | None:
    """The compiled extension wrapped as a backend, or None."""
    if AOT_MODULE is None:
        return None
    return SplattingBackend("aot", AOT_MODULE.forward, AOT_MODULE.backward, _no_prepare)


def _nvrtc_backend() -> tuple[SplattingBackend | None, str | None]:
    """The NVRTC backend, or (None, why NVRTC cannot run)."""
    from luxar.gsplats._nvrtc import nvrtc_unavailable_reason

    reason = nvrtc_unavailable_reason()
    if reason is not None:
        return None, reason
    from luxar.gsplats.models.gsplats.cuda import nvrtc_backend

    return (
        SplattingBackend(
            "nvrtc",
            nvrtc_backend.forward,
            nvrtc_backend.backward,
            nvrtc_backend.prepare,
        ),
        None,
    )


def _cuda_available() -> bool:
    """Whether torch can see a CUDA device (False without torch)."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


_resolved: dict[str, SplattingBackend | None] = {}
_reasons: dict[str, str] = {}
_lock = threading.Lock()


def _resolve(choice: str) -> SplattingBackend | None:
    """Resolve one ``LUXAR_CUDA_BACKEND`` choice (uncached; see the module docstring)."""
    if choice == "torch":
        _reasons[choice] = f"{BACKEND_ENV}=torch"
        return None
    if not _cuda_available():
        if choice != "auto":
            raise RuntimeError(
                f"{BACKEND_ENV}={choice} but torch cannot access a CUDA device"
            )
        _reasons[choice] = "no CUDA device"
        return None
    if choice in ("auto", "aot"):
        aot = _aot_backend()
        if aot is not None:
            return aot
        if choice == "aot":
            raise RuntimeError(
                f"{BACKEND_ENV}=aot but the compiled extension cuda_splatting_backend "
                "is not importable (build it with `make build-cuda`)"
            )
    nvrtc, reason = _nvrtc_backend()
    if nvrtc is not None:
        return nvrtc
    if choice == "nvrtc":
        raise RuntimeError(f"{BACKEND_ENV}=nvrtc but NVRTC is unavailable: {reason}")
    _reasons[choice] = f"no compiled extension, and NVRTC is unavailable: {reason}"
    return None


def get_splatting_backend() -> SplattingBackend | None:
    """The CUDA kernel backend to use, or None for the pure-PyTorch path.

    Resolved once per ``LUXAR_CUDA_BACKEND`` value and logged once. Cheap
    after the first call. NVRTC compilation itself is deferred to
    ``backend.prepare(dim, device)`` / the first kernel launch.
    """
    choice = requested_backend()
    with _lock:
        if choice in _resolved:
            return _resolved[choice]
        backend = _resolve(choice)
        _resolved[choice] = backend
    if backend is not None:
        how = {
            "aot": "compiled extension (make build-cuda)",
            "nvrtc": "runtime-compiled with NVRTC, cached on disk",
        }[backend.name]
        aprint(f"CUDA splatting kernels: {backend.name} ({how})")
    elif _cuda_available():
        aprint(
            f"CUDA splatting kernels: none, using PyTorch ({_reasons.get(choice, '')})"
        )
    return backend


def splatting_backend_unavailable_reason() -> str | None:
    """Why :func:`get_splatting_backend` returned None (None when it did not)."""
    if get_splatting_backend() is not None:
        return None
    return _reasons.get(requested_backend(), "unknown")


def reset_for_tests() -> None:
    """Forget resolved backends (tests only)."""
    with _lock:
        _resolved.clear()
        _reasons.clear()
