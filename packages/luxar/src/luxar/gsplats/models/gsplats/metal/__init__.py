# mypy: warn-unreachable=False
# Reason: this package's availability checks are gated by `sys.platform !=
# "darwin"` early-returns. mypy on Linux narrows sys.platform there and flags
# the macOS-only branches as unreachable. The narrowing is platform-correct
# on Linux and platform-incorrect on macOS, so per-line ignores disagree
# across mypy versions/platforms; disabling the warning at file scope is
# the cleanest expression of "this file is platform-conditional".

"""
Metal-accelerated Gaussian splatting renderer for Apple Silicon.

The 3D splat-centric kernels in ``src/kernels.metal`` run through one of two
interchangeable backends (see :mod:`.backend`):

- ``runtime`` (the default for a pip install): compiled on first use by
  ``torch.mps.compile_shader`` with the Metal compiler built into macOS.
  Needs only an MPS-capable torch >= 2.7 (2.6 via its private API) -- no
  Xcode, no compiled extension.
- ``aot``: the optional ahead-of-time extension, built explicitly by a
  developer with ``python setup.py build_ext --inplace --force`` (needs
  Xcode). Preferred in ``auto`` when present and up to date;
  ``LUXAR_METAL_BACKEND=runtime`` selects the equivalent runtime path.

``LUXAR_METAL_BACKEND=auto|aot|runtime|torch`` overrides the choice. Importing
this package never compiles or builds anything.
"""

from __future__ import annotations

import sys

from .backend import (
    BACKEND_ENV,
    MetalKernelBackend,
    get_metal_backend,
    metal_backend_unavailable_reason,
    requested_backend,
)
from .gsplat_model_metal import GaussianSplatModelMetal


def is_metal_available() -> bool:
    """
    Check if a Metal splatting kernel backend (AOT or runtime) is usable.

    Off macOS this returns False without consulting ``LUXAR_METAL_BACKEND``.
    On macOS it returns True only with an MPS device where either the compiled
    extension imports or the runtime compiler built ``kernels.metal``, and
    False when ``LUXAR_METAL_BACKEND=torch``. There, a forced backend that
    cannot run (``LUXAR_METAL_BACKEND=aot|runtime``) raises ``RuntimeError``
    and an unknown value raises ``ValueError``.

    If False, callers should use the reference PyTorch `GaussianSplatModel`.
    """
    if sys.platform != "darwin":
        return False
    return get_metal_backend() is not None


def get_metal_status() -> str:
    """
    Get a human-readable status message about the Metal backend.

    Returns:
        Status message explaining current Metal backend state
    """
    if sys.platform != "darwin":
        return "Metal backend only available on macOS"
    try:
        backend = get_metal_backend()
    except (RuntimeError, ValueError) as exc:  # forced backend fails / bad value
        return f"Metal backend not available: {exc}"
    if backend is not None:
        return f"Metal backend available ({backend.name} kernels)"
    return f"Metal backend not available: {metal_backend_unavailable_reason()}"


__all__ = [
    "BACKEND_ENV",
    "GaussianSplatModelMetal",
    "MetalKernelBackend",
    "get_metal_backend",
    "get_metal_status",
    "is_metal_available",
    "metal_backend_unavailable_reason",
    "requested_backend",
]
