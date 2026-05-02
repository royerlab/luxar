"""PyTorch device-selection helpers for Gaussian splat code."""

from __future__ import annotations

from typing import Optional

import torch


def is_mps_available() -> bool:
    """Return True when PyTorch's MPS backend is available.

    Some older or CPU-only PyTorch builds do not expose ``torch.backends.mps``.
    Centralizing the check keeps device auto-detection robust across builds.
    """
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_torch_device(
    device: Optional[str | torch.device] = None,
    *,
    use_cuda: bool = True,
    use_metal: bool = True,
) -> torch.device:
    """Resolve an explicit or auto-selected PyTorch device.

    Explicit ``device`` values always win. When ``device`` is ``None``, CUDA is
    preferred over MPS/Metal, and both accelerator classes honor their
    corresponding opt-in flags before falling back to CPU.
    """
    if device is not None:
        return torch.device(device)

    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if use_metal and is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")
