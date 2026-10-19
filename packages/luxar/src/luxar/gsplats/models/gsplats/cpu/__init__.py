"""Native CPU Gaussian splatting: fused numba kernels for 3D fits on the CPU.

The CPU sibling of the Metal and CUDA backends: a splat-centric forward plus an
analytic backward, compiled at runtime by numba from pip wheels (no C++
toolchain), parallel without atomics, and numerically the PyTorch reference
renderer. See :mod:`.backend` for ``LUXAR_CPU_BACKEND`` / ``LUXAR_CPU_THREADS``.

Importing this package imports neither numba nor the kernels; they load when a
CPU fit first selects them.
"""

from __future__ import annotations

from luxar.gsplats.utils.device import physical_core_count

from .backend import (
    BACKEND_ENV,
    THREADS_ENV,
    CpuNativeStatus,
    cpu_native_unavailable_reason,
    get_cpu_kernels,
    prepare_cpu_native,
    requested_backend,
    resolve_thread_count,
)
from .gsplat_model_cpu import (
    CpuSplatFunction,
    GaussianSplatModelCPUNative,
    cpu_render_backward_numpy,
    cpu_render_numpy,
    render_gaussians_cpu_native,
)

__all__ = [
    "BACKEND_ENV",
    "THREADS_ENV",
    "CpuNativeStatus",
    "CpuSplatFunction",
    "GaussianSplatModelCPUNative",
    "cpu_native_unavailable_reason",
    "cpu_render_backward_numpy",
    "cpu_render_numpy",
    "get_cpu_kernels",
    "physical_core_count",
    "prepare_cpu_native",
    "render_gaussians_cpu_native",
    "requested_backend",
    "resolve_thread_count",
]
