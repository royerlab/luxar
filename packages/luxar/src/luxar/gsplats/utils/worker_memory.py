"""Shared worker-memory estimates for Gaussian splat launchers.

A fit keeps the volume tensor plus model and optimizer state, approximately
twice the raw float32 volume. Each CUDA worker also pays hundreds of MiB for
its CUDA context and library handles even when its task is tiny.
"""

from __future__ import annotations

from typing import Optional

_CUDA_PROCESS_OVERHEAD_BYTES = 512 * 1024**2


def task_working_set_bytes(
    task_voxels: int,
    dtype_bytes: int = 4,
    safety_factor: float = 2.0,
) -> int:
    """Estimate one fit worker's variable memory footprint."""
    return max(1, int(safety_factor * task_voxels * dtype_bytes))


def cuda_worker_memory_bytes(
    task_voxels: int,
    dtype_bytes: int = 4,
    safety_factor: float = 2.0,
) -> int:
    """Estimate one CUDA fit worker's task data plus process overhead."""
    return (
        task_working_set_bytes(task_voxels, dtype_bytes, safety_factor)
        + _CUDA_PROCESS_OVERHEAD_BYTES
    )


def cuda_worker_count(memory_budget: Optional[int], task_bytes: int) -> int:
    """Return the number of CUDA workers that fit a memory budget."""
    if memory_budget is None:
        return 1
    per_worker = task_bytes + _CUDA_PROCESS_OVERHEAD_BYTES
    return max(1, int(memory_budget // per_worker))
