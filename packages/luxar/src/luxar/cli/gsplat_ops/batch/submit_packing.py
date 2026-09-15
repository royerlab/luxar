"""Packing heuristics for ``batch-fit submit`` Slurm arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

_CUDA_PROCESS_OVERHEAD_BYTES = 512 * 1024**2
_FIT_BYTES_PER_VOXEL = 2 * 4

PackingLimit = Literal[
    "GPU memory",
    "node CPU count",
    "node RAM",
    "packing cap",
    "requested count",
    "unknown node capacity",
]


@dataclass(frozen=True)
class PackingResolution:
    """Resolved tasks per Slurm job and the binding constraint."""

    count: int
    limit: PackingLimit


def _gpu_parallel_capacity(
    max_shape: Optional[Sequence[int]], tile_voxels: int, gpus_per_task: int
) -> int:
    if not max_shape:
        return 1
    profile_budget = (
        math.prod(max_shape) * _FIT_BYTES_PER_VOXEL + _CUDA_PROCESS_OVERHEAD_BYTES
    )
    per_worker = tile_voxels * _FIT_BYTES_PER_VOXEL + _CUDA_PROCESS_OVERHEAD_BYTES
    per_gpu = max(1, profile_budget // max(per_worker, 1))
    return max(1, int(per_gpu) * max(1, gpus_per_task))


def _node_parallel_limit(
    node_resources: Optional[Sequence[tuple[int, int]]],
    cpus_per_task: int,
    mem_gb_per_task: int,
) -> PackingResolution:
    if not node_resources:
        return PackingResolution(4, "unknown node capacity")
    candidates: list[PackingResolution] = []
    for cores, memory_mb in node_resources:
        cpu_limit = max(1, cores // max(1, cpus_per_task))
        memory_limit = max(1, memory_mb // max(1, mem_gb_per_task * 1024))
        if cpu_limit <= memory_limit:
            candidates.append(PackingResolution(cpu_limit, "node CPU count"))
        else:
            candidates.append(PackingResolution(memory_limit, "node RAM"))
    return max(candidates, key=lambda candidate: candidate.count)


def resolve_tasks_per_job(
    tasks_per_job: Optional[int],
    *,
    parallel: bool,
    uses_backfill: bool,
    no_job_limit: bool,
    max_shape: Optional[Sequence[int]],
    tile_voxels: int,
    est_seconds: float,
    gpus_per_task: int,
    cpus_per_task: int,
    mem_gb_per_task: int,
    node_resources: Optional[Sequence[tuple[int, int]]],
) -> PackingResolution:
    """Resolve effective ``tasks_per_job`` using scheduler-aware heuristics."""
    if tasks_per_job is not None:
        return PackingResolution(max(1, tasks_per_job), "requested count")

    if parallel:
        cap = 4 if uses_backfill and no_job_limit else 10
        candidates = [
            _node_parallel_limit(node_resources, cpus_per_task, mem_gb_per_task),
            PackingResolution(
                _gpu_parallel_capacity(max_shape, tile_voxels, gpus_per_task),
                "GPU memory",
            ),
            PackingResolution(cap, "packing cap"),
        ]
        return min(candidates, key=lambda candidate: candidate.count)

    max_safe = math.prod(max_shape) if max_shape else tile_voxels
    packing = max(1, int(max_safe / max(tile_voxels, 1)))

    if uses_backfill and no_job_limit:
        if est_seconds > 0:
            packing = min(packing, max(1, int(300 / est_seconds)))
        packing = min(packing, 3)
    else:
        packing = min(packing, 10)
    return PackingResolution(max(1, packing), "packing cap")
