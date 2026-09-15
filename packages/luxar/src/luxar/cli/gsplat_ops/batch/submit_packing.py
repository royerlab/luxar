"""Packing heuristics for ``batch-fit submit`` Slurm arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from luxar.gsplats.utils.device import cuda_worker_memory_bytes, gpu_worker_capacity

# A fit holds the volume tensor plus model/optimizer state, approximately twice
# the raw float32 volume. The benchmark OOM ceiling already paid for one CUDA
# context, so its task bytes plus the shared fixed overhead recover the usable
# device-memory budget for packed workers.
_FIT_BYTES_PER_VOXEL = 2 * 4

PackingLimit = Literal[
    "GPU memory",
    "node CPU count",
    "node RAM",
    "packing cap",
    "backfill window",
    "requested count",
    "unknown node capacity",
    "volume ratio",
]


@dataclass(frozen=True)
class PackingResolution:
    """Resolved tasks per Slurm job and the binding constraint."""

    count: int
    limit: PackingLimit
    node_class: Optional[tuple[int, int]] = None


def _gpu_parallel_capacity(
    max_shape: Optional[Sequence[int]], tile_voxels: int, gpus_per_task: int
) -> int:
    """Bound packed fits by the profiled per-GPU OOM ceiling."""
    if not max_shape:
        return 1
    profile_budget = cuda_worker_memory_bytes(math.prod(max_shape))
    task_bytes = max(1, tile_voxels * _FIT_BYTES_PER_VOXEL)
    per_gpu = gpu_worker_capacity(profile_budget, task_bytes).count
    return max(1, per_gpu * max(1, gpus_per_task))


def _node_classes(
    node_resources: Optional[Sequence[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Return distinct scheduler node classes in stable order."""
    if not node_resources:
        return []
    return list(dict.fromkeys(node_resources))


def _node_parallel_limit(
    node_resources: Optional[Sequence[tuple[int, int]]],
    cpus_per_task: int,
    mem_gb_per_task: int,
) -> PackingResolution:
    """Bound packing so the request remains eligible for every node class."""
    node_classes = _node_classes(node_resources)
    if not node_classes:
        return PackingResolution(4, "unknown node capacity")
    candidates: list[PackingResolution] = []
    for cores, memory_mb in node_classes:
        cpu_limit = max(1, cores // max(1, cpus_per_task))
        memory_limit = max(1, memory_mb // max(1, mem_gb_per_task * 1024))
        if cpu_limit <= memory_limit:
            candidates.append(
                PackingResolution(cpu_limit, "node CPU count", (cores, memory_mb))
            )
        else:
            candidates.append(
                PackingResolution(memory_limit, "node RAM", (cores, memory_mb))
            )
    return min(candidates, key=lambda candidate: candidate.count)


def largest_node_class(
    node_resources: Optional[Sequence[tuple[int, int]]],
    cpus_per_task: int,
    mem_gb_per_task: int,
) -> Optional[tuple[int, int]]:
    """Return the node class that can host the most requested workers."""
    node_classes = _node_classes(node_resources)
    if not node_classes:
        return None
    return max(
        node_classes,
        key=lambda node: min(
            node[0] // max(1, cpus_per_task),
            node[1] // max(1, mem_gb_per_task * 1024),
        ),
    )


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
    limit: PackingLimit = "volume ratio"

    if uses_backfill and no_job_limit:
        if est_seconds > 0:
            backfill_limit = max(1, int(300 / est_seconds))
            if backfill_limit < packing:
                packing = backfill_limit
                limit = "backfill window"
        if 3 < packing:
            packing = 3
            limit = "packing cap"
    else:
        if 10 < packing:
            packing = 10
            limit = "packing cap"
    return PackingResolution(max(1, packing), limit)
