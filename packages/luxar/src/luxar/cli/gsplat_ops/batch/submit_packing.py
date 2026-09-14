"""Packing heuristics for ``batch-fit submit`` Slurm arrays."""

from __future__ import annotations

import math
from typing import Optional, Sequence


def resolve_tasks_per_job(
    tasks_per_job: Optional[int],
    *,
    parallel: bool,
    uses_backfill: bool,
    no_job_limit: bool,
    max_shape: Optional[Sequence[int]],
    tile_voxels: int,
    est_seconds: float,
) -> int:
    """Resolve effective ``tasks_per_job`` using scheduler-aware heuristics."""
    if tasks_per_job is None:
        max_safe = math.prod(max_shape) if max_shape else tile_voxels

        if parallel:
            # Local worker sizing uses the shared policy in gsplats/utils/device.py;
            # Slurm packing needs allocation-aware limits tracked in #2756.
            # Each concurrent fit holds the volume tensor + model params
            # + optimizer state. ~2× the raw volume is a safe estimate.
            packing = max(1, int(max_safe / max(tile_voxels * 2, 1)))
        else:
            packing = max(1, int(max_safe / max(tile_voxels, 1)))

        # On backfill clusters with no job limit, prefer shorter jobs
        # (more jobs = more backfill opportunities = faster throughput).
        # Cap packing lower so individual jobs stay short.
        if uses_backfill and no_job_limit:
            if parallel:
                # Parallel: already short, keep the memory-based packing.
                packing = min(packing, 4)
            else:
                # Sequential: each extra task adds wall-time.
                # Keep jobs under ~5 min for best backfill scheduling.
                if est_seconds > 0:
                    max_tasks_for_5min = max(1, int(300 / est_seconds))
                    packing = min(packing, max_tasks_for_5min)
                packing = min(packing, 3)
        else:
            packing = min(packing, 10)

        tasks_per_job = packing

    return max(1, tasks_per_job)
