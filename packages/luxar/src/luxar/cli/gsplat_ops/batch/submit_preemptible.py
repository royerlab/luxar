"""Preemptible-partition detection/validation for ``batch-fit submit``."""

from __future__ import annotations

from typing import Optional

from arbol import aprint


def resolve_preemptible_partition(
    *,
    preemptible: bool,
    preemptible_partition_opt: Optional[str],
) -> Optional[str]:
    """Resolve an accessible preemptible partition, or ``None`` if unavailable."""
    if not preemptible:
        return None

    preempt_partition: Optional[str]
    if preemptible_partition_opt:
        preempt_partition = preemptible_partition_opt
    else:
        from luxar.gsplats.batch.env_capture import detect_preemptible_gpu_partition

        preempt_partition = detect_preemptible_gpu_partition()

    if preempt_partition is None:
        aprint(
            "No preemptible GPU partition found on this cluster.\n"
            "  Checked all partitions for: preemptible naming + GPU resources.\n"
            "  Use --preemptible-partition to specify one explicitly.\n"
            "  Continuing with guaranteed partition only."
        )
        return None

    from luxar.gsplats.batch.env_capture import validate_partition_access

    if not validate_partition_access(preempt_partition):
        aprint(
            f"Cannot submit to preemptible partition '{preempt_partition}'.\n"
            f"  Your account may not have access.\n"
            f"  To check: sacctmgr show assoc user=$USER partition={preempt_partition}\n"
            "  Continuing with guaranteed partition only."
        )
        return None

    aprint(f"Preemptible partition: {preempt_partition}")
    return preempt_partition
