"""Wall-time estimation for batch fitting jobs from GPU profile data."""

from __future__ import annotations

import math
from typing import Any, Dict, List

# Safety margin applied to time estimates
SAFETY_FACTOR = 1.5

# Overhead fractions
SEEDING_OVERHEAD_FRAC = 0.10  # 10% of training time
IO_OVERHEAD_SECONDS = 5.0  # Flat I/O cost per tile


def estimate_tile_wall_seconds(
    tile_voxels: int,
    n_iters: int,
    throughput_table: List[Dict[str, Any]],
) -> float:
    """Estimate wall time in seconds for fitting a single tile.

    Algorithm:
        1. Find bracketing entries in the throughput table (by voxel count).
        2. Log-interpolate ``amp_train_ms`` (or ``fp32_train_ms``) for the
           tile's voxel count.
        3. ``total = ms_per_iter * n_iters / 1000``
        4. Add 10% seeding overhead + 5s I/O flat cost.
        5. Multiply by 1.5x safety margin.

    Args:
        tile_voxels: Number of voxels in the tile.
        n_iters: Optimization iterations (from preset/config).
        throughput_table: Averaged 3D throughput table from the GPU profile
            (list of dicts with ``voxels``, ``amp_train_ms``/``fp32_train_ms``).

    Returns:
        Estimated wall time in seconds (with safety margin).
    """
    ms_per_iter = _interpolate_train_ms(tile_voxels, throughput_table)

    # Total training time
    training_seconds = ms_per_iter * n_iters / 1000.0

    # Add overheads
    total = training_seconds * (1 + SEEDING_OVERHEAD_FRAC) + IO_OVERHEAD_SECONDS

    # Safety margin
    return total * SAFETY_FACTOR


def estimate_slurm_time_limit(max_tile_seconds: float) -> str:
    """Convert a wall-time estimate (seconds) to a Slurm ``--time`` string.

    Rounds up to the next 15-minute boundary.

    Returns:
        Time string in ``HH:MM:SS`` format.
    """
    # Round up to next 15-minute boundary
    minutes = math.ceil(max_tile_seconds / 60.0)
    minutes = math.ceil(minutes / 15) * 15

    # Enforce minimum of 15 minutes
    minutes = max(minutes, 15)

    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:00"


def _interpolate_train_ms(
    tile_voxels: int,
    throughput_table: List[Dict[str, Any]],
) -> float:
    """Log-interpolate ms per training iteration for a given voxel count.

    Falls back to linear extrapolation with a 2x factor if tile_voxels
    exceeds the profile's range.
    """
    # Filter to non-OOM entries with train_ms data, sorted by voxels
    entries = []
    for e in throughput_table:
        if e.get("oom"):
            continue
        ms = e.get("amp_train_ms") or e.get("fp32_train_ms")
        if ms is not None and ms > 0:
            entries.append((e["voxels"], ms))
    entries.sort(key=lambda x: x[0])

    if not entries:
        # No data — return a conservative default (50ms per iter)
        return 50.0

    if len(entries) == 1:
        # Single data point — scale linearly
        v0, ms0 = entries[0]
        if v0 > 0:
            return float(ms0 * (tile_voxels / v0))
        return float(ms0)

    # Check bounds
    v_min, ms_min = entries[0]
    v_max, ms_max = entries[-1]

    if tile_voxels <= v_min:
        return float(ms_min)
    if tile_voxels >= v_max:
        # Extrapolate with 2x safety
        ratio = tile_voxels / v_max
        return float(ms_max * ratio * 2.0)

    # Find bracketing pair
    for i in range(len(entries) - 1):
        v_lo, ms_lo = entries[i]
        v_hi, ms_hi = entries[i + 1]
        if v_lo <= tile_voxels <= v_hi:
            # Log-log interpolation
            if v_lo > 0 and v_hi > 0 and ms_lo > 0 and ms_hi > 0 and v_lo != v_hi:
                log_v = math.log(tile_voxels)
                log_v_lo = math.log(v_lo)
                log_v_hi = math.log(v_hi)
                log_ms_lo = math.log(ms_lo)
                log_ms_hi = math.log(ms_hi)
                frac = (log_v - log_v_lo) / (log_v_hi - log_v_lo)
                log_ms = log_ms_lo + frac * (log_ms_hi - log_ms_lo)
                return float(math.exp(log_ms))
            else:
                # Linear fallback
                frac = (tile_voxels - v_lo) / (v_hi - v_lo) if v_hi != v_lo else 0
                return float(ms_lo + frac * (ms_hi - ms_lo))

    # Should not reach here, but just in case
    return float(ms_max)
