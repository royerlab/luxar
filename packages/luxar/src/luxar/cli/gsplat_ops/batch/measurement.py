"""Measurement helpers for batch tile stores."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from luxar._zarr_compat import read_array_meta
from luxar.cli.gsplat_ops.recipe_shared import measure_store_bytes


def measure_tiles_bytes_per_splat(
    tiles_dir: Path, tile_names: List[str]
) -> Tuple[Optional[float], int]:
    """Measure on-wire bytes/splat from completed tile stores.

    Returns ``(bytes_per_splat, n_tiles_measured)``; ``(None, 0)`` when nothing
    usable could be measured.
    """
    total_bytes = 0
    total_splats = 0
    n_measured = 0
    for name in tile_names:
        tile = tiles_dir / name
        if not tile.is_dir():
            continue
        tile_bytes = measure_store_bytes(tile)
        tile_splats = 0
        # Glob the array DIRECTORY and read its metadata through the facade:
        # only format 2 has a `.zarray`, so globbing that name measured zero
        # splats in every format-3 tile — which is not an error here, it just
        # skips the tile and silently drops the caller back to the analytic
        # bytes/splat estimate. Both formats spell `shape` the same way.
        for node in tile.rglob("centers"):
            if not node.is_dir():
                continue
            meta = read_array_meta(node)
            if meta is None:
                continue
            try:
                tile_splats += int(meta["shape"][0])
            except (ValueError, KeyError, IndexError, TypeError):
                continue
        if tile_bytes > 0 and tile_splats > 0:
            total_bytes += tile_bytes
            total_splats += tile_splats
            n_measured += 1
    if total_bytes <= 0 or total_splats <= 0:
        return None, 0
    return total_bytes / total_splats, n_measured
