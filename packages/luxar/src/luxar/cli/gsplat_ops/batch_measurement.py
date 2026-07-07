"""Measurement helpers for batch tile stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from luxar.cli.lod import measure_store_bytes


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
        for zarray in tile.rglob("centers/.zarray"):
            try:
                tile_splats += int(json.loads(zarray.read_text())["shape"][0])
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                continue
        if tile_bytes > 0 and tile_splats > 0:
            total_bytes += tile_bytes
            total_splats += tile_splats
            n_measured += 1
    if total_bytes <= 0 or total_splats <= 0:
        return None, 0
    return total_bytes / total_splats, n_measured
