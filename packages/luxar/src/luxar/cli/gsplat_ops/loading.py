"""Shared loading policy for gsplat CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from arbol import aprint

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import is_matrix_shaped


def load_matrix_gsplats(
    path: Path,
    *,
    include_stats: bool,
    command: str,
) -> GSplatData:
    """Load a matrix-shaped dataset or give the shared flatten-first guidance."""
    node, stats = load_gsplat_node(path, include_stats=include_stats)
    if not is_matrix_shaped(node):
        aprint(
            f"❌ Error: {path.name}: 'luxar gsplat {command}' needs a flat "
            "(matrix-shaped) store and cannot preserve partition/nested-tree "
            f"topology. Collapse it first with 'luxar gsplat flatten {path.name} "
            "flat.gsplats.zarr'."
        )
        raise typer.Exit(1)
    return GSplatData.from_tree(node, stats=stats)
