"""The ``luxar info`` command — inspect a Zarr scene's hierarchy and stats.

Registered onto the root Typer app by :func:`register_info_command` (mirroring
the ``cli/gsplat_ops/*::register_*_commands`` pattern), so ``cli/main.py`` stays
a thin entry point. ``_print_tree`` / ``_dfs`` are module-level helpers;
``_dfs`` is re-exported from ``cli/main.py`` for the existing test import.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Optional, Tuple

import typer
import zarr
from arbol import aprint

from ..typing_utils._format_contract import GEOMETRY_TYPES
from .utils import (
    format_memory_size,
    format_tree_node,
    get_zarr_info,
)


def register_info_command(app: typer.Typer) -> None:
    """Attach the ``info`` command to ``app``."""

    @app.command()
    def info(
        path: Path,
        tree: bool = typer.Option(True, "--tree/--no-tree", help="Show tree view"),
        stats: bool = typer.Option(
            False, "--stats", "-s", help="Show detailed statistics"
        ),
        depth: Optional[int] = typer.Option(
            None, "--depth", "-d", help="Max tree depth"
        ),
        format: str = typer.Option(
            "text", "--format", help="Output format (text/json)"
        ),
    ) -> None:
        """Show detailed information about a Zarr scene.

        Args:
            path (Path): Path to the Zarr store.
            tree (bool, optional): Show tree view. Defaults to True.
            stats (bool, optional): Show detailed statistics. Defaults to False.
            depth (int, optional): Max tree depth. Defaults to None (unlimited).
            format (str, optional): Output format, "text" or "json". Defaults to "text".
        """
        if format not in ("text", "json"):
            aprint(f"❌ Unknown format: {format}. Use 'text' or 'json'.")
            raise typer.Exit(1)

        try:
            if not path.exists():
                aprint(f"❌ Error: path does not exist: {path}")
                raise typer.Exit(1)

            # Get zarr info
            info_dict = get_zarr_info(path, detailed=stats)

            if format == "json":
                import json

                # Use print() not aprint() to avoid ANSI color codes in JSON output
                print(json.dumps(info_dict, indent=2))
                return

            # Text format output
            root = zarr.open_group(path, mode="r")

            # Header
            aprint(f"\n📁 Zarr Store: {path}")
            aprint(f"💾 Size: {format_memory_size(info_dict['size'])}")
            aprint("")

            # Root attributes
            if root.attrs:
                aprint("🎯 Root Attributes:")
                for k, v in root.attrs.items():
                    if k == "scene_dimensions":
                        # Special formatting for dimensions
                        aprint(f"  {k}:")
                        if isinstance(v, dict) and "dimensions" in v:
                            for dim in v["dimensions"]:
                                aprint(
                                    f"    - {dim.get('name', '?')}: {dim.get('unit', '?')} [display: {dim.get('display', False)}]"
                                )
                    elif isinstance(v, (dict, list)) and len(str(v)) > 80:
                        aprint(f"  {k}: <{type(v).__name__} with {len(v)} items>")
                    else:
                        aprint(f"  {k}: {v}")
                aprint("")

            # Tree view
            if tree:
                aprint("🌳 Scene Hierarchy:")
                _print_tree(root, max_depth=depth, show_stats=stats)
                aprint("")

            # Statistics
            aprint("📊 Summary Statistics:")
            aprint(f"  🗂️  Groups: {info_dict['n_groups']}")
            aprint(f"  📦 Arrays: {info_dict['n_arrays']}")
            if info_dict["points_objects"]:
                aprint(f"  ⭕ Points objects: {len(info_dict['points_objects'])}")
                aprint(f"  ✨ Total points: {info_dict['n_points_total']:,}")
            if info_dict["lines_objects"]:
                aprint(f"  📏 Lines objects: {len(info_dict['lines_objects'])}")
                aprint(f"  ✨ Total vertices: {info_dict['n_lines_vertices_total']:,}")
            if info_dict["gsplats_objects"]:
                aprint(f"  💠 GSplats objects: {len(info_dict['gsplats_objects'])}")
                aprint(f"  ✨ Total splats: {info_dict['n_gsplats_total']:,}")

            if stats:
                if info_dict["points_objects"]:
                    aprint("\n📦 Points Objects Details:")
                    for pc in info_dict["points_objects"]:
                        aprint(f"  {pc['path']}:")
                        aprint(f"    Points: {pc['n_points']:,}")
                        aprint(f"    Dimensions: {pc['n_dims']}")
                        aprint(f"    Has colors: {pc['has_colors']}")
                        aprint(f"    Has radii: {pc['has_radii']}")
                        aprint(f"    Has sharpness: {pc['has_sharpness']}")
                if info_dict["lines_objects"]:
                    aprint("\n📏 Lines Objects Details:")
                    for lo in info_dict["lines_objects"]:
                        aprint(f"  {lo['path']}:")
                        aprint(f"    Vertices: {lo['n_vertices']:,}")
                        aprint(f"    Dimensions: {lo['n_dims']}")
                        aprint(f"    Has colors: {lo['has_colors']}")
                        aprint(f"    Has widths: {lo['has_widths']}")
                if info_dict["gsplats_objects"]:
                    aprint("\n💠 GSplats Objects Details:")
                    for gs in info_dict["gsplats_objects"]:
                        aprint(f"  {gs['path']}:")
                        aprint(f"    Splats: {gs['n_splats']:,}")
                        aprint(f"    Dimensions: {gs['n_dims']}")
                        aprint(f"    Has colors: {gs['has_colors']}")
        except typer.Exit:
            raise
        except Exception as e:
            aprint(f"❌ Error reading info for {path}: {e}")
            raise typer.Exit(1)


def _print_tree(
    group: zarr.Group,
    depth: int = 0,
    max_depth: Optional[int] = None,
    prefix: str = "",
    is_last: bool = True,
    show_stats: bool = False,
) -> None:
    """Print a tree view of the zarr hierarchy."""
    if max_depth is not None and depth > max_depth:
        return

    # Determine node type from zarr attrs (set by compiler)
    stored_type = group.attrs.get("type", "")
    if depth == 0:
        node_type = "scene"
    # Contract vocabulary, not a literal tuple: a geometry type added to
    # ``geometry_types`` but missed here would be reported as a plain "group".
    elif stored_type in GEOMETRY_TYPES:
        node_type = stored_type
    else:
        node_type = "group"
    attrs = {}

    if node_type == "points" and "positions" in group:
        positions = group["positions"]
        attrs["n_points"] = positions.shape[0]
        if show_stats:
            attrs["shape"] = positions.shape
            attrs["dtype"] = str(positions.dtype)
    elif node_type == "lines" and "vertices" in group:
        vertices = group["vertices"]
        attrs["n_vertices"] = vertices.shape[0]
        if show_stats:
            attrs["shape"] = vertices.shape
            attrs["dtype"] = str(vertices.dtype)
    elif node_type == "gsplats" and "centers" in group:
        centers = group["centers"]
        attrs["n_splats"] = centers.shape[0]
        if show_stats:
            attrs["shape"] = centers.shape
            attrs["dtype"] = str(centers.dtype)

    # Print node
    if depth == 0:
        aprint(format_tree_node("/", depth, is_last, prefix, node_type, attrs))
    else:
        name = group.basename or "?"
        aprint(format_tree_node(name, depth, is_last, prefix, node_type, attrs))

    # Update prefix for children
    if depth > 0:
        if is_last:
            new_prefix = prefix + "    "
        else:
            new_prefix = prefix + "│   "
    else:
        new_prefix = ""

    # Get children
    subgroups = list(group.group_keys())

    # Print children
    for i, subgroup_name in enumerate(subgroups):
        is_last_child = i == len(subgroups) - 1
        subgroup = group[subgroup_name]
        _print_tree(
            subgroup, depth + 1, max_depth, new_prefix, is_last_child, show_stats
        )


def _dfs(
    group: zarr.Group, depth: int = 0
) -> Generator[Tuple[int, zarr.Group], None, None]:
    """Depth-first walk that yields (depth, group) for the given group and
    every nested subgroup.

    Args:
        group (zarr.Group): Zarr group to traverse.
        depth (int, optional): Starting depth for the root group. Defaults to 0.

    Yields:
        Tuple[int, zarr.Group]: (depth, group) for the input group and each
        descendant subgroup.
    """
    try:
        yield depth, group
        for name in group.group_keys():
            yield from _dfs(group[name], depth + 1)
    except Exception as e:
        aprint(f"Error traversing Zarr group hierarchy: {e}")
        raise
