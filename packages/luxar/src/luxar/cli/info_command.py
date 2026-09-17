"""The ``luxar info`` command — inspect a Zarr scene's hierarchy and stats.

Registered onto the root Typer app by :func:`register_info_command` (mirroring
the ``cli/gsplat_ops/*::register_*_commands`` pattern), so ``cli/main.py`` stays
a thin entry point. ``_print_tree`` / ``_dfs`` are module-level helpers;
``_dfs`` is re-exported from ``cli/main.py`` for the existing test import.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import typer
import zarr
from arbol import aprint

from .._zarr_compat import group_keys, suppress_payload_member_warning
from .._zarr_compat import open_group as zarr_open_group
from ..typing_utils._format_contract import GEOMETRY_TYPES
from ..typing_utils.constants import RESERVED_ROOT_GROUPS
from ._traceback import exit_with_error
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
            root = zarr_open_group(path, mode="r")

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
                _print_chunk_layout(root)
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
                        aprint(f"    Has categorical labels: {gs['has_label_ids']}")
        except typer.Exit:
            raise
        except Exception as e:
            exit_with_error(f"❌ Error reading info for {path}: {e}", e)


#: Mean chunk payload below which a full load is dominated by per-request
#: round trips on an HTTP/1.1 host (six connections, no multiplexing). Distinct
#: from ``MIN_CHUNK_BYTES`` (the optimizer's re-chunk floor): a 20 KB chunk is
#: above the floor and still costs a 30 ms RTT per 20 KB on such a host.
HTTP1_RTT_BOUND_CHUNK_BYTES = 32 * 1024


def _print_chunk_layout(root: zarr.Group) -> None:
    """Report the store's STREAMING shape — the ``luxar optimize`` diagnostic.

    Computed off the same helper the optimizer plans from, so a badly chunked
    store is visible from ``luxar info --stats`` rather than only after hosting
    it and counting round trips. The projected request count is the number of
    chunk files a full load fetches, which is what dominates a cold load over
    object storage (measured: 245 s / 9,390 requests for 38.6 MB).
    """
    from ..io.optimize import plan_optimization, summarize_plan
    from ..typing_utils.constants import MIN_CHUNK_BYTES, TARGET_CHUNK_BYTES

    # ONE walk. The summary and the "try `luxar optimize`" hint both need the
    # same per-array metadata, and opening every array twice on top of the walk
    # `get_zarr_info(detailed=True)` already did is three passes over a store
    # that, in the corpus this diagnostic exists for, holds 606,349 files.
    plan = plan_optimization(root)
    summary = summarize_plan(plan)
    if summary.n_arrays == 0:
        return
    aprint("\n🧩 Chunk Layout:")
    aprint(
        f"  Average chunk: {summary.mean_chunk_bytes / 1024:.1f} KB "
        f"(target {TARGET_CHUNK_BYTES // 1024} KB)"
    )
    aprint(
        f"  Under the {MIN_CHUNK_BYTES // 1024} KB floor: "
        f"{summary.n_arrays_under_floor}/{summary.n_arrays} arrays "
        f"({summary.share_under_floor:.0%})"
    )
    aprint(f"  Projected requests for a full load: {summary.n_chunks:,} chunks")
    if summary.mean_chunk_bytes < HTTP1_RTT_BOUND_CHUNK_BYTES:
        # The request count, not the byte count, is what a cold load pays on an
        # HTTP/1.1 host — and `luxar serve` (uvicorn) IS HTTP/1.1 only, as is
        # any plain static server. Measured on a 1.5 M-point example at
        # 25 Mbps / 30 ms RTT (2026-09 viewer audit): 10.6 s over HTTP/1.1
        # against 4.0 s for the same 1,548 chunks multiplexed over HTTP/2.
        aprint(
            f"  ⚠️  Chunks under {HTTP1_RTT_BOUND_CHUNK_BYTES // 1024} KB are round-trip "
            "bound on HTTP/1.1 hosts (`luxar serve` and plain static servers): "
            "measured 10.6 s vs 4.0 s over HTTP/2 for the same 1.5 M points at "
            "25 Mbps / 30 ms. Re-chunk with `luxar optimize --profile hosting`, "
            "or serve behind an HTTP/2 front (CDN, nginx, Caddy)."
        )
    if summary.mean_chunk_bytes >= MIN_CHUNK_BYTES:
        return
    # Gated on a REAL plan, not on the average alone. A store of ten 1 KB
    # single-chunk arrays is under the floor and yet has nothing to re-chunk;
    # recommending the tool there is advice that does nothing.
    if plan.n_rechunked:
        aprint(
            f"  → `luxar optimize` would cut this to "
            f"{plan.target_n_chunks:,} chunks; try --dry-run."
        )


#: Per leaf type: the array whose row count is the node's size, and its label.
_LEAF_COUNT_ARRAYS: Dict[str, Tuple[str, str]] = {
    "points": ("positions", "n_points"),
    "lines": ("vertices", "n_vertices"),
    "gsplats": ("centers", "n_splats"),
    "mesh": ("vertices", "n_vertices"),
}


def _sound_tree_attrs(group: zarr.Group) -> Dict[str, Any]:
    """A sound node's tree summary: codec, trigger, bus, rows, duration."""
    attrs: Dict[str, Any] = {
        "format": group.attrs.get("format", "?"),
        "trigger": group.attrs.get("trigger", "?"),
        "bus": group.attrs.get("bus", "?"),
    }
    if group.attrs.get("has_positions"):
        attrs["n_positions"] = group.attrs.get("n_positions", 0)
    if "duration_ms" in group.attrs:
        attrs["duration_s"] = round(float(group.attrs["duration_ms"]) / 1000.0, 1)
    return attrs


def _leaf_tree_attrs(
    group: zarr.Group, node_type: str, show_stats: bool
) -> Dict[str, Any]:
    """The per-type summary a tree line shows (counts; shapes/dtypes under --stats)."""
    if node_type == "sound":
        return _sound_tree_attrs(group)
    spec = _LEAF_COUNT_ARRAYS.get(node_type)
    if spec is None or spec[0] not in group:
        return {}
    array = group[spec[0]]
    attrs: Dict[str, Any] = {spec[1]: array.shape[0]}
    # Faces too for a mesh, unlike the other types' single count: a mesh's
    # vertex count says little about its size on its own (a coarse surface and
    # a dense one can share a vertex budget), and the face count is what the
    # render cost tracks.
    if node_type == "mesh" and "faces" in group:
        attrs["n_faces"] = group["faces"].shape[0]
    if show_stats:
        attrs["shape"] = array.shape
        attrs["dtype"] = str(array.dtype)
    return attrs


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
    elif stored_type in GEOMETRY_TYPES or stored_type == "sound":
        # `sound` is a node type but not a geometry type (heard, not drawn), so
        # it is named here rather than reached through the geometry vocabulary.
        node_type = stored_type
    else:
        node_type = "group"
    attrs = _leaf_tree_attrs(group, node_type, show_stats)

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

    # Get children. A reserved root group (a baked `environment` map, a
    # `.gsplats.zarr` bookkeeping bucket) is metadata, not a node, and the tree
    # is a tree of NODES. Payload members (an overlay's image, a sound clip)
    # are not zarr nodes either; zarr 3 warns when it enumerates past them.
    with suppress_payload_member_warning():
        subgroups = [
            name
            for name in group.group_keys()
            if not (depth == 0 and name in RESERVED_ROOT_GROUPS)
        ]

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
        for name in group_keys(group):
            if depth == 0 and name in RESERVED_ROOT_GROUPS:
                continue
            yield from _dfs(group[name], depth + 1)
    except Exception as e:
        aprint(f"Error traversing Zarr group hierarchy: {e}")
        raise
