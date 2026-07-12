"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import zarr

from luxar.gsplats import GSplatData
from luxar.gsplats.io._archive import extract_compressed_zarr


def load_gsplats(
    path: str | Path,
    include_stats: bool = False,
) -> GSplatData:
    """Load Gaussian splats from .gsplats.zarr format.

    Supports both uncompressed (.gsplats.zarr) and compressed formats
    (.gsplats.zarr.zip, .gsplats.zarr.tar.gz). Compressed archives are
    automatically extracted to a temporary directory.

    Arrays are automatically decoded from their stored encoding (quantization,
    broadcasting, etc.) to float32.

    Only **matrix-shaped** node trees map to a ``GSplatData`` — a leaf, or a
    ``kind=lod`` group whose children are all leaves (the substitutive × additive
    matrix). A genuinely nested tree (a ``kind=partition`` root, or a lod group
    with non-leaf children) has no flat ``GSplatData`` equivalent and raises
    ``ValueError``; consume those via the node tree directly (``read_gsplat_node``).

    Args:
        path: Path to .gsplats.zarr directory or compressed archive
        include_stats: Whether to include fitting/provenance metadata in stats

    Returns:
        GSplatData with decoded arrays and optional stats

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If the format is invalid/incompatible, or the file is a
            non-matrix (partition/nested) tree.
    """
    node, stats = load_gsplat_node(path, include_stats=include_stats)
    return GSplatData.from_tree(node, stats=stats)


def load_gsplat_node(
    path: str | Path,
    include_stats: bool = False,
) -> "tuple[Any, Dict[str, Any]]":
    """Load the raw v3.0 node-tree (a :class:`~luxar.gsplats.tree.GSplatNode`).

    Unlike :func:`load_gsplats`, this does NOT flatten to a ``GSplatData`` and so
    works for **every** shape — including ``kind=partition`` roots and genuinely
    nested trees that have no flat matrix equivalent. Use this to graft a
    standalone ``.gsplats.zarr`` into a scene, or to inspect a partition/nested
    file. Handles ``.zip`` / ``.tar.gz`` archives transparently.

    Returns:
        ``(node, stats)`` — the tree root and the (optional) root-level stats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats zarr not found: {path}")

    # Handle compressed archives
    temp_dir = None
    zarr_path = path

    compressed_suffixes = (".zip", ".tar.gz")
    is_compressed = any(str(path).endswith(s) for s in compressed_suffixes)
    if is_compressed:
        # Compressed archive - extract to temp
        zarr_path = extract_compressed_zarr(path)
        temp_dir = zarr_path.parent
    elif path.is_file():
        raise ValueError(
            f"Expected a zarr directory or compressed archive (.zip/.tar.gz), "
            f"got regular file: {path}"
        )

    try:
        # Open zarr store
        root = zarr.open_group(str(zarr_path), mode="r")

        # Validate format
        format_type = root.attrs.get("format_type")
        if format_type != "gsplats_zarr":
            raise ValueError(
                f"Invalid format_type: {format_type}, expected 'gsplats_zarr'"
            )

        from luxar.gsplats.io.save_gsplats import SUPPORTED_FORMAT_VERSIONS

        format_version = root.attrs.get("format_version")
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported format_version: {format_version!r}. The on-disk "
                f"format is now v3.2 (a detached node-tree subtree; v3.0 / v3.1 "
                f"are also read). Convert legacy v1.x / v2.0 files (and old "
                f"substitutive directories) with "
                f"`luxar gsplat migrate-format <input> <output.gsplats.zarr>`."
            )

        # Read the node-tree subtree rooted at the file.
        from luxar.io._compiler.gsplat_tree import read_gsplat_node

        node = read_gsplat_node(root, root)

        # Gather root-level stats (fitting / pipeline / provenance / header).
        stats: Dict[str, Any] = {}
        if include_stats:
            if "fitting" in root:
                for key, value in root["fitting"].attrs.items():
                    stats[key] = value
            if "pipeline" in root:
                # Reduction/topology stats (lod_kind, method, coverage_inflation,
                # refine, ...) — split_fitting_info's fourth bucket. setdefault:
                # fitting/header keys keep precedence on any collision.
                for key, value in root["pipeline"].attrs.items():
                    stats.setdefault(key, value)
            if "provenance" in root:
                stats["provenance"] = dict(root["provenance"].attrs)
            stats["format_version"] = format_version
            stats["timestamp"] = root.attrs.get("timestamp")
            stats["luxar_gsplats_version"] = root.attrs.get("luxar_gsplats_version")
            if "description" in root.attrs:
                stats["description"] = root.attrs["description"]

        return node, stats

    finally:
        # Cleanup temporary directory if we extracted a compressed archive
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
