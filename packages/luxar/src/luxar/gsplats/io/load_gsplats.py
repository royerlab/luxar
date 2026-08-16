"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from luxar._zarr_compat import open_group as zc_open_group
from luxar.gsplats import GSplatData
from luxar.gsplats.io._archive import read_archive_root_attrs, resolve_store_path


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


def read_authored_appearance(path: str | Path) -> Dict[str, Any]:
    """Read the authored appearance attrs off a ``.gsplats.zarr`` ROOT.

    A structure-only rebuild (``gsplat lod`` and friends) constructs fresh nodes
    that know nothing about the input's appearance, so without this the authored
    values are silently dropped and the writer's own defaults take their place —
    ``blending_mode`` vanishes and ``opacity``/``gamma``/``intensity``/
    ``absorption`` snap back to their identity. Feed the result to
    ``write_gsplats_tree(root_attrs=...)`` (or ``GSplatData.save(root_attrs=...)``).

    Works on a ``.gsplats.zarr`` directory and on a ``.gsplats.zarr.zip`` /
    ``.gsplats.zarr.tar.gz`` archive alike — both are first-class inputs to the
    rebuild commands, so appearance must survive both (#1604).

    Not every dropped attr is fixed by this: an authored ``colormap`` still
    reverts to gray, and the 4x4 ``transform`` is deliberately left behind
    because feeding a stored (column-major) matrix back through the writer
    transposes it a second time. Both are documented on the key set below.

    Only keys actually present are returned, so an input that authored nothing
    yields ``{}`` and the writer's defaults apply unchanged. Missing/unreadable
    stores yield ``{}`` rather than raising: this is a best-effort carry-over
    alongside the real load, which reports its own errors.

    See :data:`~luxar.core.group.compositing.AUTHORED_APPEARANCE_ATTRS` for the
    key set and https://github.com/royerlab/luxar/issues/1600 for the invariant.
    """
    from luxar.core.group.compositing import AUTHORED_APPEARANCE_ATTRS

    p = Path(path)
    try:
        if p.is_dir():
            # The facade, not a bare ``zarr.open_group``: every zarr read routes
            # through ``luxar._zarr_compat``, and it opts reads out of
            # consolidated metadata, so a directory store is read from the same
            # per-node ``.zattrs`` the archive peek below reads. Measured on zarr
            # 3.3, the two spellings agree for a ROOT GROUP'S OWN ATTRS even with
            # a stale ``.zmetadata`` present — ``.zmetadata`` governs child
            # lookups, not the root's attrs — so this is the module convention
            # holding rather than a divergence being papered over.
            root = zc_open_group(p, mode="r")
            attrs = dict(root.attrs)
        else:
            # An archive is peeked, not extracted: only the root `.zattrs`
            # member's bytes are read, and nothing is written to disk.
            # A regular file that is not an archive yields {} from the helper.
            attrs = read_archive_root_attrs(p)
    except Exception:
        return {}
    return {k: attrs[k] for k in sorted(AUTHORED_APPEARANCE_ATTRS) if k in attrs}


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

    # Archive resolution is shared with inspect_gsplats_zarr, but this caller
    # deliberately does NOT opt into `flat_zip_in_place` (see its docstring).
    zarr_path, temp_dir = resolve_store_path(path)

    try:
        # Open zarr store
        root = zc_open_group(str(zarr_path), mode="r")

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
                f"Unsupported format_version: {format_version!r} "
                f"(expected one of {SUPPORTED_FORMAT_VERSIONS}). The on-disk "
                f"format is a detached node-tree subtree. Convert legacy "
                f"v1.x / v2.0 files (and old substitutive directories) with "
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
