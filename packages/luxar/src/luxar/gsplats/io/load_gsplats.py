"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import zarr

from luxar.gsplats import GSplatData


def _extract_compressed_zarr(compressed_path: Path) -> Path:
    """Extract compressed zarr archive to temporary directory.

    Args:
        compressed_path: Path to .gsplats.zarr.zip or .gsplats.zarr.tar.gz

    Returns:
        Path to extracted .gsplats.zarr directory (in temp)
    """
    import tarfile
    import zipfile

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="luxar_gsplat_"))

    # Determine compression type and extract
    if compressed_path.suffix == ".zip" or str(compressed_path).endswith(
        ".gsplats.zarr.zip"
    ):
        # ZIP extraction (with path traversal protection)
        with zipfile.ZipFile(compressed_path, "r") as zip_ref:
            temp_dir_resolved = Path(temp_dir).resolve()
            for zip_member in zip_ref.namelist():
                # Reject absolute paths, parent traversal, and backslash sep.
                if "\\" in zip_member or zip_member.startswith("/"):
                    raise ValueError(
                        f"Zip member '{zip_member}' has unsafe path separator"
                    )
                zip_member_path = (Path(temp_dir) / zip_member).resolve()
                try:
                    zip_member_path.relative_to(temp_dir_resolved)
                except ValueError as exc:
                    raise ValueError(
                        f"Zip member '{zip_member}' would escape extraction directory"
                    ) from exc
            zip_ref.extractall(temp_dir)
    elif compressed_path.suffix == ".gz" or str(compressed_path).endswith(
        (".tar.gz", ".gsplats.zarr.tar.gz")
    ):
        # TAR.GZ extraction (with path traversal protection)
        with tarfile.open(compressed_path, "r:gz") as tar_ref:
            temp_dir_resolved = Path(temp_dir).resolve()
            # Validate every member BEFORE extracting anything, so a malicious
            # archive cannot write a single byte (CVE-2007-4559 + symlink escape).
            for member in tar_ref.getmembers():
                # A legitimate .gsplats.zarr archive is only regular files and
                # directories — never links. Reject sym/hard links outright:
                # otherwise a symlink member pointing outside temp_dir followed
                # by a file member written "through" it escapes the extraction
                # dir (name-only validation does not catch this — the file's own
                # name resolves inside temp_dir).
                if member.issym() or member.islnk():
                    raise ValueError(
                        f"Tar member '{member.name}' is a link; refusing "
                        "(gsplats archives must contain only regular files)"
                    )
                member_path = Path(temp_dir) / member.name
                if not member_path.resolve().is_relative_to(temp_dir_resolved):
                    raise ValueError(
                        f"Tar member '{member.name}' would escape extraction directory"
                    )
            tar_ref.extractall(temp_dir)
    else:
        raise ValueError(f"Unsupported compression format: {compressed_path}")

    # Find the extracted .gsplats.zarr directory
    # It should be the only directory in temp_dir or have .gsplats.zarr suffix
    extracted_dirs = list(temp_dir.iterdir())
    zarr_dir = None

    for d in extracted_dirs:
        if d.is_dir() and d.name.endswith(".gsplats.zarr"):
            zarr_dir = d
            break

    if zarr_dir is None:
        # Fallback: use first directory
        if extracted_dirs and extracted_dirs[0].is_dir():
            zarr_dir = extracted_dirs[0]
        else:
            raise ValueError(f"No .gsplats.zarr directory found in {compressed_path}")

    return zarr_dir


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
        zarr_path = _extract_compressed_zarr(path)
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
                f"format is now v3.1 (a detached node-tree subtree; v3.0 is also "
                f"read). Convert legacy v1.x / v2.0 files (and old substitutive "
                f"directories) with "
                f"`luxar gsplat migrate-format <input> <output.gsplats.zarr>`."
            )

        # Read the node-tree subtree rooted at the file.
        from luxar.io._compiler.gsplat_tree import read_gsplat_node

        node = read_gsplat_node(root, root)

        # Gather root-level stats (fitting / provenance / header).
        stats: Dict[str, Any] = {}
        if include_stats:
            if "fitting" in root:
                for key, value in root["fitting"].attrs.items():
                    stats[key] = value
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
