"""Inspect .gsplats.zarr metadata without loading arrays."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import zarr

from luxar._zarr_compat import open_group as zc_open_group
from luxar.gsplats.io._archive import resolve_store_path


def _add_label_info(info: Dict[str, Any], attrs: Dict[str, Any]) -> None:
    has_label_ids = attrs.get("has_label_ids", False)
    info["has_label_ids"] = has_label_ids
    info.update(
        {"label_vocabulary": dict(attrs.get("label_vocabulary", {}))}
        if has_label_ids
        else {}
    )


def inspect_gsplats_zarr(path: str | Path) -> Dict[str, Any]:
    """Inspect .gsplats.zarr metadata without loading arrays.

    Accepts a ``.gsplats.zarr`` directory or a ``.gsplats.zarr.zip`` /
    ``.gsplats.zarr.tar.gz`` archive, resolved through the same store-root
    helper the loader uses (an archive nests its store one directory deep) —
    matching the loader's resolution is deliberate.

    "Without loading arrays" is about the ARRAY DATA, not about I/O in general.
    A directory store and a *flat* zip (store at the archive root, opened in
    place as a ``ZipStore`` — the one point where this resolves a shape the
    loader resolves by EXTRACTION instead, see ``resolve_store_path``'s
    ``flat_zip_in_place``) cost nothing beyond reading metadata documents. A
    NESTED archive — what
    ``save_gsplats(..., compress=…)`` writes — is EXTRACTED to a temp directory
    to resolve its store root, so inspecting one costs its full uncompressed
    size in temp space for the duration of the call (measured: a 3.17 MB archive
    writes 3.15 MB across 105 files, linear in dataset size). The temp directory
    is removed before returning, on success and on failure alike.

    Args:
        path: Path to a .gsplats.zarr directory or compressed archive

    Returns:
        Dictionary with format information. ``storage_bytes``/``storage_mb``
        measure the path as given — the archive's own bytes, not the extracted
        copy — and ``compression_ratio`` is ``None`` when that size is unknown
        or zero, rather than an invented 1.0.

        Pre-existing limitation, unchanged here: ``compression_ratio`` and
        ``uncompressed_mb`` are modelled from ``n_splats``, which for a
        multi-node tree (``kind=lod`` / ``kind=partition``) is only the
        REPRESENTATIVE leaf's count while ``storage_bytes`` covers the whole
        tree, so the ratio is only meaningful for a single-leaf store (measured:
        a 4-part partition of 4x1000 splats reports ``n_splats=1000`` and
        ``compression_ratio=0.46``, where the whole tree's own ratio is ~1.9).

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If format is invalid, or the path is a regular file that is
            not a supported archive
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats zarr not found: {path}")

    # `flat_zip_in_place`: a flat zip is metadata-readable in place as a
    # ZipStore, for no temp space at all — extracting one (which is how every
    # other reader resolves the shape) would cost this metadata-only call the
    # dataset's full uncompressed size. See `resolve_store_path`.
    zarr_path, temp_dir = resolve_store_path(path, flat_zip_in_place=True)
    try:
        return _inspect_store(path, zarr_path)
    finally:
        # Remove the extraction temp dir (None for a directory store).
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _inspect_store(path: Path, zarr_path: Path) -> Dict[str, Any]:
    """Inspect an opened-on-disk store.

    Args:
        path: The user-supplied path, whose own bytes are the meaningful
            on-disk size (for an archive that is the archive file itself).
        zarr_path: The resolved store to open — a directory, or the zip FILE
            itself for a flat zip opened in place as a ``ZipStore``.
    """
    # Open zarr store (read-only)
    root = zc_open_group(str(zarr_path), mode="r")

    # Validate format
    format_type = root.attrs.get("format_type")
    if format_type != "gsplats_zarr":
        raise ValueError(f"Invalid format_type: {format_type}, expected 'gsplats_zarr'")

    from luxar.gsplats.io.save_gsplats import SUPPORTED_FORMAT_VERSIONS

    format_version = root.attrs.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported format_version: {format_version!r} "
            f"(expected one of {SUPPORTED_FORMAT_VERSIONS}). "
            f"Convert legacy files with `luxar gsplat migrate-format`."
        )

    # Extract metadata
    info: Dict[str, Any] = {}

    # Root attributes
    info["format_version"] = format_version
    info["format_type"] = format_type
    info["timestamp"] = root.attrs.get("timestamp")
    info["luxar_gsplats_version"] = root.attrs.get("luxar_gsplats_version")

    if "description" in root.attrs:
        info["description"] = root.attrs["description"]

    # v3.0: the file root IS the node. Describe the tree shape, then surface a
    # representative leaf's data attrs for the at-a-glance summary fields.
    info["kind"] = root.attrs.get("kind", "gsplats")

    def _representative_leaf_node(group: "zarr.Group") -> "zarr.Group":
        """Descend through groups to a representative leaf NODE group.

        For a kind=lod group, descend into the FINEST child so the headline
        stats (n_splats, ndim, bounds, …) match the data-model default
        (``default_substitutive=0`` = finest) and the ``gsplat info`` CLI path
        (which loads via ``GSplatData``). On disk children are coarsest-first
        (``child_0`` = coarsest, ``child_{n-1}`` = finest), so we must NOT read
        the on-disk ``default_level`` here — that is the viewer's COARSEST
        render hint, a separate concept.
        """
        kind = group.attrs.get("kind")
        if kind == "lod":
            n = sum(1 for c in group if str(c).startswith("child_"))
            finest = max(n - 1, 0)
            key = f"child_{finest}" if f"child_{finest}" in group else "child_0"
            return _representative_leaf_node(group[key])
        if kind == "partition":
            return _representative_leaf_node(group["part_0"])
        return group

    if info["kind"] == "lod":
        info["n_substitutive"] = sum(1 for n in root if str(n).startswith("child_"))
    elif info["kind"] == "partition":
        info["n_parts"] = sum(1 for n in root if str(n).startswith("part_"))
        info["n_substitutive"] = 1
    else:
        info["n_substitutive"] = 1
    # The data-model default substitutive level is the FINEST (index 0, the
    # GSplatData.default_substitutive convention). Do NOT read the on-disk
    # ``default_level`` here — that is the viewer's COARSEST initial-load hint,
    # a separate concept (conflating the two is the kind=lod default_level bug).
    info["default_substitutive"] = 0
    info["default_lod_level"] = int(root.attrs.get("default_level", 0))

    leaf_node = _representative_leaf_node(root)
    n_additive = int(leaf_node.attrs.get("n_additive_sublods", 1))
    info["n_additive_sublods_default"] = n_additive
    # Data attrs live on the leaf node (single set) or its additive_0 subgroup.
    data_group = leaf_node["additive_0"] if n_additive > 1 else leaf_node
    splats_attrs = dict(data_group.attrs)

    info["n_splats"] = splats_attrs.get("n_splats")
    info["ndim"] = splats_attrs.get("ndim")
    info["has_colors"] = splats_attrs.get("has_colors", False)
    _add_label_info(info, splats_attrs)
    info["ordering"] = splats_attrs.get("ordering", "none")
    info["chunk_size"] = splats_attrs.get("chunk_size")

    # Field-name compatibility for externally produced or older datasets:
    # some files use `morton_*` / `hilbert_resolution` keys, while current
    # writers emit `ordering_*` keys.
    if info["ordering"] in ["morton", "hilbert"]:
        info["ordering_min"] = splats_attrs.get("ordering_min") or splats_attrs.get(
            "morton_min"
        )
        info["ordering_max"] = splats_attrs.get("ordering_max") or splats_attrs.get(
            "morton_max"
        )
        info["ordering_bits_per_dim"] = splats_attrs.get(
            "ordering_bits_per_dim"
        ) or splats_attrs.get("morton_bits_per_dim")
        # Legacy-only key: current writers emit a per-axis bit budget, not a
        # grid resolution — include `ordering_resolution` only when a value
        # actually exists (pre-v3.0 morton_/hilbert_resolution files).
        legacy_key = (
            "morton_resolution"
            if info["ordering"] == "morton"
            else "hilbert_resolution"
        )
        resolution = splats_attrs.get("ordering_resolution") or splats_attrs.get(
            legacy_key
        )
        if resolution is not None:
            info["ordering_resolution"] = resolution

    # Ranges
    info["amplitude_range"] = splats_attrs.get("amplitude_range")
    info["center_bounds"] = splats_attrs.get("center_bounds")

    # Fitting info (optional)
    if "fitting" in root:
        fitting_group = root["fitting"]
        fitting_info = dict(fitting_group.attrs)
        info["fitting"] = fitting_info

        # Fitting config (optional)
        if "config" in fitting_group:
            config_group = fitting_group["config"]
            info["fitting_config"] = dict(config_group.attrs)

    # Provenance info (optional)
    if "provenance" in root:
        provenance_group = root["provenance"]
        info["provenance"] = dict(provenance_group.attrs)

    # Compute storage size. A directory store is walked; anything else is a
    # single file (an archive), whose own st_size IS its on-disk size — summing
    # zero there used to report "0.0 MB" for every archive.
    try:
        if path.is_dir():
            total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            total_bytes = path.stat().st_size
        info["storage_bytes"] = total_bytes
        info["storage_mb"] = round(total_bytes / (1024 * 1024), 2)

        # Compute compression ratio (estimate)
        n_splats = info["n_splats"]
        ndim = info["ndim"]
        chol_size = ndim * (ndim + 1) // 2

        # Uncompressed size estimate (float32 for all)
        uncompressed_bytes = n_splats * (
            ndim * 4  # centers
            + 4  # amplitudes
            + chol_size * 4  # cholesky_factors
            + (12 if info["has_colors"] else 0)  # colors (float32)
            + (4 if info.get("has_label_ids") else 0)  # label ids (upper bound)
        )

        # A ratio against an unmeasurable size is not a measurement — report it
        # as absent rather than publishing an invented 1.0.
        info["compression_ratio"] = (
            round(uncompressed_bytes / total_bytes, 2) if total_bytes > 0 else None
        )
        info["uncompressed_mb"] = round(uncompressed_bytes / (1024 * 1024), 2)

    except Exception:
        # Storage size computation failed (permissions, etc.)
        info["storage_bytes"] = None
        info["storage_mb"] = None
        info["compression_ratio"] = None

    return info


def format_gsplats_info(info: Dict[str, Any]) -> str:
    """Format inspection info as human-readable string.

    Args:
        info: Info dictionary from inspect_gsplats_zarr()

    Returns:
        Formatted string
    """
    lines = []

    # Header
    lines.append(f"GSplats: {info['n_splats']:,} splats, {info['ndim']}D")

    # Ordering — current files carry the per-axis bit budget; only legacy
    # (pre-v3.0 morton_*) files still have a grid resolution.
    ordering = info["ordering"]
    if ordering in ("morton", "hilbert"):
        bits = info.get("ordering_bits_per_dim")
        if bits is not None:
            lines.append(f"Ordering: {ordering} (bits_per_dim={bits})")
        else:
            resolution = info.get("ordering_resolution") or "unknown"
            lines.append(f"Ordering: {ordering} (resolution={resolution})")
    else:
        lines.append("Ordering: none")

    # Optional arrays
    if info["has_colors"]:
        lines.append("Optional arrays: colors")
    if info.get("has_label_ids"):
        lines.append(
            "Categorical labels: "
            f"{len(info.get('label_vocabulary', {}))} vocabulary entries"
        )

    # Storage
    if info.get("storage_mb") is not None:
        mb = info["storage_mb"]
        ratio = info["compression_ratio"]
        uncompressed_mb = info["uncompressed_mb"]
        if ratio is None:
            # Size measured but no ratio available (a zero/unknown size) — print
            # what is known rather than a made-up compression figure.
            lines.append(f"Size: {mb:.1f} MB ({uncompressed_mb:.1f} MB uncompressed)")
        else:
            lines.append(
                f"Size: {mb:.1f} MB ({uncompressed_mb:.1f} MB uncompressed, "
                f"{ratio:.1f}x compression)"
            )

    # Fitting info
    if "fitting" in info:
        fitting = info["fitting"]
        time_sec = fitting.get("time_seconds")
        iterations = fitting.get("iterations")
        converged = fitting.get("converged")

        if time_sec is not None and iterations is not None:
            status = "converged" if converged else "stopped"
            lines.append(
                f"Fitting: {time_sec:.1f}s, {iterations} iterations ({status})"
            )

    return "\n".join(lines)
