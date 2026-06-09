"""Inspect .gsplats.zarr metadata without loading arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import zarr


def inspect_gsplats_zarr(path: str | Path) -> Dict[str, Any]:
    """Inspect .gsplats.zarr metadata without loading arrays.

    Args:
        path: Path to .gsplats.zarr directory

    Returns:
        Dictionary with format information

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats zarr not found: {path}")

    # Open zarr store (read-only)
    root = zarr.open_group(str(path), mode="r")

    # Validate format
    format_type = root.attrs.get("format_type")
    if format_type != "gsplats_zarr":
        raise ValueError(f"Invalid format_type: {format_type}, expected 'gsplats_zarr'")

    format_version = root.attrs.get("format_version")
    if format_version != "3.0":
        raise ValueError(
            f"Unsupported format_version: {format_version!r} (expected '3.0'). "
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
        """Descend through groups to a representative leaf NODE group."""
        kind = group.attrs.get("kind")
        if kind == "lod":
            idx = int(group.attrs.get("default_level", 0))
            key = f"child_{idx}" if f"child_{idx}" in group else "child_0"
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
    info["default_substitutive"] = int(root.attrs.get("default_level", 0))

    leaf_node = _representative_leaf_node(root)
    n_additive = int(leaf_node.attrs.get("n_additive_sublods", 1))
    info["n_additive_sublods_default"] = n_additive
    # Data attrs live on the leaf node (single set) or its additive_0 subgroup.
    data_group = leaf_node["additive_0"] if n_additive > 1 else leaf_node
    splats_attrs = dict(data_group.attrs)

    info["n_splats"] = splats_attrs.get("n_splats")
    info["ndim"] = splats_attrs.get("ndim")
    info["has_colors"] = splats_attrs.get("has_colors", False)
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
        if info["ordering"] == "morton":
            info["ordering_resolution"] = splats_attrs.get(
                "ordering_resolution"
            ) or splats_attrs.get("morton_resolution")
        else:
            info["ordering_resolution"] = splats_attrs.get(
                "ordering_resolution"
            ) or splats_attrs.get("hilbert_resolution")

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

    # Compute storage size
    try:
        total_bytes = sum(
            sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            if p.is_dir()
            else 0
            for p in [path]
        )
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
        )

        compression_ratio = uncompressed_bytes / total_bytes if total_bytes > 0 else 1.0
        info["compression_ratio"] = round(compression_ratio, 2)
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

    # Ordering
    ordering = info["ordering"]
    if ordering == "morton":
        resolution = info.get("ordering_resolution") or "unknown"
        lines.append(f"Ordering: morton (resolution={resolution})")
    elif ordering == "hilbert":
        resolution = info.get("ordering_resolution") or "unknown"
        lines.append(f"Ordering: hilbert (resolution={resolution})")
    else:
        lines.append("Ordering: none")

    # Optional arrays
    if info["has_colors"]:
        lines.append("Optional arrays: colors")

    # Storage
    if info.get("storage_mb") is not None:
        mb = info["storage_mb"]
        ratio = info["compression_ratio"]
        uncompressed_mb = info["uncompressed_mb"]
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
