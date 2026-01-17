"""Inspect .gsplats.zarr metadata without loading arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
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

    # Extract metadata
    info: Dict[str, Any] = {}

    # Root attributes
    info["format_version"] = root.attrs.get("format_version")
    info["format_type"] = format_type
    info["timestamp"] = root.attrs.get("timestamp")
    info["luxar_gsplats_version"] = root.attrs.get("luxar_gsplats_version")

    if "description" in root.attrs:
        info["description"] = root.attrs["description"]

    # Splats group attributes
    splats_group = root["splats"]
    splats_attrs = dict(splats_group.attrs)

    info["n_splats"] = splats_attrs.get("n_splats")
    info["ndim"] = splats_attrs.get("ndim")
    info["has_colors"] = splats_attrs.get("has_colors", False)
    info["has_sharpness"] = splats_attrs.get("has_sharpness", False)
    info["ordering"] = splats_attrs.get("ordering", "none")
    info["chunk_size"] = splats_attrs.get("chunk_size")

    # Ordering info (try new names first, fall back to old for backward compat)
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
    info["sharpness_bounds"] = splats_attrs.get("sharpness_bounds")
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
            + (4 if info["has_sharpness"] else 0)  # sharpness
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
        resolution = info.get("morton_resolution", "unknown")
        lines.append(f"Ordering: morton (resolution={resolution})")
    elif ordering == "hilbert":
        resolution = info.get("hilbert_resolution", "unknown")
        lines.append(f"Ordering: hilbert (resolution={resolution})")
    else:
        lines.append("Ordering: none")

    # Optional arrays
    optional_arrays = []
    if info["has_colors"]:
        optional_arrays.append("colors")
    if info["has_sharpness"]:
        optional_arrays.append("sharpness")

    if optional_arrays:
        lines.append(f"Optional arrays: {', '.join(optional_arrays)}")

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


def render_gsplats_to_volume(
    centers: np.ndarray,
    cholesky_factors: np.ndarray,
    amplitudes: np.ndarray,
    volume_shape: Tuple[int, int, int],
    sharpness: np.ndarray | None = None,
) -> np.ndarray:
    """Render Gaussian splats into a 3D volume for visualization.

    .. deprecated:: 0.2.0
        This function is deprecated and will be removed in a future version.
        Use :func:`luxar.gsplats.rendering.render_to_volume` or the
        :meth:`GSplatData.render_to_volume` method instead for 100-1000x
        faster GPU-accelerated rendering.

    This is a legacy NumPy implementation that is very slow (O(N_splats × N_voxels))
    and runs on CPU only. The new GPU-accelerated renderer is much faster.

    Args:
        centers: Splat centers (N, 3) in voxel coordinates
        cholesky_factors: Packed Cholesky factors (N, 6) [L00, L10, L11, L20, L21, L22]
        amplitudes: Splat amplitudes (N,)
        volume_shape: Output volume shape (Z, Y, X)
        sharpness: Optional sharpness values (N,). If None, uses s=2.0 (standard Gaussian)

    Returns:
        3D volume (Z, Y, X) with accumulated splat contributions

    See Also:
        :func:`luxar.gsplats.rendering.render_to_volume`: Fast GPU-accelerated rendering
        :meth:`GSplatData.render_to_volume`: Convenience method on GSplatData
    """
    import warnings

    warnings.warn(
        "render_gsplats_to_volume() is deprecated and will be removed in a future version. "
        "Use luxar.gsplats.rendering.render_to_volume() or GSplatData.render_to_volume() "
        "instead for 100-1000x faster GPU-accelerated rendering.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import the new fast renderer
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.rendering.volume_rendering import render_to_volume

    # Default sharpness to 2.0 (standard Gaussian)
    if sharpness is None:
        sharpness = np.full(len(centers), 2.0, dtype=np.float32)

    # Wrap in GSplatData and use new renderer
    gsplat_data = GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        sharpnesses=sharpness,
        colors=None,
        stats={},
    )

    return render_to_volume(gsplat_data, shape=volume_shape)
