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

    This function evaluates each Gaussian splat on a 3D grid and accumulates
    the contributions to create a volumetric representation.

    Args:
        centers: Splat centers (N, 3) in voxel coordinates
        cholesky_factors: Packed Cholesky factors (N, 6) [L00, L10, L11, L20, L21, L22]
        amplitudes: Splat amplitudes (N,)
        volume_shape: Output volume shape (Z, Y, X)
        sharpness: Optional sharpness values (N,). If None, uses s=2.0 (standard Gaussian)

    Returns:
        3D volume (Z, Y, X) with accumulated splat contributions
    """
    n_splats = centers.shape[0]
    volume = np.zeros(volume_shape, dtype=np.float32)

    # Default sharpness to 2.0 (standard Gaussian)
    if sharpness is None:
        sharpness = np.full(n_splats, 2.0, dtype=np.float32)

    # Create coordinate grid
    Z, Y, X = volume_shape
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(Z, dtype=np.float32),
        np.arange(Y, dtype=np.float32),
        np.arange(X, dtype=np.float32),
        indexing="ij",
    )

    # Stack into (Z*Y*X, 3) coordinate array
    coords = np.stack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()], axis=1)

    # Render each splat
    for i in range(n_splats):
        center = centers[i]
        amplitude = amplitudes[i]
        s = sharpness[i]

        # Unpack Cholesky factor (lower triangular)
        L = np.array(
            [
                [cholesky_factors[i, 0], 0, 0],
                [cholesky_factors[i, 1], cholesky_factors[i, 2], 0],
                [
                    cholesky_factors[i, 3],
                    cholesky_factors[i, 4],
                    cholesky_factors[i, 5],
                ],
            ]
        )

        # Compute difference from center
        diff = coords - center  # (Z*Y*X, 3)

        # Solve L @ y = diff using forward substitution (vectorized)
        y = np.zeros_like(diff)
        y[:, 0] = diff[:, 0] / L[0, 0]
        y[:, 1] = (diff[:, 1] - L[1, 0] * y[:, 0]) / L[1, 1]
        y[:, 2] = (diff[:, 2] - L[2, 0] * y[:, 0] - L[2, 1] * y[:, 1]) / L[2, 2]

        # Mahalanobis distance: ||y||
        mahal_sq = np.sum(y * y, axis=1)

        # Generalized Gaussian: exp(-0.5 * ||y||^s)
        # For s=2 (standard Gaussian), this is exp(-0.5 * mahal_sq)
        # For s≠2, we use ||y||^s = (mahal_sq)^(s/2)
        if s == 2.0:
            density = amplitude * np.exp(-0.5 * mahal_sq)
        else:
            mahal_dist = np.sqrt(mahal_sq)
            density = amplitude * np.exp(-0.5 * np.power(mahal_dist, s))

        # Accumulate into volume
        volume += density.reshape(volume_shape)

    return volume
