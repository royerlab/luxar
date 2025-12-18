"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import zarr

from luxar.encoding import ArrayDecoder
from luxar.gsplats import GSplatData


def load_gsplats(
    path: str | Path,
    include_stats: bool = False,
) -> GSplatData:
    """Load Gaussian splats from .gsplats.zarr format.

    Arrays are automatically decoded from their stored encoding (quantization,
    broadcasting, etc.) to float32.

    Args:
        path: Path to .gsplats.zarr directory
        include_stats: Whether to include fitting/provenance metadata in stats

    Returns:
        GSplatData with decoded arrays and optional stats

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If format is invalid or incompatible
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats zarr not found: {path}")

    # Open zarr store
    root = zarr.open_group(str(path), mode="r")

    # Validate format
    format_type = root.attrs.get("format_type")
    if format_type != "gsplats_zarr":
        raise ValueError(f"Invalid format_type: {format_type}, expected 'gsplats_zarr'")

    format_version = root.attrs.get("format_version")
    if format_version != "1.0":
        raise ValueError(
            f"Unsupported format_version: {format_version}, expected '1.0'"
        )

    # Get splats group
    splats_group = root["splats"]

    # Create decoder
    decoder = ArrayDecoder()

    # Decode arrays (automatically handles encoding metadata)
    centers = decoder.decode(splats_group["centers"], root)
    amplitudes = decoder.decode(splats_group["amplitudes"], root)
    cholesky_factors = decoder.decode(splats_group["cholesky_factors"], root)

    # Decode optional arrays
    if "colors" in splats_group:
        colors = decoder.decode(splats_group["colors"], root)
    else:
        colors = None

    if "sharpnesses" in splats_group:
        sharpnesses = decoder.decode(splats_group["sharpnesses"], root)
    else:
        # Default to standard Gaussian (s=2.0)
        n_splats = centers.shape[0]
        sharpnesses = np.full(n_splats, 2.0, dtype=np.float32)

    # Build stats dictionary
    stats: Dict[str, Any] = {}

    if include_stats:
        # Add basic metadata
        stats["n_splats"] = splats_group.attrs.get("n_splats")
        stats["ndim"] = splats_group.attrs.get("ndim")
        stats["ordering"] = splats_group.attrs.get("ordering", "none")

        # Add fitting info if present
        if "fitting" in root:
            fitting_group = root["fitting"]
            for key, value in fitting_group.attrs.items():
                stats[key] = value

        # Add provenance info if present
        if "provenance" in root:
            provenance_group = root["provenance"]
            stats["provenance"] = dict(provenance_group.attrs)

        # Add root metadata
        stats["format_version"] = format_version
        stats["timestamp"] = root.attrs.get("timestamp")
        stats["luxar_gsplats_version"] = root.attrs.get("luxar_gsplats_version")

        if "description" in root.attrs:
            stats["description"] = root.attrs["description"]

    # Create data object with colors
    data = GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        sharpnesses=sharpnesses,
        colors=colors,
        stats=stats,
    )

    return data
