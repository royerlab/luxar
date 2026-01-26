"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import zarr

from luxar.encoding import ArrayDecoder
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
        # ZIP extraction
        with zipfile.ZipFile(compressed_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
    elif compressed_path.suffix == ".gz" or str(compressed_path).endswith(
        (".tar.gz", ".gsplats.zarr.tar.gz")
    ):
        # TAR.GZ extraction
        with tarfile.open(compressed_path, "r:gz") as tar_ref:
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

    Args:
        path: Path to .gsplats.zarr directory or compressed archive
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

    # Handle compressed archives
    temp_dir = None
    zarr_path = path

    if path.is_file() or str(path).endswith(
        (".zip", ".tar.gz", ".gsplats.zarr.zip", ".gsplats.zarr.tar.gz")
    ):
        # Compressed archive - extract to temp
        zarr_path = _extract_compressed_zarr(path)
        temp_dir = zarr_path.parent

    try:
        # Open zarr store
        root = zarr.open_group(str(zarr_path), mode="r")

        # Validate format
        format_type = root.attrs.get("format_type")
        if format_type != "gsplats_zarr":
            raise ValueError(
                f"Invalid format_type: {format_type}, expected 'gsplats_zarr'"
            )

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

    finally:
        # Cleanup temporary directory if we extracted a compressed archive
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
