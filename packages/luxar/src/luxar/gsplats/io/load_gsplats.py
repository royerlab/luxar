"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import zarr

from luxar.encoding import ArrayDecoder
from luxar.gsplats import GSplatData
from luxar.gsplats.gsplat_data import GSplatLOD


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
            for member in zip_ref.namelist():
                # Reject absolute paths, parent traversal, and backslash sep.
                if "\\" in member or member.startswith("/"):
                    raise ValueError(
                        f"Zip member '{member}' has unsafe path separator"
                    )
                member_path = (Path(temp_dir) / member).resolve()
                try:
                    member_path.relative_to(temp_dir_resolved)
                except ValueError as exc:
                    raise ValueError(
                        f"Zip member '{member}' would escape extraction directory"
                    ) from exc
            zip_ref.extractall(temp_dir)
    elif compressed_path.suffix == ".gz" or str(compressed_path).endswith(
        (".tar.gz", ".gsplats.zarr.tar.gz")
    ):
        # TAR.GZ extraction (with path traversal protection)
        with tarfile.open(compressed_path, "r:gz") as tar_ref:
            # Validate no path traversal (CVE-2007-4559)
            for member in tar_ref.getmembers():
                member_path = Path(temp_dir) / member.name
                if not member_path.resolve().is_relative_to(Path(temp_dir).resolve()):
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

        format_version = root.attrs.get("format_version")
        if format_version not in ("1.0", "1.1"):
            raise ValueError(
                f"Unsupported format_version: {format_version}, expected '1.0' or '1.1'"
            )

        # Get splats group
        splats_group = root["splats"]

        # Create decoder
        decoder = ArrayDecoder()

        # Build root-level stats dictionary
        stats: Dict[str, Any] = {}
        if include_stats:
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

        # Read truncation radius (default 3.0 for backward compatibility with old files)
        truncation_radius = float(splats_group.attrs.get("truncation_radius", 3.0))

        if format_version == "1.1":
            # v1.1: Multi-LOD format with per-LOD groups
            n_lods = splats_group.attrs.get("n_lods", 1)
            lods = []
            for i in range(n_lods):
                lod_group = splats_group[f"lod_{i}"]
                lod_centers = decoder.decode(lod_group["centers"], root)
                lod_amplitudes = decoder.decode(lod_group["amplitudes"], root)
                lod_cholesky = decoder.decode(lod_group["cholesky_factors"], root)
                lod_colors = (
                    decoder.decode(lod_group["colors"], root)
                    if "colors" in lod_group
                    else None
                )
                lod_stats: Dict[str, Any] = {}
                if include_stats:
                    lod_stats_raw = lod_group.attrs.get("lod_stats", {})
                    if isinstance(lod_stats_raw, dict):
                        lod_stats = dict(lod_stats_raw)
                    lod_stats["n_splats"] = lod_group.attrs.get("n_splats")
                    lod_stats["ndim"] = lod_group.attrs.get("ndim")
                    lod_stats["ordering"] = lod_group.attrs.get("ordering", "none")
                lods.append(
                    GSplatLOD(
                        centers=lod_centers,
                        amplitudes=lod_amplitudes,
                        cholesky_factors=lod_cholesky,
                        colors=lod_colors,
                        stats=lod_stats,
                        truncation_radius=truncation_radius,
                    )
                )
            data = GSplatData(lods=lods, stats=stats)
        else:
            # v1.0: Flat format (single LOD)
            centers = decoder.decode(splats_group["centers"], root)
            amplitudes = decoder.decode(splats_group["amplitudes"], root)
            cholesky_factors = decoder.decode(splats_group["cholesky_factors"], root)
            colors = (
                decoder.decode(splats_group["colors"], root)
                if "colors" in splats_group
                else None
            )
            # Note: old files may contain a "sharpnesses" array — we simply ignore it.
            if include_stats:
                stats["n_splats"] = splats_group.attrs.get("n_splats")
                stats["ndim"] = splats_group.attrs.get("ndim")
                stats["ordering"] = splats_group.attrs.get("ordering", "none")
            data = GSplatData(
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky_factors,
                colors=colors,
                stats=stats,
                truncation_radius=truncation_radius,
            )

        return data

    finally:
        # Cleanup temporary directory if we extracted a compressed archive
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
