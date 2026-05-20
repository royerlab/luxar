"""Save Gaussian splat results to .gsplats.zarr format."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import zarr
from zarr.storage import DirectoryStore

from luxar.encoding import ArrayEncoder, EncodingMode, SemanticType
from luxar.io.ordering import (
    compute_chunk_bounds_gsplats,
    sort_splats_spatial,
)
from luxar.io.reader import DEFAULT_COMP
from luxar.typing_utils import TARGET_CHUNK_BYTES

# Get luxar.gsplats version
try:
    from luxar.gsplats import (  # type: ignore[attr-defined]
        __version__ as GSPLATS_VERSION,
    )
except ImportError:
    GSPLATS_VERSION: str = "unknown"  # type: ignore[no-redef]


def _compute_chunk_size(n_splats: int, ndim: int) -> int:
    """Compute chunk size in elements from byte target.

    Args:
        n_splats: Total number of splats
        ndim: Number of dimensions

    Returns:
        Number of splats per chunk
    """
    # Estimate bytes per splat (conservative):
    # centers: ndim * 4 bytes
    # amplitudes: 4 bytes
    # cholesky: (ndim * (ndim + 1) // 2) * 4 bytes
    # colors (optional): 12 bytes
    # Use conservative estimate
    bytes_per_splat = ndim * 4 + 4 + (ndim * (ndim + 1) // 2) * 4 + 12

    chunk_elements = max(1024, TARGET_CHUNK_BYTES // bytes_per_splat)

    # Cap at total splats
    chunk_elements = min(chunk_elements, n_splats)

    return chunk_elements


def _save_splat_arrays_to_group(
    splats_group: "zarr.Group",
    centers: np.ndarray,
    amplitudes: np.ndarray,
    cholesky_factors: np.ndarray,
    colors: Optional[np.ndarray],
    ordering: Literal["morton", "hilbert", "none"],
    encoding_mode: "EncodingMode",
    color_mode: Optional[Literal["sdr", "hdr"]],
    positive_scalar_encoding: Literal["linear", "log"],
    float16_allowed: bool,
    lod_stats: Optional[Dict[str, Any]] = None,
    compressor: Optional[Any] = None,
    truncation_radius: float = 3.0,
) -> None:
    """Write splat arrays into an existing zarr group.

    This is the inner workhorse used by both ``save_gsplats()`` (single-LOD)
    and the LOD-aware save path in ``GSplatData.save()``.
    """
    n_splats, ndim = centers.shape
    expected_chol_size = ndim * (ndim + 1) // 2

    # Validation
    if amplitudes.shape != (n_splats,):
        raise ValueError(
            f"Amplitudes shape {amplitudes.shape} doesn't match centers ({n_splats},)"
        )
    if cholesky_factors.shape != (n_splats, expected_chol_size):
        raise ValueError(
            f"Cholesky factors shape {cholesky_factors.shape} doesn't match "
            f"expected ({n_splats}, {expected_chol_size})"
        )
    if colors is not None:
        if colors.shape != (n_splats, 3):
            raise ValueError(
                f"Colors shape {colors.shape} doesn't match ({n_splats}, 3)"
            )
        if np.issubdtype(colors.dtype, np.floating) and color_mode is None:
            raise ValueError(
                "color_mode must be specified ('sdr' or 'hdr') when colors are float32"
            )

    # Apply spatial ordering (skip for empty data)
    if ordering != "none" and n_splats > 0:
        sort_indices, ordering_metadata = sort_splats_spatial(centers, method=ordering)
        centers = centers[sort_indices]
        amplitudes = amplitudes[sort_indices]
        cholesky_factors = cholesky_factors[sort_indices]
        if colors is not None:
            colors = colors[sort_indices]
    else:
        ordering_metadata = {"ordering": "none"}

    # Compute chunk size and bounds
    chunk_size = _compute_chunk_size(n_splats, ndim)
    chunk_bounds = compute_chunk_bounds_gsplats(
        centers, cholesky_factors, chunk_size, coverage_sigma=truncation_radius
    )

    # Compute amplitude ranges for metadata
    if n_splats == 0:
        amplitude_min, amplitude_max = 0.0, 0.0
        center_min = [0.0] * ndim
        center_max = [0.0] * ndim
    else:
        amplitude_min = float(amplitudes.min())
        amplitude_max = float(amplitudes.max())
        center_min = centers.min(axis=0).tolist()
        center_max = centers.max(axis=0).tolist()

    # Splats group attributes
    splats_attrs: Dict[str, Any] = {
        "type": "gsplats",
        "n_splats": n_splats,
        "ndim": ndim,
        "has_colors": colors is not None,
        "chunk_size": chunk_size,
        "amplitude_range": {"min": amplitude_min, "max": amplitude_max},
        "center_bounds": {"min": center_min, "max": center_max},
        "truncation_radius": truncation_radius,
    }
    splats_attrs.update(ordering_metadata)
    if lod_stats is not None:
        splats_attrs["lod_stats"] = lod_stats
    splats_group.attrs.update(splats_attrs)

    # Create ArrayEncoder
    encoder = ArrayEncoder(float16_allowed=float16_allowed)

    # Compute chunks for arrays, capped at n_splats to avoid zero-padding
    # in small LODs (e.g. 464 splats in a 5461-row chunk).
    cap = max(1, n_splats)

    centers_bytes_per_row = ndim * 4
    centers_chunk_elements = min(TARGET_CHUNK_BYTES // centers_bytes_per_row, cap)
    centers_chunks = (centers_chunk_elements, ndim)

    chol_bytes_per_row = expected_chol_size * 4
    chol_chunk_elements = min(TARGET_CHUNK_BYTES // chol_bytes_per_row, cap)
    chol_chunks = (chol_chunk_elements, expected_chol_size)

    scalar_chunk_elements = min(TARGET_CHUNK_BYTES // 4, cap)
    scalar_chunks = (scalar_chunk_elements,)

    # Write centers
    encoder.encode(
        data=centers,
        zarr_group=splats_group,
        name="centers",
        semantic_type=SemanticType.COORDINATE,
        mode=encoding_mode,
        chunks=centers_chunks,
        compressor=compressor,
    )

    # Write amplitudes
    encoder.encode(
        data=amplitudes,
        zarr_group=splats_group,
        name="amplitudes",
        semantic_type=SemanticType.POSITIVE_SCALAR,
        mode=encoding_mode,
        positive_scalar_encoding=positive_scalar_encoding,
        chunks=scalar_chunks,
        compressor=compressor,
    )

    # Write cholesky_factors
    encoder.encode(
        data=cholesky_factors,
        zarr_group=splats_group,
        name="cholesky_factors",
        semantic_type=SemanticType.CHOLESKY,
        mode=encoding_mode,
        chunks=chol_chunks,
        compressor=compressor,
    )

    # Write colors (optional)
    if colors is not None:
        colors_bytes_per_row = 3 if colors.dtype == np.uint8 else 12
        colors_chunk_elements = min(TARGET_CHUNK_BYTES // colors_bytes_per_row, cap)
        colors_chunks = (colors_chunk_elements, 3)
        encoder.encode(
            data=colors,
            zarr_group=splats_group,
            name="colors",
            semantic_type=SemanticType.COLOR,
            mode=encoding_mode,
            color_mode=color_mode,
            chunks=colors_chunks,
            compressor=compressor,
        )

    # Write chunk_bounds (no encoding, single chunk)
    splats_group.create_dataset(
        "chunk_bounds",
        data=chunk_bounds,
        chunks=(chunk_bounds.shape[0], ndim, 2),
        dtype=np.float32,
        compressor=compressor,
    )


def save_gsplats(
    path: str | Path,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    cholesky_factors: np.ndarray,
    colors: Optional[np.ndarray] = None,
    ordering: Literal["morton", "hilbert", "none"] = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    color_mode: Optional[Literal["sdr", "hdr"]] = None,
    positive_scalar_encoding: Literal["linear", "log"] = "linear",
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    float16_allowed: bool = False,
    compress: Optional[Literal["zip", "tar.gz"]] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    zip_deflate: bool = False,
    truncation_radius: float = 3.0,
) -> None:
    """Save Gaussian splats to .gsplats.zarr format.

    Args:
        path: Output path (should end with .gsplats.zarr or .gsplats.zarr.zip/.tar.gz if compress is used)
        centers: Splat centers, shape (N, d), float32
        amplitudes: Splat amplitudes, shape (N,), float32
        cholesky_factors: Packed Cholesky factors, shape (N, d*(d+1)//2), float32
        colors: Optional RGB colors, shape (N, 3), float32 or uint8
        ordering: Spatial ordering method ("morton", "hilbert", or "none")
        encoding_mode: Encoding mode (AUTO, PRECISION, or MEMORY)
        color_mode: Required if colors are float32 ("sdr" or "hdr")
        positive_scalar_encoding: Encoding for amplitudes ("linear" or "log")
        fitting_info: Optional fitting statistics (fitter-agnostic)
        fitting_config: Optional fitter-specific configuration
        provenance_info: Optional image provenance metadata
        description: Optional user description
        float16_allowed: Enable float16 encoding (default: False for compatibility)
        compress: Optional compression format ("zip" or "tar.gz"). Creates compressed archive.
        zip_deflate: Use DEFLATE compression for the outer zip (default: STORED).
        truncation_radius: Gaussian truncation radius in standard deviations (default 3.0).

    Raises:
        ValueError: If arrays have incompatible shapes or invalid parameters
    """
    import shutil
    import tempfile

    path = Path(path)

    # Determine zarr directory path (may be temporary if compressing)
    temp_dir = None
    if compress:
        # Create zarr in temp, then compress
        temp_dir = Path(tempfile.mkdtemp(prefix="luxar_gsplat_save_"))
        # Extract base name without compression suffix
        zarr_name = path.name
        for suffix in [".zip", ".tar.gz", ".gz"]:
            if zarr_name.endswith(suffix):
                zarr_name = zarr_name[: -len(suffix)]
        if not zarr_name.endswith(".gsplats.zarr"):
            zarr_name = zarr_name + ".gsplats.zarr"
        zarr_path = temp_dir / zarr_name
    else:
        zarr_path = path

    # Create zarr store
    store = DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)

    # Write root attributes (format v2.0 — 2-D LOD: substitutive × additive)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root.attrs.update(
        {
            "format_version": "2.0",
            "format_type": "gsplats_zarr",
            "timestamp": timestamp,
            "luxar_gsplats_version": GSPLATS_VERSION,
            "n_substitutive": 1,
            "default_substitutive": 0,
        }
    )
    if description:
        root.attrs["description"] = description

    # Create splats / substitutive_0 / additive_0 nested group structure.
    # For the trivial 1×1 case (today's "single splat set"), n_substitutive=1
    # and there's a single additive sub-LOD at index 0. The same wiring scales
    # to [N, M_i] when written via GSplatData._save_multi_lod.
    splats_group = root.create_group("splats")
    splats_group.attrs.update(
        {
            "type": "gsplats",
            "n_substitutive": 1,
            "default_substitutive": 0,
            "truncation_radius": truncation_radius,
        }
    )
    sub_group = splats_group.create_group("substitutive_0")
    sub_group.attrs.update(
        {
            "n_additive_sublods": 1,
            "compression_factor": 1,
            "parent_method": "",  # "" sentinel == None at the finest level
            "level_index": 0,
        }
    )
    additive_group = sub_group.create_group("additive_0")
    _save_splat_arrays_to_group(
        splats_group=additive_group,
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        colors=colors,
        ordering=ordering,
        encoding_mode=encoding_mode,
        color_mode=color_mode,
        positive_scalar_encoding=positive_scalar_encoding,
        float16_allowed=float16_allowed,
        compressor=compressor,
        truncation_radius=truncation_radius,
    )

    # Write fitting info (optional)
    if fitting_info is not None:
        fitting_group = root.create_group("fitting")
        fitting_group.attrs.update(fitting_info)

        # Write fitting config (optional, fitter-specific)
        if fitting_config is not None:
            config_group = fitting_group.create_group("config")
            config_group.attrs.update(fitting_config)

    # Write provenance info (optional)
    if provenance_info is not None:
        provenance_group = root.create_group("provenance")
        provenance_group.attrs.update(provenance_info)

    # Consolidate metadata for fast loading
    zarr.consolidate_metadata(store)

    # Compress if requested
    if compress:
        try:
            import tarfile
            import zipfile

            if compress == "zip":
                zip_method = zipfile.ZIP_DEFLATED if zip_deflate else zipfile.ZIP_STORED
                with zipfile.ZipFile(path, "w", zip_method) as zipf:
                    for file_path in zarr_path.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(zarr_path.parent)
                            zipf.write(file_path, arcname)

            elif compress == "tar.gz":
                # tar.gz applies gzip on top of already-compressed zarr chunks,
                # but the user explicitly chose this format.
                with tarfile.open(path, "w:gz") as tarf:
                    tarf.add(zarr_path, arcname=zarr_path.name)

        finally:
            # Cleanup temp directory
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
