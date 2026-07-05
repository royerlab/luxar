"""GSplat assembly: validation, spatial ordering, array writes, group attrs.

Private support module shared by the scene compiler
(:class:`~luxar.io.compiler.LuxarZarrCompiler`) and the root-agnostic node-tree
writer (:mod:`luxar.io._compiler.gsplat_tree`). Holds the gsplat-specific pipeline
both sequence: validate inputs, (optionally) spatially order, write the zarr
arrays, then stamp the group attributes. Because both paths call these same four
free functions, a scene leaf is byte-identical to a standalone one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ...core.dimensions import Dimensions
from ...encoding import SemanticType
from .chunking import calculate_intelligent_chunks
from .colormap import write_colormap_lut_if_needed
from .context import DatasetCtx, OrderingCtx
from .dataset_writers.colors import write_colors
from .dataset_writers.scalars import write_positive_scalar


def validate_gsplat_inputs(
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]] = None,
) -> Tuple[
    NDArray[np.float32],
    Union[NDArray[np.float32], float],
    NDArray[np.float32],
    Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    int,
    int,
    bool,
]:
    """Validate and normalize gsplat inputs.

    Returns:
        (centers, amplitudes, cholesky_factors, colors,
         n_splats, n_dims, cholesky_is_uniform)
    """
    from ...validation.base import validate_positions_for_writing

    n_splats, n_dims = validate_positions_for_writing(centers)
    expected_k = n_dims * (n_dims + 1) // 2
    cholesky_is_uniform = False

    if cholesky_factors.ndim == 1:
        if cholesky_factors.shape[0] != expected_k:
            raise ValueError(
                f"Cholesky factors shape mismatch: expected ({expected_k},), "
                f"got {cholesky_factors.shape}"
            )
        cholesky_factors = cholesky_factors.reshape(1, expected_k)
        cholesky_is_uniform = True
    elif cholesky_factors.shape != (n_splats, expected_k):
        raise ValueError(
            f"Cholesky factors shape mismatch: expected ({n_splats}, {expected_k}), "
            f"got {cholesky_factors.shape}"
        )

    # Validate amplitudes
    if isinstance(amplitudes, np.ndarray):
        if amplitudes.shape[0] != n_splats:
            raise ValueError(
                f"Amplitudes shape {amplitudes.shape} doesn't match n_splats {n_splats}"
            )
        if np.any(amplitudes < 0):
            min_val = float(np.min(amplitudes))
            raise ValueError(
                f"Amplitudes must be non-negative (>= 0). Found minimum value: {min_val:.3f}"
            )
    elif isinstance(amplitudes, (int, float)) and amplitudes < 0:
        raise ValueError(f"Amplitude must be non-negative (>= 0). Got {amplitudes}")

    return (
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        n_splats,
        n_dims,
        cholesky_is_uniform,
    )


def apply_gsplat_spatial_ordering(
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    n_splats: int,
    n_dims: int,
    cholesky_is_uniform: bool,
    ctx: OrderingCtx,
    coverage_sigma: float = 3.0,
) -> Tuple[
    NDArray[np.float32],
    Union[NDArray[np.float32], float],
    NDArray[np.float32],
    Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    Optional[Dict[str, Any]],
]:
    """Apply spatial ordering to gsplat arrays.

    Returns:
        (centers, amplitudes, cholesky_factors, colors, ordering_data)
        where ordering_data is None if ordering was not applied.
    """
    ordering_data = None
    if ctx.enable_spatial_index and n_splats > 0:
        from ..ordering import compute_chunk_bounds_gsplats, sort_splats_spatial

        aprint(f"  🔍 Applying {ctx.ordering_method} ordering to gsplats...")
        sort_indices, ordering_metadata = sort_splats_spatial(
            centers, method=ctx.ordering_method
        )

        centers = centers[sort_indices]
        if not cholesky_is_uniform:
            cholesky_factors = cholesky_factors[sort_indices]
        if isinstance(amplitudes, np.ndarray):
            amplitudes = amplitudes[sort_indices]
        if colors is not None and isinstance(colors, np.ndarray):
            if colors.shape[0] > 1:
                colors = colors[sort_indices]

        from ...typing_utils import TARGET_CHUNK_BYTES

        expected_k = n_dims * (n_dims + 1) // 2
        bytes_per_splat = n_dims * 4 + 4 + expected_k * 4 + 16
        chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_splat)
        chunk_size = min(chunk_size, n_splats)

        chunk_bounds = compute_chunk_bounds_gsplats(
            centers,
            cholesky_factors,
            chunk_size,
            coverage_sigma=coverage_sigma,
        )

        ordering_data = {
            "sort_order": sort_indices,
            "chunk_bounds": chunk_bounds,
            "chunk_size": chunk_size,
            **ordering_metadata,
        }

        aprint(
            f"  ✓ Spatial ordering complete: {ordering_metadata['ordering']} "
            f"with {len(chunk_bounds)} chunks"
        )

    return centers, amplitudes, cholesky_factors, colors, ordering_data


def write_gsplat_arrays(
    group: zarr.Group,
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    n_splats: int,
    n_dims: int,
    cholesky_is_uniform: bool,
    ordering_data: Optional[Dict[str, Any]],
    ctx: DatasetCtx,
) -> dict[str, Any]:
    """Write gsplat arrays to a zarr group and return metadata.

    This is the core array-writing routine used by both single-LOD
    and multi-LOD writers.

    Returns:
        Metadata dict with n_splats, ndim, has_colors, amplitude_range,
        center_bounds, and ordering info.
    """
    from ...validation.base import validate_colors_for_writing

    # Write centers
    chunks_centers = calculate_intelligent_chunks(
        centers.shape, spatial_index_data=ordering_data
    )
    ctx.encoder.encode(
        data=centers,
        zarr_group=group,
        name="centers",
        semantic_type=SemanticType.COORDINATE,
        mode=ctx.encoding_mode,
        chunks=chunks_centers,
        compressor=ctx.compressor,
    )

    # Compute amplitude range up-front for the layer-control
    # metadata block below. The helper recomputes max internally;
    # we surface min here because the GSplat metadata dict needs
    # both bounds.
    if isinstance(amplitudes, (int, float)):
        amplitude_min = amplitude_max = float(amplitudes)
    else:
        amplitude_min, amplitude_max = (
            float(np.min(amplitudes)),
            float(np.max(amplitudes)),
        )

    # Canonical POSITIVE_SCALAR helper — shared with Points "radii"
    # and Lines "widths" so the default-precision selection is
    # symmetric across all three geometries.
    write_positive_scalar(
        group=group,
        data=amplitudes,
        name="amplitudes",
        spatial_index_data=ordering_data,
        n_elements=n_splats,
        ctx=ctx,
        log_label_singular="amplitude",
    )

    if isinstance(amplitudes, np.ndarray) and amplitudes.size > 0:
        group.attrs["amplitude_data_range"] = [
            float(amplitudes.min()),
            float(amplitudes.max()),
        ]

    # Write cholesky_factors as two arrays. The diagonal (positive, scale-like)
    # and the off-diagonal (signed, zero-centred) are split so each can be
    # encoded/quantised independently on disk. They are recombined into the
    # packed (N, k) form immediately on read (Python reader + viewer loader),
    # so nothing downstream of the storage boundary sees the split.
    from ...gsplats.utils.trils import split_tril

    chol_diag, chol_offdiag = split_tril(cholesky_factors, n_dims)

    if cholesky_is_uniform:
        n_elems_chol: Optional[int] = n_splats
        chunks_diag: Optional[tuple] = None
        chunks_offdiag: Optional[tuple] = None
    else:
        n_elems_chol = None
        # Chunk both halves with the SAME row-chunk size (derived from the
        # packed shape) so the viewer's aligned per-chunk range reads line up.
        chunk_rows = calculate_intelligent_chunks(
            cholesky_factors.shape, spatial_index_data=ordering_data
        )[0]
        chunks_diag = (chunk_rows, chol_diag.shape[1])
        chunks_offdiag = (chunk_rows, chol_offdiag.shape[1])

    # One joint call: the encoder owns the whole pair policy (same tier for
    # both halves, AUTO's u8→u16 certificate escalation, the 1D no-offdiag
    # case, and deduplicate=False so the per-channel scales stay in each
    # array's own attrs for the viewer).
    ctx.encoder.encode_cholesky_split(
        zarr_group=group,
        diag=chol_diag,
        offdiag=chol_offdiag,
        ndim=n_dims,
        mode=ctx.encoding_mode,
        n_elements=n_elems_chol,
        chunks_diag=chunks_diag,
        chunks_offdiag=chunks_offdiag,
        compressor=ctx.compressor,
    )

    # Compute metadata
    if n_splats > 0:
        center_min = centers.min(axis=0).tolist()
        center_max = centers.max(axis=0).tolist()
    else:
        center_min = [0.0] * n_dims
        center_max = [0.0] * n_dims

    metadata: dict[str, Any] = {
        "n_splats": n_splats,
        "ndim": n_dims,
        "has_colors": False,
        "amplitude_range": {"min": amplitude_min, "max": amplitude_max},
        "center_bounds": {"min": center_min, "max": center_max},
    }

    if ordering_data is not None:
        metadata.update(
            {
                "ordering": ordering_data["ordering"],
                "ordering_min": ordering_data["ordering_min"],
                "ordering_max": ordering_data["ordering_max"],
                "ordering_bits_per_dim": ordering_data["ordering_bits_per_dim"],
                "chunk_size": ordering_data["chunk_size"],
            }
        )
    else:
        metadata["ordering"] = "none"

    # Write colors via the canonical COLOR helper (shared with
    # Points + Lines). The helper handles the tuple/list/ndarray
    # branch, picks the right `color_mode`, and writes
    # `color_data_range` attrs identically across geometries.
    if colors is not None:
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_splats)
        write_colors(
            group=group,
            colors=colors,
            spatial_index_data=ordering_data,
            n_elements=n_splats,
            ctx=ctx,
        )
        metadata["has_colors"] = True

    # Write chunk_bounds
    if ordering_data is not None:
        chunk_bounds = ordering_data["chunk_bounds"]
        if len(chunk_bounds) > 0:
            group.create_dataset(
                "chunk_bounds",
                data=chunk_bounds,
                chunks=(chunk_bounds.shape[0], n_dims, 2),
                dtype=np.float32,
                compressor=ctx.compressor,
                overwrite=True,
            )
            aprint(f"  ✓ Chunk bounds written: {len(chunk_bounds)} chunks")

    return metadata


def apply_gsplat_group_attrs(
    group: zarr.Group,
    metadata: dict[str, Any],
    attrs: dict[str, Any],
    store: zarr.Group,
    scene_tone_mapping: Optional[str],
    lut_tone_mapping_warned: bool,
) -> bool:
    """Set standard gsplats group attributes and rendering defaults.

    Mutates both group.attrs and attrs dict in-place. The colormap-LUT
    tone-mapping warn flag is threaded by value (see
    :func:`~luxar.io._compiler.colormap.write_colormap_lut_if_needed`) and the
    updated flag is returned for the caller to store back.

    Returns:
        The updated ``lut_tone_mapping_warned`` flag.
    """
    # Default colormap if no colors and no colormap
    if not metadata.get("has_colors") and "colormap" not in attrs:
        attrs["colormap"] = "gray"

    # Write colormap LUT if colormap is a custom array
    lut_tone_mapping_warned = write_colormap_lut_if_needed(
        group, attrs, scene_tone_mapping, lut_tone_mapping_warned
    )

    # Process transform if present
    if "transform" in attrs:
        from ...core.transforms import prepare_transform_for_zarr

        attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

    # Validate nd_transform if present
    if "nd_transform" in attrs:
        from ...validation.nd_transforms import validate_nd_transform

        dims = None
        if "scene_dimensions" in store.attrs:
            dims = Dimensions.from_dict(store.attrs["scene_dimensions"])
        attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"], dims)

    # Set rendering defaults
    for key, default in [
        ("opacity", 1.0),
        ("gamma", 1.0),
        ("intensity", 1.0),
        ("offset", 0.0),
        ("blending_mode", "additive"),
        ("truncation_radius", 3.0),
    ]:
        if key not in attrs:
            attrs[key] = default

    # Write all attrs, then override with authoritative metadata
    group.attrs.update(attrs)
    group.attrs["type"] = "gsplats"
    group.attrs["n_splats"] = metadata["n_splats"]
    group.attrs["ndim"] = metadata["ndim"]
    group.attrs["has_colors"] = metadata["has_colors"]
    group.attrs["amplitude_range"] = metadata["amplitude_range"]
    group.attrs["center_bounds"] = metadata["center_bounds"]
    group.attrs["ordering"] = metadata["ordering"]

    if metadata["ordering"] != "none":
        for key in [
            "ordering_min",
            "ordering_max",
            "ordering_bits_per_dim",
            "chunk_size",
        ]:
            if key in metadata:
                group.attrs[key] = metadata[key]
    else:
        default_chunk_size = min(1024, max(64, metadata["n_splats"]))
        group.attrs["chunk_size"] = default_chunk_size

    # Position bounds
    center_bounds = metadata["center_bounds"]
    position_bounds = {"min": center_bounds["min"], "max": center_bounds["max"]}
    group.attrs["position_bounds"] = position_bounds
    metadata["position_bounds"] = position_bounds

    return lut_tone_mapping_warned
