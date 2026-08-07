"""Scalar zarr serializers: positive, bounded, colormap scalars, + geometry wrappers.

Canonical writers for the ``POSITIVE_SCALAR`` and ``BOUNDED_SCALAR`` semantic types,
shared across Points (radii/sharpnesses), Lines (widths/sharpnesses), and GSplats
(amplitudes), plus the colormap ``scalars`` writer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ....encoding import SemanticType
from ....typing_utils.constants import SHARPNESS_MAX
from ..chunking import calculate_intelligent_chunks
from ..context import DatasetCtx


def write_positive_scalar(
    group: zarr.Group,
    data: Union[NDArray[np.float32], float, int],
    name: str,
    spatial_index_data: Optional[Dict[str, Any]],
    n_elements: int,
    ctx: DatasetCtx,
    log_label_singular: Optional[str] = None,
    per_array_bytes: bool = False,
) -> float:
    """Canonical writer for the ``POSITIVE_SCALAR`` semantic type.

    Used by Points (``radii``), Lines (``widths``), and GSplats
    (``amplitudes``). Returns the maximum value so callers that
    cache it for layer-controls (e.g. radius/amplitude range
    metadata) don't have to recompute.

    ``log_label_singular`` controls the aprint output —
    ``"radius"`` / ``"width"`` / ``"amplitude"``. Defaults to
    ``name`` if not provided.

    Args:
        group: Zarr group to write to.
        data: Array of positive scalars or a single broadcast value.
        name: Dataset name in the zarr group (e.g. ``"radii"``,
            ``"widths"``, ``"amplitudes"``).
        spatial_index_data: Optional spatial-ordering metadata used
            by ``calculate_intelligent_chunks`` to pick chunk
            boundaries that line up with the spatial index.
        n_elements: Logical element count; must not be inferred from
            the positions array because deduplicated positions can be
            stored as an array_ref with physical shape ``(0, D)``.
        ctx: Encoder configuration (encoder, mode, compressor).
        log_label_singular: Optional singular form for the aprint
            log line ("radius" / "width" / "amplitude"). Defaults to
            the dataset name.
        per_array_bytes: Opt-in per-array dtype byte-budget chunking (points
            only; defaults ``False`` so lines/gsplats keep atom-sized chunks).

    Returns:
        Maximum value across ``data``.
    """
    label = log_label_singular or name

    if isinstance(data, (int, float)):
        max_value = float(data)
        n_elems = n_elements
        chunks = None
    else:
        max_value = float(np.max(data))
        if data.shape[0] == 1:
            n_elems = n_elements
            chunks = None
        else:
            n_elems = None
            chunks = calculate_intelligent_chunks(
                data.shape,
                spatial_index_data=spatial_index_data,
                dtype=data.dtype,
                per_array_bytes=per_array_bytes,
            )

    aprint(f"  ✓ Max {label}: {max_value:.3f}")

    ctx.encoder.encode(
        data=data,
        zarr_group=group,
        name=name,
        semantic_type=SemanticType.POSITIVE_SCALAR,
        mode=ctx.encoding_mode,
        n_elements=n_elems,
        chunks=chunks,
        compressor=ctx.compressor,
    )

    enc = group[name].attrs.get("encoding", {})
    enc_name = enc.get("name", "unknown")
    if enc_name == "broadcasted":
        aprint(f"  ✓ Wrote {name} (broadcasted - uniform)")
    elif enc_name == "array_ref":
        aprint(f"  ✓ Wrote {name} (reference to {enc['target']})")
    elif enc_name.startswith("log_scalar"):
        aprint(f"  ✓ Wrote {name} ({enc_name} - log encoding)")
    else:
        aprint(f"  ✓ Wrote {name} ({enc_name})")

    return max_value


def write_bounded_scalar(
    group: zarr.Group,
    data: Union[NDArray[np.float32], float, int],
    name: str,
    bounds: Tuple[float, float],
    spatial_index_data: Optional[Dict[str, Any]],
    n_elements: int,
    ctx: DatasetCtx,
    log_label_singular: Optional[str] = None,
    per_array_bytes: bool = False,
) -> float:
    """Canonical writer for the ``BOUNDED_SCALAR`` semantic type.

    Used by Points + Lines (``sharpnesses``). The ``bounds`` tuple
    is forwarded to the encoder, which quantizes the data into
    Uint8 normalised to that range when the encoding mode allows.

    Returns the maximum value for callers that surface it on
    layer-control metadata. ``per_array_bytes`` opts into per-array
    dtype byte-budget chunking (points only; defaults ``False`` so
    lines keep atom-sized chunks).
    """
    label = log_label_singular or name

    if isinstance(data, (int, float)):
        max_value = float(data)
        n_elems = n_elements
        chunks = None
    else:
        max_value = float(np.max(data))
        if data.shape[0] == 1:
            n_elems = n_elements
            chunks = None
        else:
            n_elems = None
            chunks = calculate_intelligent_chunks(
                data.shape,
                spatial_index_data=spatial_index_data,
                dtype=data.dtype,
                per_array_bytes=per_array_bytes,
            )

    aprint(f"  ✓ Max {label}: {max_value:.3f}")

    ctx.encoder.encode(
        data=data,
        zarr_group=group,
        name=name,
        semantic_type=SemanticType.BOUNDED_SCALAR,
        mode=ctx.encoding_mode,
        bounds=bounds,
        n_elements=n_elems,
        chunks=chunks,
        compressor=ctx.compressor,
    )

    enc = group[name].attrs.get("encoding", {})
    enc_name = enc.get("name", "unknown")
    if enc_name == "broadcasted":
        aprint(f"  ✓ Wrote {name} (broadcasted - uniform)")
    elif enc_name == "array_ref":
        aprint(f"  ✓ Wrote {name} (reference to {enc['target']})")
    elif enc_name == "bounded_scalar_uint8":
        aprint(f"  ✓ Wrote {name} (uint8, quantized to [{bounds[0]:g}, {bounds[1]:g}])")
    else:
        aprint(f"  ✓ Wrote {name} ({enc_name})")

    return max_value


def write_radii(
    group: zarr.Group,
    radii: Union[NDArray[np.float32], float, int],
    spatial_index_data: Optional[Dict[str, Any]],
    n_points: int,
    ctx: DatasetCtx,
) -> float:
    """Thin Points-specific wrapper over :func:`write_positive_scalar`.

    Kept as a named helper because ``write_points`` reads more
    clearly with the geometry-specific name. New geometries should
    call :func:`write_positive_scalar` directly with their own
    dataset name.
    """
    return write_positive_scalar(
        group=group,
        data=radii,
        name="radii",
        spatial_index_data=spatial_index_data,
        n_elements=n_points,
        ctx=ctx,
        log_label_singular="radius",
        per_array_bytes=True,
    )


def write_sharpness(
    group: zarr.Group,
    sharpness: Union[NDArray[np.float32], float, int],
    spatial_index_data: Optional[Dict[str, Any]],
    n_points: int,
    ctx: DatasetCtx,
) -> float:
    """Thin Points-specific wrapper over :func:`write_bounded_scalar`.

    Like :func:`write_radii`, kept for readability in ``write_points``.
    New geometries call :func:`write_bounded_scalar` directly with the
    appropriate bounds tuple.
    """
    return write_bounded_scalar(
        group=group,
        data=sharpness,
        name="sharpnesses",
        bounds=(0.0, SHARPNESS_MAX),
        spatial_index_data=spatial_index_data,
        n_elements=n_points,
        ctx=ctx,
        log_label_singular="sharpness",
        per_array_bytes=True,
    )


def write_scalars(
    group: zarr.Group,
    scalars: Union[NDArray[np.float32], float, int],
    spatial_index_data: Optional[Dict[str, Any]],
    n_elements: int,
    ctx: DatasetCtx,
    per_array_bytes: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
) -> None:
    """Write scalars dataset to Zarr for colormap lookup.

    Args:
        group: Zarr group to write to
        scalars: Scalar array or uniform value
        spatial_index_data: Optional spatial index for chunk optimization
        n_elements: Logical element count. This must not be inferred from
            the position zarr array because duplicate positions/vertices may
            be stored as an array_ref with physical shape ``(0, D)``.
        ctx: Encoder configuration (encoder, mode, compressor).
        per_array_bytes: Opt-in per-array dtype byte-budget chunking (points
            only; defaults ``False`` so lines/gsplats keep atom-sized chunks).
        bounds: Explicit ``(min, max)`` display window to stamp instead of this
            array's own min/max. For a node that is one LEVEL of a larger field
            — an LOD ladder's coarsened copy — the array's own range is a
            contracted one, and the viewer windows each level's colormap on the
            stamped range, so per-level ranges make the same value a different
            colour at every LOD switch. The supplied pair is WIDENED to cover
            the data if the data exceeds it, because it doubles as the
            BOUNDED_SCALAR quantization range (see the comment below) and a
            value outside it would come back clipped.
    """
    # Validate that this is a geometry group. The logical element count is
    # passed by the caller; physical zarr shape can be zero for array_ref.
    pos_key = next(
        (k for k in ("positions", "vertices", "centers") if k in group),
        None,
    )
    if pos_key is None:
        raise RuntimeError(
            f"No position data found in group '{group.path}' "
            f"(expected 'positions', 'vertices', or 'centers')"
        )

    if isinstance(scalars, (int, float)):
        n_elems = n_elements
        chunks = None
        scalar_min = float(scalars)
        scalar_max = float(scalars)
    else:
        scalars = np.asarray(scalars, dtype=np.float32)
        scalar_min = float(np.min(scalars))
        scalar_max = float(np.max(scalars))
        if scalars.shape[0] == 1:
            n_elems = n_elements
            chunks = None
        else:
            n_elems = None
            chunks = calculate_intelligent_chunks(
                scalars.shape,
                spatial_index_data=spatial_index_data,
                dtype=scalars.dtype,
                per_array_bytes=per_array_bytes,
            )

    # Applied to BOTH branches, so the parameter means one thing wherever it is
    # passed — a uniform-valued level otherwise stamped [v, v] and ignored the
    # ladder's window. Widened, never narrowed: the pair is also the
    # quantization range below, so narrowing it onto the data would put a datum
    # outside its own encoding range.
    if bounds is not None:
        scalar_min = min(scalar_min, float(bounds[0]))
        scalar_max = max(scalar_max, float(bounds[1]))

    # Colormap scalars are legitimately signed (z-scores, velocities,
    # divergence). Encode as BOUNDED_SCALAR (no sign constraint) quantized
    # to the explicit [min, max] just computed, so the stored
    # ``scalar_data_range`` and the quantization range stay identical.
    ctx.encoder.encode(
        data=scalars,
        zarr_group=group,
        name="scalars",
        semantic_type=SemanticType.BOUNDED_SCALAR,
        mode=ctx.encoding_mode,
        bounds=(scalar_min, scalar_max),
        n_elements=n_elems,
        chunks=chunks,
        compressor=ctx.compressor,
    )

    # Store scalar data range for layer controls
    group.attrs["scalar_data_range"] = [scalar_min, scalar_max]
    aprint(f"  ✓ Wrote scalars (range [{scalar_min:.4f}, {scalar_max:.4f}])")
