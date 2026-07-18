"""GSplats write pipelines (bodies of ``write_gsplats`` /
``write_gsplat_leaf_subtree``).

Extracted from the orchestrator. The heavy lifting already lives in
``_compiler/gsplat_assembly.py`` and ``_compiler/gsplat_tree.py``; these
functions are the thin pipelines that sequence those steps behind a narrow
:class:`~luxar.io._compiler.context.GSplatsWriteCtx`. Behavior-preserving.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ....typing_utils.aliases import NodePath
from ..context import GSplatsWriteCtx
from ..gsplat_assembly import (
    apply_gsplat_spatial_ordering,
    validate_gsplat_inputs,
    write_gsplat_arrays,
)
from ..labels.image_labels import write_image_labels_csr
from ..labels.text_labels import write_labels_csr
from ..node_common import validate_render_attrs


def scene_barrier_dims(store: zarr.Group, n_dims: int) -> Optional[List[int]]:
    """Barrier (categorical) axes from the scene's ``Dimension`` metadata.

    Mirrors the Points/Lines slice-dim split (``d.discrete and not d.display``).
    Returns ``None`` when the scene carries no dimensions (→ per-leaf
    auto-detect); an explicit list (possibly empty) otherwise.
    """
    if "scene_dimensions" not in store.attrs:
        return None
    from ....core.dimensions import Dimensions

    dims = Dimensions.from_dict(store.attrs["scene_dimensions"]).dimensions
    return [
        i for i, d in enumerate(dims) if i < n_dims and d.discrete and not d.display
    ]


def write_gsplats(
    ctx: GSplatsWriteCtx,
    path: NodePath,
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]] = None,
    labels: Optional["Sequence[str]"] = None,
    image_labels: Optional[Any] = None,
    **attrs: Any,
) -> dict[str, Any]:
    """Write Gaussian splats data to Zarr (single-LOD, flat layout).

    Returns the node metadata; the caller records it in the metadata cache.
    """
    # Fail fast on invalid render attrs BEFORE creating the group, so a bad
    # value cannot leave a partial node on disk.
    validate_render_attrs(attrs)

    path = path.lstrip("/")
    group = ctx.store.require_group(path)

    # Validate
    (
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        n_splats,
        n_dims,
        cholesky_is_uniform,
    ) = validate_gsplat_inputs(centers, amplitudes, cholesky_factors, colors)

    # Extract truncation_radius for spatial ordering (default 3.0)
    truncation_radius = float(attrs.get("truncation_radius", 3.0))

    aprint(f"📝 Writing {n_splats:,} gsplats ({n_dims}D) to {path}")
    if isinstance(amplitudes, (int, float)):
        aprint(f"  → Uniform amplitude {amplitudes:.3f} for all splats")
    if colors is not None and isinstance(colors, (list, tuple)):
        aprint(f"  → Uniform color RGB{list(colors)} for all splats")
    if cholesky_is_uniform:
        aprint(
            f"  → Uniform Cholesky factors (shape {n_dims * (n_dims + 1) // 2}) "
            f"for all splats"
        )

    # Spatial ordering. Scene gsplats carry authoritative dimension semantics —
    # use the discrete non-displayed axes as the ordering barrier, exactly like
    # the Points/Lines scene path (sort_points_compound slice_dims). This is
    # authoritative regardless of coordinate value spacing, so a non-integer
    # categorical axis (e.g. physical-time seconds) is handled where the
    # value-based auto-detect fallback would miss it.
    (
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        ordering_data,
    ) = apply_gsplat_spatial_ordering(
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        n_splats,
        n_dims,
        cholesky_is_uniform,
        ctx.ordering_ctx,
        truncation_radius,
        barrier_dims=scene_barrier_dims(ctx.store, n_dims),
    )

    # Write arrays
    metadata = write_gsplat_arrays(
        group,
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        n_splats,
        n_dims,
        cholesky_is_uniform,
        ordering_data,
        ctx.dataset_ctx,
    )

    # Set group attrs (owns the warn-once colormap-LUT flag on the orchestrator)
    ctx.apply_gsplat_group_attrs(group, metadata, attrs)

    # Update scene-level bounds
    ctx.update_scene_bounds(metadata["position_bounds"])

    # Write labels if provided (CSR-style: label_offsets + label_bytes)
    if labels is not None:
        sort_order = ordering_data["sort_order"] if ordering_data is not None else None
        write_labels_csr(group, labels, n_splats, ctx.compressor, sort_order)
        metadata["has_labels"] = True

    # Write image labels if provided (CSR-style, no compression on blobs)
    if image_labels is not None:
        sort_order = ordering_data["sort_order"] if ordering_data is not None else None
        write_image_labels_csr(
            group, image_labels, n_splats, ctx.compressor, sort_order
        )
        metadata["has_image_labels"] = True

    aprint(f"✅ GSplats written to {path}")

    return metadata


def write_gsplat_leaf_subtree(
    ctx: GSplatsWriteCtx,
    path: NodePath,
    leaf: Any,  # luxar.gsplats.tree.GSplatLeaf
    **attrs: Any,
) -> dict[str, Any]:
    """Write a ``GSplatLeaf`` (single set or additive ladder) into the scene.

    Seam onto :func:`~luxar.io._compiler.gsplat_tree.write_gsplat_leaf` — the
    single authoring path also used by the standalone ``.gsplats.zarr`` writer,
    so a scene additive ladder is byte-identical to a standalone one.

    Returns the aggregate metadata; the caller records it in the metadata cache.
    """
    from ..gsplat_tree import write_gsplat_leaf

    # Fail fast on invalid render attrs BEFORE creating the group, so a bad
    # value cannot leave a partial node on disk.
    validate_render_attrs(attrs)

    path = path.lstrip("/")
    group = ctx.store.require_group(path)

    metadata = write_gsplat_leaf(
        group,
        leaf,
        dataset_ctx=ctx.dataset_ctx,
        ordering_ctx=ctx.ordering_ctx,
        store=ctx.store,
        attrs=attrs,
        scene_tone_mapping=ctx.scene_tone_mapping,
        # Scene-embedded GSplatData (flat leaf / additive ladder) uses the same
        # authoritative scene-dimension barrier as the array path, so a
        # non-integer categorical axis is grouped correctly (not left to the
        # value-based auto-detect fallback).
        barrier_dims=scene_barrier_dims(ctx.store, leaf.ndim),
    )

    ctx.update_scene_bounds(metadata["position_bounds"])
    n_sub = metadata.get("n_additive_sublods", 1)
    aprint(
        f"✅ GSplats leaf written to {path} "
        f"({metadata['n_splats']:,} splats, {n_sub} additive sub-LOD(s))"
    )
    return metadata
