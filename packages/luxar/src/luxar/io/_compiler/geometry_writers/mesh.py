"""Mesh write pipeline (body of ``LuxarZarrCompiler.write_mesh``).

Structurally the simplest of the four geometry writers, and deliberately so.
Mesh has no spatial ordering (``ordering`` is always ``"none"`` in v1 — the
loader is whole-node, so there is nothing for a chunk index to skip), no primary
size scalar (a triangle's extent comes from its own vertices, not a per-element
radius/width/covariance), and no LOD or partition path. What remains is: encode
``vertices`` + ``faces``, encode the optional per-vertex channels, stamp the
attrs.

The one structural quirk worth knowing is shared with Lines: ``faces`` is
``SemanticType.INDEX`` written with ``deduplicate=False`` and ``allow_lut=False``,
because the loader reads it as raw chunked zarr and does not resolve
``array_ref``. Dedup would silently drop the topology for a byte-identical
sibling, and LUT encoding of grid-snapped indices would decode as garbage.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from arbol import aprint
from numpy.typing import NDArray

from ....encoding import SemanticType
from ....typing_utils.aliases import NodePath
from ..bounds import compute_position_bounds
from ..chunking import calculate_intelligent_chunks
from ..context import GeometryWriteCtx
from ..dataset_writers.colors import write_colors
from ..dataset_writers.scalars import write_scalars
from ..labels.image_labels import write_image_labels_csr
from ..labels.text_labels import write_labels_csr
from ..node_common import (
    MESH_RESERVED_ATTRS,
    apply_default_render_attrs,
    prepare_transform_attrs,
    validate_broadcast_color,
    validate_node_path,
    validate_render_attrs,
    validate_scalars_preflight,
)

#: Below this face count the unwelded-vertices lint stays quiet — a handful of
#: independent triangles is a normal test fixture, not an authoring mistake.
_AUTHORING_LINT_MIN_FACES = 8


def _is_unwelded(faces: NDArray[np.integer], n_vertices: int) -> bool:
    """Whether the mesh looks like independent triangles rather than a surface.

    The signature is ``V == 3F`` *and* no vertex index used more than once: every
    triangle then owns its three vertices outright, which is what you get by
    flattening a triangle soup instead of building an indexed surface. Checking
    both conditions matters — ``V == 3F`` alone also holds for some legitimately
    welded meshes, and "no shared index" alone holds trivially for a single
    triangle.

    Mirrors ``_exploded_chain_fraction`` in the lines writer: a cheap warn-only
    heuristic for a mistake that otherwise shows up only as bad shading.

    Distinctness is tested with ``bincount``, not ``unique``: ``unique`` sorts,
    which measured 1.5 s on a 1M-face soup — a full second added to a write, for a
    warning, in exactly the case that triggers it. By the time this line runs
    ``V == 3F`` holds and the caller's validator has already established every
    index is in ``[0, V)``, so "all distinct" is equivalent to "every index used
    exactly once", which ``bincount`` answers in one linear pass (~9 ms at 1M
    faces, same verdict on both soup and one-shared-vertex inputs).
    """
    n_faces = faces.shape[0]
    if n_faces < _AUTHORING_LINT_MIN_FACES or n_vertices != 3 * n_faces:
        return False
    return bool(np.bincount(faces.ravel(), minlength=n_vertices).max() == 1)


def _mesh_authoring_warning_key(ctx: GeometryWriteCtx, path: str) -> str:
    """Collapse partition leaves to their logical parent warning key.

    Mesh cannot be partitioned in v1, so this cannot fire today; it mirrors the
    lines writer so the two stay diffable, and so the behaviour is already right
    if a partition path lands later.
    """
    parent_path, separator, _leaf = path.rpartition("/")
    if separator and parent_path in ctx.store:
        parent = ctx.store[parent_path]
        if getattr(parent, "attrs", {}).get("kind") == "partition":
            return parent_path
    return path


def write_mesh(
    ctx: GeometryWriteCtx,
    path: NodePath,
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    normals: Optional[NDArray[np.float32]] = None,
    normal_dims: Optional[Sequence[int]] = None,
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]] = None,
    scalars: Optional[Union[NDArray[np.float32], float]] = None,
    shading: Optional[str] = None,
    double_sided: bool = True,
    labels: Optional["Sequence[str]"] = None,
    image_labels: Optional[Any] = None,
    **attrs: Any,
) -> dict[str, Any]:
    """Write mesh data to Zarr (see ``LuxarZarrCompiler.write_mesh``).

    Returns the node metadata; the caller records it in the metadata cache.
    """
    from ....validation.base import (
        validate_colors_for_writing,
        validate_faces_for_writing,
        validate_labels_for_writing,
        validate_normal_dims_for_writing,
        validate_normals_for_writing,
        validate_positions_for_writing,
        validate_vertices_for_writing,
    )

    # 0. Fail-fast pre-write gate: everything here runs BEFORE the zarr group is
    # created and before any array lands on disk, so an invalid input cannot
    # leave a partial node behind. Same best-effort caveat as the sibling
    # writers: validators needing the store (image_labels, custom colormap LUT
    # resolution) still run post-write.
    #
    # 0a. Pure attr validators + reserved writer-stamp collisions.
    validate_render_attrs(attrs, reserved_attrs=MESH_RESERVED_ATTRS)
    # 0b. Node path: an empty path would resolve require_group("") to the scene
    # ROOT and clobber it.
    path = validate_node_path(path)
    # 0c. Vertices shape/finiteness (shared coordinate path), then the mesh-only
    # vertex-count ceiling. Order matters: the cap reads shape[0], which is only
    # meaningful once the array is known to be 2D.
    n_vertices, n_dims = validate_positions_for_writing(vertices, context="vertices")
    validate_vertices_for_writing(vertices)
    # 0d. Faces: layout, integer dtype, and both index bounds. Runs before the
    # uint32 cast below, which is what makes the bounds check meaningful.
    validate_faces_for_writing(faces, n_vertices)
    # 0e. Normals and their companion attr are a PAIR — each is meaningless
    # without the other. A normal array with no normal_dims cannot be oriented
    # (storing normals against an implicit "first three dimensions" is the bug
    # the attr exists to prevent: for a (t, x, y, z) mesh those are (t, x, y)),
    # and normal_dims with no normals describes nothing.
    if normals is not None:
        validate_normals_for_writing(normals, n_vertices)
        if normal_dims is None:
            raise ValueError(
                "normal_dims is required when normals are supplied: it names "
                "which three dimension indices the 3-component normals describe. "
                "Pass e.g. normal_dims=(0, 1, 2)."
            )
        validate_normal_dims_for_writing(normal_dims, n_dims)
    elif normal_dims is not None:
        raise ValueError(
            "normal_dims was supplied without normals. It names the dimensions "
            "that a normals array describes, so it has no meaning on its own — "
            "pass normals=..., or drop normal_dims."
        )
    # 0f. Shading is metadata the viewer acts on, so a typo must not reach zarr:
    # an unrecognised value would silently take the stored-normal path.
    if shading is not None and shading not in ("smooth", "flat"):
        raise ValueError(f"shading must be 'smooth' or 'flat', got {shading!r}")
    if not isinstance(double_sided, bool):
        raise ValueError(
            f"double_sided must be a bool, got {type(double_sided).__name__}"
        )
    # 0g. Optional per-vertex channels.
    if colors is not None:
        if isinstance(colors, np.ndarray):
            # channels=(3, 4): the optional 4th component is per-vertex opacity,
            # load-bearing in every blending mode — mirrors points/lines.
            validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
        elif isinstance(colors, (list, tuple)):
            validate_broadcast_color(colors, "colors")
    if scalars is not None:
        validate_scalars_preflight(scalars, n_vertices)
    if labels is not None:
        validate_labels_for_writing(labels, n_vertices)
    # 0h. Transform / nd_transform normalization is pure attr processing, so it
    # belongs in the gate too — and prepare_transform_attrs is NOT idempotent
    # (it transposes the matrix), so it must run exactly once.
    prepare_transform_attrs(attrs, ctx.store)

    # Normalize faces to (F, 3) uint32. Safe now, not before: the validator has
    # established an integer dtype and both bounds, and MAX_MESH_VERTICES keeps
    # every admitted index far below 2^32, so the cast is value-preserving.
    faces_arr = np.asarray(faces).reshape(-1, 3).astype(np.uint32, copy=False)
    n_faces = int(faces_arr.shape[0])

    # `shading` resolution: default to smooth only when there are normals to
    # smooth. An explicit value is stamped AS GIVEN and never rewritten — an
    # explicit "smooth" with no stored normals is honoured by the viewer falling
    # back to a derived flat normal at render time, and an explicit "flat" gives a
    # faceted surface even when normals are present.
    resolved_shading = (
        shading
        if shading is not None
        else ("smooth" if normals is not None else "flat")
    )

    # 1. Setup: create the group.
    group = ctx.store.require_group(path)

    aprint(
        f"📝 Writing {n_vertices:,} mesh vertices / {n_faces:,} faces "
        f"({n_dims}D) to {path}"
    )

    if _is_unwelded(faces_arr, n_vertices):
        warning_key = _mesh_authoring_warning_key(ctx, path)
        if ctx.claim_authoring_warning("mesh", warning_key):
            aprint(
                f"  ⚠️ Node '{warning_key}': {n_vertices:,} vertices for "
                f"{n_faces:,} faces with no shared vertex indices — this mesh "
                "looks like independent triangles rather than a welded indexed "
                "surface. Smooth shading is impossible (each vertex belongs to "
                "one face, so there are no neighbours to average across), "
                "per-vertex normals are meaningless, and the vertex array is "
                "~3x larger than needed. Weld coincident vertices and index the "
                "faces into them."
            )

    if colors is not None and isinstance(colors, (list, tuple)):
        aprint(f"  → Uniform color RGB(A){list(colors)} for all vertices")

    # 2. Write vertices (COORDINATE). No spatial ordering: v1 mesh is
    # whole-node-resident, so there is no chunk index to build.
    ctx.dataset_ctx.encoder.encode(
        data=vertices,
        zarr_group=group,
        name="vertices",
        semantic_type=SemanticType.COORDINATE,
        mode=ctx.dataset_ctx.encoding_mode,
        chunks=calculate_intelligent_chunks((n_vertices, n_dims), dtype=vertices.dtype),
        compressor=ctx.dataset_ctx.compressor,
        # Same reasoning as the lines writer's structural arrays: the loader
        # reads vertices/faces as raw chunked zarr without resolving array_ref,
        # so dedup would silently drop geometry for a byte-identical sibling, and
        # LUT encoding of grid-snapped coordinates would decode as garbage.
        deduplicate=False,
        allow_lut=False,
    )

    # 3. Write faces (INDEX). Chunked over triangles.
    ctx.dataset_ctx.encoder.encode(
        data=faces_arr,
        zarr_group=group,
        name="faces",
        semantic_type=SemanticType.INDEX,
        mode=ctx.dataset_ctx.encoding_mode,
        chunks=calculate_intelligent_chunks((n_faces, 3), dtype=faces_arr.dtype),
        compressor=ctx.dataset_ctx.compressor,
        deduplicate=False,  # see the vertices note above
        allow_lut=False,
    )
    aprint(f"  ✓ Wrote faces ({n_faces:,} triangles)")

    metadata: dict[str, Any] = {
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "ndim": n_dims,
        "has_normals": False,
        "has_colors": False,
        "has_scalars": False,
        "shading": resolved_shading,
        "double_sided": bool(double_sided),
    }

    # 4. Optional per-vertex arrays.
    if normals is not None:
        # COORDINATE, not a bounded scalar: it quantizes each component over its
        # own [-1, 1] range (a free 2x over float32) and correctly BLOCKS
        # broadcasting, since a normal is always per-vertex.
        normals_f32 = normals.astype(np.float32, copy=False)
        ctx.dataset_ctx.encoder.encode(
            data=normals_f32,
            zarr_group=group,
            name="normals",
            semantic_type=SemanticType.COORDINATE,
            mode=ctx.dataset_ctx.encoding_mode,
            # `normals_f32.dtype`, not `np.float32`: chunking reads `.itemsize`,
            # which on the numpy *scalar type* is an unbound descriptor rather
            # than a number.
            chunks=calculate_intelligent_chunks(
                (n_vertices, 3), dtype=normals_f32.dtype
            ),
            compressor=ctx.dataset_ctx.compressor,
        )
        metadata["has_normals"] = True
        # Stored as a plain list of ints: a numpy integer is not JSON
        # serializable, and zarr attrs are JSON.
        metadata["normal_dims"] = [int(d) for d in normal_dims]  # type: ignore[union-attr]
        aprint(f"  ✓ Wrote normals (dims {metadata['normal_dims']})")

    if colors is not None:
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
        write_colors(group, colors, None, n_vertices, ctx.dataset_ctx)
        metadata["has_colors"] = True

    if scalars is not None:
        write_scalars(group, scalars, None, n_vertices, ctx.dataset_ctx)
        metadata["has_scalars"] = True

    # Write the colormap LUT when `colormap` is a custom array.
    ctx.write_colormap_lut(group, attrs)

    # 5. Attrs. Transform / nd_transform were already normalized in the gate.
    apply_default_render_attrs(attrs)
    group.attrs.update(attrs)
    group.attrs["type"] = "mesh"
    group.attrs["n_vertices"] = n_vertices
    group.attrs["n_faces"] = n_faces
    group.attrs["ndim"] = n_dims
    # Below attrs.update, like the sibling writers: a user-supplied attrs dict
    # must never clobber the writer's presence truth.
    group.attrs["has_normals"] = metadata["has_normals"]
    group.attrs["has_colors"] = metadata["has_colors"]
    group.attrs["has_scalars"] = metadata["has_scalars"]
    group.attrs["shading"] = resolved_shading
    group.attrs["double_sided"] = bool(double_sided)
    if metadata["has_normals"]:
        group.attrs["normal_dims"] = metadata["normal_dims"]
    # v1 has no spatial index; the key is stamped anyway so a reader never has to
    # distinguish "no ordering" from "attr missing".
    group.attrs["ordering"] = "none"
    metadata["ordering"] = "none"

    position_bounds = compute_position_bounds(vertices)
    group.attrs["position_bounds"] = position_bounds
    metadata["position_bounds"] = position_bounds

    if not attrs.pop("_skip_scene_bounds", False):
        ctx.update_scene_bounds(position_bounds)

    # 6. Labels (CSR). Per-vertex, like the lines writer.
    if labels is not None:
        write_labels_csr(group, labels, n_vertices, ctx.compressor, None)
        metadata["has_labels"] = True
        group.attrs["has_labels"] = True

    if image_labels is not None:
        write_image_labels_csr(group, image_labels, n_vertices, ctx.compressor, None)
        metadata["has_image_labels"] = True
        group.attrs["has_image_labels"] = True

    aprint(f"✅ Mesh written to {path}")

    return metadata
