"""Mesh write pipeline (body of ``LuxarZarrCompiler.write_mesh``).

Structurally the simplest of the four geometry writers, and deliberately so.
Mesh has no spatial ordering (``ordering`` is always ``"none"`` in v1 — the
loader is whole-node, so there is nothing for a chunk index to skip), no primary
size scalar (a triangle's extent comes from its own vertices, not a per-element
radius/width/covariance), and neither an LOD nor a partition path *of its own*:
``add_mesh(substitutive_lod=...)`` decimates and ``add_mesh(partition=...)``
splits the surface upstream, and this writer just sees one independent leaf per
level or per part. (``write_mesh_multi_lod`` routes its additive levels through
this same writer too, one call per level — but a prefix of an index buffer is a
holed surface, not a coarser one, so what it writes is a REVEAL ladder, not an
LOD; see ``MESH_NODE_SPEC.md`` §9.) What remains is: encode
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
from ..dataset_writers.texture import write_texture
from ..labels.image_labels import (
    validate_image_labels_for_writing,
    write_image_labels_csr,
)
from ..labels.text_labels import write_string_channels_csr
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

#: Accepted shading values, public so the core read type can stay pinned to the writer.
VALID_SHADING_MODES = ("smooth", "flat", "none")


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

    A partitioned mesh writes one leaf per part, so the lint must warn once for
    the whole object rather than once per leaf. Mirrors the lines writer so the
    two stay diffable.
    """
    parent_path, separator, _leaf = path.rpartition("/")
    if separator and parent_path in ctx.store:
        parent = ctx.store[parent_path]
        if getattr(parent, "attrs", {}).get("kind") == "partition":
            return parent_path
    return path


def _validate_normal_pair(
    normals: Any, normal_dims: Any, n_vertices: int, n_dims: int
) -> None:
    """Normals and their companion attr are a PAIR — each is meaningless alone.

    A normal array with no ``normal_dims`` cannot be oriented (storing normals
    against an implicit "first three dimensions" is the bug the attr exists to
    prevent: for a ``(t, x, y, z)`` mesh those are ``(t, x, y)``), and
    ``normal_dims`` with no normals describes nothing.
    """
    from ....validation.base import (
        validate_normal_dims_for_writing,
        validate_normals_for_writing,
    )

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


def _validate_mesh_metadata(shading: Optional[str], double_sided: Any) -> None:
    """Validate mesh metadata that directly controls viewer rendering."""
    # A typo must not reach zarr: an unrecognised shading value would silently
    # take the stored-normal path.
    if shading is not None and shading not in VALID_SHADING_MODES:
        raise ValueError(
            f"shading must be one of {VALID_SHADING_MODES}, got {shading!r}"
        )
    if not isinstance(double_sided, bool):
        raise ValueError(
            f"double_sided must be a bool, got {type(double_sided).__name__}"
        )


def _write_mesh_texture_arrays(
    group: Any,
    ctx: GeometryWriteCtx,
    n_vertices: int,
    uvs: Optional[NDArray[np.float32]],
    texture: Optional[NDArray[Any]],
    texture_encoding: str,
    texture_width: Optional[int],
    texture_height: Optional[int],
    texture_channels: Optional[int],
    texture_color_space: str,
    texture_ktx2_mode: str,
    texture_ktx2_quality: Optional[int],
    texture_ktx2_rdo_l: Optional[float],
    texture_ktx2_zcmp: Optional[int],
    encoded_ktx2: Optional[NDArray[np.uint8]],
) -> dict[str, Any]:
    """Write the UV and texture arrays, and return the attrs they imply.

    Extracted so :func:`write_mesh` gains no branches for them — it sat at the
    complexity ratchet's limit, and three more `if`s tipped it over. The two
    arrays travel together because they are a pair the adder already refuses
    apart, so there is no caller that wants one function and not the other.

    Returns the presence flags always, and the dimension attrs only alongside a
    texture, so the caller can stamp whatever comes back without deciding again.
    """
    meta: dict[str, Any] = {"has_uvs": uvs is not None, "has_texture": False}

    if uvs is not None:
        # COORDINATE, like `normals`: a per-vertex 2-vector that quantizes over
        # its own range and must never broadcast. NOT a bounded scalar — a UV is
        # deliberately allowed outside [0, 1] so `texture_wrap="repeat"` can tile.
        uvs_f32 = np.asarray(uvs).astype(np.float32, copy=False)
        ctx.dataset_ctx.encoder.encode(
            data=uvs_f32,
            zarr_group=group,
            name="uvs",
            semantic_type=SemanticType.COORDINATE,
            mode=ctx.dataset_ctx.encoding_mode,
            chunks=calculate_intelligent_chunks((n_vertices, 2), dtype=uvs_f32.dtype),
            compressor=ctx.dataset_ctx.compressor,
        )
        aprint(f"  ✓ Wrote uvs ({n_vertices:,} texture coordinates)")

    if texture is not None:
        tex_h, tex_w, tex_c = write_texture(
            group,
            texture,
            texture_encoding,
            texture_width,
            texture_height,
            texture_channels,
            texture_color_space,
            ctx.dataset_ctx,
            texture_ktx2_mode,
            texture_ktx2_quality,
            texture_ktx2_rdo_l,
            texture_ktx2_zcmp,
            encoded_ktx2,
        )
        meta.update(
            has_texture=True,
            texture_encoding=texture_encoding,
            # Stamped unconditionally, including for `raw` where they are
            # redundant with the array shape: a reader must never have to open the
            # payload to learn how large it decodes to.
            texture_height=tex_h,
            texture_width=tex_w,
            texture_channels=tex_c,
            texture_color_space=texture_color_space,
        )
    return meta


def validate_mesh_arrays(
    vertices: Any,
    faces: Any,
    *,
    normals: Any = None,
    normal_dims: Any = None,
    colors: Any = None,
    scalars: Any = None,
    uvs: Any = None,
    texture: Any = None,
    texture_encoding: str = "raw",
    texture_width: Optional[int] = None,
    texture_height: Optional[int] = None,
    texture_channels: Optional[int] = None,
    texture_color_space: str = "srgb",
    texture_ktx2_mode: str = "uastc",
    texture_ktx2_quality: Optional[int] = None,
    texture_ktx2_rdo_l: Optional[float] = None,
    texture_ktx2_zcmp: Optional[int] = None,
    shading: Optional[str] = None,
    double_sided: bool = True,
    labels: Any = None,
    image_labels: Any = None,
    keys: Any = None,
) -> Tuple[int, int]:
    """Validate a mesh's arrays and channels. Pure — reads nothing, writes nothing.

    Steps 0c-0h of :func:`write_mesh`'s fail-fast gate, factored out because a
    SECOND caller needs exactly them and nothing else:
    ``add_mesh(substitutive_lod=…)`` runs this before it decimates anything and
    before ``add_lod_group`` creates the zarr group. Without that, a malformed
    optional channel that the decimator ALSO validates per level — colours
    with two components, a wrong-length normals array, a typo'd ``shading`` —
    was refused only from inside whichever child's write hit it first
    (typically ``child_0``, the coarsest, since levels are written
    coarsest-to-finest), with the ``kind=lod`` group already on disk: that
    left a childless ladder if the very first child failed, or a
    finest-child-less one if a later level was the one to fail. The
    plain-leaf path writes nothing in the same situation, and the two must
    agree.

    ``labels`` and a wrong-length ``image_labels`` (#1491) fail a DIFFERENT
    way: both are forwarded ONLY to the FINEST child (the original,
    undecimated surface), written LAST, never to a coarse level. Pre-fix, a
    bad one of these was refused deep inside that finest child's own
    ``write_mesh`` call — AFTER its vertices, faces, normals, colours and
    scalars were already on disk — so the ladder was neither childless nor
    finest-child-less: every level, including the finest, was fully written
    and independently loadable. Only the finest level's own label channel
    (the text-label CSR pair, or ``image_label_offsets`` /
    ``image_label_bytes``) was silently absent. That is harder to notice than
    a missing level, not milder — the ladder looks and loads like a
    correctly-authored object that simply carries no labels.

    Sharing the function rather than repeating the checks is what keeps that
    promise true as the rules change: the ladder gate cannot drift from what the
    child write accepts, because it IS what the child write runs.

    Returns:
        ``(n_vertices, n_dims)``, since the caller needs both and only the
        vertex validator can produce them.
    """
    from ....validation.base import (
        _texture_decoded_bytes,
        validate_colors_for_writing,
        validate_faces_for_writing,
        validate_labels_for_writing,
        validate_mesh_decode_budget,
        validate_positions_for_writing,
        validate_texture_for_writing,
        validate_uvs_for_writing,
        validate_vertices_for_writing,
    )

    # Vertices shape/finiteness (shared coordinate path), then the mesh-only
    # vertex-count ceiling. Order matters: the cap reads shape[0], which is only
    # meaningful once the array is known to be 2D.
    n_vertices, n_dims = validate_positions_for_writing(vertices, context="vertices")
    validate_vertices_for_writing(vertices)
    # Faces: layout, integer dtype, and both index bounds. Runs before the
    # writer's uint32 cast, which is what makes the bounds check meaningful.
    validate_faces_for_writing(faces, n_vertices)
    _validate_normal_pair(normals, normal_dims, n_vertices, n_dims)
    if uvs is not None:
        validate_uvs_for_writing(uvs, n_vertices)
    _validate_mesh_metadata(shading, double_sided)
    # Optional per-vertex channels.
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
    # Keys ride the same pre-flight as labels: the CSR serializer
    # UTF-8-encodes each entry, so a non-str or a length mismatch must be
    # caught BEFORE any array reaches disk (#1917).
    if keys is not None:
        validate_labels_for_writing(keys, n_vertices, context="keys", noun="Keys")
    if image_labels is not None:
        validate_image_labels_for_writing(image_labels, n_vertices)
    # Last, because it needs every channel's presence and the validated shapes.
    # A store over the viewer's per-node ceiling does not render at all, so it is
    # refused here rather than in a browser (#2145).
    texture_decoded_bytes = 0
    if texture is not None:
        texture_height, texture_width, texture_channels = validate_texture_for_writing(
            texture,
            texture_encoding,
            texture_width,
            texture_height,
            texture_channels,
            texture_color_space,
            ktx2_mode=texture_ktx2_mode,
            ktx2_quality=texture_ktx2_quality,
            ktx2_rdo_l=texture_ktx2_rdo_l,
            ktx2_zcmp=texture_ktx2_zcmp,
        )
        texture_decoded_bytes = _texture_decoded_bytes(
            texture_encoding, texture_width, texture_height, texture_channels
        )
    validate_mesh_decode_budget(
        n_vertices,
        n_dims,
        int(np.asarray(faces).size // 3),
        normals=normals,
        colors=colors,
        scalars=scalars,
        uvs=uvs,
        texture_decoded_bytes=texture_decoded_bytes,
    )
    return n_vertices, n_dims


def write_mesh(
    ctx: GeometryWriteCtx,
    path: NodePath,
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    normals: Optional[NDArray[np.float32]] = None,
    normal_dims: Optional[Sequence[int]] = None,
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]] = None,
    scalars: Optional[Union[NDArray[np.float32], float]] = None,
    uvs: Optional[NDArray[np.float32]] = None,
    texture: Optional[NDArray[Any]] = None,
    texture_encoding: str = "raw",
    texture_width: Optional[int] = None,
    texture_height: Optional[int] = None,
    texture_channels: Optional[int] = None,
    texture_color_space: str = "srgb",
    texture_ktx2_mode: str = "uastc",
    texture_ktx2_quality: Optional[int] = None,
    texture_ktx2_rdo_l: Optional[float] = None,
    texture_ktx2_zcmp: Optional[int] = None,
    shading: Optional[str] = None,
    double_sided: bool = True,
    labels: Optional["Sequence[str]"] = None,
    image_labels: Optional[Any] = None,
    keys: Optional["Sequence[str]"] = None,
    **attrs: Any,
) -> dict[str, Any]:
    """Write mesh data to Zarr (see ``LuxarZarrCompiler.write_mesh``).

    Returns the node metadata; the caller records it in the metadata cache.
    """
    from ....validation.base import validate_colors_for_writing

    # 0. Fail-fast pre-write gate: everything here runs BEFORE the zarr group is
    # created and before any array lands on disk, so an invalid input cannot
    # leave a partial node behind. Same best-effort caveat as the sibling
    # writers: validators needing the store (custom colormap LUT resolution)
    # still run post-write. image_labels' LENGTH/index and its per-item TYPE
    # dispatch — including the (H,W[,3|4]) ndarray-shape check, which
    # check_image_label_type validates eagerly since ndim/shape[2] need no PIL
    # round-trip — now run here too (step 0h, below, inside validate_mesh_arrays,
    # via validate_image_labels_for_writing / check_image_label_type); only
    # normalize_image_label's actual blob normalization still runs post-write
    # — reading a str/Path file and the PIL encode itself (including the
    # Pillow-not-installed ImportError, which the adder's
    # except (ValueError, TypeError) funnel does not catch either).
    #
    # 0a. Pure attr validators + reserved writer-stamp collisions — after
    # consuming the one INTERNAL key mesh takes: the explicit display window an
    # LOD level stamps in place of its own contracted min/max. Popped before the
    # validator rather than declared in the shared `_ALLOWED_NODE_ATTRS`,
    # because that list is global: allowing it there let the key through
    # `write_points` / `write_lines` / `write_group`, none of which pop it, and
    # it landed on disk as a private attr sitting next to the range it exists to
    # replace. Only mesh consumes it, so only mesh admits it; every other writer
    # gives the ordinary unknown-attribute rejection.
    scalar_bounds = attrs.pop("_scalar_data_range", None)
    validate_render_attrs(attrs, reserved_attrs=MESH_RESERVED_ATTRS)
    # 0b. Node path: an empty path would resolve require_group("") to the scene
    # ROOT and clobber it.
    path = validate_node_path(path)
    # 0c-0h. Vertices, faces, and every optional channel — one shared gate,
    # because `add_mesh(substitutive_lod=…)` runs the very same function before
    # it creates the `kind=lod` group. See :func:`validate_mesh_arrays`.
    n_vertices, n_dims = validate_mesh_arrays(
        vertices,
        faces,
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
        uvs=uvs,
        texture=texture,
        texture_encoding=texture_encoding,
        texture_width=texture_width,
        texture_height=texture_height,
        texture_channels=texture_channels,
        texture_color_space=texture_color_space,
        texture_ktx2_mode=texture_ktx2_mode,
        texture_ktx2_quality=texture_ktx2_quality,
        texture_ktx2_rdo_l=texture_ktx2_rdo_l,
        texture_ktx2_zcmp=texture_ktx2_zcmp,
        shading=shading,
        double_sided=double_sided,
        labels=labels,
        image_labels=image_labels,
        keys=keys,
    )
    encoded_ktx2: Optional[NDArray[np.uint8]] = None
    if texture is not None and texture_encoding == "ktx2":
        from ..dataset_writers.texture import _encode_ktx2

        encoded_ktx2 = _encode_ktx2(
            np.asarray(texture),
            texture_ktx2_mode,
            texture_ktx2_quality,
            texture_ktx2_rdo_l,
            texture_ktx2_zcmp,
            texture_color_space,
        )
    # 0i. Transform / nd_transform normalization is pure attr processing, so it
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
    #
    # "none" is never a DEFAULT, only ever explicit. It suppresses the diffuse and
    # specular terms entirely, so the base colour reaches the screen unmodulated —
    # what a data basemap wants (a textured globe whose colours must stay faithful),
    # and what every other Luxar geometry type does, since the other three are
    # purely emissive. Defaulting to it would silently un-light every existing mesh.
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

    texture_attrs = _write_mesh_texture_arrays(
        group,
        ctx,
        n_vertices,
        uvs,
        texture,
        texture_encoding,
        texture_width,
        texture_height,
        texture_channels,
        texture_color_space,
        texture_ktx2_mode,
        texture_ktx2_quality,
        texture_ktx2_rdo_l,
        texture_ktx2_zcmp,
        encoded_ktx2,
    )
    metadata.update(texture_attrs)

    if colors is not None:
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
        write_colors(group, colors, None, n_vertices, ctx.dataset_ctx)
        metadata["has_colors"] = True

    if scalars is not None:
        write_scalars(
            group,
            scalars,
            None,
            n_vertices,
            ctx.dataset_ctx,
            bounds=scalar_bounds,
        )
        metadata["has_scalars"] = True

    # Write the colormap LUT when `colormap` is a custom array.
    ctx.write_colormap_lut(group, attrs)

    # 5. Attrs. Transform / nd_transform were already normalized in the gate.
    # POPPED BEFORE the attrs land, not after. `_skip_scene_bounds` is private
    # plumbing between the ladder writers and this one — it says "the parent will
    # aggregate the bbox, do not do it per level" — and popping it below the
    # `group.attrs.update(attrs)` wrote it to disk on every sub-LOD of every
    # ladder. Harmless to a reader that ignores unknown keys, but it is an
    # internal flag in the on-disk format, and it round-trips: a tool that reads
    # a level's attrs and re-writes them hands it back as a caller attr.
    skip_scene_bounds = bool(attrs.pop("_skip_scene_bounds", False))
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
    # Every key the texture helper returned, verbatim — `update`, not a loop with
    # a membership test, because the helper already decided what applies. It
    # returns `has_uvs` / `has_texture` always, and the DIMENSION attrs only
    # alongside a texture, for the same reason `normal_dims` is conditional: a
    # stray dimension attr with no payload would be handed on as if it described
    # something, and here it would additionally be CHARGED, since the viewer
    # budgets a node from these numbers before it fetches anything.
    group.attrs.update(texture_attrs)
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

    if not skip_scene_bounds:
        ctx.update_scene_bounds(position_bounds)

    write_string_channels_csr(
        group,
        labels=labels,
        keys=keys,
        n_elements=n_vertices,
        compressor=ctx.compressor,
        sort_order=None,
        metadata=metadata,
    )

    if image_labels is not None:
        write_image_labels_csr(group, image_labels, n_vertices, ctx.compressor, None)
        metadata["has_image_labels"] = True
        group.attrs["has_image_labels"] = True

    aprint(f"✅ Mesh written to {path}")

    return metadata
