"""add_mesh body.

Pure function taking a ``group: Group`` parameter as the first arg. Called by
``Group.add_mesh`` (a thin signature + docstring + delegate) in
``core/group/group.py``.

Still shorter than its siblings, and structurally so: mesh has a ``partition``
branch but no ``additive_lod`` / ``substitutive_lod`` one, because those paths do
not exist for a mesh — for two different reasons, spelled out in the rejections
below and in MESH_NODE_SPEC.md §9. It also keeps one rejection those adders never
need, since a ``kind=lod`` parent is the one way a mesh could still end up
somewhere it cannot be rendered.

The partition branch differs from the sibling adders' in the one way that
matters: theirs hand each part a SLICE of the element arrays, because their
elements are independent rows. A triangle is not a row — it is three references
into a shared vertex table — so a mesh part is a re-indexing, not a slice, and
the split lives in :mod:`luxar.mesh.split`.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ...mesh import Mesh
from ..compositing import (
    COMPOSITING_ATTRS,
    position_bounds_from_array,
    slice_optional_array,
    sync_custom_colormap_attr,
)
from ..dim_order import apply_dim_order_positions

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def _reject_specialized_parent(parent_node: "Node", name: str) -> None:
    """Refuse to write a mesh leaf under a ``kind=lod`` group.

    Mesh has no LOD ladder (spec §9), and the guard that would otherwise catch it
    resolves the display type from the children at FINALIZE — i.e. after the
    mesh's arrays are already on disk. Catching it here keeps the failure
    fail-fast and leaves no partial node. It is a caller mistake with no valid
    interpretation, so it raises rather than warns.

    The two LOD flavours are refused for DIFFERENT reasons and the message says so
    (spec §9). The additive prefix ladder is excluded on principle — a prefix of an
    index buffer is a *holed* surface, not a coarse one, which is why it degrades
    gracefully for independent elements and produces a wrong picture here.
    Substitutive levels are excluded only for want of a producer: that machinery
    makes no independence assumption at all, since a level is an
    independently-authored ``(vertices, faces)`` pair chosen by
    ``coverage_fraction``. Conflating the two (as this message once did) tells a
    user the feature is impossible when it is merely unwritten.

    ``kind=partition`` is NOT refused any more. It used to be, on the grounds that
    a BSP cut runs through faces — which is true, and is exactly what
    ``add_mesh(partition=...)`` now handles by duplicating boundary vertices, so
    the mesh children it writes must be allowed through here.
    """
    kind = parent_node.attrs.get("kind")
    if kind == "lod":
        raise ValueError(
            f"Cannot add mesh '{name}' to a kind=lod group. Mesh has no LOD "
            "ladder yet, for two different reasons: an ADDITIVE prefix ladder "
            "cannot apply at all (a prefix of an index buffer is a surface with "
            "holes in it, not a coarser surface), while SUBSTITUTIVE levels are "
            "structurally fine and simply have no producer — mesh decimation does "
            "not exist yet. Add the mesh to a plain group instead."
        )
    # No ``kind == "partition"`` arm: mesh IS partition-capable now. A mesh leaf
    # under a kind=partition group is exactly what ``add_mesh(partition=...)``
    # writes, so refusing it here would refuse this adder's own output.


# Reason per structural parameter the sibling adders take and mesh does not. Same
# three exclusions as the parent-group refusals above and as ``Group.add_mesh``'s
# docstring, worded for a caller who passed the knob; insertion order decides which
# one a multi-parameter call is told about.
_UNSUPPORTED_STRUCTURE_PARAMS: Dict[str, str] = {
    "additive_lod": (
        "an ADDITIVE prefix ladder cannot apply at all — a prefix of an index "
        "buffer is a surface with holes in it, not a coarser surface"
    ),
    "substitutive_lod": (
        "SUBSTITUTIVE levels are structurally fine and simply have no producer — "
        "mesh decimation does not exist yet"
    ),
    # NOTE: ``partition`` is deliberately absent — it is a real ``add_mesh``
    # parameter now, so it never reaches ``**attrs``.
}


def _reject_structure_params(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse the sibling adders' ``additive_lod`` / ``substitutive_lod``.

    ``add_mesh`` has no such parameters, so a caller who passes one lands in
    ``**attrs`` and gets the generic UNKNOWN-attribute rejection from
    ``validate_render_attrs`` — "The viewer would silently ignore it… Remove it or
    use a supported attribute". That is a typo diagnostic, and this caller made no
    typo: they asked for a real feature the other three geometry types have, by
    its real name.

    The refusal is correct either way; only its stated reason was wrong, which is the
    same defect the parent-group message above carried. Answering with the per-flavour
    reason (spec §9) is what tells a user whether to wait for the feature.

    ``partition`` is no longer in this table — it is a real ``add_mesh``
    parameter, so it is bound by name and never reaches ``**attrs``.
    """
    for key, reason in _UNSUPPORTED_STRUCTURE_PARAMS.items():
        if key in attrs:
            raise ValueError(
                f"Cannot add mesh '{name}' with '{key}'. That is a structural "
                "parameter of the sibling adders and mesh has no such path yet: "
                f"{reason} (spec §9). Write the mesh as a plain leaf instead."
            )


def _reject_volumetric_blending(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse ``blending_mode='volumetric'`` on a mesh (spec §9).

    The other §9 exclusions are refused already — LOD by
    :func:`_reject_specialized_parent`, the additive ladder by the viewer's
    progressive-loader factory — but this one was documented and never enforced, so
    a volumetric mesh wrote and loaded cleanly.

    It cannot mean anything. Volumetric blending is emission-absorption integration
    through a participating medium: the shader scales each element's contribution by
    its extent along the view ray and maps ``absorption`` into optical depth over
    that path length. A triangle is a zero-thickness surface, so its path length is
    identically zero and there is no medium to absorb anything — the mode has no
    per-element quantity to integrate. That makes it a caller mistake with no valid
    interpretation, which is the same bar the LOD/partition refusals are held to, so
    it raises rather than warns.

    Refused at the ADDER rather than in ``validate_blending_mode``, which is
    deliberately geometry-agnostic and shared by all four types.
    """
    if attrs.get("blending_mode") == "volumetric":
        raise ValueError(
            f"Cannot add mesh '{name}' with blending_mode='volumetric'. Volumetric "
            "blending integrates emission and absorption along the view ray through a "
            "participating medium, and a triangle is a zero-thickness surface: its "
            "path length through the medium is zero, so there is nothing for "
            "'absorption' to attenuate. Use 'normal' for an opaque surface, or "
            "'additive' for a translucent one."
        )


def _reject_energy_stamps(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse hand-supplied LOD quality stamps on a mesh (spec §9.1).

    ``level_stats`` / ``lod_stats`` carry the ``energy_fraction_cum`` and
    ``reference_energy`` pair that an additive ladder writes. The allow-list in
    ``io/_compiler/node_common.py`` is geometry-blind, so a caller can set them on a
    mesh today and the node writes clean.

    The hazard is on the viewer side. ``energyCompensation`` multiplies a leaf's
    brightness by ``1/e(k)`` while a ladder is incomplete, and it is gated on
    ``BLENDABLE_MODES = {additive, luminous, volumetric}`` — **not** on geometry
    type. Mesh supports ``additive`` and ``luminous``, so a stamped mesh in either
    mode would be brightened. That is right for a splat prefix, which really is a
    dimmer version of the whole; it is backwards for a surface, where a partial draw
    is a *holed* picture at full brightness. Mesh's ``opaque`` default escapes it
    today by luck, not design.

    This is deliberately prophylactic rather than a fix for a live bug: mesh cannot
    currently be in a ``kind=lod`` group (so the fade pass never visits it) and the
    mesh commit never stamps ``committedEnergyFraction`` (so the compensation
    factor is 1). Substitutive LOD would remove the first latch and a reveal ladder
    the second. Refusing now means §9.1's "must NOT carry energy stamps" rule is
    enforced before either lands, rather than being a note someone has to remember.

    A reveal ladder (§9.1) is expected to arrive as a face ORDER plus a reveal
    fraction on a single leaf, which needs neither key — so this refusal does not
    stand in its way. Substitutive mesh levels are the one thing it would touch:
    ``level_stats`` also carries the non-energy ``quality`` stamp a level may
    legitimately want, so whoever lands the decimator narrows this to the energy keys
    rather than working around it.

    Until then the check is on KEY PRESENCE, deliberately broader than the energy
    fields themselves: neither attribute has anything to say about a mesh today, so
    there is no value worth inspecting, and refusing the container is the rule §9.1
    states. The message says which attribute is refused and what it is FOR — it does
    not claim the supplied dict actually holds a stamp.
    """
    supplied = sorted(k for k in ("level_stats", "lod_stats") if k in attrs)
    if supplied:
        raise ValueError(
            f"Cannot add mesh '{name}' with {' and '.join(supplied)}. That is where "
            "an additive ladder writes its energy stamps, and the viewer's brightness "
            "compensation is gated on the BLENDING MODE, not the geometry type — "
            "so a stamped mesh in 'additive' or 'luminous' would be scaled by "
            "1/energy_fraction_cum. That brightens a dimmer prefix correctly and a "
            "holed surface wrongly (spec §9.1). Mesh has no additive ladder, so "
            "neither attribute has anything to say about one; omit them."
        )


def add_mesh_impl(
    group: "Group",
    *,
    name: str,
    vertices: Any,
    faces: Any,
    normals: Any = None,
    normal_dims: Optional[Sequence[int]] = None,
    colors: Any = None,
    scalars: Any = None,
    shading: Optional[str] = None,
    double_sided: bool = True,
    labels: Optional[Union[List[str], Sequence[str]]] = None,
    image_labels: Optional[Any] = None,
    partition: Any = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    **attrs: Any,
) -> Union[Mesh, "Group"]:
    try:
        # Fail-fast pre-write gate: reject invalid names (empty/'/'/dot-prefixed —
        # an empty name resolves to the zarr ROOT group and would clobber the
        # scene root) and duplicate siblings BEFORE any zarr write. Node.__init__
        # re-checks both post-write (belt and braces).
        from ....validation.base import validate_node_name

        validate_node_name(name)
        (parent or group)._ensure_no_duplicate_child(name)
        _reject_specialized_parent(parent or group, name)
        _reject_structure_params(name, attrs)
        _reject_volumetric_blending(name, attrs)
        _reject_energy_stamps(name, attrs)

        scene = group._find_scene()

        vert_arr: np.ndarray = (
            vertices if isinstance(vertices, np.ndarray) else np.asarray(vertices)
        )
        if vert_arr.ndim != 2:
            raise ValueError(
                f"Vertices must have shape (V, D), got shape {vert_arr.shape}"
            )
        # A floor of 2 dimensions, which the sibling adders deliberately do NOT have.
        # Points and Lines are meaningful in 1D — a scatter along an axis, segments with
        # length — so they take whatever width they are given. A TRIANGLE needs two
        # dimensions to enclose any area: in 1D every face is collinear, so the mesh
        # writes and loads successfully and then renders nothing at all, with no
        # diagnostic anywhere. Refusing at the adder is the only place that can say why.
        if vert_arr.shape[1] < 2:
            raise ValueError(
                f"Vertices must have at least 2 dimensions, got shape {vert_arr.shape}. "
                "A triangle needs two dimensions to have any area — in 1D every face is "
                "collinear and the surface renders nothing. Use Points or Lines for "
                "1D data."
            )

        # Apply dim_order before validation. Vertices are coordinates and get
        # reordered like every other geometry type's positions; `faces` is INDEX
        # data addressing vertex ROWS, so it is deliberately NOT reordered —
        # permuting columns of the coordinate array leaves row indices valid.
        vert_arr, extend_to_all = apply_dim_order_positions(
            vert_arr, scene, dim_order, fill, extend_to_all
        )

        n_vertices = vert_arr.shape[0]
        ndim = vert_arr.shape[1]

        # Colormap / colors mutual exclusivity, matching the sibling adders.
        if colors is not None and attrs.get("colormap") is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )
        if scalars is not None and attrs.get("colormap") is None:
            raise ValueError(
                "'scalars' requires a 'colormap' attribute to map values to colors."
            )

        faces_arr: np.ndarray = (
            faces if isinstance(faces, np.ndarray) else np.asarray(faces)
        )
        n_faces = int(faces_arr.size // 3)

        aprint(
            f"Adding mesh node '{name}' with {n_vertices:,} vertices and "
            f"{n_faces:,} faces in {ndim}D."
        )

        scene._validate_data_dimensions(vert_arr, name, data_type="vertices")

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, vert_arr, "mesh"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        parent_node = parent or group

        # Spatial partition — split the surface into independently drawable
        # parts under a kind=partition wrapper, so the viewer can frustum-cull
        # per part. Placed here, after dim_order/extend_to_all resolution, so
        # every part inherits coordinates and visibility already in final form
        # and the per-part recursion must not re-apply them.
        if partition is not None and partition is not False:
            wrapper = _add_mesh_partition(
                group,
                name=name,
                vert_arr=vert_arr,
                faces_arr=faces_arr,
                partition=partition,
                normals=normals,
                normal_dims=normal_dims,
                colors=colors,
                scalars=scalars,
                shading=shading,
                double_sided=double_sided,
                labels=labels,
                image_labels=image_labels,
                parent_node=parent_node,
                extend_to_all=extend_to_all,
                **attrs,
            )
            if wrapper is not None:
                return wrapper
            # 1 part → fall through to the plain single-leaf write, exactly as
            # the sibling adders do. A wrapper around one part is pure overhead.

        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        metadata = writer.write_mesh(
            path,
            vert_arr.astype(np.float32),
            faces_arr,
            normals=normals,
            normal_dims=normal_dims,
            colors=cast(Any, colors),
            scalars=scalars,
            shading=shading,
            double_sided=double_sided,
            labels=labels,
            image_labels=image_labels,
            **attrs,
        )

        # Sync colormap attr with what the compiler wrote to zarr
        sync_custom_colormap_attr(attrs)

        if labels is not None:
            scene._notify_labels_added()
        if image_labels is not None:
            scene._notify_image_labels_added()

        return Mesh(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )
    except (ValueError, TypeError) as e:
        aprint(f"Failed to add mesh node '{name}': {e}")
        raise ValueError(f"Could not add mesh '{name}': {e}") from e


def _resolve_mesh_partition(partition: Any) -> tuple[int, str]:
    """Validate ``partition=`` and return ``(max_elements, rule)``.

    Same vocabulary as the sibling adders (``True`` / ``{max_elements, rule}``)
    so a caller who knows ``add_points(partition=...)`` already knows this one.

    ``max_elements`` counts **faces**, not vertices. The BSP recurses on face
    centroids — one triangle is one indivisible unit of the split — so faces are
    the quantity the cap can actually bound. A part's vertex count is whatever
    its faces reference (at most ``3 * max_elements``, in practice far less).
    """
    from ..partition import DEFAULT_MAX_ELEMENTS

    if partition is True:
        return DEFAULT_MAX_ELEMENTS, "median"
    if isinstance(partition, dict):
        max_elements = int(partition.get("max_elements", DEFAULT_MAX_ELEMENTS))
        if max_elements < 1:
            raise ValueError(f"partition max_elements must be >= 1, got {max_elements}")
        rule = str(partition.get("rule", "median"))
        if rule not in ("median", "midpoint", "sah"):
            raise ValueError(
                f"partition rule must be 'median', 'midpoint', or 'sah'; got {rule!r}"
            )
        return max_elements, rule
    raise TypeError(
        f"partition must be None, True, or dict; got {type(partition).__name__}"
    )


def _add_mesh_partition(
    group: "Group",
    *,
    name: str,
    vert_arr: np.ndarray,
    faces_arr: np.ndarray,
    partition: Any,
    normals: Any,
    normal_dims: Optional[Sequence[int]],
    colors: Any,
    scalars: Any,
    shading: Optional[str],
    double_sided: bool,
    labels: Any,
    image_labels: Any,
    parent_node: "Node",
    extend_to_all: Optional[Union[List[str], str]],
    **attrs: Any,
) -> Optional["Group"]:
    """Write a kind=partition wrapper with one independent Mesh child per part.

    Returns ``None`` when the BSP could not split into more than one part, so the
    caller falls through to a plain single-leaf write.

    Unlike the sibling wrappers this cannot slice its inputs: a part's faces
    reference a shared vertex table, so each part gathers and renumbers its own
    vertices (see :mod:`luxar.mesh.split`). Boundary vertices are therefore
    duplicated across parts — that is what makes each part independently
    drawable, and it is invisible in the render because both copies carry
    identical position and identical stored normal.

    Per-VERTEX attributes (``normals`` / ``colors`` / ``scalars`` / ``labels``)
    are gathered through the part's ``vertex_index``; per-face data has no
    attribute today.
    """
    from ....mesh.split import duplication_factor, face_centroids, split_mesh_by_faces
    from ..partition import (
        median_bsp_partition,
        midpoint_bsp_partition,
        sah_bsp_partition,
        warn_if_oversized_single_part,
    )

    max_elements, rule = _resolve_mesh_partition(partition)

    # No `warn_if_partition_needs_more_dims` call: it guards against <2 spatial
    # dims, and add_mesh_impl has already refused those outright with a
    # mesh-specific message (a 1D triangle encloses no area). Unreachable here.

    if image_labels is not None:
        raise ValueError(
            "image_labels is not supported alongside partition=. It is a "
            "whole-node image-to-label mapping with no per-part meaning, and "
            "splitting it would silently change what each part's labels index. "
            "Decompose manually or omit image_labels."
        )

    faces2d = faces_arr.reshape(-1, 3)
    centroids = face_centroids(vert_arr, faces2d)
    if rule == "sah":
        face_parts = sah_bsp_partition(centroids, max_elements)
    elif rule == "midpoint":
        face_parts = midpoint_bsp_partition(centroids, max_elements)
    else:
        face_parts = median_bsp_partition(centroids, max_elements)

    warn_if_oversized_single_part(
        len(face_parts),
        int(face_parts[0].size) if face_parts else 0,
        max_elements,
        name,
    )
    if len(face_parts) <= 1:
        return None

    parts = split_mesh_by_faces(faces2d, face_parts)
    n_vertices = int(vert_arr.shape[0])

    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    leaf_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}

    wrapper = parent_node.add_partition_group(
        name=name,
        display_type="mesh",
        max_elements=max_elements,
        **wrapper_attrs,
    )

    aprint(
        f"  ✂️  Partitioned mesh '{name}' into {len(parts)} parts via BSP "
        f"(max_elements={max_elements:,} faces, "
        f"face counts={[int(p.faces.shape[0]) for p in parts]}, "
        f"vertex duplication x{duplication_factor(parts, n_vertices):.3f})"
    )

    for i, part in enumerate(parts):
        # `slice_optional_array` is the same helper the sibling wrappers use, and
        # it is the right one here for the same reason: it gathers ONLY when the
        # value's leading length matches the element count, so a per-vertex array
        # follows its vertices while a uniform RGB triple or a colormap name is
        # handed to every part untouched. `labels` is per-VERTEX (hover tooltips),
        # so it is gathered too — passing it whole would give every part V labels
        # for its own Vi vertices.
        take = part.vertex_index
        wrapper.add_mesh(
            name=f"part_{i}",
            vertices=vert_arr[take],
            faces=part.faces,
            normals=slice_optional_array(normals, take, n_vertices),
            normal_dims=normal_dims,
            colors=slice_optional_array(colors, take, n_vertices),
            scalars=slice_optional_array(scalars, take, n_vertices),
            shading=shading,
            double_sided=double_sided,
            labels=slice_optional_array(labels, take, n_vertices),
            image_labels=None,
            extend_to_all=extend_to_all,
            # dim_order / fill were applied to vert_arr upstream — re-applying
            # per part would permute already-permuted coordinates.
            dim_order=None,
            fill=None,
            # Explicit no-partition, so a part can never recurse into another
            # partition (mirrors the `False` sentinel the sibling wrappers use).
            partition=False,
            **leaf_attrs,
        )

    # Union of the parts' bounds == the whole input's bounds, computed straight
    # from the source rather than round-tripped through the children's attrs.
    wrapper._persist_attr("position_bounds", position_bounds_from_array(vert_arr))
    return wrapper
