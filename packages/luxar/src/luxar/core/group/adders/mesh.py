"""add_mesh body.

Pure function taking a ``group: Group`` parameter as the first arg. Called by
``Group.add_mesh`` (a thin signature + docstring + delegate) in
``core/group/group.py``.

Shorter than its siblings, and structurally so. Mesh has a ``substitutive_lod``
branch like the other three, but no ``additive_lod`` and no ``partition``, for two
different reasons spelled out in the rejections below and in MESH_NODE_SPEC.md §9.
Its substitutive branch is also much smaller than the Points/Lines one, because
those coarsen by LIFTING to gsplats (scalar baking, amplitude conservation, an
anisotropy cap) while a mesh is simply decimated.
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
from ..compositing import COMPOSITING_ATTRS, sync_custom_colormap_attr
from ..dim_order import apply_dim_order_positions

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def _reject_specialized_parent(parent_node: "Node", name: str) -> None:
    """Refuse to write a mesh leaf under a ``kind=partition`` group.

    ``kind=lod`` used to be refused here too and no longer is: a mesh IS a valid
    substitutive level now that a producer exists (``luxar.mesh.decimate``), and
    ``substitutive_lod=`` builds exactly this shape — a ``kind=lod`` group whose
    children are progressively decimated meshes. The additive prefix ladder stays
    impossible for a surface, but that was never what this guard covered:
    ``kind=lod`` holds levels that REPLACE one another, while an additive ladder
    is ``additive_<i>/`` subgroups inside a leaf. See
    :data:`_UNSUPPORTED_STRUCTURE_PARAMS` for the flavour that is still refused.

    ``kind=partition`` remains refused. ``add_partition_group`` already rejects
    ``display_type='mesh'``, but nothing stops a caller from creating a ``points``
    partition and then adding a mesh child into it, which would make the group's
    declared display type a lie. It is a caller mistake with no valid
    interpretation, so it raises rather than warns.
    """
    kind = parent_node.attrs.get("kind")
    if kind == "partition":
        raise ValueError(
            f"Cannot add mesh '{name}' to a kind=partition group. Mesh has no "
            "spatial-partition path yet: a BSP cut runs through faces, so each "
            "part needs its boundary vertices duplicated and the per-vertex label "
            "CSR split to match. Add the mesh to a plain group instead."
        )


# Reason per structural parameter the sibling adders take and mesh does not.
# ``substitutive_lod`` left this table when the decimator landed — mesh now takes
# it as a real parameter. Worded for a caller who passed the knob; insertion order
# decides which one a multi-parameter call is told about.
_UNSUPPORTED_STRUCTURE_PARAMS: Dict[str, str] = {
    "additive_lod": (
        "an ADDITIVE prefix ladder cannot apply at all — a prefix of an index "
        "buffer is a surface with holes in it, not a coarser surface"
    ),
    "partition": (
        "a BSP cut runs through faces, so each part needs its boundary vertices "
        "duplicated and the per-vertex label CSR split to match"
    ),
}


def _reject_structure_params(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse the sibling adders' ``additive_lod`` / ``substitutive_lod`` / ``partition``.

    ``add_mesh`` has no such parameters, so a caller who passes one lands in
    ``**attrs`` and gets the generic UNKNOWN-attribute rejection from
    ``validate_render_attrs`` — "The viewer would silently ignore it… Remove it or
    use a supported attribute", plus a "Did you mean 'absorption'?" hint for
    ``partition``. That is a typo diagnostic, and this caller made no typo: they
    asked for a real feature the other three geometry types have, by its real name.

    The refusal is correct either way; only its stated reason was wrong, which is the
    same defect the parent-group message above carried. Answering with the per-flavour
    reason (spec §9) is what tells a user whether to wait for the feature.
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

    The other §9 exclusions are refused already — LOD and partition by
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
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    substitutive_lod: Any = None,
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

        # Substitutive-LOD branch — coarse levels are DECIMATED meshes under a
        # kind=lod Group whose finest child is the original surface. Placed after
        # validation (so a malformed mesh fails the same way either path) and
        # before the write (the wrapper writes every child itself, including the
        # finest, so falling through would write the leaf twice).
        #
        # `vert_arr` is already dim_order-transformed, so children are written
        # with dim_order=None/fill=None to avoid double application.
        if substitutive_lod is not None:
            from ..lod.mesh import resolve_substitutive_axis_mesh

            substitutive_spec = resolve_substitutive_axis_mesh(substitutive_lod)
            if substitutive_spec is not None:
                return add_mesh_substitutive_lod_wrapper_impl(
                    group,
                    name=name,
                    vert_arr=vert_arr,
                    faces_arr=faces_arr,
                    normals=normals,
                    normal_dims=normal_dims,
                    colors=colors,
                    scalars=scalars,
                    shading=shading,
                    double_sided=double_sided,
                    labels=labels,
                    image_labels=image_labels,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    spec=substitutive_spec,
                    scene=scene,
                    **attrs,
                )

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


def add_mesh_substitutive_lod_wrapper_impl(
    group: "Group",
    *,
    name: str,
    vert_arr: np.ndarray,
    faces_arr: np.ndarray,
    normals: Any,
    normal_dims: Optional[Sequence[int]],
    colors: Any,
    scalars: Any,
    shading: Optional[str],
    double_sided: bool,
    labels: Any,
    image_labels: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    spec: Dict[str, Any],
    scene: Any,
    **attrs: Any,
) -> Union["Group", Mesh]:
    """Write a mesh whose coarse LOD levels are DECIMATED copies of the surface.

    The Mesh peer of ``add_points_substitutive_lod_wrapper_impl`` and much
    smaller than it, because Points and Lines coarsen by lifting to gsplats —
    scalar→RGB baking, mass-preserving amplitudes, an anisotropy cap — and a mesh
    is simply decimated. What is shared is the SHAPE: a ``kind=lod`` group,
    children coarsest→finest, viewport-relative ``coverage_fraction`` per child
    from :func:`luxar.core.group.lod.group.coverage_fractions`, compositing attrs
    on the group and everything else on the children.

    **Level targets are vertex counts**, ``V / K**i``, because that is the
    currency the decimator's search is expressed in. Triangle count would be an
    equally defensible proxy for rendered detail (and is roughly ``2V`` on a
    closed manifold), but mixing the two would mean asking for one and thresholding
    on the other.

    A requested level is DROPPED rather than written when it cannot be a real
    level: below the decimator's 4-vertex floor, or reducing to no fewer vertices
    than the level before it. Writing it anyway would put two identical surfaces
    in the ladder and give ``coverage_fractions`` a duplicate ratio, which is not
    strictly ascending and raises. If every level drops — a surface already too
    coarse to reduce — the ladder is abandoned and a plain leaf is written, which
    is the same degenerate-path behaviour the Points wrapper has.
    """
    from ....mesh.decimate import decimate_cluster
    from ..lod.group import coverage_fractions, resolve_coarsen_dims

    n_vertices = int(vert_arr.shape[0])
    ndim = int(vert_arr.shape[1])
    compression_factor = int(spec["compression_factor"])
    levels = int(spec["levels"])

    # `coarsen_dims` is the authoring name for the decimator's `spatial_dims` —
    # the dims the reduction may merge across; the complement are hard barriers.
    #
    # Resolved against the SCENE, through the same helper Points and Lines use, so
    # dimension NAMES and the `"display"` default work as the resolver's docstring
    # promises. An earlier version only accepted an already-integer list and
    # silently passed `None` for anything else, which turned
    # `coarsen_dims=["x", "y", "z", "time"]` into "coarsen the first three columns
    # and treat time as a barrier" — the opposite of the request, with no warning.
    #
    # The `None` return needs translating rather than forwarding: it means
    # "coarsen over ALL dims" to the resolver, and "default to the first three" to
    # the decimator. Those coincide for a 3D mesh and diverge for anything else,
    # so `None` becomes an explicit all-columns tuple here.
    coarsen = resolve_coarsen_dims(scene, ndim, spec.get("coarsen_dims"))
    spatial_dims: tuple = coarsen if coarsen is not None else tuple(range(ndim))

    # Coarsest first, so the ladder reads the way it is written out. Counts must
    # come out strictly ASCENDING in that order, which is what the comparison
    # below enforces — against the previously KEPT (coarser) level, and against
    # the original at the top. Comparing the other way round drops every level
    # after the first, since each is legitimately larger than the one before it.
    # PER-VERTEX colours are averaged per cluster by the decimator; a UNIFORM
    # colour is not per-vertex data at all and must be forwarded verbatim to every
    # level instead. `isinstance(colors, np.ndarray)` was the wrong discriminator
    # for that and failed two ways: a uniform TUPLE fell through to `None`, so the
    # coarse levels came out colourless against a coloured finest one (a visible
    # colour pop on every LOD switch), and a uniform NDARRAY of shape (3,) reached
    # the decimator, where `colors.shape[1]` raised a bare `IndexError` that the
    # adder's ValueError/TypeError funnel does not even catch.
    per_vertex_colors = (
        colors
        if isinstance(colors, np.ndarray)
        and colors.ndim == 2
        and colors.shape[0] == n_vertices
        else None
    )

    coarse: List[Any] = []
    previous = 0
    for power in range(levels, 0, -1):
        target = n_vertices // (compression_factor**power)
        if target < 4:
            continue
        level = decimate_cluster(
            vert_arr,
            faces_arr.reshape(-1, 3).astype(np.uint32),
            target_vertices=target,
            normals=normals if normals is not None else None,
            # The normal FRAME, which is not the coarsening axes — a grid may merge
            # over any number of dims while a normal always lives in exactly three.
            normal_dims=tuple(normal_dims) if normal_dims is not None else None,
            colors=per_vertex_colors,
            spatial_dims=spatial_dims,
        )
        count = int(level.vertices.shape[0])
        # Strictly between the previous (coarser) level and the original, or it
        # adds nothing: a duplicate count would also give `coverage_fractions` a
        # repeated ratio, which is not strictly ascending and raises.
        if count <= previous or count >= n_vertices:
            continue
        coarse.append(level)
        previous = count

    if not coarse:
        aprint(
            f"  📐 Substitutive-LOD '{name}': no level reduced the surface "
            f"({n_vertices:,} vertices at K={compression_factor}) — writing a "
            "plain mesh leaf instead."
        )
        return add_mesh_impl(
            group,
            name=name,
            vertices=vert_arr,
            faces=faces_arr,
            normals=normals,
            normal_dims=normal_dims,
            colors=colors,
            scalars=scalars,
            shading=shading,
            double_sided=double_sided,
            labels=labels,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            substitutive_lod=None,
            **attrs,
        )

    counts = [int(c.vertices.shape[0]) for c in coarse] + [n_vertices]
    explicit = spec.get("coverage_fractions")
    if explicit is not None:
        if len(explicit) != len(counts):
            raise ValueError(
                f"coverage_fractions has {len(explicit)} entries but the LOD ladder "
                f"has {len(counts)} levels ({len(coarse)} decimated + 1 original). "
                "Levels that could not reduce the surface are dropped, so the ladder "
                "can be shorter than the requested `levels`."
            )
        coverage_vals = list(explicit)
    else:
        coverage_vals = coverage_fractions(counts)

    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    child_attrs.pop("coverage_fraction", None)
    lod_attrs.setdefault("display_type", "mesh")

    parent_node = parent or group
    aprint(
        f"  📐 Substitutive-LOD '{name}': {len(coarse)} decimated levels + original "
        f"(vertex counts coarsest→finest={counts}, K={compression_factor})"
    )
    lod_group_node = parent_node.add_lod_group(name, **lod_attrs)

    for idx, level in enumerate(coarse):
        lod_group_node.add_mesh(
            f"child_{idx}",
            level.vertices,
            level.faces,
            normals=level.normals,
            # Normals are recomputed from the COARSE surface, so they describe the
            # same axis triple the input's did.
            normal_dims=normal_dims if level.normals is not None else None,
            # Averaged per cluster when per-vertex; the original uniform value
            # otherwise, since every vertex shares it and nothing needs merging.
            colors=level.colors if per_vertex_colors is not None else colors,
            shading=shading,
            double_sided=double_sided,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            coverage_fraction=coverage_vals[idx],
            **child_attrs,
        )

    # Finest child: the original surface, with everything the coarse levels
    # cannot carry — scalars (the decimator averages colours, not scalars) and
    # the label channels.
    lod_group_node.add_mesh(
        f"child_{len(coarse)}",
        vert_arr,
        faces_arr,
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
        shading=shading,
        double_sided=double_sided,
        labels=labels,
        image_labels=image_labels,
        extend_to_all=extend_to_all,
        dim_order=None,
        fill=None,
        coverage_fraction=coverage_vals[-1],
        **child_attrs,
    )

    return lod_group_node
