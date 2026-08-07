"""add_mesh body.

Pure function taking a ``group: Group`` parameter as the first arg. Called by
``Group.add_mesh`` (a thin signature + docstring + delegate) in
``core/group/group.py``.

Notably shorter than its siblings, and structurally so: mesh has no
``additive_lod`` / ``substitutive_lod`` / ``partition`` branches, because none of
those paths exist for a mesh yet — for three different reasons, spelled out in the
rejections below and in MESH_NODE_SPEC.md §9. What remains is the
single-leaf write path the other adders reach after their LOD/partition
dispatch — plus one rejection those adders never need, since a specialized-group
parent is the one way a mesh could end up somewhere it cannot be rendered.
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
from ..compositing import sync_custom_colormap_attr
from ..dim_order import apply_dim_order_positions

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def _reject_specialized_parent(parent_node: "Node", name: str) -> None:
    """Refuse to write a mesh leaf under a ``kind=lod`` / ``kind=partition`` group.

    Mesh supports neither (spec §9), and the two group kinds fail differently if
    a mesh slips in:

    * ``kind=partition`` — ``add_partition_group`` already rejects
      ``display_type='mesh'``, but nothing stops a caller from creating a
      ``points`` partition and then adding a mesh child into it, which would make
      the group's declared display type a lie.
    * ``kind=lod`` — there is no mesh LOD producer, and the display type resolved
      from a mesh child is refused by the LOD guard. That guard fires at finalize,
      i.e. AFTER the mesh's arrays are on disk; catching it here keeps the failure
      fail-fast and leaves no partial node.

    Both are caller mistakes with no valid interpretation, so they raise rather
    than warn.

    The two LOD flavours are refused for DIFFERENT reasons and the message says so
    (spec §9). The additive prefix ladder is excluded on principle — a prefix of an
    index buffer is a *holed* surface, not a coarse one, which is why it degrades
    gracefully for independent elements and produces a wrong picture here.
    Substitutive levels are excluded only for want of a producer: that machinery
    makes no independence assumption at all, since a level is an
    independently-authored ``(vertices, faces)`` pair chosen by
    ``coverage_fraction``. Conflating the two (as this message once did) tells a
    user the feature is impossible when it is merely unwritten.
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
    if kind == "partition":
        raise ValueError(
            f"Cannot add mesh '{name}' to a kind=partition group. Mesh has no "
            "spatial-partition path yet: a BSP cut runs through faces, so each "
            "part needs its boundary vertices duplicated and the per-vertex label "
            "CSR split to match. Add the mesh to a plain group instead."
        )


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
                f"Cannot add mesh '{name}' with '{key}'. That is a Points / Lines / "
                f"GSplats parameter and mesh has no such path yet: {reason} (spec "
                "§9). Write the mesh as a plain leaf instead."
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
    """
    supplied = sorted(k for k in ("level_stats", "lod_stats") if k in attrs)
    if supplied:
        raise ValueError(
            f"Cannot add mesh '{name}' with {' and '.join(supplied)}. Those carry "
            "the additive ladder's energy stamps, and the viewer's brightness "
            "compensation is gated on the BLENDING MODE, not the geometry type — "
            "so a stamped mesh in 'additive' or 'luminous' would be scaled by "
            "1/energy_fraction_cum. That brightens a dimmer prefix correctly and a "
            "holed surface wrongly (spec §9.1). Mesh has no additive ladder; omit "
            "them."
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
    **attrs: Any,
) -> Mesh:
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
