"""add_mesh body.

Pure function taking a ``group: Group`` parameter as the first arg. Called by
``Group.add_mesh`` (a thin signature + docstring + delegate) in
``core/group/group.py``.

Still shorter than its siblings, and structurally so: mesh has ``partition`` and
``substitutive_lod`` branches like the other three, but no ``additive_lod`` — a
prefix of an index buffer is a *holed* surface, not a coarse one (MESH_NODE_SPEC.md
§9, and the rejections below).

Its substitutive branch is much smaller than the Points/Lines one, because those
coarsen by LIFTING to gsplats (scalar baking, amplitude conservation, an
anisotropy cap) while a mesh is simply decimated.

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
    reject_lines_only_join,
    slice_optional_array,
    sync_custom_colormap_attr,
)
from ..dim_order import apply_dim_order_positions
from ..partition import reject_mismatched_partition_parent

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


# NOTE on specialized-group parents. Mesh used to carry its OWN pre-write guard
# (`_reject_specialized_parent`) refusing a `kind=lod` or `kind=partition` parent
# outright. Neither refusal survives, for two different reasons:
#
# * ``kind=lod`` — a mesh IS a valid substitutive level now that a producer
#   exists (``luxar.mesh.decimate``), and ``substitutive_lod=`` builds exactly
#   this shape: a ``kind=lod`` group whose children are progressively decimated
#   meshes. The ADDITIVE prefix ladder stays impossible for a surface, but that
#   was never what the guard covered — ``kind=lod`` holds levels that REPLACE one
#   another, while an additive ladder is ``additive_<i>/`` subgroups inside a
#   leaf. See :data:`_UNSUPPORTED_STRUCTURE_PARAMS` for the flavour still refused.
# * ``kind=partition`` — mesh is partition-capable now, and a mesh leaf under a
#   ``display_type='mesh'`` partition is exactly what ``add_mesh(partition=...)``
#   writes. A partition declaring some OTHER geometry type is still refused, by
#   the shared ``reject_mismatched_partition_parent`` every leaf adder calls; the
#   rule is symmetric (a points leaf under a ``display_type='mesh'`` partition is
#   refused the same way), so it does not belong to mesh. That is why the adder
#   below calls the shared helper directly, exactly like its three siblings.


# Reason per structural parameter the sibling adders take and mesh does not.
# ``substitutive_lod`` left this table when the decimator landed and ``partition``
# when the splitter did — mesh now takes both as real parameters. Worded for a
# caller who passed the knob; insertion order decides which one a multi-parameter
# call is told about.
_UNSUPPORTED_STRUCTURE_PARAMS: Dict[str, str] = {
    "additive_lod": (
        "an ADDITIVE prefix ladder cannot apply at all — a prefix of an index "
        "buffer is a surface with holes in it, not a coarser surface"
    ),
    # NOTE: ``substitutive_lod`` and ``partition`` are both deliberately absent —
    # each is a real ``add_mesh`` parameter now, bound by name, so neither ever
    # reaches ``**attrs``.
}


def _reject_structure_params(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse the sibling adders' ``additive_lod``.

    ``add_mesh`` has no such parameter, so a caller who passes it lands in
    ``**attrs`` and gets the generic UNKNOWN-attribute rejection from
    ``validate_render_attrs`` — "The viewer would silently ignore it… Remove it or
    use a supported attribute". That is a typo diagnostic, and this caller made no
    typo: they asked for a real feature the other three geometry types have, by
    its real name.

    The refusal is correct either way; only its stated reason was wrong. Answering
    with the per-flavour reason (spec §9) is what tells a user whether to wait for
    the feature — and for the additive prefix ladder the answer is "never on a
    surface", not "not yet".

    ``substitutive_lod`` and ``partition`` are no longer in this table — both are
    real ``add_mesh`` parameters, bound by name, so neither reaches ``**attrs``.
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

    The other §9 exclusions are refused already — the additive ladder by
    :func:`_reject_structure_params` and by the viewer's progressive-loader
    factory, a mismatched partition parent by the shared
    ``reject_mismatched_partition_parent`` — but this one was documented and never
    enforced, so a volumetric mesh wrote and loaded cleanly.

    It cannot mean anything. Volumetric blending is emission-absorption integration
    through a participating medium: the shader scales each element's contribution by
    its extent along the view ray and maps ``absorption`` into optical depth over
    that path length. A triangle is a zero-thickness surface, so its path length is
    identically zero and there is no medium to absorb anything — the mode has no
    per-element quantity to integrate. That makes it a caller mistake with no valid
    interpretation, which is the same bar the surviving structural refusals are held to
    (``additive_lod``, and a partition parent declaring a non-mesh ``display_type``),
    so it raises rather than warns.

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

    This is deliberately prophylactic rather than a fix for a live bug: the mesh
    commit never stamps ``committedEnergyFraction``, so the compensation factor is 1.
    Substitutive LOD has already removed the OTHER latch — a mesh can be a
    ``kind=lod`` level now, so the fade pass does visit it — and a reveal ladder
    would remove this one. Refusing now means §9.1's "must NOT carry energy stamps"
    rule is enforced before that lands, rather than being a note someone has to
    remember.

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


def validate_scalar_data_range(name: str, value: Any) -> Optional[tuple[float, float]]:
    """Check the internal ``_scalar_data_range`` plumbing key (or pass ``None``).

    Public despite validating a private key, because ``luxar mesh lod`` has to run
    it BEFORE it deletes its output — the same reason that command validates the
    decimation method up front. Sharing this function is what keeps the CLI's
    early check and the adder's real one from disagreeing about what is accepted.

    Internal, but reachable: it is a keyword like any other, and an ill-formed
    one used to travel all the way to ``write_scalars``, where ``bounds[1]``
    raised a bare ``IndexError`` — a type the adder's ValueError/TypeError funnel
    does not catch, so the exception escaped mid-write with the vertices and
    faces already on disk. Everything it can be wrong about is cheap to check
    here, before anything is written.

    Reversed and non-finite pairs are refused rather than repaired: the pair is
    also the scalars quantization range, so a silently swapped or NaN window
    would encode the whole field wrongly with no diagnostic.
    """
    if value is None:
        return None
    try:
        lo, hi = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Mesh '{name}': _scalar_data_range must be a (min, max) pair of "
            f"numbers; got {value!r}"
        ) from exc
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(
            f"Mesh '{name}': _scalar_data_range must be finite; got ({lo}, {hi})"
        )
    if lo > hi:
        raise ValueError(
            f"Mesh '{name}': _scalar_data_range is reversed — ({lo}, {hi}). It is "
            "also the scalars quantization range, so the order is load-bearing."
        )
    return (lo, hi)


def _shared_scalar_window(
    explicit: Optional[tuple[float, float]], scalars: Any, n_vertices: int
) -> Optional[tuple[float, float]]:
    """The ONE display window every child of a structural wrapper stamps.

    Shared by both wrappers because both split a single scalar field across
    several nodes, and the viewer windows a node's colormap on that node's OWN
    stamped ``scalar_data_range``: a level or a part that stamps its own subset
    min/max renders the same value as a different colour, and a subset that is
    constant stamps a degenerate ``[v, v]`` the viewer maps to the LUT midpoint.

    Derived from the WHOLE field, before any split or decimation, and UNIONED
    with the caller's explicit window (already validated) when there is one.
    The union rather than the explicit pair verbatim, because ``write_scalars``
    widens (never narrows) the window onto each node's own values — the pair is
    also that node's quantization range, so a datum outside it would come back
    clipped. A deliberately narrower window therefore came out different on
    every child: an explicit ``(0, 1)`` over parts holding ``[-10, -5]`` and
    ``[5, 10]`` stamped ``[-10, 1]`` and ``[0, 10]``, which is exactly the
    discontinuity this helper exists to prevent. Unioning up front makes every
    child stamp the one window a plain leaf over the same field would.

    Only a per-VERTEX array needs one — the same discriminator the levels use
    for forwarding, since a broadcast value is identical on every child already
    and needs nothing shared. Callers must run their fail-fast scalars gate
    first; that is what rules out a non-finite field here (an explicit window is
    refused non-finite, a derived one would not be).
    """
    if not (isinstance(scalars, np.ndarray) and scalars.shape[:1] == (n_vertices,)):
        return explicit
    field = (float(np.min(scalars)), float(np.max(scalars)))
    if explicit is None:
        return field
    return (min(explicit[0], field[0]), max(explicit[1], field[1]))


def _resolve_mesh_vertices(vertices: Any) -> np.ndarray:
    """Coerce ``vertices`` to an array and refuse a shape a mesh cannot render.

    Extracted whole from ``add_mesh_impl`` so the coercion and the two shape
    refusals read as one step, and so the adder body stays under the C901 limit
    the complexity ratchet enforces. Pure: no scene state, no writes — it runs
    BEFORE ``dim_order`` is applied so that both refusals judge the AUTHORED
    array. ``dim_order`` does not merely permute the coordinate columns, it also
    WIDENS them, padding unmapped scene dimensions with constant ``fill`` values;
    a 1-column input would come out a technically-valid 2-D one whose extra axis
    is a constant, i.e. still arealess. Checking first keeps the error about what
    the caller actually wrote.
    """
    vert_arr: np.ndarray = (
        vertices if isinstance(vertices, np.ndarray) else np.asarray(vertices)
    )
    if vert_arr.ndim != 2:
        raise ValueError(f"Vertices must have shape (V, D), got shape {vert_arr.shape}")
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
    return vert_arr


def _reject_partition_with_substitutive_lod(
    partition: Any, substitutive_lod: Any
) -> None:
    """Refuse ``partition=`` together with ``substitutive_lod=``.

    The same refusal ``add_points`` / ``add_lines`` carry, with the same message: a
    ``kind=partition`` of per-part LOD ladders is a topology nothing writes yet.
    Mesh needs it for one extra reason — the substitutive branch RETURNS before the
    partition branch is reached, so accepting both would silently drop the split.

    ``False`` is an explicit no-op sentinel on BOTH sides, so neither trips this:
    ``partition=False`` is what :func:`_add_mesh_partition` hands each part (a part
    must never recurse into another partition), and ``substitutive_lod=False`` is
    ``resolve_substitutive_axis_mesh``'s documented "no ladder" spelling. Refusing
    either would refuse a call that asked for exactly one of the two features.

    Tested with ``is`` rather than ``in (None, False)``, because the latter compares
    by EQUALITY: ``0 == False``, so ``partition=0`` read as "not requested" here
    while the real dispatch (``partition is not None and partition is not False``)
    read it as requested and silently dropped the split behind a ladder. The guard
    and the dispatch must agree by construction.

    Its own function rather than an inline ``if`` because ``add_mesh_impl`` sits at
    the C901 limit the complexity ratchet enforces — the same reason
    :func:`_resolve_mesh_vertices` and :func:`_validate_partition_sources` were
    hoisted out of it.
    """
    wants_partition = partition is not None and partition is not False
    wants_ladder = substitutive_lod is not None and substitutive_lod is not False
    if wants_partition and wants_ladder:
        raise ValueError(
            "partition= and substitutive_lod= cannot be combined yet "
            "(partition-of-substitutive is not implemented). Use one or the other."
        )


def _maybe_add_mesh_substitutive_lod(
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
    scalar_data_range: Optional[tuple[float, float]],
    substitutive_lod: Any,
    scene: Any,
    **attrs: Any,
) -> Optional[Union[Mesh, "Group"]]:
    """Dispatch the substitutive-LOD branch, or return ``None`` to fall through.

    Extracted whole from ``add_mesh_impl`` — the child-gate preflight and the
    wrapper hand-off are one step, and hoisting them keeps that function under the
    C901 limit the complexity ratchet enforces (mesh now dispatches BOTH structural
    branches, so the adder body would otherwise sit over it).

    Runs the CHILD's gates and throws the results away, purely to keep the
    fail-fast pre-write gate intact. Every one of them runs again inside
    ``child_0`` — but by then every level has been decimated and ``add_lod_group``
    has created the zarr group, so a bad ``extend_to_all``, ``colormap`` or
    ``blending_mode`` surfaced as a childless kind=lod group in an incomplete store
    rather than as a clean refusal that wrote nothing. Same validators the children
    run, so the two cannot disagree about what is accepted.

    ``extend_to_all`` is guarded on ``is not None`` because that branch is the one
    that emits the advisory candidate warning, which must fire exactly once. The
    attr gate gets a COPY, since it is the child write's job to consume the real
    dict.
    """
    if substitutive_lod is None:
        return None

    from ....io._compiler.node_common import (
        MESH_RESERVED_ATTRS,
        validate_render_attrs,
    )
    from ..lod.mesh import resolve_substitutive_axis_mesh

    if extend_to_all is not None:
        scene._resolve_extend_to_all(extend_to_all, vert_arr, "mesh")
    validate_render_attrs(dict(attrs), reserved_attrs=MESH_RESERVED_ATTRS)

    substitutive_spec = resolve_substitutive_axis_mesh(substitutive_lod)
    if substitutive_spec is None:
        return None
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
        scalar_data_range=scalar_data_range,
        spec=substitutive_spec,
        scene=scene,
        **attrs,
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
    substitutive_lod: Any = None,
    **attrs: Any,
) -> Union[Mesh, "Group"]:
    # Outside the ``try`` for the same reason the siblings raise it there: an
    # argument error, not a write failure, so it must not be re-wrapped as
    # "Could not add mesh '<name>': …".
    _reject_partition_with_substitutive_lod(partition, substitutive_lod)
    try:
        # Fail-fast pre-write gate: reject invalid names (empty/'/'/dot-prefixed —
        # an empty name resolves to the zarr ROOT group and would clobber the
        # scene root) and duplicate siblings BEFORE any zarr write. Node.__init__
        # re-checks both post-write (belt and braces).
        from ....validation.base import validate_node_name

        validate_node_name(name)
        (parent or group)._ensure_no_duplicate_child(name)
        reject_mismatched_partition_parent(parent or group, "mesh", name)
        _reject_structure_params(name, attrs)
        _reject_volumetric_blending(name, attrs)
        _reject_energy_stamps(name, attrs)
        reject_lines_only_join("mesh", name, attrs)

        # Consumed HERE, at the top, so nothing downstream — the substitutive
        # wrapper, the attr gate, the returned node object — ever sees the
        # private key. Validated because it is a real parameter with a real
        # shape: a 1-tuple used to reach the writer and raise a bare IndexError,
        # which this funnel does not catch, leaving a half-written node.
        scalar_data_range = validate_scalar_data_range(
            name, attrs.pop("_scalar_data_range", None)
        )

        scene = group._find_scene()

        vert_arr = _resolve_mesh_vertices(vertices)

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

        # Substitutive-LOD branch — coarse levels are DECIMATED meshes under a
        # kind=lod Group whose finest child is the original surface. Placed after
        # validation (so a malformed mesh fails the same way either path) and
        # before the write (the wrapper writes every child itself, including the
        # finest, so falling through would write the leaf twice).
        #
        # ALSO before the `extend_to_all` resolution below, which is why the
        # sibling adders dispatch their structural branches here too: that
        # resolution lands in `attrs`, and the wrapper takes `**attrs` alongside
        # its own `extend_to_all=` parameter — so resolving first made every
        # `extend_to_all="all"` ladder a "multiple values for keyword argument"
        # TypeError. The wrapper forwards the RAW value to each child's
        # `add_mesh`, which resolves it per child.
        #
        # `vert_arr` is already dim_order-transformed, so children are written
        # with dim_order=None/fill=None to avoid double application.
        laddered = _maybe_add_mesh_substitutive_lod(
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
            scalar_data_range=scalar_data_range,
            substitutive_lod=substitutive_lod,
            scene=scene,
            **attrs,
        )
        if laddered is not None:
            return laddered

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
            # ``extend_to_all`` was just RESOLVED into ``attrs`` above, and this
            # call also passes it by name — so hand the split a copy of attrs
            # without the key, or the two collide as a duplicate keyword argument
            # and every partitioned+extended mesh fails. The parts get the
            # resolved dimension names (re-resolving a name list is a no-op).
            partition_attrs = {k: v for k, v in attrs.items() if k != "extend_to_all"}
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
                extend_to_all=final_extend_dims or extend_to_all,
                scalar_data_range=scalar_data_range,
                **partition_attrs,
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
            _scalar_data_range=scalar_data_range,
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
    scalar_data_range: Optional[tuple[float, float]],
    spec: Dict[str, Any],
    scene: Any,
    **attrs: Any,
) -> Union["Group", Mesh]:
    """Write a mesh whose coarse LOD levels are DECIMATED copies of the surface.

    The Mesh peer of ``add_points_substitutive_lod_wrapper_impl`` and much
    smaller than it, because Points and Lines coarsen by lifting to gsplats —
    scalar→RGB baking, mass-preserving amplitudes, an anisotropy cap — and a mesh
    is simply decimated (its per-vertex colours and scalars are averaged per
    cluster, so both reach every level). What is shared is the SHAPE: a
    ``kind=lod`` group, children coarsest→finest, viewport-relative ``coverage_fraction`` per child
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
    from ....io._compiler.geometry_writers.mesh import validate_mesh_arrays
    from ....mesh.decimate import decimate_cluster
    from ..lod.group import coverage_fractions, resolve_coarsen_dims

    # Fail-fast pre-write gate, part two: the ARRAYS, run BEFORE any decimation
    # and before `add_lod_group` creates the group. The adder already ran the
    # child's attr/extend_to_all gates on the way in; these are the rest of what
    # a child write checks, and without them a malformed channel was refused
    # only from inside a child — colours with two components, a wrong-length
    # scalars array or a typo'd `shading` from `child_0`, a bad normals array or
    # `labels` length from the FINEST child, which is written last. Either way
    # the store was left holding a `kind=lod` group with no children (or with
    # its finest one missing), where the plain-leaf path writes nothing at all.
    #
    # The authored arrays are the right thing to check: a level's decimated
    # arrays are derived from them (cluster means keep the dtype, the channel
    # count and the value range; recomputed normals are always (V, 3)), so an
    # input the writer accepts cannot produce a level it refuses. This is also
    # what catches a NON-FINITE scalar field, which no level can store.
    validate_mesh_arrays(
        vert_arr,
        faces_arr,
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
        shading=shading,
        double_sided=double_sided,
        labels=labels,
    )

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
    # Same discriminator, same reason. Scalars ARE the colour of a colormapped
    # mesh, and `colormap` is a child attr copied onto every level — so a coarse
    # level without scalars carries a colormap with nothing to map and renders
    # unmapped against a mapped finest level, which is a pop at every switch.
    per_vertex_scalars = (
        scalars
        if isinstance(scalars, np.ndarray) and scalars.shape[:1] == (n_vertices,)
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
            scalars=per_vertex_scalars,
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
            # Re-supplied because the adder popped it on the way in; a leaf's
            # own range equals the field's, so this only matters when the
            # caller set one explicitly.
            _scalar_data_range=scalar_data_range,
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

    # ONE display window for the whole ladder, stamped on every child. Each
    # level would otherwise stamp its own min/max, and cluster-averaging
    # strictly CONTRACTS the range — so the viewer, which windows a level's
    # colormap on its stamped `scalar_data_range`, would map the same value to a
    # different colour at every level and the surface would recolour as you
    # zoom. That is the pop this ladder exists to avoid, one layer down. Same
    # rule the gsplat lift states for its beads ("share the finest node's
    # scalar_data_range, not a per-segment one"). Same helper the partition
    # wrapper uses, so the two topologies cannot disagree about what the shared
    # window is; `validate_mesh_arrays` above is this path's fail-fast scalars
    # gate, which the helper's docstring requires.
    field_range = _shared_scalar_window(scalar_data_range, scalars, n_vertices)

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
            # Averaged per cluster like the colours, for the same reason: the
            # colormap rides on every child, so a level without scalars is a
            # level the colormap cannot reach.
            scalars=level.scalars if per_vertex_scalars is not None else scalars,
            _scalar_data_range=field_range,
            shading=shading,
            double_sided=double_sided,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            coverage_fraction=coverage_vals[idx],
            **child_attrs,
        )

    # Finest child: the original surface, with everything the coarse levels
    # cannot carry — the label channels, which are per-vertex CSR text and have
    # no meaningful merge (colours and scalars are both averaged, so those DO
    # reach every level).
    lod_group_node.add_mesh(
        f"child_{len(coarse)}",
        vert_arr,
        faces_arr,
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
        # Passed explicitly even though the finest child's own range already
        # equals it: stamping it here is what makes the ladder's single shared
        # window visible at the one level that could have got away without it.
        _scalar_data_range=field_range,
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


def _is_broadcast_color(colors: Any) -> bool:
    """Whether ``colors`` is a uniform RGB(A) sequence rather than per-vertex data.

    Classifies on SHAPE only — a list/tuple of 3 or 4 numeric components — which is
    the admission test
    :func:`~luxar.io._compiler.node_common.validate_broadcast_color` applies at the
    writer. Values (finite, non-negative, alpha in range) are deliberately left to
    that validator, so a bad uniform color fails with the same message it gets
    without ``partition=``.

    A per-vertex list of triples fails the component test (its entries are
    sequences, not numbers) and is gathered normally, as are numpy colors of any
    shape — the writer refuses a 1-D numpy color outright, so only a list/tuple can
    be the broadcast form.
    """
    if not isinstance(colors, (list, tuple)) or len(colors) not in (3, 4):
        return False
    return all(isinstance(c, (int, float, np.integer, np.floating)) for c in colors)


def _validate_partition_sources(
    faces_arr: np.ndarray,
    n_vertices: int,
    *,
    normals: Any,
    colors: Any,
    scalars: Any,
    labels: Any,
    image_labels: Any,
) -> None:
    """Run the plain-leaf write gates against the SOURCE arrays, before the split.

    Extracted whole from :func:`_add_mesh_partition` — the rejections are one
    cohesive step (everything that must fail before a single index is used to
    gather), and hoisting them keeps that function under the C901 limit the
    complexity ratchet enforces. Order is load-bearing and preserved exactly:
    ``image_labels`` first, then faces, then the per-vertex channels in
    ``normals``, ``colors``, ``scalars``, ``labels`` order — a call that trips
    several is told about the same one it was told about before.
    """
    from ....io._compiler.node_common import (
        validate_broadcast_color,
        validate_scalars_preflight,
    )
    from ....validation.base import (
        validate_colors_for_writing,
        validate_faces_for_writing,
        validate_labels_for_writing,
        validate_normals_for_writing,
    )

    if image_labels is not None:
        raise ValueError(
            "image_labels is not supported alongside partition=. It is a "
            "whole-node image-to-label mapping with no per-part meaning, and "
            "splitting it would silently change what each part's labels index. "
            "Decompose manually or omit image_labels."
        )

    # Validate the ORIGINAL indices before they are used to gather anything. On
    # the plain-leaf path the writer does this, but the split runs first and
    # every one of these failures is silent or unrecognisable here: numpy WRAPS a
    # negative index while gathering, so `-1` would quietly become the last
    # vertex and write a triangle the author never wound, and an out-of-range or
    # float index surfaces as a bare IndexError from inside the centroid gather
    # instead of the guided message the same input gets without partition=.
    validate_faces_for_writing(faces_arr, n_vertices)

    # Same argument for the per-VERTEX channels, and the failure here is worse
    # than a bad message: `slice_optional_array` gathers only when the leading
    # length matches the vertex count and otherwise passes the value through
    # WHOLE, which is exactly what makes a uniform RGB triple or a colormap name
    # work. A wrong-length per-vertex array takes the same pass-through branch —
    # and if its length happens to equal a part's OWN vertex count, that part's
    # writer accepts it and silently pairs the values with the wrong vertices.
    # (Two disconnected triangles: six source vertices, two three-vertex parts,
    # three normals — rejected outright without `partition=`, accepted by both
    # parts with it.) So run the plain-leaf gate against the SOURCE count first;
    # a given input then fails identically whether or not it is partitioned.
    if normals is not None:
        validate_normals_for_writing(normals, n_vertices)
    if isinstance(colors, np.ndarray):
        validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
    elif isinstance(colors, (list, tuple)):
        validate_broadcast_color(colors, "colors")
    if scalars is not None:
        validate_scalars_preflight(scalars, n_vertices)
    if labels is not None:
        validate_labels_for_writing(labels, n_vertices)


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
    scalar_data_range: Optional[tuple[float, float]] = None,
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

    Every part is handed ONE display window as ``_scalar_data_range=`` —
    :func:`_shared_scalar_window` over the whole field, unioned with the caller's
    explicit one — because the viewer windows a node's colormap on that node's OWN
    stamped ``scalar_data_range``, so a per-part subset min/max renders the same
    scalar value as a different colour either side of a BSP cut. The union is what
    makes the window survive verbatim: ``write_scalars`` widens (never narrows)
    the pair onto each node's own data, so a window narrower than the field would
    otherwise come back out per-part.
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

    n_vertices = int(vert_arr.shape[0])
    _validate_partition_sources(
        faces_arr,
        n_vertices,
        normals=normals,
        colors=colors,
        scalars=scalars,
        labels=labels,
        image_labels=image_labels,
    )

    # Derived here, AFTER that gate (it is the fail-fast scalars check the helper
    # requires), and before the split: an explicit window is only the minority
    # case, and without a derived one the DEFAULT `scalars=` call has every part
    # stamping its own subset min/max — a colour discontinuity at every cut, plus
    # a degenerate `[v, v]` (viewer: LUT midpoint) for any part whose subset is
    # constant.
    part_range = _shared_scalar_window(scalar_data_range, scalars, n_vertices)

    # A uniform RGB(A) list/tuple is the one leaf parameter whose OWN length can
    # collide with the vertex count, and mesh is the geometry where that collision
    # is reachable: every part holds at least three vertices (a triangle's worth),
    # so a 4-vertex source with an RGBA color satisfies `slice_optional_array`'s
    # length test and is gathered as if its four channels were four vertex rows.
    # Each part then receives a different rotated 3-slice of the components —
    # SILENTLY, because a 3-element result is itself a valid uniform RGB. Classify
    # the broadcast form up front so every part gets the color the caller wrote.
    uniform_color = _is_broadcast_color(colors)

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
        f"vertex duplication x{duplication_factor(parts):.3f})"
    )

    for i, part in enumerate(parts):
        # `slice_optional_array` is the same helper the sibling wrappers use, and
        # it is the right one here for the same reason: it gathers ONLY when the
        # value's leading length matches the element count, so a per-vertex array
        # follows its vertices while a uniform RGB triple or a colormap name is
        # handed to every part untouched. `labels` is per-VERTEX (hover tooltips),
        # so it is gathered too — passing it whole would give every part V labels
        # for its own Vi vertices. `colors` is the one exception: a broadcast
        # RGB(A) sequence is classified by SHAPE above rather than by length, since
        # its length can coincide with the vertex count (see `uniform_color`).
        take = part.vertex_index
        wrapper.add_mesh(
            name=f"part_{i}",
            vertices=vert_arr[take],
            faces=part.faces,
            normals=slice_optional_array(normals, take, n_vertices),
            normal_dims=normal_dims,
            colors=(
                colors
                if uniform_color
                else slice_optional_array(colors, take, n_vertices)
            ),
            scalars=slice_optional_array(scalars, take, n_vertices),
            _scalar_data_range=part_range,
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
