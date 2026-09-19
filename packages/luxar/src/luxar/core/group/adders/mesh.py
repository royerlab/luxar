"""add_mesh body.

Pure function taking a ``group: Group`` parameter as the first arg. Called by
``Group.add_mesh`` (a thin signature + docstring + delegate) in
``core/group/group.py``.

All three structural branches the sibling adders have are here — ``partition``,
``substitutive_lod`` and ``additive_lod`` — but the additive one is narrower by
design: a mesh ladder is a REVEAL (concentric shells) and nothing else, because a
prefix of an *arbitrarily ordered* index buffer is a holed surface rather than a
coarse one (MESH_NODE_SPEC.md §9; see
:data:`luxar.core.group.lod.mesh.MESH_ADDITIVE_METHODS`). No two of the three
COMPOSE yet, which the sibling adders partly do; each pairing is refused by name.

Its substitutive branch is much smaller than the Points/Lines one, because those
coarsen by LIFTING to gsplats (scalar baking, amplitude conservation, an
anisotropy cap) while a mesh is simply decimated.

Its additive branch is the one that is BIGGER than theirs, for the same reason the
partition branch is: a level is a re-indexing rather than a slice, so the ladder
splits FACES, re-indexes each group through :mod:`luxar.mesh.split`, and pays a
boundary-vertex duplication the element geometries never do.

The partition branch differs from the sibling adders' in the one way that
matters: theirs hand each part a SLICE of the element arrays, because their
elements are independent rows. A triangle is not a row — it is three references
into a shared vertex table — so a mesh part is a re-indexing, not a slice, and
the split lives in :mod:`luxar.mesh.split`.
"""

from __future__ import annotations

import warnings
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
from arbol import aprint, asection

from ....validation.types import (
    HOUSE_SHADER_ONLY_ATTRS,
    PHYSICAL_MATERIAL_ATTRS,
    PHYSICAL_TRANSMISSION_DEPENDENT_ATTRS,
    validate_mesh_material,
)
from ....validation.writing import (
    MESH_RESERVED_ATTRS,
    validate_broadcast_color,
    validate_mesh_arrays,
    validate_render_attrs,
    validate_scalars_preflight,
)
from ...mesh import Mesh
from ..compositing import (
    ABSENT_WHEN_NONE_RENDER_ATTRS,
    COMPOSITING_ATTRS,
    funnel_add_error,
    is_broadcast_color,
    position_bounds_from_array,
    preflight_extend_to_all,
    reject_layer_order_inside_specialized_group,
    reject_lines_only_join,
    slice_optional_array,
    strip_absent_attr_kwargs,
    sync_custom_colormap_attr,
    unnest_add_error,
)
from ..dim_order import apply_dim_order_positions, warn_if_dim_order_reverses_winding
from ..partition import is_requested, reject_mismatched_partition_parent

if TYPE_CHECKING:
    from ....mesh.split import MeshPart
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
#
# INTENTIONALLY EMPTY as of the reveal ladder: every structural parameter the
# sibling adders take is now a real ``add_mesh`` parameter, bound by name, so none
# of them can reach ``**attrs`` any more. ``substitutive_lod`` left this table when
# the decimator landed, ``partition`` when the splitter did, and ``additive_lod``
# when :func:`add_mesh_multi_lod_wrapper_impl` landed — a mesh ladder is a REVEAL
# (concentric shells, every prefix a contiguous partial surface at full
# brightness), which is the one additive family a surface admits; the *prefix of an
# arbitrary order* the old entry refused is still refused, by
# ``MESH_ADDITIVE_METHODS`` naming ``radial`` as the only accepted method.
#
# The table and :func:`_reject_structure_params` stay as the extension point: a
# future structural knob mesh cannot serve (say a ``kind=partition`` of per-part
# ladders, refused today by name in :func:`_reject_additive_lod_compositions`)
# belongs here the moment it becomes a sibling parameter mesh does not bind, so
# that a caller gets the per-flavour reason instead of a typo diagnostic. Worded
# for a caller who passed the knob; insertion order decides which one a
# multi-parameter call is told about.
_UNSUPPORTED_STRUCTURE_PARAMS: Dict[str, str] = {}


def _reject_structure_params(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse any sibling-adder structural parameter mesh does not bind by name.

    :data:`_UNSUPPORTED_STRUCTURE_PARAMS` is empty today, so this is a no-op — see
    that table's comment for why it is kept rather than deleted.

    The failure mode it exists to prevent: a parameter ``add_mesh`` does not bind
    lands in ``**attrs`` and gets the generic UNKNOWN-attribute rejection from
    ``validate_render_attrs`` — "The viewer would silently ignore it… Remove it or
    use a supported attribute". That is a typo diagnostic, and such a caller made
    no typo: they asked for a real feature the other three geometry types have, by
    its real name. The refusal would be correct either way; only its stated reason
    would be wrong, and the per-flavour reason (spec §9) is what tells a user
    whether to wait for the feature or stop waiting.
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

    The other §9 exclusions are refused already — an additive ladder that is not a
    pure reveal by ``MESH_ADDITIVE_METHODS`` and
    :func:`_reject_additive_lod_compositions`, a mismatched partition parent by the
    shared ``reject_mismatched_partition_parent`` — but this one was documented and
    never enforced, so a volumetric mesh wrote and loaded cleanly.

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


def _reject_physical_material_conflicts(
    name: str,
    attrs: Dict[str, Any],
    shading: Optional[str],
    texture: Any,
) -> None:
    """Cross-check ``material`` against the knobs that only one family understands.

    The value validators in ``validate_render_attrs`` judge each key alone. The
    rules here are about PAIRS, and they are refusals rather than warnings for the
    same reason ``blending_mode='volumetric'`` is (spec
    ``MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.1): every case below is an authoring
    mistake with no valid interpretation, and letting it through would write a
    store in which a knob the author typed does nothing — which reads, in the
    viewer, exactly like a working setting.

    - A physical knob (``roughness`` …, :data:`PHYSICAL_MATERIAL_ATTRS`) without
      ``material="physical"`` is dead metadata: the house shader has no such term.
    - Under ``material="physical"`` the house-shader knobs
      (:data:`HOUSE_SHADER_ONLY_ATTRS`) parameterise a lighting model that is not
      running; ``blending_mode`` names a compositing family three's material does
      not implement (it blends by its own ``transparent`` rule); ``colormap`` and
      ``texture`` are base-colour sources the Phase 1 material does not read (it
      takes vertex colours only); and ``shading="none"`` asks an environment-lit
      surface to be unlit. ``shading="flat"`` / ``"smooth"`` are NOT refused —
      flat-versus-smooth normals is a property of any lit surface and maps onto
      three's ``flatShading`` — and neither is ``alpha_cutoff``, which maps onto
      ``alphaTest``.
    - The glass knobs that only act inside three's transmission block
      (:data:`PHYSICAL_TRANSMISSION_DEPENDENT_ATTRS`: ``thickness``,
      ``attenuation_color``, ``attenuation_distance``, ``dispersion``, and the
      Phase 3 ``refract_data`` flag) without a ``transmission`` above zero are
      dead metadata too — three compiles the knobs under ``USE_TRANSMISSION``,
      and a surface that transmits nothing has nothing to refract — so
      ``thickness=0.4`` or ``refract_data=True`` on an opaque metal is refused.
      ``ior`` is exempt: it also sets an opaque surface's reflectance.

    Runs BEFORE the shared attrs gate so the reason a caller sees is the pairing,
    not a downstream symptom, and validates the family value first so a typo in
    ``material`` itself is reported as such rather than as a missing opt-in.
    """
    material = attrs.get("material")
    if material is not None:
        validate_mesh_material(material)
    physical_knobs = sorted(PHYSICAL_MATERIAL_ATTRS & attrs.keys())

    if material != "physical":
        if physical_knobs:
            raise ValueError(
                f"Cannot add mesh '{name}' with {physical_knobs}: these are "
                "physically based material knobs and the house shader has no such "
                "term, so they would be written and silently ignored. Pass "
                "material='physical' to opt this mesh into three's physical "
                "material (see MESH_PHYSICAL_MATERIALS_SPEC.md §3.1), or drop them."
            )
        return

    house_knobs = sorted(HOUSE_SHADER_ONLY_ATTRS & attrs.keys())
    if house_knobs:
        raise ValueError(
            f"Cannot add mesh '{name}' with material='physical' and {house_knobs}: "
            "those parameterise the house shader's view-anchored key "
            "(MESH_NODE_SPEC.md §6.2), which a physical material does not run. Use "
            "roughness/metalness/clearcoat/clearcoat_roughness/iridescence/sheen/"
            "sheen_color (and the transmission family) instead, or drop "
            "material='physical'."
        )
    _reject_physical_transmission_conflicts(name, attrs)
    if "blending_mode" in attrs:
        raise ValueError(
            f"Cannot add mesh '{name}' with material='physical' and blending_mode="
            f"{attrs['blending_mode']!r}: a physical mesh composites by three's own "
            "rule — opaque unless its opacity or vertex alpha is below 1 — and has "
            "no additive/luminous/max emission to select. Drop blending_mode (an "
            "inherited group mode is ignored for physical meshes with a notice)."
        )
    if attrs.get("colormap") is not None:
        raise ValueError(
            f"Cannot add mesh '{name}' with material='physical' and a colormap: the "
            "Phase 1 physical material takes its base colour from per-vertex "
            "colours only. Bake the colormap into `colors` instead."
        )
    if texture is not None:
        raise ValueError(
            f"Cannot add mesh '{name}' with material='physical' and a texture: the "
            "Phase 1 physical material takes its base colour from per-vertex "
            "colours only (a textured physical surface is a later phase)."
        )
    if shading == "none":
        raise ValueError(
            f"Cannot add mesh '{name}' with material='physical' and shading='none': "
            "'none' means unlit, and a physical material is lit by the scene "
            "environment by definition. Use 'smooth' (stored normals) or 'flat'."
        )


def _reject_physical_transmission_conflicts(name: str, attrs: Dict[str, Any]) -> None:
    glass_only = sorted(PHYSICAL_TRANSMISSION_DEPENDENT_ATTRS & attrs.keys())
    if not glass_only:
        return
    transmission = attrs.get("transmission")
    # A non-numeric transmission is the writer's diagnostic, not this one's.
    transmitting = isinstance(transmission, (int, float)) and transmission > 0
    if not transmitting:
        raise ValueError(
            f"Cannot add mesh '{name}' with material='physical' and {glass_only} "
            "but no transmission above zero: thickness, attenuation, dispersion "
            "and refract_data only act inside three's transmission path, so "
            "they would be written and silently ignored. Pass transmission=... "
            "(a fraction in (0, 1]) alongside them, or drop them."
        )


def _reject_energy_stamps(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse hand-supplied additive-energy fields on a mesh (spec §9.1).

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
    ``kind=lod`` level now, so the fade pass does visit it.

    **Unchanged by the arrival of the reveal ladder**, which is what makes this the
    guard rather than a note. ``add_mesh(additive_lod=…)`` writes these two keys
    ITSELF (:func:`add_mesh_multi_lod_wrapper_impl`), and it runs AFTER this gate —
    so what is refused here is a HAND-SUPPLIED value, never the ladder's own. A
    mesh ladder is reveal-only by construction
    (:data:`~luxar.core.group.lod.mesh.MESH_ADDITIVE_METHODS`), and
    :func:`~luxar.core.group.lod.group.additive_level_stats` suppresses
    ``energy_fraction_cum`` / ``reference_energy`` for every reveal method, so a
    mesh ladder carries no energy stamps at all — the only route by which a mesh
    could acquire one is a caller writing it by hand, which is exactly this.

    The check is deliberately on the ENERGY FIELDS, not the containers: substitutive
    levels use ``level_stats.geometric_error`` for their separate surface-error
    currency, which must never be mistaken for mixture ``quality`` or acquire the
    energy pair that makes the viewer's brightness compensation engage.
    """
    for container in ("level_stats", "lod_stats"):
        value = attrs.get(container)
        if value is not None and not isinstance(value, dict):
            raise TypeError(
                f"{container} must be a dict when adding a mesh; "
                f"got {type(value).__name__}"
            )

    energy_keys = {"reference_energy", "energy_fraction_cum"}
    supplied = sorted(
        f"{container}.{key}"
        for container in ("level_stats", "lod_stats")
        if isinstance(attrs.get(container), dict)
        for key in energy_keys & set(attrs[container])
    )
    if supplied:
        raise ValueError(
            f"Cannot add mesh '{name}' with {' and '.join(supplied)}. Those are the "
            "fields an additive ladder writes as energy stamps, and the viewer's brightness "
            "compensation is gated on the BLENDING MODE, not the geometry type — "
            "so a stamped mesh in 'additive' or 'luminous' would be scaled by "
            "1/energy_fraction_cum. That brightens a dimmer prefix correctly and a "
            "holed surface wrongly (spec §9.1). A mesh ladder writes both attributes "
            "itself, and it is reveal-only — a partial surface at full brightness — "
            "so it carries no energy stamp by construction; supplying one by hand is "
            "what is refused. Non-energy level_stats fields remain valid."
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


#: The "was this structural knob asked for?" predicate, kept under its local name
#: for the guards below. It moved next to ``resolve_partition_spec`` for #1550,
#: when the gsplats ``partition=``-beside-a-ladder gates needed the same question
#: one level above the leaf — see :func:`~luxar.core.group.partition.is_requested`
#: for the ``False`` sentinel rule and the ``is``-vs-``==`` hazard behind it.
_requested = is_requested


def _reject_additive_lod_compositions(
    additive_lod: Any, substitutive_lod: Any, partition: Any
) -> None:
    """Refuse ``additive_lod=`` together with ``substitutive_lod=`` or ``partition=``.

    Same shape and placement as :func:`_reject_partition_with_substitutive_lod`, and
    for the same two reasons: the composition is a topology nothing writes yet, and
    mesh's branches RETURN in order, so accepting both would silently drop one of the
    two features rather than fail.

    Points and Lines DO compose additive with substitutive (an additive ladder inside
    each substitutive level — see ``lod/group.py``'s "Composed axes" section) and
    with partition (a per-part ladder). Neither composition is implemented for mesh,
    so both are refused by name and the messages say so plainly rather than implying
    the combination is meaningless: a per-shell ladder inside each decimated level,
    and a per-tile ladder inside each BSP part, are both coherent things to want.

    Two messages, not one, because the two obstacles are different — the substitutive
    one needs a ladder built per decimated level (whose face count and reveal center
    differ from the source's), the partition one needs a ladder built per part (whose
    faces are already re-indexed once).
    """
    if _requested(additive_lod) and _requested(substitutive_lod):
        raise ValueError(
            "additive_lod= and substitutive_lod= cannot be combined for a mesh "
            "(additive-under-substitutive — a reveal ladder inside each DECIMATED "
            "level — is not implemented). Points and Lines do compose the two; for a "
            "mesh each decimated level would need its own shell ladder built from its "
            "own faces, which nothing does yet. Use one or the other: "
            "substitutive_lod= to make the surface genuinely coarser when it is "
            "small on screen, additive_lod= to grow it in as it loads."
        )
    if _requested(additive_lod) and _requested(partition):
        raise ValueError(
            "additive_lod= and partition= cannot be combined for a mesh "
            "(a kind=partition of per-part reveal ladders is not implemented). "
            "Points and Lines do compose the two — each BSP part ladders "
            "independently — but a mesh part's faces have already been re-indexed "
            "into that part's own vertex table, and laddering it would re-index them "
            "again per shell, which nothing does yet. Use one or the other: "
            "partition= for frustum culling, additive_lod= for progressive reveal."
        )


def _reject_partition_with_substitutive_lod(
    partition: Any, substitutive_lod: Any
) -> None:
    """Refuse ``partition=`` together with ``substitutive_lod=``.

    Lines carries the same refusal. Points instead supports a global overview LOD
    above partitioned fine detail; Mesh does not implement that topology yet.
    Mesh needs it for one extra reason — the substitutive branch RETURNS before the
    partition branch is reached, so accepting both would silently drop the split.

    ``False`` is an explicit no-op sentinel on BOTH sides, so neither trips this —
    see :func:`~luxar.core.group.partition.is_requested`, which is where that rule
    and the ``is``-vs-``==`` hazard behind it are stated. Refusing either would
    refuse a call that asked for exactly one of the two features.

    Its own function rather than an inline ``if`` because ``add_mesh_impl`` sits at
    the C901 limit the complexity ratchet enforces — the same reason
    :func:`_resolve_mesh_vertices` and :func:`_validate_partition_sources` were
    hoisted out of it.
    """
    if _requested(partition) and _requested(substitutive_lod):
        raise ValueError(
            "partition= and substitutive_lod= cannot be combined yet "
            "(Points supports a global overview LOD above partitioned fine detail; "
            "Mesh does not implement that topology yet). Use one or the other."
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
    uvs: Any,
    shading: Optional[str],
    double_sided: bool,
    labels: Any,
    keys: Any = None,
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

    Runs the CHILD's ``extend_to_all`` preflight and throws the result away,
    purely to keep the fail-fast pre-write gate intact: it runs again inside
    ``child_0``, but by then every level has been decimated and ``add_lod_group``
    has created the zarr group, so a bad value surfaced as a childless kind=lod
    group in an incomplete store rather than as a clean refusal that wrote
    nothing. Same validator the children run, so the two cannot disagree about
    what is accepted.

    The attrs gate itself (``validate_render_attrs``) no longer lives here — it
    moved to the top of ``add_mesh_impl`` (#1534), above this dispatch, so it now
    also outranks ``extend_to_all`` resolution on every mesh path (matching
    Points/Lines: see the module-level `#1529`/`#1534` note in
    ``tests/group/lod/test_source_validation.py``).

    ``extend_to_all`` is guarded on ``is not None`` because that branch is the one
    that emits the advisory candidate warning, which must fire exactly once.
    """
    if substitutive_lod is None:
        return None

    from ..lod.mesh import resolve_substitutive_axis_mesh

    preflight_extend_to_all(scene, extend_to_all, vert_arr, "mesh")

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
        uvs=uvs,
        shading=shading,
        double_sided=double_sided,
        labels=labels,
        keys=keys,
        image_labels=image_labels,
        parent=parent,
        extend_to_all=extend_to_all,
        scalar_data_range=scalar_data_range,
        spec=substitutive_spec,
        scene=scene,
        **attrs,
    )


def _reject_texture_conflicts(texture: Any, uvs: Any, colors: Any, attrs: Any) -> None:
    """Refuse the texture combinations a single base colour cannot express.

    Mesh admits exactly ONE base-colour source. `colors` and `colormap` were
    already mutually exclusive (:func:`_reject_colors_colormap_conflict`); a
    texture is the third arm of the same rule, not an independent channel that
    modulates the others. Tinting a texture per-vertex is a reasonable thing to
    want and is deliberately NOT what this does — it would need its own shader
    variant and its own composition semantics, so it stays a follow-up rather
    than an accident of leaving the gate open.

    The `uvs`/`texture` pairing is the mesh peer of `normals`/`normal_dims`
    (§3.4): each is meaningless alone. A texture with no UVs has no mapping and
    would sample one arbitrary texel across every triangle; UVs with no texture
    describe a mapping into nothing and cost a per-vertex array to say so. Both
    are refused rather than warned, because both render *something* — the failure
    is silent, which is the case §3.4 already argues must be made loud.
    """
    if texture is not None and colors is not None:
        raise ValueError(
            "Cannot specify both 'texture' and 'colors'. A mesh has one base "
            "colour source: per-vertex colours, a colormap over scalars, or a "
            "texture. (Per-vertex tinting OF a texture is not implemented.)"
        )
    if texture is not None and attrs.get("colormap") is not None:
        raise ValueError(
            "Cannot specify both 'texture' and 'colormap'. A mesh has one base "
            "colour source: per-vertex colours, a colormap over scalars, or a "
            "texture."
        )
    if texture is not None and uvs is None:
        raise ValueError(
            "'texture' requires 'uvs'. Without texture coordinates there is no "
            "mapping from the surface into the image, so every triangle would "
            "sample the same arbitrary texel."
        )
    if uvs is not None and texture is None:
        raise ValueError(
            "'uvs' requires 'texture'. Texture coordinates describe a mapping "
            "into an image; with no image they cost a per-vertex array and "
            "affect nothing."
        )
    # The sampling attrs are refused on a mesh with no texture for the same
    # reason `reject_mesh_only_appearance` refuses them on a POINTS node: there
    # is nothing to sample, so the setting persists to the store, reads back
    # exactly as authored, and changes no pixel. A silent no-op that survives a
    # round trip is the hardest kind of mistake to notice.
    if texture is None:
        orphaned = sorted(k for k in ("texture_filter", "texture_wrap") if k in attrs)
        if orphaned:
            raise ValueError(
                f"{', '.join(repr(k) for k in orphaned)} "
                f"{'require' if len(orphaned) > 1 else 'requires'} 'texture'. "
                "Sampling attrs configure how a texture is read; with no texture "
                "they affect nothing."
            )


def _reject_texture_with_structural_routes(
    texture: Any, partition: Any, substitutive_lod: Any, additive_lod: Any
) -> None:
    """Refuse a texture on the three structural routes, naming why for each.

    Not "meaningless" — each is a coherent thing to want, and each needs work
    nothing does yet, which is the distinction the sibling refusals in this module
    are careful to draw:

    * ``partition=`` — a part is a re-indexing, so its UVs gather through
      ``vertex_index`` like colours do (easy), but the IMAGE is node-level. Each
      part would either duplicate the whole texture on disk or need a shared
      sibling array with a reference from every part, and neither exists.
    * ``substitutive_lod=`` — a decimated level has its own vertices, so its UVs
      must be resampled at the collapse targets. ``luxar.mesh.decimate``
      re-derives normals but carries no UV interpolation.
    * ``additive_lod=`` — a reveal shell duplicates boundary vertices, so its UVs
      gather like colours (easy), but the texture would be stored once per shell
      and charged per level by the viewer's ladder budget.

    Deliberately not blocking for the motivating use case: a UV sphere is ~20k
    triangles and needs none of the three.
    """
    from ..partition import is_requested

    if texture is None:
        return
    for value, label, reason in (
        (partition, "partition", "each part would duplicate the whole image"),
        (
            substitutive_lod,
            "substitutive_lod",
            "a decimated level needs its UVs resampled at the collapse targets",
        ),
        (
            additive_lod,
            "additive_lod",
            "the image would be stored once per reveal shell",
        ),
    ):
        if is_requested(value):
            raise ValueError(
                f"'texture' cannot be combined with {label}= yet ({reason}). "
                "Write the textured surface as a plain mesh leaf; a UV sphere or "
                "a UV-mapped surface rarely needs either."
            )


def _reject_colors_colormap_conflict(colors: Any, scalars: Any, attrs: Any) -> None:
    """Refuse the two colour/colormap combinations no geometry type accepts.

    Extracted whole from ``add_mesh_impl``, with the messages byte-identical to the
    sibling adders' so a given mistake reads the same on all four — the same reason
    :func:`_resolve_mesh_vertices`, :func:`_validate_partition_sources` and
    :func:`_reject_partition_with_substitutive_lod` were hoisted: the adder body sits
    at the C901 limit the complexity ratchet enforces, and it now dispatches three
    structural branches.

    Stays a pre-write gate on the AUTHORED values, above every structural branch, so
    an invalid combination is refused identically whether the mesh is written as a
    plain leaf, a decimated ladder, a partition or a reveal ladder.
    """
    if colors is not None and attrs.get("colormap") is not None:
        raise ValueError(
            "Cannot specify both 'colors' and 'colormap'. Use one or the other."
        )
    if scalars is not None and attrs.get("colormap") is None:
        raise ValueError(
            "'scalars' requires a 'colormap' attribute to map values to colors."
        )


def _maybe_add_mesh_additive_lod(
    group: "Group",
    *,
    name: str,
    vert_arr: np.ndarray,
    faces_arr: np.ndarray,
    normals: Any,
    normal_dims: Optional[Sequence[int]],
    colors: Any,
    scalars: Any,
    uvs: Any,
    shading: Optional[str],
    double_sided: bool,
    labels: Any,
    keys: Any = None,
    image_labels: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[List[str]],
    scalar_data_range: Optional[tuple[float, float]],
    additive_lod: Any,
    scene: Any,
    **attrs: Any,
) -> Optional[Mesh]:
    """Dispatch the additive (reveal) ladder branch, or ``None`` to fall through.

    Its own function for the same reason :func:`_maybe_add_mesh_substitutive_lod` is
    one: the resolve → order → split → hand-off sequence is one step, and hoisting it
    keeps ``add_mesh_impl`` under the C901 limit the complexity ratchet enforces.

    Returns ``None`` — meaning "write a plain leaf" — in three cases, each of which
    is the sibling adders' behaviour for the same input:

    * nothing was requested (``None`` / ``False``);
    * the mesh carries ``labels``, ``image_labels`` or ``keys`` (see the warning below);
    * the ladder came out with one level or none, which is a ladder in name only.
    """
    if not _requested(additive_lod):
        return None

    from ....mesh.split import split_mesh_by_faces
    from ..lod.mesh import make_additive_lod_mesh, resolve_additive_axis_mesh
    from ..lod.reveal import resolve_reveal_spatial_dims

    if labels is not None or image_labels is not None or keys is not None:
        # PRECEDENT: ``adders/points.py``'s image_labels guard, which resolves the
        # spec, discards it and warns. Refuse the LADDER, not the labels: fall
        # through to the single-leaf write, which forwards both channels. This
        # branch only fires for an EXPLICIT request (``_requested`` excludes
        # ``None``/``False``), so the caller asked for something that cannot be
        # honoured and must be told.
        #
        # Mesh degrades for ALL THREE per-element channels where Points degrades
        # only for ``image_labels``, and the extra ones are structural rather than
        # a missing writer feature: a Points ladder writes one union CSR over the
        # concatenated levels, but a mesh level RE-INDEXES its own vertices, so a
        # source vertex on a shell boundary occupies a slot in several levels and
        # the union index space is ill-defined. ``keys`` is in the list for
        # exactly the reason ``labels`` is — same CSR, same undefined index space
        # — and omitting it did not produce a ladder with keys, it produced a
        # ladder that dropped them without a word (#1917). See ``write_mesh_multi_lod``, which
        # refuses a labelled level outright rather than pairing text with the
        # wrong vertices.
        #
        # Resolve and discard so a malformed ``additive_lod=`` fails fast here
        # exactly as it would without labels.
        resolve_additive_axis_mesh(additive_lod)
        channels = " and ".join(
            n
            for n, v in (
                ("labels", labels),
                ("image_labels", image_labels),
                ("keys", keys),
            )
            if v is not None
        )
        warnings.warn(
            f"'{name}': the requested reveal ladder cannot be honoured "
            f"({channels} is set); the surface will load all-at-once. A mesh level "
            "re-indexes its own vertices, so there is no single index space for a "
            "union label CSR to describe.",
            UserWarning,
            stacklevel=2,
        )
        return None

    additive_spec = resolve_additive_axis_mesh(additive_lod)
    if additive_spec is None:  # pragma: no cover - _requested already excluded these
        return None

    # Fail-fast pre-write gate, run BEFORE the faces are reshaped or used to gather
    # anything: ``face_centroids`` indexes the vertex table with them, and numpy
    # WRAPS a negative index while gathering (so ``-1`` would quietly become the last
    # vertex) while an out-of-range or float index surfaces as a bare IndexError from
    # inside the centroid gather instead of the guided message the same input gets
    # without ``additive_lod=``. Handed the AUTHORED ``faces_arr`` rather than a
    # reshaped view, because a size not divisible by 3 must fail with the validator's
    # message and not numpy's reshape error — the same order, and the same helper,
    # as the partition path's ``_validate_partition_sources``.
    validate_mesh_arrays(
        vert_arr,
        faces_arr,
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
        uvs=uvs,
        shading=shading,
        double_sided=double_sided,
        labels=None,
    )
    faces2d = faces_arr.reshape(-1, 3)

    face_groups = make_additive_lod_mesh(
        vert_arr,
        faces2d,
        method=additive_spec["method"],
        n_lods=additive_spec["n_lods"],
        counts=additive_spec["counts"],
        reveal_center=additive_spec.get("reveal_center"),
        # Shells must not grow along a stacked time/channel column, which has
        # positional extent exactly like a spatial axis and so cannot be told apart
        # by the scorer's own extent rule. The scene can: a stacked axis is a
        # NON-DISPLAYED dimension. Same resolver the Points/Lines adders use.
        spatial_dims=resolve_reveal_spatial_dims(
            additive_spec, scene, int(vert_arr.shape[1])
        ),
    )
    if len(face_groups) <= 1:
        # One level (or none — an empty mesh) is not a ladder: fall through to the
        # plain leaf, exactly as the Points and Lines branches do.
        return None

    # The re-indexing that makes each level independently drawable. The face groups
    # are a true partition of the faces by construction (``make_additive_lod_mesh``
    # slices one permutation), which is precisely the contract
    # ``split_mesh_by_faces`` enforces — so its partition check is a free
    # cross-check of the ladder rather than a second requirement.
    parts = split_mesh_by_faces(faces2d, face_groups)

    return add_mesh_multi_lod_wrapper_impl(
        group,
        name=name,
        vert_arr=vert_arr,
        parts=parts,
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
        shading=shading,
        double_sided=double_sided,
        parent=parent,
        extend_to_all=extend_to_all,
        scalar_data_range=scalar_data_range,
        method=additive_spec["method"],
        counts=additive_spec["counts"],
        **attrs,
    )


def add_mesh_multi_lod_wrapper_impl(
    group: "Group",
    *,
    name: str,
    vert_arr: np.ndarray,
    parts: List["MeshPart"],
    normals: Any,
    normal_dims: Optional[Sequence[int]],
    colors: Any,
    scalars: Any,
    shading: Optional[str],
    double_sided: bool,
    parent: Optional["Node"],
    extend_to_all: Optional[List[str]],
    scalar_data_range: Optional[tuple[float, float]],
    method: str,
    counts: Any = None,
    **attrs: Any,
) -> Mesh:
    """Write a Mesh node with an additive (reveal) ladder of ``additive_<i>/`` levels.

    The Mesh peer of ``add_points_multi_lod_wrapper_impl``. Produces one
    ``<path>/additive_<i>/`` subgroup per level, coarsest (innermost shell) first;
    the parent carries ``n_additive_sublods``, the summed ``n_vertices`` /
    ``n_faces``, a global ``position_bounds`` and the compositing attrs. The returned
    :class:`Mesh` is that parent — the user's logical "one node" — and the viewer's
    progressive loader forms level *i* by concatenating levels 0..i.

    Two things differ from the element geometries, both because a triangle is three
    REFERENCES into a shared vertex table rather than a row of its own:

    * A level is not a slice. ``parts`` are already-re-indexed
      :class:`~luxar.mesh.split.MeshPart`\\ s, so each per-vertex channel is gathered
      through that level's ``vertex_index`` (the same helper and the same reason as
      the partition wrapper), and the level's own renumbered ``faces`` are written.
      A vertex on a shell boundary is therefore stored once per level that touches
      it — the honest cost of independently drawable levels, logged below via
      :func:`~luxar.mesh.split.duplication_factor`.
    * There is no per-FACE channel to carry, so ``MeshPart.face_index`` is unused
      here. It exists for callers that have one.

    **No energy stamps, and not by omission.** ``level_stats`` / ``lod_stats`` ARE
    written — they carry the ladder's provenance (``lod_method``, ``lod_level``,
    ``lod_n_elements``, ``lod_cumulative_n``) — but
    :func:`~luxar.core.group.lod.group.additive_level_stats` suppresses
    ``energy_fraction_cum`` and ``reference_energy`` for every reveal method, so a
    mesh ladder cannot acquire them. That suppression is the mechanism, not a
    convention: the viewer's ``1/e(k)`` brightness compensation is gated on the
    BLENDING MODE and never on geometry type, and applied to a reveal it would blow
    out the innermost shell and then dim it as the surface completes — the exact
    inverse of growing in. The all-zero energies passed below say the same thing a
    second way: a mesh has no per-element energy at all, because a triangle's
    brightness is a property of the surface it belongs to.
    """
    from ....mesh.split import duplication_factor
    from ..lod.group import additive_level_stats, breakpoints_kind_of

    n_vertices = int(vert_arr.shape[0])
    uniform_color = is_broadcast_color(colors)

    scene = group._find_scene()
    writer = group._require_scene_writer(scene)
    parent_node = parent or group
    path = f"{parent_node.path}/{name}" if parent_node.path else name

    # Ladder stamps. Counts are FACE counts: the ladder splits faces (a triangle is
    # the indivisible unit of a reveal, exactly as it is of the BSP), so that is the
    # currency ``lod_n_elements`` must be in. Energies are all-zero and the
    # ``energy_kind`` says why — see this function's docstring.
    per_level_stats, _reference_energy, parent_level_stats = additive_level_stats(
        [0.0] * len(parts),
        [int(p.faces.shape[0]) for p in parts],
        method=method,
        breakpoints_kind=breakpoints_kind_of(counts),
        energy_kind="mesh-reveal-no-energy",
    )
    # Assigned unconditionally rather than merged with a caller's dict (which is
    # what the Points wrapper has to do): ``_reject_energy_stamps`` runs in the adder
    # above and refuses a hand-supplied ``level_stats`` / ``lod_stats`` outright, so
    # there is nothing to merge with.
    attrs["level_stats"] = parent_level_stats

    # ONE display window for the whole ladder, stamped on every level — the same
    # rule, and the same helper, as the substitutive and partition wrappers. Each
    # level would otherwise stamp its own subset min/max, and the viewer windows a
    # node's colormap on that node's OWN stamped ``scalar_data_range``: the same
    # scalar value would then be a different colour in each shell, and a shell whose
    # subset is constant would stamp a degenerate ``[v, v]`` (viewer: LUT midpoint).
    # ``validate_mesh_arrays`` in the caller is this path's fail-fast scalars gate,
    # which the helper's docstring requires.
    field_range = _shared_scalar_window(scalar_data_range, scalars, n_vertices)

    levels: List[Dict[str, Any]] = []
    for level_i, part in enumerate(parts):
        take = part.vertex_index
        levels.append(
            {
                "vertices": vert_arr[take].astype(np.float32),
                "faces": part.faces,
                "normals": slice_optional_array(normals, take, n_vertices),
                # Only meaningful alongside the array, and the writer rejects it
                # without one.
                "normal_dims": normal_dims if normals is not None else None,
                # `colors` is the one channel classified by SHAPE rather than by
                # length: a broadcast RGB(A) sequence's own length can coincide with
                # the vertex count, and mesh is where that is reachable (a 4-vertex
                # surface with an RGBA colour), so `slice_optional_array` would
                # gather its four COMPONENTS as if they were four vertex rows and
                # hand each level a different rotated 3-slice — silently, since a
                # 3-element result is itself a valid uniform RGB.
                "colors": (
                    colors
                    if uniform_color
                    else slice_optional_array(colors, take, n_vertices)
                ),
                "scalars": slice_optional_array(scalars, take, n_vertices),
                "_scalar_data_range": field_range,
                "shading": shading,
                "double_sided": double_sided,
                "lod_stats": per_level_stats[level_i],
            }
        )

    # The ladder's levels concatenate into ONE node's buffers on the viewer side
    # and all stay resident, so the loader charges their SUM against a single
    # budget (`mesh-progressive-loader.ts`). The flat check only ever sees one
    # level, and a shell ladder duplicates every boundary vertex — so an
    # under-budget surface can still write a ladder the viewer refuses. This is
    # the one gap in that accounting that is multiplicative rather than a bounded
    # constant, which is why it is closed here rather than documented (#2145).
    from ....validation.base import validate_mesh_ladder_decode_budget

    validate_mesh_ladder_decode_budget(levels, context=f"mesh '{name}' reveal ladder")

    with asection(f"Additive-LOD mesh '{name}'"):
        aprint(
            f"📐 {len(parts)} reveal levels (method={method!r}, "
            f"face counts coarsest→finest={[int(p.faces.shape[0]) for p in parts]})"
        )
        aprint(
            f"🧩 Vertex duplication x{duplication_factor(parts):.3f} — a vertex on a "
            "shell boundary is stored once per level that touches it"
        )

    metadata = writer.write_mesh_multi_lod(
        path,
        levels,
        # Already resolved by the adder (mesh dispatches its structural branches
        # BELOW the `extend_to_all` resolution, unlike Points and Lines), so the
        # shared `resolve_ladder_extend_to_all` has nothing left to do here — and
        # calling it would re-emit the advisory the adder already emitted. An empty
        # list means "nothing requested"; the writer stamps only a non-empty one.
        extend_to_all=extend_to_all or None,
        **attrs,
    )

    # Mirror the writer's custom-colormap resolution (ndarray / matplotlib name →
    # 'custom') so the returned node matches what zarr stores.
    sync_custom_colormap_attr(attrs)

    # No `_notify_labels_added` / `_notify_image_labels_added`: a labelled mesh never
    # reaches this wrapper (the adder degrades it to a flat leaf with a warning).

    return Mesh(
        name,
        metadata=metadata,
        parent=cast(Any, parent_node),
        writer=writer,
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
    labels: Optional[Union[List[str], Sequence[str]]] = None,
    image_labels: Optional[Any] = None,
    keys: Optional[Union[List[str], Sequence[str]]] = None,
    partition: Any = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    substitutive_lod: Any = None,
    additive_lod: Any = None,
    **attrs: Any,
) -> Union[Mesh, "Group"]:
    # Outside the ``try`` for the same reason the siblings raise it there: an
    # argument error, not a write failure, so it must not be re-wrapped as
    # "Could not add mesh '<name>': …".
    _reject_partition_with_substitutive_lod(partition, substitutive_lod)
    _reject_additive_lod_compositions(additive_lod, substitutive_lod, partition)
    _reject_texture_with_structural_routes(
        texture, partition, substitutive_lod, additive_lod
    )
    try:
        # "An explicit None means absent" (#1574), applied ONCE here rather than
        # at each consumer: above the refusals below (none of which judges these
        # two keys), above the entry attrs gate, and above every structural
        # branch, so no route can see the raw None. Only render attrs are in the
        # set — the structural keys whose None also means absent
        # (``colors``/``labels``/``image_labels``/``partition``) are named params
        # of this function and can never reach ``**attrs``, which is exactly what
        # ``_reject_structure_params`` relies on too.
        strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_RENDER_ATTRS)

        # Fail-fast pre-write gate: reject invalid names (empty/'/'/dot-prefixed —
        # an empty name resolves to the zarr ROOT group and would clobber the
        # scene root) and duplicate siblings BEFORE any zarr write. Node.__init__
        # re-checks both post-write (belt and braces).
        from ....validation.base import validate_node_name

        validate_node_name(name)
        (parent or group)._ensure_no_duplicate_child(name)
        reject_mismatched_partition_parent(parent or group, "mesh", name)
        reject_layer_order_inside_specialized_group(
            "mesh", name, attrs, parent or group
        )
        _reject_structure_params(name, attrs)
        _reject_volumetric_blending(name, attrs)
        _reject_physical_material_conflicts(name, attrs, shading, texture)
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
        _reject_colors_colormap_conflict(colors, scalars, attrs)
        _reject_texture_conflicts(texture, uvs, colors, attrs)

        faces_arr: np.ndarray = (
            faces if isinstance(faces, np.ndarray) else np.asarray(faces)
        )
        n_faces = int(faces_arr.size // 3)

        # `dim_order` renumbers the vertex COLUMNS, and an orientation-reversing
        # permutation reflects space — so a triangle wound counter-clockwise in the
        # caller's own column order is clockwise in the scene's. Warn rather than
        # repair: `normal_dims` names SCENE dimensions, so a caller who followed the
        # contract literally wound against the scene frame and is already correct,
        # and flipping their faces would BREAK them. Only the caller knows which
        # frame they used. See `dim_order_reverses_winding` (#2141).
        warn_if_dim_order_reverses_winding(name, normal_dims, scene, dim_order)

        aprint(
            f"Adding mesh node '{name}' with {n_vertices:,} vertices and "
            f"{n_faces:,} faces in {ndim}D."
        )

        scene._validate_data_dimensions(vert_arr, name, data_type="vertices")

        # Node-attrs gate (#1534) — the Mesh peer of the Points/Lines hoist
        # (#1529). All THREE structural branches below (substitutive_lod=,
        # partition=, additive_lod=) forward the non-compositing remainder of
        # ``**attrs`` to a synthesised child (a decimated ``child_0``, a
        # re-indexed ``part_i``) or straight to the multi-LOD writer's own
        # UNRESERVED ``validate_render_attrs`` call (additive_lod=), so a bad
        # attr used to be refused late and, on the additive path specifically,
        # sometimes not at all: a genuinely mesh-reserved key that is not also
        # in ``_ALLOWED_NODE_ATTRS`` (``ordering=``) got the wrong verdict
        # (*unknown* instead of *reserved*), while ``position_bounds=`` —
        # reserved for mesh but ALSO generically allowed — didn't raise at
        # all: a real ladder was written and the writer's own stamp was
        # silently clobbered by the returned node's own construction,
        # breaking `finalize()` later far from the actual cause (see the
        # comment above ``TestMeshAdditiveNodeAttrsGate`` in
        # ``tests/group/lod/test_source_validation.py`` for the measured
        # repro). Below the colours and dimension gates above (same
        # precedence those already keep) and above every structural branch,
        # so nothing is written before it runs — including BEFORE
        # `_maybe_add_mesh_substitutive_lod`'s own `extend_to_all` preflight,
        # so an attrs fault now outranks an `extend_to_all` one on every mesh
        # path too, matching Points/Lines. The flat writer below still
        # validates the same dict once more inside ``write_mesh`` — the
        # validator is read-only, so running it here on the live ``attrs``
        # (not a copy) is safe and idempotent.
        #
        # MUST stay below the ``_scalar_data_range`` pop above (near the top
        # of this function): that private key is deliberately
        # absent from ``_ALLOWED_NODE_ATTRS`` (it never reaches a Node or a
        # caller), and it is real, caller-supplied input on the one live path
        # that sets it — ``luxar mesh lod`` re-authoring a scalars-carrying
        # mesh (``cli/mesh_ops/lod_commands.py``). Popped before this gate
        # runs, it is invisible here; moved below it, this gate would reject
        # it as an unknown attribute and break that command.
        #
        # NOT byte-identical to Points/Lines in one respect worth naming: this
        # gate sits below the FULL ``scene._validate_data_dimensions`` call
        # just above (which also emits the per-dimension out-of-range
        # ``UserWarning``), where Points/Lines hoist only the dimension COUNT
        # half above their own attrs gate and leave the range warning in
        # their flat write. A refused mesh call can therefore still emit that
        # warning where a refused points/lines call cannot — equivalent in
        # OUTCOME (both refuse, nothing written) but not identical in every
        # observable side effect. Deliberately left this way: hoisting only
        # the count half here (to make it identical) risks the #1446
        # warning-count controls, and it is only warnings, not a state
        # divergence.
        validate_render_attrs(attrs, reserved_attrs=MESH_RESERVED_ATTRS)

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
            uvs=uvs,
            shading=shading,
            double_sided=double_sided,
            labels=labels,
            keys=keys,
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
                keys=keys,
                image_labels=image_labels,
                parent_node=parent_node,
                split_axes=scene.dimensions.displayed,
                extend_to_all=final_extend_dims or extend_to_all,
                scalar_data_range=scalar_data_range,
                **partition_attrs,
            )
            if wrapper is not None:
                return wrapper
            # 1 part → fall through to the plain single-leaf write, exactly as
            # the sibling adders do. A wrapper around one part is pure overhead.

        # Additive-LOD branch — a REVEAL ladder of `additive_<i>/` levels inside a
        # single leaf, coarsest (innermost shell) first. Last of the three, and
        # deliberately so: `additive_lod=` is refused alongside either of the other
        # two up front (`_reject_additive_lod_compositions`), so by here both are
        # ruled out and this branch cannot mask a dropped split or ladder. It also
        # sits AFTER the 1-part partition fall-through, mirroring `add_points`, so
        # `partition=` that could not split still leaves the ladder reachable —
        # except that for mesh the composition guard means such a call never arrives.
        #
        # `extend_to_all` was RESOLVED into `attrs` above and this call also passes
        # it by name, so hand the ladder a copy without the key or the two collide as
        # a duplicate keyword argument (the same trap the partition call documents).
        ladder_attrs = {k: v for k, v in attrs.items() if k != "extend_to_all"}
        revealed = _maybe_add_mesh_additive_lod(
            group,
            name=name,
            vert_arr=vert_arr,
            faces_arr=faces_arr,
            normals=normals,
            normal_dims=normal_dims,
            colors=colors,
            scalars=scalars,
            uvs=uvs,
            shading=shading,
            double_sided=double_sided,
            labels=labels,
            keys=keys,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=final_extend_dims,
            scalar_data_range=scalar_data_range,
            additive_lod=additive_lod,
            scene=scene,
            **ladder_attrs,
        )
        if revealed is not None:
            return revealed

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
            keys=keys,
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
        # Un-nest BEFORE printing too, or arbol still echoes an internal child
        # name (`part_0` / `child_0`) that the raised exception no longer
        # names (#1491) — see funnel_add_error.
        inner = unnest_add_error("mesh", name, e)
        aprint(f"Failed to add mesh node '{name}': {inner}")
        raise ValueError(funnel_add_error("mesh", name, e)) from e


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
    uvs: Any,
    shading: Optional[str],
    double_sided: bool,
    labels: Any,
    keys: Any = None,
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
    ``kind=lod`` group, children coarsest→finest, viewport-relative
    ``coverage_fraction`` per child AND the group ``selector`` naming their units
    from :func:`luxar.core.group.lod.group.resolve_lod_ladder` — which calls
    ``derive_coverage_fractions`` underneath when no explicit
    ``coverage_fractions=`` list was given, so a ladder hand-placed under a
    ``kind=partition`` wrapper gets the fills-screen per-tile anchor, exactly as
    the Points/Lines wrappers do — compositing attrs on the group and everything
    else on the children.

    **Level targets are vertex counts**, ``V / K**i``, because that is the
    currency the decimator's search is expressed in. Triangle count would be an
    equally defensible proxy for rendered detail (and is roughly ``2V`` on a
    closed manifold), but mixing the two would mean asking for one and thresholding
    on the other.

    A requested level is DROPPED rather than written when it cannot be a real
    level: below the decimator's 4-vertex floor, or reducing to no fewer vertices
    than the level before it. Writing it anyway would put two identical surfaces
    in the ladder, each claiming its own halving of screen occupancy — so the
    viewer would swap between them and pay a load for nothing. (The thresholds
    come from the ladder's LENGTH, not from count ratios, so a duplicate count is
    not a duplicate threshold and nothing downstream objects.) If every level
    drops — a surface already too
    coarse to reduce — the ladder is abandoned and a plain leaf is written, which
    is the same degenerate-path behaviour the Points wrapper has.
    """
    from ....mesh.decimate import decimate_ladder
    from ..lod.group import resolve_coarsen_dims, resolve_lod_ladder

    # Fail-fast pre-write gate, part two: the ARRAYS, run BEFORE any decimation
    # and before `add_lod_group` creates the group. The adder already ran the
    # child's attr/extend_to_all gates on the way in; these are the rest of what
    # a child write checks, and without them a malformed channel was refused
    # only from inside a child write. Colours with two components, a
    # wrong-length scalars array, a typo'd `shading`, or a bad normals array
    # is validated identically for EVERY level, so it was refused from
    # whichever child's write hit it first (typically `child_0`, the
    # coarsest, since levels write coarsest-to-finest) — leaving the store
    # holding a `kind=lod` group with no children at all (if the first one
    # failed) or missing only its finest one (if a later level did), where
    # the plain-leaf path writes nothing.
    # `labels` and a wrong-length `image_labels` (#1491) fail a DIFFERENT way:
    # both are forwarded ONLY to the FINEST child, written last, so pre-fix
    # they were refused deep inside that child's OWN write — after its
    # vertices, faces, normals, colours and scalars were already on disk.
    # That left every level, the finest included, fully written and
    # independently loadable; only the finest level's own label channel was
    # silently missing — harder to notice than a missing level, not milder.
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
        uvs=uvs,
        shading=shading,
        double_sided=double_sided,
        labels=labels,
        keys=keys,
        image_labels=image_labels,
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

    targets = [
        n_vertices // (compression_factor**power)
        for power in range(levels, 0, -1)
        if n_vertices // (compression_factor**power) >= 4
    ]
    candidates = decimate_ladder(
        vert_arr,
        faces_arr.reshape(-1, 3).astype(np.uint32),
        target_vertices=targets,
        method=spec["method"],
        normals=normals if normals is not None else None,
        # Normals live in their own 3D FRAME, not necessarily the coarsening axes.
        normal_dims=tuple(normal_dims) if normal_dims is not None else None,
        colors=per_vertex_colors,
        scalars=per_vertex_scalars,
        spatial_dims=spatial_dims,
        attribute_weight=spec["attribute_weight"],
    )

    coarse: List[Any] = []
    previous = 0
    for level in candidates:
        count = int(level.vertices.shape[0])
        # Strictly between the previous (coarser) level and the original, or it
        # adds nothing: a duplicate would still be handed its own halving of
        # screen occupancy, so the viewer would swap between identical surfaces.
        if count <= previous or count >= n_vertices:
            continue
        coarse.append(level)
        previous = count

    if not coarse:
        weight_note = (
            f" A positive attribute_weight={spec['attribute_weight']} can prevent "
            "attribute-incompatible vertices from merging."
            if spec["attribute_weight"] > 0
            else ""
        )
        aprint(
            f"  📐 Substitutive-LOD '{name}': no level reduced the surface "
            f"({n_vertices:,} vertices at K={compression_factor}) — writing a "
            f"plain mesh leaf instead.{weight_note}"
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
            keys=keys,
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
    parent_node = parent or group
    # Thresholds AND the selector naming their units, from the one shared rule
    # (``lod.group.resolve_lod_ladder``): an explicit ``coverage_fractions=[...]``
    # is used verbatim under the legacy units it was authored in, otherwise the
    # screen-area halving ladder is derived — re-anchored at fills-screen when the
    # insertion point is partition-bound. ``add_mesh`` rejects ``partition=``
    # together with ``substitutive_lod=``, so a caller who wants per-tile mesh
    # ladders MUST hand-build the ``kind=partition`` wrapper and call this once per
    # part, which is how that anchor is reached here.
    coverage_vals, lod_selector = resolve_lod_ladder(
        spec.get("coverage_fractions"),
        counts,
        parent_node,
        name=name,
        length_error=lambda n_explicit, n_levels: (
            f"coverage_fractions has {n_explicit} entries but the LOD ladder "
            f"has {n_levels} levels ({len(coarse)} decimated + 1 original). "
            "Levels that could not reduce the surface are dropped, so the ladder "
            "can be shorter than the requested `levels`."
        ),
    )

    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    child_attrs.pop("coverage_fraction", None)
    caller_level_stats = child_attrs.pop("level_stats", None)
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

    aprint(
        f"  📐 Substitutive-LOD '{name}': {len(coarse)} decimated levels + original "
        f"(vertex counts coarsest→finest={counts}, K={compression_factor})"
    )
    lod_group_node = parent_node.add_lod_group(name, selector=lod_selector, **lod_attrs)

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
            level_stats={
                **(caller_level_stats if isinstance(caller_level_stats, dict) else {}),
                "geometric_error": level.geometric_error,
            },
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
        keys=keys,
        image_labels=image_labels,
        extend_to_all=extend_to_all,
        dim_order=None,
        fill=None,
        coverage_fraction=coverage_vals[-1],
        level_stats={
            **(caller_level_stats if isinstance(caller_level_stats, dict) else {}),
            "geometric_error": 0.0,
        },
        **child_attrs,
    )

    return lod_group_node


def _validate_partition_sources(
    faces_arr: np.ndarray,
    n_vertices: int,
    *,
    normals: Any,
    colors: Any,
    scalars: Any,
    labels: Any,
    keys: Any = None,
    image_labels: Any,
) -> None:
    """Run the plain-leaf write gates against the SOURCE arrays, before the split.

    Extracted whole from :func:`_add_mesh_partition` — the rejections are one
    cohesive step (everything that must fail before a single index is used to
    gather), and hoisting them keeps that function under the C901 limit the
    complexity ratchet enforces. Order is load-bearing and preserved exactly:
    ``image_labels`` first, then faces, then the per-vertex channels in
    ``normals``, ``colors``, ``scalars``, ``labels``, ``keys`` order — a call
    that trips several is told about the same one it was told about before.
    """
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
    if keys is not None:
        validate_labels_for_writing(keys, n_vertices, context="keys", noun="Keys")


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
    keys: Any = None,
    image_labels: Any,
    parent_node: "Node",
    split_axes: Sequence[int],
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
        bsp_leaf_parts,
        persist_pruned_bsp_tree,
        resolve_partition_spec,
        spatial_bsp_tree,
        warn_if_oversized_single_part,
    )

    # Same vocabulary as the sibling adders — and for a mesh ``max_elements``
    # counts FACES, not vertices (see :func:`resolve_partition_spec`).
    max_elements, rule = resolve_partition_spec(partition)

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
        keys=keys,
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
    uniform_color = is_broadcast_color(colors)

    faces2d = faces_arr.reshape(-1, 3)
    centroids = face_centroids(vert_arr, faces2d)
    tree = spatial_bsp_tree(centroids, max_elements, rule=rule, split_axes=split_axes)
    face_parts = bsp_leaf_parts(tree)

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
            keys=slice_optional_array(keys, take, n_vertices),
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
    # Unlike lines, split_mesh_by_faces returns one written part per BSP leaf.
    persist_pruned_bsp_tree(wrapper, tree.to_serializable(), range(len(parts)))

    # Union of the parts' bounds == the whole input's bounds, computed straight
    # from the source rather than round-tripped through the children's attrs.
    wrapper._persist_attr("position_bounds", position_bounds_from_array(vert_arr))
    return wrapper
