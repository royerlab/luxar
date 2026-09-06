"""Compositing primitives used by Group's partition-wrapping path.

These helpers are shared by the kind=partition and kind=lod wrapper builders
(see ``adders/`` and ``gsplats_pipeline/``). Most are pure data operations with
no Group/Node references; :func:`preflight_extend_to_all` accepts the owning
Scene solely for read-only validation before a wrapper write.

Exposed:

* :data:`COMPOSITING_ATTRS` — frozenset of attribute names that ride on
  a wrapper Group (where the user thinks of the wrapper as "their
  layer") rather than getting copied onto each internal child.
* :data:`WRITER_STAMPED_APPEARANCE_DEFAULTS` + :data:`IDENTITY_COMPOSITING_ATTRS`
  — the value the writer manufactures for an appearance attr nobody set, shared
  by every stamp site and by the reader that has to tell a stamp from an
  authored value (``gsplat merge``'s agreement rule).
* :func:`slice_optional_array` — slice an array-valued leaf parameter by
  index, leaving scalars / None / mis-sized inputs untouched.
* :func:`is_broadcast_color` — classify a uniform RGB(A) sequence (which must
  reach every part whole) apart from per-element color data.
* :func:`validate_labels_before_split` — reject a wrong-length ``labels``
  before any partition / LOD decomposition.
* :func:`validate_points_channels_before_split`,
  :func:`validate_lines_channels_before_split`,
  :func:`validate_gsplats_channels_before_split` — the same pre-split gate for
  every other per-element channel (colors / radii / widths / sharpness /
  scalars / amplitudes / Cholesky / image_labels), delegating to the writers'
  own sweeps.
* :func:`validate_line_indices_before_split` — the topology half of the Lines
  gate: the flat ``indexed`` layout/parity check, run before the channels.
* :func:`position_bounds_from_array` — per-axis min/max of an (N, D)
  position array, in the writer's shape.
* :func:`preflight_extend_to_all` — validate an explicit scene-level extension
  spec before a wrapper is written, while leaving the warning-producing
  ``None`` branch to each written child.
* :func:`strip_absent_attr_kwargs` + :data:`ABSENT_WHEN_NONE_RENDER_ATTRS` —
  delete the caller-named keys whose present-but-``None`` value means ABSENT,
  and the leaf adders' set of them (``colormap`` / ``coverage_fraction``). The
  set is a required argument: the gsplats pipeline passes its own wider one.
* :func:`sync_custom_colormap_attr` — mirror the writer's custom-colormap
  resolution (`ndarray / non-builtin name -> 'custom'`) into the adder's
  attrs dict so the returned node object matches what zarr stores.
* :func:`mirror_written_colormap` — copy the colormap the writer actually
  stamped (its ``"gray"`` default for a colorless gsplats leaf, or nothing at
  all when an ancestor authored a palette) onto the adder's attrs, so the
  returned node's attr write-back cannot contradict the store (#1600).
* :func:`reject_lines_only_join` — refuse the lines-only ``join`` attr on a
  points / gsplats / mesh leaf, where it would write cleanly and do nothing.
* :func:`reject_lines_only_join_assignment` — the same refusal for the second
  door into the same attr, the ``node.join = ...`` property setter.
* :func:`reject_layer_order_inside_specialized_group` — refuse ``layer_order``
  inside partition / LOD groups through adders; the matching post-hoc attrs
  door is guarded by ``core/node/node.py::_WriteThroughAttrs``.
* :func:`reject_mesh_only_appearance` — refuse mesh-only appearance attrs on
  points / lines / gsplats leaves and on Groups, where they do not compose. The
  second door into the same attrs (a post-hoc ``node.attrs[...] = ...``) is guarded
  by ``core/node/node.py::_WriteThroughAttrs._reject_mesh_only_on_non_mesh``.
* :data:`MESH_ONLY_APPEARANCE_ATTRS` — the mesh-only authored keys (the
  appearance and texture-sampling controls plus the slab-membership tolerance)
  refused on every non-mesh node by those two guards.
* :func:`unnest_add_error` — strip a SAME-geometry inner adder's own ``Could
  not add <geometry> '<child>': …`` prefix from a caught exception's message,
  so a refusal from inside a synthesised same-kind split child (``child_3``,
  ``part_0``) reads the same as the flat path's refusal for the same input.
* :func:`funnel_add_error` — build an adder's outer ``Could not add <geometry>
  '<name>': …`` message from :func:`unnest_add_error`'s un-nested inner text.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ...validation.writing import (
    validate_gsplat_inputs,
    validate_line_indices,
    validate_lines_channels,
    validate_points_channels,
)

#: The four addable geometry words — the only tokens a nested funnel prefix is
#: ever allowed to name. Deliberately excludes ``group``: ``Node.add_lod_group``
#: / ``add_partition_group`` build their OWN ``Could not create child … group
#: '<x>': …`` prefix ahead of a wrapper-creation failure, and that prefix must
#: never be mistaken for (or stripped as) a geometry adder's — see
#: :func:`funnel_add_error`.
_GEOMETRY_WORDS = ("points", "lines", "mesh", "gsplats")

#: The mesh-only authored keys refused on every non-mesh node by the
#: adder/group and write-through guards.
#:
#: Twenty-one are appearance: the five house-shader controls, the two
#: texture-sampling ones, the ``material`` family selector and the thirteen
#: physically based knobs it unlocks — seven surface knobs and the six-knob glass
#: family (``MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.1, §3.4) — which is what the
#: name records. ``slab_tolerance`` is the exception and is
#: deliberately here anyway: it is a LOADING knob — the half-width of the nD
#: membership slab a continuous hidden dimension is culled against (spec §5.2.1)
#: — but it is mesh-only for exactly the same reason and must be refused on the
#: other three types by exactly the same guard. A second frozenset would be a
#: second thing to forget. Pinned equal to the validator table's keys by
#: ``io/tests/test_compiler_improvements.py``.
MESH_ONLY_APPEARANCE_ATTRS = frozenset(
    {
        "alpha_cutoff",
        "ambient",
        "attenuation_color",
        "attenuation_distance",
        "clearcoat",
        "clearcoat_roughness",
        "dispersion",
        "ior",
        "iridescence",
        "material",
        "metalness",
        "roughness",
        "shade_exponent",
        "sheen",
        "sheen_color",
        "shininess",
        "slab_tolerance",
        "specular",
        "texture_filter",
        "texture_wrap",
        "thickness",
        "transmission",
    }
)

#: Matches the prefix an adder's own funnel produces, e.g.
#: ``Could not add points 'child_3': ...``. Anchored to the start of the
#: string so it only strips an ACTUAL nested funnel prefix, never a substring
#: that happens to appear mid-message, and the token is restricted to
#: :data:`_GEOMETRY_WORDS` so a ``Could not add group '<x>': …`` prefix (an
#: unrelated wrapper-creation failure, never a geometry adder's own) can never
#: match.
_NESTED_ADD_ERROR_PREFIX_RE = re.compile(
    r"^Could not add (" + "|".join(_GEOMETRY_WORDS) + r") '[^']*': "
)


def preflight_extend_to_all(
    scene: Any,
    extend_to_all: Optional[Union[List[str], str]],
    positions: Any,
    data_type: str,
) -> None:
    """Validate an explicit scene-level spec before a wrapper is written.

    The explicit branches are position-independent and can be judged once at
    the wrapper door. ``None`` stays leaf-only because its candidate analysis
    emits one advisory warning per written child.
    """
    if extend_to_all is not None:
        scene._resolve_extend_to_all(extend_to_all, positions, data_type)


def unnest_add_error(geometry: str, name: str, exc: BaseException) -> str:
    """Strip a SAME-geometry inner ``Could not add <geometry> '<child>': …``
    prefix from ``exc``'s message, returning the UN-NESTED inner text alone
    (never re-prefixed with the caller's own ``geometry``/``name`` — that is
    :func:`funnel_add_error`'s job, built on top of this).

    Every ``add_points`` / ``add_lines`` / ``add_mesh`` / ``add_gsplats``
    funnels every failure inside its ``try`` block through one
    ``raise ValueError(f"Could not add {geometry} '{name}': {e}") from e``. On
    a ``partition=`` split, on the Mesh ``substitutive_lod=`` ladder, or on the
    FINEST child of a Points/Lines ``substitutive_lod=`` ladder (the one child
    of that ladder written by the SAME adder — its coarse siblings are lifted
    to gsplats, see below), a child level is written by calling the SAME adder
    again for a synthesised child name (``child_3``, ``part_0``) — so when
    that recursive call fails, its own funnel has ALREADY produced
    ``Could not add <geometry> '<child>': …`` before the outer call catches it
    and wraps it a second time: ``Could not add points 'p': Could not add
    points 'part_1': Image labels length (200) must match element count
    (400)``. That doubly indirects the caller away from the actual fault and
    names an internal node they never passed and have no way to address
    (issue #1491).

    Stripping the inner prefix — and, in :func:`funnel_add_error`, re-prefixing
    with the CALLER's own ``geometry``/``name`` — makes the split path's
    message identical to what the flat (non-split) path raises for the same
    invalid input — the "same verdict" parity this module's docstrings already
    promise between the two write paths.

    The strip is CONDITIONAL on the inner token matching this call's OWN
    ``geometry``, which is the whole point rather than an edge case: on the
    Points/Lines ``substitutive_lod=`` ladder the coarse children are
    GSPLATS nodes (the lift target), so an outer ``add_points`` call's inner
    failure is genuinely ``Could not add gsplats 'child_0': …`` — a DIFFERENT
    geometry's own fault, not a same-kind recursion artefact. Un-nesting that
    would hide which geometry actually failed and, worse, could relabel it
    with an attribute the outer geometry does not even recognise (a gsplats-
    only attr like ``amplitude_range`` reported as a Points fault). Only a
    same-geometry inner prefix — the actual recursion-into-itself case this
    function exists for — is stripped; a cross-geometry or a
    ``Could not add group '<x>': …`` inner prefix (see
    :data:`_GEOMETRY_WORDS`) is left exactly as raised.

    Only a message whose token matches the CALLER's ``geometry`` is touched;
    any other message (a plain ``ValueError`` from a leaf write, a
    ``TypeError``, a cross-geometry or ``group`` inner prefix, ...) passes
    through byte-for-byte, so an unrelated error is never mangled.

    Not every raise site funnels through this. The ones that do are the four
    ``add_{points,lines,mesh,gsplats}_impl`` outer ``except`` blocks — see
    ``adders/{points,lines,mesh,gsplats}.py``. A handful of OTHER sites build
    the identical ``Could not add <geometry> '<name>': …`` prefix by hand and
    do not call this function: ``gsplats_pipeline/lod_dispatch.py``,
    ``gsplats_pipeline/from_data.py``, ``gsplats_pipeline/from_io.py`` (two
    sites), and ``scene/scene.py``. None of those is reachable as an INNER
    exception this function would need to un-nest today — they are the
    outermost raise on their own call paths — so the gap is latent, not a
    live bug; it means only that a future recursive call THROUGH one of them
    would need its own :func:`funnel_add_error` call, same as the four
    adders. This helper also does not reach the wrapper-GROUP creation chain:
    e.g. ``scene.add_points("g", pos, substitutive_lod=True,
    transform=[1,2,3])`` still raises ``Could not add points 'g': Could not
    create child kind=lod group 'g': Could not add group 'g': Could not
    create child group 'g': Invalid transform for node 'g': …`` — four levels
    of nesting, none of it un-nested by this function (the innermost failure
    is a ``group``, which :data:`_GEOMETRY_WORDS` excludes by design). That
    chain is a separate, still-open problem; this function closes only the
    geometry-adder same-kind recursion case #1491 is about.

    Args:
        geometry: The geometry kind for THIS raise site — ``"points"`` /
            ``"lines"`` / ``"mesh"`` / ``"gsplats"`` — matching what the flat
            write would say for the same node type.
        name: The node name the CALLER passed to this adder call (never a
            synthesised child/part name — those only ever appear on an INNER
            call, which is exactly what gets stripped when the inner geometry
            matches this one). Unused by this function directly (the strip
            only inspects ``exc``'s own message), but taken for symmetry with
            :func:`funnel_add_error` and so a future geometry-aware strip rule
            has somewhere to use it.
        exc: The caught exception. Only its ``str()`` is used; the call site
            keeps its own ``raise ... from e`` so the full chain (including
            this exact exc) is still available to a debugger/traceback.

    Returns:
        ``exc``'s message with a same-geometry inner ``Could not add …``
        prefix removed, or unchanged if there is none.
    """
    message = str(exc)
    match = _NESTED_ADD_ERROR_PREFIX_RE.match(message)
    if match and match.group(1) == geometry:
        message = message[match.end() :]
    return message


def funnel_add_error(geometry: str, name: str, exc: BaseException) -> str:
    """Build the ``Could not add <geometry> '<name>': …`` message, un-nested.

    Thin wrapper around :func:`unnest_add_error`: re-prefixes its un-nested
    inner text with THIS call's own ``geometry``/``name``, producing the
    string an adder's outer ``except`` block passes to ``ValueError(...)``.
    See :func:`unnest_add_error` for why the un-nesting is needed and exactly
    which prefix it strips.

    Idempotent, but not because the regex has "nothing left to match" on a
    second pass — it does: this function's OWN output is
    ``Could not add <geometry> '<name>': <message>``, which matches the same
    pattern :func:`unnest_add_error` strips. Idempotence instead holds because
    every un-nesting is immediately followed by re-prefixing with the SAME
    ``geometry``/``name`` that were just stripped, so running the result
    through this function again with the same arguments strips exactly what
    it just added back and reproduces the identical string.

    Args:
        geometry: The geometry kind for THIS raise site — ``"points"`` /
            ``"lines"`` / ``"mesh"`` / ``"gsplats"`` — matching what the flat
            write would say for the same node type.
        name: The node name the CALLER passed to this adder call.
        exc: The caught exception — forwarded to :func:`unnest_add_error`
            unchanged.

    Returns:
        The message to pass to ``ValueError(...)``.
    """
    return f"Could not add {geometry} '{name}': {unnest_add_error(geometry, name, exc)}"


#: Attrs that ride on a kind=lod / kind=partition wrapper Group (where the user
#: thinks of the wrapper as "their layer") rather than getting copied onto
#: each internal child. Compositing semantics (opacity, gamma, ...) flow
#: down to the children through Group inheritance at render time, so writing
#: them once on the parent is correct. ``truncation_radius`` is deliberately
#: NOT compositing: the writer auto-defaults it per leaf, which under
#: nearest-ancestor-wins would shadow a parent's setting.
#:
#: ``colormap`` is not here either, but for a weaker reason now that it DOES
#: compose in the viewer (#1600 — see :data:`AUTHORED_APPEARANCE_ATTRS`):
#: copying it onto each child of a wrapper the user built with
#: ``add_gsplats(partition=…, colormap=…)`` is still correct (a colormap on
#: every leaf and a colormap on their wrapper render identically), and the
#: Layers panel's ``deriveColormapFromDescendants`` reads the panel row's
#: palette back out of exactly that shape. Moving it would be a behaviour
#: change with no user-visible gain.
COMPOSITING_ATTRS = frozenset(
    {
        "transform",
        "opacity",
        "absorption",
        "gamma",
        "intensity",
        "offset",
        "blending_mode",
        # The authored cross-layer draw order. Compositing for the strongest
        # version of the blending_mode reason: the wrapper IS the layer, so a
        # partitioned / LOD node must carry ONE level for the whole block.
        # Copying it onto each internal child would split the wrapper across
        # draw-order bands and destroy its exact BSP part order — which is why
        # authoring one strictly inside a specialized group is refused outright
        # (:func:`reject_layer_order_inside_specialized_group`) rather than
        # merely discouraged. See docs/guides/specs/LAYER_ORDER_SPEC.md.
        "layer_order",
        # Lines-only, but compositing for the same reason blending_mode is: the
        # user thinks of the wrapper as their layer, so a partitioned / LOD
        # lines node must not silently drop back to the default join style on
        # every internal child.
        "join",
        "layer",
        "visible",
        "nd_transform",
    }
)


#: Attrs a STRUCTURE-ONLY rebuild (``gsplat lod`` and friends: the input is
#: re-laddered / re-tiled, the appearance is not the command's business) must
#: carry from the source root to the output root. Without this the reduction's
#: fresh nodes know nothing about the input and the writer's own defaults take
#: over — ``blending_mode`` vanishes and the multiplicative attrs snap back to
#: their identity (#1600).
#:
#: :data:`COMPOSITING_ATTRS` minus ``transform`` (see below), because those are
#: precisely the attrs a root-level stamp actually reaches the leaf with. The
#: viewer's ``composeAttrs`` (``viewer/src/data/attrs-composer.ts``) MULTIPLIES
#: opacity/absorption/gamma/intensity and ADDS offset along the root→leaf chain,
#: so a per-level identity stamp composes to the root's authored value; and
#: ``blending_mode``/``join`` are nearest-SETTER-wins with no per-leaf default
#: stamped, so the root's choice wins.
#:
#: ``transform`` is EXCLUDED for a different reason — carrying it corrupts it.
#: The stored value is already COLUMN-major (THREE.js), and the leaf writer runs
#: whatever it is handed through ``prepare_transform_for_zarr``, which reads its
#: input as ROW-major and transposes. Handing the stored list straight back
#: transposes it a SECOND time on a leaf-rooted result (``flat``/``stream``/
#: ``levels``): an authored translation lands in the bottom row and the command
#: dies on ``validate_transform`` ("bottom row must be [0, 0, 0, 1]", measured
#: exit 1 where the un-carried command exited 0), and a rotation is silently
#: INVERTED. A group-rooted result (``adaptive``/``tiles``) writes caller attrs
#: verbatim and does round-trip — so one carry would mean two different things
#: depending on the recipe. Carrying it needs the writer to tell an
#: already-stored transform from an authored one; tracked with the rest of the
#: sweep in #1600, and the reason the ``gsplat`` rebuilds leave the attr alone
#: (the same status quo as before the carry existed).
#:
#: ``colormap`` is carried too, even though it is not in
#: :data:`COMPOSITING_ATTRS` (the wrapper-vs-children ROUTING question is a
#: different one — see that set's own note). It used to be excluded because a
#: root stamp was SHADOWED and therefore only LOOKED preserved: the writer
#: manufactured a ``"gray"`` on every colorless leaf, which sits nearer the
#: leaf than the root, and nothing composed the attr anyway. Both halves are
#: fixed (#1600): ``apply_gsplat_group_attrs`` now stamps the gray default only
#: when no ancestor authored a palette (see ``inherited_gsplat_colormap`` for
#: the two write paths' mechanics), and the viewer composes ``colormap``
#: nearest-setter-wins root→leaf like ``blending_mode``/``join``
#: (``viewer/src/data/attrs-composer.ts``). So a root stamp now genuinely
#: reaches every leaf.
#:
#: ALSO DELIBERATELY EXCLUDED, because a root stamp would be SHADOWED and
#: therefore only look preserved:
#:
#: * ``amplitude_data_range`` / ``scalar_data_range`` — not composed, and each
#:   level re-derives its own from its (post-reduction) values, which sits
#:   nearer the leaf than the root. The gsplat window harmonization
#:   (``finalize/amplitude_window.py``) does not change that: it only ever
#:   rewrites windows on LEAVES, so a root stamp on a group-rooted result
#:   survives untouched — and is still shadowed by every leaf's own. Unlike
#:   ``colormap``, dropping the per-leaf value is NOT the fix: the window is a
#:   property of that leaf's own values, so composing it nearest-setter-wins
#:   across a ``kind=lod`` boundary reintroduces exactly the basis mismatch
#:   ``leafScalarWindow`` / ``composedWindowIsInReferenceBasis``
#:   (``viewer/src/ui/layers/layer-apply.ts``) had to gate. Carrying it needs
#:   that basis gate applied at COMPOSE time — a separate change, still tracked
#:   in https://github.com/royerlab/luxar/issues/1600.
#: * ``truncation_radius`` — auto-defaulted per leaf by design (see the note on
#:   ``COMPOSITING_ATTRS``); each leaf already carries the source value through
#:   ``GSplatData.truncation_radius``, so the footprint survives anyway.
AUTHORED_APPEARANCE_ATTRS = (COMPOSITING_ATTRS - {"transform"}) | {"colormap"}


#: The value the WRITER manufactures for an appearance attr the author never
#: set — the single source of truth for every stamp site, so a reader that has
#: to tell "the author chose this" from "nobody chose anything" cannot drift
#: from the writer that produced the file.
#:
#: Stamped by three places, all of which read their value from here:
#:
#: * :func:`~luxar.io._compiler.node_common.apply_default_render_attrs` and
#:   :func:`~luxar.io._compiler.gsplat_assembly.apply_gsplat_group_attrs` —
#:   :data:`IDENTITY_COMPOSITING_ATTRS`, unconditionally, on every leaf.
#: * ``apply_gsplat_group_attrs`` again for ``colormap`` — but only on a
#:   COLORLESS leaf with no ancestor palette, so this one is conditional and a
#:   colored store legitimately carries no ``colormap`` at all.
#: * ``gsplats/io/save_gsplats.py`` for ``layer``, on a standalone
#:   ``.gsplats.zarr`` root (the file IS the layer when opened directly).
#:
#: ``blending_mode`` / ``visible`` / ``nd_transform`` / ``join`` are absent
#: here on purpose: they have no identity value, so nothing is stamped for
#: them and their absence on disk is genuine silence.
#: Read-only (``MappingProxyType``): it is the single source of truth several
#: modules index into, and a stamp site that mutated it would silently redefine
#: what "the author never set this" means for the reader.
WRITER_STAMPED_APPEARANCE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "opacity": 1.0,
        "absorption": 1.0,
        "gamma": 1.0,
        "intensity": 1.0,
        "offset": 0.0,
        "layer": True,
        "colormap": "gray",
    }
)


#: The compositing attrs with an identity value, in the order the writers stamp
#: them. Multiplicative (opacity/absorption/gamma/intensity) or additive
#: (offset) no-ops under the viewer's hierarchical composition, which is what
#: makes stamping them on every leaf harmless — and is also why they cannot be
#: distinguished from a deliberate authored identity (see
#: :data:`WRITER_STAMPED_APPEARANCE_DEFAULTS`).
IDENTITY_COMPOSITING_ATTRS = (
    "opacity",
    "absorption",
    "gamma",
    "intensity",
    "offset",
)


def lines_only_join_reason(geometry_type: str) -> str:
    """The one explanation of why ``join`` is refused on a non-lines leaf.

    Shared verbatim by both surfaces that can reach the attr — the adders
    (:func:`reject_lines_only_join`) and the geometry classes' ``join`` setters
    (:func:`reject_lines_only_join_assignment`) — so a user who trips either one
    gets the same account of what ``join`` is and where it belongs.
    """
    return (
        "'join' is a lines-only attribute — it selects the join style at a "
        f"degree-2 polyline joint, and a {geometry_type} node has no polyline "
        "joints to style, so the viewer would ignore it. Remove it, or set it "
        "on a lines node (or on a Group above one, from where it composes down)."
    )


def reject_lines_only_join_assignment(
    geometry_type: str, name: str, value: Any
) -> None:
    """Refuse ``node.join = ...`` on a non-lines leaf (issue #790).

    :func:`reject_lines_only_join` closes the add-time door; this closes the
    assignment-time one, which the ``join`` property inherited from
    :class:`~luxar.core.node.node.Node` would otherwise leave wide open one line
    later (``set_join`` assigns through the property, so it is covered too).
    Called from the ``join`` setter overrides on ``Points`` / ``GSplats`` /
    ``Mesh``, exactly as the mesh ``volumetric`` blending refusal is.
    """
    raise ValueError(
        f"Cannot set join={value!r} on {geometry_type} '{name}'. "
        + lines_only_join_reason(geometry_type)
    )


def reject_lines_only_join(
    geometry_type: str, name: str, attrs: Dict[str, Any]
) -> None:
    """Refuse ``join=`` on a non-lines LEAF (issue #790).

    ``KNOWN_RENDER_ATTRS`` in ``io/_compiler/node_common.py`` is one set shared by
    all four geometry writers, so ``add_points(..., join="none")`` wrote a dead
    ``join`` into a points ``.zattrs`` with no warning at all — contradicting both
    the format spec ("Optional, LINES ONLY") and this module's own note above.

    Only a LEAF is refused. A ``join`` on a Group — including a ``kind=partition``
    / ``kind=lod`` wrapper — is correct and must keep working: it is a compositing
    attr precisely so it can be authored once on the wrapper and inherited by the
    lines descendants.

    Refused at the ADDER rather than in the shared writer gate, which is
    deliberately geometry-blind, and rather than in :func:`validate_line_join`,
    which validates the VALUE and never sees the geometry type. Same shape and
    same bar as the mesh ``volumetric`` refusal: a caller mistake with no valid
    interpretation raises rather than warns.
    """
    if "join" in attrs:
        raise ValueError(
            f"Cannot add {geometry_type} '{name}' with join={attrs['join']!r}. "
            + lines_only_join_reason(geometry_type)
        )


def _enclosing_specialized_group(node: Any) -> Optional[Any]:
    """The nearest ``kind=partition`` / ``kind=lod`` ancestor of ``node``, if any.

    Walks the whole parent chain, unlike
    :func:`~luxar.core.group.partition.reject_mismatched_partition_parent`, which
    only inspects the IMMEDIATE parent — a level authored two levels down (on a
    part's own LOD wrapper, say) is exactly as damaging as one on the part.
    ``node`` itself is included, so passing a wrapper reports that wrapper.
    """
    current = node
    while current is not None:
        kind = None
        try:
            kind = current.attrs.get("kind")
        except Exception:  # pragma: no cover - a detached//partial node
            kind = None
        if kind in ("partition", "lod"):
            return current
        current = getattr(current, "parent", None)
    return None


def layer_order_inside_specialized_group_reason(kind: str) -> str:
    """Why ``layer_order`` cannot be authored inside a partition / LOD group."""
    if kind == "partition":
        return (
            "A partition's parts are ordered EXACTLY against each other from the "
            "stored BSP planes, valid from any camera pose. layer_order bands are "
            "the outer key, so a level on one part would move it into a different "
            "band and its parts would interleave by band instead of by the tree, "
            "destroying that exactness. Set layer_order on the partition WRAPPER "
            "instead — the wrapper is the layer, and a level there moves the whole "
            "block while the part order travels with it intact."
        )
    return (
        "A kind=lod group's levels are ALTERNATIVES — only one renders at a time — "
        "so a level on one of them would be inert. Set layer_order on the LOD "
        "wrapper instead, where it applies to whichever level is live."
    )


def reject_layer_order_inside_specialized_group(
    geometry_type: str, name: str, attrs: Dict[str, Any], parent: Any
) -> None:
    """Refuse ``layer_order=`` on a node inside a partition / LOD group.

    ``layer_order`` may participate only when authored on a node that IS a layer
    — a plain group or a top-level leaf. A value on the scene root is accepted as
    carrier metadata and ignored. The two reasons differ in severity (a partition
    would lose an exactness guarantee; a LOD level would be inert), but the rule
    is one ancestry check, which also covers the nested case (a LOD group inside
    a partition part) with no extra branch.

    Refused rather than ignored, and at the adder rather than in the value
    validator, for the same reasons as :func:`reject_lines_only_join`: the
    validator never sees the tree, and an attr that writes cleanly and silently
    does nothing is this codebase's most expensive failure mode. The viewer, which
    must render whatever it is handed, instead warns once and uses one level for
    the whole group, keeping the first member's — strict write, tolerant read.
    """
    if "layer_order" not in attrs:
        return
    wrapper = _enclosing_specialized_group(parent)
    if wrapper is None:
        return
    kind = str(wrapper.attrs.get("kind"))
    raise ValueError(
        f"Cannot add {geometry_type} '{name}' with "
        f"layer_order={attrs['layer_order']!r} inside a kind={kind} group. "
        + layer_order_inside_specialized_group_reason(kind)
    )


def reject_mesh_only_appearance(
    node_type: str, name: str, attrs: Dict[str, Any]
) -> None:
    """Refuse mesh-only appearance attrs on non-mesh nodes."""
    invalid = sorted(MESH_ONLY_APPEARANCE_ATTRS & attrs.keys())
    if invalid:
        raise ValueError(
            f"Cannot add {node_type} '{name}' with mesh-only attribute(s) {invalid}. "
            "The viewer applies these attributes only to mesh leaves, and they do "
            "not compose through Groups. Remove them, set them on each mesh leaf "
            "(part_<i> / child_<i>), or pass them to add_mesh(...), which stamps "
            "every generated mesh leaf."
        )


# The RENDER attrs for which a present-but-``None`` value means ABSENT. Exactly
# the two whose None is caught by no value validator and therefore reaches disk:
# ``validate_render_attrs`` guards its colormap check on ``is not None``, and
# ``coverage_fraction`` (an LOD selector threshold) has no validator at all. Both
# are ACCEPTED with a None and then write something WRONG, which is why they are
# here and the rest of the render attrs are not: measured, ``opacity=None``,
# ``blending_mode=None``, ``layer=None``, ``visible=None``, ``gamma=None``,
# ``intensity=None``, ``offset=None``, ``absorption=None`` and
# ``truncation_radius=None`` each refuse outright with a "must be convertible to
# float / must be a boolean / …, got NoneType" (their validators run
# unconditionally), so for those a None is loud and reading it as "absent" would
# only mask typos.
#
# Spelled here rather than in any one adder because every door that can be handed
# a stray None needs the identical answer — each of the four leaf adders (#1574)
# and the three ``**attrs``-forwarding gsplats pipeline doors (#1496), whose own
# wider ``ABSENT_WHEN_NONE_ATTRS`` is derived from this tuple rather than
# repeating it.
ABSENT_WHEN_NONE_RENDER_ATTRS = ("colormap", "coverage_fraction")


def strip_absent_attr_kwargs(attrs: Dict[str, Any], keys: Sequence[str]) -> None:
    """Delete every ``keys`` entry of ``attrs`` whose value is ``None``.

    One rule, stated once: for a key whose ABSENT case has a working default, an
    explicit ``None`` means "absent", and the whole fix is to drop the key before
    anything downstream looks at it. Which keys those are is the caller's
    question, not this function's — :data:`ABSENT_WHEN_NONE_RENDER_ATTRS` for a
    leaf adder, the wider ``gsplats_pipeline.from_data.ABSENT_WHEN_NONE_ATTRS``
    (which adds that adder's structurally-forwarded leaf params) for the
    ``add_gsplats_from_data`` / ``add_gsplats_from_file`` / graft doors. The set
    is a required argument precisely so neither door can silently inherit the
    other's.

    What goes wrong without it, measured. ``colormap=None`` is the worst case
    because it is SILENT: the key survives ``validate_render_attrs`` (whose
    colormap check is guarded on ``is not None``) and then
    :func:`sync_custom_colormap_attr` rewrites the None to ``'custom'`` (it is
    not a str in ``BUILTIN_COLORMAP_NAMES``) WITHOUT writing any
    ``colormap_lut`` — so the node ships a LUT-less custom colormap, and the
    viewer's ``build-scene-graph.ts`` reacts to that by warning and falling back
    to VIRIDIS, where omitting the key gives ``gray``. ``coverage_fraction=None``
    persists a literal ``coverage_fraction: null`` LOD selector threshold into
    the node's zarr attrs. On the gsplats pipeline's wider set the same shape
    also STRANDS: a present-but-None ``labels`` / ``partition`` is rejected by
    NAME by ``validate_render_attrs``, which never looks at the value, so an
    idiomatic ``labels=maybe_labels`` left a childless ``kind=lod`` wrapper on
    disk (#1471).

    Mutates in place and returns None: every caller owns the dict it passes (its
    own ``**attrs``), and handing back a copy would only invite one of them to
    forget to use it.
    """
    for key in keys:
        if key in attrs and attrs[key] is None:
            del attrs[key]


def sync_custom_colormap_attr(attrs: Dict[str, Any]) -> None:
    """Sync ``attrs['colormap']`` with what the compiler wrote to zarr.

    The writer resolves any non-builtin colormap — an ndarray LUT or a
    matplotlib/colorcet name — to a ``colormap_lut`` dataset plus
    ``colormap='custom'`` (``io/_compiler/colormap.py``), but it mutates its
    OWN copy of the attrs (the ``**attrs`` packing boundary), so the adder
    must mirror the substitution for the node object it returns. No-op when
    ``colormap`` is absent or a builtin name.
    """
    if "colormap" not in attrs:
        return
    from ...colormaps.builtins import BUILTIN_COLORMAP_NAMES

    cm = attrs["colormap"]
    if not isinstance(cm, str) or cm not in BUILTIN_COLORMAP_NAMES:
        attrs["colormap"] = "custom"


def mirror_written_colormap(attrs: Dict[str, Any], writer: Any, path: str) -> None:
    """Copy the colormap the WRITER actually stamped onto an adder's attrs.

    The compiler manufactures ``colormap="gray"`` on a colorless gsplats leaf,
    and the adders mirror that onto the node object they return so the
    in-memory node matches zarr. The mirror is not cosmetic: the returned
    node's attrs are written straight back through ``Node.__init__`` →
    ``write_group``, so a mirror that stamps a gray the writer DECLINED puts it
    on disk after all.

    Since #1600 the writer declines whenever an ancestor authored a palette
    (a nearer gray would shadow it under the viewer's nearest-setter-wins
    composition — see
    ``io._compiler.gsplat_assembly.inherited_gsplat_colormap``). Re-deriving
    that rule here would be a second implementation that can disagree: the
    in-memory parent chain cannot see attrs written through the RAW compiler
    API (``compiler.write_group("/", colormap=…)``), while the writer's store
    walk can. So read back the decision instead of reproducing it.

    No-op when the leaf already carries an explicit ``colormap`` (nothing to
    mirror), and for a writer with no zarr-shaped ``store`` (a stub in a test).

    Args:
        attrs: The adder's attrs dict, mutated in place.
        writer: The writer the leaf was just written through.
        path: The leaf's store-relative node path.
    """
    if "colormap" in attrs:
        return
    store = getattr(writer, "store", None)
    if store is None:
        return
    try:
        written = store[path.lstrip("/")].attrs.get("colormap")
    except (KeyError, TypeError, AttributeError, IndexError):
        return
    if written is not None:
        attrs["colormap"] = written


def slice_optional_array(value: Any, indices: np.ndarray, n_elements: int) -> Any:
    """Slice an array-valued leaf parameter by index; pass non-per-element values through.

    Used by the ``partition=`` wrapping path on the leaf adders. Returns
    unchanged when:
      * ``value`` is ``None`` or a scalar (``int`` / ``float`` / ``bool``
        / ``str``) — applies uniformly to every part.
      * ``value`` is a 0-D array.
      * ``value``'s first-axis length doesn't match ``n_elements`` (e.g.
        a 3-vector RGB broadcast, or a length-1 sentinel).
    Slices the first axis when the input is a list of length
    ``n_elements`` (string labels) or an array whose first axis matches.
    """
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, list):
        if len(value) == n_elements:
            return [value[i] for i in indices]
        return value
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if arr.ndim == 0:
        return value
    if arr.shape[0] == n_elements:
        return arr[indices]
    return value


def validate_labels_before_split(labels: Any, n_elements: int) -> None:
    """Reject a wrong-length ``labels`` BEFORE any partition / LOD decomposition.

    The companion guard to :func:`slice_optional_array`, which passes a list whose
    length does not match ``n_elements`` through **unchanged** rather than slicing
    it (that pass-through is deliberate — it is how broadcast values reach every
    part). For labels that is a trap: every part / LOD level would receive the
    same unsliced list, and the write would SUCCEED with labels in the wrong
    slots. The downstream per-part / per-level length checks cannot catch it,
    because a level's own length may coincidentally match. So the full-count check
    has to happen upstream of the split.

    Two callers reach this directly now. :func:`validate_gsplats_channels_before_split`
    is the one geometry whose channel validator does not cover labels — no OTHER
    wrapper impl calls this on its own account; Points and Lines get the
    equivalent check from the writer sweeps their gates delegate to
    (``validate_labels_for_writing``, last in the flat order). The second is
    ``gsplats_pipeline.from_io._validate_labelled_leaf_length``, called from the
    graft door's ``_reject_labels_on_a_grafted_wrapper`` (#1505) — a genuine
    EXCEPTION to the rule below, not another instance of it: that call site is a
    pre-WRAPPER gate, not a pre-split one (a bare-leaf graft builds no wrapper and
    triggers no split at all), and it DELIBERATELY changes which fault a
    multi-fault call reports — pinned by
    ``test_a_wrong_length_label_outranks_an_unknown_attr_here`` — for the same
    trade :func:`validate_points_channels_before_split` already sanctions below
    for "a NaN position, an unknown attr".

    Every OTHER caller's check belongs to a wrapper's pre-split gate — entering a
    wrapper is exactly "a split is about to happen" — and deliberately NOT to the
    top of a leaf adder: the plain-leaf path validates in the writer, and
    hoisting the check above the adder's positions/attr gates (and above the
    range half of ``_validate_data_dimensions``, which still runs only in the
    single-leaf write) would change which error a multi-fault call reports. The
    count half of that validator is one acknowledged exception — #1446 moved it
    to the top of every leaf adder, so a wrong column count outranks this gate on
    both paths, by design. The node-attrs gate (``validate_render_attrs``) is a
    second: since #1529 it runs at the Points/Lines adder entry, above THEIR OWN
    channel gates, so an attrs fault outranks a channel one on those two paths
    too. GSplats — the one geometry that actually reaches THIS function, via
    :func:`validate_gsplats_channels_before_split` — gained the identical entry
    gate later, in #1534: before that it had no such gate and an attrs fault on
    ``add_gsplats(partition=..., labels=<wrong length>, ...)`` was reported from
    behind this label check instead of ahead of it; now it outranks this gate
    too, matching Points/Lines. Same reasoning, and the same house rule, as
    ``adders/mesh.py::_validate_partition_sources``.

    No-op when ``labels`` is ``None``.

    Args:
        labels: The caller's ``labels`` argument (per-point for Points, per-splat
            for GSplats, per-vertex for Lines).
        n_elements: The node's FULL element count, before any decomposition.

    Raises:
        ValidationError: If ``labels`` is not a sequence of one string per element.
    """
    if labels is None:
        return
    from ...validation.base import validate_labels_for_writing

    validate_labels_for_writing(labels, n_elements)


def is_broadcast_color(colors: Any) -> bool:
    """Whether ``colors`` is a uniform RGB(A) sequence rather than per-element data.

    Classifies on TYPE/SHAPE only — a list/tuple of 3 or 4 numeric components,
    which is exactly the admission test
    :func:`~luxar.io._compiler.node_common.validate_broadcast_color` applies at
    the writer (on the flat path a list/tuple ``colors`` is ALWAYS the broadcast
    form). Values are deliberately left to that validator, so a bad uniform
    color fails with the same message it gets without ``partition=``.

    Needed because a uniform color's OWN length can collide with the element
    count: a 3-point node with ``colors=(1.0, 0.0, 0.0)`` satisfies
    :func:`slice_optional_array`'s length test and is gathered as if its three
    components were three point rows.

    On Points / Lines / GSplats the consequence is a SPURIOUS REJECTION, not a
    silent mis-write: those parts are disjoint, so with 3 or 4 elements split
    over at least two parts the slice lengths sum to at most 4 and some part
    always gets a length outside ``{3, 4}``, which its writer refuses. Which
    refusal you get depends on list vs tuple, because :func:`slice_optional_array`
    keeps a list a list and ``np.asarray``s a tuple. Measured without this
    classifier: 3 points + an RGB *list* raises "Uniform color must have 3 (RGB)
    or 4 (RGBA) components, got 1" at every cap, where the same triple as a
    *tuple* raises "Expected shape (1, 3) or (1, 3), got (1,)" from the array
    validator instead. And 4 points + an RGBA *list* under ``midpoint`` / ``sah``
    (which split them 3 + 1, where the default ``median`` splits 2 + 2) raises
    from ``part_1`` with ``part_0`` ALREADY WRITTEN, carrying the RGB and the
    authored alpha dropped — a legal input refused, sometimes only after
    stranding a partial node. Mesh,
    which solved this first, is the one geometry where it can be silent instead:
    its parts SHARE vertices, so two parts can each take a valid 3-of-4 slice
    (see the rationale in ``adders/mesh.py``). Same rule, both places.

    Only a list/tuple can be the broadcast form at all, which is why numpy colors
    of every shape classify as ``False`` here (the writer refuses a 1-D numpy
    color outright and wants ``(1, c)``). A list of triples classifies as
    ``False`` too, because its entries are sequences rather than numbers — but it
    is not thereby "gathered normally": a list/tuple ``colors`` is ALWAYS the
    broadcast form to the writer, so ``[[1.0, 0.0, 0.0]] * 200`` is refused on
    both paths ("Uniform color must have 3 (RGB) or 4 (RGBA) components, got
    200"), and a 3-long list of triples is refused for its components ("component
    0 must be a finite number"). Per-element colors are an ndarray.
    """
    if not isinstance(colors, (list, tuple)) or len(colors) not in (3, 4):
        return False
    return all(isinstance(c, (int, float, np.integer, np.floating)) for c in colors)


def validate_points_channels_before_split(
    n_points: int,
    *,
    colors: Any = None,
    radii: Any = None,
    sharpness: Any = None,
    scalars: Any = None,
    labels: Any = None,
    image_labels: Any = None,
    keys: Any = None,
) -> None:
    """Run the flat Points write gate over every per-point channel, pre-split.

    The generalisation of :func:`validate_labels_before_split` to the rest of
    the channels (issue #1437). Same trap, same reasoning: a wrong-length
    per-point array takes :func:`slice_optional_array`'s pass-through branch, so
    every part / LOD level receives the whole unsliced array, and a part whose
    own element count happens to equal that array's length ACCEPTS it — the
    write succeeds with values paired to the wrong points.

    This does not re-implement the checks: it calls
    :func:`~luxar.io._compiler.geometry_writers.points.validate_points_channels`,
    which IS the writer's own step-0d/0e sweep (colors → radii → sharpness →
    scalars → labels → keys → image labels), just against the source count. Sharing one
    implementation is deliberate — a channel added to the writer's gate is
    covered here the same day, so this gate cannot drift from what the child
    write accepts. Every legal broadcast form the flat path accepts therefore
    passes THIS GATE too (a scalar radius, a ``(1, c)`` colors row, an RGB
    triple) — and reaches disk on every path, ``substitutive_lod=`` included:
    the gsplat lift broadcasts a uniform ``colors`` onto the coarse levels,
    alpha column and all, rather than refusing it as it did before #1444. A
    per-element ``(N, 4)`` RGBA is still refused by the lift (the substitutive
    merge is untested on a varying alpha) — but that is not a broadcast form, so
    it is not this gate's parity promise.

    ``image_labels`` (#1491) does not fit the ``slice_optional_array`` trap
    above at all — it has no per-part/per-level SLICER in the first place. On
    ``substitutive_lod=`` the value is forwarded ONLY to the finest child
    (Points itself), which the wrapper writes LAST, after every coarse gsplat
    level is already committed. Pre-fix, the length/index check ran INSIDE
    ``write_image_labels_csr``, near the very end of that finest child's own
    write — AFTER its ``type``, ``n_points``, positions, radii, colors,
    sharpness, scalars and ``labels`` were already on disk. (``finalize()``,
    which back-fills a ``kind=lod`` wrapper's own ``position_bounds`` from its
    finest child, runs only from ``LuxarZarrCompiler.finalize()`` — AFTER
    every ``add_*`` call has returned — so it never ran on this path: the
    raising ``add_points`` call propagated the error straight out, and the
    wrapper node was left with no ``position_bounds`` of its own at all.) So a
    wrong-length ``image_labels`` left a COMPLETE, fully
    loadable N-level ``kind=lod`` ladder on disk — every level present, every
    other array on every level present — silently missing only the
    ``image_label_offsets`` / ``image_label_bytes`` the caller actually asked
    for. That is HARDER to notice than a missing level, not milder: the
    ladder loads and renders exactly like a smaller, correctly-authored object
    that simply has no images, rather than announcing a failed write.
    (``partition=`` refuses ``image_labels`` outright, and an additive ladder
    falls back to a single leaf when it is set, so the substitutive wrapper is
    the only one exposed to this.)

    The CHANNEL verdict is identical with and without a wrapper. Note the gate
    runs ABOVE the positions checks on the split paths, so a call that ALSO
    trips a bad position (a NaN) reports the channel fault first here and the
    positions fault on the plain-leaf path. Both refuse, and neither writes.
    The scene-DIMENSION count is one exception: since #1446 every leaf adder
    checks it above its split branches, so a wrong column count is reported
    first on BOTH paths and this gate is never reached (see
    :func:`validate_labels_before_split`). The node-attrs gate is a second
    exception here: since #1529 ``validate_render_attrs`` also runs at the
    Points/Lines adder entry, above this gate, so an unknown/reserved attr
    wins there too instead of reporting the channel fault.

    Call as the FIRST statement of a wrapper impl, never from a leaf adder — see
    :func:`validate_labels_before_split` for why the placement is load-bearing.

    Args:
        n_points: The node's FULL point count, before any decomposition.
        colors: Per-point ``(n, 3|4)`` array, a ``(1, c)`` broadcast row, or a
            uniform RGB(A) list/tuple.
        radii: Per-point array, ``(1,)`` broadcast array, or scalar.
        sharpness: Per-point array, ``(1,)`` broadcast array, or scalar.
        scalars: Per-point array, ``(1,)`` broadcast array, or scalar.
        labels: One string per point.
        image_labels: Per-point images (dense sequence or sparse dict) — only
            ever carried by the finest child of a ``substitutive_lod=``
            ladder; see the paragraph above for why it needs this gate
            specifically.
        keys: One machine-readable string per point.

    Raises:
        ValidationError: If any channel is not a legal per-point or broadcast
            value for ``n_points`` elements.
    """
    validate_points_channels(
        n_points,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
        image_labels=image_labels,
    )


def validate_lines_channels_before_split(
    n_vertices: int,
    *,
    widths: Any,
    colors: Any = None,
    sharpness: Any = None,
    scalars: Any = None,
    labels: Any = None,
    image_labels: Any = None,
    keys: Any = None,
) -> None:
    """Run the flat Lines write gate over every per-vertex channel, pre-split.

    The Lines twin of :func:`validate_points_channels_before_split` — read that
    docstring for the trap this closes (including the ``image_labels``
    paragraph: the substitutive wrapper forwards it only to the finest child,
    written LAST, so this gate is what keeps a wrong length from stranding a
    truncated ``kind=lod`` ladder), for why the implementation is shared with
    the writer rather than repeated, and for the exact scope of the
    identical-verdict promise. All seven channels are per-VERTEX (not
    per-segment), and ``widths`` is required, so it is validated first and
    unconditionally. The topology half of the same gate is
    :func:`validate_line_indices_before_split`, which must run BEFORE this one.

    Args:
        n_vertices: The node's FULL vertex count, before any decomposition.
        widths: Per-vertex array, ``(1,)`` broadcast array, or scalar.
        colors: Per-vertex ``(n, 3|4)`` array, a ``(1, c)`` broadcast row, or a
            uniform RGB(A) list/tuple.
        sharpness: Per-vertex array, ``(1,)`` broadcast array, or scalar.
        scalars: Per-vertex array, ``(1,)`` broadcast array, or scalar.
        labels: One string per vertex.
        image_labels: Per-vertex images (dense sequence or sparse dict) —
            only ever carried by the finest child of a ``substitutive_lod=``
            ladder; see :func:`validate_points_channels_before_split` for why
            it needs this gate specifically.
        keys: One machine-readable string per vertex.

    Raises:
        ValidationError: If any channel is not a legal per-vertex or broadcast
            value for ``n_vertices`` elements.
    """
    validate_lines_channels(
        n_vertices,
        widths=widths,
        colors=colors,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
        image_labels=image_labels,
    )


def validate_line_indices_before_split(
    indices: Any, n_vertices: int, line_type: str
) -> None:
    """Reject a malformed ``indexed`` edge list BEFORE any split touches it.

    The topology half of the Lines pre-split gate, and it runs FIRST — mesh
    validates ``faces`` before any per-vertex channel for exactly this reason
    (``adders/mesh.py``): a bad edge list makes every channel verdict moot.

    Needed because the split paths do not go through the writer's ``indexed``
    gate before they interpret the edges. ``lod.lines.identify_polylines``
    checks dtype and bounds and then does ``reshape(-1, 2)``, so an ``(E, 3)``
    array was reinterpreted as ``3E/2`` edges the author never wound and WROTE
    CLEANLY (the flat writer refuses it), and an odd flat count raised a raw
    ``cannot reshape array of size 23 into shape (2)`` instead of the guided
    "even element count" message. Called from each split branch of
    ``add_lines`` immediately before that branch's topology builder, which is
    the first consumer — the wrapper impls are too late for the raw-reshape
    case.

    No-op unless ``line_type == "indexed"`` with a non-``None`` ``indices``; the
    "requires indices array" refusal stays where it already is on each path.

    Raises:
        ValueError: If the edge list is not a flat ``(2E,)`` or ``(E, 2)``
            integer array with an even element count and in-bounds indices.
    """
    if line_type != "indexed" or indices is None:
        return
    validate_line_indices(indices, n_vertices)


def validate_gsplats_channels_before_split(
    centers: np.ndarray,
    amplitudes: Any,
    cholesky_factors: np.ndarray,
    *,
    colors: Any = None,
    labels: Any = None,
    keys: Any = None,
) -> bool:
    """Run the flat GSplats write gate over the source arrays, pre-split.

    The GSplats twin of :func:`validate_points_channels_before_split`. The whole
    amplitudes / Cholesky / colors trio is checked by one validator
    (:func:`~luxar.io._compiler.gsplat_assembly.validate_gsplat_inputs`), which
    derives the splat count from ``centers`` itself — so handing it the SOURCE
    arrays yields the flat verdict, including the uniform ``(k,)`` Cholesky and
    scalar-amplitude broadcast forms.

    Args:
        centers: The node's full ``(N, D)`` centers array.
        amplitudes: Per-splat ``(N,)`` array or a scalar.
        cholesky_factors: Per-splat ``(N, k)`` array or a uniform ``(k,)`` one.
        colors: Per-splat ``(N, 3|4)`` array, a ``(1, c)`` broadcast row, or a
            uniform RGB(A) list/tuple.
        labels: One string per splat.
        keys: One stable string key per splat.

    Returns:
        ``cholesky_is_uniform`` — whether ``cholesky_factors`` is the uniform
        ``(k,)`` form, straight from the validator that already decided it. The
        caller needs this to skip slicing that array, and recomputing the rule at
        the call site would be one more copy of exactly the kind of duplication
        this gate exists to remove.

    Raises:
        ValueError: If the trio's shapes/values are not a legal per-splat or
            broadcast combination for ``len(centers)`` splats.
        ValidationError: If ``labels`` or ``keys`` is not one string per splat.
    """
    from ...validation.base import validate_labels_for_writing

    (*_normalized, n_splats, _n_dims, cholesky_is_uniform) = validate_gsplat_inputs(
        centers, amplitudes, cholesky_factors, colors
    )
    validate_labels_before_split(labels, n_splats)
    if keys is not None:
        validate_labels_for_writing(keys, n_splats, context="keys", noun="Keys")
    return bool(cholesky_is_uniform)


def position_bounds_from_array(positions: np.ndarray) -> Dict[str, List[float]]:
    """Per-axis min/max of an ``(N, D)`` position array, in the writer's shape.

    Matches what the compiler's ``_compute_position_bounds`` writes onto
    each leaf node, so the partition-kind wrapper's ``position_bounds`` is
    the same shape as its children's. Used by the ``partition=`` wrapping
    path to compute the parent bbox directly from the source array
    instead of round-tripping through the per-leaf zarr writes.
    """
    if positions.size == 0:
        raise ValueError("Cannot compute position_bounds from empty array")
    return {
        "min": positions.min(axis=0).astype(float).tolist(),
        "max": positions.max(axis=0).astype(float).tolist(),
    }
