"""Compositing primitives used by Group's partition-wrapping path.

These helpers are shared by the kind=partition and kind=lod wrapper builders
(see ``adders/`` and ``gsplats_pipeline/``). They are pure data
operations — no Group/Node references — and have no side effects.

Exposed:

* :data:`COMPOSITING_ATTRS` — frozenset of attribute names that ride on
  a wrapper Group (where the user thinks of the wrapper as "their
  layer") rather than getting copied onto each internal child.
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
  scalars / amplitudes / Cholesky), delegating to the writers' own sweeps.
* :func:`validate_line_indices_before_split` — the topology half of the Lines
  gate: the flat ``indexed`` layout/parity check, run before the channels.
* :func:`position_bounds_from_array` — per-axis min/max of an (N, D)
  position array, in the writer's shape.
* :func:`sync_custom_colormap_attr` — mirror the writer's custom-colormap
  resolution (`ndarray / non-builtin name -> 'custom'`) into the adder's
  attrs dict so the returned node object matches what zarr stores.
* :func:`reject_lines_only_join` — refuse the lines-only ``join`` attr on a
  points / gsplats / mesh leaf, where it would write cleanly and do nothing.
* :func:`reject_lines_only_join_assignment` — the same refusal for the second
  door into the same attr, the ``node.join = ...`` property setter.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

#: Attrs that ride on a kind=lod / kind=partition wrapper Group (where the user
#: thinks of the wrapper as "their layer") rather than getting copied onto
#: each internal child. Compositing semantics (opacity, gamma, ...) flow
#: down to the children through Group inheritance at render time, so writing
#: them once on the parent is correct. ``colormap`` and ``truncation_radius``
#: are deliberately NOT compositing: the writer auto-defaults them per leaf,
#: which under nearest-ancestor-wins would shadow a parent's setting.
COMPOSITING_ATTRS = frozenset(
    {
        "transform",
        "opacity",
        "absorption",
        "gamma",
        "intensity",
        "offset",
        "blending_mode",
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

    No wrapper impl calls this directly any more: it is reached from
    :func:`validate_gsplats_channels_before_split`, which is the one geometry
    whose channel validator does not cover labels. Points and Lines get the
    equivalent check from the writer sweeps their gates delegate to
    (``validate_labels_for_writing``, last in the flat order). Whichever door, the
    call belongs to a wrapper's pre-split gate — entering a wrapper is exactly "a
    split is about to happen" — and deliberately NOT to the top of a leaf adder:
    the plain-leaf path validates in the writer, and hoisting the check above
    ``_validate_data_dimensions`` there would change which error a multi-fault
    call reports. Same reasoning, and the same house rule, as
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
    scalars → labels), just against the source count. Sharing one
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

    The CHANNEL verdict is identical with and without a wrapper. Note the gate
    runs ABOVE the positions / dimension / attr checks on the split paths, so a
    call that ALSO trips one of those (a NaN position, a wrong column count, an
    unknown attr) reports the channel fault first here and the positions/attr
    fault on the plain-leaf path. Both refuse, and neither writes.

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

    Raises:
        ValidationError: If any channel is not a legal per-point or broadcast
            value for ``n_points`` elements.
    """
    from ...io._compiler.geometry_writers.points import validate_points_channels

    validate_points_channels(
        n_points,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
    )


def validate_lines_channels_before_split(
    n_vertices: int,
    *,
    widths: Any,
    colors: Any = None,
    sharpness: Any = None,
    scalars: Any = None,
    labels: Any = None,
) -> None:
    """Run the flat Lines write gate over every per-vertex channel, pre-split.

    The Lines twin of :func:`validate_points_channels_before_split` — read that
    docstring for the trap this closes, for why the implementation is shared
    with the writer rather than repeated, and for the exact scope of the
    identical-verdict promise. All four channels are per-VERTEX (not
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

    Raises:
        ValidationError: If any channel is not a legal per-vertex or broadcast
            value for ``n_vertices`` elements.
    """
    from ...io._compiler.geometry_writers.lines import validate_lines_channels

    validate_lines_channels(
        n_vertices,
        widths=widths,
        colors=colors,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
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
    from ...io._compiler.geometry_writers.lines import validate_line_indices

    validate_line_indices(indices, n_vertices)


def validate_gsplats_channels_before_split(
    centers: np.ndarray,
    amplitudes: Any,
    cholesky_factors: np.ndarray,
    *,
    colors: Any = None,
    labels: Any = None,
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

    Returns:
        ``cholesky_is_uniform`` — whether ``cholesky_factors`` is the uniform
        ``(k,)`` form, straight from the validator that already decided it. The
        caller needs this to skip slicing that array, and recomputing the rule at
        the call site would be one more copy of exactly the kind of duplication
        this gate exists to remove.

    Raises:
        ValueError: If the trio's shapes/values are not a legal per-splat or
            broadcast combination for ``len(centers)`` splats.
        ValidationError: If ``labels`` is not one string per splat.
    """
    from ...io._compiler.gsplat_assembly import validate_gsplat_inputs

    (*_normalized, n_splats, _n_dims, cholesky_is_uniform) = validate_gsplat_inputs(
        centers, amplitudes, cholesky_factors, colors
    )
    validate_labels_before_split(labels, n_splats)
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
