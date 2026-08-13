"""`add_gsplats_from_data` body — high-level dispatch over GSplatData.

Resolves the two LOD axes (substitutive via ``lod_group=`` and additive
via ``additive_lod=``), then routes to the appropriate write path:

* multi-substitutive → kind=lod Group via :func:`add_gsplats_as_lod_group_impl`
* single-substitutive + multi-additive → multi-LOD subgroups via
  :func:`add_gsplats_multi_lod_impl`
* single-substitutive + single-additive → flat single-leaf gsplats node
  via ``Group.add_gsplats``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .lod_dispatch import (
    add_gsplats_as_lod_group_impl,
    add_gsplats_multi_lod_impl,
)

if TYPE_CHECKING:
    from ...gsplats import GSplats
    from ...node import Node
    from ..group import Group


# The two structures that reach :func:`labels_on_wrapper_reason`, spelled once so
# the wording cannot drift between the ``lod_group=`` door and the graft door.
#
# The lod_group one names the STRUCTURE rather than the kwarg on purpose: the
# most likely real door is the AUTO-LOWER route, where the caller passed no
# ``lod_group=`` at all (``add_gsplats_from_file`` on any matrix-shaped
# multi-level file takes it), and a message quoting a kwarg that is not in the
# call sends the reader looking for something that is not there.
LOD_GROUP_STRUCTURE = (
    "a multi-level substitutive pyramid (auto-lowered to a kind=lod group)"
)
LOD_GROUP_REMEDY = (
    "Label a single-level node instead (lod_group=False collapses to the finest level)."
)
GRAFT_STRUCTURE = (
    "a grafted multi-node .gsplats.zarr subtree (kind=lod / kind=partition)"
)
GRAFT_REMEDY = (
    "Label a single-leaf file instead ('gsplat flatten' collapses this one to "
    "one leaf)."
)


def labels_on_wrapper_reason(kwarg: str, structure: str, remedy: str) -> str:
    """Why per-element labels cannot ride into a multi-child gsplats wrapper.

    ONE template for all four doors — ``labels`` / ``image_labels`` crossed with
    the ``lod_group=`` gate below and the graft gate in ``from_io`` — because they
    all fail for the identical reason, and hand-written near-copies would be free
    to drift. Only two things vary: ``structure`` names the wrapper the caller
    actually asked for, and ``remedy`` is that door's single-node escape hatch.
    Both come from the module constants above.
    """
    return (
        f"{kwarg} is not supported on {structure}. Each child carries its own "
        "set of splats — a coarser level holds merged representatives, a "
        "partition part holds one tile's share — so no single list has a "
        "per-element correspondence to carry: slicing it would pair entries with "
        "the wrong splats, and passing it whole would only fit whichever child "
        f"happened to match. {remedy} Or build the wrapper yourself "
        "(add_lod_group() / add_partition_group()) and give each add_gsplats() "
        f"child its own {kwarg}. An additive_lod= ladder is not an alternative "
        "here: a gsplats additive ladder carries no labels at all."
    )


def labels_on_a_laddered_leaf_reason(kwarg: str) -> str:
    """Why a SINGLE gsplats leaf still cannot take labels once it is laddered.

    Deliberately NOT the shared template above. That one's whole argument is
    "several children, no per-element correspondence, so it cannot be sliced" —
    every clause of which is false here: there is one leaf, holding every splat,
    with a perfect correspondence. The fault is simply that the writer this leaf
    goes to has nowhere to put them. Bending the template to cover both would
    make it read wrong at one door or the other, so this forks — and the two live
    side by side, sharing the concluding fact (the template's last sentence states
    the same "no labels channel" limitation from the other direction).
    """
    return (
        f"{kwarg} is not supported on a gsplats additive ladder. The ladder "
        "writer (write_gsplat_leaf_subtree) has no labels channel at all — "
        "labels are a leaf-only feature of write_gsplats — so a laddered leaf "
        "cannot carry them however they are supplied, and the ladder-union "
        "labels Points and Lines get have no gsplats equivalent. Collapse the "
        "ladder and label the single leaf ('gsplat flatten' does exactly that), "
        f"or keep the ladder and drop {kwarg}."
    )


def strip_absent_label_kwargs(attrs: Dict[str, Any]) -> None:
    """Delete ``labels`` / ``image_labels`` from ``attrs`` when their value is None.

    ``labels=None`` means "no labels" — that is how the leaf adders read it, since
    they bind both as named params defaulting to None. Here they arrive inside
    ``**attrs``, where a present-but-None KEY is a different thing entirely: it
    survives into ``child_attrs`` and reaches ``validate_render_attrs``, which
    rejects an unknown key by NAME and never looks at its value. So the idiomatic
    ``labels=maybe_labels`` call stranded a childless wrapper with ``Unknown node
    attribute 'labels'`` raised from inside ``child_0``, whenever a child took the
    additive-ladder writer — which every level of a stock ``gsplat lod --recipe
    levels`` file does, its stream ladders being on by default (#1471).

    Mutates in place and returns None: every caller owns the dict it passes (its
    own ``**attrs``), and handing back a copy would only invite one of them to
    forget to use it.
    """
    for key in ("labels", "image_labels"):
        if key in attrs and attrs[key] is None:
            del attrs[key]


def _reject_before_wrapper(
    group: "Group",
    *,
    name: str,
    result: Any,  # GSplatData
    dim_order: Optional[List[str]],
    fill: Optional[Dict[str, float]],
    fill_sigma: Optional[Dict[str, float]],
    colormap: Any,
    attrs: Optional[Dict[str, Any]] = None,
    labels: Any = None,
    image_labels: Any = None,
) -> None:
    """Refuse what is judgeable up front BEFORE the ``kind=lod`` group exists (#1446).

    :func:`add_gsplats_as_lod_group_impl` calls ``add_lod_group`` first and writes
    children afterwards, so without this the refusal came from inside ``child_0``
    — blaming an internal child the caller never wrote and leaving ``name`` in the
    store as a childless ``kind=lod`` group, where the same data with
    ``lod_group=False`` writes nothing at all. The other two dispatch targets need
    no such gate: the flat path is ``Group.add_gsplats``, which checks inside its
    own funnel, and :func:`add_gsplats_multi_lod_impl` validates every sub-LOD
    while building them, before its single write — including its own
    ``validate_render_attrs(attrs, reserved_attrs=GSPLATS_RESERVED_ATTRS)`` call
    inside ``write_gsplat_leaf_subtree``, which already runs BEFORE that writer
    creates any group, so the ``additive_lod=`` door never stranded and needed no
    change here.

    ``attrs`` (#1534, which hoisted the same check to the top of
    ``add_mesh_impl`` / ``add_gsplats_impl``) closes the SAME stranding class
    on THIS door, one level removed: ``child_0``'s own
    write already answers a ``GSPLATS_RESERVED_ATTRS`` key (``ordering=``,
    ``position_bounds=``, ...) with the correct *reserved* verdict — unlike
    the Points/Lines/Mesh ADDITIVE bug (#1529/#1534), nothing here calls
    ``validate_render_attrs`` with an empty reserved set, so there is no
    wrong-verdict/silent-accept-and-clobber half to this bug — but that
    correct refusal fires from inside ``child_0``, by which point
    ``add_gsplats_as_lod_group_impl`` has already called ``add_lod_group``
    and created the ``kind=lod`` wrapper. So the verdict was always right and
    the store was always wrong: ``name`` survived as a childless ``kind=lod``
    group, exactly the #1529/#1534 partition/substitutive shape one adder
    layer up. Running the same check here, before ``add_gsplats_as_lod_group_impl``
    is even called, closes it the same way: nothing written on a refusal.

    The ORDER inside the gate copies the flat path's statement order exactly
    AMONG THE CHECKS IT CONTAINS — not the flat path's order as a whole, which
    starts higher up with ``validate_node_name`` and
    ``_ensure_no_duplicate_child`` (``adders/gsplats.py``, above everything
    mirrored here), so a duplicate name is still reported differently on the two
    paths. That gap predates this gate, came in with #1446, and is deliberately
    left alone. For everything the gate DOES contain the order is what decides
    which fault a call tripping more than one of them is told about: the rank
    raise first (the ``(N, D)`` check at the top
    of ``add_gsplats_impl``), then the ``dim_order`` spec — which the flat path
    runs while APPLYING the transform, i.e. above everything else — then the
    colours/colormap exclusion, and the width last. Getting that wrong is the
    recurring bug here: with the width first the split path answered ``Dimension
    mismatch …`` where the flat path answered ``Cannot specify both …``; with the
    colours gate above the ``dim_order`` spec it answered ``Cannot specify both
    …`` where the flat path answered ``dim_order has 3 names but data has 4
    columns``. The gate also closes the same stranding class for a
    colours+colormap call whose width is perfectly fine, which used to reach
    ``child_0`` with the wrapper on disk.

    WHAT else is checkable up front depends on ``dim_order``. Without one, the
    incoming column count must already equal the scene's, which is exactly
    ``validate_dimension_count`` (rank guard included). With one, the
    post-transform width is ``scene_ndim`` by construction (``apply_dim_order``
    allocates it that way), so the count check can never fire downstream — what
    fires there instead is one of the SPEC refusals: ``dim_order``'s
    length-vs-columns, duplicate names, a name absent from the scene, a bad
    ``fill`` key (all in ``validate_dim_order_spec``) or a bad ``fill_sigma`` key
    (``validate_fill_sigma_keys``). All of them are judged from the spec plus the
    column count alone, so all of them are checkable here. Everything left
    downstream genuinely needs the transformed per-level arrays.

    The node-attrs gate (``validate_render_attrs``) sits right after the width
    check and before the per-rung colour-validator loop: below colours/width,
    matching the precedence the leaf adders already keep between their own
    colours gate, dimension-count gate and node-attrs gate (see
    ``adders/gsplats.py::add_gsplats_impl``), and above the labels/structural
    checks below (the colour-validator loop has no flat-path counterpart of its
    own either, but it exists only to pre-check what the CHILDREN will each
    validate again, so it stays the closest thing to a "structural" check this
    function has and the attrs gate outranks it the same way the leaf adder's
    own attrs gate outranks its channel-validator wrappers).

    ``labels`` / ``image_labels`` are the ONE check here with no flat-path
    counterpart (#1471), which is why they come LAST. Both ride into every child
    unsliced through ``child_attrs``, so a real ladder — whose levels are merged
    representatives with DIFFERENT splat counts — refused from inside ``child_0``
    with the wrapper already on disk, and no per-level slicing could have saved
    it: there is no per-element correspondence between a coarse level's
    representatives and the finest level's splats. So they are REFUSED outright
    rather than hoisted. The flat path, by contrast, happily ACCEPTS ``labels``
    and validates it LAST of all, in the writer's own sweep — so ranking this
    refusal any higher would let it outrank a fault the flat path reports first,
    which is exactly what the parity assertions above pin. ``labels`` is asked
    before ``image_labels`` (the leaf adder's signature order), so a call passing
    both hears about ``labels``.

    Called from INSIDE the multi-substitutive branch, below the
    ``coverage_fraction`` refusal, so that structural-kwarg fault outranks
    everything here — the same shape as the ordering inside the gate itself. The
    cost is that an invalid call still pays for the substitutive resolve above;
    correct-precedence-first is the deliberate trade.

    ``TypeError`` is caught alongside ``ValueError`` because the spec validators
    can raise it on a non-sequence ``dim_order`` (``len()`` of an int), and the
    leaf adders' funnel converts both to a ``ValueError`` — a split path that
    leaked the raw ``TypeError`` would diverge in exception TYPE, which is
    precisely what the parity tests compare. The ``Could not add gsplats
    '<name>': `` prefix is applied by hand because this module has no try/except
    funnel of its own; it is byte-identical to what ``add_gsplats_impl``
    produces, which the #1446 tests pin against a direct ``add_gsplats`` call.
    """
    import warnings

    import numpy as np

    from ....validation.base import validate_colors_for_writing
    from ...scene.dim_order import validate_dim_order_spec
    from ...scene.validation import validate_array_rank
    from ..dim_order import validate_fill_sigma_keys

    centers = result.centers
    try:
        # The flat path's first statement, and it also makes ``shape[1]`` below
        # safe: ``AdditiveSubLOD`` accepts 1-D centers, so a bare ``IndexError``
        # would otherwise escape this funnel.
        validate_array_rank(centers, "centers")
        if dim_order is not None:
            # Same order as the flat path, which applies dim_order to the centers
            # (spec + fill) before it embeds the Cholesky factors (fill_sigma) —
            # and does both BEFORE it reaches its colours/colormap gate.
            scene = group._find_scene()
            validate_dim_order_spec(scene, dim_order, centers.shape[1], fill)
            validate_fill_sigma_keys(scene, dim_order, fill_sigma)
        # Asked of EVERY level's ladder, not just ``result.colors`` (which is the
        # FINEST level's — what the flat path would forward as ``colors=``): each
        # child is written through an adder that refuses colours+colormap in its
        # own right, so a pyramid whose finest level is uncoloured and whose
        # coarse level is not would pass a finest-only gate and then raise from
        # inside that coarse child, with the wrapper already on disk — the exact
        # stranding this function exists to prevent. Same ``any`` question
        # ``add_gsplats_multi_lod_impl`` asks of its sub-LODs.
        has_colors = any(
            sub.colors is not None
            for level in result.substitutive_levels
            for sub in level.additive_sublods
        )
        if has_colors and colormap is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )
        if dim_order is None:
            # Under a ``dim_order`` there is nothing left to count: the
            # post-transform width is ``scene_ndim`` by construction (see above).
            group._find_scene()._validate_dimension_count(
                centers, name, data_type="centers"
            )
        # Node-attrs gate — the ``lod_group=`` peer of the #1529/#1534 hoist on
        # ``add_points``/``add_lines``/``add_mesh``/``add_gsplats``'s own
        # structural branches (see the docstring). Below colours/width, matching
        # the leaf adders' own precedence, and above the colour-validator loop
        # and the labels checks below — a call tripping one of those too still
        # hears about the attrs fault first, exactly as ``add_gsplats_impl``'s
        # own gate outranks ITS post-entry checks.
        #
        # ``labels`` / ``image_labels`` / ``partition`` are excluded here. The
        # RULE, not just the list: a key is excluded exactly when it is a NAMED
        # parameter of the LEAF ``Group.add_gsplats`` (so genuinely meaningful
        # inside ``**attrs`` on this adder — unlike on ``add_points``/
        # ``add_lines``/``add_mesh``/``add_gsplats`` themselves, where all three
        # are named params and the caller's ``**attrs`` never contains them, so
        # the exclusion is a no-op there) AND is forwarded onward STRUCTURALLY
        # by this adder rather than colliding with a value this adder already
        # passes positionally. ``labels``/``image_labels`` ride unsliced into
        # every child via ``child_attrs`` (see the dedicated, more informative
        # refusal below); ``partition`` rides the same way, all the way down to
        # each per-level ``add_gsplats`` call, where it drives that leaf's own
        # BSP split — legitimate and structural, not appearance data this
        # function computes itself.
        #
        # ``colors`` / ``amplitudes`` / ``cholesky_factors`` are deliberately
        # NOT excluded: unlike ``partition``, this adder itself forwards those
        # three positionally from the ``GSplatData`` (``result.colors``, etc.)
        # on the single-substitutive path, so a caller-supplied value under the
        # same name would collide with what this adder already passes rather
        # than travel through untouched — refusing it here, with nothing
        # written, is an improvement over the raw ``TypeError`` (Python's own
        # "multiple values for keyword argument") plus childless-wrapper strand
        # that open issue #1496 documents for that shape, though it does not
        # close #1496's broader present-but-None/conflicting-``**attrs``
        # question. ``scalars`` is likewise left unexcluded: GSplats has no
        # scalars channel at all (unlike Points/Mesh), so it is correctly
        # unknown on this door too.
        from ....io._compiler.node_common import (
            GSPLATS_RESERVED_ATTRS,
            validate_render_attrs,
        )

        attrs_for_gate = {
            k: v
            for k, v in (attrs or {}).items()
            if k not in ("labels", "image_labels", "partition")
        }
        validate_render_attrs(attrs_for_gate, reserved_attrs=GSPLATS_RESERVED_ATTRS)
        # The leaf's WHOLE colours validator, run against EVERY rung of EVERY
        # level, for the same reason the colours/colormap question above is asked
        # of every level — and below the width, because that is where the flat
        # path puts it (the leaf writer's channel sweep runs after the dimension
        # count). Any colours fault otherwise surfaces from inside a child with
        # the wrapper already on disk (#1489): measured with this loop removed,
        # an all-``int64`` two-level ladder left ``name`` holding a PARTIAL
        # ``child_0`` (centers/amplitudes/cholesky, no colours), and a
        # fine-bad/coarse-clean one left a COMPLETE ``child_0`` beside a partial
        # ``child_1`` — a half-written ladder. Dtype is only the loudest of four
        # such doors; negative values, NaN and an out-of-range RGBA alpha all did
        # the same thing.
        #
        # The FULL validator, never just the dtype rule it was written for: dtype
        # is the fifth check inside it, so hoisting that one alone jumps it over
        # the four above and an all-negative ``int32`` array then answered with
        # the dtype where the flat path answers "Colors cannot be negative" — and
        # advised ``astype(np.uint8)``, which turns -1 into 255. One call, one
        # internal order, parity restored.
        #
        # Levels are walked finest-first, which is the order
        # ``substitutive_levels`` holds them in, so an all-bad ladder reports the
        # same level a flat call would — the finest, which is what
        # ``result.colors`` forwards. Each rung is checked against its OWN splat
        # count: a ladder's rungs are prefixes of different lengths.
        #
        # Its WARNINGS are suppressed for the duration, and only its warnings:
        # the validator also warns on float colours above 10.0, and every rung
        # this loop inspects is validated again by the child that writes it, so
        # letting the gate warn too simply doubles the count (measured: a 2-level
        # HDR ladder emitted 4 where the flat path emits one per leaf). Same
        # concern as ``TestRangeWarningsAreNotMultipliedByTheHoist`` pins for the
        # #1446 count hoist — a pre-write gate must add refusals, not noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for level in result.substitutive_levels:
                for sub in level.additive_sublods:
                    if sub.colors is not None:
                        validate_colors_for_writing(
                            np.asarray(sub.colors), int(sub.n_splats), channels=(3, 4)
                        )
        # LAST on purpose — see the docstring: the only check here with no
        # flat-path counterpart at all (the flat path ACCEPTS labels and
        # validates them last of all, in the writer sweep), so it must not
        # outrank any fault the flat path would report first.
        for kwarg, value in (("labels", labels), ("image_labels", image_labels)):
            if value is not None:
                raise ValueError(
                    labels_on_wrapper_reason(
                        kwarg, LOD_GROUP_STRUCTURE, LOD_GROUP_REMEDY
                    )
                )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e


def add_gsplats_from_data_impl(
    group: "Group",
    *,
    name: str,
    result: Any,  # GSplatData (runtime-typed; importing it here pulls heavy deps)
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    lod_group: Any = None,
    additive_lod: Any = None,
    **attrs: Any,
) -> Union["GSplats", "Group"]:
    from luxar.gsplats.gsplat_data import GSplatData

    from ..lod.gsplats import (
        resolve_additive_axis_gsplats,
        resolve_substitutive_axis_gsplats,
    )

    if not isinstance(result, GSplatData):
        raise TypeError(f"Expected GSplatData, got {type(result).__name__}")

    # Normalise "no labels" to "no key" for the whole dispatch — a present-but-None
    # key is otherwise an unknown ATTR to every writer downstream. Done here rather
    # than per-branch because all three targets forward ``**attrs`` verbatim, and
    # it is a no-op on the flat route (``Group.add_gsplats`` binds both as named
    # params defaulting to None). See :func:`strip_absent_label_kwargs`.
    strip_absent_label_kwargs(attrs)

    # Propagate truncation_radius through attrs (unless caller overrode it)
    if "truncation_radius" not in attrs:
        attrs["truncation_radius"] = result.truncation_radius

    # Resolve coarsen_dims (which dims substitutive coarsening may merge over)
    # on the COMPUTE path only — scene + data ndim are available here, but the
    # stored-pyramid path rejects compute kwargs. Default Auto = displayed dims,
    # grouping by non-displayed dims (same semantics as Points/Lines).
    if isinstance(lod_group, dict):
        computing = result.n_substitutive <= 1 or bool(lod_group.get("recompute"))
        if computing:
            from ..lod.group import _validate_coarsen_dims_spec, resolve_coarsen_dims

            raw = _validate_coarsen_dims_spec(lod_group.get("coarsen_dims"))
            resolved = resolve_coarsen_dims(group._find_scene(), int(result.ndim), raw)
            lod_group = {**lod_group, "coarsen_dims": resolved}

    # Resolve the two LOD axes. Substitutive first (it can produce a
    # multi-level result), then additive (uniform across levels).
    result, explicit_coverage_fractions = resolve_substitutive_axis_gsplats(
        result, lod_group
    )
    result = resolve_additive_axis_gsplats(result, additive_lod)

    # Multi-substitutive → kind=lod Group with one gsplats child per level
    if result.n_substitutive > 1:
        if "coverage_fraction" in attrs:
            raise ValueError(
                "coverage_fraction must not be passed when the resolved "
                "result is multi-substitutive: thresholds are derived "
                "per-child (or set via lod_group=dict(coverage_fractions="
                "[...]))."
            )
        _reject_before_wrapper(
            group,
            name=name,
            result=result,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            colormap=attrs.get("colormap"),
            # The un-split caller attrs, for the node-attrs gate — see the
            # docstring. Handed whole (colormap/labels/image_labels included):
            # ``validate_render_attrs`` does not stop at KEYS (unknown-attr /
            # reserved-attr refusals) — it also runs the render-attrs VALUE
            # validators (11 of them: blending_mode, join, absorption, opacity,
            # truncation_radius, gamma, intensity, offset, layer, visible,
            # colormap), so a bad ``colormap``/``opacity`` VALUE is caught right
            # here, ahead of the dedicated labels refusal below (measured: a bad
            # ``colormap`` plus ``labels=[...]`` on the same call answers "Unknown
            # colormap ...", not the labels refusal). That matches the flat
            # path's own precedence between its attrs gate and its labels
            # handling, so it is desirable parity, not an accident — see the
            # gate ordering discussion above.
            attrs=attrs,
            # Neither is a named kwarg of this function: both are named params of
            # the LEAF adder and travel here inside ``**attrs``, from where they
            # would ride into every child through ``child_attrs``.
            labels=attrs.get("labels"),
            image_labels=attrs.get("image_labels"),
        )
        return add_gsplats_as_lod_group_impl(
            group,
            name=name,
            result=result,
            explicit_coverage_fractions=explicit_coverage_fractions,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **attrs,
        )

    # Single-substitutive: flat or multi-additive path
    if result.n_additive_sublods <= 1:
        return group.add_gsplats(
            name=name,
            centers=result.centers,
            amplitudes=result.amplitudes,
            cholesky_factors=result.cholesky_factors,
            colors=result.colors,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **attrs,
        )

    return add_gsplats_multi_lod_impl(
        group,
        name=name,
        result=result,
        parent=parent,
        extend_to_all=extend_to_all,
        dim_order=dim_order,
        fill=fill,
        fill_sigma=fill_sigma,
        **attrs,
    )
