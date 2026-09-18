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

from ....validation.writing import (
    GSPLATS_RESERVED_ATTRS,
    validate_render_attrs,
)
from ..compositing import (
    ABSENT_WHEN_NONE_RENDER_ATTRS,
    funnel_add_error,
    preflight_extend_to_all,
    reject_layer_order_inside_specialized_group,
    reject_lines_only_join,
    reject_mesh_only_appearance,
    strip_absent_attr_kwargs,
)
from ..partition import reject_mismatched_partition_parent
from .amplitude_norm import (
    NormalizeSpec,
    normalize_gsplat_data,
)
from .amplitude_norm import (
    stamp_factor as stamp_amplitude_factor,
)
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
    "one leaf), or pass flatten=True when adding the file."
)


# The two structures that reach :func:`partition_beside_a_ladder_reason`, spelled
# once for the reason the pair above is: the ladder can arrive two ways and only
# one of them is a kwarg. On the ``additive_lod=`` door "drop one of the two" is
# the remedy; on the FILE / graft door there is no ``additive_lod=`` in the call
# to drop — the ladder is in the store — so naming it sends the reader looking
# for something that is not there, which is the exact failure the labels pair
# above exists to avoid.
LADDER_STRUCTURE = "an additive_lod= ladder"
LADDER_REMEDY = (
    "Drop one of the two, or partition the data yourself and add each part with "
    "its own ladder."
)
STORED_LADDER_STRUCTURE = "the additive ladder this .gsplats.zarr already carries"
STORED_LADDER_REMEDY = (
    "Collapse the ladder first ('gsplat flatten' rewrites the file as a single "
    "unladdered leaf, which partitions normally), or pass flatten=True when "
    "adding the file."
)


def reject_invalid_gsplat_compositing_attrs(
    group: "Group", name: str, parent: Optional["Node"], attrs: Dict[str, Any]
) -> None:
    """Apply the flat gsplat adder's compositing gates with its error prefix."""
    try:
        reject_mismatched_partition_parent(parent or group, "gsplats", name)
        reject_layer_order_inside_specialized_group(
            "gsplats", name, attrs, parent or group
        )
        reject_lines_only_join("gsplats", name, attrs)
        reject_mesh_only_appearance("gsplats", name, attrs)
    except ValueError as error:
        raise ValueError(funnel_add_error("gsplats", name, error)) from error


def partition_beside_a_ladder_reason(structure: str, remedy: str) -> str:
    """Why ``partition=`` and an additive ladder cannot ride the same node.

    ONE template for both doors, for the reason :func:`labels_on_wrapper_reason`
    is one: the obstacle is identical — ``add_gsplats_multi_lod_impl`` writes
    every laddered leaf and has no ``partition`` parameter at all — and only the
    caller's own escape hatch differs. Both come from the module constants above.
    """
    return (
        f"partition= is not supported alongside {structure}. A laddered leaf is "
        "written by add_gsplats_multi_lod_impl, which cannot also split it into "
        "parts, and the two are different decompositions of the same splats — a "
        "prefix ordering and a BSP split — that nothing downstream carries "
        f"together. {remedy}"
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


# The ``**attrs`` keys the ``lod_group=`` node-attrs gate must NOT judge, because
# they are named parameters of the LEAF ``Group.add_gsplats`` that this adder
# forwards onward STRUCTURALLY rather than colliding with a value of its own —
# see the rule stated in full at the gate itself (:func:`_reject_before_wrapper`).
GATE_FORWARDED_LEAF_PARAMS = ("labels", "image_labels", "keys", "partition")

# The keys for which a present-but-``None`` value means ABSENT, and which are
# therefore DELETED from ``**attrs`` before dispatch (#1471, #1496). Four
# families, and the rule for each is why the set is not simply the tuple above:
#
#   * every key in ``GATE_FORWARDED_LEAF_PARAMS`` — each is a named param of the
#     leaf ``Group.add_gsplats`` DEFAULTING TO None, so None already means
#     "absent" one level down;
#   * ``colors`` — the same kind of leaf named param (default None), except that
#     this adder passes it POSITIONALLY from the data, so a NON-None value is a
#     collision rather than something to forward (see
#     :func:`reject_data_owned_channels`) and it must stay OUT of the gate's
#     exclusion tuple above;
#   * ``truncation_radius`` — not a leaf named param at all but a key this adder
#     INJECTS from ``result.truncation_radius`` a few lines into
#     :func:`add_gsplats_from_data_impl`; an explicit None used to OVERWRITE the
#     data's own value and then fail the render-attr validator, so under the same
#     rule it must mean "no override" and let the data's value stand. A NON-None
#     value here is a legitimate override, so it likewise does not belong in the
#     gate's exclusion tuple — the gate should and does still validate it;
#   * :data:`~luxar.core.group.compositing.ABSENT_WHEN_NONE_RENDER_ATTRS` —
#     ``colormap`` and ``coverage_fraction``, the two render attrs whose None is
#     caught by no value validator and so reaches disk. Not this module's
#     property but every adder's (the leaf adders hit the identical shape, #1574),
#     which is why it is taken from ``compositing`` rather than restated here; the
#     measured argument for those two and against the rest of the render attrs
#     lives with the definition. ``scalars=None`` is the one member of the family
#     specific to this door, and it stays OUT: it refuses as an unknown attribute
#     (GSplats has no scalars channel at all), so its None is already loud.
#
# Spelled once, derived from BOTH sources, and commented on purpose: the sets
# answer different questions ("may this key ride onward untouched?" vs "does None
# mean absent for this key?" vs "is this key's None a render fault every adder
# shares?") and a hand-copied second list would be free to drift.
ABSENT_WHEN_NONE_ATTRS = (
    GATE_FORWARDED_LEAF_PARAMS
    + (
        "colors",
        "truncation_radius",
    )
    + ABSENT_WHEN_NONE_RENDER_ATTRS
)

# The channels this adder supplies FROM the ``GSplatData`` itself, so a
# caller-supplied value under the same name is a collision, never a forward.
# All FOUR of them: the flat route's ``group.add_gsplats`` call passes
# ``centers=result.centers`` alongside the other three, so ``centers`` meets this
# tuple's criterion exactly like they do — it was missed in the first cut of
# #1496 and was the one member still able to strand a childless wrapper. Asked in
# THIS order (see :func:`reject_data_owned_channels`).
DATA_OWNED_CHANNELS = ("colors", "centers", "amplitudes", "cholesky_factors")


def data_owned_channel_reason(kwarg: str) -> str:
    """Why a caller cannot pass a splat channel this adder reads off the data.

    ONE template for the four channels and for both doors that refuse them (the
    ``from_data`` dispatch and the file/graft entries in ``from_io``), for the same
    reason :func:`labels_on_wrapper_reason` is one template: the argument is
    identical in every instance and near-copies would drift.

    Only the closing sentences vary, and only for ``colors``, which needs two
    things the other three do not. An appearance alternative worth naming
    (``colormap=``). And an admission that the opening clause — "cannot be passed
    as a keyword here" — is not literally true for it: ``colors=None`` IS passed as
    a keyword and IS accepted, being read as "absent" under the #1496 rule. That
    makes ``colors`` the one key in this family where neither spelling does what a
    naive caller expects (a None is now silently ignored where it used to raise, a
    value is refused where it used to raise differently), so the message says so
    rather than leaving it to be discovered.
    """
    colormap_hint = (
        " If you meant to recolour the node rather than replace its per-splat "
        "colours, pass colormap= instead. (An explicit colors=None is the one "
        "value this key does accept: it is read as 'absent', so the GSplatData's "
        "own colours are used.)"
        if kwarg == "colors"
        else ""
    )
    return (
        f"{kwarg} cannot be passed as a keyword here: this adder supplies it from "
        f"the GSplatData itself (result.{kwarg} on the flat route, each additive "
        "sub-LOD's own array on a split one), so a caller-supplied value collides "
        "with the one already being passed rather than travelling through — on "
        'the flat route Python itself answers "got multiple values for keyword '
        f"argument '{kwarg}'\". Nor could it be split across a wrapper's "
        "children: a coarser substitutive level holds merged representatives and "
        "an additive rung holds a prefix, each with its own element count, so no "
        f"single array has a per-element correspondence to carry. Set {kwarg} on "
        f"the GSplatData before adding it.{colormap_hint}"
    )


def reject_data_owned_channels(name: str, attrs: Dict[str, Any]) -> None:
    """Refuse a ``colors`` with a value, or a ``centers``/``amplitudes``/``cholesky_factors`` at all.

    The asymmetry is in the summary line on purpose: this function tests ``kwarg in
    attrs``, with no value check of its own, so it refuses ALL FOUR channels
    whatever they hold — ``amplitudes=None`` included. ``colors`` looks like an
    exception only because :func:`strip_absent_attr_kwargs` has already deleted a
    None one before this runs (it is the one of the four with a None default at the
    leaf, so its None means "absent"). Saying "a present, non-None value" of all
    four would read as a promise that ``amplitudes=maybe_amps`` is safe, and it is
    not.

    Run at the ENTRY of :func:`add_gsplats_from_data_impl` (and of both ``from_io``
    doors), before any route is chosen and before anything is written, so every
    route gives the same answer and none of them strands a wrapper. Pre-#1496 the
    four routes disagreed TWO ways for the identical mistake: a raw ``TypeError``
    from Python on the flat route, and — byte-identically across the other three —
    ``Unknown node attribute 'colors'. Did you mean 'colormap'?``, a real but
    misleading verdict, since the key is not unknown, it is taken.

    WHERE that mattered differs by door, and only one of them stranded. On this
    adder's own four routes nothing was ever written for any of the four channels
    (measured with this refusal disabled: an empty store on all four), so here the
    fix is purely the verdict — the right exception type and a message that names
    the real fault. The GRAFT door is where it stranded: ``graft_gsplat_node``
    builds its ``kind=partition`` / ``kind=lod`` wrapper from the on-disk tree
    BEFORE the first leaf write, so the collision was judged one level down and
    left ``name`` on disk as a childless wrapper surviving ``finalize()``.

    The exception TYPE is load-bearing, not incidental: the leaf adders' funnel
    converts both ``ValueError`` and ``TypeError`` to a ``ValueError``, and the
    parity tests compare types, so leaking the flat route's ``TypeError`` would be
    a divergence in its own right. The ``Could not add gsplats '<name>': `` prefix
    is applied by hand for the reason :func:`_reject_before_wrapper` gives at
    length: this module has no try/except funnel, and the prefixed form is what
    every sibling refusal here produces.

    A present-but-None ``colors`` is NOT refused — :func:`strip_absent_attr_kwargs`
    has already deleted it by the time this runs, under the "an explicit None means
    absent" rule (#1496). The other three have no such reading: they are REQUIRED
    positional params of the leaf adder with no None default, so a None there is a
    collision like any other value and is refused with everything else.

    ``colors`` is asked first, then ``centers``, ``amplitudes``,
    ``cholesky_factors``, so a call passing several is answered deterministically —
    same tie-break convention as the ``labels`` before ``image_labels`` order the
    label gates keep. ``colors`` leads rather than following the leaf signature's
    order because it is the one of the four a caller plausibly reaches for on
    purpose (an appearance override); the three required data channels follow in
    that signature's own order.
    """
    for kwarg in DATA_OWNED_CHANNELS:
        if kwarg in attrs:
            raise ValueError(
                f"Could not add gsplats '{name}': {data_owned_channel_reason(kwarg)}"
            )


def reject_bad_partition_spec(attrs: Optional[Dict[str, Any]], ndim: int) -> None:
    """Judge ``partition``'s VALUE, which the node-attrs gate cannot (#1550).

    ``partition`` is in :data:`GATE_FORWARDED_LEAF_PARAMS`, so its KEY is
    excluded from ``validate_render_attrs`` — deliberately, since that exclusion
    is what lets a valid spec ride into each substitutive child, where it drives
    that child's own BSP split. An INVALID one rode it too, unread, and was
    refused only from inside ``child_0``, after ``add_lod_group`` had created
    the ``kind=lod`` wrapper.

    Calls the leaf adder's own validator for the verdict alone — the resolved
    cap and rule belong to the child that actually splits — so the message and
    exception type are byte-identical to the flat path's. Its own function
    rather than three lines inline purely to keep :func:`_reject_before_wrapper`
    under the C901 limit the complexity ratchet enforces; it is also the graft
    door's gate (``from_io.graft_gsplat_node``, which builds its wrapper from the
    on-disk tree before the first leaf write and had the identical bug).

    Two values are skipped, because the leaf never judges them either and a gate
    that refuses what the flat path accepts is its own regression:

    * ``False`` is a first-class member of the :data:`~luxar.core.group.partition.PartitionSpec`
      vocabulary — the explicit no-partition bypass ``resolve_auto_partition``
      normalises to ``None`` (and the recursion guard the partition wrappers use),
      so it is the only way to opt a ``lod_group=`` ladder out of a compiler-level
      ``auto_partition_max_elements``. Every leaf adder normalises it away before
      resolving, so it must be normalised away here too.
    * ``ndim < 2`` is where the leaf DROPS the partition request with a warning
      (``warn_if_partition_needs_more_dims``) BEFORE it resolves the spec, so a
      1-dimension scene accepts even a nonsense spec. Skipped silently: the child
      still emits that warning, and a second copy from here would only double it.

    ``ndim`` is the EFFECTIVE post-transform width — see the call site.
    """
    from ..partition import is_requested, resolve_partition_spec

    partition = (attrs or {}).get("partition")
    if not is_requested(partition) or ndim < 2:
        return
    resolve_partition_spec(partition)


def resolve_partition_beside_an_additive_ladder(
    name: str,
    attrs: Dict[str, Any],
    result: Any,  # GSplatData
) -> None:
    """Refuse a REAL ``partition=`` beside an additive ladder; DROP a ``False`` one.

    ``add_gsplats_multi_lod_impl`` — the writer every multi-rung level goes
    through — has no ``partition`` parameter, so the key stays in ``**attrs`` and
    reaches ``validate_render_attrs`` as an unknown node attribute. On the
    multi-substitutive route that fired from inside ``child_0`` with the
    ``kind=lod`` wrapper already on disk, so a VALID spec stranded exactly like an
    invalid one did (#1550): ``Could not add gsplats 'child_0': Unknown node
    attribute 'partition'. Did you mean 'absorption'?``, ``g`` surviving
    ``finalize()``. Refused here instead, above both routes, so the two report the
    same thing and nothing is written.

    ``False`` is NOT a request, so it is not refused — but it stranded the very
    same wrapper, and for the very same reason: measured, ``lod_group=True,
    additive_lod={"n_lods": 2}, partition=False`` gave ``Could not add gsplats
    'child_0': Unknown node attribute 'partition'`` with ``g`` on disk as a
    childless ``kind=lod`` group, where the single-substitutive twin merely
    refused (naming ``'g'``, writing nothing). So it is DELETED instead, and the
    call succeeds on both routes — which is what the bypass means: ``False`` opts
    out of a compiler-level ``auto_partition_max_elements``, so honouring it is
    writing no partition, which is exactly what this destination does.

    The scope of that deletion is narrow on purpose. ``False`` is load-bearing
    wherever a LEAF adder resolves it — ``resolve_auto_partition`` reads
    ``user_partition is False`` to bypass the compiler threshold, so stripping the
    key at this adder's entry (the way an explicit ``None`` is stripped) would
    silently re-enable auto-partitioning on every un-laddered route. It is dropped
    only when the multi-LOD writer is THIS call's destination
    (``n_substitutive <= 1 < n_additive_sublods``) — a writer that cannot
    partition at all, so there is no auto-partition left to bypass. A
    multi-substitutive call keeps the key and hands it to each child, where the
    recursion asks this same question per level: a laddered level drops it, an
    un-laddered one still gets its bypass.

    Called from ABOVE the route branch rather than from the pre-wrapper gate's
    partition slot, which costs it the precedence its siblings have — a call also
    carrying a bad node attr, a bad ``dim_order`` or colours+colormap hears about
    this conflict where the same call without ``additive_lod=`` hears about the
    other fault. That is the deliberate trade, not an oversight: the two routes
    below MUST answer identically (that divergence is half of what #1550 fixes),
    and only the multi-substitutive one has a pre-wrapper gate to sit in — the
    multi-additive route's attrs gate lives inside the writer, so a check placed
    "at the same slot" there would still outrank nothing. Every combination
    refuses with an EMPTY store either way, so what is at stake is which fault is
    named first, and answering the same on both routes is worth more than
    matching the flat path on a call the flat path cannot make. Pinned by
    ``TestTheLadderConflictOutranksTheOtherPreWrapperFaults``.
    """
    from ..partition import is_requested

    partition = attrs.get("partition")
    if not is_requested(partition):
        if (
            partition is False
            and result.n_substitutive <= 1
            and result.n_additive_sublods > 1
        ):
            del attrs["partition"]
        return
    if not any(len(level.additive_sublods) > 1 for level in result.substitutive_levels):
        return
    raise ValueError(
        f"Could not add gsplats '{name}': "
        + partition_beside_a_ladder_reason(LADDER_STRUCTURE, LADDER_REMEDY)
    )


def _reject_before_wrapper(
    group: "Group",
    *,
    name: str,
    result: Any,  # GSplatData
    dim_order: Optional[List[str]],
    fill: Optional[Dict[str, float]],
    fill_sigma: Optional[Dict[str, float]],
    colormap: Any,
    extend_to_all: Optional[Union[List[str], str]],
    attrs: Optional[Dict[str, Any]] = None,
    labels: Any = None,
    image_labels: Any = None,
    keys: Any = None,
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

    Immediately below it sits the ``partition``-SPEC check (#1550), the one
    thing the attrs gate cannot reach: ``partition`` is in
    :data:`GATE_FORWARDED_LEAF_PARAMS`, so its key is excluded from
    ``validate_render_attrs`` — deliberately, since that is what lets a VALID
    spec ride into each child's own BSP split — which left an INVALID one
    (``partition="nonsense"``, ``{"max_elements": 0}``) judged one level down,
    inside ``child_0``, with the wrapper already written. Same stranding shape,
    closed the same way by :func:`reject_bad_partition_spec`, which calls
    ``partition.resolve_partition_spec`` — the very function the leaf adder
    calls — for its verdict alone. Directly
    below the attrs gate because that is the flat path's own order — the leaf
    validates node attrs at its entry and resolves the spec inside its
    partition branch, further down.

    An explicit ``extend_to_all`` is checked next. Its invalid branches depend
    only on the scene dimensions, not on a particular LOD child, so they can be
    refused before the wrapper exists. It remains below the partition-spec gate
    to match the leaf's order and above the per-rung colours loop because the
    leaf resolves it before the writer's channel sweep. ``None`` stays
    child-only so candidate analysis still warns once per written child.

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
        # The width the CHILDREN will actually be handed, which is what decides
        # whether a partition can run at all (see the ``partition``-spec check
        # below): the incoming column count, unless a ``dim_order`` reallocates
        # it to the scene's own ``ndim``.
        effective_ndim = int(centers.shape[1])
        if dim_order is not None:
            # Same order as the flat path, which applies dim_order to the centers
            # (spec + fill) before it embeds the Cholesky factors (fill_sigma) —
            # and does both BEFORE it reaches its colours/colormap gate.
            scene = group._find_scene()
            effective_ndim = int(scene._dimensions.ndim)
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
        # ``labels`` / ``image_labels`` / ``partition`` are excluded here —
        # :data:`GATE_FORWARDED_LEAF_PARAMS`, spelled once at module level
        # because :data:`ABSENT_WHEN_NONE_ATTRS` is derived from it. The
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
        # All FOUR :data:`DATA_OWNED_CHANNELS` — ``colors``, ``centers``,
        # ``amplitudes``, ``cholesky_factors`` — are deliberately NOT excluded,
        # and since #1496 none of them reaches this gate at all: unlike
        # ``partition``, this adder itself forwards every one of them
        # positionally from the ``GSplatData`` (``result.centers``,
        # ``result.colors``, …) on the single-substitutive path, so a
        # caller-supplied value under the same name collides with what this adder
        # already passes rather than travelling through untouched. Each is
        # answered at the entry of ``add_gsplats_from_data_impl`` instead — a None
        # ``colors`` deleted by :func:`strip_absent_attr_kwargs`, anything else
        # refused by :func:`reject_data_owned_channels` — with the same verdict on
        # all four routes, in place of the raw ``TypeError`` (Python's own
        # "multiple values for keyword argument") on the flat one and the
        # misleading ``Unknown node attribute 'colors'. Did you mean 'colormap'?``
        # on the other three (#1496). ``colormap`` and ``coverage_fraction`` stay
        # unexcluded too, and are unreachable here only with a NONE value (the
        # strip deletes those); a real value of either still rides in and is still
        # validated, a bad colormap NAME by this very call. Leaving all six
        # unexcluded is what keeps this gate honest should that entry ever be
        # bypassed. ``scalars`` is left unexcluded with nothing done to it at all:
        # GSplats has no scalars channel (unlike Points/Mesh), so it is correctly
        # unknown on this door and a ``scalars=None`` correctly refused.
        attrs_for_gate = {
            k: v
            for k, v in (attrs or {}).items()
            if k not in GATE_FORWARDED_LEAF_PARAMS
        }
        validate_render_attrs(attrs_for_gate, reserved_attrs=GSPLATS_RESERVED_ATTRS)
        # …and immediately after it, the one thing that exclusion leaves
        # unjudged: ``partition``'s VALUE (#1550). Same stranding shape this
        # function exists to prevent, one key over — see the helper, which also
        # states why ``False`` and a sub-2-D width are skipped rather than judged.
        reject_bad_partition_spec(attrs, effective_ndim)
        preflight_extend_to_all(group._find_scene(), extend_to_all, centers, "splats")
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
        for kwarg, value in (
            ("labels", labels),
            ("image_labels", image_labels),
            ("keys", keys),
        ):
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
    normalize_amplitudes: NormalizeSpec = True,
    _source_dtype: Optional[str] = None,
    **attrs: Any,
) -> Union["GSplats", "Group"]:
    from luxar.gsplats.gsplat_data import GSplatData

    from ..lod.gsplats import (
        resolve_additive_axis_gsplats,
        resolve_substitutive_axis_gsplats,
    )

    if not isinstance(result, GSplatData):
        raise TypeError(f"Expected GSplatData, got {type(result).__name__}")

    # Normalise "an explicit None" to "no key" for the whole dispatch — a
    # present-but-None key is otherwise an unknown ATTR to every writer downstream.
    # Done here rather than per-branch because all three targets forward ``**attrs``
    # verbatim, and it is a no-op on the flat route for the keys the leaf adder
    # binds as named params defaulting to None. See
    # :func:`~luxar.core.group.compositing.strip_absent_attr_kwargs`.
    strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_ATTRS)
    # Then refuse a channel this adder owns, whatever its value, before ANY route
    # is taken and before anything is written. Above every branch on purpose
    # (#1496) — including above the ``coverage_fraction`` refusal, the
    # ``dim_order`` spec check and the rank guard inside
    # :func:`_reject_before_wrapper`, so a call that trips one of those TOO hears
    # about the collision. That is deliberate rather than incidental: unlike a
    # ``dim_order`` typo, a channel collision means this adder cannot even build
    # the call it is about to make, so it is the more fundamental fault of the two
    # and there is nothing below it worth reporting first.
    reject_data_owned_channels(name, attrs)
    reject_invalid_gsplat_compositing_attrs(group, name, parent, attrs)

    # Propagate truncation_radius through attrs (unless caller overrode it — and
    # an explicit ``truncation_radius=None`` is NOT an override, having just been
    # stripped above, so the data's own value applies).
    if "truncation_radius" not in attrs:
        attrs["truncation_radius"] = result.truncation_radius

    # Normalise amplitudes into [0, ~1] BEFORE the LOD axes are resolved. A
    # fitted archive stores raw source units (detector counts), which the shader
    # turns directly into radiance and optical depth with no display-side
    # compensation available — see :mod:`.amplitude_norm` for the measurement.
    # Done here rather than after resolution because scaling is linear and
    # commutes with the mass-conserving coarsening, so one pass over the single
    # input is equivalent to (and cheaper than) one pass per built level.
    result, _amp_factor = normalize_gsplat_data(result, normalize_amplitudes)
    stamp_amplitude_factor(attrs, _amp_factor)

    # Resolve coarsen_dims (which dims substitutive coarsening may merge over)
    # on the COMPUTE path only — scene + data ndim are available here, but the
    # stored-pyramid path rejects compute kwargs. Default Auto = displayed dims,
    # grouping by non-displayed dims (same semantics as Points/Lines).
    if isinstance(lod_group, dict):
        computing = result.n_substitutive <= 1 or bool(lod_group.get("recompute"))
        if computing:
            from ..lod.group import _validate_coarsen_dims_spec, resolve_coarsen_dims

            raw = _validate_coarsen_dims_spec(lod_group.get("coarsen_dims"))
            resolved = resolve_coarsen_dims(
                group._find_scene(),
                int(result.ndim),
                raw,
                dim_order=dim_order,
            )
            lod_group = {**lod_group, "coarsen_dims": resolved}

    if _source_dtype is None:
        value = result.stats.get("source_dtype")
        _source_dtype = value if isinstance(value, str) else None

    # Resolve the two LOD axes. Substitutive first (it can produce a
    # multi-level result), then additive (uniform across levels).
    result, explicit_coverage_fractions = resolve_substitutive_axis_gsplats(
        result, lod_group
    )
    result = resolve_additive_axis_gsplats(result, additive_lod)
    # ``partition=`` and an additive ladder are mutually exclusive on this door,
    # and the answer must be the same on both routes below — hence above the
    # branch, beside the other structural-kwarg refusals (#1550). Also DELETES a
    # ``partition=False`` bound for the multi-LOD writer, which is the one
    # destination where that bypass has nothing to bypass and everything to
    # strand; see the helper for why the deletion cannot be hoisted to the strip.
    resolve_partition_beside_an_additive_ladder(name, attrs, result)

    # Multi-substitutive → kind=lod Group with one gsplats child per level
    if result.n_substitutive > 1:
        # A present-but-None ``coverage_fraction`` no longer reaches this: the
        # strip at the top deleted it, so a caller who effectively passed nothing
        # is no longer told they passed something (#1496). The prefix is the same
        # ``Could not add gsplats '<name>': `` every sibling refusal in this module
        # applies by hand; this one was the only refusal here without it, so the
        # same fault was reported in two different shapes depending on which check
        # caught it.
        if "coverage_fraction" in attrs:
            raise ValueError(
                f"Could not add gsplats '{name}': coverage_fraction must not be "
                "passed when the resolved result is multi-substitutive: "
                "thresholds are derived per-child (or set via "
                "substitutive_lod=dict(coverage_fractions=[...]))."
            )
        _reject_before_wrapper(
            group,
            name=name,
            result=result,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            colormap=attrs.get("colormap"),
            extend_to_all=extend_to_all,
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
            keys=attrs.get("keys"),
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
            _source_dtype=_source_dtype,
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
            label_ids=result.label_ids,
            label_vocabulary=result.label_vocabulary,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            _source_dtype=_source_dtype,
            **attrs,
        )

    for kwarg in ("labels", "image_labels", "keys"):
        if attrs.get(kwarg) is not None:
            raise ValueError(
                f"Could not add gsplats '{name}': "
                + labels_on_a_laddered_leaf_reason(kwarg)
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
        _source_dtype=_source_dtype,
        **attrs,
    )
