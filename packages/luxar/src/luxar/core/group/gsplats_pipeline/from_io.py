"""`add_gsplats_from_file` and `add_gsplats_from_volume` impls.

Both load (or fit) a GSplatData object and then call
:func:`add_gsplats_from_data_impl` to write it into the scene.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from luxar.validation.writing import validate_image_labels_for_writing

from ..compositing import reject_mesh_only_appearance, strip_absent_attr_kwargs
from .amplitude_norm import (
    NormalizeSpec,
    normalize_node_in_place,
)
from .amplitude_norm import (
    stamp_factor as stamp_amplitude_factor,
)
from .from_data import (
    ABSENT_WHEN_NONE_ATTRS,
    GRAFT_REMEDY,
    GRAFT_STRUCTURE,
    LADDER_REMEDY,
    LADDER_STRUCTURE,
    STORED_LADDER_REMEDY,
    STORED_LADDER_STRUCTURE,
    add_gsplats_from_data_impl,
    labels_on_a_laddered_leaf_reason,
    labels_on_wrapper_reason,
    partition_beside_a_ladder_reason,
    reject_bad_partition_spec,
    reject_data_owned_channels,
)

if TYPE_CHECKING:
    from luxar.gsplats.tree import GSplatPartition

    from ...gsplats import GSplats
    from ...node import Node
    from ..group import Group


def _grafted_partition_bsp_tree(
    node: "GSplatPartition", axes: "List[int]"
) -> Optional[Dict[str, Any]]:
    """Return stored or exactly recoverable planes and report recovery."""
    if node.bsp_tree is not None:
        return node.bsp_tree

    from arbol import aprint

    from luxar.core.group.partition import (
        reconstruct_serialized_bsp_tree,
        serialized_bsp_tree_separates,
    )
    from luxar.gsplats.tree import center_bounds

    child_bounds = [center_bounds(child) for child in node.children]
    if any(bounds is None for bounds in child_bounds):
        aprint("  🧭 No exact BSP split planes recovered; using centroid part ordering")
        return None
    boxes = [bounds for bounds in child_bounds if bounds is not None]
    recovered = reconstruct_serialized_bsp_tree(boxes, axes=axes)
    if serialized_bsp_tree_separates(recovered, boxes):
        if recovered is not None and "part" not in recovered:
            aprint(f"  🧭 Recovered BSP split planes for {len(node.children)} parts")
        return recovered
    aprint("  🧭 No exact BSP split planes recovered; using centroid part ordering")
    return None


def add_gsplats_from_file_impl(
    group: "Group",
    *,
    name: str,
    path: Union[str, Path],
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    partition: Any = None,
    lod_group: Any = None,
    additive_lod: Any = None,
    flatten: bool = False,
    **attrs: Any,
) -> Union["GSplats", "Group"]:
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import is_matrix_shaped

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats file not found: {path}")

    # The #1496 ``**attrs`` pair, run HERE rather than left to the two branches
    # below, because this one public method has two of them and they must not word
    # the same fault differently — the concern the stored-column-count check below
    # already states for itself. A matrix-shaped file goes to
    # ``add_gsplats_from_data_impl``, which runs this pair at its own entry; a
    # nested one goes to ``graft_gsplat_node``, which runs it at ITS entry, below
    # this function's ``dim_order`` refusal and column-count check. So without a
    # copy here the two halves disagreed: measured with a 4-column store into a
    # 3-dimension scene plus ``colors=<arr>``, the matrix-shaped file answered the
    # collision while the partition file answered ``Dimension mismatch for 'g':
    # centers array has 4 columns …`` — the same call, the same mistake, a
    # different verdict decided by a structural property of the file the caller
    # may not even know. Running it above both branches makes the collision
    # outrank this function's own checks on BOTH, which is the same precedence
    # ``add_gsplats_from_data_impl`` gives it. Below the existence check, because
    # a missing file is not an attrs question at all.
    strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_ATTRS)
    reject_data_owned_channels(name, attrs)
    reject_mesh_only_appearance("gsplats", name, attrs)
    if partition is not None:
        attrs["partition"] = partition
    if additive_lod is not None:
        attrs["additive_lod"] = additive_lod

    # Classical (photogrammetric) splat files — INRIA/SuperSplat .ply,
    # antimatter15 .splat, Niantic .spz — are imported on the fly and embedded
    # through the normal data path, so `scene.add_gsplats_from_file("garden",
    # "garden.splat")` works one-line. Luxar's own stores never carry these
    # suffixes (.gsplats.zarr / .zip / .tar.gz), so extension sniffing is safe.
    if path.suffix.lower() in (".ply", ".splat", ".spz"):
        from luxar.gsplats.interop.classical_splats import import_gsplats

        return add_gsplats_from_data_impl(
            group,
            name=name,
            result=import_gsplats(path),
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            lod_group=lod_group,
            **attrs,
        )

    # Read the v3.0 node tree once. A matrix-shaped tree (leaf / additive ladder
    # / kind=lod of leaves) maps to a GSplatData and embeds via the normal data
    # path (which applies dim_order / extend_to_all / fill). A genuinely nested
    # tree (kind=partition root, or lod with non-leaf children) has no flat
    # GSplatData equivalent, so it is GRAFTED node-for-node, reusing the scene's
    # own builders — the same subtree the file already holds.
    node, stats = load_gsplat_node(path)

    if flatten:
        from luxar.gsplats.gsplat_data import GSplatData

        return add_gsplats_from_data_impl(
            group,
            name=name,
            result=GSplatData.from_default_selection(node, stats=stats).flattened(),
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            lod_group=lod_group,
            **attrs,
        )

    if is_matrix_shaped(node):
        from luxar.gsplats.gsplat_data import GSplatData

        # The stored-ladder half of the #1550 gate, on the branch that never
        # reaches ``graft_gsplat_node``. Not for stranding — the data door
        # refuses before anything is written either way — but for the WORDING:
        # its own refusal names ``additive_lod=`` and tells the caller to drop
        # one of the two, and there is no ``additive_lod=`` in an
        # ``add_gsplats_from_file`` call and nothing to drop. Same question, same
        # template, this door's own remedy ('gsplat flatten').
        _reject_a_partition_beside_a_stored_ladder(name, node, attrs)

        return add_gsplats_from_data_impl(
            group,
            name=name,
            result=GSplatData.from_tree(node),
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            lod_group=lod_group,
            **attrs,
        )

    if dim_order is not None or fill is not None or fill_sigma is not None:
        raise ValueError(
            "dim_order / fill / fill_sigma are not supported when grafting a "
            "partition / nested .gsplats.zarr (the file is already a full node "
            "subtree). Re-author the file in the target scene dims, or embed a "
            "matrix-shaped (leaf / additive / kind=lod) file instead, or pass "
            "flatten=True to materialize the default finest selection first."
        )

    if lod_group is not None and lod_group is not False:
        raise ValueError(
            "flatten=True is required to apply substitutive_lod= to a "
            "partition / nested .gsplats.zarr"
        )

    # Scene-dimension COUNT check on the STORED tree, before the graft creates
    # any wrapper group (#1446) — the file door's counterpart of the check the
    # three leaf adders run above their split branches. Without it the refusal
    # came from inside ``part_0`` / ``child_0``, blaming a child the caller never
    # wrote and leaving the wrapper chain on disk. Below the ``dim_order``
    # refusal above, so that kwarg fault keeps precedence, as everywhere else.
    #
    # The first leaf's centers speak for the whole subtree: the graft applies no
    # ``dim_order`` (just refused), so the stored width must already be the
    # scene's, and both container nodes reject mixed-``ndim`` children in
    # ``__post_init__`` — so one ndim per tree, recursively. Every tree has at
    # least one leaf (both containers require >= 1 child) and every leaf at least
    # one sub-LOD, both enforced at construction, so neither index can miss.
    # The ``Could not add gsplats '<name>': `` prefix is applied by hand for the
    # same reason it is in ``from_data._reject_before_wrapper``: this module has
    # no try/except funnel, and the matrix-shaped door of this very method reports
    # the prefixed form, so a bare raise here would make the two halves of
    # ``add_gsplats_from_file`` word the same fault differently.
    from luxar.gsplats.tree import iter_leaves

    first_leaf = next(iter_leaves(node))
    try:
        group._find_scene()._validate_dimension_count(
            first_leaf.additive_sublods[0].centers, name, data_type="centers"
        )
    except ValueError as e:
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e

    return graft_gsplat_node(
        group, name=name, node=node, parent=parent, extend_to_all=extend_to_all, **attrs
    )


def _reject_a_partition_beside_a_stored_ladder(
    name: str, node: Any, attrs: Dict[str, Any]
) -> None:
    """The file door's half of the #1550 partition-vs-ladder gate.

    The conflict itself is judged by ``add_gsplats_from_data_impl``, which the
    graft reaches PER LEAF — after :func:`graft_gsplat_node` has built its
    ``kind=partition`` / ``kind=lod`` wrapper from the on-disk tree. So a
    perfectly valid spec stranded here exactly as it did on the ``lod_group=``
    door: measured against a ``kind=partition`` of two 2-sublod leaves,
    ``add_gsplats_from_file("g", …, partition={"max_elements": 4})`` gave ``Could
    not add gsplats 'part_0': partition= is not supported alongside …`` with ``g``
    surviving ``finalize()`` as a childless ``kind=partition``. That is the most
    commonly produced store shape in the whole gsplat pipeline — ``gsplat lod
    --recipe tiles|overview|adaptive`` and ``batch-fit merge --recipe stream`` all
    emit it, stream ladders being on by default — and the ladder is knowable up
    front from ``iter_leaves``, so the question belongs in the same slot as
    :func:`_reject_a_bad_partition_spec_on_a_graft`.

    ABOVE that spec-shape gate, so the two doors agree: on the data door the
    conflict is hoisted above ``_reject_before_wrapper`` (which holds the shape
    check), so a bad-spec-plus-ladder call hears about the conflict there and must
    hear about it here too. It is also the more fundamental of the two — if
    ``partition=`` cannot be honoured on this node at all, whether the spec is
    well-formed is moot.

    ``False`` is not judged here at all: it is the no-partition bypass, and the
    per-leaf :func:`~.from_data.resolve_partition_beside_an_additive_ladder`
    deletes it once it reaches a laddered leaf, so the graft succeeds instead of
    stranding (measured pre-fix: the same childless ``kind=partition``, this time
    from ``Unknown node attribute 'partition'``).

    THE LADDER THIS GATE JUDGES IS THE RESOLVED ONE, not the stored one (#1632).
    Reading only ``iter_leaves`` was wrong in both directions, because
    ``additive_lod=`` can add a ladder the store has not got AND take away one it
    has:

    * an UNLADDERED nested store plus ``additive_lod={"n_lods": 2}`` stranded
      exactly the wrapper this gate exists to prevent — ``additive_lod=`` is not a
      parameter of :func:`graft_gsplat_node`, it rides in ``**attrs`` down to each
      part's ``add_gsplats_from_data_impl``, which BUILDS the ladder there, one
      level below the ``kind=partition`` that already exists. Measured: ``Could
      not add gsplats 'part_0': partition= is not supported alongside an
      additive_lod= ladder …`` with ``g`` surviving ``finalize()`` childless.
    * a LADDERED store plus a spec that COLLAPSES it —
      ``additive_lod={"n_lods": 1, "recompute": True}`` — resolves to a single
      rung, takes the flat route and partitions fine, yet the store-only test
      refused it (the ``False`` spelling was skipped by hand; this one was not).

    So each leaf is asked through :func:`~luxar.core.group.lod.gsplats.resolve_additive_rungs`,
    which states ``resolve_additive_axis_gsplats``'s own vocabulary as a rung
    count, over ``leaf.n_splats`` — the ladder's UNION, which is exactly the N
    ``make_additive_lod`` resolves its cuts against.

    AN UNKNOWN COUNT FALLS BACK TO THE STORE. ``None`` (energy-fraction
    breakpoints, a malformed dict, a non-dict junk spec) is a statement about the
    KWARG, not about the store: it says this gate cannot tell what the kwarg does,
    not that the store's ladder went away. Skipping the leaf instead threw away a
    conflict already visible in ``iter_leaves`` and re-stranded exactly the
    childless wrapper this gate exists to prevent — measured on a stored
    ``kind=partition`` of 2-sublod leaves with ``partition={"max_elements": 4}``,
    all three of ``{"recompute": True, "breakpoints": [0.3, 1.0]}``,
    ``{"recompute": True, "n_lods": 0}`` and a junk ``"stream"`` answered from
    inside ``part_0`` (the latter two with the builder's own ``n_lods must be
    positive`` / ``additive_lod must be None, bool, or dict``) and left ``g`` on
    disk as a childless ``kind=partition``. Falling back to ``stored_rungs``
    restores, byte for byte, the verdict the pre-#1632 store-only gate gave those
    same three calls.

    The rule "a query must not pre-empt the builder's own fault report" is
    preserved exactly where it belongs: on an UNLADDERED store the fallback is 1
    rung, so the leaf still skips and the builder still reports the malformed spec
    one level down, on a store that has no conflict of its own (the reporting
    residual below). The accepted cost is one over-refusal, on the other side:
    ``{"recompute": True, "breakpoints":
    [1.0]}`` on a LADDERED store is refused although it would in fact resolve to a
    single rung. That is identical to pre-#1632 behaviour — nothing regresses —
    and it is the conservative direction: refuse with an empty store, never write
    a wrapper and strand it.

    WHICH REMEDY is decided by the OFFENDING leaf's store alone —
    ``len(leaf.additive_sublods) > 1``, asked of a leaf whose RESOLVED count
    already exceeds 1 — not by whether the kwarg also builds that leaf's ladder.
    If any offending leaf is stored-laddered, the stored pair wins ('gsplat
    flatten'): unchanged wording for every call that passes no ``additive_lod=``,
    and the right answer in the usual MIXED case (one leaf stored-laddered,
    another laddered by the kwarg) and in the BOTH case (a laddered store plus
    ``{"n_lods": 4, "recompute": True}``) too, because the store's ladder is the
    obstacle that survives dropping the kwarg. Naming ``additive_lod=`` there
    would tell the caller to drop a kwarg whose removal only buys a SECOND
    refusal. Otherwise the ladder exists only because the kwarg builds it, and
    the ``additive_lod=`` pair applies.

    Note what keying on the OFFENDING leaf costs, deliberately: a stored ladder
    somewhere in the tree does NOT by itself force the stored wording. In a
    MIXED-COLLAPSE store — a laddered 4-splat leaf beside a flat 8-splat one
    under ``{"recompute": True, "breakpoints": [4]}``, which collapses the first
    to one rung and ladders the second — the only offending leaf is the FLAT one,
    so the message is the ``additive_lod=`` pair although a stored ladder sits
    untouched next door. That is the right call, not a miss: 'gsplat flatten'
    would send the caller to rewrite a ladder that is not in the way, whereas
    dropping either half of "drop one of the two" genuinely resolves it. The
    discriminator is which leaf BLOCKS the call, not which ladders exist.

    The reason BODY of that kwarg-built refusal is byte-identical to the one
    :func:`~.from_data.resolve_partition_beside_an_additive_ladder` raises for the
    same call one step later — :func:`partition_beside_a_ladder_reason` is
    literally the shared text. The whole MESSAGE matches only on the
    MATRIX-shaped branch of ``add_gsplats_from_file``, where the data door names
    the caller's own node anyway. On the GRAFT branch it deliberately does not:
    the data door would have said ``part_0``, and saying ``'g'`` instead is the
    entire point of hoisting the question up here — the tests assert that
    difference. And even the body matches only for a call whose ONLY fault is
    this conflict. This gate runs ABOVE the data door's ``lod_group=`` resolution, so a
    call carrying a second fault can be answered differently by NAME: measured,
    ``add_gsplats_from_file(..., lod_group="bogus", partition={…},
    additive_lod={"n_lods": 2})`` answers the conflict (``ValueError``) where
    ``add_gsplats_from_data`` with the same effective arguments answers
    ``TypeError: substitutive_lod (or lod_group alias) must be None, bool, or
    dict; got str``. Both refuse and
    both write nothing, so the divergence is in naming only — the same sanctioned
    class this module already documents for "a NaN position, an unknown attr".
    For the same reason a fault the COUNT cannot see (a bad ``method``, a stray
    ``substitutive_level`` key) is now MASKED by this refusal rather than reported
    by the builder: the same trade, taken because the conflict is the more
    fundamental fault and nothing is written either way.

    One reporting residual stays open on an UNLADDERED store, where the fallback
    is 1 rung and there is no stored conflict to judge: an INVALID call, reported
    by the builder — as it should be.
    ``additive_lod=True`` on an unladdered store counts 1 rung, skips here, and
    fails from inside ``part_0`` with "additive_lod=True requires every
    substitutive level to already carry an additive ladder"; a malformed spec
    does the same with its own message, there being no
    ``_reject_a_bad_additive_lod_spec_on_a_graft`` to hold it. Both fail
    identically WITHOUT ``partition=`` at all, so the fault is the call's and the
    verdict is the builder's, and pre-empting either would be the very thing this
    gate must not do. The outer transaction removes any partial graft.

    A VALID energy-fraction spec also reaches the child because its rung count is
    unknowable without building the energy curve. What fires is THIS refusal,
    not a builder fault. Measured:
    ``partition={"max_elements": 4}, additive_lod={"breakpoints": [0.5, 1.0]}``
    on an unladdered nested store answers ``Could not add gsplats 'part_0':
    partition= is not supported alongside an additive_lod= ladder …`` with ``g``
    being removed with every written descendant by the outer graft transaction.
    The spec is well-formed and
    ``make_additive_lod`` really does build 2 rungs from it; the count is UNKNOWN
    only because energy cuts need the ORDERING and the energy curve — the
    expensive half this query exists to avoid.

    Refusing on UNKNOWN instead was considered and rejected. ``{"breakpoints":
    [1.0]}`` resolves to a SINGLE rung and partitions perfectly well, so a gate
    that refused every uncountable spec would refuse a call the flat path
    accepts. That is its own regression, and a broader one than a strand reached
    only by asking for a multi-rung ENERGY ladder beside ``partition=``.
    """
    from luxar.gsplats.tree import iter_leaves

    from ..lod.gsplats import resolve_additive_rungs
    from ..partition import is_requested

    if not is_requested(attrs.get("partition")):
        return
    spec = attrs.get("additive_lod")
    offending = False
    from_the_store = False
    for leaf in iter_leaves(node):
        stored_rungs = len(leaf.additive_sublods)
        rungs = resolve_additive_rungs(
            spec,
            stored_rungs=stored_rungs,
            n_splats=leaf.n_splats,
        )
        if rungs is None:
            # UNKNOWN is about the KWARG, not the store — see the docstring's
            # AN UNKNOWN COUNT paragraph. The store's own ladder still stands.
            rungs = stored_rungs
        if rungs <= 1:
            continue
        offending = True
        if stored_rungs > 1:
            # A stored ladder outranks a kwarg-built one — see WHICH REMEDY
            # above; nothing later in the walk can change the answer.
            from_the_store = True
            break
    if not offending:
        return
    structure, remedy = (
        (STORED_LADDER_STRUCTURE, STORED_LADDER_REMEDY)
        if from_the_store
        else (LADDER_STRUCTURE, LADDER_REMEDY)
    )
    raise ValueError(
        f"Could not add gsplats '{name}': "
        + partition_beside_a_ladder_reason(structure, remedy)
    )


def _reject_a_bad_partition_spec_on_a_graft(
    name: str, node: Any, attrs: Dict[str, Any]
) -> None:
    """The graft door's half of the #1550 partition-spec gate.

    ``partition`` is not a kwarg this door consumes: it rides in ``child_attrs``
    all the way down to each part's own ``add_gsplats``, where it drives that
    leaf's BSP split. So an invalid spec was judged one level down — measured,
    ``Could not add gsplats 'part_0': partition must be None, True, or dict; got
    str`` — by which point :func:`graft_gsplat_node` had already built the
    ``kind=partition`` wrapper from the on-disk tree, and that childless wrapper
    survived ``finalize()``. Identical shape, and the identical fix, to the
    ``lod_group=`` door (``from_data._reject_before_wrapper``): the same
    ``reject_bad_partition_spec``, for its verdict alone.

    The ndim is the STORED tree's — every leaf of a tree shares one
    (``node_ndim``) — because that is the width each part would be split on, and
    it is what makes the sub-2-D skip mean the same thing here as there. The
    ``Could not add gsplats '<name>': `` prefix is applied by hand for the reason
    :func:`add_gsplats_from_file_impl`'s dimension-count check gives: this module
    has no funnel of its own.
    """
    from luxar.gsplats.tree import node_ndim

    try:
        reject_bad_partition_spec(attrs, node_ndim(node))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e


def _reject_labels_on_a_grafted_wrapper(
    name: str, node: Any, attrs: Dict[str, Any]
) -> None:
    """The graft door's half of the #1471 labels gate, run before any wrapper.

    :func:`graft_gsplat_node` builds its wrappers by calling ``add_lod_group`` /
    ``add_partition_group`` DIRECTLY, so it never meets
    ``from_data._reject_before_wrapper`` — and it is the door for every shape that
    gate cannot see (``gsplat lod --recipe tiles|overview|adaptive``, ``gsplat
    partition``, a ``batch-fit merge`` kind=partition). BOTH failure modes were
    live here: a wrong-length list refused from inside ``part_0`` with a childless
    ``kind=partition`` already on disk, and — worse — a list whose length HAPPENS
    to equal a part's own count is passed through whole by
    ``slice_optional_array`` and written onto EVERY part, silently, with no error
    at all.

    What is exempted is exactly what CAN carry labels: one leaf, with one
    additive sub-LOD. Both halves are load-bearing.

    * ONE LEAF, not "a leaf node" — a wrapper resolving to a single leaf still
      has an exact per-element correspondence, because that leaf holds every
      splat. ``luxar gsplat partition in out --parts 1`` emits exactly that (a
      ``kind=partition`` of one part holding a flat leaf, which the partition
      branch below already treats as "not a tiling" for its anchor choice), and
      it labelled correctly before this gate existed. Note ``gsplat lod --recipe
      tiles`` on a small dataset does NOT: it emits a one-part partition of a
      LADDERED leaf, which the sub-LOD half below refuses. ``iter_leaves``
      recurses, so nesting smuggles nothing past this: a one-child wrapper
      around a MULTI-leaf wrapper counts every leaf underneath and is refused
      HERE, at the entry call, naming the caller's node — the recursion never
      runs. A one-child wrapper whose whole nest still resolves to a single flat
      leaf is exempt for the reason above: that one leaf holds every splat —
      but the correspondence being EXACT is not the same as it being the right
      LENGTH (#1505). A wrong-length ``labels`` / ``keys`` / ``image_labels`` on this
      branch used to sail through here and refuse one level down, from inside
      ``part_0``'s own leaf write, with the wrapper this function was supposed
      to guard already on disk — the exact strand this gate exists to close,
      just on its exempt side rather than its refused one. So the exempt
      branch below now runs the same length/content validators the flat writer
      runs (``labels`` then ``keys`` then ``image_labels``, its own order), and re-raises
      with the same hand-applied prefix — giving a verdict byte-identical to
      what the flattened control (the remedy this gate already recommends)
      would raise on the same arrays, for a SINGLE-FAULT call.

      That closes the wrong-LENGTH case, which is the only one anything
      upstream of the wrapper can judge. A RIGHT-length list can still be
      refused one level down when a DECOMPOSITION kwarg is forwarded through
      the graft onto this same exempt leaf: ``partition=`` (including
      implicitly, under an auto-partitioning compiler), ``lod_group=`` or
      ``additive_lod=`` is each refused by the leaf ADDER (``Group.add_gsplats``
      and the ``from_data`` dispatch below it), for that kwarg's own reason. The
      outer transaction removes the partial wrapper, but the child-named verdict
      remains the pre-existing #1496-family residual, out of scope for a LENGTH
      fix — and untouched by #1496's own ``**attrs`` normalisation at the top of
      ``graft_gsplat_node`` either, which only reads a ``None`` as absent and
      refuses a data-owned CHANNEL: a decomposition kwarg carrying a real
      value is neither.

      This check runs at the TOP of ``graft_gsplat_node`` — below only the
      #1496 normalisation pair (:func:`~luxar.core.group.compositing.strip_absent_attr_kwargs`
      and :func:`~luxar.core.group.gsplats_pipeline.from_data.reject_data_owned_channels`,
      which touch none of the faults listed next) — so a call
      that ALSO trips an unrelated fault (an unknown attr, a bad node name, a
      duplicate sibling name, NaN centers) hears the label fault here, where
      the flat path answers the other fault first — both refuse, and neither
      writes. That is
      the same SANCTIONED divergence
      :func:`~luxar.core.group.compositing.validate_points_channels_before_split`
      documents for "a NaN position, an unknown attr", and which
      :func:`~luxar.core.group.gsplats_pipeline.from_data._reject_before_wrapper`
      already makes on the ``lod_group=`` door — not a regression to fix here,
      only a claim to state accurately.
    * ONE SUB-LOD — because ``write_gsplat_leaf_subtree``, where a laddered leaf
      goes, has no labels channel at all. Exempting a laddered leaf would make
      this gate's contract a lie: the refusal then comes from inside ``part_0``
      one level down. The transaction removes the one-part wrapper, but that is
      still a DIFFERENT fault, so it gets
      :func:`labels_on_a_laddered_leaf_reason` instead of the shared wrapper
      template, whose slicing argument would read as nonsense for a single leaf.

    Why multi-leaf cannot simply SLICE the way ``add_gsplats(partition=…)`` does:
    that path slices because it is handed the BSP ``parts: List[np.ndarray]`` it
    just computed. A stored :class:`~luxar.gsplats.tree.GSplatPartition` carries
    no per-part index arrays, so the only correspondence definable here would be
    implicit leaf-CONCATENATION order — precisely the storage-order coupling this
    gate exists to remove. ``gsplat flatten`` is an adequate remedy for the same
    reason: it emits that concatenation order (and collapses each leaf's ladder
    on the way), so the list a caller would have had to guess is exactly the list
    that works on the flattened file.

    Extracted to its own function rather than inlined so ``graft_gsplat_node``
    stays under the C901 limit the complexity ratchet enforces.
    """
    from luxar.gsplats.tree import iter_leaves

    # Defensive only — ``graft_gsplat_node`` now strips at its own entry, above
    # this call, and every terminal write under a graft funnels through
    # ``add_gsplats_from_data_impl``, which strips again at the top of its body.
    # Kept so this function's own ``is not None`` test reads the same normalised
    # attrs its callers will, rather than depending on either of them.
    strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_ATTRS)
    # ``labels`` before ``image_labels`` (the leaf adders' signature order), so a
    # call passing both is answered deterministically — same tie-break as the
    # ``lod_group=`` half of the gate.
    kwarg = next(
        (k for k in ("labels", "image_labels", "keys") if attrs.get(k) is not None),
        None,
    )
    if kwarg is None:
        return

    leaves = iter_leaves(node)
    only = next(leaves, None)
    if only is None:
        return  # no leaves at all — nothing this gate can say about it
    if next(leaves, None) is not None:
        reason = labels_on_wrapper_reason(kwarg, GRAFT_STRUCTURE, GRAFT_REMEDY)
    elif len(only.additive_sublods) > 1:
        reason = labels_on_a_laddered_leaf_reason(kwarg)
    else:
        # One flat leaf — the one shape that can carry them, PROVIDED the list is
        # also the right length (#1505; see the docstring's ONE LEAF bullet).
        _validate_labelled_leaf_length(name, only.n_splats, attrs)
        return
    # Prefixed by hand: this module has no try/except funnel, and its sibling
    # checks already report the prefixed form.
    raise ValueError(f"Could not add gsplats '{name}': {reason}")


def _validate_labelled_leaf_length(
    name: str, n_splats: int, attrs: Dict[str, Any]
) -> None:
    """Length/content-validate whichever string channel(s) are present.

    Reached only from the one-flat-leaf exempt branch above: that leaf's
    correspondence is exact, but a wrong-length list still has to be caught
    HERE, before this function returns and ``graft_gsplat_node`` proceeds to
    build the wrapper — for a wrong-LENGTH list specifically, there is no gate
    below this one that runs before ``part_0``'s own leaf write (#1505). (A
    RIGHT-length list can still be refused from inside the child via a
    decomposition kwarg forwarded onto the leaf — see the ONE LEAF bullet of
    :func:`_reject_labels_on_a_grafted_wrapper`'s docstring; the transaction
    cleans the wrapper, but this length check does not change that verdict.)

    ``n_splats`` is the exempt leaf's OWN count (``GSplatLeaf.n_splats`` sums
    over every additive sub-LOD, but this is only ever called with a leaf that
    has exactly one — the caller already refused more than one above — so the
    sum degenerates to that sub-LOD's own count, unambiguously).

    Runs ``labels`` then ``keys`` then ``image_labels`` — the flat writer's own
    order (``write_gsplats`` steps 0d then 0e) — and reuses its validators rather
    than re-implementing any rule:

    * ``labels`` → :func:`~luxar.core.group.compositing.validate_labels_before_split`,
      which no-ops on ``None`` and otherwise delegates to
      ``validate_labels_for_writing`` (the same function step 0d calls).
    * ``keys`` → ``validate_labels_for_writing`` directly with the key channel's
      own context and noun (the same function step 0d calls).
    * ``image_labels`` → guarded on ``is not None`` here (that validator has no
      built-in no-op) and then
      :func:`~luxar.io._compiler.labels.image_labels.validate_image_labels_for_writing`
      — the store-free pre-write validator #1491 extracted for exactly this
      kind of hoist, covering the dense length check, the sparse-dict key
      type/bounds, the one-shot-iterable refusal, and every entry's type.

    These validators raise a plain ``ValueError`` (``ValidationError`` is a subclass) or,
    for a malformed ``image_labels`` (a bad dict key type, a one-shot
    iterable, a wrongly-typed entry), a ``TypeError`` — caught here and
    re-raised as ``ValueError`` with the prefix this module applies by hand,
    ``from e`` so the original stays chained. Catching ``TypeError`` too
    matters: the leaf adders' own ``except (ValueError, TypeError)`` funnel
    does the same conversion on the flat path, so skipping it here would make
    the two paths diverge in exception TYPE, not just message. For a
    SINGLE-FAULT call the result is a verdict byte-identical to the flattened
    control's — same validators, same order, same prefix. It is not identical
    for a call that ALSO trips an unrelated fault (an unknown attr, a bad node
    name, NaN centers): this function runs first, above where those are
    checked, so it reports the label fault where the flat path would report
    the other one — see the ONE LEAF bullet of
    :func:`_reject_labels_on_a_grafted_wrapper`'s docstring for why that
    divergence is the same sanctioned trade
    ``compositing.validate_points_channels_before_split`` already documents.
    """
    from ....validation.base import validate_labels_for_writing
    from ..compositing import validate_labels_before_split

    try:
        validate_labels_before_split(attrs.get("labels"), n_splats)
        keys = attrs.get("keys")
        if keys is not None:
            validate_labels_for_writing(keys, n_splats, context="keys", noun="Keys")
        image_labels = attrs.get("image_labels")
        if image_labels is not None:
            validate_image_labels_for_writing(image_labels, n_splats)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e


def _graft_gsplat_node_transaction(
    group: "Group",
    *,
    name: str,
    node: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    attrs: Dict[str, Any],
) -> Union["GSplats", "Group"]:
    """Run one top-level graft with store, graph, and compiler-state rollback."""
    from ..lod.group import is_partition_bound

    parent_node = parent or group
    writer = group._require_scene_writer(group._find_scene())
    path = f"{parent_node.path}/{name}" if parent_node.path else name
    children_before = list(parent_node.children)
    try:
        with writer.transaction(path):
            return graft_gsplat_node(
                group,
                name=name,
                node=node,
                parent=parent,
                extend_to_all=extend_to_all,
                _under_partition=is_partition_bound(parent_node),
                **attrs,
            )
    except BaseException:
        parent_node.children[:] = children_before
        raise


def _ensure_graft_normalized(
    node: Any,
    attrs: Dict[str, Any],
    normalize_amplitudes: NormalizeSpec,
    normalized: bool,
) -> None:
    """Apply the one tree-wide insertion factor before graft recursion."""
    if normalized:
        return
    factor = normalize_node_in_place(node, normalize_amplitudes)
    stamp_amplitude_factor(attrs, factor)


def graft_gsplat_node(
    group: "Group",
    *,
    name: str,
    node: Any,  # luxar.gsplats.tree.GSplatNode
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    normalize_amplitudes: NormalizeSpec = True,
    _under_partition: Optional[bool] = None,
    _normalized: bool = False,
    **attrs: Any,
) -> Union["GSplats", "Group"]:
    """Graft a pre-built ``GSplatNode`` subtree into the scene, node-for-node.

    Composes the scene's own builders — ``add_gsplats_from_data`` (leaf /
    additive ladder), ``add_lod_group`` (kind=lod), ``add_partition_group``
    (kind=partition) — so a standalone ``.gsplats.zarr`` of any shape (including
    partition / nested) embeds as the identical subtree it holds on disk. The
    per-child ``coverage_fraction`` selector thresholds ride from each child's
    ``meta`` (so a nested lod combo stays selectable). Compositing attrs land on
    a wrapper Group; the rest fall through to children.

    ``_under_partition`` carries the partition binding of this node's surroundings
    and is TRI-STATE. ``None`` — the caller default — means "entry call: read the
    binding off the SCENE" by walking ``parent`` links up from the insertion point
    (:func:`~luxar.core.group.lod.group.is_partition_bound`), which is resolved
    once by :func:`_graft_gsplat_node_transaction`; the recursion then threads a
    concrete bool down. It is ``True`` once a ``kind=partition`` with **more than
    one part** has been crossed (a one-part partition is not a tiling — see the
    partition branch), and an outer ``True`` is never lost on the way down. It
    selects which anchor the FALLBACK ``coverage_fraction`` derivation uses (see
    the lod branch). Callers leave it at the default.

    The entry call is transactional. If any descendant builder fails after a
    wrapper has been written, the new top-level subtree is removed from the Zarr
    store, the insertion parent's child list and compiler-side bounds/warning
    state are restored, and the original exception is re-raised. Recursive calls
    carry a concrete ``_under_partition`` and participate in that one outer
    transaction rather than opening their own.

    Note what does NOT come here: only a **non-matrix-shaped** subtree is grafted
    at all. ``add_gsplats_from_file_impl`` sends every matrix-shaped tree — a bare
    leaf, an additive ladder, or a ``kind=lod`` group whose children are all
    leaves, which is what ``levels`` / ``stream`` / a plain fit writes — down
    ``add_gsplats_from_data_impl``, so the ordinary "one
    ``add_gsplats_from_file`` per part into a hand-built ``kind=partition``" case
    derives its thresholds in
    :func:`~luxar.core.group.gsplats_pipeline.lod_dispatch.add_gsplats_as_lod_group_impl`
    (via ``derive_coverage_fractions``), not in this function. The public
    ``Group.add_gsplats_from_file`` entry covers that matrix-shaped route with
    the same outer writer transaction as this graft path. Explicit
    ``extend_to_all`` values are still preflighted before its wrapper is written
    so that known failures keep the caller-facing node name.
    """
    if _under_partition is None:
        # Entry call: re-dispatch through the writer transaction, which comes
        # back into this function with a concrete ``_under_partition``. The
        # transaction forwards only ``attrs``, so ``normalize_amplitudes`` has to
        # ride in there or the caller's choice is silently replaced by the
        # default on the second frame — measured: an explicit
        # ``normalize_amplitudes=False`` normalised anyway.
        return _graft_gsplat_node_transaction(
            group,
            name=name,
            node=node,
            parent=parent,
            extend_to_all=extend_to_all,
            attrs={**attrs, "normalize_amplitudes": normalize_amplitudes},
        )

    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    from ..compositing import COMPOSITING_ATTRS

    parent_node = parent or group

    # The graft door's half of the #1496 ``**attrs`` normalisation — the identical
    # pair the ``from_data`` entry and ``add_gsplats_from_file_impl`` run, so a
    # DIRECT ``graft_gsplat_node(..., colors=None)`` (this function is called by
    # tests and by its own recursion, not only through the file door) behaves like
    # every other door and a non-None ``colors=`` is refused here rather than from
    # inside ``part_0``.
    #
    # Only the REFUSAL can be observed here, and it is genuinely load-bearing for
    # the error contract: this function builds its wrappers from the on-disk tree
    # BEFORE the first leaf write, so a collision judged one level down (by
    # ``add_gsplats_from_data_impl``, where every terminal leaf write does funnel
    # through) names ``part_0`` / ``child_0`` after transiently writing a wrapper.
    # The transaction now removes that wrapper, but this gate preserves the
    # caller's top-level name and avoids the write altogether. The
    # ``strip_absent_attr_kwargs``
    # call, by contrast, is UNREACHABLE IN EFFECT and no test can cover it: the
    # very next statement strips the same dict again (defensively, see
    # :func:`_reject_labels_on_a_grafted_wrapper`), and the per-leaf calls
    # underneath strip it a third time. It is kept as the entry-level statement of
    # intent — the pair is one rule and reads as one — not because anything
    # depends on it; if you delete it, nothing observable changes.
    strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_ATTRS)
    reject_data_owned_channels(name, attrs)
    reject_mesh_only_appearance("gsplats", name, attrs)
    _reject_a_partition_beside_a_stored_ladder(name, node, attrs)
    _reject_a_bad_partition_spec_on_a_graft(name, node, attrs)
    _reject_labels_on_a_grafted_wrapper(name, node, attrs)

    # Normalise ONCE, over the WHOLE subtree, at the entry call — then thread
    # ``_normalized`` down so no descendant repeats it. This is the one place a
    # partition / nested tree can be normalised at all: it never becomes a
    # ``GSplatData``, so it never reaches the data path's own normalisation.
    #
    # Doing it per node would be actively wrong, not merely redundant. Each part
    # of a tiling has its own amplitude distribution — the tile holding the
    # brightest region has a higher p99.9 than its neighbours — so a per-part
    # factor would scale adjacent tiles differently and reintroduce exactly the
    # cross-seam exposure mismatch ``finalize/amplitude_window.py`` exists to
    # prevent. Idempotence does not save us here: after a tree-wide scale a hot
    # tile's OWN p99.9 can still exceed 1.0, so ``auto`` would fire again on it
    # alone.
    _ensure_graft_normalized(node, attrs, normalize_amplitudes, _normalized)
    _normalized = True

    if isinstance(node, GSplatLeaf):
        # Matrix-shaped → the normal data path. A graft preserves the file's own
        # coordinates, so no scene-embed transforms are applied here.
        return add_gsplats_from_data_impl(
            group,
            name=name,
            result=GSplatData.from_tree(node),
            parent=parent,
            extend_to_all=extend_to_all,
            # Already normalised above, tree-wide. Re-running it on this leaf's
            # own distribution is the per-part hazard described there.
            normalize_amplitudes=False,
            **attrs,
        )

    # The entry transaction resolved the SCENE side of the binding exactly once
    # and every recursive call threads a concrete bool. It must not be re-asked
    # below the entry: ``parent_node`` is then a wrapper this graft itself just
    # created, so the walk would answer about our own freshly-written partition
    # and override the recursion's one-part exclusion.
    under_partition = _under_partition
    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    # `blending_mode` stays on the WRAPPER ONLY, like every other compositing
    # attr. It is nearest-setter-wins, so re-stamping it on each grafted part
    # (a former "belt-and-suspenders" duplication) made the parts SHADOW the
    # wrapper: the layers panel composes root→leaf and the part's own copy won,
    # so changing Blend on a partition / nested-lod layer changed nothing at all
    # while flat / stream / levels layers responded normally. The viewer's
    # ancestor inheritance resolves a wrapper-only mode on its own (the leaf
    # writers no longer stamp a shadowing "additive" default).
    # A ``coverage_fraction`` passed down by a parent lod-group is THIS node's own
    # selector threshold. For a leaf it is applied via ``add_gsplats_from_data``
    # (the leaf branch above); for a kind=lod / kind=partition WRAPPER it must land
    # on the wrapper group itself — NOT be silently dropped and NOT propagate to
    # the parts. Without this, a kind=partition child of a kind=lod group (the
    # ``multiscale`` recipe's fine branch) loses its threshold, so the viewer's
    # coverage selector can never switch to it and the lod is stuck on it. This
    # matches the standalone writer (``gsplat_tree.write_gsplat_node``), which
    # stamps coverage_fraction on the partition wrapper.
    self_coverage_fraction = child_attrs.pop("coverage_fraction", None)
    if self_coverage_fraction is not None:
        wrapper_attrs["coverage_fraction"] = self_coverage_fraction

    if isinstance(node, GSplatLodGroup):
        from luxar.gsplats.tree import total_splats

        from ..lod.group import coverage_fractions, partitioned_coverage_fractions

        wrapper_attrs.setdefault("display_type", "gsplats")
        # In-memory children are coarsest→finest, the same order add_lod_group
        # wants and the on-disk child_<i> layout uses — no reversal.
        on_disk = list(node.children)
        # Per-child coverage_fraction selector thresholds: prefer each child's
        # authored ``meta`` value, else derive — screen-occupancy halving (count-
        # ratios), so a meta-less grafted tree still gets ascending thresholds the
        # selector accepts (never all-zero).
        #
        # The fallback is TOPOLOGY-AWARE, matching the standalone writer
        # (``gsplat_tree.write_gsplat_node``): a ladder bound to a spatial partition
        # keeps the fills-screen anchor (see ``partitioned_coverage_fractions``).
        # Without this a meta-less partition-bound tree grafted with
        # ``add_gsplats_from_file`` would get whole-object anchors while
        # ``gsplat migrate-format`` on the same store gave partitioned ones. The
        # live trigger is a legacy pre-v3.2 store, whose ``min_pixel_size`` is not
        # lifted into ``meta``, so nothing authored wins over this fallback.
        #
        # TWO ways to be bound, and the first one already folds in the scene side:
        #   * ``under_partition`` — a partition crossed on the way here. Either one
        #     higher up THIS grafted subtree (the recursion's own flag, which
        #     excludes a one-part partition — see the partition branch), or, at the
        #     entry call, a ``kind=partition`` already in the SCENE above the
        #     insertion point (the ``is_partition_bound`` walk in
        #     ``_graft_gsplat_node_transaction``).
        #   * a child that IS a ``GSplatPartition`` — the ``overview`` recipe's
        #     cap↔fine pair, the common reachable case here.
        # The scene-side half is DEFENSIVE, not the fix for the everyday per-part
        # graft: only a non-matrix-shaped subtree reaches this function at all
        # (see the docstring), so with no binding from above the shape that gets
        # here and needs it is a nested lod-of-lods — which no library producer
        # writes today (``levels`` → lod of leaves, ``overview`` → lod of [leaf,
        # partition], ``adaptive``/``tiles`` → partition of …). A per-part
        # ``add_gsplats_from_file`` of an ordinary ladder file is matrix-shaped and
        # is anchored by ``add_gsplats_as_lod_group_impl`` instead. It is kept
        # because it costs one cheap parent walk and closes the asymmetry for a
        # hand-built / future nested tree — but ONLY as the entry call's seed, so
        # it can never contradict the recursion's own one-part exclusion.
        partition_bound = under_partition or any(
            isinstance(c, GSplatPartition) for c in on_disk
        )
        derive_cov = (
            partitioned_coverage_fractions if partition_bound else coverage_fractions
        )
        # SELECTOR/THRESHOLD CONSISTENCY — the shared all-or-none gate (see
        # ``gsplats.tree.gate_authored_selector``), the same one the standalone
        # writer runs, so a store grafted into a scene renders identically to
        # the same store opened directly.
        from luxar.gsplats.tree import gate_authored_selector

        on_disk, selector_out = gate_authored_selector(
            on_disk,
            (node.meta or {}).get("selector"),
            source="kind=lod group (scene graft)",
        )
        derived_cov = derive_cov([total_splats(c) for c in on_disk])
        # default_level = 0 = the COARSEST child (child_0): the viewer's initial
        # progressive-load level, decoupled from the data-model default (see
        # gsplat_tree.write_gsplat_node / add_gsplats_as_lod_group_impl). Loading
        # the finest by default would render "backwards".
        wrapper = parent_node.add_lod_group(
            name,
            selector=selector_out,
            **wrapper_attrs,
        )
        for i, child in enumerate(on_disk):
            cov = float((child.meta or {}).get("coverage_fraction", derived_cov[i]))
            graft_gsplat_node(
                wrapper,
                name=f"child_{i}",
                node=child,
                extend_to_all=extend_to_all,
                coverage_fraction=cov,
                _normalized=True,
                # A nested ladder inside a partition-bound one is still inside the
                # same tile, so the binding propagates down.
                _under_partition=partition_bound,
                **child_attrs,
            )
        return wrapper

    if isinstance(node, GSplatPartition):
        # Local import: the sibling `GSplatLodGroup` branch above imports this
        # too, but that branch does not run on this path.
        from luxar.gsplats.tree import total_splats

        partition_attrs = dict(wrapper_attrs)
        # Carry the BSP split planes into the scene so the viewer keeps its
        # exact back-to-front part ordering. Legacy partitions may predate the
        # stored tree; recover one only when the child bounds admit an exact
        # separating BSP. Overlapping/interlocking parts keep the viewer's
        # centroid fallback rather than receiving an invented ordering.
        bsp_tree = _grafted_partition_bsp_tree(
            node, group._find_scene().dimensions.displayed
        )
        if bsp_tree is not None:
            partition_attrs["bsp_tree"] = bsp_tree
        # `max_elements` is a per-part CAP, so only a capped splitter sets it
        # (uniform tiling, BSP `--parts`/`--max-elements`). A CONTENT-tiled fit
        # balances its boxes by feature density instead and leaves the field at
        # the `GSplatPartition` default of 0 — which `add_partition_group`
        # rejects, since it requires >= 1. Derive the honest value in that case:
        # the largest part IS this partition's effective per-part cap. The
        # attribute is descriptive downstream (the viewer logs it and does not
        # branch on it), so deriving cannot change rendering — whereas failing
        # here made every `fit --tiling content` result ungraftable.
        cap = int(node.max_elements)
        if cap < 1:
            # Floored at 1: an all-empty partition would otherwise derive a cap
            # of 0 and trip the very validator this branch exists to satisfy.
            cap = max(1, max(total_splats(child) for child in node.children))
        wrapper = parent_node.add_partition_group(
            name=name,
            display_type="gsplats",
            max_elements=cap,
            **partition_attrs,
        )
        # Everything under a kind=partition is partition-bound — EXCEPT when the
        # partition holds a single part, which is not a tiling: that part covers
        # the whole object, so a ladder underneath it keeps the whole-object
        # anchor. Mirrors the standalone writer (``gsplat_tree.write_gsplat_node``)
        # and the shape ``build_adaptive`` emits below ``max_elements``. The
        # exclusion only ever ADDS a binding, never drops an outer one: a one-part
        # partition nested inside a real tiling is still inside that one tile, so
        # OR the incoming binding in rather than overwriting it.
        child_under_partition = under_partition or len(node.children) > 1
        for i, child in enumerate(node.children):
            graft_gsplat_node(
                wrapper,
                name=f"part_{i}",
                node=child,
                extend_to_all=extend_to_all,
                _under_partition=child_under_partition,
                _normalized=True,
                **child_attrs,
            )
        return wrapper

    raise TypeError(f"Cannot graft unknown gsplat node type: {type(node).__name__}")


def add_gsplats_from_volume_impl(
    group: "Group",
    *,
    name: str,
    volume: np.ndarray,
    seeds: Optional[Union[int, float]] = None,
    n_iters: int = 1000,
    device: Optional[str] = None,
    progressive: bool = False,
    max_splats_per_pass: int = 5000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    normalize_amplitudes: NormalizeSpec = True,
    opacity: Optional[float] = None,
    absorption: Optional[float] = None,
    blending_mode: Optional[str] = None,
    **fit_kwargs: Any,
) -> Union["GSplats", "Group"]:
    if progressive:
        from luxar.gsplats import fit_progressive_gaussian_splats

        # Resolve the max-splats budget honoring the documented `seeds` contract
        # (int = exact count, float in (0, 1] = compression ratio). A float must
        # not be silently dropped — convert it the same way the non-progressive
        # fitter does. Only an unspecified seeds (None) falls back to the default.
        if seeds is None:
            max_splats = 50000
        elif isinstance(seeds, bool):  # guard: bool is an int subclass
            raise TypeError("seeds must be an int count or a float ratio, not bool")
        elif isinstance(seeds, int):
            max_splats = seeds
        elif isinstance(seeds, float):
            from luxar.gsplats.fitting.preprocessing import (
                _compression_ratio_to_target_count,
            )

            max_splats = _compression_ratio_to_target_count(seeds, volume.shape)
        else:
            raise TypeError(
                f"seeds must be an int count or float ratio, got {type(seeds).__name__}"
            )
        result = fit_progressive_gaussian_splats(
            volume,
            max_splats=max_splats,
            max_splats_per_pass=max_splats_per_pass,
            iters_per_pass=n_iters,
            psnr_patience=psnr_patience,
            max_passes=max_passes,
            device=device,
            **fit_kwargs,
        )
    else:
        from luxar.gsplats import fit_gaussian_splats

        result = fit_gaussian_splats(
            volume,
            seeds=seeds,
            n_iters=n_iters,
            device=device,
            **fit_kwargs,
        )

    scene_attrs: Dict[str, Any] = {}
    if opacity is not None:
        scene_attrs["opacity"] = opacity
    if absorption is not None:
        # Multiplicative compositing attr (identity 1.0): on a partitioned
        # result COMPOSITING_ATTRS routes it to the wrapper only, and the
        # children's stamped 1.0 defaults are no-ops under the product.
        scene_attrs["absorption"] = absorption
    if blending_mode is not None:
        scene_attrs["blending_mode"] = blending_mode

    return add_gsplats_from_data_impl(
        group,
        name=name,
        result=result,
        parent=parent,
        extend_to_all=extend_to_all,
        dim_order=dim_order,
        fill=fill,
        fill_sigma=fill_sigma,
        normalize_amplitudes=normalize_amplitudes,
        **scene_attrs,
    )
