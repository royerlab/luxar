"""The LOD topology each cached demo artifact ships with.

A ladder is not something a fit acquires by default. ``GSplatData`` carries
``additive_sublods``, and the progressive fitter happens to produce several as a
by-product of its passes while ``fit_gaussian_splats`` produces exactly one — so
which demos shipped a streaming ladder was decided by which fitter they called,
not by what their data needed. Five of the twenty-one shipped archives had no
ladder at all and none had a substitutive level, for that reason alone.

This module makes the choice explicit and reviewable, and gives the one call
demos need between fitting and saving. The recipe a dataset gets is a property
of the dataset, so it is recorded here rather than buried in each demo:

``stream``
    One leaf plus a progressive prefix ladder. Fast first paint; the right
    answer whenever the whole object is loaded anyway.
``levels``
    Coarse→fine replacement levels *and* a ladder per level. Worth its ~40%
    extra bytes on a large single object that is orbited, because the finest
    level still shows whenever the object fills half the screen or more.
``adaptive``
    Spatial tiles, each choosing its own level. For data panned and zoomed
    rather than orbited, where most tiles are off-screen most of the time.

Measured cost of ``levels`` over ``stream``, on ct_atlas (660,934 splats):
6,966 KB → 9,635 KB, i.e. +38%. Do not extrapolate that from a small fit — the
same comparison on a 743-splat fit reads 3.6x, which is per-zarr-group overhead
rather than data.

A recipe costlier than ``stream`` is only worth choosing if the demo's SCENE can
carry it, and that depends on which adder the demo builds with — see
:data:`TOPOLOGY_PRESERVING_ADDERS`.

CHOOSING: one object, viewed whole, under the cap -> ``stream``
---------------------------------------------------------------

The default should be ``stream``, and the burden of proof is on anything
costlier. All three qualifiers below are load-bearing; drop any one and the
answer can flip.

**one object** — nothing to frustum-cull. Parts exist so the viewer can skip
geometry that is off-screen. A compact specimen that is always fully in frame
has none, so a partition buys nothing and costs a request per node to bootstrap.
Measured (2026-08-26): first paint was **63 requests** for a single stacked leaf
(8 nodes) against **689** for a ``kind=partition`` of 44 parts x 4 levels x 4
rungs (704 nodes) — ~15-18x, like-for-like on chunking. Note the scaling is
SUBLINEAR (88x the nodes, ~16x the requests), so halving a node count does not
halve the cost.

**viewed whole** — the LOD selector is screen-occupancy based, so at a full-frame
view the FINEST substitutive level is what shows. Coarse levels are then bytes
nobody fetches. They earn their keep only where the object is genuinely small on
screen. Two live exceptions in this repo, and they are different from each other:

* the two 2D pathology slides author ``adaptive`` on ``--recompute`` because
  they are panned and zoomed, so most tiles are off-screen most of the time —
  which is what PARTS are for. Their pinned record archives currently serve
  four-part partitions without per-tile levels;
* ``milky_way_dust`` authors ``levels`` on ``--recompute``, and its pinned
  record archive preserves those levels, because the galaxy is orbited at range
  as well as inspected close up. A coarse level really is selected and fetched
  — which is what LEVELS are for. It pays +39% (7.88 -> 10.97 MB) on purpose.

``cryoem_virus`` is the counter-example: a single compact particle, always
full-frame, so its coarse levels were never selected and it moved to ``stream``
for a 28% saving when regenerated. Publishing that regenerated archive remains
tracked in #1879.

**under the cap** — ``MAX_SPLATS_PER_GSPLATS_NODE`` is 4,194,304 resident on a
4096-class GPU (8.38M at 8192). Above it the viewer reports the clamp at load
time and drops the tail; because storage is Hilbert-ordered that tail is one
contiguous lobe: a clean-edged hole rather than noise. **Only the RESIDENT slice
counts**, so an nD node sliced on a hidden axis is measured per-slice — a
500-timepoint node at ~165k splats/frame is 25x under the cap despite holding
82M in total. For a STATIC object above the cap, parts stop being a distraction
and become load-bearing. The compiler also warns on the node total rather than
the resident slice, so that warning is expected for a sliced nD node that
satisfies the runtime limit.

THE CAP IS PER GEOMETRY TYPE, AND RESIDENT IS EASY TO MISMEASURE
-----------------------------------------------------------------

Two mistakes cost real time on the 2026-08-28 Points/Lines pass, and both look
like a clean answer rather than an error.

**There is no single element ceiling.** ``typing_utils.constants`` derives one per
geometry from the element-texture layout: **2,793,472** segments for Lines,
**5,591,040** points for Points, 4,194,304 splats for GSplats. Comparing a Points
node against the gsplat number over-flags it by 1.33x; ``desi_galaxies``'
``SCENE_MAX_POINTS_PER_NODE = 4_000_000`` is a deliberate safety MARGIN under
5,591,040, not the cap.

**Measuring the resident slice of a PARTITIONED node has two traps.** Measure one
``part_N`` and you under-report by the part count — that read
``nuclear_pore_complex`` as 164,633 when it is 4,937,064, a 30x error that turned
a load-bearing partition into an apparently obvious removal. Sum each part's
LARGEST slice instead and you over-report, because different parts can peak on
different hidden coordinates (5,708,398 for the same node, which crosses the cap
and would have argued the opposite way). The resident set is, for ONE hidden
coordinate, the sum over every part: group globally by hidden coordinate FIRST,
then take the max. Likewise a leaf's own array plus its ``additive_<i>`` rungs
are one level's UNION — and its total is ``max(level count, sum of rungs)``,
never the sum of both, which double-counts to exactly 2x.

The practical upshot for Points demos: an embedding cloud stacked over 6 colouring
views at 6,248,730 total is 1,041,455 resident, five times under its cap. Its
substitutive levels are ~17-20% of the store serving a framing the screen-area
selector never picks, because the finest level is anchored at half-screen
occupancy and these demos open auto-fitted. Those go to :func:`stream_ladder`.
What stays: a node genuinely over its cap (``desi_galaxies``, 9,751,955 with no
hidden axis), and a partition that is load-bearing for a second reason —
``nuclear_pore_complex``'s subunits are concave and interpenetrate, so its
``bsp_tree`` split planes are the only valid draw order, camera inside the
channel included.

Measured cost of the alternatives, same flat fit, same knobs:

    cryoem_virus   1.01M splats   levels 16 nodes 16.03 MB -> stream 4 nodes 11.50 MB  (-28%)
    milkyway_dust  0.67M splats   levels 16 nodes 10.97 MB -> stream 4 nodes  7.88 MB  (-28%)
    droso 4D leaf  (3-frame proto)  +49% for levels, +9% for a partition-per-timepoint

The partition figure is the sharpest: on a time-stacked node it bought
*nothing*, because the writer already lexsorts by the time barrier and so gives
per-timepoint chunk locality with no partition at all.

ONE CAVEAT that comes with ``stream``: a flat store needs re-chunking or
scrubbing gets WORSE, not better. Measured on a 4D leaf, per timepoint step:
**173 requests as-built, 2 after ``luxar optimise --profile archive``** — the
as-built figure is worse than a partitioned store's re-chunked 12. Additive-only
and re-chunking are a package.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from arbol import aprint

from luxar.utils.lod_breakpoints import (
    DEFAULT_SLICED_LADDER_MAX_DEPTH,
    hidden_coordinate_count,
)

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData

#: Recipes usable from a demo. A subset of the full recipe set on purpose: the
#: rest need knobs (a density, a source volume, a tiled fit) that a demo's cache
#: step does not have in hand.
DemoRecipe = Literal["stream", "levels", "adaptive"]

#: Recipes whose artifact is a node TREE with no flat matrix form.
#:
#: This is not a detail of the writer — it decides how the demo READS its own
#: cache back. ``GSplatData.load`` (and therefore
#: :func:`luxar.demos.load_dataset_gsplats`) refuses a
#: ``kind=partition`` store outright, so a demo choosing one of these must graft
#: the PATH with :meth:`~luxar.core.group.Group.add_gsplats_from_file` instead.
#: Getting this wrong breaks the demo's DEFAULT path while leaving the
#: ``--recompute`` path — the one an author exercises — perfectly green.
TREE_RECIPES: frozenset[str] = frozenset({"adaptive"})

#: Scene adders that carry a stored topology into the scene.
#:
#: ``add_gsplats_from_data`` is handed the whole ``GSplatData``, whose matrix form
#: holds the substitutive levels, and re-emits them as a ``kind=lod`` group;
#: ``add_gsplats_from_file`` grafts the stored subtree node-for-node. The third
#: adder, plain ``add_gsplats(centers=…, amplitudes=…)``, is handed loose arrays
#: — every one of which is a view of the FINEST level only — so it writes a flat
#: leaf. Measured on a 4,000-splat fit: the same archive added via
#: ``add_gsplats`` gives a node with no ``kind`` and no child groups whether it
#: was written ``stream`` or ``levels``, while the ``levels`` archive is 4.1x the
#: bytes on disk (the small-fit figure — +38% on a real one, above).
#:
#: So a demo that rebuilds its scene from arrays gets nothing for the extra bytes
#: and should stay on ``stream``. It is not a rule the choice below can enforce
#: on its own — the archive is written in the fit step and read in the scene step,
#: often hundreds of lines apart — so ``tests/test_lod_policy.py`` gates it.
TOPOLOGY_PRESERVING_ADDERS: frozenset[str] = frozenset(
    {"add_gsplats_from_data", "add_gsplats_from_file"}
)

#: Recipes whose extra bytes only pay off if the topology reaches the scene.
#:
#: ``stream`` is excluded because a prefix ladder is the same splats regrouped —
#: it costs essentially nothing, so it is the right floor even for a demo whose
#: scene flattens it (the archive is still downloadable and ``luxar gsplat view``
#: honours the ladder).
SCENE_TOPOLOGY_RECIPES: frozenset[str] = frozenset({"levels", "adaptive"})

#: Per-recipe parameters, so two demos choosing ``levels`` cannot drift apart.
#:
#: ``n_lods=4`` throughout: four rungs is the shipped default and what every
#: laddered archive already carries, so it keeps regenerated artifacts
#: comparable to the ones they replace.
_RECIPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "stream": {"n_lods": 4},
    "levels": {"n_lods": 4, "compression_factor": 4, "levels": 3},
    # max_elements caps splats per tile; K/L are the per-tile level ladder.
    # L=2 rather than 3 because a tile is already a fraction of the object, so
    # a third level would coarsen past anything a viewer requests.
    #
    # max_elements was 250,000, and that made `adaptive` by far the most
    # node-expensive thing in the corpus. Measured on the built
    # gsplats_2d_codex_pancreas store: 12 channels x 172 tiles x 3 levels x 4
    # rungs = 12 + 172 + 516 + 2,064 = **2,764 groups**, three times the next
    # largest store and 24% of the whole 91-store corpus's 6,924. On the
    # published, post-`optimise --profile archive` store, hosted first paint
    # costs roughly one request per node, and node count is what Loic named as
    # the thing that slows loading.
    #
    # 1,000,000 quarters the tile count. MEASURED, by flattening two of codex's
    # real per-channel archives and rebuilding this recipe at both values (no
    # refit needed — `adaptive` is applied to a GSplatData at save time):
    #
    #   channel   splats     250k -> tiles/groups    1M -> tiles/groups
    #   ch01        562,180        4 / 65                 1 / 17
    #   ch05      1,114,331        8 / 129                2 / 33
    #
    # ch01's 4 tiles at 250k is exactly what the shipped store holds for that
    # channel, which is what anchors the extrapolation: 172 tiles -> ~43, so
    # 12 wrappers + 43 lod + 129 levels + 516 rungs = **~700 groups**, a 3.9x
    # reduction. A tile stays 4x under the 4,194,304 gsplat cap. What it
    # costs is culling granularity — ~43 tiles over a 46000x33000 slide is ~6-7
    # per side instead of ~13 — which is the right trade for a recipe whose whole
    # point is that most tiles are off screen anyway: at 6-7 per side a
    # full-screen view still holds a minority of them.
    #
    # Shared rather than per-demo on purpose. The only two `adaptive` users are
    # the two 2D pathology slides, which want the same thing, and this table
    # exists so they cannot drift apart — a per-demo override would defeat that
    # for no benefit here. Both need a REFIT for the change to reach their
    # artifacts (they fit locally and graft their own output).
    "adaptive": {
        "n_lods": 4,
        "max_elements": 1_000_000,
        "compression_factor": 4,
        "levels": 2,
    },
}


#: Deepest additive ladder a SLICED node may carry, as a share of its frame.
#:
#: `1/8` floors the first rung at 12.5% of the node, hence 12.5% per resident
#: slice on average. Sized to clear the viewer-side gate for #2374 — which fails
#: a sliced node below **10%** of its frame, or below 250 elements at the
#: 5th-percentile stop — with margin.
#: :func:`stream_ladder` rejects a sliced leaf whose share would exceed the
#: 900,000-element commit ceiling; partition after moving any stacked axis last,
#: or supply explicit capped cuts, rather than silently delivering less than
#: this contract. Deepening this past 10 walks a sliced node into the failing band.
#:
#: THE 12.5% IS AN AGGREGATE, NOT A PER-STOP GUARANTEE. Rung 0 is a prefix of a
#: global ordering, so it concentrates where the signal is rather than spreading
#: over stops in proportion to slice size. On a non-uniform hidden axis the
#: sparsest stops get less than `1/L` of their own already-small slice —
#: measured on a published 500-timepoint demo, `additive_0` is a median of 45
#: splats per timepoint but **p05 = 7, min = 1, and 30 stops under 10**, against
#: 41.7 predicted by a uniform assumption. A categorical axis of 2-7 stops has
#: little room to be non-uniform, which is the case this constant was sized for;
#: a long timelapse over a growing specimen does not, and wants its per-stop
#: histogram checked rather than this constant trusted.
SLICED_LADDER_MAX_DEPTH = DEFAULT_SLICED_LADDER_MAX_DEPTH


def hidden_axis_stops(positions: Any, hidden_dims: Sequence[int]) -> int:
    """How many slices a node splits into along its NON-DISPLAYED axes.

    This is an upper bound on the number of resident selections: exact when each
    element belongs to one hidden coordinate, and an over-count when pooled or
    marginal layouts duplicate elements across selections. :func:`stream_ladder`
    only needs the ``> 1`` predicate, so that over-count is safe there.

    Counts distinct COMBINATIONS across all hidden columns, not the product of
    each column's cardinality: a node stacked on time *and* channel is only
    sliced by the pairs that actually occur, and on sparse data the product
    massively overestimates.

    Args:
        positions: ``(N, d)`` element coordinates — the same array passed to
            ``add_points`` / ``add_lines``.
        hidden_dims: Column indices that are NOT displayed. Pass
            ``dims.non_displayed`` rather than literals — demos do not agree on
            where the hidden axis sits (the multiome demos put it at column
            **0**, the timelapse demos put it last), so a hardcoded index is
            wrong somewhere and silently: pointing it at a displayed spatial axis
            counts thousands of distinct floats and inflates the count enormously.
            Empty means the node is shown whole, which returns 1.

    Returns:
        The number of distinct hidden coordinates, at least 1.
    """
    import numpy as np

    cols = [int(c) for c in hidden_dims]
    if not cols:
        return 1
    arr = np.asarray(positions)
    if arr.ndim != 2:
        raise ValueError(f"positions must be 2-D (N, d); got shape {arr.shape}")
    bad = [c for c in cols if not 0 <= c < arr.shape[1]]
    if bad:
        raise ValueError(
            f"hidden_dims {bad} out of range for positions with {arr.shape[1]} columns"
        )
    return hidden_coordinate_count(arr, cols)


def stream_ladder(
    n: int, *, geometry: str = "points", slices: int = 1
) -> dict[str, Any]:
    """The additive ladder a Points/Lines leaf shown whole should carry.

    The Points/Lines counterpart of choosing ``stream`` above. Those adders take
    a ladder spec directly rather than going through a recipe, and — unlike the
    substitutive path, where a ladder is composed in by default — an additive
    ladder on a PLAIN leaf is strictly opt-in. So dropping ``substitutive_lod=``
    from a demo silently drops its ladder too unless this is passed; that is the
    single easiest mistake to make in this rework, and
    ``scripts/check_demo_ladders.py`` is the backstop (it fails an un-laddered
    leaf above 200,000).

    Two numbers, and both are chosen rather than inherited:

    **First rung = the ~200 ms download budget** (39,062 elements at 25 Mbps and
    16 B/element), matching ``default_composed_additive_lod`` when unsliced;
    both paths apply the same resident-share floor once sliced. Not desi's 2,000:
    that is sized to land in a single zarr chunk because its EAGER COARSEST
    SUBSTITUTIVE LEVEL is what paints first. An additive-only leaf has no coarse
    level, so its first rung IS first paint, and a 2,000-point opening frame buys
    latency nobody asked for while costing rungs. In the same order as the
    runbook table — mouse, ESM3, zebrahub, cellxgene, human and arxiv — measured
    group counts (wrapper + rungs) at 2,000 are 12/12/15/14/17/18 against
    13/13/17/18/17/18 today, so there is no reduction at all. At the budget
    chunk they are **7/7/11/9/13/13** unsliced and **5/5/7/6/9/9** with the
    sliced share floor, because every rung saved is a node saved, and on the
    published, post-``optimise --profile archive`` store, hosted first paint
    costs roughly one request per node.

    **Capped increments**, via :func:`~luxar.utils.lod_breakpoints.
    capped_stream_cuts` — a plain doubling ladder's last commit grows with ``n``
    and would block the main thread on these leaves.

    A SLICED NODE WANTS A SHARE, NOT A BUDGET — PASS ``slices``
    -----------------------------------------------------------

    The first rung is an ABSOLUTE count, but the viewer draws one hidden-axis
    coordinate at a time. So on a sliced nD node a rung of 39,062 rows arrives as
    39,062/S rows for the coordinate on screen. Measured on the demos this
    laddered, per resident slice:

    ======================== ====== ============ ===========
    demo                      slices rung 0/slice first paint
    ======================== ====== ============ ===========
    human_multiome_peak_umap      6        6,510      0.63 %
    zebrahub_multiome_umap        7        5,580      0.87 %
    cellxgene_census_umap         3       13,021      1.30 %
    mouse_multiome_peak_umap      6        6,510      3.39 %
    esm3_protein_landscape        2       19,531      3.39 %
    arxiv_papers_kaggle           2       19,531      0.59 %
    ======================== ====== ============ ===========

    Scaling the rung to ``budget × S`` would restore the intended *latency* — 200
    ms per slice — but Loic ruled for the **share** contract instead (2026-08-30),
    and the measurements are why: what predicts whether an opening frame is
    recognisable is the share of the frame, not the bytes spent. Predicted
    rung-0-per-slice against observed render: **0.03% renders blank** (a
    500-timepoint gsplat demo, decoded and confirmed against playback at 20-51
    splats), **12.5% is soft but usable** (neuromast, 110,614 measured at rest
    against a 113,947 metadata mean), and **54% is fine on only 1,735 absolute
    elements** — which is what rules out an absolute floor. A budget contract
    would have left all six demos above under 10% of their frame.

    So above one slice the first rung is floored at ``n /
    SLICED_LADDER_MAX_DEPTH``, giving every sliced node ``1/L`` of its frame **on
    average** — see that constant for why the sparsest stop can be worse, and why
    that is acceptable on a 2-7 stop categorical axis but not on a long
    timelapse. Slice-invariant, so a demo that gains a dimension cannot silently
    regress. Derive ``slices`` with :func:`hidden_axis_stops`, not by hand.

    The cost is accepted rather than hidden: first paint on these demos goes from
    ~200 ms to ~123-2,100 ms, with the upper end from arxiv's two-stop shape.
    Part of that is repaid in requests — hosted first paint is dominated by
    request COUNT, and fewer, fatter rungs mean fewer nodes to fetch, so the
    wall-clock penalty is smaller than the byte arithmetic suggests.

    **The fatal version of this is on a PLAYED axis, and it wants a different
    ladder.** Under a playback frame budget a cold ladder's opening frame starts
    from level 0; since #2377, any further cache-resident rungs can join it. The
    published playback measurements below used the old LOD-0-only policy, but
    level 0 remains the floor each timepoint starts from. A time-budget rung
    leaves each timepoint with a handful of elements — measured on a published
    500-timepoint gsplat demo, a median of 45 splats per timepoint, which renders
    as nothing (#2374). An
    EQUAL-COUNT ladder (``--n-lods L``) gives ``1/L`` of the node in aggregate,
    making the average slice share independent of ``S``. Its global prefix can
    still concentrate away from sparse slices, so check the per-stop histogram
    on a long or non-uniform played axis. Do NOT reach for it here: on a
    keypress-navigated categorical axis it would put ~40x the 200 ms budget into
    first paint and buy a slow opening frame these demos do not need. Two
    regimes, two answers.

    LINES SPELL THIS DIFFERENTLY, AND THE UNITS DISAGREE
    ----------------------------------------------------

    On ``add_lines`` an explicit ``counts`` list is in **POLYLINES**, while
    ``"stream:<c>"`` is in **VERTICES** (the writer converts each vertex target
    to the first whole-polyline boundary that reaches it, because a ladder can
    only cut on polyline boundaries without breaking segment topology).

    That asymmetry fails SILENTLY in the direction you would hit by accident.
    Measured on 4,000 polylines x 27 vertices = 108,000 vertices: a vertex-sized
    list ``[39062, 78124, 108000]`` exceeds the 4,000 polylines, so
    ``_validate_counts`` clamps every entry to 4,000, collapses the ladder to one
    level, and writes **no rungs at all** — no error, no warning. The same node
    with ``"stream:39062"`` gets three rungs of 39,069 / 39,069 / 29,862
    vertices. (``scripts/check_demo_ladders.py`` is the only thing that catches
    the silent case, and only above 200,000.)

    So for ``geometry="lines"`` this returns the STRING form, which is the one
    whose unit matches the ``n`` a caller naturally has. The cost is that the
    string form is a plain doubling ladder — its last increment approaches ``n/2``
    rather than being capped. The resolved ladder is checked against the 900,000
    ceiling because the crossing point depends on the effective first chunk;
    a larger Lines leaf needs capped cuts expressed in polylines instead.

    Args:
        n: Element count of the leaf — points for ``"points"``, VERTICES for
            ``"lines"``.
        geometry: ``"points"`` or ``"lines"``. Selects the spelling, because the
            two are not interchangeable (above).
        slices: Number of hidden-axis coordinates the node splits into, from
            :func:`hidden_axis_stops`. Above 1, the first rung is floored at
            ``n / SLICED_LADDER_MAX_DEPTH`` so first paint is a usable SHARE of
            the resident slice. Leaving it at 1 on a sliced node leaves that node
            with a first paint of ``budget / slices`` elements.

    Returns:
        A spec for ``additive_lod=`` on :meth:`Group.add_points` /
        :meth:`Group.add_lines`.

    Raises:
        ValueError: ``geometry`` is neither ``"points"`` nor ``"lines"``, or a
            sliced leaf cannot deliver its share within the commit ceiling, or a
            Lines leaf is too large for the uncapped vertex-count string form.
    """
    if geometry not in ("points", "lines"):
        raise ValueError(
            f"geometry must be 'points' or 'lines'; got {geometry!r} "
            "(the two spell their ladder in different units)"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if slices < 1:
        raise ValueError(
            f"slices must be >= 1; got {slices} "
            "(1 means the node is shown whole — see hidden_axis_stops)"
        )
    from luxar.core.group.lod.group import (
        DEFAULT_LADDER_BYTES_PER_ELEMENT,
        DEFAULT_LADDER_TARGET_MS,
    )
    from luxar.utils.lod_breakpoints import (
        DEFAULT_BANDWIDTH_MBPS,
        DEFAULT_MAX_ADDITIVE_COMMIT,
        capped_stream_cuts,
        sliced_ladder_first_chunk,
        stream_cuts,
        streaming_chunk_splats,
    )

    first_chunk = streaming_chunk_splats(
        DEFAULT_LADDER_TARGET_MS,
        DEFAULT_BANDWIDTH_MBPS,
        DEFAULT_LADDER_BYTES_PER_ELEMENT,
    )
    if slices > 1:
        # Loic's ruling, 2026-08-30: on a sliced node the contract is a SHARE of
        # the frame, not a download budget. A share is what tracks whether the
        # opening frame is recognisable (measured: 0.03% blank, 0.87% thin, 12.5%
        # soft-but-usable, 54% fine), and an aggregate `1/L` share is invariant
        # to the slice count, so it cannot regress merely because a demo gains a
        # dimension. The per-stop distribution can still be uneven; see
        # SLICED_LADDER_MAX_DEPTH.
        #
        # Note the floor needs no `slices` term. Requiring
        # `first_chunk/S >= share * (n/S)` cancels to `first_chunk >= share * n`,
        # so the arithmetic is slice-independent and `slices` only decides
        # WHETHER the floor applies — an unsliced node keeps its budget ladder
        # untouched, which is why no existing unsliced caller moves.
        share_chunk = sliced_ladder_first_chunk(
            first_chunk,
            elements=n,
            slices=slices,
            max_depth=SLICED_LADDER_MAX_DEPTH,
        )
        if share_chunk > DEFAULT_MAX_ADDITIVE_COMMIT:
            raise ValueError(
                f"Sliced node with {n:,} elements cannot deliver its "
                f"{100 / SLICED_LADDER_MAX_DEPTH:g}% first "
                f"rung within the {DEFAULT_MAX_ADDITIVE_COMMIT:,}-element commit "
                "ceiling. Partition this leaf after moving any stacked axis last, "
                "or supply explicit capped cuts instead of stream_ladder."
            )
        first_chunk = share_chunk
    if geometry == "lines":
        cuts = stream_cuts(int(n), first_chunk)
        largest_commit = max(
            (cut - previous for previous, cut in zip([0, *cuts[:-1]], cuts)),
            default=0,
        )
        if largest_commit > DEFAULT_MAX_ADDITIVE_COMMIT:
            raise ValueError(
                "Lines streaming ladder with resolved chunk "
                f"{first_chunk:,} exceeds the {DEFAULT_MAX_ADDITIVE_COMMIT:,}-vertex "
                f"commit ceiling (largest commit {largest_commit:,}) for n={n:,}. "
                "Supply capped polyline-count cuts for this leaf instead."
            )
    counts: Any = (
        f"stream:{first_chunk}"
        if geometry == "lines"
        else capped_stream_cuts(int(n), first_chunk)
    )
    return {
        "counts": counts,
        # `random` is the house default and the right reveal for a density cloud:
        # a random prefix reads as a sparser version of the whole. The
        # spatial-uniform sampler walks a doubling grid over the BOUNDING BOX,
        # which is measurably worse on a shell (see the ocean-currents notes).
        "method": "random",
        "seed": 0,
    }


def save_with_lod(
    data: "GSplatData",
    path: Path,
    *,
    recipe: DemoRecipe,
    device: Optional[str] = None,
    quiet: bool = False,
    **save_kwargs: Any,
) -> None:
    """Apply ``recipe`` to ``data`` and write the cached artifact.

    Replaces a bare ``data.save(path, ...)`` in a demo's cache step. Handles
    both shapes a recipe can return: the matrix recipes give back a
    ``GSplatData`` (whose own ``save`` is used, so ``save_kwargs`` pass through
    untouched), while ``adaptive`` gives back a node tree that needs
    ``write_gsplats_tree`` and an explicitly split stats dict.

    The fit's ``stats`` survive the recipe unchanged, which is what keeps the
    foreground PSNR and the source-grid provenance on the artifact — a LOD step
    that dropped them would leave the published compression figure and quality
    numbers with nothing behind them.

    Parameters
    ----------
    data
        The fitted splats.
    path
        Destination, e.g. ``.../name.gsplats.zarr.zip``.
    recipe
        Which topology to build. See the module docstring for why each exists.
    device
        Passed to the substitutive reduction (``levels``/``adaptive`` only).
        ``None`` lets it choose; pass ``"cpu"`` on Apple silicon, where the MPS
        backend lacks the float64 support the reduction wants and warns as it
        falls back anyway.
    quiet
        Suppress the one-line report of what was built.
    **save_kwargs
        Forwarded to the writer: ``encoding_mode``, ``compress``,
        ``zip_deflate``, ``ordering``, ``include_fitting_info``, ``description``.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
    from luxar.gsplats.lod import RecipeParams, build_recipe

    params_kwargs = dict(_RECIPE_DEFAULTS[recipe])
    if device is not None:
        params_kwargs["device"] = device
    result = build_recipe(data, recipe, RecipeParams(**params_kwargs))

    # Build provenance: which recipe made this. Distinct from the `lod_kind` the
    # builder stamps, which is the mechanism the viewer actually reads.
    if isinstance(result, GSplatData):
        result.stats["recipe"] = recipe
        result.save(path, **save_kwargs)
    else:
        fitting_info, fitting_config, provenance_info, pipeline_info = (
            split_fitting_info(
                data.stats,
                include_fitting_info=save_kwargs.pop("include_fitting_info", True),
                include_provenance=save_kwargs.pop("include_provenance", False),
            )
        )
        # `include_*` are consumed above; the rest are writer kwargs.
        write_gsplats_tree(
            path,
            result,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            pipeline_info={**(pipeline_info or {}), "recipe": recipe},
            **save_kwargs,
        )

    if not quiet:
        aprint(f"  LOD topology: {recipe} ({_describe(recipe)})")


def _describe(recipe: str) -> str:
    """One phrase per recipe, for the demo's console output."""
    defaults = _RECIPE_DEFAULTS[recipe]
    if recipe == "stream":
        return f"{defaults['n_lods']}-rung progressive ladder"
    if recipe == "levels":
        return (
            f"{defaults['levels']} coarse levels at 1/{defaults['compression_factor']} "
            f"each, {defaults['n_lods']}-rung ladder per level"
        )
    return (
        f"spatial tiles of <={defaults['max_elements']:,} splats, "
        f"{defaults['levels']} levels per tile"
    )
