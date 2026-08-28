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

* the two 2D pathology slides keep ``adaptive`` because they are panned and
  zoomed, so most tiles are off-screen most of the time — which is what PARTS
  are for;
* ``milky_way_dust`` keeps ``levels`` because the galaxy is orbited at range as
  well as inspected close up, so a coarse level really is selected and really is
  fetched — which is what LEVELS are for. It pays +39% (7.88 -> 10.97 MB) on
  purpose.

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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from arbol import aprint

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
    "adaptive": {
        "n_lods": 4,
        "max_elements": 250_000,
        "compression_factor": 4,
        "levels": 2,
    },
}


def stream_ladder(n: int, *, geometry: str = "points") -> dict[str, Any]:
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
    16 B/element), matching ``default_composed_additive_lod``. Not desi's 2,000:
    that is sized to land in a single zarr chunk because its EAGER COARSEST
    SUBSTITUTIVE LEVEL is what paints first. An additive-only leaf has no coarse
    level, so its first rung IS first paint, and a 2,000-point opening frame buys
    latency nobody asked for while costing rungs. Measured group counts (wrapper
    + rungs) for the six embedding demos: at 2,000 they are 12/12/14/15/17/18
    against 13/13/18/17/17/18 today — no reduction at all. At the budget chunk
    they are **7/7/9/11/13/13**, because every rung saved is a node saved, and
    hosted first paint costs roughly one request per node.

    **Capped increments**, via :func:`~luxar.utils.lod_breakpoints.
    capped_stream_cuts` — a plain doubling ladder's last commit grows with ``n``
    and would block the main thread on these leaves.

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
    rather than being capped. That is fine up to a few million vertices (at
    1.8M it is under the 900,000 ceiling) and both Lines demos using this are far
    below that; a much larger Lines leaf would want capped cuts expressed in
    polylines instead.

    Args:
        n: Element count of the leaf — points for ``"points"``, VERTICES for
            ``"lines"``.
        geometry: ``"points"`` or ``"lines"``. Selects the spelling, because the
            two are not interchangeable (above).

    Returns:
        A spec for ``additive_lod=`` on :meth:`Group.add_points` /
        :meth:`Group.add_lines`.

    Raises:
        ValueError: ``geometry`` is neither ``"points"`` nor ``"lines"``.
    """
    if geometry not in ("points", "lines"):
        raise ValueError(
            f"geometry must be 'points' or 'lines'; got {geometry!r} "
            "(the two spell their ladder in different units)"
        )
    from luxar.core.group.lod.group import (
        DEFAULT_LADDER_BYTES_PER_ELEMENT,
        DEFAULT_LADDER_TARGET_MS,
    )
    from luxar.utils.lod_breakpoints import (
        DEFAULT_BANDWIDTH_MBPS,
        capped_stream_cuts,
        streaming_chunk_splats,
    )

    first_chunk = streaming_chunk_splats(
        DEFAULT_LADDER_TARGET_MS,
        DEFAULT_BANDWIDTH_MBPS,
        DEFAULT_LADDER_BYTES_PER_ELEMENT,
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
