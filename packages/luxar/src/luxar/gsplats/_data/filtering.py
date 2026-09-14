"""Attribute/geometry filtering mixin for ``GSplatData``, and the ``stats`` rules.

Inherited-``stats`` hygiene comes in THREE categories, because a rewritten
artifact can invalidate a stamp along any of three independent axes and no single
predicate covers more than one:

* :data:`_REGION_SCOPED_STATS_KEYS` — what the splats REPRESENT. Invalidated by a
  spatial restriction (a bbox/slice that excluded splats), and by nothing else:
  an amplitude threshold leaves the represented region exactly as it was.
  Predicate: :func:`_is_crop`.
* :data:`_CONTENT_SCOPED_STATS_KEYS` — MEASURED reconstruction scores of the
  splat set against the SOURCE VOLUME (the PSNR family), plus the
  :data:`_CONTENT_SCOPED_OP_RECORD_KEYS` record of the reduction that produced
  the artifact. Invalidated whenever the splat set changes, spatially or not: an
  amplitude-threshold cull leaves the region untouched while changing the
  reconstruction completely, and a merge-family reduction can hit the requested
  count exactly while replacing every splat with a representative. Predicate:
  "did the content change" — :func:`_stats_after_content_change`.
* :data:`_STRUCTURE_SCOPED_STATS_KEYS` — the artifact's OWN TOPOLOGY: which LOD
  mechanism built it, how many substitutive levels and additive rungs it has,
  where the ladder cutpoints fall, which recipe was run. Invalidated when a
  rewrite changes the STRUCTURE KIND, and by nothing else: a cull rewrites the
  content while leaving a substitutive pyramid a substitutive pyramid (and its
  root ladder counts are RE-STAMPED, not dropped — see
  :func:`~luxar.gsplats._data.transforms._refresh_ladder_summary`, reached from
  ``_map_substitutive`` / ``_map_additive`` through
  ``_stats_after_ladder_rebuild``; the neighbouring
  :func:`~luxar.gsplats.lod.restamp.refresh_reduction_lod_stats` passes the ROOT
  dict through verbatim and rewrites only the per-level / per-rung stamps),
  whereas ``flatten`` turns that pyramid into one flat leaf without touching a
  single splat. Predicate: the rewriting command's own knowledge that it produced
  a different kind of thing — :func:`stats_after_structure_change`.

Reusing the region predicate for the metrics is what #1600 was: a ``cull -r 0.5``
that halved the splat count published the pre-cull PSNR as its own, and ``gsplat
info`` reads ``psnr_db`` as THE dataset's reconstruction quality. The third axis
is the same issue's other half: ``flatten`` / ``partition`` / ``lod`` /
``decimate`` published ``lod_kind: substitutive``, ``n_substitutive_levels: 4``
and ``lod_cutpoints: [2, 4, 5, 7]`` for a store that is one flat leaf (or four
bare parts) — ``lod --recipe flat`` printing that block one line above its own
``recipe: flat``. No category subsumes another — a whole-volume bbox that removed nothing
keeps all three, a non-spatial cull keeps the region and topology stamps and
loses the metrics, an actual crop loses region and metrics but keeps the topology,
and ``flatten`` loses only the topology.

The LOD Q·e ladder stamps are a fourth, artifact-local category —
``lod_stats.energy_fraction_cum`` (the prefix energy e(k) of a rung),
``level_stats.reference_energy`` (its weight w) and ``level_stats.quality`` (the
measured Q of a level against its group's finest). Those are measured on the
artifact's OWN content rather than against a source volume, so a coarse level's
stamps are true of that coarse level and must survive a plain accessor —
``at_substitutive`` is used directly on the scene-authoring path
(``core/group/gsplats_pipeline/lod_dispatch.py``), which copies exactly these
numbers onto every coarse child of a ``kind=lod`` group. A content-changing
rewrite instead recomputes their counts, e(k), and group-consistent w from the
rewritten artifact, while dropping Q until ``annotate-quality --with-quality``
remeasures it; deleting w would license ``annotate-quality``'s leaf-local fallback
to fabricate a group-inconsistent value.

One source-dependent stamp sits beside them but cannot be recomputed: a
``--refine l2|volume`` level records its build step as ``level_stats.refine_stats``
(``mse_seed`` / ``mse_refit``), measured against the source volume. A reduction
removes that nested measurement while keeping the descriptive ``refine`` method;
the rewriter has no source volume from which to remeasure it.

Every scrub here is by KEY, never by dropping a whole nested container, and never
reaches into a dict the caller still owns: ``GSplatData`` is conceptually
immutable, and every call site hands over a result it has just built (nested
``pass_stats`` and ``part_provenance`` lists are REPLACED with scrubbed copies
rather than edited in place, because a shallow ``dict(self.stats)`` shares those
lists with the input).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData


#: Authored reduction-LOD stamp keys shared with
#: :func:`luxar.gsplats.lod.restamp._has_ladder_stamps`. They live in core
#: ``_data`` so this module can mirror ``refresh_reduction_lod_stats``' guard
#: without importing the optional LOD package.
_REDUCTION_LOD_LEVEL_STATS_KEYS = (
    "median_footprint",
    "footprint_dims",
    "quality",
    "reference_energy",
    "n_splats_total",
    "refine_stats",
)
_REDUCTION_LOD_RUNG_STATS_KEYS = (
    "energy_fraction_cum",
    "lod_n_splats",
    "lod_cumulative_n",
)


#: Source-provenance ``stats`` keys that describe the REGION the splats
#: represent. A bbox crop keeps only part of that region, so carrying them over
#: would make ``gsplat info`` quote a compression ratio (and an occupancy) for a
#: volume this artifact no longer represents — inflated by the crop factor. They
#: are dropped instead: staying silent about an unknown source grid is what
#: ``info`` already does for a dataset fitted before these stamps existed.
#: ``source_dtype`` is deliberately EXEMPT — the element type of the source
#: volume is unchanged by a crop, so it remains true. ``voxels_per_splat`` IS
#: region-scoped: it is a ratio over the fitted grid, so after a crop it
#: describes a region the object no longer represents — and its ``fitted_voxels``
#: denominator has just been dropped, leaving it unanchored.
_REGION_SCOPED_STATS_KEYS = (
    "source_shape",
    # Goes with the grid it qualifies: on its own it is a flag saying a shape that
    # is no longer there was declared.
    "source_declared",
    "source_voxels",
    "source_bytes",
    # The size the WHOLE acquisition occupies on disk. Region-scoped for the
    # same reason as the decoded size, and doubly so once the grid it belongs
    # to has gone: `info`'s source block bails out without a `source_shape`, so
    # a survivor would surface as a bare number in "Additional Metadata"
    # claiming the original download for a crop of it.
    "source_stored_bytes",
    "fitted_shape",
    "fitted_voxels",
    "occupancy",
    "voxels_per_splat",
)


def stamp_region_scoped_stats(
    stats: "MutableMapping[str, Any]",
    *,
    source_shape: "Sequence[int]",
    fitted_shape: "Sequence[int]",
    n_splats: int,
    occupancy: float,
    source_itemsize: "int | None" = None,
) -> None:
    """Replace the source-grid record with one measured for this region.

    ``source_declared`` stays absent because this grid was measured rather than
    declared; ``source_stored_bytes`` describes the whole acquisition, not the
    extracted region.
    """
    for key in _REGION_SCOPED_STATS_KEYS:
        stats.pop(key, None)
    source = [int(size) for size in source_shape]
    fitted = [int(size) for size in fitted_shape]
    source_voxels = int(np.prod(source)) if source else 0
    fitted_voxels = int(np.prod(fitted)) if fitted else 0
    stats.update(
        {
            "source_shape": source,
            "source_voxels": source_voxels,
            "fitted_shape": fitted,
            "fitted_voxels": fitted_voxels,
            "occupancy": float(occupancy),
        }
    )
    if source_itemsize is not None:
        stats["source_bytes"] = source_voxels * int(source_itemsize)
    if fitted_voxels and n_splats:
        stats["voxels_per_splat"] = float(fitted_voxels / n_splats)


#: Final best-state populations measured against candidate scales. They are
#: content-scoped, but unlike reconstruction scores they are cheap counts rather
#: than expensive renders, so the fitters' closing-trim snapshot must not carry
#: them onto a different splat set.
_FINAL_SCALE_POPULATION_STATS_KEYS = (
    "splats_near_fit_init_sigma_count",
    "splats_near_fit_init_sigma_fraction",
    "splats_near_relocation_init_sigma_count",
    "splats_near_relocation_init_sigma_fraction",
    "splats_near_sigma_min_count",
    "splats_near_sigma_min_fraction",
)


#: MEASURED reconstruction scores — every number that was obtained by rendering
#: a specific splat set and comparing it to the source volume. They describe the
#: SPLATS, not the source and not the run, so any operation that changes which
#: splats the artifact holds makes them a score for a different object. Kept in
#: lockstep with :func:`luxar.gsplats.metrics.compute_quality_metrics` (which
#: produces the first block wholesale) and with ``_FITTING_INFO_KEYS`` in
#: ``gsplats/io/save_gsplats.py`` (which persists them).
_CONTENT_SCOPED_STATS_KEYS = (
    # The compute_quality_metrics() suite, copied into `stats` verbatim by
    # fitting/results.py, fit_tiled_gsplats.py and the progressive fitter.
    "mse",
    "psnr_db",
    "ssim",
    "rel_l2",
    "max_abs_error",
    "foreground_psnr_db",
    # The threshold and the foreground share are measured on the TARGET, so on
    # their own they would survive a cull — but they exist only to say what
    # `foreground_psnr_db` was taken over (`info` prints them inside its
    # sentence). Orphaned they are a bare Otsu number in "Additional Metadata"
    # qualifying a score that is no longer there, so they go with it.
    "foreground_threshold",
    "foreground_fraction",
    # The optimizer's own residuals at its best iteration (fitting/results.py).
    # Same measurement against the same source volume as `rel_l2` /
    # `max_abs_error` above, just read off the fit loop rather than recomputed —
    # and `final_loss` is what `info` prints one line above `psnr_db`. The
    # iteration COUNTERS beside them (`iterations`, `best_iteration`,
    # `converged`, `early_stopped`, `time_seconds`) stay: they describe the run,
    # which happened, and no rewrite makes them false.
    "final_loss",
    "final_rel_l2",
    "final_max_abs_error",
    # Final best-state populations measured against candidate scales. The
    # candidate/config values and initial-covariance summaries remain run-scoped,
    # but these counts and fractions describe the exact splat set being reduced.
    *_FINAL_SCALE_POPULATION_STATS_KEYS,
    # Progressive fit, per additive sub-LOD: the PSNR of the prefix up to and
    # including this pass, and its increment. Persisted as the leaf's
    # `lod_stats` and read back by `GSplatData.lod_psnrs`, so a cull that fixed
    # only the top level would leave the stale ladder scores one level down.
    "cumulative_psnr_db",
    "delta_psnr_db",
    # The same per-pass ladder rolled up on the top-level dict.
    "pass_psnrs",
    # The error-budget cull's own measurement: the joint L∞ error bound it
    # achieved for ONE specific removal, measured against the target volume, plus
    # the search counters that only mean anything beside it. A later
    # `cull -m cumulative` overwrites the heuristic stamps (`culling_method`,
    # `n_culled`) but would leave these, so the store would publish an error
    # bound for a splat set that no longer exists next to a `culling_method`
    # saying the measuring run never happened. Safe because the error-budget cull
    # re-stamps its own AFTER `filter()` has scrubbed (`culling.py`).
    "error_budget",
    "max_joint_error",
    "phase1_candidates",
    "phase2_iterations",
)

#: The record of the REDUCTION that produced the artifact: which cull ran, how
#: many splats it started from and removed, and how much amplitude mass survived.
#: True of the operation that stamped them and false of anything downstream, so
#: they are content-scoped like the scores above. Nothing published them stale
#: before, only because ``decimate`` published no provenance AT ALL; threading it
#: through (as it now does) is what exposed them — a ``decimate`` of a culled store
#: would otherwise carry the INPUT's ``amplitude_retention: 0.95`` onto a prefix
#: reduction that had just discarded ~75% of the amplitude mass.
#:
#: Safe only because every op that stamps one of these does so AFTER its own
#: ``filter()`` (hence after the scrub): ``_cull_heuristic`` and the
#: error-budget/redundancy path in ``culling.py`` both ``update`` the result of
#: ``self.filter(...)``, as does ``filter_by``'s single-level path (``n_original``).
#: The two multi-substitutive branches (``cull`` and ``filter_by``) build their
#: top-level dict themselves, so they scrub BEFORE stamping — see the comments
#: there. ``filtered`` / ``filter_criteria`` / ``n_removed`` / ``truncate`` are
#: deliberately not here: narrowing this set is a judgement call per key, and
#: those describe a filter rather than quantifying what is left.
_CONTENT_SCOPED_OP_RECORD_KEYS = (
    "culled",
    "culling_method",
    "n_original",
    "n_culled",
    "amplitude_retention",
)

#: The artifact's OWN TOPOLOGY record: what shape the LOD/partition builders gave
#: it. Enumerated from the PRODUCERS, not from a reader's expectations —
#: ``make_substitutive_lod`` (``gsplats/lod/substitutive.py``),
#: ``make_additive_lod`` (``gsplats/lod/additive.py``), the ``recipe`` stamp on the
#: ``gsplat lod`` / demo LOD-policy write paths, and
#: ``_recipe_pipeline_info`` (``gsplats/batch/merge_orchestrator.py``), which is
#: the only site that stamps ``per_part`` / ``n_lods`` / ``breakpoints`` /
#: ``levels`` / ``additive_ladders``.
#:
#: A DENY-LIST rather than "drop the ``pipeline/`` group", because that group is
#: shared: it also carries
#: :data:`~luxar.gsplats.io.save_gsplats.NORMALIZATION_STATS_KEYS` (the input
#: volume's intensity scale, true whatever shape the splats end up in) and
#: ``coarsen_dims``, which is load-bearing — see
#: :data:`_STRUCTURE_SCOPE_EXEMPT_KEYS`.
#:
#: Three names here are generic enough to be worth stating: ``method``,
#: ``refine`` and ``refine_iters``. As TOP-LEVEL ``stats`` keys they are stamped
#: only by the two LOD producers above (the substitutive merge method, and the
#: ``--refine l2|volume`` pass over merged levels), so they are unambiguous
#: today — a fitter's own choices are spelled ``fitter_name`` / ``seed_method`` /
#: ``culling_method``, and the per-level ``refine`` / ``refine_stats`` live in a
#: level's own ``meta["stats"]`` dict, which this rule never touches. Should a
#: future producer stamp a top-level ``method`` meaning something else, the
#: conservative resolution is to RENAME that one rather than narrow this set: a
#: surviving false ``method: auto`` on a flattened leaf is the defect, while a
#: dropped-but-still-true one costs a line ``gsplat info`` prints under
#: "Additional Metadata".
_STRUCTURE_SCOPED_STATS_KEYS = (
    # make_substitutive_lod's out_stats.
    "lod_kind",
    "compression_factor",
    "method",
    "n_substitutive_levels",
    "coverage_inflation",
    "conserve_mass",
    "refine",
    "refine_iters",
    # make_additive_lod's out_stats (the ladder over the summary level).
    "lod_method",
    "lod_n_lods",
    "lod_breakpoints_kind",
    "lod_cutpoints",
    "lod_substitutive_level",
    # The build instruction, stamped by the `lod` CLI and the demo LOD policy.
    "recipe",
    # _recipe_pipeline_info (batch-fit merge): the per-part recipe knobs.
    "per_part",
    "n_lods",
    "breakpoints",
    "levels",
    "additive_ladders",
)

#: Topology-ADJACENT keys that share the ``pipeline/`` group and must survive a
#: structure change. Not a taste call in either case:
#:
#: * ``coarsen_dims`` is READ BACK BY THE WRITER.
#:   :func:`~luxar.gsplats.io.save_gsplats._barrier_from_coarsen_dims` derives
#:   ``write_gsplats_tree``'s chunk-ordering barrier axes from its COMPLEMENT, so
#:   scrubbing it does not merely delete a stamp — it silently changes the
#:   output's chunk layout (a stacked time/channel axis loses its barrier and
#:   falls back to per-leaf auto-detection, smearing every chunk across
#:   timepoints and destroying per-slice read locality). The exemption rests on
#:   that mechanism alone, NOT on the value being true — a rewrite that CHANGES
#:   which axes it coarsened over owes the output a fresh stamp, and scrubbing
#:   the key here would neither produce one nor leave the chunks where they were.
#:   ``decimate`` is the case that had to learn this: its ``merge`` family now
#:   re-stamps the dims it resolved — see
#:   :func:`~luxar.gsplats.lod.substitutive.resolved_merge_coarsen_dims`, called
#:   right after this scrub — while its ``prefix`` family keeps the inherited
#:   value because it blends no axis. That resolution is SHARED with
#:   ``make_substitutive_lod`` and the ``batch-fit merge`` per-part record, so
#:   the three paths that WRITE this key spell coarsen-everything the same
#:   explicit way rather than as a ``null`` the writer reads as no provenance at
#:   all. (``lod --recipe adaptive`` / ``overview`` and ``fit --recipe levels``
#:   coarsen too but publish no stamp at all — still on #1600.)
#:
#:   The exemption has a CONSEQUENCE worth stating, because #1600 made the
#:   inherited value load-bearing where it used to be an inert ``null``: a
#:   ``levels`` store that coarsened everything now carries ``[0, …, d-1]``, and
#:   every structure-preserving rewrite of it — ``cull`` / ``filter`` / ``slice``
#:   / ``transform`` / ``reencode``, plus ``flatten`` / ``partition`` /
#:   ``additive`` / a rebuilt ``lod`` — inherits that list and writes
#:   ``slice_dims: []`` where it used to write the auto-detected barrier. The
#:   direction is the safe one (:func:`~luxar.io._ordering.compound
#:   .detect_barrier_dims` documents the asymmetry: a MISSING barrier costs
#:   over-fetch, a false one gives a spatial axis tight chunk bounds and can drop
#:   splats), and the value stays TRUE of every output EXCEPT the finest level —
#:   none of those rewrites coarsens anything, so an axis blended upstream is
#:   still blended, but the finest level is the input UNREDUCED and that axis was
#:   never blended in it. So ``flatten`` of a coarsen-everything ``levels`` store
#:   hands back the original splats still claiming ``[0, …, d-1]``: a false claim
#:   as well as a lost barrier, costing the per-slice locality one there would
#:   have been legitimate. One layout per ladder, chosen by the producer, instead
#:   of a heuristic answering each level on its own.
#: * The :data:`~luxar.gsplats.io.save_gsplats.NORMALIZATION_STATS_KEYS` block
#:   (``floor`` / ``image_min`` / ``image_max`` / ``intensity_range``) describes
#:   the INPUT VOLUME's intensity scale. Regrouping splats cannot change what
#:   pedestal was subtracted before fitting. Named here for the record; they are
#:   exempt by simply not appearing above, and the completeness test asserts the
#:   two sets stay disjoint.
#:
#: Nothing in production READS this tuple — the writer never consults it. It is
#: the classification anchor for the completeness test, which requires every key
#: a builder stamps to be either scrubbed or listed here with its reason.
_STRUCTURE_SCOPE_EXEMPT_KEYS = ("coarsen_dims",)

#: ``stats`` keys whose value is a LIST of nested stats dicts. The progressive
#: fitter stores one dict per pass here, mixing measured scores
#: (``cumulative_psnr_db``) with descriptive counts (``pass_index``,
#: ``seeds_requested``, ``splats_after_culling``) that stay true. Dropping the
#: whole list would throw the counts away with the scores, so the nested dicts
#: are scrubbed key-by-key instead.
_NESTED_STATS_LIST_KEYS = ("pass_stats",)


def _drop_part_provenance_fitting_keys(
    stats: "MutableMapping[str, Any]", dropped: "Sequence[str]"
) -> None:
    provenance = stats.get("part_provenance")
    if not isinstance(provenance, list):
        return
    dropped_set = set(dropped)
    scrubbed: list[Any] = []
    for entry in provenance:
        if not isinstance(entry, dict):
            scrubbed.append(entry)
            continue
        record = dict(entry)
        fitting = record.get("fitting")
        if isinstance(fitting, dict):
            scrubbed_fitting = {
                key: value for key, value in fitting.items() if key not in dropped_set
            }
            _drop_part_provenance_fitting_keys(scrubbed_fitting, dropped)
            record["fitting"] = scrubbed_fitting
        scrubbed.append(record)
    stats["part_provenance"] = scrubbed


def drop_content_scoped_stats(stats: "MutableMapping[str, Any]") -> None:
    """Remove the measured reconstruction scores from ONE stats dict, in place.

    The dict-level primitive, so a caller holding raw ``stats`` (the CLI's
    node-tree path, which never builds a ``GSplatData``) scrubs exactly what the
    ``GSplatData`` paths do. Use :func:`_stats_after_content_change` when you have
    a dataset — it also reaches each additive sub-LOD's own dict (the leaf's
    on-disk ``lod_stats``), where a progressive fit's per-pass scores live.

    Mutates ``stats`` (which the caller owns) but nothing REACHABLE from it:
    nested lists are replaced with scrubbed copies rather than edited
    entry-by-entry. Every ``GSplatData`` call site reaches this through a shallow
    ``dict(source.stats)``, which shares those lists with the input — so editing
    the entries would delete the input's own scores and break the
    "conceptually immutable, operations return new instances" contract (#1600
    review). The top-level dict is a copy, the nested list must be made one.
    """
    dropped = (*_CONTENT_SCOPED_STATS_KEYS, *_CONTENT_SCOPED_OP_RECORD_KEYS)
    for key in dropped:
        stats.pop(key, None)
    _drop_part_provenance_fitting_keys(stats, dropped)
    for key in _NESTED_STATS_LIST_KEYS:
        nested = stats.get(key)
        if isinstance(nested, list):
            stats[key] = [
                (
                    {k: v for k, v in entry.items() if k not in dropped}
                    if isinstance(entry, dict)
                    else entry
                )
                for entry in nested
            ]


def stats_after_structure_change(stats: "Mapping[str, Any]") -> "Dict[str, Any]":
    """A COPY of ``stats`` with the artifact's inherited TOPOLOGY record removed.

    For a rewrite that KNOWS it produced a different kind of thing: ``gsplat
    flatten`` and :func:`~luxar.gsplats.lod.decimate.decimate` emit one flat leaf,
    ``gsplat partition`` emits a ``kind=partition`` of bare leaves, and ``gsplat
    lod`` emits whatever ``--recipe`` says from any matrix-shaped input (every
    recipe starts by flattening, so none of them preserves it). All four thread
    the input's stats through to the output's ``fitting/`` / ``provenance/`` /
    ``pipeline/`` groups (correct — the fit provenance is still true), which is how
    a flattened pyramid came to advertise ``lod_kind: substitutive`` with four
    levels and a four-rung ladder it does not have (#1600). ``lod`` applies this to
    the LOADED INPUT rather than to a result, because its two write paths both
    descend from that one dict and each builder then stamps its own true record
    over the cleaned copy.

    Returns a copy rather than mutating: the call sites hand over the dict they
    loaded off disk, or the dataset's own ``stats``, and the rest of this
    module's contract is that a rewrite never edits what the caller still owns.
    Scrubs BY KEY (:data:`_STRUCTURE_SCOPED_STATS_KEYS`), never by dropping the
    ``pipeline/`` group, which also holds the normalization block and the
    load-bearing ``coarsen_dims`` — see :data:`_STRUCTURE_SCOPE_EXEMPT_KEYS`.

    Deliberately NOT applied inside ``flattened()`` / ``concatenate`` /
    ``to_spatial_partition``: those are general-purpose domain methods with other
    callers (a recipe builds an intermediate flat view of a level it is ABOUT to
    wrap in a lod group again, and scrubbing there would erase a record that is
    still true of the result). Their RETURN VALUE does not settle what kind of
    thing will be published, so the knowledge belongs to whoever publishes.
    ``decimate()`` is the opposite case and does apply it itself, at the domain
    layer: its contract is one flat leaf whatever it was handed, so no caller can
    want the topology record kept — leaving the scrub to the CLI left the public
    ``luxar.gsplats.lod.decimate`` API reproducing the defect.
    """
    scrubbed = dict(stats)
    for key in _STRUCTURE_SCOPED_STATS_KEYS:
        scrubbed.pop(key, None)
    return scrubbed


def content_scoped_stats(stats: "MutableMapping[str, Any]") -> "Dict[str, Any]":
    """Snapshot the measured scores so a PRODUCER can re-attach them, deep.

    For the fitters' own last step, which is a high-retention cumulative cull
    (``cull_retention``, default 0.95 flat/tiled and 0.98 progressive) applied
    AFTER ``finalize_results`` has scored the reconstruction. That trim is part of
    producing the artifact rather than a later rewrite of a published one, and the
    score cannot be retaken without a second full render of the volume — so the
    fitters snapshot their measurement across it and put it back. Scrubbing there
    instead would leave EVERY default fit with no ``psnr_db`` at all, which is a
    worse answer than one taken before a trim that drops 5% of total amplitude
    (the fit's own summary quotes exactly this number, so it would also start
    disagreeing with the store it wrote).

    Post-fit ``cull`` / ``filter`` / ``slice`` / ``decimate`` on a stored artifact
    get no such exemption — carrying the score across an arbitrary retention the
    user picked is #1600 itself.

    Covers the measured SCORES only, not final scale populations or
    :data:`_CONTENT_SCOPED_OP_RECORD_KEYS`: restoring populations would attach
    pre-trim counts to a different splat set, and the op that scrubbed the record
    re-stamps its own right after. Restoring an older cull's ``n_original`` /
    ``amplitude_retention`` over it would publish the wrong reduction (a tiled
    fit culls each tile, then culls the merge).

    The nested per-pass lists are deep-copied so the snapshot is independent of
    the dataset it was taken from — a later edit of the source (or of the trimmed
    result, which shares the list until it is scrubbed) cannot reach back into it.
    """
    import copy

    snapshot = {
        key: stats[key]
        for key in _CONTENT_SCOPED_STATS_KEYS
        if key in stats and key not in _FINAL_SCALE_POPULATION_STATS_KEYS
    }
    for key in _NESTED_STATS_LIST_KEYS:
        if key in stats:
            snapshot[key] = copy.deepcopy(stats[key])
    return snapshot


def _measured_stats_dicts(data: "GSplatData") -> "List[MutableMapping[str, Any]]":
    """Every dict a measured stamp of ``data`` lives in, in a fixed order.

    Two scopes, ordered so a snapshot and a restore line up: the top-level dict,
    then every additive sub-LOD's (persisted as the leaf's ``lod_stats``). The
    sub-LOD objects are shared with the node, so mutating their ``stats`` reaches
    what gets written.

    Deliberately NOT the leaves' ``meta["stats"]`` (the on-disk ``level_stats``),
    which this rule does not claim — see the module docstring for the Q·e stamps
    and for the one OTHER measured thing that lives there: a ``--refine
    l2|volume`` level's nested ``refine_stats`` (``mse_seed`` / ``mse_refit``,
    taken against the source volume). It goes with the Q·e stamps rather than
    with the scores above, for the same reason: it is the record of the build
    step that produced THAT level. A reduction drops it because remeasurement
    requires the unavailable source volume; rebuilding with ``--refine`` restores it.
    """
    return [data.stats, *(lod.stats for lod in _all_sublods(data))]


def measured_stats_snapshot(data: "GSplatData") -> "List[Dict[str, Any]]":
    """:func:`content_scoped_stats` over every scope of a dataset.

    The per-sub-LOD walk is what makes the snapshot the exact inverse of the
    scrub: :func:`scrub_measured_stats` reaches those dicts, so a top-level-only
    snapshot would restore less than was taken and leave a laddered dataset
    half-stamped. (It is not about what a progressive fit PUBLISHES: that fitter
    ends with ``final_result.flattened()``, which collapses the ladder, so only
    the rolled-up ``pass_psnrs`` / ``pass_stats`` on the top-level dict reach
    disk.)
    """
    return [content_scoped_stats(stats) for stats in _measured_stats_dicts(data)]


def restore_measured_stats(
    data: "GSplatData", snapshot: "List[Dict[str, Any]]"
) -> "GSplatData":
    """Put a :func:`measured_stats_snapshot` back, positionally.

    Per-sub-LOD scores are restored only when the ladder shape is unchanged.  If
    a mask-based op pruned an empty rung, positional pairing is no longer sound;
    in that case only the top-level measurement is restored.
    """
    targets = _measured_stats_dicts(data)
    if len(targets) != len(snapshot):
        data.stats.update(snapshot[0])
        return data
    for target, saved in zip(targets, snapshot):
        target.update(saved)
    return data


def scrub_measured_stats(result: "GSplatData") -> None:
    """Drop every MEASURED stamp of ``result`` — top level and every sub-LOD.

    The dataset-level counterpart of :func:`drop_content_scoped_stats`, reaching
    the one scope a rewrite otherwise leaves behind: a progressive fit stamps each
    pass's scores into that pass's sub-LOD, and the writer persists those as the
    leaf's ``lod_stats`` — so a fix applied only to the top-level dict would ship
    the stale ladder PSNRs one level down.
    """
    for stats in _measured_stats_dicts(result):
        drop_content_scoped_stats(stats)


def scrub_region_scoped_stats(result: "GSplatData") -> None:
    """Drop every source-grid stamp of ``result`` — top level and sub-LODs."""
    for stats in _measured_stats_dicts(result):
        for key in _REGION_SCOPED_STATS_KEYS:
            stats.pop(key, None)


def _stats_after_content_change(
    result: "GSplatData", *, changed: bool, source: _GSplatDataOps
) -> "GSplatData":
    """Drop the measured scores from ``result`` when the splat set ``changed``.

    ``changed`` is the caller's honest answer to "does this artifact hold
    different splats than the one the metrics were measured on" — a count that
    moved (filter/cull/slice), amplitudes that were rewritten (intensity ops:
    PSNR is not scale-invariant), or splats replaced by merged representatives
    (a decimate that hit the requested count exactly). A rewrite that changes
    nothing — an all-passing threshold, a re-encode, a re-ladder — keeps the
    metrics, which are still true of it.

    Mutates ``result``'s own stats in place (never the caller's — see
    :func:`drop_content_scoped_stats`), for the same reasons spelled out on
    :func:`_stats_after_filter`.

    LOD restamping belongs to the optional gsplats extra; the shared helper
    keeps that deferred dependency boundary in one place.
    """
    if not changed:
        return result
    scrub_measured_stats(result)

    return _refresh_reduction_lod_stats_if_needed(result, source)


def _needs_reduction_lod_restamp(
    result: _GSplatDataOps, source: _GSplatDataOps
) -> bool:
    """Whether a rewrite has authored LOD stamps to refresh.

    This mirrors ``refresh_reduction_lod_stats``' first guard exactly, across
    every substitutive level, without importing the optional LOD package.
    """

    source_levels = source.substitutive_levels
    result_levels = result.substitutive_levels
    return bool(
        source_levels
        and result_levels
        and any(
            any(key in level.stats for key in _REDUCTION_LOD_LEVEL_STATS_KEYS)
            or any(
                any(key in lod.stats for key in _REDUCTION_LOD_RUNG_STATS_KEYS)
                for lod in level.additive_sublods
            )
            for level in source_levels
        )
    )


def _refresh_reduction_lod_stats_if_needed(
    result: "GSplatData", source: _GSplatDataOps
) -> "GSplatData":
    """Refresh authored LOD stamps without loading the extra for unstamped data."""
    if not _needs_reduction_lod_restamp(result, source):
        return result

    from luxar.gsplats.lod.restamp import refresh_reduction_lod_stats

    return refresh_reduction_lod_stats(result, source)


def _substitutive_counts_changed(
    before: "_GSplatDataOps", after: "_GSplatDataOps"
) -> bool:
    before_levels = before.substitutive_levels
    after_levels = after.substitutive_levels
    return len(before_levels) != len(after_levels) or any(
        old.n_splats_total != new.n_splats_total
        for old, new in zip(before_levels, after_levels)
    )


def _is_crop(bbox: object, n_before: int, n_after: int) -> bool:
    """Whether a filter actually RESTRICTED the region the splats represent.

    A bbox that excluded nothing (``slice_by([slice(None)] * ndim)``, or a
    ``--bbox`` enclosing the whole volume — the natural spelling when one axis of
    a scripted sweep is unbounded) leaves an artifact representing exactly the
    same content, so its source stamp is still true and must survive. Only a
    bbox that removed splats invalidates it.
    """
    return bbox is not None and n_after < n_before


def _stats_after_filter(result: "GSplatData", *, cropped: bool) -> "GSplatData":
    """Drop the region-scoped source stamps from ``result`` when ``cropped``.

    A non-spatial filter (amplitude/scale/mass/... thresholds) does NOT change
    which region the splats represent, so it keeps the whole stamp; only a
    bbox/slice restriction that actually excluded splats invalidates it (see
    :func:`_is_crop`).

    Mutates the stats dicts IN PLACE rather than copying: every call site hands
    over a result it has just built, never a caller's object (``filter()`` and the
    ``_map_*`` rebuilds all copy ``stats``).

    Cleans the TOP-LEVEL stats and every additive sub-LOD's, because the writer
    persists a sub-LOD's dict as the leaf's ``lod_stats`` — a crop that fixed only
    the top level would leave the uncropped stamp on disk one level down. On a
    single leaf those are the SAME dict (``GSplatData.__init__`` aliases it) and
    the extra pass is a no-op; a ladder needs it, and a progressive fit builds one
    whose every sub-LOD carries that pass's full fit stats.
    """
    if not cropped:
        return result
    for stats in (result.stats, *(lod.stats for lod in _all_sublods(result))):
        for key in _REGION_SCOPED_STATS_KEYS:
            stats.pop(key, None)
        _drop_part_provenance_fitting_keys(stats, _REGION_SCOPED_STATS_KEYS)
    return result


def _all_sublods(result: "GSplatData") -> "List[AdditiveSubLOD]":
    """Every additive sub-LOD of every substitutive level.

    ``substitutive_levels`` is a view rebuilt from the node, but it shares the
    ``AdditiveSubLOD`` objects themselves, so mutating their ``stats`` reaches
    what gets written.
    """
    return [
        lod for level in result.substitutive_levels for lod in level.additive_sublods
    ]


class FilteringMixin(_GSplatDataOps):
    """``filter`` / ``filter_by`` / ``slice_by`` and the threshold resolver."""

    def filter(self, mask: np.ndarray) -> "GSplatData":
        """Return new GSplatData with only the splats where mask is True.

        Args:
            mask: Boolean array of shape (N,).

        Returns:
            New GSplatData with filtered arrays.

        Removing any splat drops the inherited measured reconstruction scores and
        the inherited reduction record (see :data:`_CONTENT_SCOPED_STATS_KEYS` and
        :data:`_CONTENT_SCOPED_OP_RECORD_KEYS`): this is the single chokepoint
        every mask-based rewrite goes through — ``filter_by``, ``slice_by`` and
        every ``cull`` strategy — so scrubbing here covers all of them, and the
        record each of those stamps AFTERWARDS (``culled``, ``n_original``,
        ``filter_criteria``, ...) describes THIS operation and lands on a clean
        dict.

        Example:
            >>> filtered = data.filter(data.volumes() < 100)
            >>> filtered = data.filter((data.amplitudes > 0.1) & (data.eccentricities() < 5))
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_splats,):
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match splat count ({self.n_splats},)"
            )

        # A raw boolean mask is sized to the default substitutive level, so it
        # cannot be applied per-level — coarser substitutive levels are dropped.
        # Warn loudly (never silent) and point at the criteria-based ops, which
        # DO preserve the full pyramid (see filter_by / cull).
        if self.n_substitutive > 1:
            warnings.warn(
                "filter(mask) keeps only the default substitutive level "
                f"(n_substitutive={self.n_substitutive}); coarser levels are "
                "dropped. Use filter_by(...) / cull(...) to filter every "
                "substitutive level and preserve the pyramid.",
                UserWarning,
                stacklevel=2,
            )

        # An all-passing mask rewrites nothing, so the metrics still describe the
        # result and must survive (the counterpart of _is_crop's "a bbox that
        # excluded nothing keeps its region stamp").
        removed_any = bool(int(np.count_nonzero(mask)) < self.n_splats)

        def _filter_lod(lod: AdditiveSubLOD, offset: int, n: int) -> AdditiveSubLOD:
            lod_mask = mask[offset : offset + n]
            return AdditiveSubLOD(
                centers=lod.centers[lod_mask],
                amplitudes=lod.amplitudes[lod_mask],
                cholesky_factors=lod.cholesky_factors[lod_mask],
                colors=lod.colors[lod_mask] if lod.colors is not None else None,
                label_ids=(
                    lod.label_ids[lod_mask] if lod.label_ids is not None else None
                ),
                label_vocabulary=lod.label_vocabulary,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )

        return _stats_after_content_change(
            self._map_additive(_filter_lod), changed=removed_any, source=self
        )

    @staticmethod
    def _resolve_threshold(
        val: float | None,
        normalized: bool,
        dataset_values: np.ndarray,
        percentile: bool = False,
    ) -> float | None:
        """Resolve a threshold to an absolute value.

        - ``percentile``: ``val`` in [0,100] → the ``val``-th percentile of
          ``dataset_values`` (robust on heavy-tailed attributes; preferred over
          ``normalized``).
        - ``normalized``: ``val`` in [0,1] → linear map onto [min, max].
        - otherwise: ``val`` is already absolute.
        """
        if val is None:
            return None
        if percentile:
            return float(np.percentile(dataset_values, val))
        if normalized:
            dmin, dmax = float(dataset_values.min()), float(dataset_values.max())
            return dmin + val * (dmax - dmin)
        return val

    def filter_by(
        self,
        *,
        bbox: list[tuple[float, float]] | None = None,
        volume_min: float | None = None,
        volume_max: float | None = None,
        volume_normalized: bool = False,
        volume_percentile: bool = False,
        scale_min: float | None = None,
        scale_max: float | None = None,
        scale_normalized: bool = False,
        scale_percentile: bool = False,
        amplitude_min: float | None = None,
        amplitude_max: float | None = None,
        amplitude_normalized: bool = False,
        amplitude_percentile: bool = False,
        eccentricity_min: float | None = None,
        eccentricity_max: float | None = None,
        eccentricity_percentile: bool = False,
        mass_min: float | None = None,
        mass_max: float | None = None,
        mass_normalized: bool = False,
        mass_percentile: bool = False,
        sigma_axis: int | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        sigma_percentile: bool = False,
        isolation_max: float | None = None,
        isolation_percentile: bool = False,
        min_neighbors: int | None = None,
        neighbor_radius: float | None = None,
        spatial_dims: Sequence[int] | None = None,
        truncate: float | None = None,
    ) -> "GSplatData":
        """Filter splats by multiple criteria (AND logic).

        All criteria are optional. Only specified criteria are applied.
        Multiple criteria combine with AND — a splat must satisfy all
        active criteria to be kept.

        Args:
            bbox: Bounding box per dimension as [(min0, max0), (min1, max1), ...].
                  Length must equal ndim. Filters by center position.
            volume_min: Minimum volume (characteristic length * truncate).
            volume_max: Maximum volume.
            volume_normalized: If True, interpret volume thresholds as 0-1
                mapped to the dataset's [min, max] volume range.
            amplitude_min: Minimum amplitude.
            amplitude_max: Maximum amplitude.
            amplitude_normalized: If True, interpret amplitude thresholds as 0-1
                mapped to the dataset's [min, max] amplitude range.
            eccentricity_min: Minimum eccentricity (1.0 = isotropic).
            eccentricity_max: Maximum eccentricity.
            mass_min: Minimum mass (amplitude * volume).
            mass_max: Maximum mass.
            mass_normalized: If True, interpret mass thresholds as 0-1
                mapped to the dataset's [min, max] mass range.
            sigma_axis: Axis index for per-axis sigma filtering.
            sigma_min: Minimum marginal sigma on sigma_axis.
            sigma_max: Maximum marginal sigma on sigma_axis.
            scale_min/scale_max: Characteristic size (geometric-mean marginal
                sigma over the spatial/``spatial_dims`` axes; see ``scale()``).
                The recommended "remove large diffuse background" knob — cleaner
                than ``volume`` on nD timelapses.
            isolation_max: Remove splats whose nearest-neighbour distance (over
                the spatial axes, grouped by the non-spatial axes) EXCEEDS this
                — i.e. spatially isolated noise splats.
            min_neighbors / neighbor_radius: Remove splats with fewer than
                ``min_neighbors`` other splats within ``neighbor_radius``.
            spatial_dims: Override the axes used for scale / eccentricity /
                isolation (default: auto-detected non-degenerate axes).
            *_percentile: For volume/scale/amplitude/mass/sigma/eccentricity/
                isolation — interpret the corresponding min/max as a percentile
                in [0,100] of that attribute (robust on heavy-tailed data).
            truncate: Sigma truncation factor for volume computation.
                Defaults to ``self.truncation_radius``.

        Returns:
            New GSplatData with only splats that pass all criteria.

        Raises:
            ValueError: If bbox length doesn't match ndim, sigma_axis is out
                of range, or sigma_min/sigma_max given without sigma_axis.

        Examples:
            >>> # Keep splats with amplitude >= 0.1 and eccentricity <= 5
            >>> filtered = data.filter_by(amplitude_min=0.1, eccentricity_max=5.0)
            >>>
            >>> # Spatial crop to a bounding box (3D)
            >>> filtered = data.filter_by(bbox=[(0, 50), (0, 50), (0, 50)])
            >>>
            >>> # Remove top 10% largest volumes (normalized)
            >>> filtered = data.filter_by(volume_max=0.9, volume_normalized=True)
        """
        if truncate is None:
            truncate = self.truncation_radius

        # Short-circuit for empty data
        if self.n_splats == 0:
            result = self.filter(np.ones(0, dtype=bool))
            result.stats.update(
                {
                    "filtered": True,
                    "filter_criteria": {},
                    "n_original": 0,
                    "n_removed": 0,
                    "truncate": truncate,
                }
            )
            # Nothing to remove from an empty dataset, so no bbox can be a crop.
            return _stats_after_filter(result, cropped=_is_crop(bbox, 0, 0))

        # Validate sigma_axis usage
        if (sigma_min is not None or sigma_max is not None) and sigma_axis is None:
            raise ValueError("sigma_min/sigma_max require sigma_axis to be specified")
        if sigma_axis is not None and not (0 <= sigma_axis < self.ndim):
            raise ValueError(
                f"sigma_axis={sigma_axis} out of range for {self.ndim}D data"
            )
        # Local-density filter needs both knobs (mirrors the sigma_axis rule).
        if (min_neighbors is None) != (neighbor_radius is None):
            raise ValueError(
                "min_neighbors and neighbor_radius must be specified together"
            )

        # Multi-substitutive: apply the SAME criteria to every substitutive
        # level and rebuild the pyramid (decision 6) rather than silently
        # collapsing to the default level. Each level is filtered through the
        # single-substitutive path below (a per-level view); thresholds with
        # *_normalized resolve per-level (each level to its own range).
        if self.n_substitutive > 1:
            out = self._map_substitutive(
                lambda lvl: lvl.filter_by(
                    bbox=bbox,
                    volume_min=volume_min,
                    volume_max=volume_max,
                    volume_normalized=volume_normalized,
                    volume_percentile=volume_percentile,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    scale_normalized=scale_normalized,
                    scale_percentile=scale_percentile,
                    amplitude_min=amplitude_min,
                    amplitude_max=amplitude_max,
                    amplitude_normalized=amplitude_normalized,
                    amplitude_percentile=amplitude_percentile,
                    eccentricity_min=eccentricity_min,
                    eccentricity_max=eccentricity_max,
                    eccentricity_percentile=eccentricity_percentile,
                    mass_min=mass_min,
                    mass_max=mass_max,
                    mass_normalized=mass_normalized,
                    mass_percentile=mass_percentile,
                    sigma_axis=sigma_axis,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    sigma_percentile=sigma_percentile,
                    isolation_max=isolation_max,
                    isolation_percentile=isolation_percentile,
                    min_neighbors=min_neighbors,
                    neighbor_radius=neighbor_radius,
                    spatial_dims=spatial_dims,
                    truncate=truncate,
                )
            )
            # The measured scores go BEFORE this filter's own record is stamped:
            # _map_substitutive rebuilds the top level from `dict(self.stats)`, so
            # the per-level scrub inside filter() reaches the sub-LOD dicts but not
            # this one — and the scrub also takes `n_original`, which is exactly
            # the key stamped just below (the single-level path gets the same order
            # for free, scrubbing inside `self.filter(mask)`).
            out = _stats_after_content_change(
                out, changed=_substitutive_counts_changed(self, out), source=self
            )
            out.stats.update(
                {
                    "filtered": True,
                    "n_original": self.n_splats,
                    "n_removed": self.n_splats - out.n_splats,
                    "truncate": truncate,
                }
            )
            # A crop restricts WHICH REGION the splats represent (the per-level
            # recursion above cannot fix the rebuilt top-level stats).
            return _stats_after_filter(
                out, cropped=_is_crop(bbox, self.n_splats, out.n_splats)
            )

        mask = np.ones(self.n_splats, dtype=bool)
        criteria: dict[str, object] = {}

        # -- Bounding box (center position)
        if bbox is not None:
            if len(bbox) != self.ndim:
                raise ValueError(
                    f"bbox has {len(bbox)} dimensions, expected {self.ndim}"
                )
            criteria["bbox"] = bbox
            for i, (lo, hi) in enumerate(bbox):
                mask &= (self.centers[:, i] >= lo) & (self.centers[:, i] <= hi)

        # -- Volume (characteristic length * truncate)
        if volume_min is not None or volume_max is not None:
            vols = self.volumes() * truncate
            vmin = self._resolve_threshold(
                volume_min, volume_normalized, vols, volume_percentile
            )
            vmax = self._resolve_threshold(
                volume_max, volume_normalized, vols, volume_percentile
            )
            if vmin is not None:
                mask &= vols >= vmin
                criteria["volume_min"] = vmin
            if vmax is not None:
                mask &= vols <= vmax
                criteria["volume_max"] = vmax
            if volume_normalized:
                criteria["volume_normalized"] = True

        # -- Scale (geometric-mean marginal sigma over the spatial axes)
        if scale_min is not None or scale_max is not None:
            scl = self.scale(axes=spatial_dims)
            smin = self._resolve_threshold(
                scale_min, scale_normalized, scl, scale_percentile
            )
            smax = self._resolve_threshold(
                scale_max, scale_normalized, scl, scale_percentile
            )
            if smin is not None:
                mask &= scl >= smin
                criteria["scale_min"] = smin
            if smax is not None:
                mask &= scl <= smax
                criteria["scale_max"] = smax

        # -- Amplitude
        if amplitude_min is not None or amplitude_max is not None:
            amps = self.amplitudes
            amin = self._resolve_threshold(
                amplitude_min, amplitude_normalized, amps, amplitude_percentile
            )
            amax = self._resolve_threshold(
                amplitude_max, amplitude_normalized, amps, amplitude_percentile
            )
            if amin is not None:
                mask &= amps >= amin
                criteria["amplitude_min"] = amin
            if amax is not None:
                mask &= amps <= amax
                criteria["amplitude_max"] = amax
            if amplitude_normalized:
                criteria["amplitude_normalized"] = True

        # -- Eccentricity (spatial isotropy; auto-ignores degenerate axes)
        if eccentricity_min is not None or eccentricity_max is not None:
            ecc = self.eccentricities(axes=spatial_dims)
            emin = self._resolve_threshold(
                eccentricity_min, False, ecc, eccentricity_percentile
            )
            emax = self._resolve_threshold(
                eccentricity_max, False, ecc, eccentricity_percentile
            )
            if emin is not None:
                mask &= ecc >= emin
                criteria["eccentricity_min"] = emin
            if emax is not None:
                mask &= ecc <= emax
                criteria["eccentricity_max"] = emax

        # -- Mass (amplitude * volume)
        if mass_min is not None or mass_max is not None:
            m = self.masses()
            mmin = self._resolve_threshold(
                mass_min, mass_normalized, m, mass_percentile
            )
            mmax = self._resolve_threshold(
                mass_max, mass_normalized, m, mass_percentile
            )
            if mmin is not None:
                mask &= m >= mmin
                criteria["mass_min"] = mmin
            if mmax is not None:
                mask &= m <= mmax
                criteria["mass_max"] = mmax
            if mass_normalized:
                criteria["mass_normalized"] = True

        # -- Per-axis sigma
        if sigma_axis is not None and (sigma_min is not None or sigma_max is not None):
            sigmas = self.marginal_sigmas()[:, sigma_axis]
            criteria["sigma_axis"] = sigma_axis
            smn = self._resolve_threshold(sigma_min, False, sigmas, sigma_percentile)
            smx = self._resolve_threshold(sigma_max, False, sigmas, sigma_percentile)
            if smn is not None:
                mask &= sigmas >= smn
                criteria["sigma_min"] = smn
            if smx is not None:
                mask &= sigmas <= smx
                criteria["sigma_max"] = smx

        # -- Isolation (remove spatially-isolated noise splats)
        if isolation_max is not None:
            nn = self.nearest_neighbor_distances(spatial_axes=spatial_dims)
            finite = nn[np.isfinite(nn)]
            if isolation_percentile and finite.size == 0:
                # Every splat is an isolated singleton (no finite NN distance);
                # a percentile is undefined → drop them all.
                mask &= False
                criteria["isolation_max"] = "all-isolated"
            else:
                imax = self._resolve_threshold(
                    isolation_max, False, finite, isolation_percentile
                )
                if imax is not None:
                    # +inf (no neighbour) always exceeds the threshold → removed.
                    mask &= nn <= imax
                    criteria["isolation_max"] = imax

        # -- Local density (keep only well-supported splats)
        if min_neighbors is not None and neighbor_radius is not None:
            counts = self.neighbor_counts(neighbor_radius, spatial_axes=spatial_dims)
            mask &= counts >= int(min_neighbors)
            criteria["min_neighbors"] = int(min_neighbors)
            criteria["neighbor_radius"] = float(neighbor_radius)

        # Apply mask
        result = self.filter(mask)
        result.stats.update(
            {
                "filtered": True,
                "filter_criteria": criteria,
                "n_original": self.n_splats,
                "n_removed": self.n_splats - result.n_splats,
                "truncate": truncate,
            }
        )
        # A bbox crop that actually excluded splats invalidates the source-region
        # stamps inherited from the fit (see _stats_after_filter / _is_crop); a
        # non-spatial threshold, or a bbox that removed nothing, keeps them.
        return _stats_after_filter(
            result, cropped=_is_crop(bbox, self.n_splats, result.n_splats)
        )

    def slice_by(self, slices: list[slice]) -> "GSplatData":
        """Slice splats by coordinate ranges per dimension (numpy-style).

        Each slice specifies a [start, stop] range for that dimension's center
        coordinate. ``None`` in start/stop means unbounded.

        Args:
            slices: One slice per dimension. ``slice(lo, hi)`` keeps splats
                with center in [lo, hi]. ``slice(None, None)`` keeps all.

        Returns:
            New GSplatData with only splats inside all ranges.

        Raises:
            ValueError: If number of slices doesn't match ndim.

        Examples:
            >>> # Keep x in [0,50], all y, z in [10,90]
            >>> sliced = data.slice_by([slice(0, 50), slice(None, None), slice(10, 90)])
            >>>
            >>> # Open-ended: x >= 50
            >>> sliced = data.slice_by([slice(50, None), slice(None, None), slice(None, None)])
        """
        if len(slices) != self.ndim:
            raise ValueError(f"Expected {self.ndim} slices, got {len(slices)}")
        bbox = []
        for s in slices:
            lo = float(s.start) if s.start is not None else float("-inf")
            hi = float(s.stop) if s.stop is not None else float("inf")
            bbox.append((lo, hi))
        return self.filter_by(bbox=bbox)
