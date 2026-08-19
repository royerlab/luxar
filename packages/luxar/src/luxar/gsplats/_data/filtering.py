"""Attribute/geometry filtering mixin for ``GSplatData``, and the ``stats`` rules.

Inherited-``stats`` hygiene comes in TWO categories, because a rewritten artifact
can invalidate a stamp along either of two independent axes and no single
predicate covers both:

* :data:`_REGION_SCOPED_STATS_KEYS` — what the splats REPRESENT. Invalidated by a
  spatial restriction (a bbox/slice that excluded splats), and by nothing else:
  an amplitude threshold leaves the represented region exactly as it was.
  Predicate: :func:`_is_crop`.
* :data:`_CONTENT_SCOPED_STATS_KEYS` — MEASURED scores of the splat set (against
  the source volume, or against its own finest content). Invalidated whenever the
  splat set changes, spatially or not: an amplitude-threshold cull leaves the
  region untouched while changing the reconstruction completely, and a
  merge-family reduction can hit the requested count exactly while replacing
  every splat with a representative. Predicate: "did the content change" —
  :func:`_stats_after_content_change`, which also reaches the per-sub-LOD
  (:data:`_CONTENT_SCOPED_SUBLOD_KEYS`) and per-level
  (:data:`_CONTENT_SCOPED_LEVEL_KEYS`) ladder stamps.

Reusing the region predicate for the metrics is what #1600 was: a ``cull -r 0.5``
that halved the splat count published the pre-cull PSNR as its own, and ``gsplat
info`` reads ``psnr_db`` as THE dataset's reconstruction quality. Neither category
subsumes the other — a whole-volume bbox that removed nothing keeps both, a
non-spatial cull keeps the region stamp and loses the metrics, an actual crop
loses both.

Every scrub here is by KEY, never by dropping a whole nested container, and never
reaches into a dict the caller still owns: ``GSplatData`` is conceptually
immutable, and every call site hands over a result it has just built (a nested
``pass_stats`` list is REPLACED with scrubbed copies rather than edited in place,
because a shallow ``dict(self.stats)`` shares that list with the input).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Sequence

import numpy as np

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData


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

#: MEASURED stamps that live one level down, in each additive sub-LOD's own
#: ``stats`` (the leaf's on-disk ``lod_stats``). ``energy_fraction_cum`` is the
#: cumulative self-energy fraction e(k) of the ladder prefix through this rung —
#: a measurement over the splat set, and the one stale stamp that is RENDERING-
#: visible: the viewer multiplies an incomplete ladder's brightness by ``1/e(k)``
#: inside a ``kind=lod`` group, so a rung that still claims 0.69 after a cull left
#: it holding everything over-brightens the fully-loaded level by ~1.44x. Absent
#: it degrades exactly right — ``energyCompensation(undefined)`` returns 1 and the
#: display gate falls back to committed-count comparison.
_CONTENT_SCOPED_SUBLOD_KEYS = ("energy_fraction_cum",)

#: MEASURED stamps on each substitutive level, persisted as the level group's
#: ``level_stats``. ``quality`` is the measured mixture-L² Q of this level against
#: its lod group's finest content; ``reference_energy`` is the absolute weight w
#: those e(k) fractions are aggregated by. w goes WITH
#: :data:`_CONTENT_SCOPED_SUBLOD_KEYS` — ``lod/additive.py`` calls the e/w pairing
#: a contract (a weight with nothing to weight is a half-written stamp), and the
#: viewer's display gate needs both or neither.
_CONTENT_SCOPED_LEVEL_KEYS = ("quality", "reference_energy")

#: ``stats`` keys whose value is a LIST of nested stats dicts. The progressive
#: fitter stores one dict per pass here, mixing measured scores
#: (``cumulative_psnr_db``) with descriptive counts (``pass_index``,
#: ``seeds_requested``, ``splats_after_culling``) that stay true. Dropping the
#: whole list would throw the counts away with the scores, so the nested dicts
#: are scrubbed key-by-key instead.
_NESTED_STATS_LIST_KEYS = ("pass_stats",)


def drop_content_scoped_stats(stats: "MutableMapping[str, Any]") -> None:
    """Remove the measured reconstruction scores from ONE stats dict, in place.

    The dict-level primitive, so a caller holding raw ``stats`` (the CLI's
    node-tree path, which never builds a ``GSplatData``) scrubs exactly what the
    ``GSplatData`` paths do. Use :func:`_stats_after_content_change` when you have
    a dataset — it also reaches the per-sub-LOD and per-level ladder stamps.

    Mutates ``stats`` (which the caller owns) but nothing REACHABLE from it: a
    nested ``pass_stats`` list is replaced with scrubbed copies rather than edited
    entry-by-entry. Every ``GSplatData`` call site reaches this through a shallow
    ``dict(source.stats)``, which shares that list with the input — so editing the
    entries would delete the input's own per-pass scores and break the
    "conceptually immutable, operations return new instances" contract (#1600
    review). The top-level dict is a copy, the nested list must be made one.
    """
    for key in _CONTENT_SCOPED_STATS_KEYS:
        stats.pop(key, None)
    for key in _NESTED_STATS_LIST_KEYS:
        nested = stats.get(key)
        if isinstance(nested, list):
            stats[key] = [
                (
                    {
                        k: v
                        for k, v in entry.items()
                        if k not in _CONTENT_SCOPED_STATS_KEYS
                    }
                    if isinstance(entry, dict)
                    else entry
                )
                for entry in nested
            ]


def content_scoped_stats(
    stats: "MutableMapping[str, Any]",
    extra: "Sequence[str]" = (),
) -> "Dict[str, Any]":
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

    ``extra`` names the keys of the dict's OWN scope on top of the shared metric
    set — the per-sub-LOD / per-level ladder stamps
    (:data:`_CONTENT_SCOPED_SUBLOD_KEYS` / :data:`_CONTENT_SCOPED_LEVEL_KEYS`), so
    the snapshot restores exactly what :func:`_stats_after_content_change` takes.

    The nested per-pass lists are deep-copied so the snapshot is independent of
    the dataset it was taken from — a later edit of the source (or of the trimmed
    result, which shares the list until it is scrubbed) cannot reach back into it.
    """
    import copy

    keys = (*_CONTENT_SCOPED_STATS_KEYS, *extra)
    snapshot = {key: stats[key] for key in keys if key in stats}
    for key in _NESTED_STATS_LIST_KEYS:
        if key in stats:
            snapshot[key] = copy.deepcopy(stats[key])
    return snapshot


def _measured_stats_targets(data: "GSplatData") -> "List[tuple[Any, Sequence[str]]]":
    """The (dict, own-scope keys) pairs every measured stamp of ``data`` lives in.

    Three scopes, in a fixed order so a snapshot and a restore line up: the
    top-level dict, then every additive sub-LOD's (the leaf's ``lod_stats``), then
    every leaf's ``meta["stats"]`` (the on-disk ``level_stats``). The last one has
    to go through the NODE: ``substitutive_levels`` is rebuilt on access and copies
    ``meta["stats"]``, so mutating the view's ``SubstitutiveLevel.stats`` reaches
    nothing that gets written (the ``AdditiveSubLOD`` objects, by contrast, are
    shared with the node).
    """
    from luxar.gsplats.tree import iter_leaves

    targets: "List[tuple[Any, Sequence[str]]]" = [(data.stats, ())]
    targets += [(lod.stats, _CONTENT_SCOPED_SUBLOD_KEYS) for lod in _all_sublods(data)]
    for leaf in iter_leaves(data.tree):
        level_stats = leaf.meta.get("stats")
        if isinstance(level_stats, dict):
            targets.append((level_stats, _CONTENT_SCOPED_LEVEL_KEYS))
    return targets


def measured_stats_snapshot(data: "GSplatData") -> "List[Dict[str, Any]]":
    """:func:`content_scoped_stats` over every scope of a dataset.

    A progressive fit's ladder scores (``cumulative_psnr_db`` / ``delta_psnr_db``)
    live in the per-sub-LOD dicts, so a top-level-only snapshot would restore the
    overall PSNR and silently lose the per-pass one; the per-level scope is here
    for the same reason (nothing on the fitters' paths stamps a ``quality`` today,
    but the exemption must restore whatever the scrub takes, not a subset of it).
    """
    return [
        content_scoped_stats(stats, extra)
        for stats, extra in _measured_stats_targets(data)
    ]


def restore_measured_stats(
    data: "GSplatData", snapshot: "List[Dict[str, Any]]"
) -> "GSplatData":
    """Put a :func:`measured_stats_snapshot` back, positionally.

    Sound because a mask-based op emits one sub-LOD per input sub-LOD (the ladder
    is preserved, only its members are trimmed). A rebuild that changed the ladder
    LENGTH restores only the positions both share, which is the safe direction:
    an unmatched sub-LOD keeps no score rather than borrowing another's.
    """
    for (target, _extra), saved in zip(_measured_stats_targets(data), snapshot):
        target.update(saved)
    return data


def scrub_measured_stats(result: "GSplatData") -> None:
    """Drop every MEASURED stamp of ``result`` — top level, sub-LODs, levels.

    The dataset-level counterpart of :func:`drop_content_scoped_stats`, reaching
    the two scopes a rewrite otherwise leaves behind: the per-sub-LOD
    ``energy_fraction_cum`` (rendering-visible — see
    :data:`_CONTENT_SCOPED_SUBLOD_KEYS`) and the per-level ``quality`` /
    ``reference_energy``, which a ``_map_substitutive`` rebuild carries over
    verbatim from the input's levels. e(k) and w go together so the ladder is
    never left half-stamped.

    The structural ladder COUNTS are re-stamped from the result rather than
    dropped: the writer persists them verbatim, and a reduction makes them wrong
    (not unknown) while the result knows the truth.
    """
    for stats, extra in _measured_stats_targets(result):
        drop_content_scoped_stats(stats)
        for key in extra:
            stats.pop(key, None)
    _restamp_ladder_counts(result)


def _restamp_ladder_counts(result: "GSplatData") -> None:
    """Refresh ``lod_n_splats`` / ``lod_cumulative_n`` / ``n_splats_total``.

    Only where they are already present — this re-states a count that was
    authored, it does not start stamping one on a dataset that carried none. Each
    key is re-derived from the scope that owns it (two per sub-LOD, one per level),
    which is why they are spelled out inline rather than looped over a key tuple.
    """
    from luxar.gsplats.tree import iter_leaves

    for leaf in iter_leaves(result.tree):
        cumulative = 0
        for sublod in leaf.additive_sublods:
            cumulative += int(sublod.n_splats)
            if "lod_n_splats" in sublod.stats:
                sublod.stats["lod_n_splats"] = int(sublod.n_splats)
            if "lod_cumulative_n" in sublod.stats:
                sublod.stats["lod_cumulative_n"] = cumulative
        level_stats = leaf.meta.get("stats")
        if isinstance(level_stats, dict) and "n_splats_total" in level_stats:
            level_stats["n_splats_total"] = cumulative


def _stats_after_content_change(result: "GSplatData", *, changed: bool) -> "GSplatData":
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
    """
    if not changed:
        return result
    scrub_measured_stats(result)
    return result


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

        Removing any splat drops the inherited measured reconstruction scores
        (see :data:`_CONTENT_SCOPED_STATS_KEYS`): this is the single chokepoint
        every mask-based rewrite goes through — ``filter_by``, ``slice_by`` and
        every ``cull`` strategy — so scrubbing here covers all of them, and the
        provenance each of those stamps AFTERWARDS (``culled``, ``n_original``,
        ``filter_criteria``, ...) is untouched by the scrub.

        Example:
            >>> filtered = data.filter(data.volumes() < 100)
            >>> filtered = data.filter((data.amplitudes > 0.1) & (data.eccentricities() < 5))
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

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

        # Multi-LOD path: split mask across LODs
        if self.n_additive_sublods > 1:

            def _filter_lod(lod: AdditiveSubLOD, offset: int, n: int) -> AdditiveSubLOD:
                lod_mask = mask[offset : offset + n]
                return AdditiveSubLOD(
                    centers=lod.centers[lod_mask],
                    amplitudes=lod.amplitudes[lod_mask],
                    cholesky_factors=lod.cholesky_factors[lod_mask],
                    colors=lod.colors[lod_mask] if lod.colors is not None else None,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )

            return _stats_after_content_change(
                self._map_additive(_filter_lod), changed=removed_any
            )

        return _stats_after_content_change(
            GSplatData(
                centers=self.centers[mask],
                amplitudes=self.amplitudes[mask],
                cholesky_factors=self.cholesky_factors[mask],
                colors=self.colors[mask] if self.colors is not None else None,
                stats=dict(self.stats),
                truncation_radius=self.truncation_radius,
            ),
            changed=removed_any,
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
            out.stats.update(
                {
                    "filtered": True,
                    "n_original": self.n_splats,
                    "n_removed": self.n_splats - out.n_splats,
                    "truncate": truncate,
                }
            )
            # A crop restricts WHICH REGION the splats represent (the per-level
            # recursion above cannot fix the rebuilt top-level stats). Same for
            # the measured scores: _map_substitutive rebuilds the top level from
            # `dict(self.stats)`, so the per-level scrub inside filter() reaches
            # the sub-LOD dicts but not this one.
            _stats_after_content_change(out, changed=out.n_splats != self.n_splats)
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
