"""A rewritten artifact must not publish the fit's score as its own.

``gsplat info`` reads ``psnr_db`` as THE reconstruction quality of the dataset it
is pointed at, so a ``cull -r 0.5`` that kept 12 of 28 splats and carried the
pre-cull 44.46 dB over was publishing a score for a splat set that no longer
exists (#1600). The measured metrics are *content*-scoped: they are invalidated
by any change to which splats the artifact holds, spatially motivated or not —
which is why the pre-existing *region*-scoped predicate (``_is_crop``) could not
serve them, an amplitude threshold being the exact counter-example.

The table below is the contract, one row per operation, and every row states
whether it changes content. ``test_every_rewrite_method_is_classified`` closes it
against the actual ``GSplatData`` surface: a new public method that returns a
``GSplatData`` has to appear in a table or in ``_UNAFFECTED`` (with a reason) or
this file goes red.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

from luxar.gsplats._data.filtering import (
    _CONTENT_SCOPED_OP_RECORD_KEYS,
    _CONTENT_SCOPED_STATS_KEYS,
    _REGION_SCOPED_STATS_KEYS,
)
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel

#: Every measured reconstruction score, at a NON-DEFAULT value so a survivor is
#: unmistakable (a 0.0 could be a fresh dict's own idea of the number). Spelled
#: out LITERALLY rather than derived from the implementation constant: a fixture
#: built by iterating ``_CONTENT_SCOPED_STATS_KEYS`` would keep every row green
#: after someone deleted a key from it (the exact regression this file exists to
#: catch). ``test_the_fixture_covers_the_implementation_constant`` ties the two
#: together in the one direction that is safe.
_METRICS: Dict[str, Any] = {
    "mse": 1.11e-4,
    "psnr_db": 44.4587,
    "ssim": 0.9876,
    "rel_l2": 0.0333,
    "max_abs_error": 0.0444,
    "foreground_psnr_db": 32.8791,
    "foreground_threshold": 0.1234,
    "foreground_fraction": 0.0567,
    "final_loss": 0.001234,
    "final_rel_l2": 0.0321,
    "final_max_abs_error": 0.0456,
    "cumulative_psnr_db": 41.0,
    "delta_psnr_db": 1.5,
    "pass_psnrs": [30.0, 40.0, 44.4587],
    # The error-budget cull's measured bound and the search counters that mean
    # nothing without it: a later `cull -m cumulative` overwrites the heuristic
    # stamps but would leave these, publishing an error bound for a splat set
    # that no longer exists.
    "error_budget": 0.0271,
    "max_joint_error": 0.0269,
    "phase1_candidates": 31,
    "phase2_iterations": 7,
}

#: Provenance that stays TRUE no matter what is done to the splats: the run
#: happened, and its counters describe the run rather than the artifact. Every
#: case asserts these survive — a scrub that took them too would be a
#: regression, not a fix.
_DESCRIPTIVE: Dict[str, Any] = {
    "fitter_name": "probe",
    "iterations": 123,
    "best_iteration": 99,
    "converged": True,
    "time_seconds": 4.5,
    "source_dtype": "uint16",
}

#: Every region-scoped stamp, at a recognisable value. ALL of them, so the
#: "survived a crop" assertion is testing nine keys rather than six plus three
#: that were never there.
_REGION: Dict[str, Any] = {
    "source_shape": [64, 64, 64],
    "source_declared": True,
    "source_voxels": 64**3,
    "source_bytes": 2 * 64**3,
    "source_stored_bytes": 12345,
    "fitted_shape": [64, 64, 64],
    "fitted_voxels": 64**3,
    "occupancy": 0.01,
    "voxels_per_splat": 1000.0,
}

#: The nested per-pass dicts a progressive fit stores, mixing a measured score
#: with counts that stay true.
_PASS_STATS: List[Dict[str, Any]] = [
    {"pass_index": 0, "seeds_requested": 100, "cumulative_psnr_db": 40.0},
    {"pass_index": 1, "seeds_requested": 50, "cumulative_psnr_db": 44.4587},
]

#: The record of the reduction that produced the artifact. Content-scoped like the
#: scores — true of the op that stamped it, false of anything downstream — but kept
#: in its own dict because an op that stamps its own record REPLACES these rather
#: than leaving the dict clean, so they cannot ride the shared metric loop.
_OP_RECORD: Dict[str, Any] = {
    "culled": True,
    "culling_method": "error_budget",
    "n_original": 999,
    "n_culled": 993,
    "amplitude_retention": 0.95,
}

#: The LOD Q·e ladder stamps, at deliberately false values. A content-changing
#: rewrite must replace these with measurements of the rewritten artifact; a
#: content-preserving rewrite must keep them byte-for-byte.
_SUBLOD_LADDER: Dict[str, Any] = {
    "energy_fraction_cum": 0.6923,
    "lod_n_splats": 2,
    "lod_cumulative_n": 2,
}
_LEVEL_LADDER: Dict[str, Any] = {
    "quality": 0.9,
    "reference_energy": 1234.5,
    "n_splats_total": 4,
}


def _stats() -> Dict[str, Any]:
    return {
        **_METRICS,
        **_OP_RECORD,
        **_DESCRIPTIVE,
        "pass_stats": [dict(d) for d in _PASS_STATS],
        "part_provenance": [
            {
                "coordinate": 0,
                "fit_reference": {"kind": "acquisition", "note": "raw stack"},
                "fitting": {
                    **_METRICS,
                    **_OP_RECORD,
                    **_REGION,
                    **_DESCRIPTIVE,
                },
            }
        ],
    }


def _chol(n: int) -> np.ndarray:
    return np.tile(np.array([2, 0, 2, 0, 0, 2], dtype=np.float32), (n, 1))


def _fitted(n: int = 6) -> GSplatData:
    """A flat fit carrying the full stamp: 6 splats spread over [0, 100]³."""
    rng = np.random.default_rng(0)
    return GSplatData(
        centers=(rng.random((n, 3)) * 100).astype(np.float32),
        amplitudes=np.linspace(1.0, 0.1, n).astype(np.float32),
        cholesky_factors=_chol(n),
        stats=_stats(),
    )


def _laddered() -> GSplatData:
    """Two additive sub-LODs, each carrying its own copy of the stamp.

    A progressive fit stamps every pass's stats into that pass's sub-LOD, and the
    writer persists those as the leaf's ``lod_stats`` — so a fix that only
    rebound the top-level dict would still ship the stale score one level down.
    The e(k)/w pair rides along on the sub-LODs and the level.
    """

    def lod(centers: List[List[float]]) -> AdditiveSubLOD:
        n = len(centers)
        return AdditiveSubLOD(
            centers=np.array(centers, dtype=np.float32),
            amplitudes=np.linspace(1.0, 0.2, n).astype(np.float32),
            cholesky_factors=_chol(n),
            stats={**_stats(), **_SUBLOD_LADDER},
        )

    data = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[
                    lod([[10, 10, 10], [30, 30, 30]]),
                    lod([[70, 70, 70], [90, 90, 90]]),
                ],
                compression_factor=1,
                parent_method=None,
                level_index=0,
                stats={**_stats(), **_LEVEL_LADDER},
            )
        ],
        stats=_stats(),
    )
    return data


def _pyramid() -> GSplatData:
    """A 2-level substitutive pyramid; the coarse level is the merged one.

    ``filter_by`` / ``cull`` / the intensity ops take a per-level branch here
    that rebuilds the TOP-level stats from ``dict(self.stats)``, bypassing the
    per-level scrub — the branch this fixture exists to cover.
    """

    def level(centers: List[List[float]], index: int, factor: int) -> SubstitutiveLevel:
        n = len(centers)
        return SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=np.array(centers, dtype=np.float32),
                    amplitudes=np.linspace(1.0, 0.2, n).astype(np.float32),
                    cholesky_factors=_chol(n),
                    stats={**_stats(), **_SUBLOD_LADDER},
                )
            ],
            compression_factor=factor,
            parent_method="kmeans_lloyd",
            level_index=index,
            stats={**_stats(), **_LEVEL_LADDER},
        )

    return GSplatData.from_substitutive_levels(
        [
            level([[10, 10, 10], [30, 30, 30], [70, 70, 70], [90, 90, 90]], 0, 1),
            level([[20, 20, 20], [80, 80, 80]], 1, 2),
        ],
        stats=_stats(),
    )


def _replace_level_amplitudes(
    data: GSplatData, level_index: int, amplitudes: List[float]
) -> GSplatData:
    levels = data.substitutive_levels
    level = levels[level_index]
    lod = level.additive_sublods[0]
    levels[level_index] = SubstitutiveLevel(
        additive_sublods=[
            AdditiveSubLOD(
                centers=lod.centers,
                amplitudes=np.asarray(amplitudes, dtype=np.float32),
                cholesky_factors=lod.cholesky_factors,
                colors=lod.colors,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )
        ],
        compression_factor=level.compression_factor,
        parent_method=level.parent_method,
        level_index=level.level_index,
        stats=dict(level.stats),
    )
    return GSplatData.from_substitutive_levels(levels, stats=dict(data.stats))


def _decimate(
    target: object, method: str = "auto"
) -> Callable[[GSplatData], GSplatData]:
    from luxar.gsplats.lod.decimate import decimate

    return lambda gs: decimate(gs, target=target, method=method)  # type: ignore[arg-type]


#: ``(id, GSplatData method exercised, operation, changes_content)``.
#: ``changes_content`` is the honest answer to "does the result hold different
#: splats than the fit that was scored" — a moved count, rewritten amplitudes
#: (PSNR is an absolute-error metric, so a global x0.5 changes it outright), or
#: merged representatives.
_CASES: List[Tuple[str, str, Callable[[GSplatData], GSplatData], bool]] = [
    # ── content changes: the score must go ──────────────────────────────
    ("filter_mask", "filter", lambda gs: gs.filter(gs.amplitudes > 0.3), True),
    (
        "filter_by_amplitude",
        "filter_by",
        lambda gs: gs.filter_by(amplitude_min=0.3),
        True,
    ),
    (
        "filter_by_bbox_crop",
        "filter_by",
        lambda gs: gs.filter_by(bbox=[(0.0, 50.0)] * 3),
        True,
    ),
    ("slice_by_crop", "slice_by", lambda gs: gs.slice_by([slice(0, 50)] * 3), True),
    (
        "cull_cumulative",
        "cull",
        lambda gs: gs.cull(method="cumulative", retention=0.5),
        True,
    ),
    (
        "cull_amplitude_percentile",
        "cull",
        lambda gs: gs.cull(method="amplitude_percentile", amplitude_percentile=50.0),
        True,
    ),
    ("decimate_prefix", "filter", _decimate(3, "prefix"), True),
    # A merge lands on the requested count with every splat REPLACED by a
    # representative — the case a count-only predicate would wave through.
    ("decimate_merge", "filter", _decimate(3, "merge"), True),
    ("scale_intensity", "scale_intensity", lambda gs: gs.scale_intensity(0.5), True),
    (
        "normalize_intensity",
        "normalize_intensity",
        lambda gs: gs.normalize_intensity(4.0),
        True,
    ),
    (
        "affine_intensity",
        "affine_intensity",
        lambda gs: gs.affine_intensity(scale=2.0, offset=0.1),
        True,
    ),
    (
        "clamp_intensity",
        "clamp_intensity",
        lambda gs: gs.clamp_intensity(max=0.2),
        True,
    ),
    (
        "soft_scale_filter",
        "soft_scale_filter",
        lambda gs: gs.soft_scale_filter(highpass=1.0),
        True,
    ),
    (
        "reweight_amplitude",
        "reweight_amplitude",
        lambda gs: gs.reweight_amplitude(np.full(gs.n_splats, 0.5)),
        True,
    ),
    # ── content preserved: the score is still true of it ────────────────
    (
        "filter_mask_all_true",
        "filter",
        lambda gs: gs.filter(np.ones(gs.n_splats, bool)),
        False,
    ),
    (
        "filter_by_bbox_noop",
        "filter_by",
        lambda gs: gs.filter_by(bbox=[(-1.0, 1e4)] * 3),
        False,
    ),
    (
        "slice_by_unbounded",
        "slice_by",
        lambda gs: gs.slice_by([slice(None, None)] * 3),
        False,
    ),
    (
        "cull_retaining_all",
        "cull",
        lambda gs: gs.cull(method="cumulative", retention=1.0),
        False,
    ),
    ("decimate_full_count", "filter", _decimate(1.0), False),
    (
        "scale_intensity_by_one",
        "scale_intensity",
        lambda gs: gs.scale_intensity(1.0),
        False,
    ),
    (
        "soft_scale_filter_no_cutoff",
        "soft_scale_filter",
        lambda gs: gs.soft_scale_filter(),
        False,
    ),
    # Geometry only: the SAME splats in a moved/rescaled frame. The score is a
    # statement about how well these Gaussians reproduce the source volume, and
    # the splat set is untouched — so it survives, unlike an amplitude edit which
    # changes the rendered values. (Weaker than reproducibility: `transform`
    # records no scale factor, so the number cannot be re-derived afterwards.)
    ("transform_scale", "transform", lambda gs: gs.transform(np.eye(3) * 2.0), False),
    (
        "transform_affine",
        "transform",
        lambda gs: gs.transform(np.diag([2.0, 1.0, 1.0, 1.0])),
        False,
    ),
    (
        "translate",
        "translate",
        lambda gs: gs.translate(np.array([5.0, 0.0, -5.0])),
        False,
    ),
    (
        "center_at_centroid",
        "center_at_centroid",
        lambda gs: gs.center_at_centroid(),
        False,
    ),
    ("with_colors", "with_colors", lambda gs: gs.with_colors((1.0, 0.5, 0.25)), False),
    (
        "with_label_ids",
        "with_label_ids",
        lambda gs: gs.with_label_ids(
            np.arange(gs.n_splats, dtype=np.uint32) % 2,
            {0: "background", 1: "foreground"},
        ),
        False,
    ),
    (
        "without_label_ids",
        "without_label_ids",
        lambda gs: gs.with_label_ids(
            np.arange(gs.n_splats, dtype=np.uint32) % 2,
            {0: "background", 1: "foreground"},
        ).without_label_ids(),
        False,
    ),
    ("flattened", "flattened", lambda gs: gs.flattened(), False),
]

#: The reduced-VIEW half of the table, separate because these rows need a
#: multi-rung / multi-level input to BE reductions (a prefix of a one-rung ladder
#: is the whole thing). ``(id, method, fixture, operation, changes_content)``.
_VIEW_CASES: List[
    Tuple[str, str, Callable[[], GSplatData], Callable[[GSplatData], GSplatData], bool]
] = [
    # A strict prefix holds fewer splats than the object that was scored.
    (
        "additive_prefix_strict",
        "additive_prefix",
        _laddered,
        lambda gs: gs.additive_prefix(0),
        True,
    ),
    # The full prefix IS the input content.
    (
        "additive_prefix_full",
        "additive_prefix",
        _laddered,
        lambda gs: gs.additive_prefix(1),
        False,
    ),
    # A COARSER substitutive level is a different, merged splat set — the view
    # `lod --recipe overview` writes out as its merged coarse cap.
    (
        "at_substitutive_coarse",
        "at_substitutive",
        _pyramid,
        lambda gs: gs.at_substitutive(1),
        True,
    ),
    (
        "at_substitutive_finest",
        "at_substitutive",
        _pyramid,
        lambda gs: gs.at_substitutive(0),
        False,
    ),
]

#: Public ``GSplatData`` methods that return a ``GSplatData`` and deliberately
#: have no row, each with the reason. ``test_every_rewrite_method_is_classified``
#: enumerates the class and fails on anything in neither place, so an omission is
#: a decision on the record rather than a gap.
_UNAFFECTED = {
    "concatenate": "builds merged_stats from scratch (no metric keys)",
    "combine_as_new_dimension": "goes through concatenate; fresh stats, then may "
    "attach caller-supplied per-part provenance whose nested fitting keys follow "
    "the same content/region scrub rules on later rewrites; record cardinality "
    "cannot be pruned because the current record does not identify its center axis",
    "merge_with_channel_colors": "builds its own merged_stats from scratch",
    "embed_dimension": "widens the center columns; dataset-level source-volume "
    "metrics survive, while promoted-dimensional count/energy stamps are "
    "recomputed and stale quality/refine measurements are removed",
    "from_tree": "constructor — the caller supplies the stats",
    "from_default_selection": "tree constructor — the caller supplies root stats "
    "for the default-rendered content",
    "from_additive_sublods": "constructor — the caller supplies the stats",
    "from_substitutive_levels": "constructor — the caller supplies the stats",
    "load": "reads stats off disk; it inherits nothing from another artifact",
}

#: Non-method rewrite paths deliberately not in the table, with reasons.
_UNAFFECTED_PATHS = {
    "to_spatial_partition": "structure only: every part leaf is built with an "
    "EMPTY stats dict, so no part can inherit a score. (The ROOT fitting/ group "
    "of a partition store is the caller's to thread through the tree writer — "
    "see the sibling CLI test's partition fixture, which does exactly that.)",
}

#: Cases whose op returns a dataset with no INHERITED sub-LOD stats — ``decimate``
#: rebuilds from bare arrays, so its result is a single bare rung. Nothing can be
#: stale in an empty dict, so these rows still assert the metric half; they are
#: exempted only from the per-sub-LOD "descriptive provenance survives" half, on
#: the record rather than by an `if stats:` that would quietly excuse any op.
_DROPS_SUBLOD_STATS = {"decimate_prefix", "decimate_merge", "decimate_full_count"}

#: Cases that collapse the ladder to ONE rung whose stats are rebuilt from the
#: TOP-level dict, so a per-rung stamp like ``energy_fraction_cum`` has no
#: meaningful home in the result (there is a single, complete rung). They still
#: assert the shared metric set, which the top-level dict carries.
_REBUILDS_SUBLOD_FROM_TOP = {"flattened"}

_PARAMS = [pytest.param(cid, op, ch, id=cid) for cid, _m, op, ch in _CASES]


def _metric_keys_present(stats: Dict[str, Any]) -> "set[str]":
    return {k for k in _CONTENT_SCOPED_STATS_KEYS if k in stats}


def _assert_descriptive_intact(stats: Dict[str, Any], label: str) -> None:
    for key, value in _DESCRIPTIVE.items():
        assert stats.get(key) == value, f"{label} lost descriptive stat {key!r}"


def _assert_q_e_counts_match(data: GSplatData, label: str) -> None:
    """The authored ladder family is complete and describes ``data``."""
    for level_index, level in enumerate(data.substitutive_levels):
        assert level.stats["n_splats_total"] == level.n_splats_total, (
            f"{label} level {level_index} kept a stale total count"
        )
        assert level.stats["reference_energy"] >= 0.0
        cumulative_n = 0
        fractions: list[float] = []
        missing_fraction = False
        for rung_index, lod in enumerate(level.additive_sublods):
            cumulative_n += lod.n_splats
            assert lod.stats["lod_n_splats"] == lod.n_splats, (
                f"{label} level {level_index} rung {rung_index} kept a stale count"
            )
            assert lod.stats["lod_cumulative_n"] == cumulative_n
            if "energy_fraction_cum" in lod.stats:
                fractions.append(float(lod.stats["energy_fraction_cum"]))
            else:
                missing_fraction = True
        if missing_fraction:
            assert not fractions, f"{label} left a half-stamped energy ladder"
            assert level.stats["reference_energy"] == 0.0
            continue
        assert fractions == sorted(fractions)
        assert fractions[-1] == pytest.approx(1.0)


def test_the_fixture_covers_the_implementation_constant() -> None:
    """The fixture is the anchor; the constant may not shrink out from under it.

    ``_METRICS`` is written out literally, so a key deleted from
    ``_CONTENT_SCOPED_STATS_KEYS`` stops being scrubbed while the loop-based
    assertions (which iterate the constant) stay green. Comparing the two sets
    makes that deletion a failure here.
    """
    assert set(_METRICS) == set(_CONTENT_SCOPED_STATS_KEYS), (
        "the fixture and the implementation constant disagree; add the new key to "
        "_METRICS with a non-default value (or explain the removal here)"
    )
    # The keys the rest of the file names in prose, pinned literally.
    for key in ("psnr_db", "ssim", "foreground_psnr_db", "final_loss", "error_budget"):
        assert key in _CONTENT_SCOPED_STATS_KEYS
    assert set(_OP_RECORD) == set(_CONTENT_SCOPED_OP_RECORD_KEYS), (
        "the reduction-record fixture and its implementation constant disagree"
    )
    assert set(_REGION) == set(_REGION_SCOPED_STATS_KEYS)
    # The Q·e stamps stay OUT of the source-volume metric scrub: reductions
    # recompute them from the rewritten artifact instead of deleting them.
    scoped = set(_CONTENT_SCOPED_STATS_KEYS) | set(_CONTENT_SCOPED_OP_RECORD_KEYS)
    for key in ("energy_fraction_cum", "reference_energy", "quality"):
        assert key not in scoped, (
            f"{key!r} is a Q·e ladder stamp — recompute it on a reduction rather "
            "than scrubbing it; scrubbing breaks the scene-authoring path and "
            "licenses annotate-quality's leaf-local reference_energy fallback"
        )


@pytest.mark.parametrize("case_id,op,changes_content", _PARAMS)
def test_measured_scores_follow_the_content(
    case_id: str,
    op: Callable[[GSplatData], GSplatData],
    changes_content: bool,
) -> None:
    """One row of the table, on a flat fit."""
    out = op(_fitted())
    survivors = _metric_keys_present(out.stats)
    if changes_content:
        assert not survivors, f"stale measured scores survived: {sorted(survivors)}"
        # Named literally as well as by the loop, so the assertion cannot be
        # satisfied by an empty key list.
        assert "psnr_db" not in out.stats
        assert "final_loss" not in out.stats
    else:
        for key, value in _METRICS.items():
            assert out.stats[key] == value, (
                f"{key!r} dropped by a content-preserving op"
            )
    _assert_descriptive_intact(out.stats, "result")


@pytest.mark.parametrize("case_id,op,changes_content", _PARAMS)
def test_sublod_dicts_follow_the_top_level(
    case_id: str,
    op: Callable[[GSplatData], GSplatData],
    changes_content: bool,
) -> None:
    """The persisted twin: a leaf's ``lod_stats`` comes from the sub-LOD dicts."""
    out = op(_laddered())
    for i, lod in enumerate(out.additive_sublods):
        survivors = _metric_keys_present(lod.stats)
        if changes_content:
            assert not survivors, (
                f"stale measured scores survived in sub-LOD {i}: {sorted(survivors)}"
            )
        elif case_id not in _DROPS_SUBLOD_STATS:
            for key, value in _METRICS.items():
                assert lod.stats[key] == value, (
                    f"{key!r} dropped from sub-LOD {i} by a content-preserving op"
                )
            if case_id not in _REBUILDS_SUBLOD_FROM_TOP:
                assert lod.stats["energy_fraction_cum"] == 0.6923
    if changes_content and case_id not in _DROPS_SUBLOD_STATS:
        _assert_q_e_counts_match(out, case_id)


@pytest.mark.parametrize("case_id,op,changes_content", _PARAMS)
def test_the_input_is_never_mutated(
    case_id: str,
    op: Callable[[GSplatData], GSplatData],
    changes_content: bool,
) -> None:
    """``GSplatData`` is conceptually immutable — the scrub must respect that.

    The scrub reaches nested containers, and every call site gets there through a
    shallow ``dict(source.stats)`` that SHARES the nested ``pass_stats`` list. So
    an in-place edit of those entries deleted the per-pass scores of the object
    the caller still held: ``d.cull(...)`` emptied ``d``'s own ladder. That also
    poisons the module-scoped ``shared_progressive_fit`` fixture in
    ``conftest.py`` (a "frozen shared artifact" two other test files read), which
    makes it a seed-dependent cross-test failure under pytest-randomly.
    """
    source = _laddered()
    before_top = dict(source.stats)
    before_passes = [dict(p) for p in source.stats["pass_stats"]]
    before_sublods = [dict(lod.stats) for lod in source.additive_sublods]
    before_levels = [dict(lv.stats) for lv in source.substitutive_levels]

    out = op(source)

    assert source.stats == before_top, f"{case_id} mutated the input's top stats"
    assert [dict(p) for p in source.stats["pass_stats"]] == before_passes, (
        f"{case_id} scrubbed the INPUT's nested pass_stats in place"
    )
    assert [dict(lod.stats) for lod in source.additive_sublods] == before_sublods, (
        f"{case_id} mutated the input's sub-LOD stats"
    )
    assert [dict(lv.stats) for lv in source.substitutive_levels] == before_levels, (
        f"{case_id} mutated the input's level stats"
    )
    # ...and a SCRUBBED result does not share the nested container: the scrub
    # replaced it rather than editing the entries, which is what makes the
    # assertions above hold. (A content-preserving op legitimately still shares
    # the list — it is a shallow copy, and nothing has edited it.)
    if changes_content:
        assert out.stats["pass_stats"] is not source.stats["pass_stats"], (
            f"{case_id} shares the nested pass_stats list with its input"
        )


@pytest.mark.parametrize(
    "case_id,fixture,op,changes_content",
    [pytest.param(cid, fx, op, ch, id=cid) for cid, _m, fx, op, ch in _VIEW_CASES],
)
def test_a_reduced_view_does_not_inherit_the_score(
    case_id: str,
    fixture: Callable[[], GSplatData],
    op: Callable[[GSplatData], GSplatData],
    changes_content: bool,
) -> None:
    """A view holding fewer/different splats is a reduction like any other.

    ``lod --recipe overview`` builds its merged coarse cap as
    ``capped.at_substitutive(n - 1).flattened()``, so before this the recipe
    published the INPUT fit's ``psnr_db`` on a level made of merged
    representatives.
    """
    source = fixture()
    view = op(source)
    survivors = _metric_keys_present(view.stats)
    if changes_content:
        assert not survivors, f"a reduced view kept {sorted(survivors)}"
        assert "psnr_db" not in view.stats
    else:
        assert view.stats["psnr_db"] == _METRICS["psnr_db"]
    if case_id == "additive_prefix_strict":
        _assert_q_e_counts_match(view, case_id)
    else:
        # Full-prefix and substitutive views are accessors, not rewrites. The
        # scene-authoring path relies on their authored stamps verbatim.
        for i, lod in enumerate(view.additive_sublods):
            assert lod.stats["energy_fraction_cum"] == 0.6923, (
                f"{case_id} rung {i} lost e(k)"
            )
    if case_id.startswith("at_substitutive"):
        level_stats = view.substitutive_levels[0].stats
        assert level_stats["reference_energy"] == 1234.5, f"{case_id} lost w"
        assert level_stats["quality"] == 0.9, f"{case_id} lost Q"
    _assert_descriptive_intact(view.stats, case_id)
    # The source is untouched either way (a view must not scrub what it views).
    assert source.stats["psnr_db"] == _METRICS["psnr_db"]
    assert all(
        lod.stats["psnr_db"] == _METRICS["psnr_db"] for lod in source.additive_sublods
    )


def test_an_error_budget_bound_does_not_survive_a_later_cull() -> None:
    """The measured L∞ bound is for ONE removal, and it is not self-labelling.

    A second ``cull -m cumulative`` overwrites the heuristic stamps
    (``culling_method``, ``n_culled``) but used to leave ``error_budget`` /
    ``max_joint_error`` behind, so the store published an error bound for a splat
    set that no longer exists beside a ``culling_method`` saying the measuring run
    never happened. Safe to classify content-scoped because the error-budget cull
    re-stamps its own AFTER ``filter()`` scrubs — pinned by
    ``test_culling.py::TestGSplatDataCull::test_stats_populated``, which asserts
    ``error_budget`` / ``phase1_candidates`` ARE present on its own output.
    """
    first = _fitted()  # already carries a previous run's error-budget stamps
    assert first.stats["error_budget"] == _METRICS["error_budget"]
    second = first.cull(method="cumulative", retention=0.5)
    assert second.stats["culling_method"] == "cumulative"
    for key in (
        "error_budget",
        "max_joint_error",
        "phase1_candidates",
        "phase2_iterations",
    ):
        assert key not in second.stats, (
            f"{key!r} outlived the removal it was measured for"
        )


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda gs: gs.filter_by(amplitude_min=0.3), id="filter_by"),
        pytest.param(lambda gs: gs.cull(method="cumulative", retention=0.5), id="cull"),
        pytest.param(lambda gs: gs.scale_intensity(0.5), id="scale_intensity"),
        pytest.param(lambda gs: gs.soft_scale_filter(highpass=1.0), id="soft_filter"),
    ],
)
def test_a_pyramid_scrubs_its_rebuilt_top_level(
    op: Callable[[GSplatData], GSplatData],
) -> None:
    """The multi-substitutive branch rebuilds the top-level dict on its own."""
    out = op(_pyramid())
    survivors = _metric_keys_present(out.stats)
    assert not survivors, (
        f"a pyramid rewrite kept its top-level scores: {sorted(survivors)}"
    )
    _assert_descriptive_intact(out.stats, "pyramid result")


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda gs: gs.filter_by(amplitude_min=0.3), id="filter_by"),
        pytest.param(lambda gs: gs.cull(method="cumulative", retention=0.5), id="cull"),
        pytest.param(lambda gs: gs.scale_intensity(0.5), id="scale_intensity"),
    ],
)
def test_a_reduction_recomputes_energy_stamps_and_drops_quality(
    op: Callable[[GSplatData], GSplatData],
) -> None:
    """Cheap artifact-local energy stamps describe the rewritten splats."""
    from luxar.gsplats.lod.quality import total_self_energy

    source = _pyramid()
    for level in source.substitutive_levels:
        level.stats["lod_cutpoints"] = [999]
    from luxar.gsplats.tree import iter_leaves

    for leaf in iter_leaves(source.tree):
        leaf.meta["stats"]["refine_stats"] = {"mse_seed": 9.0, "mse_refit": 3.0}

    out = op(source)
    assert not _metric_keys_present(out.stats), (
        "the source-volume scrub stopped working"
    )

    levels = out.substitutive_levels
    finest = out.at_substitutive(0).flattened()
    expected_w = total_self_energy(finest)
    for level in levels:
        assert level.stats["n_splats_total"] == level.n_splats_total
        assert level.stats["reference_energy"] == pytest.approx(expected_w)
        assert "quality" not in level.stats
        assert "median_footprint" not in level.stats
        assert "footprint_dims" not in level.stats
        assert "refine_stats" not in level.stats
        assert "lod_cutpoints" not in level.stats

        cumulative_n = 0
        energies = [
            total_self_energy(GSplatData.from_additive_sublods([lod]))
            for lod in level.additive_sublods
        ]
        total = sum(energies)
        cumulative_energy = 0.0
        for lod, energy in zip(level.additive_sublods, energies):
            cumulative_n += lod.n_splats
            cumulative_energy += energy
            assert lod.stats["lod_n_splats"] == lod.n_splats
            assert lod.stats["lod_cumulative_n"] == cumulative_n
            assert lod.stats["energy_fraction_cum"] == pytest.approx(
                cumulative_energy / total
            )


def test_filter_restamps_when_only_a_coarse_level_changes() -> None:
    source = _replace_level_amplitudes(_pyramid(), 1, [0.01, 0.2])
    out = source.filter_by(amplitude_min=0.1)
    coarse = out.substitutive_levels[1]
    assert out.n_splats == source.n_splats
    assert coarse.n_splats_total == 1
    assert coarse.stats["n_splats_total"] == 1
    assert coarse.additive_sublods[0].stats["lod_cumulative_n"] == 1
    assert "quality" not in coarse.stats


def test_intensity_restamps_when_only_a_coarse_level_changes() -> None:
    source = _replace_level_amplitudes(_pyramid(), 1, [4.0, 0.2])
    out = source.clamp_intensity(max=1.0)
    coarse = out.substitutive_levels[1]
    assert np.array_equal(out.amplitudes, source.amplitudes)
    assert coarse.stats["reference_energy"] != pytest.approx(1234.5)
    assert "quality" not in coarse.stats


@pytest.mark.parametrize(
    ("level_stats", "lod_stats"),
    [
        pytest.param({}, {}, id="empty"),
        pytest.param({"quality": 0.9}, {}, id="level-stamp"),
        pytest.param({}, {"energy_fraction_cum": 1.0}, id="rung-stamp"),
        pytest.param(
            {"reference_energy": 1.0},
            {"lod_cumulative_n": 2},
            id="both-stamps",
        ),
        pytest.param({"label": "fine"}, {"note": "keep"}, id="other-metadata"),
    ],
)
def test_restamp_prefilter_matches_exact_stamp_guard(
    level_stats: Dict[str, Any], lod_stats: Dict[str, Any]
) -> None:
    from luxar.gsplats._data.filtering import _needs_reduction_lod_restamp
    from luxar.gsplats.lod.restamp import _has_ladder_stamps

    source_lod = _fitted(2).additive_sublods[0]
    source = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=source_lod.centers,
                        amplitudes=source_lod.amplitudes,
                        cholesky_factors=source_lod.cholesky_factors,
                        stats=lod_stats,
                    )
                ],
                stats=level_stats,
            )
        ]
    )
    level = source.substitutive_levels[0]

    assert _needs_reduction_lod_restamp(source, source) == _has_ladder_stamps(level)


def test_restamp_prefilter_checks_coarse_only_stamps() -> None:
    from luxar.gsplats._data.filtering import _needs_reduction_lod_restamp
    from luxar.gsplats.lod.restamp import _has_ladder_stamps

    fine = _fitted(4).substitutive_levels[0]
    coarse = _fitted(2).substitutive_levels[0]
    coarse = SubstitutiveLevel(
        additive_sublods=coarse.additive_sublods,
        compression_factor=2,
        parent_method="kmeans_lloyd",
        level_index=1,
        stats={"refine_stats": {"mse_seed": 1.0}},
    )
    source = GSplatData.from_substitutive_levels([fine, coarse])
    levels = source.substitutive_levels

    assert not _has_ladder_stamps(levels[0])
    assert _has_ladder_stamps(levels[1])
    assert _needs_reduction_lod_restamp(source, source)


def test_nonfinite_energy_is_reset_and_fraction_is_removed() -> None:
    source = _replace_level_amplitudes(_pyramid(), 0, [np.inf, 0.7, 0.4, 0.2])
    out = source.scale_intensity(0.5)
    for level in out.substitutive_levels:
        assert level.stats["reference_energy"] == 0.0
    assert "energy_fraction_cum" not in out.additive_sublods[0].stats


def test_decimate_recomputes_a_complete_single_rung_stamp() -> None:
    from luxar.gsplats.lod.quality import total_self_energy

    out = _decimate(2, "prefix")(_laddered())
    level = out.substitutive_levels[0]
    lod = level.additive_sublods[0]
    assert lod.stats["lod_n_splats"] == out.n_splats
    assert lod.stats["lod_cumulative_n"] == out.n_splats
    assert lod.stats["energy_fraction_cum"] == 1.0
    assert level.stats["n_splats_total"] == out.n_splats
    assert "quality" not in level.stats
    assert level.stats["reference_energy"] == pytest.approx(total_self_energy(out))


@pytest.mark.parametrize("n_rungs", [1, 2])
@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda gs: gs.transform(np.diag([2.0, 2.0, 2.0])), id="transform"),
        pytest.param(
            lambda gs: gs.translate(np.array([1.0, 2.0, 3.0])), id="translate"
        ),
        pytest.param(lambda gs: gs.with_colors((0.2, 0.4, 0.6)), id="with_colors"),
        pytest.param(lambda gs: gs.scale_intensity(1.0), id="scale_intensity_noop"),
        pytest.param(lambda gs: gs.filter_by(amplitude_min=0.0), id="filter_by_noop"),
    ],
)
def test_content_preserving_rewrite_keeps_authored_q_e_stamps(
    n_rungs: int, op: Callable[[GSplatData], GSplatData]
) -> None:
    source = _laddered()
    if n_rungs == 1:
        first_lod = source.additive_sublods[0]
        source = GSplatData.from_substitutive_levels(
            [
                SubstitutiveLevel(
                    additive_sublods=[first_lod],
                    compression_factor=1,
                    parent_method=None,
                    level_index=0,
                    stats={**_stats(), **_LEVEL_LADDER},
                )
            ],
            stats=_stats(),
        )
    out = op(source)
    assert out.substitutive_levels[0].stats == source.substitutive_levels[0].stats
    assert [lod.stats for lod in out.additive_sublods] == [
        lod.stats for lod in source.additive_sublods
    ]


def test_single_rung_filter_preserves_level_provenance_after_reduction() -> None:
    source_lod = _laddered().additive_sublods[0]
    source = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[source_lod],
                compression_factor=4,
                parent_method="kmeans_lloyd",
                level_index=2,
                stats={**_stats(), **_LEVEL_LADDER},
            )
        ],
        stats=_stats(),
    )

    out = source.filter(source.amplitudes > 0.5)
    level = out.substitutive_levels[0]
    assert (level.compression_factor, level.parent_method, level.level_index) == (
        4,
        "kmeans_lloyd",
        2,
    )
    assert level.stats["n_splats_total"] == out.n_splats
    assert "quality" not in level.stats


@pytest.mark.parametrize(
    "source_factory",
    [
        pytest.param(
            lambda: GSplatData.from_substitutive_levels(
                [
                    SubstitutiveLevel(
                        additive_sublods=[_laddered().additive_sublods[0]],
                        compression_factor=4,
                        parent_method="kmeans_lloyd",
                        level_index=2,
                        stats={**_stats(), **_LEVEL_LADDER},
                    )
                ],
                stats=_stats(),
            ),
            id="single-rung",
        ),
        pytest.param(_laddered, id="additive-ladder"),
        pytest.param(_pyramid, id="substitutive-pyramid"),
    ],
)
def test_embed_dimension_restamps_lod_metadata(
    source_factory: Callable[[], GSplatData],
) -> None:
    from luxar.gsplats.lod.quality import total_self_energy

    source = source_factory()
    out = source.embed_dimension(0.0, sigma=1.0)
    expected_energy = total_self_energy(out.at_substitutive(0).flattened())

    assert expected_energy != pytest.approx(_LEVEL_LADDER["reference_energy"])
    assert [
        (level.compression_factor, level.parent_method, level.level_index)
        for level in out.substitutive_levels
    ] == [
        (level.compression_factor, level.parent_method, level.level_index)
        for level in source.substitutive_levels
    ]
    for level in out.substitutive_levels:
        assert level.stats["reference_energy"] == pytest.approx(expected_energy)
        assert level.stats["n_splats_total"] == level.n_splats_total
        assert "quality" not in level.stats
        assert level.additive_sublods[-1].stats["energy_fraction_cum"] == 1.0


def test_reveal_ladder_stays_without_energy_compensation_after_reduction() -> None:
    from luxar.gsplats.tree import iter_leaves

    source = _laddered()
    for leaf in iter_leaves(source.tree):
        leaf.meta["stats"].pop("reference_energy", None)
        for lod in leaf.additive_sublods:
            lod.stats["lod_method"] = "radial"
            lod.stats.pop("energy_fraction_cum", None)

    out = source.scale_intensity(0.5)
    assert "reference_energy" not in out.substitutive_levels[0].stats
    assert all("energy_fraction_cum" not in lod.stats for lod in out.additive_sublods)


def test_tree_restamp_shares_finest_energy_and_drops_quality() -> None:
    from luxar.gsplats.lod.quality import total_self_energy
    from luxar.gsplats.lod.restamp import refresh_reduction_lod_tree
    from luxar.gsplats.tree import map_leaves

    source = _pyramid().tree
    for child in source.children:
        child.meta["stats"]["median_footprint"] = 4.0
        child.meta["stats"]["footprint_dims"] = [0, 1, 2]

    def scale_leaf(leaf: Any) -> Any:
        return GSplatData.from_tree(leaf).scale_intensity(0.5).tree

    result = refresh_reduction_lod_tree(map_leaves(source, scale_leaf), source)
    data = GSplatData.from_tree(result)
    finest = data.at_substitutive(0).flattened()
    expected_w = total_self_energy(finest)
    for level in data.substitutive_levels:
        assert "quality" not in level.stats
        assert "median_footprint" not in level.stats
        assert "footprint_dims" not in level.stats
        assert level.stats["reference_energy"] == pytest.approx(expected_w)


def test_the_scene_authoring_path_keeps_a_coarse_level_stamped() -> None:
    """The regression guard for the scene-authoring path, at its own seam.

    ``core/group/gsplats_pipeline/lod_dispatch.py`` walks ``at_substitutive(s)``
    for every substitutive level and copies ``reference_energy`` / ``quality`` /
    ``energy_kind`` off ``level_view.substitutive_levels[0].stats`` into the child
    leaf's ``level_stats``, plus each rung's ``energy_fraction_cum`` from
    ``dict(lod.stats)``. ``at_substitutive`` is a pure ACCESSOR there, not a
    rewrite, so treating a coarse level as a reduction left every coarse child of a
    ``kind=lod`` group unstamped on disk. This walks the same accessor rather than
    compiling a scene, so it costs nothing; the on-disk equivalent is covered by
    the sibling CLI test's pyramid fixtures.
    """
    source = _pyramid()
    for s in range(source.n_substitutive):
        view = source.at_substitutive(s)
        level_stats = view.substitutive_levels[0].stats
        assert level_stats["reference_energy"] == 1234.5, (
            f"coarse child {s} would be written with no reference_energy — the "
            "viewer falls back to counting elements and annotate-quality would "
            "fabricate a leaf-local w"
        )
        assert level_stats["quality"] == 0.9, f"coarse child {s} lost its measured Q"
        for lod in view.additive_sublods:
            assert lod.stats["energy_fraction_cum"] == 0.6923, (
                f"coarse child {s} would be written with no e(k)"
            )


@pytest.mark.parametrize(
    "case_id,op",
    [
        pytest.param("decimate", _decimate(2, "prefix"), id="decimate"),
        pytest.param(
            "filter_by", lambda gs: gs.filter_by(amplitude_min=0.3), id="filter_by"
        ),
        pytest.param(
            "slice_by", lambda gs: gs.slice_by([slice(0, 50)] * 3), id="slice_by"
        ),
        pytest.param(
            "scale_intensity", lambda gs: gs.scale_intensity(0.5), id="scale_intensity"
        ),
    ],
)
def test_an_inherited_reduction_record_does_not_ride_along(
    case_id: str, op: Callable[[GSplatData], GSplatData]
) -> None:
    """A rewrite must not publish the INPUT's cull record as its own.

    ``decimate`` is the case that motivated it: threading the input's provenance
    through (so ``fitter_name`` / the source grid survive) also carried
    ``culled: True`` with ``amplitude_retention: 0.95`` onto a prefix reduction
    that had just discarded ~75% of the amplitude mass, and ``n_original`` /
    ``n_culled`` then described a removal that is not the one that happened.
    """
    out = op(_fitted())
    for key, stale in _OP_RECORD.items():
        assert out.stats.get(key) != stale, (
            f"{case_id} published the input's {key!r} for a reduction it did not do"
        )
    # None of these ops is a cull, so the cull-specific half is gone outright
    # (``filter_by`` / ``slice_by`` legitimately re-stamp their own ``n_original``).
    for key in ("culled", "culling_method", "n_culled", "amplitude_retention"):
        assert key not in out.stats, f"{case_id} kept the input's {key!r}"
    _assert_descriptive_intact(out.stats, case_id)


def test_a_cull_publishes_its_own_record_over_the_inherited_one() -> None:
    """The scrub must not eat the record the op stamps right after it.

    Every op that stamps a reduction record does so AFTER its ``filter()``, which
    is what makes the record content-scoped safe. The two multi-substitutive
    branches (``cull`` / ``filter_by``) build their top-level dict themselves and
    therefore scrub BEFORE stamping — asserted here on the pyramid as well as flat,
    since that ordering is the one a refactor can silently invert.
    """
    for source in (_fitted(), _pyramid()):
        out = source.cull(method="cumulative", retention=0.5)
        assert out.stats["culled"] is True
        assert out.stats["culling_method"] == "cumulative"
        assert out.stats["n_original"] == source.n_splats, (
            "the cull's own record was scrubbed after being stamped"
        )
        assert out.stats["n_culled"] == source.n_splats - out.n_splats
        assert out.stats["n_original"] != _OP_RECORD["n_original"]

    # `filter_by` stamps `n_original` too, on both branches.
    for source in (_fitted(), _pyramid()):
        out = source.filter_by(amplitude_min=0.3)
        assert out.stats["n_original"] == source.n_splats
        assert out.stats["filtered"] is True


def test_nested_pass_stats_lose_the_score_but_keep_the_counts() -> None:
    """``pass_stats`` mixes a measured score with counts; only the score goes."""
    out = _fitted().cull(method="cumulative", retention=0.5)
    passes = out.stats["pass_stats"]
    assert len(passes) == len(_PASS_STATS), "the per-pass list was dropped wholesale"
    for i, entry in enumerate(passes):
        assert "cumulative_psnr_db" not in entry, f"pass {i} kept a stale PSNR"
        assert entry["pass_index"] == _PASS_STATS[i]["pass_index"]
        assert entry["seeds_requested"] == _PASS_STATS[i]["seeds_requested"]


def test_part_provenance_follows_content_and_region_scope() -> None:
    """Nested component fits keep identity while stale measurements are scrubbed."""
    source = _fitted()
    original = source.stats["part_provenance"]

    culled = source.cull(method="cumulative", retention=0.5)
    culled_record = culled.stats["part_provenance"][0]
    culled_fitting = culled_record["fitting"]
    assert culled_record["coordinate"] == 0
    assert culled_record["fit_reference"] == original[0]["fit_reference"]
    assert not set(_CONTENT_SCOPED_STATS_KEYS) & culled_fitting.keys()
    assert not set(_CONTENT_SCOPED_OP_RECORD_KEYS) & culled_fitting.keys()
    for key, value in _REGION.items():
        assert culled_fitting[key] == value
    assert culled_fitting["fitter_name"] == _DESCRIPTIVE["fitter_name"]

    cropped = source.filter_by(bbox=[(0.0, 50.0)] * 3)
    cropped_fitting = cropped.stats["part_provenance"][0]["fitting"]
    assert not set(_CONTENT_SCOPED_STATS_KEYS) & cropped_fitting.keys()
    assert not set(_CONTENT_SCOPED_OP_RECORD_KEYS) & cropped_fitting.keys()
    assert not set(_REGION_SCOPED_STATS_KEYS) & cropped_fitting.keys()
    assert cropped_fitting["fitter_name"] == _DESCRIPTIVE["fitter_name"]

    assert source.stats["part_provenance"] == original
    assert culled.stats["part_provenance"] is not original
    assert cropped.stats["part_provenance"] is not original


def test_the_two_categories_stay_independent() -> None:
    """The regression guard for the region stamps: neither set subsumes the other.

    A non-spatial cull keeps every region stamp (the surviving splats still
    represent the whole fitted volume) while losing the scores; a bbox crop loses
    both. Reusing one predicate for both — what #1600 was — collapses this
    2x2 into a diagonal.
    """
    stamped = _fitted()
    stamped.stats.update(_REGION)

    culled = stamped.cull(method="cumulative", retention=0.5)
    assert not _metric_keys_present(culled.stats)
    for key, value in _REGION.items():
        assert culled.stats[key] == value, (
            f"a non-spatial cull dropped the region stamp {key!r}"
        )

    stamped = _fitted()
    stamped.stats.update(_REGION)
    cropped = stamped.filter_by(bbox=[(0.0, 50.0)] * 3)
    assert not _metric_keys_present(cropped.stats)
    for key in _REGION_SCOPED_STATS_KEYS:
        assert key not in cropped.stats, f"{key!r} survived a crop"


def test_the_fitters_keep_the_score_across_their_own_closing_trim() -> None:
    """The deliberate exemption, at the seam the fitters use.

    Every fitter ends with a high-retention cumulative cull (``cull_retention``,
    0.95 by default) applied AFTER it scored the reconstruction. Scrubbing there
    would leave every default fit with no ``psnr_db`` at all — a worse answer than
    one taken before a trim that drops 5% of the amplitude, and re-scoring costs a
    second full render. The per-pass ladder scores live in the sub-LOD dicts, so
    the snapshot reaches those too — it is the exact inverse of the scrub.
    """
    from luxar.gsplats._data.filtering import (
        measured_stats_snapshot,
        restore_measured_stats,
    )

    fitted = _laddered()
    saved = measured_stats_snapshot(fitted)
    trimmed = fitted.cull(method="cumulative", retention=0.5)
    assert not _metric_keys_present(trimmed.stats), "the cull itself must still scrub"

    restore_measured_stats(trimmed, saved)
    for key, value in _METRICS.items():
        assert trimmed.stats[key] == value, f"{key!r} was not restored"
    for i, lod in enumerate(trimmed.additive_sublods):
        assert lod.stats["cumulative_psnr_db"] == _METRICS["cumulative_psnr_db"], (
            f"sub-LOD {i}'s ladder score was not restored"
        )
    _assert_q_e_counts_match(trimmed, "closing trim")
    # The nested per-pass dicts are deep-copied into the snapshot, or the scrub
    # would have emptied them in place before the restore could read them.
    assert trimmed.stats["pass_stats"][0]["cumulative_psnr_db"] == 40.0
    # The trim's OWN record shows through — the snapshot deliberately does not
    # carry the reduction record, so restoring cannot resurrect an older cull's
    # `n_original` / `amplitude_retention` over the one that just ran (a tiled fit
    # culls each tile, then culls the merge).
    assert trimmed.stats["culling_method"] == "cumulative"
    assert trimmed.stats["n_original"] == fitted.n_splats

    # A pyramid round-trips too, on the branch that rebuilds the top level itself.
    pyr = _pyramid()
    saved = measured_stats_snapshot(pyr)
    trimmed = pyr.cull(method="cumulative", retention=0.5)
    assert not _metric_keys_present(trimmed.stats)
    restore_measured_stats(trimmed, saved)
    for key, value in _METRICS.items():
        assert trimmed.stats[key] == value, f"{key!r} was not restored on the pyramid"


def test_fitter_score_restore_refuses_a_pruned_ladder_shape() -> None:
    from luxar.gsplats._data.filtering import (
        measured_stats_snapshot,
        restore_measured_stats,
    )

    faint = AdditiveSubLOD(
        centers=np.zeros((2, 3), dtype=np.float32),
        amplitudes=np.array([0.01, 0.01], dtype=np.float32),
        cholesky_factors=_chol(2),
        stats={"pass_index": 20, "cumulative_psnr_db": 20.0},
    )
    bright = AdditiveSubLOD(
        centers=np.ones((2, 3), dtype=np.float32),
        amplitudes=np.array([1.0, 0.9], dtype=np.float32),
        cholesky_factors=_chol(2),
        stats={"pass_index": 35, "cumulative_psnr_db": 35.0},
    )
    fitted = GSplatData.from_additive_sublods([faint, bright], stats=_stats())
    saved = measured_stats_snapshot(fitted)

    trimmed = fitted.cull(method="cumulative", retention=0.95)
    assert trimmed.n_additive_sublods == 1
    assert np.all(trimmed.additive_sublods[0].centers == 1.0)

    restore_measured_stats(trimmed, saved)

    assert trimmed.stats["psnr_db"] == _METRICS["psnr_db"]
    assert trimmed.additive_sublods[0].stats["pass_index"] == 35
    assert "cumulative_psnr_db" not in trimmed.additive_sublods[0].stats


def test_a_real_fit_still_publishes_its_psnr() -> None:
    """End to end, on the default path: `cull_retention=0.95` fires every time."""
    pytest.importorskip("torch")
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    V = np.zeros((16, 16), dtype=np.float32)
    V[4:8, 4:8] = 1.0
    V[10:13, 9:12] = 0.6
    result = fit_gaussian_splats(
        V, seeds=40, n_iters=20, verbose=False, device="cpu", napari_movie=False
    )
    assert result.stats.get("culled") is True, (
        "the fit did not reach its closing trim — the test proves nothing"
    )
    assert "psnr_db" in result.stats, "a default fit lost its own measured PSNR"
    assert "final_max_abs_error" in result.stats


@pytest.mark.slow
def test_a_real_progressive_fit_keeps_its_per_pass_ladder() -> None:
    """The exemption at the ONE call site with a per-pass ladder of scores.

    The helpers are exercised directly above, which would stay green if all three
    fitter call sites were deleted. The progressive fitter is the interesting one:
    its closing trim runs while the per-pass ladder is still live, and its
    published record of that ladder is the nested ``pass_stats`` list (plus the
    rolled-up ``pass_psnrs``) — the two keys the key-by-key nested scrub exists
    for, and the ones the in-place-mutation bug destroyed. Marked ``slow``
    because it is three real optimiser runs; the volume is 24x24.
    """
    pytest.importorskip("torch")
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    rng = np.random.default_rng(0)
    V = np.zeros((24, 24), dtype=np.float32)
    V[3:7, 3:7] = 1.0
    V[10:14, 12:17] = 0.7
    V[18:22, 5:9] = 0.4
    V += rng.random(V.shape).astype(np.float32) * 0.05
    result = fit_progressive_gaussian_splats(
        V,
        max_splats=60,
        max_splats_per_pass=20,
        iters_per_pass=25,
        max_passes=3,
        psnr_patience=0.0,
        verbose=False,
        device="cpu",
    )
    assert result.stats.get("culled") is True, (
        "the progressive fit did not reach its closing trim — nothing is proven"
    )
    assert "psnr_db" in result.stats, "the progressive fit lost its overall PSNR"
    passes = result.stats["pass_stats"]
    assert len(passes) > 1, "only one pass ran; the ladder claim proves nothing"
    ladder = [entry.get("cumulative_psnr_db") for entry in passes]
    assert all(isinstance(p, float) for p in ladder), (
        f"the closing trim lost the per-pass ladder scores: {ladder}"
    )
    assert len(result.stats["pass_psnrs"]) == len(passes)


def test_every_rewrite_method_is_classified() -> None:
    """A new ``GSplatData`` rewrite cannot skip the table by not being in it.

    Enumerates the public methods that return a ``GSplatData`` — the actual API
    surface — and requires each to be exercised by a row or listed in
    ``_UNAFFECTED`` with a reason. The previous version of this test only checked
    that the ids were unique and non-empty, which closed nothing.
    """
    surface = set()

    # `ismethod` as well as `isfunction`: `concatenate` / `from_tree` / `load` are
    # classmethods, and a predicate that missed them would let a new classmethod
    # rewrite path in unclassified.
    def _callable(obj: object) -> bool:
        return inspect.isfunction(obj) or inspect.ismethod(obj)

    for name, fn in inspect.getmembers(GSplatData, predicate=_callable):
        if name.startswith("_"):
            continue
        try:
            ret = inspect.signature(fn).return_annotation
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if isinstance(ret, str) and "GSplatData" in ret:
            surface.add(name)
    assert "cull" in surface and "filter_by" in surface, (
        "the enumeration found nothing recognisable; the annotation convention "
        f"must have changed (found: {sorted(surface)})"
    )

    exercised = {m for _cid, m, _op, _ch in _CASES}
    exercised |= {m for _cid, m, _fx, _op, _ch in _VIEW_CASES}
    classified = exercised | set(_UNAFFECTED)
    assert not surface - classified, (
        "unclassified GSplatData rewrite method(s) — add a row to _CASES / "
        f"_VIEW_CASES or an entry to _UNAFFECTED: {sorted(surface - classified)}"
    )
    assert not classified - surface, (
        f"the table names methods that no longer exist: {sorted(classified - surface)}"
    )
    assert all(_UNAFFECTED.values()), "every exemption needs a reason"
    # The non-method paths are named separately (they do not return a GSplatData,
    # so the enumeration above cannot see them) but must still carry a reason.
    assert all(_UNAFFECTED_PATHS.values()), "every exemption needs a reason"
    for name in _UNAFFECTED_PATHS:
        assert hasattr(GSplatData, name), f"{name!r} no longer exists"

    ids = [case[0] for case in _CASES] + [case[0] for case in _VIEW_CASES]
    assert len(ids) == len(set(ids)), "duplicate case id in the table"
    assert any(ch for _c, _m, _o, ch in _CASES)
    assert any(not ch for _c, _m, _o, ch in _CASES)
    assert _DROPS_SUBLOD_STATS <= set(ids), "a stats-free exemption names no case"


def test_concatenate_never_inherits_content_scoped_metrics() -> None:
    """Fresh merged stats are the contract on both non-empty and empty inputs."""
    for n_splats in (0, 2):
        inputs = []
        for seed in (1, 2):
            inputs.append(
                GSplatData(
                    centers=np.zeros((n_splats, 3), dtype=np.float32),
                    amplitudes=np.ones(n_splats, dtype=np.float32),
                    cholesky_factors=np.tile(
                        np.array([1, 0, 1, 0, 0, 1], dtype=np.float32),
                        (n_splats, 1),
                    ),
                    stats={**_METRICS, "seed": seed},
                )
            )
        merged = GSplatData.concatenate(inputs)
        assert not _CONTENT_SCOPED_STATS_KEYS & merged.stats.keys()
