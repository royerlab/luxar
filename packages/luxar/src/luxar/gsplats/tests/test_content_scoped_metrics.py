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

#: The LOD Q·e ladder stamps, DELIBERATELY out of scope (see the module docstring
#: of ``_data/filtering.py``): e(k) per rung, and the measured Q with its
#: aggregation weight w per level. They are measured on the artifact's own content,
#: the scene-authoring path copies them onto every coarse child of a ``kind=lod``
#: group through ``at_substitutive``, and deleting w licenses
#: ``annotate-quality``'s leaf-local fallback. Every case asserts they are left
#: exactly as authored.
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
    "combine_as_new_dimension": "goes through concatenate; fresh stats",
    "merge_with_channel_colors": "goes through concatenate; fresh stats",
    "embed_dimension": "widens the center columns; the splat set is unchanged, "
    "and it carries stats over exactly as a geometry transform does",
    "from_tree": "constructor — the caller supplies the stats",
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

# These crops empty the second rung of ``_laddered``. Structural pruning drops
# the surviving rung's e(k), because it no longer describes the completed
# ladder; ordinary content rewrites that keep every rung still preserve e(k).
_PRUNES_EMPTY_SUBLOD = {"filter_by_bbox_crop", "slice_by_crop"}

_PARAMS = [pytest.param(cid, op, ch, id=cid) for cid, _m, op, ch in _CASES]


def _metric_keys_present(stats: Dict[str, Any]) -> "set[str]":
    return {k for k in _CONTENT_SCOPED_STATS_KEYS if k in stats}


def _assert_descriptive_intact(stats: Dict[str, Any], label: str) -> None:
    for key, value in _DESCRIPTIVE.items():
        assert stats.get(key) == value, f"{label} lost descriptive stat {key!r}"


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
    # The Q·e stamps are OUT of both content-scoped sets, on purpose: they are
    # measured on the artifact's own content and the scene-authoring path copies
    # them onto coarse levels through `at_substitutive`. Adding one here would
    # break `lod_dispatch.py` (see the module docstring of `_data/filtering.py`).
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
            if case_id not in _DROPS_SUBLOD_STATS | _REBUILDS_SUBLOD_FROM_TOP:
                if case_id in _PRUNES_EMPTY_SUBLOD:
                    assert out.n_additive_sublods == 1
                    assert "energy_fraction_cum" not in lod.stats
                else:
                    # ...and the rung's own Q·e stamp is NOT collateral when
                    # the ladder shape survives the rewrite.
                    assert lod.stats["energy_fraction_cum"] == 0.6923, (
                        f"sub-LOD {i} lost its e(k) to the PSNR scrub"
                    )
        elif case_id not in _DROPS_SUBLOD_STATS:
            for key, value in _METRICS.items():
                assert lod.stats[key] == value, (
                    f"{key!r} dropped from sub-LOD {i} by a content-preserving op"
                )
            if case_id not in _REBUILDS_SUBLOD_FROM_TOP:
                assert lod.stats["energy_fraction_cum"] == 0.6923


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
    # Either way the view's OWN Q·e stamps survive: a reduced view scrubs only the
    # INHERITED top-level scores it cannot claim. The scene-authoring path builds
    # every coarse child of a `kind=lod` group with `at_substitutive(s)` and reads
    # these off the view (`lod_dispatch.py`), so scrubbing them here silently
    # stripped the viewer's e(k) gate and 1/e(k) compensation from coarse levels.
    for i, lod in enumerate(view.additive_sublods):
        assert lod.stats["energy_fraction_cum"] == 0.6923, (
            f"{case_id} rung {i} lost e(k)"
        )
    # (An `additive_prefix` view is built from bare rungs, so it has no level
    # stats to keep; the substitutive views carry the level's own Q/w.)
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
        pytest.param(_decimate(2, "prefix"), id="decimate_prefix"),
    ],
)
def test_the_q_e_ladder_stamps_are_left_exactly_as_authored(
    op: Callable[[GSplatData], GSplatData],
) -> None:
    """The NARROWED contract, in the direction a future change must not break.

    ``energy_fraction_cum`` / ``reference_energy`` / level ``quality`` are measured
    on the artifact's own content — a coarse level's Q is that level against its
    group's finest, its e(k) is its own prefix energy — so the "measured against
    the source volume, therefore invalidated by a rewrite" argument does not reach
    them, and this rule leaves them alone. Scrubbing them broke two things at once:
    ``lod_dispatch.py`` builds every coarse child of a ``kind=lod`` group with
    ``at_substitutive(s)`` and copies these numbers off the view (so coarse levels
    lost the viewer's ``e(k) >= 0.6`` upgrade release and its ``1/e(k)`` brightness
    compensation), and ``lod/annotate.py:332`` writes a leaf-local
    ``reference_energy`` only when none is present — so deleting w licenses a
    fabricated, group-inconsistent one on a store that then LOOKS well stamped.
    """
    from luxar.gsplats.tree import iter_leaves

    source = _pyramid()
    assert source.substitutive_levels[0].stats["reference_energy"] == 1234.5
    assert source.additive_sublods[0].stats["energy_fraction_cum"] == 0.6923

    out = op(source)
    # The PSNR family really did go (or this proves nothing about the narrowing).
    assert not _metric_keys_present(out.stats), "the metric scrub stopped working"
    # Each level's own Q/w, where the level stats live: the leaf's ``meta``, which
    # is what the writer persists as ``level_stats``. (``decimate`` rebuilds from
    # bare arrays and has no level stats at all — nothing to preserve there.)
    for leaf in iter_leaves(out.tree):
        level_stats = leaf.meta.get("stats") or {}
        if level_stats:
            assert level_stats["reference_energy"] == 1234.5, "w was scrubbed"
            assert level_stats["quality"] == 0.9, "the level's measured Q was scrubbed"

    # The per-rung e(k), on the fixture whose ladder survives the rewrite: a
    # `_map_substitutive` rebuild replaces a level's single rung stats with that
    # level's rebuilt top-level dict (pre-existing, unrelated to this rule), while
    # `_map_additive` carries every rung's own dict over — which is where a scrub
    # would show.
    laddered_src = _laddered()
    laddered = op(laddered_src)
    assert not _metric_keys_present(laddered.stats)
    if laddered.n_additive_sublods == laddered_src.n_additive_sublods:
        for i, lod in enumerate(laddered.additive_sublods):
            assert lod.stats.get("energy_fraction_cum") == 0.6923, (
                f"rung {i} lost its e(k) to the PSNR scrub"
            )


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
        assert lod.stats["energy_fraction_cum"] == 0.6923
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
    assert _PRUNES_EMPTY_SUBLOD <= set(ids), "a prune exemption names no case"
