"""No ``gsplat`` command may publish a score measured on different splats.

The domain half of #1600 lives in
``gsplats/tests/test_content_scoped_metrics.py``; what needs the command is the
end-to-end claim: that the scrub survives the WRITE, on both writers. Some
commands save through ``GSplatData.save`` and others through
``write_gsplats_tree``, so a fix wired into one of them would leave the other
silently exempt — and reading a diff does not tell you which one a command uses.

Every registered ``gsplat`` subcommand — including the ``batch-fit`` GROUP's
sub-commands, one of which writes a partition with ``fitting_info`` — is
classified below, and ``test_every_gsplat_command_is_classified`` fails if a new
one appears in none of the lists: a command that rewrites a store has to state
whether it changes content rather than inheriting a pass by default.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_commands import app_gsplat
from luxar.gsplats._data.filtering import _CONTENT_SCOPED_STATS_KEYS
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node

#: Measured scores at non-default values, spelled out LITERALLY (see the domain
#: test for why deriving them from the implementation constant would make every
#: row vacuous). ``test_the_fixture_covers_the_implementation_constant`` there
#: pins the full set; this file needs only a representative, writable subset —
#: the per-sub-LOD ladder keys have no place in a flat fixture's root stats.
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
    "splats_near_fit_init_sigma_count": 12,
    "splats_near_fit_init_sigma_fraction": 0.3,
    "splats_near_relocation_init_sigma_count": 8,
    "splats_near_relocation_init_sigma_fraction": 0.2,
    "splats_near_sigma_min_count": 4,
    "splats_near_sigma_min_fraction": 0.1,
}

#: Provenance a rewrite must NOT take with it.
_DESCRIPTIVE: Dict[str, Any] = {
    "fitter_name": "probe",
    "iterations": 123,
    "converged": True,
}

#: Region-scoped stamps, so the non-spatial rows can assert the OTHER category is
#: untouched (the two rules are independent — see the domain test).
_REGION: Dict[str, Any] = {"source_shape": [100, 100, 100], "occupancy": 0.01}

#: The record of the reduction that produced the fixture — content-scoped too: it
#: is true of that cull and false of any rewrite downstream (``decimate`` used to
#: republish this ``amplitude_retention`` on a prefix reduction of its own).
_OP_RECORD: Dict[str, Any] = {
    "culled": True,
    "culling_method": "cumulative",
    "n_original": 999,
    "n_culled": 959,
    "amplitude_retention": 0.95,
}


def test_the_fixture_metrics_are_really_content_scoped() -> None:
    """Anchor: every key this file asserts on must be in the scrubbed set.

    Without this, deleting a key from ``_CONTENT_SCOPED_STATS_KEYS`` would leave
    the "survivors" list empty (it iterates the constant) and every row green.
    """
    assert set(_METRICS) <= set(_CONTENT_SCOPED_STATS_KEYS), (
        f"not content-scoped: {sorted(set(_METRICS) - set(_CONTENT_SCOPED_STATS_KEYS))}"
    )
    for key in ("psnr_db", "final_loss", "ssim", "foreground_psnr_db"):
        assert key in _CONTENT_SCOPED_STATS_KEYS
    assert not set(_DESCRIPTIVE) & set(_CONTENT_SCOPED_STATS_KEYS)


def _chol(n: int) -> np.ndarray:
    return np.tile(np.array([2, 0, 2, 0, 0, 2], dtype=np.float32), (n, 1))


def _fixture(path: Path, n: int = 40) -> Path:
    """A tiny fitted store carrying the full fit stamp."""
    rng = np.random.default_rng(0)
    GSplatData(
        centers=(rng.random((n, 3)) * 100).astype(np.float32),
        amplitudes=np.linspace(1.0, 0.1, n).astype(np.float32),
        cholesky_factors=_chol(n),
        stats={
            **_METRICS,
            **_DESCRIPTIVE,
            **_REGION,
            **_OP_RECORD,
            "n_splats": n,
        },
    ).save(path, include_fitting_info=True)
    return path


def _partition_fixture(path: Path, n: int = 24) -> Path:
    """A ``kind=partition`` store that publishes the fit stamp.

    ``gsplat partition`` now threads it through too, but this builds one directly
    so the tree-writer test does not depend on that command's behaviour: a
    ``batch-fit merge`` partition carries one, and that is the shape ``gsplat
    transform`` handles leaf-by-leaf through ``write_gsplats_tree`` instead of
    ``GSplatData.save``.
    """
    from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree

    rng = np.random.default_rng(1)
    data = GSplatData(
        centers=(rng.random((n, 3)) * 100).astype(np.float32),
        amplitudes=np.linspace(1.0, 0.1, n).astype(np.float32),
        cholesky_factors=_chol(n),
        stats={**_METRICS, **_DESCRIPTIVE, "n_splats": n},
    )
    fitting, config, provenance, pipeline = split_fitting_info(
        dict(data.stats), include_fitting_info=True
    )
    write_gsplats_tree(
        path,
        data.to_spatial_partition(max_elements=8),
        fitting_info=fitting,
        fitting_config=config,
        provenance_info=provenance,
        pipeline_info=pipeline,
    )
    return path


def _root_stats(path: Path) -> Dict[str, Any]:
    _, stats = load_gsplat_node(path, include_stats=True)
    return dict(stats or {})


def _sublod_stats(path: Path) -> List[Dict[str, Any]]:
    node, _ = load_gsplat_node(path, include_stats=True)
    sublods = getattr(node, "additive_sublods", [])
    return [dict(lod.stats) for lod in sublods]


def _all_leaf_stats(path: Path) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Per leaf, its ``level_stats`` and every rung's ``lod_stats``, off disk."""
    from luxar.gsplats.tree import iter_leaves

    node, _ = load_gsplat_node(path, include_stats=True)
    return [
        (
            dict(leaf.meta.get("stats") or {}),
            [dict(sub.stats) for sub in leaf.additive_sublods],
        )
        for leaf in iter_leaves(node)
    ]


#: ``(id, argv-after-input, changes_content)``. ``{out}`` is substituted with the
#: output path; a command with no ``{out}`` rewrites the input in place.
_REWRITE_CASES: List[Tuple[str, Sequence[str], bool]] = [
    ("cull", ("cull", "{in}", "{out}", "-m", "cumulative", "-r", "0.5"), True),
    ("filter", ("filter", "{in}", "{out}", "--amplitude-min", "0.5"), True),
    # The negative control on the same command: a threshold that removes nothing
    # leaves an artifact the score still describes.
    ("filter-noop", ("filter", "{in}", "{out}", "--amplitude-min", "0.0"), False),
    ("slice", ("slice", "{in}", "{out}", "0:50,:,:"), True),
    ("decimate", ("decimate", "{in}", "{out}", "--target", "20"), True),
    # Geometry only — same splats, moved frame (see the domain test's reasoning).
    ("transform-geometry", ("transform", "{in}", "{out}", "--scale", "2,2,2"), False),
    # ... but an intensity edit rewrites the amplitudes PSNR was measured on.
    (
        "transform-intensity",
        ("transform", "{in}", "{out}", "--scale-intensity", "0.5"),
        True,
    ),
    # ...and an intensity flag that changes NOTHING keeps them: the scrub is
    # gated on the values, not on the flag being present. (The tree-writer twin
    # of this row is the case that used to diverge — see
    # test_the_tree_writer_is_held_to_the_same_rule.)
    (
        "transform-intensity-noop",
        ("transform", "{in}", "{out}", "--scale-intensity", "1.0"),
        False,
    ),
    ("flatten", ("flatten", "{in}", "{out}"), False),
    ("additive", ("additive", "{in}", "{out}", "--n-lods", "2"), False),
    # The same splats, regrouped spatially — and it publishes the stamp now.
    ("partition", ("partition", "{in}", "{out}", "--parts", "2"), False),
    # `-e precision` is exact. `-e auto` / `-e memory` re-quantize the Cholesky
    # factors (and, for those two modes, the centers) to fixed point, so they are
    # a deliberate bounded loss rather than a no-op — the spec says the scores are
    # kept as valid to well within their own precision, and this row does not
    # claim to cover them.
    ("reencode", ("reencode", "{in}", "{out}", "-e", "precision"), False),
    ("annotate-quality", ("annotate-quality", "{in}"), False),
    # A ladder whose finest level IS the input content (see the `overview` test
    # for the recipes where that is NOT true).
    (
        "lod-stream",
        ("lod", "{in}", "{out}", "--recipe", "stream", "--n-lods", "2"),
        False,
    ),
]

#: Rewriting commands that publish NO fit provenance at all today, so there is
#: nothing for them to carry over. Verified by ``test_no_provenance_commands``
#: rather than asserted from the source, because that is exactly the kind of
#: claim that rots.
_NO_PROVENANCE: Dict[str, Sequence[str]] = {
    # `concatenate` builds a fresh stats dict; the CLI also loads with
    # include_stats=False.
    "merge": ("merge", "{in}", "{in}", "-o", "{out}"),
}

#: Commands that never rewrite a ``.gsplats.zarr`` from another one, with the
#: reason. A new command MUST be added here or above, or the guard test fails.
_NON_REWRITE: Dict[str, str] = {
    "fit": "produces the fit (and the scores) rather than inheriting them",
    "cal": "writes a calibration JSON",
    "benchmark": "GPU profile, no dataset output",
    "compare": "measures, writes no store",
    "render": "writes a volume",
    "info": "read-only",
    "view": "read-only (serves a viewer)",
    "napari": "read-only",
    "doctor": "read-only diagnostics",
    "convert": "writes a .luxar.zarr scene, not a .gsplats.zarr",
    "export": "writes a classical PLY",
    "import": "reads a foreign file that has no Luxar fit stats",
    "denoise": "operates on a volume, not on splats",
    # NOT "content-preserving by definition": `auto`/`memory` re-quantize the
    # Cholesky factors and the centers to fixed point. The loss is deliberate and
    # bounded (~93 dB), so the scores are kept — but the layout migration is the
    # command's job and it takes no dataset to inherit FROM.
    "migrate-format": "legacy → current layout of ONE store; nothing is inherited "
    "from another artifact (its re-quantization is a bounded, documented loss)",
    # ── batch-fit group ──
    "batch-fit run": "produces the fit (local multi-GPU)",
    "batch-fit submit": "produces the fit (Slurm array)",
    "batch-fit merge": "assembles the fit's own tiles; the scores it writes are "
    "measured at merge time, not inherited from a published artifact",
    "batch-fit status": "read-only",
    "batch-fit validate": "read-only (or deletes corrupt tiles with --fix)",
    "batch-fit cancel": "cancels Slurm jobs",
    "batch-fit denoise-calibrate": "operates on a volume, writes JSON",
    "batch-fit denoise-preprocess": "operates on a volume",
    "batch-fit resolve-floor": "resolves a floor level, writes JSON/manifest",
}


def _run(argv: Sequence[str], src: Path, out: Path) -> None:
    resolved = [a.format(**{"in": str(src), "out": str(out)}) for a in argv]
    result = CliRunner().invoke(app, ["gsplat", *resolved])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "argv,changes_content",
    [(argv, ch) for _, argv, ch in _REWRITE_CASES],
    ids=[case[0] for case in _REWRITE_CASES],
)
def test_stored_scores_follow_the_content(
    tmp_path: Path, argv: Sequence[str], changes_content: bool
) -> None:
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "out.gsplats.zarr"
    _run(argv, src, out)
    written = out if "{out}" in argv else src

    stats = _root_stats(written)
    survivors = [k for k in _CONTENT_SCOPED_STATS_KEYS if k in stats]
    per_lod = [
        [k for k in _CONTENT_SCOPED_STATS_KEYS if k in lod]
        for lod in _sublod_stats(written)
    ]
    if changes_content:
        assert not survivors, f"stale scores on disk: {survivors}"
        assert not any(per_lod), f"stale scores in the leaf's lod_stats: {per_lod}"
    else:
        for key, value in _METRICS.items():
            assert stats.get(key) == pytest.approx(value), (
                f"{key!r} was dropped by a content-preserving command"
            )
    # The descriptive provenance survives EITHER WAY — the scrub takes the
    # measurement, not the record that a fit happened.
    for key, value in _DESCRIPTIVE.items():
        assert stats.get(key) == value, f"{key!r} was dropped"


def test_a_culled_store_no_longer_reports_the_fits_psnr(tmp_path: Path) -> None:
    """The user-visible symptom from the issue, through ``gsplat info``.

    The culling provenance is what should be there instead — silence about a
    score nobody measured, not silence about the cull.
    """
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "culled.gsplats.zarr"
    _run(("cull", "{in}", "{out}", "-m", "cumulative", "-r", "0.5"), src, out)

    result = CliRunner().invoke(app, ["gsplat", "info", str(out), "--no-histograms"])
    assert result.exit_code == 0, result.output
    assert "psnr_db" not in result.output, result.output
    assert "44.4587" not in result.output, result.output
    assert "final_loss" not in result.output, result.output
    assert "culling_method: cumulative" in result.output, result.output
    assert "n_original: 40" in result.output, result.output

    # The control: before the cull, `info` DOES quote the fit's score, so the
    # assertions above are not passing because `info` never prints one.
    before = CliRunner().invoke(app, ["gsplat", "info", str(src), "--no-histograms"])
    assert before.exit_code == 0, before.output
    assert "psnr_db: 44.458700" in before.output, before.output


def test_decimate_keeps_the_provenance_and_drops_the_score(tmp_path: Path) -> None:
    """``decimate`` used to publish NO provenance at all, so its row was vacuous.

    It loaded with ``include_stats=False`` and called ``write_gsplats_tree``
    without ``fitting_info=``, which dropped even the descriptive keys the rule
    never touches — and left the domain-level scrub nothing to scrub.
    """
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "dec.gsplats.zarr"
    _run(("decimate", "{in}", "{out}", "--target", "20"), src, out)

    stats = _root_stats(out)
    assert stats, "decimate published no provenance at all"
    for key, value in _DESCRIPTIVE.items():
        assert stats.get(key) == value, f"decimate dropped the descriptive {key!r}"
    assert not [k for k in _CONTENT_SCOPED_STATS_KEYS if k in stats], (
        f"decimate published a score for the pre-reduction splats: {stats}"
    )
    # A reduction is not a spatial restriction, so the REGION stamps stay: the two
    # rules are independent, and only the measured half applies here.
    for key, value in _REGION.items():
        assert stats.get(key) == value, f"decimate dropped the region stamp {key!r}"
    # ...but the INPUT's cull record must not ride along either: threading the
    # provenance through carried `culled: True` with `amplitude_retention: 0.95`
    # onto a prefix reduction that had just discarded most of the amplitude mass,
    # with `n_original` / `n_culled` describing a removal that never happened here.
    for key in _OP_RECORD:
        assert key not in stats, (
            f"decimate published the input's {key!r} for a reduction it did not do"
        )


def test_a_laddered_store_recomputes_its_q_e_stamps_across_a_cull(
    tmp_path: Path,
) -> None:
    """The persisted e(k)/w pair and counts describe the culled artifact."""
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    laddered = tmp_path / "lad.gsplats.zarr"
    _run(("lod", "{in}", "{out}", "--recipe", "stream", "--n-lods", "3"), src, laddered)

    before = _all_leaf_stats(laddered)
    assert before[0][0]["reference_energy"] > 0, "the fixture carries no w"
    assert 0.0 < before[0][1][0]["energy_fraction_cum"] < 1.0, (
        "the fixture's first rung is already complete; nothing is being tested"
    )

    culled = tmp_path / "culled.gsplats.zarr"
    # 0.9, not 0.5: a harder cull empties a whole rung and the writer refuses an
    # empty splat set (a separate, pre-existing limitation).
    _run(("cull", "{in}", "{out}", "-m", "cumulative", "-r", "0.9"), laddered, culled)

    after = _all_leaf_stats(culled)
    assert len(after) == len(before), "the cull changed the tree shape"
    for i, (level_stats, rungs) in enumerate(after):
        assert len(rungs) == len(before[i][1]), f"leaf {i} changed ladder shape"
        cumulative_n = 0
        for j, rung in enumerate(rungs):
            cumulative_n += int(rung["lod_n_splats"])
            assert rung["lod_cumulative_n"] == cumulative_n
            assert 0.0 < rung["energy_fraction_cum"] <= 1.0
            # ...while the fit's MEASURED scores (this rule's actual subject) go.
            survivors = [k for k in _CONTENT_SCOPED_STATS_KEYS if k in rung]
            assert not survivors, f"leaf {i} rung {j} kept {survivors}"
        assert rungs[-1]["energy_fraction_cum"] == pytest.approx(1.0)
        assert level_stats["n_splats_total"] == cumulative_n
        assert level_stats["reference_energy"] > 0.0
        assert level_stats["reference_energy"] != pytest.approx(
            before[i][0]["reference_energy"]
        )
    assert not [k for k in _CONTENT_SCOPED_STATS_KEYS if k in _root_stats(culled)], (
        "the culled store still publishes the fit's score at the root"
    )


def test_overview_does_not_stamp_the_input_score_on_its_merged_cap(
    tmp_path: Path,
) -> None:
    """``lod --recipe overview``'s coarse cap is MERGED representatives.

    The recipe builds it as ``capped.at_substitutive(n - 1).flattened().tree``,
    and the writer persists that view's stats as the leaf's ``lod_stats`` — so the
    cap published the input fit's ``psnr_db`` for a splat set made of merged
    representatives. The ROOT keeps the score: the group's finest content IS the
    input, which is what the root ``fitting/`` group describes.
    """
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "ov.gsplats.zarr"
    _run(
        (
            "lod",
            "{in}",
            "{out}",
            "--recipe",
            "overview",
            "--compression-factor",
            "4",
            "--max-elements",
            "12",
        ),
        src,
        out,
    )
    assert _root_stats(out).get("psnr_db") == pytest.approx(_METRICS["psnr_db"]), (
        "the root lost the score of the content it still holds"
    )
    leaves = _all_leaf_stats(out)
    assert len(leaves) > 1, "no fine partition was built; the test proves nothing"
    caps = [lv for lv, _r in leaves if lv.get("quality", 1.0) < 1.0]
    assert caps, "no merged coarse cap in the output (quality < 1); nothing is proven"
    for i, (_level_stats, rungs) in enumerate(leaves):
        for j, rung in enumerate(rungs):
            survivors = [k for k in _CONTENT_SCOPED_STATS_KEYS if k in rung]
            assert not survivors, (
                f"leaf {i} rung {j} publishes the input fit's {survivors}"
            )

    transformed = tmp_path / "ov_scaled.gsplats.zarr"
    _run(("transform", "{in}", "{out}", "--scale-intensity", "0.5"), out, transformed)
    before_node, _ = load_gsplat_node(out, include_stats=True)
    node, _ = load_gsplat_node(transformed, include_stats=True)
    from luxar.gsplats.tree import (
        GSplatLeaf,
        GSplatLodGroup,
        GSplatPartition,
        iter_leaves,
    )

    assert isinstance(before_node, GSplatLodGroup)
    assert isinstance(node, GSplatLodGroup)
    before_caps = [
        child for child in before_node.children if isinstance(child, GSplatLeaf)
    ]
    caps = [child for child in node.children if isinstance(child, GSplatLeaf)]
    assert len(before_caps) == len(caps) == 1, (
        "overview output has no unique coarse cap"
    )
    before_reference = before_caps[0].meta["stats"]["reference_energy"]
    cap_reference = caps[0].meta["stats"]["reference_energy"]
    assert cap_reference == pytest.approx(before_reference * 0.25, rel=0.02)
    partitions = [
        child for child in node.children if isinstance(child, GSplatPartition)
    ]
    assert partitions, "overview output has no fine partition child"
    fine_references = [
        leaf.meta["stats"]["reference_energy"]
        for partition in partitions
        for leaf in iter_leaves(partition)
    ]
    assert cap_reference == pytest.approx(sum(fine_references))
    assert all(
        "quality" not in leaf.meta.get("stats", {}) for leaf in iter_leaves(node)
    )
    for partition in partitions:
        assert "stats" not in partition.meta, (
            "tree restamp invented level_stats on a non-leaf partition child"
        )


def test_tree_intensity_transform_preserves_substitutive_provenance(
    tmp_path: Path,
) -> None:
    """A tree rewrite must not replace authored level provenance with defaults."""
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    adaptive = tmp_path / "adaptive.gsplats.zarr"
    _run(
        (
            "lod",
            "{in}",
            "{out}",
            "--recipe",
            "adaptive",
            "--compression-factor",
            "4",
            "--levels",
            "3",
            "--max-elements",
            "12",
            "--no-additive",
            "--device",
            "cpu",
        ),
        src,
        adaptive,
    )

    from luxar.gsplats.tree import iter_leaves

    before_node, _ = load_gsplat_node(adaptive, include_stats=True)
    before = [
        (
            leaf.meta.get("compression_factor"),
            leaf.meta.get("parent_method"),
            leaf.meta.get("level_index"),
        )
        for leaf in iter_leaves(before_node)
    ]
    assert any(
        compression_factor != 1 or parent_method is not None or level_index != 0
        for compression_factor, parent_method, level_index in before
    ), "adaptive fixture has only default provenance; the test proves nothing"

    transformed = tmp_path / "adaptive_scaled.gsplats.zarr"
    _run(
        ("transform", "{in}", "{out}", "--scale-intensity", "0.5"),
        adaptive,
        transformed,
    )
    after_node, _ = load_gsplat_node(transformed, include_stats=True)
    after = [
        (
            leaf.meta.get("compression_factor"),
            leaf.meta.get("parent_method"),
            leaf.meta.get("level_index"),
        )
        for leaf in iter_leaves(after_node)
    ]
    assert after == before


@pytest.mark.parametrize("name", sorted(_NO_PROVENANCE))
def test_no_provenance_commands_publish_no_scores(tmp_path: Path, name: str) -> None:
    """These carry nothing over — checked, not assumed."""
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / f"{name}.gsplats.zarr"
    _run(_NO_PROVENANCE[name], src, out)
    stats = _root_stats(out)
    assert not [k for k in _CONTENT_SCOPED_STATS_KEYS if k in stats], (
        f"{name} DOES publish fit stats — it needs a row in _REWRITE_CASES"
    )


@pytest.mark.parametrize(
    "argv,changes_content",
    [
        pytest.param(
            ("transform", "{in}", "{out}", "--scale-intensity", "0.5"),
            True,
            id="intensity",
        ),
        # The case the two paths DIVERGED on: the tree path scrubbed on the flag
        # being present, so a no-op factor destroyed a partition's scores while
        # the flat path kept them.
        pytest.param(
            ("transform", "{in}", "{out}", "--scale-intensity", "1.0"),
            False,
            id="intensity-noop",
        ),
        # ...and `--normalize-intensity` on an ALREADY-normalized store, where the
        # old gate scrubbed without a single leaf op running.
        pytest.param(
            ("transform", "{in}", "{out}", "--normalize-intensity", "1.0"),
            False,
            id="normalize-noop",
        ),
        pytest.param(
            ("transform", "{in}", "{out}", "--scale", "2,2,2"), False, id="geometry"
        ),
        pytest.param(("flatten", "{in}", "{out}"), False, id="flatten"),
        pytest.param(
            ("additive", "{in}", "{out}", "--n-lods", "2"),
            False,
            id="additive",
        ),
    ],
)
def test_the_tree_writer_is_held_to_the_same_rule(
    tmp_path: Path, argv: Sequence[str], changes_content: bool
) -> None:
    """The other writer: a partition goes out through ``write_gsplats_tree``.

    The flat path scrubs via ``GSplatData``; this one assembles the root
    ``fitting/`` group from the stats it loaded off disk, so a fix wired only into
    the dataset would leave every tree-shaped store exempt — and a fix wired with
    a DIFFERENT predicate leaves the two paths disagreeing about the same command.
    """
    src = _partition_fixture(tmp_path / "part.gsplats.zarr")
    assert _root_stats(src).get("psnr_db") == pytest.approx(_METRICS["psnr_db"]), (
        "the fixture published no score — the test would prove nothing"
    )
    out = tmp_path / "out.gsplats.zarr"
    _run(argv, src, out)

    stats = _root_stats(out)
    survivors = [k for k in _CONTENT_SCOPED_STATS_KEYS if k in stats]
    if changes_content:
        assert not survivors, f"stale scores survived the tree writer: {survivors}"
    else:
        assert stats.get("psnr_db") == pytest.approx(_METRICS["psnr_db"])
    assert stats.get("fitter_name") == "probe"


def test_partition_root_score_survives_flatten_then_decimate_scrubs_it(
    tmp_path: Path,
) -> None:
    """The supported partition→matrix rewrite chain applies both rules."""
    src = _partition_fixture(tmp_path / "part.gsplats.zarr")
    flat = tmp_path / "flat.gsplats.zarr"
    decimated = tmp_path / "decimated.gsplats.zarr"

    _run(("flatten", "{in}", "{out}"), src, flat)
    flat_stats = _root_stats(flat)
    assert flat_stats.get("psnr_db") == pytest.approx(_METRICS["psnr_db"])

    _run(("decimate", "{in}", "{out}", "--target", "20"), flat, decimated)
    decimated_stats = _root_stats(decimated)
    assert not [key for key in _CONTENT_SCOPED_STATS_KEYS if key in decimated_stats], (
        f"decimate published a score for the pre-reduction splats: {decimated_stats}"
    )
    assert decimated_stats.get("fitter_name") == "probe"
    assert decimated_stats["n_splats"] == 20


def test_the_flat_and_tree_paths_agree_on_a_no_op_intensity(tmp_path: Path) -> None:
    """One store, two shapes, one command — the same answer.

    The measured divergence: ``--scale-intensity 1.0`` kept ``psnr_db`` on a flat
    store and deleted it on a partition, so the same store had two answers
    depending on its shape (and the partition's leaves kept their own
    ``lod_stats`` intact beside a scrubbed root).
    """
    flat_src = _fixture(tmp_path / "flat.gsplats.zarr")
    part_src = _partition_fixture(tmp_path / "part.gsplats.zarr")
    flat_out = tmp_path / "flat_out.gsplats.zarr"
    part_out = tmp_path / "part_out.gsplats.zarr"
    _run(("transform", "{in}", "{out}", "--scale-intensity", "1.0"), flat_src, flat_out)
    _run(("transform", "{in}", "{out}", "--scale-intensity", "1.0"), part_src, part_out)

    flat_kept = [k for k in _CONTENT_SCOPED_STATS_KEYS if k in _root_stats(flat_out)]
    part_kept = [k for k in _CONTENT_SCOPED_STATS_KEYS if k in _root_stats(part_out)]
    assert flat_kept == part_kept, (
        f"flat kept {flat_kept} but the partition kept {part_kept}"
    )
    assert "psnr_db" in flat_kept, "a no-op intensity edit destroyed true information"


def test_annotate_quality_rewrites_the_input_in_place(tmp_path: Path) -> None:
    """Guards the in-place row above: it must really have no ``{out}``."""
    src = _fixture(tmp_path / "fit.gsplats.zarr")
    copy = tmp_path / "copy.gsplats.zarr"
    shutil.copytree(src, copy)
    _run(("annotate-quality", "{in}"), copy, tmp_path / "unused")
    stats = _root_stats(copy)
    assert stats.get("psnr_db") == pytest.approx(_METRICS["psnr_db"])


def _registered_gsplat_commands() -> "set[str]":
    """Every invocable ``gsplat`` command name, groups included.

    ``registered_commands`` alone misses the ``batch-fit`` GROUP entirely — and
    ``batch-fit merge`` writes a partition through ``write_gsplats_tree`` with
    ``fitting_info=``, i.e. exactly the shape this file is about.
    """
    import typer

    def _name(cmd: Any) -> str:
        return cmd.name or (cmd.callback.__name__ if cmd.callback else "?")

    def walk(sub: "typer.Typer", prefix: str) -> "set[str]":
        found = {f"{prefix}{_name(c)}" for c in sub.registered_commands}
        for grp in sub.registered_groups:
            gname = grp.name or (
                grp.typer_instance.info.name if grp.typer_instance else "?"
            )
            assert grp.typer_instance is not None
            found |= walk(grp.typer_instance, f"{prefix}{gname} ")
        return found

    return walk(app_gsplat, "")


def test_every_gsplat_command_is_classified() -> None:
    """A new command cannot skip the check by not being in the table."""
    registered = _registered_gsplat_commands()
    assert "cull" in registered and "batch-fit merge" in registered, (
        f"the enumeration missed something obvious: {sorted(registered)}"
    )
    # Read the command name off the argv (its first element), not off the case
    # id: several commands appear twice under a variant id ("filter-noop",
    # "transform-intensity") and a name parsed out of the id would drift.
    classified = (
        {argv[0] for _, argv, _ in _REWRITE_CASES}
        | {argv[0] for argv in _NO_PROVENANCE.values()}
        | set(_NON_REWRITE)
    )
    assert not registered - classified, (
        "unclassified gsplat command(s) — add a row to _REWRITE_CASES (stating "
        f"whether they change content) or to _NON_REWRITE: {registered - classified}"
    )
    assert not classified - registered, (
        f"the lists name commands that no longer exist: {classified - registered}"
    )
    assert all(_NON_REWRITE.values()), "every _NON_REWRITE entry needs a reason"
