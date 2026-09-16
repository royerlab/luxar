"""Unit tests for the content-aware fit planner (scan + BSP, CPU-only)."""

from __future__ import annotations

import importlib
import math
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.calibration import SplatDensity
from luxar.gsplats.planner import (
    CONTENT_CULL_RETENTION,
    FitPlan,
    PlanBox,
    fit_planned_parallel,
    plan_partition,
    plan_volume,
    scan_content,
)
from luxar.gsplats.planner.fit_planned_parallel import max_padded_box_voxels


def _corner_blobs(shape=(128, 128, 128), n=16, corner=64, seed=0):
    """Volume with n separated blobs packed into the [:corner]^3 corner."""
    V = np.zeros(shape, np.float32)
    zz, yy, xx = np.mgrid[0:corner, 0:corner, 0:corner]
    rng = np.random.default_rng(seed)
    for i in range(n):
        cz, cy, cx = rng.integers(4, corner - 4, 3)
        V[:corner, :corner, :corner] += np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)
        ).astype(np.float32)
    return np.clip(V, 0, 1)


def _density(**kw):
    base = dict(
        feature_method="peaks",
        n_features_reference=8,
        k_star_reference=20000,
        saturation_exponent=0.44,
        saturation_cap=40000,
        splats_per_feature=2500.0,
    )
    base.update(kw)
    return SplatDensity(**base)


class TestScanContent:
    def test_field_shape_and_total(self):
        V = _corner_blobs()
        f = scan_content(V, cell=16, method="peaks")
        assert f.density.shape == (8, 8, 8)  # 128/16
        assert f.total > 0

    def test_box_weight_matches_whole_field(self):
        V = _corner_blobs()
        f = scan_content(V, cell=16, method="peaks")
        whole = f.box_weight(0, 128, 0, 128, 0, 128)
        assert whole == pytest.approx(f.total, rel=1e-6)

    def test_content_concentrated_in_corner(self):
        V = _corner_blobs()
        f = scan_content(V, cell=16, method="peaks")
        corner = f.box_weight(0, 64, 0, 64, 0, 64)
        far = f.box_weight(64, 128, 64, 128, 64, 128)
        assert corner > far  # features are in the corner

    def test_rejects_non_3d(self):
        with pytest.raises(ValueError):
            scan_content(np.zeros((10, 10), np.float32))

    def test_total_consistent_with_count_features(self):
        # CRITICAL: the planner's feature total must match calibration.count_features
        # (same detector/params) or the splats-per-feature density miscalibrates.
        from luxar.gsplats.calibration import count_features

        V = _corner_blobs()
        scan_total = scan_content(V, cell=16, method="peaks").total
        cf = count_features(V, method="peaks")
        assert abs(scan_total - cf) <= max(2, 0.1 * cf)  # within ~10%

    def test_threshold_abs_robust_to_outlier(self):
        # A hot outlier inflates the global max, so the default global-relative
        # threshold undercounts; an explicit absolute threshold (the level the
        # calibration counted at) recovers the true feature scale. This is the
        # composability fix for the cal->planner budget transfer.
        V = _corner_blobs((128, 128, 128), n=16)
        V[0, 0, 0] = 100.0  # outlier far above blob intensity (~1)
        default = scan_content(V, cell=16, method="peaks").total
        robust = scan_content(V, cell=16, method="peaks", threshold_abs=0.1).total
        assert robust > default  # absolute level is not suppressed by the outlier

    def test_scan_matches_count_features_at_recorded_threshold(self):
        # C1/H5 regression: scanning at the EXACT level the calibration recorded
        # (feature_threshold = 0.1*blurred_max) reproduces count_features even with
        # a hot outlier present — the cal->planner counts are on one scale.
        from luxar.gsplats.calibration import count_features, feature_threshold

        V = _corner_blobs((128, 128, 128), n=16)
        V[0, 0, 0] = 100.0  # hot outlier
        thr = feature_threshold(V, "peaks")
        scan = scan_content(V, cell=16, method="peaks", threshold_abs=thr).total
        cf = count_features(V, method="peaks")
        assert abs(scan - cf) <= max(2, 0.1 * cf)

    def test_intensity_scan_reconstructs_otsu_not_relative(self):
        # scan_content("intensity") with no threshold_abs must reconstruct the Otsu
        # cut (calibration's single source of truth), NOT 0.1*max — else a dim
        # population between 0.1*max and Otsu is miscounted, mis-scaling box budgets.
        from luxar.gsplats.calibration import _otsu_threshold, count_features

        V = np.zeros((48, 48, 48), np.float32)
        V[:8] = 0.2  # dim: above 0.1*max(=0.08) but below the Otsu cut
        V[40:] = 0.8  # bright
        otsu = _otsu_threshold(V)
        assert 0.2 < otsu < 0.8  # Otsu separates dim from bright
        cf = count_features(V, method="intensity")  # count(V > otsu): bright only
        scan = scan_content(V, cell=16, method="intensity").total
        assert scan == cf  # pre-fix scanned at 0.1*max and also counted the dim slab
        assert cf == int(np.count_nonzero(V > otsu))

    def test_intensity_strict_gt_matches_calibration(self):
        # foreground_mask_otsu uses strict '>'; scan_content intensity must match,
        # so voxels exactly AT the threshold are excluded (pre-fix used '>=').
        from luxar.gsplats.calibration import count_features

        V = np.zeros((32, 32, 32), np.float32)
        V[:16] = 5.0  # exactly at the threshold -> must be EXCLUDED
        V[16:] = 10.0  # above -> included
        thr = 5.0
        cf = count_features(V, method="intensity", threshold_abs=thr)
        scan = scan_content(V, cell=16, method="intensity", threshold_abs=thr).total
        assert scan == cf
        assert cf == int(np.count_nonzero(V > thr))  # the 10.0 slab only, once

    def test_no_double_count_at_slab_boundary_downsample(self):
        # zlen off-by-one double-counted a feature sitting at a strided slab
        # boundary when downsample>1; it must be counted exactly once.
        Z = 132  # spans the _SLAB=64 boundaries at 64 and 128
        V = np.zeros((Z, 16, 16), np.float32)
        V[129, 9, 9] = 1.0  # z=129 (129%3==0) is a strided sample at the boundary
        total = scan_content(
            V, cell=16, method="intensity", downsample=3, threshold_abs=0.5
        ).total
        assert total == 1.0  # pre-fix counted it in both adjacent slabs (== 2.0)


class TestSpecFitPlan:
    def test_overlap_fraction_two_sided_halo(self):
        # M7: boxes GROW by 2*overlap per axis at fit time; overhead is
        # 1 - (L/(L+2*overlap))^3, not the old (L-overlap)/L model.
        from luxar.gsplats.planner import FitPlan, PlanBox

        plan = FitPlan(
            volume_shape=[256, 256, 256],
            boxes=[PlanBox(box=[0, 256, 0, 256, 0, 256], n_features=100, budget=1000)],
            overlap=32,
            feature_method="peaks",
            min_leaf=256,
            max_leaf=512,
        )
        med, mx = plan.overlap_fraction()
        expected = 1.0 - (256 / (256 + 64)) ** 3  # ~0.488
        assert math.isclose(med, expected, rel_tol=1e-6)
        assert math.isclose(mx, expected, rel_tol=1e-6)


class TestFitPlanned:
    def test_fit_planned_cpu_smoke(self):
        # End-to-end: plan a small volume then fit it on CPU; expect splats back.
        from luxar.gsplats.planner import fit_planned

        V = _corner_blobs((64, 64, 64), n=6, corner=48)
        plan = plan_volume(V, _density(), cell=8, min_leaf=16, max_leaf=32, overlap=4)
        merged = fit_planned(
            V,
            plan,
            device="cpu",
            n_iters=30,
            early_stop_patience=30,
            use_cuda=False,
            use_metal=False,
        )
        assert merged.n_splats > 0
        c = merged.centers
        # all splats fall inside the volume bounds
        assert c[:, 0].min() >= 0 and c[:, 0].max() < 64
        assert c[:, 2].min() >= 0 and c[:, 2].max() < 64

    def test_fit_planned_scales_centers_and_box_geometry(self, monkeypatch):
        """Content boxes keep/filter in physical coordinates when requested."""
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.planner.fit_planned import _fit_one_box
        from luxar.gsplats.planner.spec import PlanBox

        def fake_fit(sub, **kwargs):
            assert sub.shape == (12, 12, 12)
            assert kwargs["voxel_size"] == (5.0, 2.0, 1.0)
            return GSplatData(
                centers=np.array([[10.0, 8.0, 4.0]], dtype=np.float32),
                amplitudes=np.ones(1, dtype=np.float32),
                cholesky_factors=np.ones((1, 6), dtype=np.float32),
            )

        monkeypatch.setattr("luxar.gsplats.fit_gsplats.fit_gaussian_splats", fake_fit)

        result = _fit_one_box(
            np.ones((24, 24, 24), dtype=np.float32),
            PlanBox(box=(8, 16, 8, 16, 8, 16), n_features=10, budget=10),
            overlap=2,
            cap=100,
            content_physical=True,
            voxel_size=(5.0, 2.0, 1.0),
            output_space="real",
        )

        np.testing.assert_allclose(result.centers, [[40.0, 20.0, 10.0]])

    def test_fit_planned_keeps_config_only_content_in_voxel_space(self, monkeypatch):
        """Physical content geometry remains opt-in rather than a config side effect."""
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.planner.fit_planned import _fit_one_box
        from luxar.gsplats.planner.spec import PlanBox

        def fake_fit(sub, **kwargs):
            assert "voxel_size" not in kwargs
            return GSplatData(
                centers=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
                amplitudes=np.ones(1, dtype=np.float32),
                cholesky_factors=np.ones((1, 6), dtype=np.float32),
            )

        monkeypatch.setattr("luxar.gsplats.fit_gsplats.fit_gaussian_splats", fake_fit)
        with pytest.warns(UserWarning, match="fitting in voxel space"):
            result = _fit_one_box(
                np.ones((24, 24, 24), dtype=np.float32),
                PlanBox(box=(4, 12, 4, 12, 4, 12), n_features=10, budget=10),
                overlap=0,
                cap=100,
                voxel_size=(5.0, 2.0, 1.0),
                output_space="real",
            )

        np.testing.assert_allclose(result.centers, [[8.0, 8.0, 8.0]])

    @pytest.mark.parametrize("partition", [False, True])
    def test_scoring_receives_the_merged_box_basis(self, monkeypatch, partition):
        from luxar.gsplats.gsplat_data import GSplatData

        fit_planned_module = importlib.import_module(
            "luxar.gsplats.planner.fit_planned"
        )
        captured = []

        def fake_fit_one_box(volume, box, pad, cap, **kwargs):
            center = np.array(
                [[(box.box[i] + box.box[i + 1]) / 2 for i in (0, 2, 4)]],
                np.float32,
            )
            return GSplatData(
                centers=center,
                amplitudes=np.ones(1, np.float32),
                cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], np.float32),
                stats={
                    "floor": 500.0,
                    "image_min": 500.0,
                    "image_max": 501.0,
                    "intensity_range": 1.0,
                },
            )

        def capture_score(*args, image_min, **kwargs):
            captured.append(image_min)

        monkeypatch.setattr(fit_planned_module, "_fit_one_box", fake_fit_one_box)
        monkeypatch.setattr(fit_planned_module, "_score_planned_merge", capture_score)
        plan = _toy_plan(n_boxes=2, budget=5)
        fit_planned_module.fit_planned(
            np.full(plan.volume_shape, 500.0, np.float32),
            plan,
            partition=partition,
            verbose=False,
        )

        assert captured == [500.0]


class TestPlanCliResolveDensity:
    def test_explicit_flags_build_density(self):
        from luxar.cli.gsplat_ops.planner import _resolve_density

        d = _resolve_density(None, 40000, 5000, 0.44, None, "peaks", 900.0)
        assert d.k_star_reference == 40000 and d.n_features_reference == 5000
        assert d.saturation_cap == 160000  # 4x default
        assert d.feature_threshold == 900.0
        assert d.feature_method == "peaks"

    def test_missing_flags_raise(self):
        import typer

        from luxar.cli.gsplat_ops.planner import _resolve_density

        with pytest.raises(typer.BadParameter):
            _resolve_density(None, None, None, 0.44, None, None, None)

    def test_none_metric_defaults_to_peaks(self):
        from luxar.cli.gsplat_ops.planner import _resolve_density

        d = _resolve_density(None, 1000, 100, 0.44, None, None, None)
        assert d.feature_method == "peaks"


class TestThresholdThreading:
    def test_plan_volume_uses_density_threshold(self):
        V = _corner_blobs((128, 128, 128), n=16)
        V[0, 0, 0] = 100.0  # outlier
        d = _density(feature_threshold=0.1)  # blob-scale absolute threshold
        plan = plan_volume(V, d, cell=16, min_leaf=32, max_leaf=64, overlap=8)
        # whole-volume features at the abs threshold match count_features at it
        total = sum(b.n_features for b in plan.boxes)
        assert total > 0
        # and budgets are non-trivial because features were not outlier-suppressed
        assert plan.total_budget > 0


class TestPlanPartition:
    def test_boxes_partition_volume_exactly(self):
        V = _corner_blobs()
        f = scan_content(V, cell=16, method="peaks")
        plan = plan_partition(f, _density(), min_leaf=32, max_leaf=64, overlap=8)
        assert sum(b.voxels for b in plan.boxes) == V.size  # exact tiling, no gaps

    def test_leaf_size_bounds(self):
        V = _corner_blobs()
        f = scan_content(V, cell=16, method="peaks")
        plan = plan_partition(f, _density(), min_leaf=32, max_leaf=64, overlap=8)
        for b in plan.boxes:
            # each leaf is either within max_leaf or was unsplittable (< 2*min_leaf)
            assert max(b.dims) <= 64 or max(b.dims) < 64
            assert min(b.dims) >= 32

    def test_budgets_track_content(self):
        V = _corner_blobs()
        f = scan_content(V, cell=16, method="peaks")
        plan = plan_partition(f, _density(), min_leaf=32, max_leaf=64, overlap=8)
        # the highest-budget box should be a content-rich (corner) box
        top = max(plan.boxes, key=lambda b: b.budget)
        assert top.n_features > 0
        assert top.box[0] < 64 and top.box[2] < 64 and top.box[4] < 64
        # empty boxes get ~0 budget
        empties = [b for b in plan.boxes if b.n_features == 0]
        if empties:
            assert all(b.budget == 0 for b in empties)

    def test_budget_capped_at_saturation(self):
        V = _corner_blobs(n=40)  # very dense corner
        f = scan_content(V, cell=16, method="peaks")
        d = _density(saturation_cap=15000)
        plan = plan_partition(f, d, min_leaf=64, max_leaf=128, overlap=8)
        assert all(b.budget <= 15000 for b in plan.boxes)

    def test_json_round_trip(self, tmp_path):
        V = _corner_blobs()
        f = scan_content(V, cell=16, method="peaks")
        plan = plan_partition(f, _density(), min_leaf=32, max_leaf=64, overlap=8)
        p = tmp_path / "plan.json"
        plan.to_json(p)
        rp = FitPlan.from_json(p)
        assert rp.n_boxes == plan.n_boxes
        assert rp.total_budget == plan.total_budget
        assert rp.boxes[0].box == plan.boxes[0].box

    def test_plan_volume_convenience(self):
        V = _corner_blobs()
        plan = plan_volume(V, _density(), cell=16, min_leaf=32, max_leaf=64, overlap=8)
        assert plan.n_boxes >= 1
        assert plan.total_budget > 0


# ── parallel box fitting (driver exercised with a fake per-box worker) ────────


def _toy_plan(n_boxes: int = 3, budget: int = 100, width: int = 16) -> FitPlan:
    """A FitPlan tiling x into ``n_boxes`` columns; no scan needed by the driver."""
    boxes = [
        PlanBox(
            box=[0, width, 0, width, j * width, (j + 1) * width],
            n_features=10,
            budget=budget,
        )
        for j in range(n_boxes)
    ]
    return FitPlan(
        volume_shape=[width, width, n_boxes * width],
        boxes=boxes,
        overlap=4,
        feature_method="peaks",
        min_leaf=8,
        max_leaf=16,
        density={"saturation_cap": 10_000},
    )


#: Per-box fit time the fake box worker records in its store's stats — far larger
#: than any (instant) merge of fake boxes can take, so a merge that reported the
#: SUM of the boxes' own times instead of wall clock is unmistakable.
_FAKE_BOX_TIME_SECONDS = 1000.0


def _fake_box_builder(
    n_per_box: int = 5,
    truncation_radius: float | None = None,
    image_min: float = 0.0,
):
    """Worker builder writing ``n_per_box`` deterministic splats — no torch/GPU.

    ``truncation_radius`` stands in for a box worker whose fit config asked for a
    non-default ``truncate`` (the real worker stamps it on the store it saves).
    Each box also records its own ``time_seconds`` fit stat, as a real box worker
    does.
    """
    radius = (
        ""
        if truncation_radius is None
        else f", truncation_radius={truncation_radius!r}"
    )

    def builder(i: int, out_path: Path) -> list[str]:
        script = textwrap.dedent(
            f"""
            import numpy as np
            from luxar.gsplats.gsplat_data import GSplatData
            rng = np.random.default_rng({i} + 1)
            k = {n_per_box}
            centers = rng.random((k, 3)).astype(np.float32) * 10.0
            amps = rng.random(k).astype(np.float32) + 0.1
            chol = np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (k, 1))
            GSplatData(centers=centers, amplitudes=amps,
                       cholesky_factors=chol{radius},
                       stats={{"time_seconds": {_FAKE_BOX_TIME_SECONDS!r},
                              "psnr_db": 42.0,
                              "source_shape": [16, 16, 16],
                              "source_voxels": 4096,
                              "floor": {image_min!r}, "image_min": {image_min!r},
                              "image_max": {image_min + 1.0!r},
                              "intensity_range": 1.0}},
                       ).save(r"{out_path}")
            """
        )
        return [sys.executable, "-c", script]

    return builder


def _empty_marker_for(*box_idxs: int, n_per_box: int = 5):
    """Builder where the given boxes write a 0-splat ``.empty`` marker instead."""
    ok = _fake_box_builder(n_per_box)

    def builder(i: int, out_path: Path) -> list[str]:
        if i in box_idxs:
            return [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path(r'{out_path}' + '.empty').write_text('')",
            ]
        return ok(i, out_path)

    return builder


class TestDefaultWorkerCmdBuilder:
    def test_forwards_resolved_floor_level(self, tmp_path):
        # The parent resolves the spec ONCE against the whole volume and hands
        # each box worker the concrete LEVEL (#1174) — a spec would be
        # re-estimated per box crop, so abutting boxes would subtract different
        # pedestals. The builder must carry a number through verbatim.
        from luxar.gsplats.planner.fit_planned_parallel import (
            _default_worker_cmd_builder,
        )

        b = _default_worker_cmd_builder("in.zarr", "plan.json", floor=4.25)
        cmd = [str(c) for c in b(0, tmp_path / "box0.gsplats.zarr")]
        assert "--floor" in cmd
        assert float(cmd[cmd.index("--floor") + 1]) == pytest.approx(4.25)
        # A disabled floor is forwarded explicitly (not omitted): the worker must
        # not fall back to its own 'auto' default.
        b_none = _default_worker_cmd_builder("in.zarr", "plan.json", floor="none")
        cmd_none = [str(c) for c in b_none(0, tmp_path / "box0.gsplats.zarr")]
        assert cmd_none[cmd_none.index("--floor") + 1] == "none"
        # A SPEC is still forwarded verbatim (documented contract): a worker
        # invoked by hand, or from a plan written before the level was resolved
        # up front, resolves it against its whole (t, c) volume — never the box
        # crop — so the deterministic sampler still makes the boxes agree.
        b_spec = _default_worker_cmd_builder("in.zarr", "plan.json", floor="p10")
        cmd_spec = [str(c) for c in b_spec(0, tmp_path / "box0.gsplats.zarr")]
        assert cmd_spec[cmd_spec.index("--floor") + 1] == "p10"
        # Nothing to forward -> no flag (the worker resolves its own config).
        b2 = _default_worker_cmd_builder("in.zarr", "plan.json")
        assert "--floor" not in [str(c) for c in b2(0, tmp_path / "box0.gsplats.zarr")]

    def test_forwards_shared_raw_normalization_range(self, tmp_path):
        from luxar.gsplats.planner.fit_planned_parallel import (
            _default_worker_cmd_builder,
        )

        builder = _default_worker_cmd_builder(
            "in.zarr", "plan.json", norm_range=(10.25, 999.5)
        )
        cmd = builder(0, tmp_path / "box0.gsplats.zarr")
        assert cmd[cmd.index("--norm-range") + 1] == "10.25,999.5"

    def test_declines_a_degenerate_shared_normalization_range(self):
        from luxar.gsplats.planner.fit_planned import _ensure_planned_norm_range

        fit_kwargs = {"norm_percentile": 1.0}
        volume = np.full((32, 32, 32), 100.0, np.float32)
        volume.flat[:100] = 200.0

        _ensure_planned_norm_range(volume, fit_kwargs, False)

        assert fit_kwargs.get("norm_range") is None

    def test_future_denoising_boxes_do_not_inherit_a_raw_shared_range(self):
        """Keep the #1813 forward guard from sharing a raw range."""
        from luxar.gsplats.planner.fit_planned import _ensure_planned_norm_range

        fit_kwargs = {"_denoise_h": 0.04, "_denoise_params": {}}
        volume = np.linspace(10.0, 110.0, 32**3, dtype=np.float32).reshape(32, 32, 32)

        _ensure_planned_norm_range(volume, fit_kwargs, False)

        assert "norm_range" not in fit_kwargs

    def test_forwards_the_runs_fit_configuration(self, tmp_path):
        # `truncate:` is settable ONLY through a YAML --config (no preset sets it,
        # there is no --truncate flag), so an unforwarded config made every `-j N`
        # box fit (and stamp) different parameters than `-j 1` (#1637).
        from luxar.gsplats.planner.fit_planned_parallel import (
            _default_worker_cmd_builder,
        )

        b = _default_worker_cmd_builder(
            "in.zarr",
            "plan.json",
            preset="draft",
            config=tmp_path / "fit.yaml",
            iters=7,
            loss="mse",
            lr=0.02,
            cull_retention=0.9,
        )
        cmd = [str(c) for c in b(0, tmp_path / "box0.gsplats.zarr")]
        assert cmd[cmd.index("--preset") + 1] == "draft"
        assert cmd[cmd.index("--config") + 1] == str(tmp_path / "fit.yaml")
        assert cmd[cmd.index("--iters") + 1] == "7"
        assert cmd[cmd.index("--loss") + 1] == "mse"
        assert float(cmd[cmd.index("--lr") + 1]) == pytest.approx(0.02)
        assert float(cmd[cmd.index("--cull-retention") + 1]) == pytest.approx(0.9)
        # A content box's budget comes from the PLAN, never from --seeds.
        assert "--seeds" not in cmd
        # Nothing supplied -> no flags (the worker resolves its own defaults).
        # `--preset` included: a "standard" default would layer that preset's
        # n_iters (5000) / cull_retention on a box the sequential path fits with
        # `load_fit_config(preset=None)` (1000 iterations).
        bare = [
            str(c)
            for c in _default_worker_cmd_builder("in.zarr", "plan.json")(
                0, tmp_path / "box0.gsplats.zarr"
            )
        ]
        for flag in (
            "--preset",
            "--config",
            "--iters",
            "--loss",
            "--lr",
            "--cull-retention",
        ):
            assert flag not in bare
        # A zero cull retention ("keep every splat") is a value, not an absence.
        zero = _default_worker_cmd_builder("in.zarr", "plan.json", cull_retention=0.0)
        cmd_zero = [str(c) for c in zero(0, tmp_path / "box0.gsplats.zarr")]
        assert float(cmd_zero[cmd_zero.index("--cull-retention") + 1]) == 0.0


def _pedestal_blobs(shape=(32, 32, 64), pedestal=5.0, step=12.0, seed=0):
    """Blobs on a background pedestal that STEPS across the x midpoint.

    The step is what makes per-box floor estimation visible: the two abutting
    boxes of the plan below sit on different pedestals, so a box that resolves
    ``auto`` against its own crop gets a different level from its neighbour —
    different subtracted baseline AND different normalization range, i.e. a
    brightness step at the box boundary.
    """
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    V = np.full(shape, pedestal, np.float32)
    V[:, :, shape[2] // 2 :] = step
    rng = np.random.default_rng(seed)
    for _ in range(12):
        cz = rng.integers(4, shape[0] - 4)
        cy = rng.integers(4, shape[1] - 4)
        cx = rng.integers(4, shape[2] - 4)
        V += 10.0 * np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
        ).astype(np.float32)
    return V


class TestContentFitSharedFloor:
    """`fit --tiling content` resolves ONE floor level for every box (#1174)."""

    @staticmethod
    def _one_splat(volume, box, overlap, cap, **fit_kwargs):
        """Stand-in for ``_fit_one_box``: one splat at the box centre."""
        from luxar.gsplats.gsplat_data import GSplatData

        z0, z1, y0, y1, x0, x1 = box.box
        c = np.array([[(z0 + z1) / 2, (y0 + y1) / 2, (x0 + x1) / 2]], np.float32)
        return GSplatData(
            centers=c,
            amplitudes=np.ones((1,), np.float32),
            cholesky_factors=np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (1, 1)),
        )

    def _run(self, tmp_path, monkeypatch, volume, floor, jobs="1"):
        """Run a content fit over a 2-box plan, capturing each box's floor kwarg."""
        import importlib

        from luxar.cli.gsplat_ops.planner import run_content_fit

        # `luxar.gsplats.planner.fit_planned` the ATTRIBUTE is the re-exported
        # function, so the module has to be fetched explicitly to patch into it.
        fp = importlib.import_module("luxar.gsplats.planner.fit_planned")
        seen: list = []

        def _spy(volume_, box, overlap, cap, **fit_kwargs):
            seen.append(fit_kwargs.get("floor"))
            return self._one_splat(volume_, box, overlap, cap, **fit_kwargs)

        monkeypatch.setattr(fp, "_fit_one_box", _spy)

        # Two abutting boxes splitting x in half (core-disjoint, no gap).
        z, y, x = volume.shape
        plan = FitPlan(
            volume_shape=[z, y, x],
            boxes=[
                PlanBox(box=[0, z, 0, y, 0, x // 2], n_features=10, budget=50),
                PlanBox(box=[0, z, 0, y, x // 2, x], n_features=10, budget=50),
            ],
            overlap=4,
            feature_method="peaks",
            min_leaf=8,
            max_leaf=32,
            density={"saturation_cap": 10_000},
        )
        plan_json = tmp_path / "plan.json"
        plan.to_json(plan_json)

        run_content_fit(
            tmp_path / "unused.npy",
            tmp_path / "out.gsplats.zarr",
            volume=volume,
            k_star_ref=4000,
            n_features_ref=200,
            plan=plan_json,
            floor=floor,
            jobs=jobs,
            flat=True,
            preset="draft",
            device="cpu",
            verbose=False,
        )
        return seen

    def test_abutting_boxes_get_one_identical_numeric_level(
        self, tmp_path, monkeypatch
    ):
        from luxar.gsplats.fitting.preprocessing import (
            _resolve_floor,
            resolve_volume_floor,
        )

        V = _pedestal_blobs()
        expected = resolve_volume_floor(V, "auto", guard_numeric=True)
        assert expected is not None and expected > 0.0  # the fixture has a pedestal

        # The pre-fix behaviour this pins down: each box crop resolves its OWN,
        # different level (that is what forwarding the spec into the per-box
        # fit does), so the boundary between them shows a brightness step.
        half = V.shape[2] // 2
        per_crop = (
            _resolve_floor(V[:, :, :half], "auto"),
            _resolve_floor(V[:, :, half:], "auto"),
        )
        assert per_crop[0] != per_crop[1]

        seen = self._run(tmp_path, monkeypatch, V, floor="auto")
        assert len(seen) == 2
        # A NUMBER, not the spec: a spec would be re-resolved per box crop.
        assert all(isinstance(f, float) for f in seen), seen
        assert seen[0] == pytest.approx(expected)
        assert seen[0] == seen[1]

    def test_disabled_and_explicit_numeric_round_trip(self, tmp_path, monkeypatch):
        V = _pedestal_blobs()
        assert self._run(tmp_path, monkeypatch, V, floor="none") == ["none", "none"]
        seen = self._run(tmp_path, monkeypatch, V, floor="3.0")
        assert seen == [pytest.approx(3.0), pytest.approx(3.0)]

    def test_percentile_spec_becomes_one_number_for_every_box(
        self, tmp_path, monkeypatch
    ):
        """`--floor pNN` too: ONE identical number for every box (#1174).

        Pre-fix each box ran its own `np.percentile` over its own crop, so
        abutting boxes were normalized against different baselines. The shared
        resolution goes through the same deterministic whole-volume sampler every
        worker would use; this fixture is well under the sample budget, so the
        sample IS the whole array and the level equals the exact percentile.
        """
        from luxar.gsplats.fitting.preprocessing import (
            _resolve_floor,
            resolve_volume_floor,
        )

        V = _pedestal_blobs()
        expected = resolve_volume_floor(V, "p10", guard_numeric=True)
        assert expected is not None
        half = V.shape[2] // 2
        assert _resolve_floor(V[:, :, :half], "p10") != _resolve_floor(
            V[:, :, half:], "p10"
        )

        seen = self._run(tmp_path, monkeypatch, V, floor="p10")
        assert all(isinstance(f, float) for f in seen), seen
        assert seen == [pytest.approx(expected), pytest.approx(expected)]

    @staticmethod
    def _capture_worker_argvs(tmp_path, monkeypatch, volume, **run_kwargs) -> list:
        """Run a 2-box content fit with ``-j 2``, capturing each worker's argv.

        Only ``fit_planned_parallel`` is faked out (no subprocess is spawned), so
        the argv comes from the REAL builder ``run_content_fit`` constructs — this
        is the seam that covers the CLI call site forwarding the run's floor and
        fit configuration to the box workers.
        """
        import importlib

        from luxar.cli.gsplat_ops.planner import run_content_fit
        from luxar.gsplats.gsplat_data import GSplatData

        fpp = importlib.import_module("luxar.gsplats.planner.fit_planned_parallel")
        argvs: list = []

        def _fake_parallel(plan, *, jobs, tmp_dir, worker_cmd_builder, **kwargs):
            for i in range(len(plan.boxes)):
                argvs.append([str(t) for t in worker_cmd_builder(i, tmp_dir / f"b{i}")])
            return GSplatData(
                centers=np.zeros((1, 3), np.float32),
                amplitudes=np.ones((1,), np.float32),
                cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], np.float32),
            )

        monkeypatch.setattr(fpp, "fit_planned_parallel", _fake_parallel)

        z, y, x = volume.shape
        plan = FitPlan(
            volume_shape=[z, y, x],
            boxes=[
                PlanBox(box=[0, z, 0, y, 0, x // 2], n_features=10, budget=50),
                PlanBox(box=[0, z, 0, y, x // 2, x], n_features=10, budget=50),
            ],
            overlap=4,
            feature_method="peaks",
            min_leaf=8,
            max_leaf=32,
            density={"saturation_cap": 10_000},
        )
        plan_json = tmp_path / "plan.json"
        plan.to_json(plan_json)
        run_content_fit(
            tmp_path / "unused.npy",
            tmp_path / "out.gsplats.zarr",
            volume=volume,
            k_star_ref=4000,
            n_features_ref=200,
            plan=plan_json,
            jobs="2",
            flat=True,
            device="cpu",
            verbose=False,
            **run_kwargs,
        )
        return argvs

    def test_parallel_worker_argv_carries_a_number_not_a_spec(
        self, tmp_path, monkeypatch
    ):
        """The -j>1 box subprocesses must be handed the level, not 'auto'."""
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        V = _pedestal_blobs()
        expected = resolve_volume_floor(V, "auto", guard_numeric=True)
        argvs = self._capture_worker_argvs(
            tmp_path, monkeypatch, V, floor="auto", preset="draft"
        )
        assert len(argvs) == 2
        levels = [float(a[a.index("--floor") + 1]) for a in argvs]
        assert levels == [pytest.approx(expected), pytest.approx(expected)]

    def test_parallel_worker_argv_carries_the_runs_fit_configuration(
        self, tmp_path, monkeypatch
    ):
        """The CLI call site must forward the run's fit config to each box worker.

        Every builder-level assertion stays green if the ``config=``/``iters=``/
        ``cull_retention=`` kwargs are deleted from the ``run_content_fit`` call
        site — this is where that is caught. ``truncate:`` is YAML-only, so an
        unforwarded ``--config`` left every ``-j N`` box fitted at the 2.75 default
        (#1637), and a ``preset or "standard"`` default made a box resolve 5000
        iterations where ``-j 1`` resolves 1000.
        """
        V = _pedestal_blobs(shape=(16, 16, 32))
        cfg = tmp_path / "fit.yaml"
        cfg.write_text("truncate: 3.5\nn_iters: 11\n")
        argvs = self._capture_worker_argvs(
            tmp_path,
            monkeypatch,
            V,
            floor="none",
            config=cfg,
            iters=9,  # a CLI override outranks the config's n_iters: 11
            cull_retention=0.5,
        )
        assert len(argvs) == 2
        for argv in argvs:
            assert argv[argv.index("--config") + 1] == str(cfg)
            assert argv[argv.index("--iters") + 1] == "9"
            assert float(argv[argv.index("--cull-retention") + 1]) == pytest.approx(0.5)
            # No --preset was asked for, so none is forwarded: "standard" would
            # layer 5000 iterations / cull_retention 0.999 on a box the sequential
            # path fits with `load_fit_config(preset=None)`.
            assert "--preset" not in argv
        # An explicit preset IS forwarded, verbatim.
        with_preset = self._capture_worker_argvs(
            tmp_path, monkeypatch, V, floor="none", preset="draft"
        )
        assert all(a[a.index("--preset") + 1] == "draft" for a in with_preset)

    def test_bad_floor_spec_is_rejected_before_the_volume_is_read(self, tmp_path):
        """An invalid spec is a usage error, paid for with zero volume reads.

        FAILS pre-fix: the volume was loaded FIRST (``fit.py`` loads before
        dispatching to the content path) and the spec was only validated inside
        ``resolve_shared_floor``, i.e. after the read — surfacing as a bare
        ``ValueError``. A ``floor:`` in a YAML ``--config`` was not validated on
        this path at all. The input path below does not exist, so a read would
        raise something else entirely.
        """
        import typer

        from luxar.cli.gsplat_ops.planner import run_content_fit

        missing = tmp_path / "does-not-exist.zarr"
        with pytest.raises(typer.BadParameter):
            run_content_fit(
                missing,
                tmp_path / "out.gsplats.zarr",
                k_star_ref=4000,
                n_features_ref=200,
                floor="p150",
                verbose=False,
            )
        cfg = tmp_path / "fit.yaml"
        cfg.write_text("floor: -5.0\n")
        with pytest.raises(typer.BadParameter):
            run_content_fit(
                missing,
                tmp_path / "out.gsplats.zarr",
                k_star_ref=4000,
                n_features_ref=200,
                config=cfg,
                verbose=False,
            )

    def test_cli_renders_a_bad_floor_spec_as_a_usage_error(self, tmp_path):
        """No traceback: `fit --tiling content --floor p150` is a usage error.

        FAILS pre-fix: `fit`'s generic ``except Exception`` printed the
        ``ValueError`` plus a full traceback (after loading the volume).
        """
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat

        vol = tmp_path / "vol.npy"
        np.save(vol, _pedestal_blobs(shape=(16, 16, 16)))
        result = CliRunner().invoke(
            app_gsplat,
            # fmt: off
            [
                "fit",
                str(vol),
                str(tmp_path / "out.gsplats.zarr"),
                "--tiling",
                "content",
                "--k-star-ref",
                "4000",
                "--n-features-ref",
                "200",
                "--floor",
                "p150",
                "--device",
                "cpu",
            ],
            # fmt: on
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "floor" in result.output.lower()


class TestContentFitCullRetention:
    """`fit --tiling content` fits every box near-losslessly (#1729).

    The planner meant to install a content default of 0.999 with
    ``fk.setdefault("cull_retention", 0.999)``, which could never fire: the
    resolved config always already carries the fitter's own signature default of
    0.95. So a preset-less content fit culled the bottom 5% of cumulative
    amplitude out of EVERY box (and the boxes are re-merged, so the cull
    compounds) while the code claimed otherwise.
    """

    @staticmethod
    def _plan(tmp_path, volume) -> Path:
        """Two abutting boxes splitting x in half (core-disjoint, no gap)."""
        z, y, x = volume.shape
        plan = FitPlan(
            volume_shape=[z, y, x],
            boxes=[
                PlanBox(box=[0, z, 0, y, 0, x // 2], n_features=10, budget=50),
                PlanBox(box=[0, z, 0, y, x // 2, x], n_features=10, budget=50),
            ],
            overlap=4,
            feature_method="peaks",
            min_leaf=8,
            max_leaf=32,
            density={"saturation_cap": 10_000},
        )
        plan_json = tmp_path / "plan.json"
        plan.to_json(plan_json)
        return plan_json

    @staticmethod
    def _spy_boxes(monkeypatch) -> list:
        """Patch ``_fit_one_box`` with a spy recording each box's resolved kwargs.

        Records ``n_iters`` alongside the retention: 0.999 alone cannot tell a
        preset apart from the command default (they agree), so a preset test needs
        a knob only the preset layer sets.
        """
        import importlib

        fp = importlib.import_module("luxar.gsplats.planner.fit_planned")
        seen: list = []

        def _spy(volume_, box, overlap, cap, **fit_kwargs):
            seen.append(
                {
                    "cull_retention": fit_kwargs.get("cull_retention"),
                    "n_iters": fit_kwargs.get("n_iters"),
                }
            )
            return TestContentFitSharedFloor._one_splat(
                volume_, box, overlap, cap, **fit_kwargs
            )

        monkeypatch.setattr(fp, "_fit_one_box", _spy)
        return seen

    def _run_boxes(self, tmp_path, monkeypatch, **run_kwargs) -> list:
        """Sequential 2-box content fit; returns each box's recorded fit kwargs."""
        from luxar.cli.gsplat_ops.planner import run_content_fit

        volume = _pedestal_blobs(shape=(16, 16, 32))
        seen = self._spy_boxes(monkeypatch)
        run_content_fit(
            tmp_path / "unused.npy",
            tmp_path / "out.gsplats.zarr",
            volume=volume,
            k_star_ref=4000,
            n_features_ref=200,
            plan=self._plan(tmp_path, volume),
            floor="none",  # keep the run cheap; the floor is #1174's business
            flat=True,
            device="cpu",
            verbose=False,
            **run_kwargs,
        )
        return seen

    def _run(self, tmp_path, monkeypatch, **run_kwargs) -> list:
        """As :meth:`_run_boxes`, projected onto each box's ``cull_retention``."""
        return [
            r["cull_retention"]
            for r in self._run_boxes(tmp_path, monkeypatch, **run_kwargs)
        ]

    def test_bare_content_fit_is_near_lossless(self, tmp_path, monkeypatch):
        """No preset, no --config, no --cull-retention → 0.999 for every box.

        FAILS pre-fix with 0.95 (the fitter's signature default falling all the
        way through), which is the regression this pins.
        """
        assert self._run(tmp_path, monkeypatch) == [
            pytest.approx(CONTENT_CULL_RETENTION),
            pytest.approx(CONTENT_CULL_RETENTION),
        ]

    def test_preset_still_gives_its_own_retention(self, tmp_path, monkeypatch):
        """A preset keeps outranking the command default.

        ``standard``'s retention is ALSO 0.999, so the retention alone cannot tell
        which layer supplied it. ``n_iters`` can: 5000 is the preset's, 1000 the
        harvested function default a preset-less run resolves.
        """
        seen = self._run_boxes(tmp_path, monkeypatch, preset="standard")
        assert [r["cull_retention"] for r in seen] == [
            pytest.approx(0.999),
            pytest.approx(0.999),
        ]
        assert [r["n_iters"] for r in seen] == [5000, 5000]  # the preset layer landed

    def test_explicit_cli_value_reaches_every_box(self, tmp_path, monkeypatch):
        seen = self._run(tmp_path, monkeypatch, cull_retention=0.5)
        assert seen == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_config_value_beats_the_command_default(self, tmp_path, monkeypatch):
        cfg = tmp_path / "fit.yaml"
        cfg.write_text("cull_retention: 0.5\n")
        seen = self._run(tmp_path, monkeypatch, config=cfg)
        assert seen == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_zero_keeps_every_splat_and_is_not_replaced(self, tmp_path, monkeypatch):
        """`--cull-retention 0` is a value, not an absence: it must win."""
        seen = self._run(tmp_path, monkeypatch, cull_retention=0.0)
        assert seen == [0.0, 0.0]

    def test_parallel_worker_argv_only_carries_an_asked_for_retention(
        self, tmp_path, monkeypatch
    ):
        """The command default must not leak into the ``-j N`` worker argv.

        The workers re-enter ``_fit_kwargs`` and resolve the same default
        themselves, so materializing it here would only make an absent flag
        indistinguishable from an explicit one.
        """
        V = _pedestal_blobs(shape=(16, 16, 32))
        bare = TestContentFitSharedFloor._capture_worker_argvs(
            tmp_path, monkeypatch, V, floor="none"
        )
        assert len(bare) == 2
        assert all("--cull-retention" not in argv for argv in bare)
        asked = TestContentFitSharedFloor._capture_worker_argvs(
            tmp_path, monkeypatch, V, floor="none", cull_retention=0.5
        )
        assert all(
            float(a[a.index("--cull-retention") + 1]) == pytest.approx(0.5)
            for a in asked
        )

    def test_worker_branch_resolves_the_same_default(self, tmp_path, monkeypatch):
        """`--plan-box K` (the ``-j N`` subprocess) must not fit differently.

        The worker re-enters ``_fit_kwargs`` in its own process, so a default
        living only on the sequential path would make ``-j N`` and ``-j 1``
        produce different splat sets — the #1637 class of bug.
        """
        from luxar.cli.gsplat_ops.planner import run_content_fit

        volume = _pedestal_blobs(shape=(16, 16, 32))
        seen = self._spy_boxes(monkeypatch)
        run_content_fit(
            tmp_path / "unused.npy",
            tmp_path / "box0.gsplats.zarr",
            volume=volume,
            k_star_ref=4000,
            n_features_ref=200,
            plan=self._plan(tmp_path, volume),
            plan_box=0,
            floor="none",
            device="cpu",
            verbose=False,
        )
        assert [r["cull_retention"] for r in seen] == [
            pytest.approx(CONTENT_CULL_RETENTION)
        ]

    def test_cli_flag_reaches_every_box_through_the_real_command(
        self, tmp_path, monkeypatch
    ):
        """`--cull-retention` must survive the real ``fit`` dispatch, not just
        ``run_content_fit``.

        Every other test in this class calls ``run_content_fit`` directly, so
        deleting ``cull_retention=cull_retention`` from ``fit.py``'s
        ``--tiling content`` dispatch would leave the flag silently ineffective
        with all of them still green — the same hole the ``-j`` builder test
        above covers for the worker argv.
        """
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat

        volume = _pedestal_blobs(shape=(16, 16, 32))
        vol = tmp_path / "vol.npy"
        np.save(vol, volume)
        seen = self._spy_boxes(monkeypatch)
        result = CliRunner().invoke(
            app_gsplat,
            # fmt: off
            [
                "fit",
                str(vol),
                str(tmp_path / "out.gsplats.zarr"),
                "--tiling",
                "content",
                "--plan",
                str(self._plan(tmp_path, volume)),
                "--k-star-ref",
                "4000",
                "--n-features-ref",
                "200",
                "--cull-retention",
                "0.5",
                "--floor",
                "none",
                "--flat",
                "--device",
                "cpu",
                "--quiet",
            ],
            # fmt: on
        )
        assert result.exit_code == 0, result.output
        assert [r["cull_retention"] for r in seen] == [
            pytest.approx(0.5),
            pytest.approx(0.5),
        ]


class TestFitPlannedParallel:
    @pytest.mark.parametrize(
        ("keep_boxes", "expected"),
        [(False, "1"), (True, "unset")],
    )
    def test_only_disposable_box_workers_suppress_restamping(
        self, tmp_path, keep_boxes, expected
    ):
        marker = tmp_path / f"worker-env-{keep_boxes}"
        base = _fake_box_builder(5)

        def builder(i: int, out_path: Path) -> list[str]:
            cmd = base(i, out_path)
            cmd[-1] = (
                "import os; from pathlib import Path; "
                f"Path(r'{marker}').write_text("
                "os.environ.get('LUXAR_INTERNAL_SKIP_CONTENT_BOX_STAMP', 'unset')); "
                + cmd[-1]
            )
            return cmd

        fit_planned_parallel(
            _toy_plan(n_boxes=1),
            jobs=1,
            tmp_dir=tmp_path / f"boxes-{keep_boxes}",
            worker_cmd_builder=builder,
            keep_boxes=keep_boxes,
            verbose=False,
        )

        assert marker.read_text() == expected

    def test_merges_all_budgeted_boxes(self, tmp_path):
        plan = _toy_plan(n_boxes=3)
        d = tmp_path / "boxes"
        merged = fit_planned_parallel(
            plan,
            jobs=2,
            tmp_dir=d,
            worker_cmd_builder=_fake_box_builder(5),
            verbose=False,
        )
        assert merged.n_splats == 5 * 3
        assert merged.stats["n_boxes"] == 3
        assert merged.stats["n_boxes_fit"] == 3
        assert merged.stats["parallel_jobs"] == 2
        assert not d.exists()  # tmp removed on success

    def test_empty_box_marker_skipped(self, tmp_path):
        plan = _toy_plan(n_boxes=3)
        merged = fit_planned_parallel(
            plan,
            jobs=2,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_empty_marker_for(1),
            verbose=False,
        )
        assert merged.n_splats == 5 * 2  # boxes 0 and 2 contribute
        assert merged.stats["n_boxes_fit"] == 3  # the empty box still "ran"

    def test_only_budgeted_boxes_spawned(self, tmp_path):
        plan = _toy_plan(n_boxes=3)
        plan.boxes[2].budget = 0  # zero-budget box must be skipped, not spawned
        spawned: list[int] = []
        base = _fake_box_builder(5)

        def builder(i: int, out_path: Path) -> list[str]:
            spawned.append(i)
            return base(i, out_path)

        merged = fit_planned_parallel(
            plan, jobs=2, tmp_dir=tmp_path / "boxes", worker_cmd_builder=builder
        )
        assert sorted(spawned) == [0, 1]  # box 2 (budget 0) never spawned
        assert merged.n_splats == 5 * 2

    def test_worker_failure_raises_and_retains_tmp(self, tmp_path):
        plan = _toy_plan(n_boxes=3)
        d = tmp_path / "boxes"
        ok = _fake_box_builder(5)

        def builder(i: int, out_path: Path) -> list[str]:
            if i == 1:
                return [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('BOOM\\n'); sys.exit(3)",
                ]
            return ok(i, out_path)

        with pytest.raises(RuntimeError, match="box fits failed"):
            fit_planned_parallel(
                plan, jobs=2, tmp_dir=d, worker_cmd_builder=builder, verbose=False
            )
        assert d.exists()  # retained for inspection on failure

    def test_clean_exit_no_output_raises(self, tmp_path):
        plan = _toy_plan(n_boxes=2)

        def builder(i: int, out_path: Path) -> list[str]:
            return [sys.executable, "-c", "pass"]  # exit 0, writes nothing

        with pytest.raises(RuntimeError, match="wrote no output"):
            fit_planned_parallel(
                plan, jobs=1, tmp_dir=tmp_path / "boxes", worker_cmd_builder=builder
            )

    def test_all_empty_raises_valueerror(self, tmp_path):
        plan = _toy_plan(n_boxes=2)
        with pytest.raises(ValueError, match="no splats"):
            fit_planned_parallel(
                plan,
                jobs=2,
                tmp_dir=tmp_path / "boxes",
                worker_cmd_builder=_empty_marker_for(0, 1),
            )

    def test_keep_boxes_retains_tmp(self, tmp_path):
        plan = _toy_plan(n_boxes=2)
        d = tmp_path / "boxes"
        fit_planned_parallel(
            plan,
            jobs=1,
            tmp_dir=d,
            worker_cmd_builder=_fake_box_builder(5),
            keep_boxes=True,
        )
        assert d.exists() and any(d.iterdir())

    def test_missing_reference_is_announced_even_when_quiet(self, tmp_path, capsys):
        merged = fit_planned_parallel(
            _toy_plan(n_boxes=1),
            jobs=1,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5),
            verbose=False,
        )

        assert "psnr_db" not in merged.stats
        notice = capsys.readouterr().out
        assert "No merged quality metrics" in notice
        assert "reference volume" in notice
        assert "gsplat compare" in notice

    def test_flat_merge_with_reference_records_quality(self, tmp_path):
        plan = _toy_plan(n_boxes=1)
        volume = _corner_blobs(tuple(plan.volume_shape), n=3, corner=12)
        merged = fit_planned_parallel(
            plan,
            jobs=1,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5),
            volume=volume,
            device="cpu",
            verbose=False,
        )

        quality_keys = {
            "mse",
            "psnr_db",
            "ssim",
            "foreground_psnr_db",
            "foreground_threshold",
            "foreground_fraction",
        }
        assert quality_keys <= merged.stats.keys()
        assert all(np.isfinite(merged.stats[key]) for key in quality_keys)

    def test_mismatched_reference_shape_is_announced(self, tmp_path, capsys):
        plan = _toy_plan(n_boxes=1)
        merged = fit_planned_parallel(
            plan,
            jobs=1,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5),
            volume=np.zeros((8, 8, 8), np.float32),
            verbose=False,
        )

        assert "psnr_db" not in merged.stats
        notice = capsys.readouterr().out
        assert "No merged quality metrics" in notice
        assert "does not match the plan grid" in notice
        assert str(tuple(plan.volume_shape)) in notice

    def test_partition_with_reference_records_flat_equivalent_quality(
        self, tmp_path, capsys
    ):
        plan = _toy_plan(n_boxes=2)
        volume = _corner_blobs(tuple(plan.volume_shape), n=3, corner=12)
        flat = fit_planned_parallel(
            plan,
            jobs=2,
            tmp_dir=tmp_path / "flat-boxes",
            worker_cmd_builder=_fake_box_builder(5),
            volume=volume,
            device="cpu",
            verbose=False,
        )
        node = fit_planned_parallel(
            plan,
            jobs=2,
            tmp_dir=tmp_path / "partition-boxes",
            worker_cmd_builder=_fake_box_builder(5),
            volume=volume,
            device="cpu",
            partition=True,
            verbose=False,
        )

        stats = node.meta["fit_stats"]
        quality_keys = {
            "mse",
            "psnr_db",
            "ssim",
            "foreground_psnr_db",
            "foreground_threshold",
            "foreground_fraction",
        }
        assert quality_keys <= stats.keys()
        assert stats["psnr_db"] == pytest.approx(flat.stats["psnr_db"])
        assert stats["mse"] == pytest.approx(flat.stats["mse"])
        assert "floor" not in stats
        assert "concatenated_from" not in stats
        assert stats["splats_per_tile"] == [5, 5]
        assert node.meta["floor"] == pytest.approx(0.0)
        assert node.meta["image_min"] == pytest.approx(0.0)
        assert node.meta["image_max"] == pytest.approx(1.0)
        assert node.meta["intensity_range"] == pytest.approx(1.0)
        assert flat.stats["floor"] == pytest.approx(node.meta["floor"])
        assert flat.stats["image_min"] == pytest.approx(node.meta["image_min"])
        assert "No merged quality metrics" not in capsys.readouterr().out

        from luxar.cli.gsplat_ops.fitting.fit_utils import save_fit_output

        output = tmp_path / "partition.gsplats.zarr"
        save_fit_output(node, output, compress=None, verbose=False)

        import zarr

        root = zarr.open_group(str(output), mode="r")
        assert root["fitting"].attrs["psnr_db"] == pytest.approx(stats["psnr_db"])
        assert root["fitting"].attrs["n_splats"] == node.n_splats
        assert root["pipeline"].attrs["planned_fit"] is True

    def test_single_region_partition_request_records_quality_without_a_notice(
        self, tmp_path, capsys
    ):
        from luxar.gsplats.tree import GSplatPartition

        plan = _toy_plan(n_boxes=1)
        volume = _corner_blobs(tuple(plan.volume_shape), n=3, corner=12)
        result = fit_planned_parallel(
            plan,
            jobs=1,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5),
            volume=volume,
            device="cpu",
            partition=True,
            verbose=False,
        )

        assert not isinstance(result, GSplatPartition)
        assert np.isfinite(result.meta["fit_stats"]["psnr_db"])
        notice = capsys.readouterr().out
        assert "No merged quality metrics" not in notice
        assert "gsplat flatten" not in notice

    def test_partition_quality_budget_skip_is_announced(
        self, tmp_path, monkeypatch, capsys
    ):
        plan = _toy_plan(n_boxes=2)
        monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "0")
        node = fit_planned_parallel(
            plan,
            jobs=2,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5),
            volume=np.zeros(plan.volume_shape, np.float32),
            partition=True,
            verbose=False,
        )

        assert "psnr_db" not in node.meta["fit_stats"]
        notice = capsys.readouterr().out
        assert "Merged quality metrics skipped" in notice
        assert "gsplat compare" in notice
        # `compare` reads a `kind=partition` store as written (#1978) — the
        # recourse must not ask for a full-disk `gsplat flatten` copy first.
        assert "gsplat flatten" not in notice

    def test_partition_missing_reference_is_announced_and_keeps_root_stats(
        self, tmp_path, capsys
    ):
        node = fit_planned_parallel(
            _toy_plan(n_boxes=2),
            jobs=2,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5),
            partition=True,
            verbose=False,
        )

        assert node.meta["fit_stats"]["planned_fit"] is True
        assert "psnr_db" not in node.meta["fit_stats"]
        notice = capsys.readouterr().out
        assert "No merged quality metrics" in notice
        assert "reference volume" in notice
        assert "gsplat compare" in notice
        # `compare` reads a `kind=partition` store as written (#1978) — the
        # recourse must not ask for a full-disk `gsplat flatten` copy first.
        assert "gsplat flatten" not in notice

    def test_partition_disagreeing_box_bases_warn_and_score_raw(self, tmp_path, capsys):
        builders = {
            0: _fake_box_builder(5, image_min=2.0),
            1: _fake_box_builder(5, image_min=3.0),
        }

        def builder(i: int, out_path: Path) -> list[str]:
            return builders[i](i, out_path)

        plan = _toy_plan(n_boxes=2)
        volume = _corner_blobs(tuple(plan.volume_shape), n=3, corner=12) + 10.0
        node = fit_planned_parallel(
            plan,
            jobs=2,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=builder,
            volume=volume,
            device="cpu",
            partition=True,
            verbose=False,
        )

        assert "image_min" not in node.meta
        assert "floor" not in node.meta
        assert np.isfinite(node.meta["fit_stats"]["psnr_db"])
        notice = capsys.readouterr().out
        assert "records no normalization basis" in notice
        assert "computed against the RAW volume" in notice

    @pytest.mark.parametrize("partition", [False, True])
    def test_quality_is_independent_of_the_removed_pedestal(self, tmp_path, partition):
        plan = _toy_plan(n_boxes=2)
        signal = _corner_blobs(tuple(plan.volume_shape), n=3, corner=12)
        psnr = []
        for pedestal in (0.0, 500.0):
            result = fit_planned_parallel(
                plan,
                jobs=2,
                tmp_dir=tmp_path / f"boxes-{pedestal:g}",
                worker_cmd_builder=_fake_box_builder(5, image_min=pedestal),
                volume=signal + pedestal,
                device="cpu",
                partition=partition,
                verbose=False,
            )
            stats = result.meta["fit_stats"] if partition else result.stats
            psnr.append(stats["psnr_db"])

        assert psnr[1] == pytest.approx(psnr[0], abs=0.05)


# ── The fit's truncation radius survives the planned path (#1637) ──
#
# `_fit_one_box` used to hand back bare (centers, amplitudes, cholesky_factors),
# so every consumer rebuilt a GSplatData from those three arrays and the fitted
# radius fell back to the 2.75 default even when the config said `truncate: 3.5`.
# The user-visible symptom is that such a content result then refuses to
# `GSplatData.concatenate` with a uniform-tiled one fitted from the SAME config.

# Fast real CPU fit knobs: tiny volume, a handful of iterations. The radius is a
# config passthrough, so the fit only has to run — not converge.
_FAST_FIT = dict(
    device="cpu",
    n_iters=5,
    early_stop_patience=5,
    use_cuda=False,
    use_metal=False,
    verbose=False,
)


def _leaf_nodes(node) -> list:
    """Every ``GSplatLeaf`` under ``node`` (a bare leaf is its own only leaf)."""
    children = getattr(node, "children", None)
    if not children:
        return [node]
    out: list = []
    for child in children:
        out.extend(_leaf_nodes(child))
    return out


class TestPlannedFitTruncationRadius:
    """A planned fit keeps the radius (and per-box stats) its config asked for."""

    @staticmethod
    def _tiny_volume_and_plan():
        V = _corner_blobs((24, 24, 24), n=4, corner=20)
        plan = plan_volume(
            V,
            _density(k_star_reference=200, saturation_cap=400, splats_per_feature=25.0),
            cell=4,
            min_leaf=12,
            max_leaf=12,
            overlap=2,
        )
        assert any(b.budget > 0 for b in plan.boxes)
        return V, plan

    def test_internal_parallel_worker_skips_disposable_box_restamp(self, monkeypatch):
        from luxar.cli.gsplat_ops.planner import _stamp_content_box_output
        from luxar.gsplats import merged_quality
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.planner import PlanBox

        stats = {"floor": 0.0, "image_min": 0.0, "intensity_range": 1.0}
        result = GSplatData(
            centers=np.zeros((1, 3), np.float32),
            amplitudes=np.ones(1, np.float32),
            cholesky_factors=np.ones((1, 6), np.float32),
            stats=stats.copy(),
        )
        box = PlanBox(box=[0, 1, 0, 1, 0, 1], n_features=1, budget=1)

        def fail_if_scored(*args, **kwargs):
            raise AssertionError("disposable parallel box was scored")

        monkeypatch.setenv("LUXAR_INTERNAL_SKIP_CONTENT_BOX_STAMP", "1")
        monkeypatch.setattr(merged_quality, "stamp_merged_quality", fail_if_scored)

        _stamp_content_box_output(
            result,
            np.ones((1, 1, 1), np.float32),
            box,
            "cpu",
            verbose=False,
        )

        assert result.stats == stats

    def test_flat_leaf_keeps_the_configured_radius(self):
        """Sequential ``partition=False``: the flat leaf carries ``truncate``."""
        from luxar.gsplats.planner import fit_planned

        V, plan = self._tiny_volume_and_plan()
        t0 = time.perf_counter()
        merged = fit_planned(V, plan, truncate=3.5, **_FAST_FIT)
        wall = time.perf_counter() - t0

        assert merged.n_splats > 0
        assert merged.truncation_radius == pytest.approx(3.5)
        # The planned-fit stats keys still describe the merge (concatenate()
        # replaces stats with its own summary, so they are re-applied after).
        assert merged.stats["planned_fit"] is True
        assert merged.stats["n_boxes"] == len(plan.boxes)
        assert merged.stats["n_boxes_fit"] >= 1
        assert merged.stats["overlap"] == plan.overlap
        assert list(merged.stats["volume_shape"]) == list(V.shape)
        quality_keys = {
            "mse",
            "psnr_db",
            "ssim",
            "foreground_psnr_db",
            "foreground_threshold",
            "foreground_fraction",
        }
        assert quality_keys <= merged.stats.keys()
        assert merged.stats["psnr_db"] > 10
        assert 0 < merged.stats["ssim"] <= 1
        assert np.isfinite(merged.stats["foreground_psnr_db"])
        # Wall clock for the fit loop, so it is bounded by the wall clock of the
        # whole call — the SUM of the boxes' own fit times need not be.
        assert 0 < merged.stats["time_seconds"] <= wall

    def test_quality_budget_skip_is_announced_and_stamps_nothing(
        self, monkeypatch, capsys
    ):
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.planner import fit_planned as fit_planned_fn

        fit_planned_module = __import__(
            "luxar.gsplats.planner.fit_planned", fromlist=["_fit_one_box"]
        )
        volume = np.ones((16, 16, 16), np.float32)
        plan = _toy_plan(n_boxes=1)

        def _fake_fit(*args, **kwargs):
            return GSplatData(
                centers=np.array([[8.0, 8.0, 8.0]], np.float32),
                amplitudes=np.ones((1,), np.float32),
                cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], np.float32),
            )

        monkeypatch.setattr(fit_planned_module, "_fit_one_box", _fake_fit)
        monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "0")
        merged = fit_planned_fn(volume, plan, verbose=False)

        assert "psnr_db" not in merged.stats
        assert "foreground_psnr_db" not in merged.stats
        notice = capsys.readouterr().out
        assert "Merged quality metrics skipped" in notice
        assert "LUXAR_TILED_QUALITY_MAX_GB" in notice

    def test_rejects_a_volume_from_a_different_plan_grid(self, monkeypatch):
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.planner import fit_planned as fit_planned_fn

        fit_planned_module = importlib.import_module(
            "luxar.gsplats.planner.fit_planned"
        )
        plan = _toy_plan(n_boxes=1)

        def _fake_fit(*args, **kwargs):
            return GSplatData(
                centers=np.array([[4.0, 4.0, 4.0]], np.float32),
                amplitudes=np.ones((1,), np.float32),
                cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], np.float32),
            )

        monkeypatch.setattr(fit_planned_module, "_fit_one_box", _fake_fit)

        with pytest.raises(ValueError, match="does not match the plan grid"):
            fit_planned_fn(np.zeros((8, 8, 8), np.float32), plan)

    def test_partition_parts_keep_the_configured_radius_and_box_stats(
        self, tmp_path, capsys
    ):
        """Sequential ``partition=True``: every part leaf carries ``truncate``."""
        from luxar.gsplats._data.filtering import _REGION_SCOPED_STATS_KEYS
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.planner import fit_planned

        V, plan = self._tiny_volume_and_plan()
        flat = fit_planned(V, plan, partition=False, truncate=3.5, **_FAST_FIT)
        node = fit_planned(V, plan, partition=True, truncate=3.5, **_FAST_FIT)
        notice = capsys.readouterr().out
        assert np.isfinite(node.meta["fit_stats"]["psnr_db"])
        assert node.meta["fit_stats"]["psnr_db"] == pytest.approx(flat.stats["psnr_db"])
        assert node.meta["fit_stats"]["mse"] == pytest.approx(flat.stats["mse"])
        assert node.meta["fit_stats"]["planned_fit"] is True
        assert node.meta["fit_stats"]["n_splats"] == node.n_splats
        assert "concatenated_from" not in node.meta["fit_stats"]
        assert node.meta["fit_stats"]["splats_per_tile"]
        assert "image_max" not in node.meta["fit_stats"]
        assert node.meta["image_max"] == pytest.approx(flat.stats["image_max"])
        assert "No merged quality metrics" not in notice

        leaves = _leaf_nodes(node)
        assert leaves
        n_cropped = 0
        for leaf in leaves:
            for sub in leaf.additive_sublods:
                assert sub.truncation_radius == pytest.approx(3.5)
            # The per-box fit stats ride along with the part they describe...
            box_stats = leaf.additive_sublods[0].stats
            assert "iterations" in box_stats
            # ... but the count must describe THIS part, not the padded crop it
            # was fitted on (it becomes the part's on-disk `lod_stats`).
            assert box_stats["n_splats"] == leaf.n_splats
            if not any(k in box_stats for k in _REGION_SCOPED_STATS_KEYS):
                n_cropped += 1
                # A rescoped box also publishes no measured SCORE: `final_loss` /
                # `psnr_db` were taken on the padded crop, with the halo splats
                # present and against a bigger target region (#1600).
                assert "final_loss" not in box_stats
                assert "psnr_db" not in box_stats
        # A box is fitted on a halo-padded crop and then core-masked, so the
        # crop's grid stamps describe a bigger region than the part (the rule
        # itself is pinned by test_core_mask_rescopes_the_box_stats and
        # test_a_halo_alone_rescopes_the_box_stats below). This plan has a halo,
        # so it happens for at least one box here.
        assert n_cropped > 0

        # ... and the radius survives the partition WRITE (the default output),
        # read back through the library's own reader.
        out = tmp_path / "part.gsplats.zarr"
        write_gsplats_tree(out, node)
        parts = sorted(p.name for p in out.iterdir() if p.name.startswith("part_"))
        assert len(parts) == len(leaves)  # one part per fitted box, all written
        reloaded, _ = load_gsplat_node(out)
        stored_leaves = _leaf_nodes(reloaded)
        assert len(stored_leaves) == len(leaves)
        for leaf in stored_leaves:
            for sub in leaf.additive_sublods:
                assert sub.truncation_radius == pytest.approx(3.5)
            # A per-box fit stat reaches the part ON DISK, not just in memory
            # (the writer persists a sub-LOD's stats as the leaf's `lod_stats`).
            assert "iterations" in leaf.additive_sublods[0].stats

    def test_zero_budget_box_carries_the_configured_radius(self):
        """The early-out has no fit to read the radius off — resolve it anyway."""
        from luxar.gsplats.planner.fit_planned import _fit_one_box
        from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

        V = np.zeros((8, 8, 8), np.float32)
        box = PlanBox(box=[0, 8, 0, 8, 0, 8], n_features=0, budget=0)

        out = _fit_one_box(V, box, 0, 0, truncate=3.5)
        assert out.n_splats == 0
        assert out.truncation_radius == pytest.approx(3.5)
        # Nothing configured -> the documented default.
        bare = _fit_one_box(V, box, 0, 0)
        assert bare.truncation_radius == pytest.approx(DEFAULT_TRUNCATION_RADIUS)

    def test_flat_merge_stamps_wall_clock_time(self, monkeypatch):
        """``time_seconds`` is elapsed, not concatenate's SUM of box times."""
        import importlib

        from luxar.gsplats.gsplat_data import GSplatData

        fp = importlib.import_module("luxar.gsplats.planner.fit_planned")

        def _one_splat(volume, box, overlap, cap, **fit_kwargs):
            z0, z1, y0, y1, x0, x1 = box.box
            return GSplatData(
                centers=np.array(
                    [[(z0 + z1) / 2, (y0 + y1) / 2, (x0 + x1) / 2]], np.float32
                ),
                amplitudes=np.ones((1,), np.float32),
                cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], np.float32),
                # A per-box fit time far larger than this (instant) merge can take.
                stats={"time_seconds": 1000.0},
            )

        monkeypatch.setattr(fp, "_fit_one_box", _one_splat)
        plan = _toy_plan(n_boxes=3)
        V = np.zeros(tuple(int(s) for s in plan.volume_shape), np.float32)
        merged = fp.fit_planned(V, plan)

        assert merged.n_splats == 3
        # Summing the boxes would give 3000s; wall clock here is a fraction of one.
        assert merged.stats["time_seconds"] < 60.0

    def test_planned_fit_resolves_one_range_for_every_box(self, monkeypatch):
        import importlib

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.utils.trils import tril_size

        fp = importlib.import_module("luxar.gsplats.planner.fit_planned")
        seen = []

        def _capture(volume, box, overlap, cap, **fit_kwargs):
            seen.append(fit_kwargs["norm_range"])
            return GSplatData(
                centers=np.array([[1.0, 1.0, 1.0]], np.float32),
                amplitudes=np.ones((1,), np.float32),
                cholesky_factors=np.zeros((1, tril_size(3)), np.float32),
            )

        monkeypatch.setattr(fp, "_fit_one_box", _capture)
        plan = _toy_plan(n_boxes=3)
        volume = np.linspace(
            10.0,
            110.0,
            num=int(np.prod(plan.volume_shape)),
            dtype=np.float32,
        ).reshape(plan.volume_shape)

        fp.fit_planned(volume, plan)

        assert len(seen) == 3
        assert all(item == pytest.approx((10.0, 110.0)) for item in seen)

    def test_core_mask_rescopes_the_box_stats(self):
        """The kept subset's stats must describe IT, not the padded crop."""
        from luxar.gsplats._data.filtering import _REGION_SCOPED_STATS_KEYS
        from luxar.gsplats.planner.fit_planned import _fit_one_box

        V = _corner_blobs((24, 24, 24), n=4, corner=20)
        box = PlanBox(box=[0, 12, 0, 12, 0, 12], n_features=50, budget=120)

        # Halo padding: the crop is 18^3, so splats fitted outside the 12^3 core
        # are dropped and the crop's grid stamps no longer hold.
        out = _fit_one_box(V, box, 6, 0, **_FAST_FIT)
        assert 0 < out.n_splats
        assert out.stats["n_splats"] == out.n_splats
        assert [k for k in _REGION_SCOPED_STATS_KEYS if k in out.stats] == []
        # The MEASURED scores go with them: they were taken on the fit of the
        # padded crop, so they describe neither this splat set (the halo splats
        # contributed) nor this region (#1600).
        assert "final_loss" not in out.stats
        assert "psnr_db" not in out.stats
        # Descriptive run metadata legitimately describes this box's own fit and
        # must survive.
        assert "iterations" in out.stats

        # Negative control: the padded crop EQUALS the core box (no halo, box
        # covering the whole volume) and nothing is dropped, so the crop's grid
        # stamps still describe this part exactly and must be kept — and so does
        # its score, which was measured on exactly these splats.
        whole = PlanBox(box=[0, 24, 0, 24, 0, 24], n_features=50, budget=120)
        full = _fit_one_box(V, whole, 0, 0, **_FAST_FIT)
        assert full.stats["n_splats"] == full.n_splats
        assert full.stats["fitted_shape"] == [24, 24, 24]
        assert "occupancy" in full.stats
        assert "final_loss" in full.stats

    def test_a_halo_alone_rescopes_the_box_stats(self, monkeypatch):
        """A padded crop LARGER than the core invalidates the grid stamps...

        ...even when the core mask dropped nothing. Gating on "the mask removed
        splats" kept an 18³ `source_shape`/`fitted_shape` (and its `occupancy` /
        `voxels_per_splat`) on a part representing 12³ — the key set a reader asks
        for the part's own source grid (`gsplat info`'s source block).
        """
        from luxar.gsplats import fit_gsplats
        from luxar.gsplats._data.filtering import _REGION_SCOPED_STATS_KEYS
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.planner.fit_planned import _fit_one_box

        V = np.zeros((24, 24, 24), np.float32)
        crop_stats = {
            "source_shape": [18, 18, 18],
            "fitted_shape": [18, 18, 18],
            "occupancy": 0.5,
            "voxels_per_splat": 43.2,
            "final_loss": 0.25,
        }

        def _fake_fit(sub, **kwargs):
            # Crop-local centres at 3.0: with the crop origin at 0 every splat
            # lands inside a 12³ core, so the mask drops NOTHING.
            n = 3
            return GSplatData(
                centers=np.full((n, 3), 3.0, np.float32),
                amplitudes=np.ones((n,), np.float32),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], np.float32), (n, 1)
                ),
                stats=dict(crop_stats),
            )

        monkeypatch.setattr(fit_gsplats, "fit_gaussian_splats", _fake_fit)

        # Halo present (crop 18³ ⊃ core 12³), mask dropped nothing -> gone.
        box = PlanBox(box=[0, 12, 0, 12, 0, 12], n_features=50, budget=120)
        out = _fit_one_box(V, box, 6, 0)
        assert out.n_splats == 3  # nothing was dropped by the core mask
        assert [k for k in _REGION_SCOPED_STATS_KEYS if k in out.stats] == []
        # The score is measured against the PADDED CROP's voxels, so a halo alone
        # invalidates it even with the splat set intact (#1600).
        assert "final_loss" not in out.stats

        # Negative control: the same halo CLAMPS to the core (the box is the whole
        # volume), so the stamps describe this part and must survive.
        whole = PlanBox(box=[0, 24, 0, 24, 0, 24], n_features=50, budget=120)
        full = _fit_one_box(V, whole, 6, 0)
        assert full.n_splats == 3
        assert full.stats["fitted_shape"] == [18, 18, 18]
        assert full.stats["occupancy"] == 0.5
        assert full.stats["final_loss"] == 0.25

    def test_box_stats_are_json_safe(self, monkeypatch, tmp_path):
        """A non-finite box stat must not reach a part's attrs.

        The leaf writer stamps `lod_stats` RAW, so an `inf` would be written as a
        bare `Infinity` token that a strict JSON parser (the viewer's) refuses. A
        signal-free crop really does fit to `psnr_db = inf`, and a content
        `batch-fit` reuses one box plan across every (t, c), so such a box is
        ordinary. numpy scalars are coerced by the same filter.
        """
        import json

        from luxar.gsplats import fit_gsplats
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.planner.fit_planned import _fit_one_box

        def _fake_fit(sub, **kwargs):
            return GSplatData(
                centers=np.full((2, 3), 3.0, np.float32),
                amplitudes=np.ones((2,), np.float32),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], np.float32), (2, 1)
                ),
                stats={
                    "psnr_db": float("inf"),
                    "final_rel_l2": float("nan"),
                    "final_loss": np.float32(0.25),
                    "iterations": 5,
                },
            )

        monkeypatch.setattr(fit_gsplats, "fit_gaussian_splats", _fake_fit)
        box = PlanBox(box=[0, 12, 0, 12, 0, 12], n_features=50, budget=120)
        out = _fit_one_box(np.zeros((24, 24, 24), np.float32), box, 0, 0)

        assert "psnr_db" not in out.stats  # inf: dropped, not persisted
        assert "final_rel_l2" not in out.stats  # nan: likewise
        assert out.stats["final_loss"] == pytest.approx(0.25)
        assert type(out.stats["final_loss"]) is float  # numpy scalar coerced
        assert out.stats["iterations"] == 5

        # And the written store parses under a strict JSON reader.
        path = tmp_path / "box.gsplats.zarr"
        write_gsplats_tree(path, out.tree)

        def _reject(token: str) -> None:
            raise AssertionError(f"non-JSON token {token!r} in the store")

        # Both spellings: zarr format 3 writes one `zarr.json` per node, format 2
        # (`LUXAR_ZARR_FORMAT=2`) a `.zattrs`. Globbing only the current default
        # would leave this examining NOTHING — and passing — under the other.
        metas = [*path.rglob("zarr.json"), *path.rglob(".zattrs")]
        assert metas, "no metadata documents found to check"
        for meta in metas:
            json.loads(meta.read_text(), parse_constant=_reject)

    def test_content_and_uniform_flat_leaves_compose(self):
        """The reported symptom: "Truncation radius mismatch" on concatenate."""
        from luxar.gsplats.fit_tiled_gsplats import fit_tiled
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.planner import fit_planned

        V, plan = self._tiny_volume_and_plan()
        content = fit_planned(V, plan, truncate=3.5, **_FAST_FIT)
        uniform = fit_tiled(
            V, tile_size=16, overlap=4, seeds=100, truncate=3.5, **_FAST_FIT
        )
        assert uniform.truncation_radius == pytest.approx(3.5)  # control

        merged = GSplatData.concatenate([content, uniform])
        assert merged.n_splats == content.n_splats + uniform.n_splats
        assert merged.truncation_radius == pytest.approx(3.5)

    def test_plan_box_worker_saves_the_configured_radius(self, tmp_path):
        """A content batch-fit tile keeps its radius and fitting stamps."""
        from typer.testing import CliRunner

        from luxar._zarr_compat import read_node_attrs
        from luxar.cli.gsplat_commands import app_gsplat
        from luxar.gsplats.gsplat_data import GSplatData

        V, plan = self._tiny_volume_and_plan()
        vol = tmp_path / "vol.npy"
        stored = np.round(V * np.iinfo(np.uint16).max).astype(np.uint16)
        np.save(vol, stored)
        plan_json = tmp_path / "plan.json"
        plan.to_json(plan_json)
        cfg = tmp_path / "fit.yaml"
        cfg.write_text("truncate: 3.5\nn_iters: 5\nearly_stop_patience: 5\n")
        box_idx = next(
            i
            for i, box in enumerate(plan.boxes)
            if box.budget > 0 and any(box.box[axis] > 0 for axis in (0, 2, 4))
        )
        out = tmp_path / "box.gsplats.zarr"

        result = CliRunner().invoke(
            app_gsplat,
            # fmt: off
            [
                "fit",
                str(vol),
                str(out),
                "--tiling",
                "content",
                "--plan",
                str(plan_json),
                "--plan-box",
                str(box_idx),
                "--config",
                str(cfg),
                "--device",
                "cpu",
            ],
            # fmt: on
        )
        assert result.exit_code == 0, result.output
        assert "Merged quality: PSNR=" in result.output
        assert out.exists(), result.output
        assert GSplatData.load(out).truncation_radius == pytest.approx(3.5)
        fitting = read_node_attrs(out / "fitting")
        assert fitting["psnr_db"] > 30
        assert fitting["foreground_psnr_db"] > 10
        assert fitting["source_shape"] == plan.boxes[box_idx].dims
        assert fitting["source_dtype"] == "uint16"
        assert fitting["source_bytes"] == int(np.prod(plan.boxes[box_idx].dims)) * 2
        pipeline = read_node_attrs(out / "pipeline")
        z0, z1, y0, y1, x0, x1 = plan.boxes[box_idx].box
        core = stored[z0:z1, y0:y1, x0:x1].astype(np.float32)
        normalized = np.clip(
            (core - pipeline["image_min"]) / pipeline["intensity_range"], 0.0, None
        )
        assert fitting["occupancy"] == pytest.approx(
            np.count_nonzero(normalized > 0.01) / normalized.size
        )

    def test_content_source_dtype_keeps_an_explicit_config_value(self):
        from luxar.cli.gsplat_ops.planner import _fill_source_dtype

        fit_config = {"source_dtype": "uint16"}
        _fill_source_dtype(fit_config, "float32")
        assert fit_config["source_dtype"] == "uint16"

    def test_parallel_flat_merge_keeps_the_boxes_radius(self, tmp_path):
        """``fit -j N --flat``: the reloaded boxes' radius survives the merge."""
        from luxar.gsplats.gsplat_data import GSplatData

        plan = _toy_plan(n_boxes=2)
        boxes_dir = tmp_path / "boxes"
        merged = fit_planned_parallel(
            plan,
            jobs=2,
            tmp_dir=boxes_dir,
            worker_cmd_builder=_fake_box_builder(5, truncation_radius=3.5),
            keep_boxes=True,  # so the boxes' own recorded fit time can be read
            verbose=False,
        )
        assert merged.n_splats == 10
        assert merged.truncation_radius == pytest.approx(3.5)
        # ... and the planned-fit stats keys are still there afterwards.
        assert merged.stats["planned_fit"] is True
        assert merged.stats["n_boxes"] == 2
        assert merged.stats["n_boxes_fit"] == 2
        assert merged.stats["parallel_jobs"] == 2
        assert merged.stats["overlap"] == int(plan.overlap)
        assert "elapsed_seconds" in merged.stats
        # Each box really did record its own (huge) fit time in its store ...
        box_time = GSplatData.load(
            boxes_dir / "box_0.gsplats.zarr", include_stats=True
        ).stats["time_seconds"]
        assert box_time == pytest.approx(_FAKE_BOX_TIME_SECONDS)
        # ... and whatever a box recorded, `time_seconds` on the merge means one
        # thing: wall clock, as the uniform tiled merge stamps it. This fake
        # persists fitting info, so the reload and `concatenate` do bring back
        # and sum the box times; the overwrite is load-bearing here too.
        assert merged.stats["time_seconds"] == pytest.approx(
            merged.stats["elapsed_seconds"]
        )
        assert merged.stats["time_seconds"] < 60.0

    def test_parallel_flat_merge_keeps_the_box_basis(self, tmp_path):
        merged = fit_planned_parallel(
            _toy_plan(n_boxes=2),
            jobs=2,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5, image_min=500.0),
            verbose=False,
        )

        assert merged.stats["image_min"] == pytest.approx(500.0)

    def test_parallel_partition_recipe_keeps_the_box_basis(self, tmp_path, monkeypatch):
        import luxar.gsplats.lod.recipes as recipes
        from luxar.gsplats.lod.recipes import RecipeParams

        captured = []

        def fake_build(part, recipe, params, *, cell=None):
            captured.append(params.image_min)
            return part

        monkeypatch.setattr(recipes, "build_part_lod", fake_build)
        fit_planned_parallel(
            _toy_plan(n_boxes=2),
            jobs=2,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5, image_min=500.0),
            partition=True,
            recipe="levels",
            recipe_params=RecipeParams(),
            verbose=False,
        )

        assert captured == [500.0, 500.0]

    def test_parallel_partition_parts_scrub_box_scoped_stats(self, tmp_path):
        """``fit -j N`` matches sequential part provenance.

        The radius and the per-box fit stats reach a part by a different route
        than the sequential path's in-memory hand-off — through the box store: the
        leaf writer stamps a box's stats as its `lod_stats`, and the reload asks
        for top-level `stats` while also restoring them onto the sub-LOD. The
        standalone worker's remeasured score and source grid describe its box
        store, not the merged partition part, so the reload must scrub them just
        as `_fit_one_box` does before the sequential hand-off.
        """
        node = fit_planned_parallel(
            _toy_plan(n_boxes=2),
            jobs=2,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=_fake_box_builder(5, truncation_radius=3.5),
            partition=True,
            verbose=False,
        )
        leaves = _leaf_nodes(node)
        assert len(leaves) == 2
        for leaf in leaves:
            sub = leaf.additive_sublods[0]
            assert sub.truncation_radius == pytest.approx(3.5)
            assert sub.stats["time_seconds"] == pytest.approx(_FAKE_BOX_TIME_SECONDS)
            assert "psnr_db" not in sub.stats
            assert "source_shape" not in sub.stats
            assert "source_voxels" not in sub.stats


class TestMaxPaddedBoxVoxels:
    def test_returns_largest_padded_budgeted_box(self):
        # volume 16 x 16 x 48, three 16-wide x-columns, overlap 4 (clamped to vol).
        # box0 x[0:16]->pad[0:20]=20; box1 x[16:32]->pad[12:36]=24; box2 x[32:48]->[28:48]=20
        plan = _toy_plan(n_boxes=3, width=16)
        assert max_padded_box_voxels(plan) == 16 * 16 * 24  # the middle box, 6144

    def test_ignores_zero_budget_boxes(self):
        plan = _toy_plan(n_boxes=3, width=16)
        plan.boxes[1].budget = 0  # the largest padded box is now unbudgeted
        # remaining budgeted boxes pad to 20 in x -> 16*16*20
        assert max_padded_box_voxels(plan) == 16 * 16 * 20


# ── Split planes: retained by the plan, carried into the fitted partition ──
#
# The planner has always BEEN a recursive BSP; it just discarded the planes,
# leaving the fitted partition unable to say how its parts stack up and the
# viewer guessing from part centroids (not a valid painter's order — it pops at
# the seams as the camera orbits, #1555).


def _plan_leaf_labels(node: dict) -> list[int]:
    if "part" in node:
        return [node["part"]]
    return _plan_leaf_labels(node["left"]) + _plan_leaf_labels(node["right"])


def test_plan_partition_retains_its_split_planes():
    """Leaves name every box exactly once, and each plane separates its subtrees."""
    V = _corner_blobs()
    plan = plan_volume(
        V,
        _density(n_features_reference=40, k_star_reference=600),
        cell=8,
        min_leaf=16,
        max_leaf=32,
    )
    assert plan.n_boxes > 1, "need a split plan for this to test anything"
    assert plan.bsp_tree is not None
    assert sorted(_plan_leaf_labels(plan.bsp_tree)) == list(range(plan.n_boxes))

    def check(node: dict) -> None:
        if "part" in node:
            return
        axis, split = node["axis"], node["split"]
        assert axis in (0, 1, 2)
        # Boxes are half-open [lo, hi) and `left` means `coord < split`, so a
        # left box must END at or before the plane and a right box START at or
        # after it — the invariant the painter's traversal rests on.
        for label in _plan_leaf_labels(node["left"]):
            assert plan.boxes[label].box[2 * axis + 1] <= split
        for label in _plan_leaf_labels(node["right"]):
            assert plan.boxes[label].box[2 * axis] >= split
        check(node["left"])
        check(node["right"])

    check(plan.bsp_tree)


def test_plan_json_round_trips_the_split_planes():
    plan = plan_volume(
        _corner_blobs(),
        _density(n_features_reference=40, k_star_reference=600),
        cell=8,
        min_leaf=16,
        max_leaf=32,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plan.json"
        plan.to_json(path)
        assert FitPlan.from_json(path).bsp_tree == plan.bsp_tree


def test_a_plan_json_without_split_planes_still_loads():
    """A plan written before the field existed must not break the merge."""
    import json
    import tempfile

    plan = FitPlan(
        volume_shape=[8, 8, 8],
        boxes=[PlanBox(box=[0, 8, 0, 8, 0, 8], n_features=1, budget=1)],
        overlap=0,
        feature_method="peaks",
        min_leaf=4,
        max_leaf=8,
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plan.json"
        plan.to_json(path)
        raw = json.loads(path.read_text())
        raw.pop("bsp_tree", None)
        path.write_text(json.dumps(raw))
        assert FitPlan.from_json(path).bsp_tree is None


def test_fit_planned_labels_parts_by_box_not_by_position(monkeypatch):
    """Boxes that fit nothing are skipped, so the Nth PART is not the Nth BOX.

    Stubs the per-box fit (the real one is far too slow for a unit test) and
    makes box 1 produce nothing: the emitted tree must name the two survivors
    0 and 1, with the plane that separated boxes 1 and 2 collapsed away.
    """
    import importlib

    from luxar.gsplats.tree import GSplatPartition

    # import_module, not `import ... as`: the package re-exports the FUNCTION
    # `fit_planned`, which shadows the submodule of the same name.
    fit_planned_mod = importlib.import_module("luxar.gsplats.planner.fit_planned")

    plan = FitPlan(
        volume_shape=[30, 4, 4],
        boxes=[
            PlanBox(box=[0, 10, 0, 4, 0, 4], n_features=1, budget=10),
            PlanBox(box=[10, 20, 0, 4, 0, 4], n_features=1, budget=10),
            PlanBox(box=[20, 30, 0, 4, 0, 4], n_features=1, budget=10),
        ],
        overlap=0,
        feature_method="peaks",
        min_leaf=4,
        max_leaf=16,
        bsp_tree={
            "axis": 0,
            "split": 10.0,
            "left": {"part": 0},
            "right": {
                "axis": 0,
                "split": 20.0,
                "left": {"part": 1},
                "right": {"part": 2},
            },
        },
    )

    def fake_fit_one_box(V, b, pad, cap, **kwargs):
        from luxar.gsplats.gsplat_data import GSplatData

        z0, z1 = b.box[0], b.box[1]
        if z0 == 10:  # box 1 legitimately yields nothing
            return GSplatData(
                centers=np.zeros((0, 3), np.float32),
                amplitudes=np.zeros((0,), np.float32),
                cholesky_factors=np.zeros((0, 6), np.float32),
            )
        centers = np.stack(
            [
                np.linspace(z0 + 1, z1 - 1, 5),
                np.full(5, 2.0),
                np.full(5, 2.0),
            ],
            axis=1,
        ).astype(np.float32)
        chol = np.zeros((5, 6), np.float32)
        chol[:, [0, 2, 5]] = 1.0
        return GSplatData(
            centers=centers,
            amplitudes=np.full(5, 0.5, np.float32),
            cholesky_factors=chol,
        )

    monkeypatch.setattr(fit_planned_mod, "_fit_one_box", fake_fit_one_box)

    node = fit_planned_mod.fit_planned(
        np.zeros((30, 4, 4), np.float32), plan, partition=True, verbose=False
    )
    assert isinstance(node, GSplatPartition)
    assert node.n_children == 2
    assert node.bsp_tree == {
        "axis": 0,
        "split": 10.0,
        "left": {"part": 0},
        "right": {"part": 1},
    }
