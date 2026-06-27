"""Unit tests for the content-aware fit planner (scan + BSP, CPU-only)."""

from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.calibration import SplatDensity
from luxar.gsplats.planner import (
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


def _fake_box_builder(n_per_box: int = 5):
    """Worker builder writing ``n_per_box`` deterministic splats — no torch/GPU."""

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
                       cholesky_factors=chol).save(r"{out_path}")
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


class TestFitPlannedParallel:
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
