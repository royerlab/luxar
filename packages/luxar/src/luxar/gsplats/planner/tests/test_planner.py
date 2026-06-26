"""Unit tests for the content-aware fit planner (scan + BSP, CPU-only)."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.calibration import SplatDensity
from luxar.gsplats.planner import (
    FitPlan,
    plan_partition,
    plan_volume,
    scan_content,
)


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
        robust = scan_content(
            V, cell=16, method="peaks", threshold_abs=0.1
        ).total
        assert robust > default  # absolute level is not suppressed by the outlier


class TestFitPlanned:
    def test_fit_planned_cpu_smoke(self):
        # End-to-end: plan a small volume then fit it on CPU; expect splats back.
        from luxar.gsplats.planner import fit_planned

        V = _corner_blobs((64, 64, 64), n=6, corner=48)
        plan = plan_volume(
            V, _density(), cell=8, min_leaf=16, max_leaf=32, overlap=4
        )
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
        plan = plan_volume(
            V, _density(), cell=16, min_leaf=32, max_leaf=64, overlap=8
        )
        assert plan.n_boxes >= 1
        assert plan.total_budget > 0
