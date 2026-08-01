"""Unit tests for the pure helpers in demo_dmri_tractography.

Exercises only the network-free / IO-free helpers (arc-length resampling,
direction colouring, the RAS -> scene rotation, the indexed edge builder, and
the deterministic subsampler). The demo is loaded by file path so importing it
never triggers the module-level ``parse_demo_flags`` scan of pytest's argv.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_dmri_tractography.py"


def _load_demo_module():
    name = "_luxar_demo_tractography_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


class TestResamplePolyline:
    def test_output_shape_and_dtype(self) -> None:
        pts = np.cumsum(np.ones((17, 3)), axis=0)
        out = _demo.resample_polyline(pts, 28)
        assert out.shape == (28, 3)
        assert out.dtype == np.float32

    def test_endpoints_are_preserved(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 5.0, 0.0], [10.0, 2.0, -3.0]])
        out = _demo.resample_polyline(pts, 40)
        np.testing.assert_allclose(out[0], pts[0], atol=1e-5)
        np.testing.assert_allclose(out[-1], pts[-1], atol=1e-5)

    def test_points_are_evenly_spaced_in_arc_length(self) -> None:
        # A straight line sampled with deliberately UNEVEN input spacing: an
        # index-space resample would inherit the clustering, an arc-length one
        # must not. This is the property the whole helper exists for.
        t = np.array([0.0, 0.01, 0.02, 0.03, 1.0])
        pts = np.column_stack([t * 100.0, np.zeros(5), np.zeros(5)])
        out = _demo.resample_polyline(pts, 21)
        step = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert step.std() < 1e-3
        np.testing.assert_allclose(step.mean(), 5.0, rtol=1e-4)

    def test_curved_path_stays_on_the_curve(self) -> None:
        theta = np.linspace(0.0, np.pi, 200)
        pts = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        out = _demo.resample_polyline(pts, 32)
        radius = np.linalg.norm(out[:, :2], axis=1)
        np.testing.assert_allclose(radius, 1.0, atol=2e-3)

    def test_degenerate_zero_length_streamline(self) -> None:
        pts = np.zeros((6, 3))
        out = _demo.resample_polyline(pts, 12)
        assert out.shape == (12, 3)
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0)

    @pytest.mark.parametrize(
        "pts,n",
        [
            (np.zeros((1, 3)), 8),  # too few input points
            (np.zeros((5, 2)), 8),  # not 3D
            (np.zeros((5, 3)), 1),  # too few output points
        ],
    )
    def test_rejects_bad_input(self, pts: np.ndarray, n: int) -> None:
        with pytest.raises(ValueError):
            _demo.resample_polyline(pts, n)


class TestDirectionColors:
    def test_axis_aligned_paths_get_pure_channels(self) -> None:
        # R = left-right (x), G = anterior-posterior (y), B = inferior-superior (z).
        steps = np.linspace(0.0, 10.0, 12)
        zero = np.zeros_like(steps)
        for axis, channel in ((0, 0), (1, 1), (2, 2)):
            cols = [zero, zero, zero]
            cols[axis] = steps
            path = np.column_stack(cols)[None, ...]
            rgb = _demo.direction_colors(path)
            assert rgb.shape == (1, 12, 3)
            assert np.all(rgb[..., channel] == 255)
            for other in range(3):
                if other != channel:
                    assert np.all(rgb[..., other] == 0)

    def test_is_orientation_only_not_direction(self) -> None:
        # Reversing a streamline must not change its colour — that is the whole
        # point of taking |tangent|.
        path = np.cumsum(np.random.default_rng(0).normal(size=(1, 30, 3)), axis=1)
        forward = _demo.direction_colors(path)
        backward = _demo.direction_colors(path[:, ::-1])
        np.testing.assert_array_equal(forward, backward[:, ::-1])

    def test_diagonal_splits_channels(self) -> None:
        steps = np.linspace(0.0, 10.0, 9)
        path = np.column_stack([steps, steps, np.zeros_like(steps)])[None, ...]
        rgb = _demo.direction_colors(path)
        expected = round(255.0 / np.sqrt(2.0))
        assert np.all(np.abs(rgb[..., 0].astype(int) - expected) <= 1)
        assert np.all(np.abs(rgb[..., 1].astype(int) - expected) <= 1)
        assert np.all(rgb[..., 2] == 0)

    def test_coincident_vertices_do_not_produce_nan(self) -> None:
        path = np.zeros((1, 10, 3))
        rgb = _demo.direction_colors(path)
        assert rgb.dtype == np.uint8
        assert np.all(rgb == 0)

    @pytest.mark.parametrize(
        "bad", [np.zeros((4, 3)), np.zeros((2, 1, 3)), np.zeros((2, 5, 2))]
    )
    def test_rejects_bad_input(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError):
            _demo.direction_colors(bad)


class TestRasToScene:
    def test_superior_maps_to_plus_y(self) -> None:
        # RAS +z (superior) must become scene +y so the brain is upright.
        out = _demo.ras_to_scene(np.array([[0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-6)

    def test_right_stays_plus_x_and_anterior_goes_minus_z(self) -> None:
        out = _demo.ras_to_scene(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[1], [0.0, 0.0, -1.0], atol=1e-6)

    def test_is_a_rigid_rotation(self) -> None:
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(50, 3))
        out = _demo.ras_to_scene(pts)
        np.testing.assert_allclose(
            np.linalg.norm(out, axis=1), np.linalg.norm(pts, axis=1), rtol=1e-6
        )

    def test_preserves_leading_shape(self) -> None:
        assert _demo.ras_to_scene(np.zeros((4, 7, 3))).shape == (4, 7, 3)

    def test_rejects_non_3_last_axis(self) -> None:
        with pytest.raises(ValueError):
            _demo.ras_to_scene(np.zeros((5, 2)))


class TestPolylineSegmentIndices:
    def test_count_and_dtype(self) -> None:
        idx = _demo.polyline_segment_indices(7, 28)
        assert idx.dtype == np.uint32
        assert len(idx) == 2 * 7 * 27

    def test_no_edge_crosses_a_streamline_boundary(self) -> None:
        n_paths, n_vertices = 11, 9
        edges = _demo.polyline_segment_indices(n_paths, n_vertices).reshape(-1, 2)
        path_of_vertex = np.repeat(np.arange(n_paths), n_vertices)
        assert np.all(path_of_vertex[edges[:, 0]] == path_of_vertex[edges[:, 1]]), (
            "an edge joined two different streamlines"
        )

    def test_every_vertex_referenced_and_joints_shared(self) -> None:
        n_paths, n_vertices = 6, 12
        edges = _demo.polyline_segment_indices(n_paths, n_vertices).reshape(-1, 2)
        degree = np.bincount(edges.reshape(-1), minlength=n_paths * n_vertices)
        assert np.all(degree > 0), "emitted an unreferenced vertex"
        # Interior vertices have degree 2 (shared joint); the two ends have 1.
        assert int((degree == 1).sum()) == 2 * n_paths
        assert int((degree == 2).sum()) == n_paths * (n_vertices - 2)

    def test_indices_are_in_range(self) -> None:
        n_paths, n_vertices = 5, 20
        idx = _demo.polyline_segment_indices(n_paths, n_vertices)
        assert int(idx.min()) >= 0
        assert int(idx.max()) < n_paths * n_vertices

    def test_rejects_single_vertex_paths(self) -> None:
        with pytest.raises(ValueError):
            _demo.polyline_segment_indices(4, 1)


class TestSubsampleIndices:
    def test_returns_all_when_under_the_cap(self) -> None:
        out = _demo.subsample_indices(500, 6000, seed=0)
        np.testing.assert_array_equal(out, np.arange(500))

    def test_caps_and_stays_sorted_and_unique(self) -> None:
        out = _demo.subsample_indices(50_000, 6_000, seed=0)
        assert len(out) == 6_000
        assert np.all(np.diff(out) > 0)
        assert int(out.max()) < 50_000

    def test_is_deterministic_for_a_given_seed(self) -> None:
        a = _demo.subsample_indices(10_000, 100, seed=7)
        b = _demo.subsample_indices(10_000, 100, seed=7)
        c = _demo.subsample_indices(10_000, 100, seed=8)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


class TestNodeBudget:
    """The two viewer limits the sizing constants exist to respect."""

    #: scripts/check_demo_ladders.py fails an un-laddered lines leaf above this.
    LADDER_GATE_VERTICES = 200_000
    #: 682 * maxTextureSize segments per node, on a 4096-class GPU.
    TEXTURE_CAP_SEGMENTS = 682 * 4096

    def test_defaults_keep_a_node_under_both_limits(self) -> None:
        vertices = _demo.DEFAULT_PER_BUNDLE * _demo.DEFAULT_POINTS
        segments = _demo.DEFAULT_PER_BUNDLE * (_demo.DEFAULT_POINTS - 1)
        assert vertices < self.LADDER_GATE_VERTICES
        assert segments < self.TEXTURE_CAP_SEGMENTS

    def test_divisions_cover_the_atlas_layout(self) -> None:
        assert set(_demo.DIVISIONS) == {
            "association",
            "projection",
            "commissural",
            "cerebellum",
            "cranial nerve",
        }
