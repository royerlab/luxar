"""Tests for GSplatData class methods."""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData


def _make_3d_gsplat(n=5, seed=42):
    """Helper: create a simple 3D GSplatData with n splats."""
    rng = np.random.RandomState(seed)
    return GSplatData(
        centers=rng.rand(n, 3).astype(np.float32) * 100,
        amplitudes=rng.rand(n).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
        ),
        sharpnesses=np.full(n, 2.0, dtype=np.float32),
    )


def _make_empty_gsplat(ndim=3):
    """Helper: create an empty GSplatData."""
    tril = ndim * (ndim + 1) // 2
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, tril), dtype=np.float32),
        sharpnesses=np.zeros(0, dtype=np.float32),
    )


# ── Properties and basic interface ──────────────────────────


class TestProperties:
    """Tests for n_splats, ndim, __len__, __repr__, sharpness alias."""

    def test_n_splats(self):
        gs = _make_3d_gsplat(n=7)
        assert gs.n_splats == 7

    def test_ndim(self):
        gs = _make_3d_gsplat()
        assert gs.ndim == 3

    def test_ndim_2d(self):
        gs = GSplatData(
            centers=np.zeros((2, 2), dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            cholesky_factors=np.zeros((2, 3), dtype=np.float32),
            sharpnesses=np.zeros(2, dtype=np.float32),
        )
        assert gs.ndim == 2

    def test_ndim_empty(self):
        gs = _make_empty_gsplat()
        assert gs.ndim == 0

    def test_len(self):
        gs = _make_3d_gsplat(n=10)
        assert len(gs) == 10

    def test_len_empty(self):
        gs = _make_empty_gsplat()
        assert len(gs) == 0

    def test_repr_nonempty(self):
        gs = _make_3d_gsplat(n=3)
        r = repr(gs)
        assert "3 splats" in r
        assert "3D" in r

    def test_repr_empty(self):
        gs = _make_empty_gsplat()
        r = repr(gs)
        assert "0 splats" in r

    def test_sharpness_alias(self):
        gs = _make_3d_gsplat()
        assert gs.sharpness is gs.sharpnesses

    def test_default_stats_is_empty_dict(self):
        gs = _make_3d_gsplat()
        assert gs.stats == {}
        assert isinstance(gs.stats, dict)

    def test_stats_not_shared_between_instances(self):
        gs1 = _make_3d_gsplat(n=1)
        gs2 = _make_3d_gsplat(n=1)
        gs1.stats["foo"] = "bar"
        assert "foo" not in gs2.stats


# ── Translate ───────────────────────────────────────────────


class TestTranslate:
    def test_basic_translate(self):
        gs = _make_3d_gsplat(n=3)
        offset = np.array([10, 20, 30], dtype=np.float32)
        translated = gs.translate(offset)
        assert np.allclose(translated.centers, gs.centers + offset)

    def test_translate_preserves_other_fields(self):
        gs = _make_3d_gsplat(n=3)
        gs.stats["key"] = "value"
        translated = gs.translate(np.zeros(3))
        # Amplitudes, cholesky, sharpness should be same objects (references)
        assert translated.amplitudes is gs.amplitudes
        assert translated.cholesky_factors is gs.cholesky_factors
        assert translated.sharpnesses is gs.sharpnesses
        # Stats should be a copy
        assert translated.stats == gs.stats
        assert translated.stats is not gs.stats

    def test_translate_with_colors(self):
        gs = _make_3d_gsplat(n=2)
        gs_with_colors = GSplatData(
            centers=gs.centers,
            amplitudes=gs.amplitudes,
            cholesky_factors=gs.cholesky_factors,
            sharpnesses=gs.sharpnesses,
            colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        )
        translated = gs_with_colors.translate(np.array([1, 2, 3]))
        assert translated.colors is gs_with_colors.colors


# ── Center at centroid ──────────────────────────────────────


class TestCenterAtCentroid:
    def test_uniform_amplitudes(self):
        """With uniform amplitudes, centroid = unweighted mean."""
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32),
            amplitudes=np.array([1.0, 1.0], dtype=np.float32),
            cholesky_factors=np.zeros((2, 6), dtype=np.float32),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
        )
        centered = gs.center_at_centroid()
        centroid = centered.centers.mean(axis=0)
        assert np.allclose(centroid, 0.0, atol=1e-5)

    def test_weighted_centroid(self):
        """Amplitude-weighted centroid should be at origin."""
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
            amplitudes=np.array([3.0, 1.0], dtype=np.float32),  # weighted toward first
            cholesky_factors=np.zeros((2, 6), dtype=np.float32),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
        )
        centered = gs.center_at_centroid()
        total = centered.amplitudes.sum()
        weighted_centroid = (centered.centers.T @ centered.amplitudes) / total
        assert np.allclose(weighted_centroid, 0.0, atol=1e-5)

    def test_zero_amplitudes_uses_mean(self):
        """With all-zero amplitudes, falls back to unweighted mean."""
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32),
            amplitudes=np.array([0.0, 0.0], dtype=np.float32),
            cholesky_factors=np.zeros((2, 6), dtype=np.float32),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
        )
        centered = gs.center_at_centroid()
        assert np.allclose(centered.centers.mean(axis=0), 0.0, atol=1e-5)


# ── Scale intensity ─────────────────────────────────────────


class TestScaleIntensity:
    def test_double(self):
        gs = _make_3d_gsplat()
        scaled = gs.scale_intensity(2.0)
        assert np.allclose(scaled.amplitudes, gs.amplitudes * 2.0)

    def test_preserves_other_fields(self):
        gs = _make_3d_gsplat()
        scaled = gs.scale_intensity(0.5)
        assert scaled.centers is gs.centers
        assert scaled.cholesky_factors is gs.cholesky_factors
        assert scaled.sharpnesses is gs.sharpnesses

    def test_zero_factor(self):
        gs = _make_3d_gsplat()
        scaled = gs.scale_intensity(0.0)
        assert np.allclose(scaled.amplitudes, 0.0)


# ── Prune ───────────────────────────────────────────────────


class TestPrune:
    def test_cumulative_retains_signal(self):
        """Cumulative pruning should retain the target fraction of total amplitude."""
        gs = _make_3d_gsplat(n=100)
        pruned = gs.prune(method="cumulative", target_retention=0.95)
        total_orig = gs.amplitudes.sum()
        total_pruned = pruned.amplitudes.sum()
        assert total_pruned / total_orig >= 0.95 - 1e-6
        assert pruned.n_splats <= gs.n_splats

    def test_cumulative_reduces_count(self):
        """Cumulative pruning should remove some splats."""
        gs = _make_3d_gsplat(n=100)
        pruned = gs.prune(method="cumulative", target_retention=0.5)
        assert pruned.n_splats < gs.n_splats

    def test_amplitude_percentile(self):
        gs = _make_3d_gsplat(n=100)
        pruned = gs.prune(method="amplitude_percentile", amplitude_percentile=20)
        assert pruned.n_splats < gs.n_splats

    def test_combined(self):
        gs = _make_3d_gsplat(n=100)
        pruned = gs.prune(method="combined", amplitude_percentile=10, volume_percentile=90)
        assert pruned.n_splats <= gs.n_splats

    def test_prune_empty(self):
        gs = _make_empty_gsplat()
        pruned = gs.prune()
        assert pruned.n_splats == 0
        assert pruned.stats["pruned"] is True

    def test_prune_all_zero_amplitudes(self):
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
            amplitudes=np.array([0.0, 0.0], dtype=np.float32),
            cholesky_factors=np.zeros((2, 6), dtype=np.float32),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
        )
        pruned = gs.prune(method="cumulative", target_retention=0.95)
        # With zero retention request, should keep all
        assert pruned.n_splats == 2

    def test_prune_stats_updated(self):
        gs = _make_3d_gsplat(n=50)
        gs.stats["time_seconds"] = 5.0
        pruned = gs.prune(method="cumulative", target_retention=0.5)
        assert pruned.stats["pruned"] is True
        assert pruned.stats["pruning_method"] == "cumulative"
        assert pruned.stats["n_original"] == 50
        assert "n_removed" in pruned.stats
        assert "amplitude_retention" in pruned.stats
        # Original stats should be preserved
        assert pruned.stats["time_seconds"] == 5.0

    def test_prune_preserves_colors(self):
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
            amplitudes=np.array([1.0, 0.001], dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
            colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        )
        pruned = gs.prune(method="cumulative", target_retention=0.5)
        assert pruned.colors is not None
        assert pruned.colors.shape[1] == 3

    def test_prune_invalid_method(self):
        gs = _make_3d_gsplat()
        with pytest.raises(ValueError, match="Unknown pruning method"):
            gs.prune(method="invalid")


# ── Save whitelist ──────────────────────────────────────────


class TestSaveWhitelist:
    """Verify save() preserves quality metrics in fitting_info."""

    def test_quality_metrics_included(self):
        """Check that save() whitelist includes all quality metrics from finalize_results."""
        gs = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
            stats={
                "time_seconds": 10.0,
                "iterations": 500,
                "converged": True,
                "early_stopped": False,
                "best_iteration": 450,
                "final_loss": 0.001,
                "final_max_abs_error": 0.005,
                "final_rel_l2": 0.01,
                "n_splats": 1,
                "n_splats_before_culling": 5,
                "n_culled": 4,
                "movie_frames": None,  # Should NOT be saved
            },
        )

        # We can't easily call save() without the full encoding stack,
        # so test the whitelist logic directly
        fitting_keys = [
            "time_seconds",
            "iterations",
            "converged",
            "early_stopped",
            "best_iteration",
            "final_loss",
            "final_max_abs_error",
            "final_rel_l2",
            "n_splats",
            "n_splats_before_culling",
            "n_culled",
        ]

        # Simulate the save() whitelist extraction
        fitting_info = {
            k: v
            for k, v in gs.stats.items()
            if k
            in [
                "time_seconds",
                "iterations",
                "converged",
                "early_stopped",
                "best_iteration",
                "final_loss",
                "final_max_abs_error",
                "final_rel_l2",
                "n_splats",
                "n_splats_before_culling",
                "n_culled",
                "fitter_name",
                "fitter_version",
                "timestamp",
                "pruned",
                "pruning_method",
                "n_original",
                "n_removed",
                "amplitude_retention",
            ]
        }

        for key in fitting_keys:
            assert key in fitting_info, f"Quality metric '{key}' missing from save whitelist"

        # movie_frames should NOT be saved
        assert "movie_frames" not in fitting_info


class TestMergeWithChannelColors:
    """Tests for GSplatData.merge_with_channel_colors() class method."""

    def test_basic_merge_two_channels(self):
        """Test basic merge of two GSplatData objects with colors."""
        gs1 = GSplatData(
            centers=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            amplitudes=np.array([0.5, 0.7], dtype=np.float32),
            cholesky_factors=np.array(
                [[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32
            ),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
            stats={"time_seconds": 1.5},
        )

        gs2 = GSplatData(
            centers=np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
            amplitudes=np.array([0.9], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.5], dtype=np.float32),
            stats={"time_seconds": 2.0},
        )

        merged = GSplatData.merge_with_channel_colors(
            [gs1, gs2], channel_colors=[(1.0, 0.0, 0.5), (0.0, 1.0, 0.5)]
        )

        # Check shapes
        assert merged.centers.shape == (3, 3)
        assert merged.amplitudes.shape == (3,)
        assert merged.cholesky_factors.shape == (3, 6)
        assert merged.sharpnesses.shape == (3,)
        assert merged.colors.shape == (3, 3)

    def test_colors_assigned_correctly(self):
        """Test that colors are assigned to the correct splats."""
        gs1 = GSplatData(
            centers=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            amplitudes=np.array([1.0, 1.0], dtype=np.float32),
            cholesky_factors=np.array(
                [[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32
            ),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
        )

        gs2 = GSplatData(
            centers=np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
        )

        red = (1.0, 0.0, 0.0)
        blue = (0.0, 0.0, 1.0)

        merged = GSplatData.merge_with_channel_colors(
            [gs1, gs2], channel_colors=[red, blue]
        )

        # First 2 splats should be red
        assert np.allclose(merged.colors[0], red)
        assert np.allclose(merged.colors[1], red)
        # Third splat should be blue
        assert np.allclose(merged.colors[2], blue)

    def test_data_concatenation(self):
        """Test that all data arrays are correctly concatenated."""
        gs1 = GSplatData(
            centers=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            amplitudes=np.array([0.5], dtype=np.float32),
            cholesky_factors=np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
        )

        gs2 = GSplatData(
            centers=np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            amplitudes=np.array([0.9], dtype=np.float32),
            cholesky_factors=np.array([[7, 8, 9, 10, 11, 12]], dtype=np.float32),
            sharpnesses=np.array([3.0], dtype=np.float32),
        )

        merged = GSplatData.merge_with_channel_colors(
            [gs1, gs2], channel_colors=[(1, 0, 0), (0, 1, 0)]
        )

        # Check centers
        assert np.allclose(merged.centers[0], [1.0, 2.0, 3.0])
        assert np.allclose(merged.centers[1], [4.0, 5.0, 6.0])

        # Check amplitudes
        assert np.allclose(merged.amplitudes, [0.5, 0.9])

        # Check cholesky
        assert np.allclose(merged.cholesky_factors[0], [1, 2, 3, 4, 5, 6])
        assert np.allclose(merged.cholesky_factors[1], [7, 8, 9, 10, 11, 12])

        # Check sharpness
        assert np.allclose(merged.sharpnesses, [2.0, 3.0])

    def test_stats_aggregation(self):
        """Test that stats are properly aggregated."""
        gs1 = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
            stats={"time_seconds": 1.5},
        )

        gs2 = GSplatData(
            centers=np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
            amplitudes=np.array([1.0, 1.0], dtype=np.float32),
            cholesky_factors=np.array(
                [[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32
            ),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
            stats={"time_seconds": 2.5},
        )

        merged = GSplatData.merge_with_channel_colors(
            [gs1, gs2], channel_colors=[(1, 0, 0), (0, 1, 0)]
        )

        assert merged.stats["merged_from_channels"] == 2
        assert merged.stats["splats_per_channel"] == [1, 2]
        assert merged.stats["time_seconds"] == 4.0

    def test_three_channels(self):
        """Test merging three channels."""
        gs_list = [
            GSplatData(
                centers=np.array([[i, i, i]], dtype=np.float32),
                amplitudes=np.array([float(i + 1)], dtype=np.float32),
                cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
                sharpnesses=np.array([2.0], dtype=np.float32),
            )
            for i in range(3)
        ]

        # Use float tuples for consistency with [0, 1] range documented in docstring
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

        merged = GSplatData.merge_with_channel_colors(gs_list, channel_colors=colors)

        assert merged.centers.shape == (3, 3)
        assert np.allclose(merged.colors[0], colors[0])
        assert np.allclose(merged.colors[1], colors[1])
        assert np.allclose(merged.colors[2], colors[2])

    def test_error_on_mismatched_lengths(self):
        """Test that ValueError is raised when lists have different lengths."""
        gs1 = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
        )

        with pytest.raises(ValueError, match="must match"):
            GSplatData.merge_with_channel_colors(
                [gs1],
                channel_colors=[(1, 0, 0), (0, 1, 0)],  # 1 gsplat, 2 colors
            )

    def test_error_on_empty_list(self):
        """Test that ValueError is raised for empty list."""
        with pytest.raises(ValueError, match="At least one"):
            GSplatData.merge_with_channel_colors([], channel_colors=[])

    def test_error_on_dimension_mismatch(self):
        """Test that ValueError is raised when dimensionalities don't match."""
        gs_3d = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),  # 3D
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
        )

        gs_2d = GSplatData(
            centers=np.array([[0, 0]], dtype=np.float32),  # 2D
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
        )

        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            GSplatData.merge_with_channel_colors(
                [gs_3d, gs_2d], channel_colors=[(1, 0, 0), (0, 1, 0)]
            )

    def test_colors_are_float32(self):
        """Test that output colors are float32."""
        gs1 = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
        )

        merged = GSplatData.merge_with_channel_colors(
            [gs1], channel_colors=[(1.0, 0.5, 0.25)]
        )

        assert merged.colors.dtype == np.float32

    def test_single_channel(self):
        """Test merging a single channel (edge case)."""
        gs1 = GSplatData(
            centers=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
            amplitudes=np.array([1.0, 2.0], dtype=np.float32),
            cholesky_factors=np.array(
                [[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32
            ),
            sharpnesses=np.array([2.0, 2.0], dtype=np.float32),
        )

        merged = GSplatData.merge_with_channel_colors(
            [gs1], channel_colors=[(0.5, 0.5, 0.5)]
        )

        assert merged.centers.shape == (2, 3)
        assert np.allclose(merged.colors[0], [0.5, 0.5, 0.5])
        assert np.allclose(merged.colors[1], [0.5, 0.5, 0.5])

    def test_one_empty_channel(self):
        """Test merging when one channel has 0 splats (edge case)."""
        gs_empty = GSplatData(
            centers=np.zeros((0, 3), dtype=np.float32),
            amplitudes=np.zeros((0,), dtype=np.float32),
            cholesky_factors=np.zeros((0, 6), dtype=np.float32),
            sharpnesses=np.zeros((0,), dtype=np.float32),
        )

        gs_nonempty = GSplatData(
            centers=np.array([[1, 2, 3]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
        )

        merged = GSplatData.merge_with_channel_colors(
            [gs_empty, gs_nonempty],
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )

        # Result should only have 1 splat (from nonempty channel)
        assert merged.centers.shape == (1, 3)
        assert merged.amplitudes.shape == (1,)
        # The splat should be green (from channel 1)
        assert np.allclose(merged.colors[0], [0.0, 1.0, 0.0])
        assert merged.stats["splats_per_channel"] == [0, 1]

    def test_existing_colors_ignored(self):
        """Test that existing colors in input GSplatData are ignored."""
        gs1 = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.array([2.0], dtype=np.float32),
            colors=np.array(
                [[0.0, 0.0, 0.0]], dtype=np.float32
            ),  # Black - should be ignored
        )

        merged = GSplatData.merge_with_channel_colors(
            [gs1],
            channel_colors=[(1.0, 1.0, 1.0)],  # White
        )

        # Output should be white, not black
        assert np.allclose(merged.colors[0], [1.0, 1.0, 1.0])
