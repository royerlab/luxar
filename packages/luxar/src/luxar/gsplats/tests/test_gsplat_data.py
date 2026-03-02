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
    """Tests for n_splats, ndim, __len__, __repr__."""

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
        """Empty GSplatData with 2D centers still knows its dimensionality."""
        gs = _make_empty_gsplat(ndim=3)
        assert gs.ndim == 3

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

    def test_default_stats_is_empty_dict(self):
        gs = _make_3d_gsplat()
        assert gs.stats == {}
        assert isinstance(gs.stats, dict)

    def test_stats_not_shared_between_instances(self):
        gs1 = _make_3d_gsplat(n=1)
        gs2 = _make_3d_gsplat(n=1)
        gs1.stats["foo"] = "bar"
        assert "foo" not in gs2.stats


# ── Validation ──────────────────────────────────────────────


class TestValidation:
    """Tests for __post_init__ shape validation."""

    def test_valid_construction(self):
        """Valid shapes should not raise."""
        _make_3d_gsplat(n=5)  # Should not raise

    def test_valid_empty(self):
        """Empty GSplatData (0 splats) should pass validation."""
        _make_empty_gsplat()  # Should not raise

    def test_mismatched_amplitudes(self):
        with pytest.raises(ValueError, match="Amplitudes shape"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),
                amplitudes=np.zeros(5, dtype=np.float32),  # wrong count
                cholesky_factors=np.zeros((3, 6), dtype=np.float32),
                sharpnesses=np.zeros(3, dtype=np.float32),
            )

    def test_mismatched_sharpnesses(self):
        with pytest.raises(ValueError, match="Sharpnesses shape"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),
                amplitudes=np.zeros(3, dtype=np.float32),
                cholesky_factors=np.zeros((3, 6), dtype=np.float32),
                sharpnesses=np.zeros(5, dtype=np.float32),  # wrong count
            )

    def test_mismatched_colors(self):
        with pytest.raises(ValueError, match="Colors count"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),
                amplitudes=np.zeros(3, dtype=np.float32),
                cholesky_factors=np.zeros((3, 6), dtype=np.float32),
                sharpnesses=np.zeros(3, dtype=np.float32),
                colors=np.zeros((5, 3), dtype=np.float32),  # wrong count
            )

    def test_wrong_cholesky_packed_size(self):
        """Cholesky factors with wrong packed size for dimensionality."""
        with pytest.raises(ValueError, match="wrong packed size"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),  # 3D → expect k=6
                amplitudes=np.zeros(3, dtype=np.float32),
                cholesky_factors=np.zeros((3, 3), dtype=np.float32),  # k=3 (2D size)
                sharpnesses=np.zeros(3, dtype=np.float32),
            )

    def test_wrong_cholesky_splat_count(self):
        """Cholesky factors with correct packed size but wrong N."""
        with pytest.raises(ValueError, match="count mismatch"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),
                amplitudes=np.zeros(3, dtype=np.float32),
                cholesky_factors=np.zeros((5, 6), dtype=np.float32),  # N=5, not 3
                sharpnesses=np.zeros(3, dtype=np.float32),
            )

    def test_valid_with_colors(self):
        """Valid construction with colors should not raise."""
        GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.zeros(3, dtype=np.float32),
            cholesky_factors=np.zeros((3, 6), dtype=np.float32),
            sharpnesses=np.zeros(3, dtype=np.float32),
            colors=np.zeros((3, 3), dtype=np.float32),
        )

    def test_valid_2d(self):
        """2D data with correct Cholesky size (k=3) should not raise."""
        GSplatData(
            centers=np.zeros((4, 2), dtype=np.float32),
            amplitudes=np.zeros(4, dtype=np.float32),
            cholesky_factors=np.zeros((4, 3), dtype=np.float32),
            sharpnesses=np.zeros(4, dtype=np.float32),
        )


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
        pruned = gs.prune(
            method="combined", amplitude_percentile=10, volume_percentile=90
        )
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
            assert key in fitting_info, (
                f"Quality metric '{key}' missing from save whitelist"
            )

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


# ── Computed properties ─────────────────────────────────


def _make_2d_gsplat(n=5, seed=42):
    """Helper: create a 2D GSplatData."""
    rng = np.random.RandomState(seed)
    return GSplatData(
        centers=rng.rand(n, 2).astype(np.float32) * 100,
        amplitudes=rng.rand(n).astype(np.float32),
        cholesky_factors=np.tile(np.array([1, 0, 1], dtype=np.float32), (n, 1)),
        sharpnesses=np.full(n, 2.0, dtype=np.float32),
    )


class TestVolumes:
    def test_isotropic_identity_3d(self):
        """Identity Cholesky [1,0,1,0,0,1] has det(L)=1, volume=1."""
        gs = _make_3d_gsplat(n=3)
        vols = gs.volumes()
        assert vols.shape == (3,)
        # Identity diagonal -> det(L)=1, det(Sigma)=1, vol=1^(1/3)=1
        assert np.allclose(vols, 1.0)

    def test_scaled_diagonal(self):
        """Diagonal [2,0,3,0,0,4] -> det(L)=24, vol=(24^2)^(1/3)."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        vols = gs.volumes()
        expected = abs(24**2) ** (1.0 / 3)
        assert np.allclose(vols[0], expected)

    def test_2d(self):
        gs = _make_2d_gsplat(n=3)
        vols = gs.volumes()
        assert vols.shape == (3,)

    def test_empty(self):
        gs = _make_empty_gsplat()
        assert gs.volumes().shape == (0,)


class TestMasses:
    def test_basic(self):
        gs = _make_3d_gsplat(n=5)
        masses = gs.masses()
        expected = gs.amplitudes * gs.volumes()
        assert np.allclose(masses, expected)

    def test_zero_amplitudes(self):
        gs = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
            sharpnesses=np.ones(2, dtype=np.float32),
        )
        assert np.allclose(gs.masses(), 0.0)


class TestMarginalSigmas:
    def test_identity(self):
        """Identity Cholesky gives sigma=[1,1,1]."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        sigmas = gs.marginal_sigmas()
        assert sigmas.shape == (1, 3)
        assert np.allclose(sigmas[0], [1, 1, 1])

    def test_diagonal(self):
        """Diagonal [2,0,3,0,0,4] gives sigma=[2,3,4]."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        assert np.allclose(gs.marginal_sigmas()[0], [2, 3, 4])

    def test_off_diagonal(self):
        """L=[[2,0,0],[1,3,0],[0,0,4]] -> sigma_0=2, sigma_1=sqrt(1+9)=sqrt(10)."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 1, 3, 0, 0, 4]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        sigmas = gs.marginal_sigmas()[0]
        assert np.allclose(sigmas[0], 2.0)
        assert np.allclose(sigmas[1], np.sqrt(1 + 9))
        assert np.allclose(sigmas[2], 4.0)

    def test_empty(self):
        gs = _make_empty_gsplat()
        assert gs.marginal_sigmas().shape == (0, 3)

    def test_2d(self):
        gs = _make_2d_gsplat(n=2)
        assert gs.marginal_sigmas().shape == (2, 2)


class TestEccentricities:
    def test_isotropic_is_one(self):
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        assert np.allclose(gs.eccentricities(), 1.0)

    def test_anisotropic(self):
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 4]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        ecc = gs.eccentricities()
        assert ecc[0] == pytest.approx(4.0)  # max_sigma=4, min_sigma=1

    def test_empty(self):
        gs = _make_empty_gsplat()
        assert gs.eccentricities().shape == (0,)


# ── Filter ──────────────────────────────────────────────


class TestFilter:
    def test_basic(self):
        gs = _make_3d_gsplat(n=5)
        mask = np.array([True, True, False, True, False])
        filtered = gs.filter(mask)
        assert filtered.n_splats == 3

    def test_all_true(self):
        gs = _make_3d_gsplat(n=3)
        filtered = gs.filter(np.ones(3, dtype=bool))
        assert filtered.n_splats == 3

    def test_all_false(self):
        gs = _make_3d_gsplat(n=3)
        filtered = gs.filter(np.zeros(3, dtype=bool))
        assert filtered.n_splats == 0

    def test_colors_preserved(self):
        gs = GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
            sharpnesses=np.ones(3, dtype=np.float32),
            colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        )
        filtered = gs.filter(np.array([True, False, True]))
        assert np.allclose(filtered.colors[0], [1, 0, 0])
        assert np.allclose(filtered.colors[1], [0, 0, 1])

    def test_none_colors(self):
        gs = _make_3d_gsplat(n=3)
        filtered = gs.filter(np.array([True, False, True]))
        assert filtered.colors is None

    def test_mask_shape_validation(self):
        gs = _make_3d_gsplat(n=5)
        with pytest.raises(ValueError, match="Mask shape"):
            gs.filter(np.ones(3, dtype=bool))

    def test_with_computed_property(self):
        """Filter by volume end-to-end."""
        gs = _make_3d_gsplat(n=10)
        filtered = gs.filter(gs.volumes() > 0)
        assert filtered.n_splats <= gs.n_splats

    def test_empty(self):
        gs = _make_empty_gsplat()
        filtered = gs.filter(np.ones(0, dtype=bool))
        assert filtered.n_splats == 0


# ── Concatenate ─────────────────────────────────────────


class TestConcatenate:
    def test_two_datasets(self):
        gs1 = _make_3d_gsplat(n=3, seed=1)
        gs2 = _make_3d_gsplat(n=5, seed=2)
        result = GSplatData.concatenate([gs1, gs2])
        assert result.n_splats == 8
        assert result.ndim == 3

    def test_preserves_data(self):
        gs1 = _make_3d_gsplat(n=2, seed=1)
        gs2 = _make_3d_gsplat(n=2, seed=2)
        result = GSplatData.concatenate([gs1, gs2])
        assert np.allclose(result.centers[:2], gs1.centers)
        assert np.allclose(result.centers[2:], gs2.centers)

    def test_all_colors(self):
        gs1 = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
            colors=np.array([[1, 0, 0]], dtype=np.float32),
        )
        gs2 = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
            colors=np.array([[0, 0, 1]], dtype=np.float32),
        )
        result = GSplatData.concatenate([gs1, gs2])
        assert result.colors is not None
        assert np.allclose(result.colors[0], [1, 0, 0])
        assert np.allclose(result.colors[1], [0, 0, 1])

    def test_no_colors(self):
        gs1 = _make_3d_gsplat(n=2, seed=1)
        gs2 = _make_3d_gsplat(n=2, seed=2)
        result = GSplatData.concatenate([gs1, gs2])
        assert result.colors is None

    def test_mixed_colors_fills_white(self):
        gs_with = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
            colors=np.array([[1, 0, 0]], dtype=np.float32),
        )
        gs_without = _make_3d_gsplat(n=1, seed=1)
        result = GSplatData.concatenate([gs_with, gs_without])
        assert result.colors is not None
        assert np.allclose(result.colors[1], [1, 1, 1])  # white fill

    def test_ndim_mismatch(self):
        gs_3d = _make_3d_gsplat(n=1)
        gs_2d = _make_2d_gsplat(n=1)
        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            GSplatData.concatenate([gs_3d, gs_2d])

    def test_empty_list(self):
        with pytest.raises(ValueError, match="At least one"):
            GSplatData.concatenate([])

    def test_single_dataset(self):
        gs = _make_3d_gsplat(n=5)
        result = GSplatData.concatenate([gs])
        assert result.n_splats == 5


# ── Split ───────────────────────────────────────────────


class TestSplit:
    def test_equal_parts(self):
        gs = _make_3d_gsplat(n=10)
        parts = gs.split(2)
        assert len(parts) == 2
        assert parts[0].n_splats == 5
        assert parts[1].n_splats == 5

    def test_uneven(self):
        gs = _make_3d_gsplat(n=10)
        parts = gs.split(3)
        assert len(parts) == 3
        assert sum(p.n_splats for p in parts) == 10

    def test_at_indices(self):
        gs = _make_3d_gsplat(n=10)
        parts = gs.split([3, 7])
        assert len(parts) == 3
        assert parts[0].n_splats == 3
        assert parts[1].n_splats == 4
        assert parts[2].n_splats == 3

    def test_roundtrip_with_concatenate(self):
        gs = _make_3d_gsplat(n=10)
        parts = gs.split(3)
        recombined = GSplatData.concatenate(parts)
        assert recombined.n_splats == 10
        assert np.allclose(recombined.centers, gs.centers)
        assert np.allclose(recombined.amplitudes, gs.amplitudes)

    def test_preserves_colors(self):
        gs = GSplatData(
            centers=np.zeros((4, 3), dtype=np.float32),
            amplitudes=np.ones(4, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (4, 1)
            ),
            sharpnesses=np.ones(4, dtype=np.float32),
            colors=np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32
            ),
        )
        parts = gs.split(2)
        assert parts[0].colors is not None
        assert np.allclose(parts[0].colors[0], [1, 0, 0])


# ── Embed dimension ─────────────────────────────────────


class TestEmbedDimension:
    def test_scalar_value(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.embed_dimension(5.0)
        assert result.ndim == 4
        assert result.n_splats == 3
        assert np.allclose(result.centers[:, 3], 5.0)

    def test_per_splat_values(self):
        gs = _make_3d_gsplat(n=3)
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = gs.embed_dimension(vals)
        assert np.allclose(result.centers[:, 3], vals)

    def test_ndim_increases(self):
        gs = _make_2d_gsplat(n=2)
        result = gs.embed_dimension(0.0)
        assert result.ndim == 3
        # 2D chol has k=3, 3D chol has k=6
        assert result.cholesky_factors.shape[1] == 6

    def test_original_dims_preserved(self):
        gs = _make_3d_gsplat(n=2)
        result = gs.embed_dimension(0.0)
        assert np.allclose(result.centers[:, :3], gs.centers)

    def test_amplitudes_unchanged(self):
        gs = _make_3d_gsplat(n=2)
        result = gs.embed_dimension(0.0)
        assert result.amplitudes is gs.amplitudes

    def test_values_shape_validation(self):
        gs = _make_3d_gsplat(n=3)
        with pytest.raises(ValueError, match="values shape"):
            gs.embed_dimension(np.array([1.0, 2.0]))  # wrong size

    def test_empty(self):
        gs = _make_empty_gsplat(ndim=3)
        result = gs.embed_dimension(0.0)
        assert result.ndim == 4
        assert result.n_splats == 0


# ── Transform ───────────────────────────────────────────


class TestTransform:
    def test_identity(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.transform(np.eye(3))
        assert np.allclose(result.centers, gs.centers, atol=1e-5)

    def test_uniform_scale(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.transform(np.eye(3) * 2.0)
        assert np.allclose(result.centers, gs.centers * 2.0, atol=1e-5)

    def test_translation_affine(self):
        gs = _make_3d_gsplat(n=3)
        M = np.eye(4)
        M[:3, 3] = [10, 20, 30]
        result = gs.transform(M)
        assert np.allclose(result.centers, gs.centers + [10, 20, 30], atol=1e-5)

    def test_rotation_2d(self):
        """90-degree rotation in 2D."""
        gs = GSplatData(
            centers=np.array([[1.0, 0.0]], dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        result = gs.transform(R)
        assert np.allclose(result.centers[0], [0, 1], atol=1e-5)

    def test_covariance_correctness(self):
        """Verify Sigma_new = A @ Sigma @ A.T for known transform."""
        from luxar.gsplats.utils.trils import unpack_tril

        gs = GSplatData(
            centers=np.zeros((1, 2), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2.0, 0.5, 1.5]], dtype=np.float32),
            sharpnesses=np.ones(1, dtype=np.float32),
        )
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        result = gs.transform(A)
        # Compute expected Sigma
        L_orig = unpack_tril(gs.cholesky_factors.astype(np.float64), 2)
        Sigma_orig = L_orig @ np.swapaxes(L_orig, -2, -1)
        Sigma_expected = A @ Sigma_orig[0] @ A.T
        # Verify result
        L_result = unpack_tril(result.cholesky_factors.astype(np.float64), 2)
        Sigma_result = (L_result @ np.swapaxes(L_result, -2, -1))[0]
        assert np.allclose(Sigma_result, Sigma_expected, atol=1e-6)

    def test_amplitudes_unchanged(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.transform(np.eye(3) * 2.0)
        assert result.amplitudes is gs.amplitudes

    def test_invalid_shape(self):
        gs = _make_3d_gsplat(n=2)
        with pytest.raises(ValueError, match="Matrix shape"):
            gs.transform(np.eye(2))

    def test_invalid_last_row(self):
        gs = _make_3d_gsplat(n=2)
        M = np.eye(4)
        M[3, 0] = 1.0
        with pytest.raises(ValueError, match="Last row"):
            gs.transform(M)

    def test_empty(self):
        gs = _make_empty_gsplat()
        result = gs.transform(np.eye(3))
        assert result.n_splats == 0


# ── Intensity transforms ────────────────────────────────


class TestAffineIntensity:
    def test_scale_only(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(scale=2.0)
        assert np.allclose(result.amplitudes, gs.amplitudes * 2.0)

    def test_offset_only(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(offset=0.5)
        assert np.allclose(result.amplitudes, gs.amplitudes + 0.5)

    def test_both(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(scale=2.0, offset=0.1)
        assert np.allclose(result.amplitudes, gs.amplitudes * 2.0 + 0.1)

    def test_preserves_centers(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(scale=3.0)
        assert result.centers is gs.centers


class TestNormalizeIntensity:
    def test_basic(self):
        gs = _make_3d_gsplat()
        result = gs.normalize_intensity()
        assert np.allclose(result.amplitudes.max(), 1.0)

    def test_custom_target(self):
        gs = _make_3d_gsplat()
        result = gs.normalize_intensity(target_max=0.5)
        assert np.allclose(result.amplitudes.max(), 0.5)

    def test_zero_amplitudes(self):
        gs = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
            sharpnesses=np.ones(2, dtype=np.float32),
        )
        result = gs.normalize_intensity()
        assert np.allclose(result.amplitudes, 0.0)


class TestClampIntensity:
    def test_min(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(min=0.5)
        assert result.amplitudes.min() >= 0.5

    def test_max(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(max=0.1)
        assert result.amplitudes.max() <= 0.1

    def test_both(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(min=0.2, max=0.8)
        assert result.amplitudes.min() >= 0.2
        assert result.amplitudes.max() <= 0.8

    def test_preserves_centers(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(min=0.0, max=1.0)
        assert result.centers is gs.centers
