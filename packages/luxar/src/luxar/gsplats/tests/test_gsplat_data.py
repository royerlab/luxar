"""Tests for GSplatData class methods."""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData


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
