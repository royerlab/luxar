"""Tests for nD validation module."""

import numpy as np
import pytest

from luxar.core.dimensions import Dimension, Dimensions
from luxar.validation.nd import (
    DimensionalCoverageError,
    broadcast_to_all_slices,
    validate_dimensional_coverage,
)


class TestDimensionalCoverageError:
    """Test the DimensionalCoverageError exception."""

    def test_error_with_missing_coverage(self) -> None:
        """Test error message with missing coverage details."""
        missing = {"time": {0.0, 1.0, 2.0}, "channel": {0.0, 1.0}}
        error = DimensionalCoverageError(
            "Test error message", "group1", missing_coverage=missing
        )

        assert "Test error message" in str(error)
        assert "Missing coverage:" in str(error)
        assert "time: [0.0, 1.0, 2.0]" in str(error)
        assert "channel: [0.0, 1.0]" in str(error)
        assert "💡 Suggestions:" in str(error)
        assert error.group_name == "group1"
        assert error.missing_coverage == missing

    def test_error_without_missing_coverage(self) -> None:
        """Test error message without missing coverage."""
        error = DimensionalCoverageError("Simple error", "group2")

        assert "Simple error" in str(error)
        assert "💡 Suggestions:" in str(error)
        assert error.group_name == "group2"
        assert error.missing_coverage is None


class TestValidateDimensionalCoverage:
    """Test dimensional coverage validation."""

    def test_no_non_displayed_dimensions(self) -> None:
        """Test validation passes when all dimensions are displayed."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("y", "um", (-10, 10), 1, display=True),
                Dimension("z", "um", (-10, 10), 1, display=True),
            ]
        )

        groups = {
            "group1": np.random.randn(100, 3).astype(np.float32),
            "group2": np.random.randn(50, 3).astype(np.float32),
        }

        # Should not raise
        validate_dimensional_coverage(dims, groups)

    def test_consistent_coverage(self) -> None:
        """Test validation passes with consistent coverage."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("y", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 10), 1, display=False),
            ]
        )

        # Create groups with same time points
        times = [0.0, 1.0, 2.0]

        group1_positions = []
        group2_positions = []
        for t in times:
            points1 = np.random.randn(10, 3).astype(np.float32)
            points1[:, 2] = t
            group1_positions.append(points1)

            points2 = np.random.randn(5, 3).astype(np.float32)
            points2[:, 2] = t
            group2_positions.append(points2)

        groups = {
            "group1": np.vstack(group1_positions),
            "group2": np.vstack(group2_positions),
        }

        # Should not raise
        validate_dimensional_coverage(dims, groups)

    def test_inconsistent_coverage_raises(self) -> None:
        """Test validation raises for inconsistent coverage."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("y", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 10), 1, display=False),
            ]
        )

        # Group1 has times [0, 1, 2], group2 only has [0, 1]
        group1_positions = []
        for t in [0.0, 1.0, 2.0]:
            points = np.random.randn(10, 3).astype(np.float32)
            points[:, 2] = t
            group1_positions.append(points)

        group2_positions = []
        for t in [0.0, 1.0]:  # Missing t=2.0
            points = np.random.randn(5, 3).astype(np.float32)
            points[:, 2] = t
            group2_positions.append(points)

        groups = {
            "group1": np.vstack(group1_positions),
            "group2": np.vstack(group2_positions),
        }

        with pytest.raises(DimensionalCoverageError) as exc_info:
            validate_dimensional_coverage(dims, groups)

        error = exc_info.value
        assert error.group_name == "group2"
        assert "incomplete coverage" in str(error)
        assert "time" in str(error)

    def test_multiple_non_displayed_dimensions(self) -> None:
        """Test validation with multiple non-displayed dimensions."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 10), 1, display=False),
                Dimension("channel", "idx", (0, 2), 1, display=False),
            ]
        )

        # Create consistent coverage for both time and channel
        positions = []
        for t in [0.0, 1.0]:
            for c in [0.0, 1.0]:
                points = np.random.randn(5, 3).astype(np.float32)
                points[:, 1] = t  # time is dimension 1
                points[:, 2] = c  # channel is dimension 2
                positions.append(points)

        # Create second group with same structure but different x values
        positions2 = []
        for t in [0.0, 1.0]:
            for c in [0.0, 1.0]:
                points = np.random.randn(5, 3).astype(np.float32)
                points[:, 1] = t  # time is dimension 1
                points[:, 2] = c  # channel is dimension 2
                positions2.append(points)

        groups = {
            "group1": np.vstack(positions),
            "group2": np.vstack(positions2),
        }

        # Should not raise
        validate_dimensional_coverage(dims, groups)

    def test_empty_groups(self) -> None:
        """Test validation with empty point groups."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 10), 1, display=False),
            ]
        )

        groups: dict[str, np.ndarray] = {}

        # Should not raise for empty groups (nothing to validate)
        validate_dimensional_coverage(dims, groups)

    def test_single_group(self) -> None:
        """Test validation with single group (no comparison needed)."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 10), 1, display=False),
            ]
        )

        positions = np.random.randn(10, 2).astype(np.float32)
        positions[:, 1] = np.random.choice([0.0, 1.0, 2.0], 10)

        groups = {"only_group": positions}

        # Should not raise for single group
        validate_dimensional_coverage(dims, groups)


class TestBroadcastToAllSlices:
    """Test broadcast_to_all_slices function."""

    def test_no_broadcasting_needed(self) -> None:
        """Test when all dimensions are displayed."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("y", "um", (-10, 10), 1, display=True),
            ]
        )

        positions = np.random.randn(10, 2).astype(np.float32)
        colors = np.random.rand(10, 3).astype(np.float32)
        radii = np.random.rand(10).astype(np.float32)

        new_pos, new_col, new_rad = broadcast_to_all_slices(
            positions, colors, radii, dims
        )

        # Should return same arrays
        assert np.array_equal(new_pos, positions)
        assert new_col is not None
        assert new_rad is not None
        assert np.array_equal(new_col, colors)
        assert np.array_equal(new_rad, radii)

    def test_broadcast_single_non_displayed_dimension(self) -> None:
        """Test broadcasting with one non-displayed dimension."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 2), 1, display=False, discrete=True),
            ]
        )

        # Original data at single time point
        positions = np.random.randn(5, 2).astype(np.float32)
        positions[:, 1] = 0  # All at time=0
        colors = np.random.rand(5, 3).astype(np.float32)
        radii = np.random.rand(5).astype(np.float32)

        new_pos, new_col, new_rad = broadcast_to_all_slices(
            positions, colors, radii, dims
        )

        # Should broadcast to 3 time points (0, 1, 2)
        assert new_pos.shape == (15, 2)  # 5 points * 3 times
        assert new_col is not None
        assert new_rad is not None
        assert new_col.shape == (15, 3)
        assert new_rad.shape == (15,)

        # Check time values are set correctly
        assert np.all(new_pos[0:5, 1] == 0)
        assert np.all(new_pos[5:10, 1] == 1)
        assert np.all(new_pos[10:15, 1] == 2)

        # Check x values are replicated
        assert np.array_equal(new_pos[0:5, 0], new_pos[5:10, 0])
        assert np.array_equal(new_pos[0:5, 0], new_pos[10:15, 0])

    def test_broadcast_multiple_non_displayed_dimensions(self) -> None:
        """Test broadcasting with multiple non-displayed dimensions."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 1), 1, display=False, discrete=True),
                Dimension("channel", "idx", (0, 1), 1, display=False, discrete=True),
            ]
        )

        # Original data
        positions = np.random.randn(3, 3).astype(np.float32)
        positions[:, 1] = 0  # time=0
        positions[:, 2] = 0  # channel=0
        colors = None
        radii = None

        new_pos, new_col, new_rad = broadcast_to_all_slices(
            positions, colors, radii, dims
        )

        # Should broadcast to 2*2=4 combinations
        assert new_pos.shape == (12, 3)  # 3 points * 2 times * 2 channels
        assert new_col is None
        assert new_rad is None

        # Check all combinations exist
        combinations = set()
        for i in range(0, 12, 3):
            combinations.add((new_pos[i, 1], new_pos[i, 2]))

        expected = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
        assert combinations == expected

    def test_broadcast_with_existing_values(self) -> None:
        """Test broadcasting uses existing unique values when no range specified."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", display=False),  # No range or discrete flag
            ]
        )

        # Data with specific time values
        positions = np.zeros((9, 2), dtype=np.float32)
        positions[0:3, 1] = 0.5
        positions[3:6, 1] = 1.5
        positions[6:9, 1] = 2.5
        colors = np.random.rand(9, 3).astype(np.float32)
        radii = np.random.rand(9).astype(np.float32)

        new_pos, new_col, new_rad = broadcast_to_all_slices(
            positions, colors, radii, dims
        )

        # Should use existing unique values [0.5, 1.5, 2.5]
        assert new_pos.shape == (27, 2)  # 9 points * 3 unique times

        # Check that all three time values are present
        unique_times = np.unique(new_pos[:, 1])
        assert np.allclose(unique_times, [0.5, 1.5, 2.5])

    def test_broadcast_preserves_colors_and_radii(self) -> None:
        """Test that colors and radii are properly replicated."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 1), 1, display=False, discrete=True),
            ]
        )

        positions = np.random.randn(2, 2).astype(np.float32)
        positions[:, 1] = 0
        colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # Red and green
        radii = np.array([0.5, 1.0], dtype=np.float32)

        new_pos, new_col, new_rad = broadcast_to_all_slices(
            positions, colors, radii, dims
        )

        # Check colors are replicated correctly
        assert new_col is not None
        assert np.array_equal(new_col[0], [1, 0, 0])  # First point, time=0
        assert np.array_equal(new_col[1], [0, 1, 0])  # Second point, time=0
        assert np.array_equal(new_col[2], [1, 0, 0])  # First point, time=1
        assert np.array_equal(new_col[3], [0, 1, 0])  # Second point, time=1

        # Check radii are replicated correctly
        assert new_rad is not None
        assert new_rad[0] == 0.5
        assert new_rad[1] == 1.0
        assert new_rad[2] == 0.5
        assert new_rad[3] == 1.0

    def test_broadcast_high_dimensional(self) -> None:
        """Test broadcasting with positions having more dimensions than scene."""
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("y", "um", (-10, 10), 1, display=True),
                Dimension("z", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 1), 1, display=False, discrete=True),
            ]
        )

        # Positions have 5 dimensions (x, y, z, time, extra)
        positions = np.random.randn(3, 5).astype(np.float32)
        positions[:, 3] = 0  # time=0
        colors = None
        radii = None

        new_pos, new_col, new_rad = broadcast_to_all_slices(
            positions, colors, radii, dims
        )

        # Should broadcast to 2 time points
        assert new_pos.shape == (6, 5)

        # Check time dimension is set correctly
        assert np.all(new_pos[0:3, 3] == 0)
        assert np.all(new_pos[3:6, 3] == 1)

        # Extra dimension should be preserved
        assert np.array_equal(new_pos[0:3, 4], new_pos[3:6, 4])

    # [Python-R5 / validation-G3] Single-point broadcasting edge case. The
    # existing tests use N≥3 points; a regression that special-cased
    # "N>1" (slicing with assumptions about repeated rows) would slip
    # past those. Pin that a single point broadcasts to exactly
    # `n_non_displayed_slices` rows, with the displayed coords identical
    # across slices and the non-displayed coord taking the documented
    # discrete values.
    def test_broadcast_single_point(self) -> None:
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("y", "um", (-10, 10), 1, display=True),
                Dimension("z", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 2), 1, display=False, discrete=True),
            ]
        )
        positions = np.array([[1.0, 2.0, 3.0, 0.0]], dtype=np.float32)
        new_pos, _, _ = broadcast_to_all_slices(positions, None, None, dims)
        # 1 input point × 3 time slices (0, 1, 2) → 3 rows
        assert new_pos.shape == (3, 4)
        # x/y/z preserved across slices
        for slice_idx in range(3):
            np.testing.assert_allclose(new_pos[slice_idx, :3], [1.0, 2.0, 3.0])
        # Each row has a distinct time value covering 0, 1, 2
        assert sorted(new_pos[:, 3].tolist()) == [0.0, 1.0, 2.0]

    # [Python-R5 / validation-G3] Empty positions input. A regression
    # that crashed on shape (0, D) instead of returning shape (0, D)
    # unchanged would silently break the no-points scene case.
    def test_broadcast_empty_positions(self) -> None:
        dims = Dimensions(
            [
                Dimension("x", "um", (-10, 10), 1, display=True),
                Dimension("y", "um", (-10, 10), 1, display=True),
                Dimension("z", "um", (-10, 10), 1, display=True),
                Dimension("time", "ms", (0, 2), 1, display=False, discrete=True),
            ]
        )
        positions = np.zeros((0, 4), dtype=np.float32)
        new_pos, new_col, new_rad = broadcast_to_all_slices(positions, None, None, dims)
        # Empty stays empty regardless of broadcast factor.
        assert new_pos.shape == (0, 4)
        assert new_col is None
        assert new_rad is None
