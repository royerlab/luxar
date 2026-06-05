"""Unit tests for ``luxar.core.group.compositing`` — the pure helpers shared
by the partition-/lod-wrapping leaf adders.

B7/[P5]: ``slice_optional_array`` and ``position_bounds_from_array`` were only
exercised incidentally through the end-to-end partition tests, leaving the
scalar / list / 0-D / mismatched-shape branches (compositing.py:62-70) and
the empty-array guard (line 83) uncovered. These direct unit tests pin every
branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.group.compositing import (
    position_bounds_from_array,
    slice_optional_array,
)


class TestSliceOptionalArray:
    def test_none_passes_through(self) -> None:
        idx = np.array([0, 2], dtype=np.intp)
        assert slice_optional_array(None, idx, n_elements=4) is None

    @pytest.mark.parametrize("scalar", [3, 1.5, True, "label"])
    def test_scalars_pass_through_unchanged(self, scalar) -> None:
        idx = np.array([0, 1], dtype=np.intp)
        assert slice_optional_array(scalar, idx, n_elements=4) == scalar

    def test_list_matching_length_is_sliced(self) -> None:
        """A per-element list (e.g. string labels) is reindexed."""
        value = ["a", "b", "c", "d"]
        idx = np.array([3, 0], dtype=np.intp)
        assert slice_optional_array(value, idx, n_elements=4) == ["d", "a"]

    def test_list_wrong_length_passes_through(self) -> None:
        """A list whose length != n_elements is left untouched."""
        value = ["only", "two"]
        idx = np.array([0, 1, 2], dtype=np.intp)
        assert slice_optional_array(value, idx, n_elements=4) == ["only", "two"]

    def test_zero_d_array_passes_through(self) -> None:
        value = np.array(5.0, dtype=np.float32)  # 0-D
        idx = np.array([0, 1], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        assert out is value  # returned unchanged (identity)

    def test_per_element_array_is_sliced(self) -> None:
        value = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        idx = np.array([2, 0], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, [30.0, 10.0])

    def test_2d_per_element_array_is_sliced_on_first_axis(self) -> None:
        value = np.arange(12, dtype=np.float32).reshape(4, 3)  # (N=4, 3)
        idx = np.array([1, 3], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, value[[1, 3]])

    def test_broadcast_vector_passes_through(self) -> None:
        """An RGB 3-vector (shape (3,)) with n_elements=4 must NOT be sliced."""
        value = np.array([255, 128, 0], dtype=np.uint8)
        idx = np.array([0, 1], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, value)
        assert out.shape == (3,)

    def test_non_ndarray_sequence_is_converted_then_sliced(self) -> None:
        """A tuple of length n_elements is asarray'd and first-axis sliced."""
        value = (1.0, 2.0, 3.0, 4.0)
        idx = np.array([3, 2], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, [4.0, 3.0])


class TestPositionBoundsFromArray:
    def test_normal_nd_bounds(self) -> None:
        pos = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
        bounds = position_bounds_from_array(pos)
        assert bounds["min"] == [1.0, -2.0, -6.0]
        assert bounds["max"] == [4.0, 5.0, 3.0]

    def test_single_row(self) -> None:
        """N=1 boundary: min == max == the single point."""
        pos = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
        bounds = position_bounds_from_array(pos)
        assert bounds["min"] == [7.0, 8.0, 9.0]
        assert bounds["max"] == [7.0, 8.0, 9.0]

    def test_empty_array_raises(self) -> None:
        pos = np.zeros((0, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="empty array"):
            position_bounds_from_array(pos)

    def test_returns_python_floats(self) -> None:
        """Values are JSON-serialisable Python floats, not numpy scalars."""
        pos = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        bounds = position_bounds_from_array(pos)
        assert all(isinstance(v, float) for v in bounds["min"])
        assert all(isinstance(v, float) for v in bounds["max"])
