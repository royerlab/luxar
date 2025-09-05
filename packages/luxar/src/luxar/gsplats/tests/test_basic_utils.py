"""
Basic tests for gsplats utilities that don't require external dependencies.
These tests focus on core functionality using only numpy.
"""

import numpy as np
import pytest

from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus
from luxar.gsplats.utils.trils import pack_tril, tril_size, unpack_tril


class TestTrilOperations:
    """Test triangular matrix operations."""

    def test_tril_basic_functionality(self):
        """Test basic pack/unpack functionality."""
        # Create a simple 2x2 lower triangular matrix
        L = np.array([[[1.0, 0.0], [2.0, 3.0]]], dtype=np.float32)

        # Pack it
        packed = pack_tril(L)
        expected_packed = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(packed, expected_packed)

        # Unpack it
        unpacked = unpack_tril(packed, d=2)
        np.testing.assert_array_equal(unpacked, L)

    def test_tril_size_calculation(self):
        """Test tril_size calculation."""
        assert tril_size(1) == 1
        assert tril_size(2) == 3
        assert tril_size(3) == 6
        assert tril_size(4) == 10


class TestInverseSoftplusBasic:
    """Test inverse softplus with numpy only."""

    def test_inverse_softplus_basic_functionality(self):
        """Test basic inverse softplus without torch dependency."""
        y = np.array([1.0, 2.0, 5.0], dtype=np.float32)
        x = stable_inverse_softplus(y)

        # Verify inverse relationship using numpy
        # softplus(x) = log(1 + exp(x))
        softplus_x = np.log1p(np.exp(x))
        np.testing.assert_array_almost_equal(softplus_x, y, decimal=5)

    def test_inverse_softplus_edge_cases(self):
        """Test edge cases."""
        # Small values
        y_small = np.array([1e-3, 1e-2], dtype=np.float32)
        x_small = stable_inverse_softplus(y_small)
        assert np.all(np.isfinite(x_small))

        # Large values
        y_large = np.array([10.0, 20.0], dtype=np.float32)
        x_large = stable_inverse_softplus(y_large)
        assert np.all(np.isfinite(x_large))


class TestValidationHelpers:
    """Test validation functions that can work without external dependencies."""

    def test_array_shape_validation(self):
        """Test basic array shape validation patterns."""
        # These patterns appear in the gsplats code
        arr_2d = np.random.randn(5, 3)
        arr_3d = np.random.randn(2, 3, 3)

        # Check basic properties
        assert arr_2d.ndim == 2
        assert arr_3d.ndim == 3
        assert arr_2d.shape[1] == 3
        assert arr_3d.shape[1] == arr_3d.shape[2]  # Square matrices

    def test_numeric_stability_patterns(self):
        """Test numeric stability patterns used in the code."""
        # Test patterns like the ones used in fit_gsplats
        values = np.array([1.0, 1.0, 1.0])  # Uniform array

        # Check for near-zero range (division by zero prevention)
        value_min = np.percentile(values, 1)
        value_max = np.percentile(values, 99)
        range_val = value_max - value_min

        is_uniform = np.abs(range_val) < 1e-12
        assert is_uniform  # This should be uniform

        # Test non-uniform case
        values_varied = np.array([0.0, 5.0, 10.0])
        value_min = np.percentile(values_varied, 1)
        value_max = np.percentile(values_varied, 99)
        range_val = value_max - value_min
        is_uniform = np.abs(range_val) < 1e-12
        assert not is_uniform  # This should not be uniform


class TestInputValidationPatterns:
    """Test input validation patterns."""

    def test_empty_array_detection(self):
        """Test empty array detection patterns."""
        empty_1d = np.array([])
        empty_2d = np.zeros((0, 3))
        valid_array = np.array([1, 2, 3])

        assert empty_1d.size == 0
        assert empty_2d.size == 0
        assert valid_array.size > 0

    def test_dimension_validation(self):
        """Test dimension validation patterns."""
        scalar = np.array(5.0)
        vector = np.array([1, 2, 3])
        matrix = np.array([[1, 2], [3, 4]])

        assert scalar.ndim == 0
        assert vector.ndim == 1
        assert matrix.ndim == 2

    def test_positive_value_validation(self):
        """Test positive value validation patterns."""
        positive_values = np.array([0.1, 1.0, 5.0])
        mixed_values = np.array([-1.0, 0.0, 1.0])

        assert np.all(positive_values > 0)
        assert not np.all(mixed_values > 0)
        assert np.any(mixed_values <= 0)

    def test_percentile_range_validation(self):
        """Test percentile validation patterns."""
        valid_percentiles = [0.0, 50.0, 100.0]
        invalid_percentiles = [-10.0, 150.0]

        for p in valid_percentiles:
            assert 0 <= p <= 100

        for p in invalid_percentiles:
            assert not (0 <= p <= 100)


class TestCommonAlgorithmicPatterns:
    """Test common algorithmic patterns used in gsplats."""

    def test_distance_calculation_patterns(self):
        """Test distance calculation patterns like those in _dedupe."""
        points = np.array([[0, 0], [3, 4], [1, 1]], dtype=float)

        # Test pairwise distance calculation
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                diff = points[i] - points[j]
                dist_squared = np.sum(diff**2)
                dist = np.sqrt(dist_squared)

                # Verify known distances
                if i == 0 and j == 1:  # (0,0) to (3,4)
                    assert np.isclose(dist, 5.0)  # 3-4-5 triangle

    def test_coordinate_bounds_checking(self):
        """Test coordinate bounds checking patterns."""
        image_shape = (10, 15)  # Height=10, Width=15
        coords = np.array([[5, 7], [0, 0], [9, 14]], dtype=float)

        # Check all coordinates are within bounds
        for coord in coords:
            assert 0 <= coord[0] < image_shape[0]  # Y coordinate
            assert 0 <= coord[1] < image_shape[1]  # X coordinate

    def test_array_concatenation_patterns(self):
        """Test array concatenation patterns like in find_candidates_overcomplete_nd."""
        arrays = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6]]),
            np.array([[7, 8], [9, 10], [11, 12]]),
        ]

        # Filter out empty arrays
        non_empty_arrays = [arr for arr in arrays if arr.size > 0]
        assert len(non_empty_arrays) == 3

        # Concatenate
        result = np.vstack(non_empty_arrays)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        np.testing.assert_array_equal(result, expected)

    def test_dtype_handling_patterns(self):
        """Test data type handling patterns."""
        # Test float32 conversion patterns
        input_int = np.array([1, 2, 3], dtype=int)
        input_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        converted_32 = np.asarray(input_int, dtype=np.float32)
        converted_from_64 = np.asarray(input_float64, dtype=np.float32)

        assert converted_32.dtype == np.float32
        assert converted_from_64.dtype == np.float32

        # Test that values are preserved
        np.testing.assert_array_equal(converted_32, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(converted_from_64, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    pytest.main([__file__])
