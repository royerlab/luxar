"""
Tests for triangular matrix packing/unpacking utilities.
"""

import numpy as np

from luxar.gsplats.utils.trils import pack_tril, tril_size, unpack_tril


class TestTrilSize:
    """Test tril_size function."""

    def test_tril_size_valid_dimensions(self):
        """Test tril_size for various valid dimensions."""
        # Test cases: (dimension, expected_size)
        test_cases = [
            (1, 1),  # 1x1 matrix: 1 element
            (2, 3),  # 2x2 matrix: 3 elements (diagonal + lower)
            (3, 6),  # 3x3 matrix: 6 elements
            (4, 10),  # 4x4 matrix: 10 elements
            (5, 15),  # 5x5 matrix: 15 elements
        ]

        for d, expected in test_cases:
            assert tril_size(d) == expected, f"Failed for dimension {d}"

    def test_tril_size_formula_consistency(self):
        """Test that tril_size follows the mathematical formula d*(d+1)/2."""
        for d in range(1, 10):
            expected = d * (d + 1) // 2
            assert tril_size(d) == expected

    def test_tril_size_zero_dimension(self):
        """Test edge case of zero dimension."""
        assert tril_size(0) == 0


class TestPackTril:
    """Test pack_tril function."""

    def test_pack_tril_2d_single_matrix(self):
        """Test packing a single 2x2 matrix."""
        L = np.array([[[1.0, 0.0], [2.0, 3.0]]])  # Shape (1, 2, 2)

        packed = pack_tril(L)
        expected = np.array([[1.0, 2.0, 3.0]])  # [L00, L10, L11]

        np.testing.assert_array_equal(packed, expected)
        assert packed.shape == (1, 3)

    def test_pack_tril_3d_single_matrix(self):
        """Test packing a single 3x3 matrix."""
        L = np.array(
            [[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]]
        )  # Shape (1, 3, 3)

        packed = pack_tril(L)
        expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])  # Row-major order

        np.testing.assert_array_equal(packed, expected)
        assert packed.shape == (1, 6)

    def test_pack_tril_batch_matrices(self):
        """Test packing a batch of matrices."""
        L = np.array(
            [[[1.0, 0.0], [2.0, 3.0]], [[4.0, 0.0], [5.0, 6.0]]]
        )  # Shape (2, 2, 2)

        packed = pack_tril(L)
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        np.testing.assert_array_equal(packed, expected)
        assert packed.shape == (2, 3)

    def test_pack_tril_ignores_upper_triangle(self):
        """Test that upper triangular elements are ignored."""
        L = np.array(
            [
                [
                    [1.0, 999.0],  # Upper triangle should be ignored
                    [2.0, 3.0],
                ]
            ]
        )  # Shape (1, 2, 2)

        packed = pack_tril(L)
        expected = np.array([[1.0, 2.0, 3.0]])  # Upper 999.0 is ignored

        np.testing.assert_array_equal(packed, expected)

    def test_pack_tril_dtype_preservation(self):
        """Test that data type is preserved."""
        L_float32 = np.array([[[1.0, 0.0], [2.0, 3.0]]], dtype=np.float32)
        L_float64 = np.array([[[1.0, 0.0], [2.0, 3.0]]], dtype=np.float64)

        packed_32 = pack_tril(L_float32)
        packed_64 = pack_tril(L_float64)

        assert packed_32.dtype == np.float32
        assert packed_64.dtype == np.float64

    def test_pack_tril_empty_batch(self):
        """Test packing empty batch."""
        L = np.zeros((0, 2, 2))
        packed = pack_tril(L)
        assert packed.shape == (0, 3)


class TestUnpackTril:
    """Test unpack_tril function."""

    def test_unpack_tril_2d_single_matrix(self):
        """Test unpacking to a single 2x2 matrix."""
        v = np.array([[1.0, 2.0, 3.0]])

        L = unpack_tril(v, d=2)
        expected = np.array([[[1.0, 0.0], [2.0, 3.0]]])

        np.testing.assert_array_equal(L, expected)
        assert L.shape == (1, 2, 2)

    def test_unpack_tril_3d_single_matrix(self):
        """Test unpacking to a single 3x3 matrix."""
        v = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        L = unpack_tril(v, d=3)
        expected = np.array([[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]])

        np.testing.assert_array_equal(L, expected)
        assert L.shape == (1, 3, 3)

    def test_unpack_tril_batch_matrices(self):
        """Test unpacking a batch of matrices."""
        v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        L = unpack_tril(v, d=2)
        expected = np.array([[[1.0, 0.0], [2.0, 3.0]], [[4.0, 0.0], [5.0, 6.0]]])

        np.testing.assert_array_equal(L, expected)
        assert L.shape == (2, 2, 2)

    def test_unpack_tril_dtype_preservation(self):
        """Test that data type is preserved."""
        v_float32 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        v_float64 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

        L_32 = unpack_tril(v_float32, d=2)
        L_64 = unpack_tril(v_float64, d=2)

        assert L_32.dtype == np.float32
        assert L_64.dtype == np.float64

    def test_unpack_tril_empty_batch(self):
        """Test unpacking empty batch."""
        v = np.zeros((0, 3))
        L = unpack_tril(v, d=2)
        assert L.shape == (0, 2, 2)


class TestPackUnpackRoundTrip:
    """Test that pack_tril and unpack_tril are inverse operations."""

    def test_roundtrip_various_sizes(self):
        """Test pack/unpack roundtrip for various matrix sizes."""
        np.random.seed(42)

        for d in [1, 2, 3, 4, 5]:
            for n_batch in [1, 3, 5]:
                # Create random lower triangular matrices
                L_orig = np.random.randn(n_batch, d, d).astype(np.float32)

                # Zero out upper triangle
                for i in range(d):
                    for j in range(i + 1, d):
                        L_orig[:, i, j] = 0.0

                # Pack and unpack
                packed = pack_tril(L_orig)
                L_recovered = unpack_tril(packed, d)

                np.testing.assert_array_almost_equal(
                    L_orig,
                    L_recovered,
                    err_msg=f"Roundtrip failed for d={d}, n_batch={n_batch}",
                )

                # Check shapes
                assert packed.shape == (n_batch, tril_size(d))
                assert L_recovered.shape == (n_batch, d, d)

    def test_roundtrip_preserves_zeros_above_diagonal(self):
        """Test that upper triangular elements remain zero after roundtrip."""
        L_orig = np.array(
            [
                [
                    [1.0, 999.0],  # Upper triangle has non-zeros
                    [2.0, 3.0],
                ]
            ]
        )

        packed = pack_tril(L_orig)
        L_recovered = unpack_tril(packed, d=2)

        expected = np.array(
            [
                [
                    [1.0, 0.0],  # Upper triangle should be zero
                    [2.0, 3.0],
                ]
            ]
        )

        np.testing.assert_array_equal(L_recovered, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_dimension_one_matrices(self):
        """Test 1x1 matrices (scalars)."""
        L = np.array([[[5.0]]])  # Shape (1, 1, 1)
        packed = pack_tril(L)
        L_recovered = unpack_tril(packed, d=1)

        np.testing.assert_array_equal(L, L_recovered)
        assert packed.shape == (1, 1)

    def test_consistency_with_numpy_tril_indices(self):
        """Test consistency with numpy's tril_indices."""
        np.random.seed(42)

        for d in [2, 3, 4]:
            L = np.random.randn(1, d, d)
            # Zero upper triangle using numpy
            for i in range(d):
                for j in range(i + 1, d):
                    L[0, i, j] = 0.0

            # Pack using our function
            packed = pack_tril(L)

            # Extract using numpy tril_indices
            i_indices, j_indices = np.tril_indices(d)
            numpy_packed = L[0][i_indices, j_indices]

            np.testing.assert_array_equal(
                packed[0], numpy_packed, err_msg=f"Inconsistency with numpy for d={d}"
            )
