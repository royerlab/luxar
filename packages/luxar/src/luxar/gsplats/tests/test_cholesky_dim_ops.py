"""Tests for Cholesky dimension permutation and embedding utilities."""

import numpy as np
import pytest

from luxar.gsplats.utils.trils import (
    embed_cholesky_packed,
    pack_tril,
    permute_cholesky_packed,
    unpack_tril,
)


class TestPermuteCholeskyPacked:
    """Tests for permute_cholesky_packed."""

    def test_identity_permutation_is_noop(self) -> None:
        """Identity permutation [0,1,2] should return equivalent factors."""
        # Create a known 3D Cholesky: L with some off-diagonal terms
        L = np.array([[[2.0, 0, 0], [0.5, 1.5, 0], [0.3, 0.2, 1.0]]])
        packed = pack_tril(L)

        result = permute_cholesky_packed(packed, 3, [0, 1, 2])

        # Unpack and verify covariance matches
        L_orig = unpack_tril(packed, 3)
        L_result = unpack_tril(result, 3)
        Sigma_orig = L_orig @ L_orig.transpose(0, 2, 1)
        Sigma_result = L_result @ L_result.transpose(0, 2, 1)
        np.testing.assert_allclose(Sigma_result, Sigma_orig, atol=1e-6)

    def test_reverse_2d_permutation(self) -> None:
        """Reversing 2D dims swaps Sigma[0,0] and Sigma[1,1]."""
        L = np.array([[[2.0, 0], [0.5, 1.0]]])
        packed = pack_tril(L)
        Sigma_orig = L @ L.transpose(0, 2, 1)

        result = permute_cholesky_packed(packed, 2, [1, 0])

        L_new = unpack_tril(result, 2)
        Sigma_new = L_new @ L_new.transpose(0, 2, 1)

        # Sigma_new[i,j] should == Sigma_orig[perm[i], perm[j]]
        np.testing.assert_allclose(Sigma_new[0, 0, 0], Sigma_orig[0, 1, 1], atol=1e-6)
        np.testing.assert_allclose(Sigma_new[0, 1, 1], Sigma_orig[0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(Sigma_new[0, 0, 1], Sigma_orig[0, 1, 0], atol=1e-6)

    def test_3d_cyclic_permutation(self) -> None:
        """Cyclic permutation [2,0,1] should correctly reorder covariance."""
        L = np.array([[[3.0, 0, 0], [1.0, 2.0, 0], [0.5, 0.3, 1.5]]])
        packed = pack_tril(L)
        Sigma_orig = L @ L.transpose(0, 2, 1)

        perm = [2, 0, 1]
        result = permute_cholesky_packed(packed, 3, perm)
        L_new = unpack_tril(result, 3)
        Sigma_new = L_new @ L_new.transpose(0, 2, 1)

        # Verify Sigma_new[i,j] == Sigma_orig[perm[i], perm[j]]
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(
                    Sigma_new[0, i, j],
                    Sigma_orig[0, perm[i], perm[j]],
                    atol=1e-6,
                    err_msg=f"Mismatch at [{i},{j}]",
                )

    def test_batch_permutation(self) -> None:
        """Permutation works on multiple splats."""
        N = 10
        # Random positive-definite: L with positive diagonal
        L = np.zeros((N, 3, 3))
        L[:, 0, 0] = np.random.rand(N) + 0.5
        L[:, 1, 0] = np.random.randn(N) * 0.3
        L[:, 1, 1] = np.random.rand(N) + 0.5
        L[:, 2, 0] = np.random.randn(N) * 0.3
        L[:, 2, 1] = np.random.randn(N) * 0.3
        L[:, 2, 2] = np.random.rand(N) + 0.5
        packed = pack_tril(L)

        result = permute_cholesky_packed(packed, 3, [1, 2, 0])
        assert result.shape == packed.shape

    def test_invalid_permutation_raises(self) -> None:
        packed = np.array([[1.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="Invalid permutation"):
            permute_cholesky_packed(packed, 2, [0, 0])
        with pytest.raises(ValueError, match="Permutation length"):
            permute_cholesky_packed(packed, 2, [0, 1, 2])


class TestEmbedCholeskyPacked:
    """Tests for embed_cholesky_packed."""

    def test_2d_to_3d_embedding(self) -> None:
        """Embed 2D Cholesky into 3D, verify original block preserved."""
        L_2d = np.array([[[2.0, 0], [0.5, 1.5]]])
        packed_2d = pack_tril(L_2d)
        Sigma_2d = L_2d @ L_2d.transpose(0, 2, 1)

        # Map src dims [0,1] → dst dims [0,1], new dim 2 gets sigma=0.5
        packed_3d = embed_cholesky_packed(packed_2d, 2, 3, [0, 1], fill_sigma={2: 0.5})

        L_3d = unpack_tril(packed_3d, 3)
        Sigma_3d = L_3d @ L_3d.transpose(0, 2, 1)

        # Original 2x2 block preserved
        np.testing.assert_allclose(Sigma_3d[0, :2, :2], Sigma_2d[0], atol=1e-6)
        # New dimension is independent with variance = 0.5^2 = 0.25
        np.testing.assert_allclose(Sigma_3d[0, 2, 2], 0.25, atol=1e-6)
        # Cross-terms with new dim are zero
        np.testing.assert_allclose(Sigma_3d[0, 0, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(Sigma_3d[0, 1, 2], 0.0, atol=1e-6)

    def test_2d_to_3d_with_reordering(self) -> None:
        """Embed 2D into 3D with non-trivial mapping: src[0,1] → dst[1,2]."""
        L_2d = np.array([[[2.0, 0], [0.3, 1.0]]])
        packed_2d = pack_tril(L_2d)
        Sigma_2d = L_2d @ L_2d.transpose(0, 2, 1)

        # src dim 0 → dst dim 1, src dim 1 → dst dim 2
        packed_3d = embed_cholesky_packed(packed_2d, 2, 3, [1, 2], fill_sigma={0: 0.7})

        L_3d = unpack_tril(packed_3d, 3)
        Sigma_3d = L_3d @ L_3d.transpose(0, 2, 1)

        # 2D block should be at positions [1:3, 1:3]
        np.testing.assert_allclose(Sigma_3d[0, 1:3, 1:3], Sigma_2d[0], atol=1e-6)
        # Dim 0 is independent
        np.testing.assert_allclose(Sigma_3d[0, 0, 0], 0.49, atol=1e-6)
        np.testing.assert_allclose(Sigma_3d[0, 0, 1], 0.0, atol=1e-6)

    def test_3d_to_5d_embedding(self) -> None:
        """Embed 3D Cholesky into 5D."""
        L_3d = np.array([[[1.5, 0, 0], [0.2, 1.0, 0], [0.1, 0.3, 0.8]]])
        packed_3d = pack_tril(L_3d)

        # src [0,1,2] → dst [0,1,2], new dims 3,4
        packed_5d = embed_cholesky_packed(
            packed_3d, 3, 5, [0, 1, 2], fill_sigma={3: 2.0, 4: 0.5}
        )

        assert packed_5d.shape == (1, 15)  # 5*(5+1)/2 = 15

        L_5d = unpack_tril(packed_5d, 5)
        Sigma_5d = L_5d @ L_5d.transpose(0, 2, 1)

        # Original 3x3 block preserved
        Sigma_3d = L_3d @ L_3d.transpose(0, 2, 1)
        np.testing.assert_allclose(Sigma_5d[0, :3, :3], Sigma_3d[0], atol=1e-6)
        # New dims independent
        np.testing.assert_allclose(Sigma_5d[0, 3, 3], 4.0, atol=1e-6)  # 2.0^2
        np.testing.assert_allclose(Sigma_5d[0, 4, 4], 0.25, atol=1e-6)  # 0.5^2

    def test_default_fill_sigma(self) -> None:
        """Unmapped dims default to sigma=1.0."""
        packed_2d = np.array([[1.0, 0.0, 1.0]])  # isotropic 2D
        packed_3d = embed_cholesky_packed(packed_2d, 2, 3, [0, 1])

        L_3d = unpack_tril(packed_3d, 3)
        Sigma_3d = L_3d @ L_3d.transpose(0, 2, 1)
        np.testing.assert_allclose(Sigma_3d[0, 2, 2], 1.0, atol=1e-6)

    def test_roundtrip_extract_block(self) -> None:
        """Embed then extract: original covariance block must match."""
        N = 5
        L = np.zeros((N, 2, 2))
        L[:, 0, 0] = np.random.rand(N) + 0.5
        L[:, 1, 0] = np.random.randn(N) * 0.3
        L[:, 1, 1] = np.random.rand(N) + 0.5
        packed_2d = pack_tril(L)
        Sigma_2d = L @ L.transpose(0, 2, 1)

        # Embed into 4D: src[0,1] → dst[2,3]
        packed_4d = embed_cholesky_packed(packed_2d, 2, 4, [2, 3])
        L_4d = unpack_tril(packed_4d, 4)
        Sigma_4d = L_4d @ L_4d.transpose(0, 2, 1)

        # Extract block at [2:4, 2:4]
        np.testing.assert_allclose(Sigma_4d[:, 2:4, 2:4], Sigma_2d, atol=1e-6)

    def test_validation_errors(self) -> None:
        packed = np.array([[1.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="Target dim"):
            embed_cholesky_packed(packed, 2, 1, [0])
        with pytest.raises(ValueError, match="dim_mapping length"):
            embed_cholesky_packed(packed, 2, 3, [0])
        with pytest.raises(ValueError, match="duplicates"):
            embed_cholesky_packed(packed, 2, 3, [0, 0])
        with pytest.raises(ValueError, match="out of range"):
            embed_cholesky_packed(packed, 2, 3, [0, 5])

    def test_fill_sigma_zero_for_discrete_dims(self) -> None:
        """fill_sigma=0 for discrete dims should not crash; produces tiny positive diagonal."""
        packed_2d = np.array([[1.0, 0.0, 1.0]])  # 2D identity

        # sigma=0 for dim 2 (discrete time) — should be handled gracefully
        packed_3d = embed_cholesky_packed(packed_2d, 2, 3, [0, 1], fill_sigma={2: 0})

        assert packed_3d.shape == (1, 6)
        L_3d = unpack_tril(packed_3d, 3)
        # Dim 2 diagonal should be tiny but positive (not zero or negative)
        assert L_3d[0, 2, 2] > 0
        assert L_3d[0, 2, 2] < 1e-5
        # Original 2x2 block should be preserved
        Sigma_3d = L_3d @ L_3d.transpose(0, 2, 1)
        np.testing.assert_allclose(Sigma_3d[0, :2, :2], np.eye(2), atol=1e-6)
