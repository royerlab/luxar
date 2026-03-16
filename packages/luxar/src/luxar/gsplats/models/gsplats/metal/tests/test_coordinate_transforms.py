"""
Unit tests for coordinate transformations between PyTorch [Z,Y,X] and Metal [X,Y,Z].

Each test focuses on one specific transformation to isolate issues.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from arbol import aprint

# Check if Metal is available (for TestEndToEndCoordinates)
_metal_available = sys.platform == "darwin" and torch.backends.mps.is_available()


class TestCentersReordering:
    """Test centers coordinate reordering: [Z,Y,X] → [X,Y,Z]."""

    def test_centers_single_splat(self):
        """Test reordering a single splat."""
        # PyTorch format: [Z, Y, X]
        centers_pytorch = torch.tensor([[10.0, 8.0, 6.0]], dtype=torch.float32)

        # Reorder to Metal format: [X, Y, Z]
        centers_metal = centers_pytorch[:, [2, 1, 0]]

        # Should be [X=6, Y=8, Z=10]
        assert centers_metal[0, 0].item() == 6.0, "X should be 6"
        assert centers_metal[0, 1].item() == 8.0, "Y should be 8"
        assert centers_metal[0, 2].item() == 10.0, "Z should be 10"

    def test_centers_multiple_splats(self):
        """Test reordering multiple splats."""
        centers_pytorch = torch.tensor(
            [[10.0, 8.0, 6.0], [5.0, 7.0, 9.0], [12.0, 11.0, 4.0]], dtype=torch.float32
        )

        centers_metal = centers_pytorch[:, [2, 1, 0]]

        # Check each splat
        assert torch.allclose(centers_metal[0], torch.tensor([6.0, 8.0, 10.0]))
        assert torch.allclose(centers_metal[1], torch.tensor([9.0, 7.0, 5.0]))
        assert torch.allclose(centers_metal[2], torch.tensor([4.0, 11.0, 12.0]))

    def test_centers_roundtrip(self):
        """Test that reordering is reversible."""
        centers_orig = torch.rand(10, 3, dtype=torch.float32)

        # [Z,Y,X] → [X,Y,Z] → [Z,Y,X]
        centers_metal = centers_orig[:, [2, 1, 0]]
        centers_back = centers_metal[:, [2, 1, 0]]

        assert torch.allclose(centers_back, centers_orig)


class TestLMatrixReordering:
    """Test L matrix coordinate reordering: [Z,Y,X] → [X,Y,Z]."""

    def test_diagonal_L(self):
        """Test that diagonal L doesn't change (symmetric)."""
        # Diagonal L in any coordinate system
        L_pytorch = torch.tensor(
            [[[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
        )

        # Reorder: permute both rows and columns [2,1,0]
        L_metal = L_pytorch[:, [2, 1, 0], :][:, :, [2, 1, 0]]

        # For diagonal, should be: [[1.0, 0, 0], [0, 1.5, 0], [0, 0, 2.0]]
        expected = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
        )

        assert torch.allclose(L_metal, expected), (
            f"Diagonal reordering failed: got {L_metal[0]}"
        )

    def test_lower_triangular_L(self):
        """Test non-diagonal lower-triangular L using correct transformation."""
        # L in [Z,Y,X] convention
        L_pytorch = torch.tensor(
            [
                [
                    [2.0, 0.0, 0.0],  # L_zz, 0, 0
                    [1.0, 1.5, 0.0],  # L_yz, L_yy, 0
                    [0.5, 0.3, 1.0],
                ]
            ],
            dtype=torch.float32,
        )  # L_xz, L_xy, L_xx

        # Correct reordering: Σ → permute → cholesky
        Sigma = L_pytorch @ L_pytorch.transpose(-2, -1)  # Covariance in [z,y,x]
        Sigma_reordered = Sigma[:, [2, 1, 0], :][:, :, [2, 1, 0]]  # Permute to [x,y,z]
        L_metal = torch.linalg.cholesky(Sigma_reordered)  # Recompute lower-triangular L

        # L_metal should be lower triangular
        upper = torch.triu(L_metal[0], diagonal=1)
        assert torch.allclose(upper, torch.zeros(3, 3), atol=1e-6), (
            f"Recomputed L is not lower triangular:\n{L_metal[0]}"
        )

        # Verify it represents the same covariance (after permutation)
        Sigma_check = L_metal @ L_metal.transpose(-2, -1)
        assert torch.allclose(Sigma_check, Sigma_reordered, atol=1e-5)

    def test_L_still_lower_triangular(self):
        """Test that reordered L (via Σ transformation) is still lower triangular."""
        L_pytorch = torch.tril(torch.randn(5, 3, 3))
        L_pytorch[:, 0, 0] = torch.abs(L_pytorch[:, 0, 0]) + 0.5
        L_pytorch[:, 1, 1] = torch.abs(L_pytorch[:, 1, 1]) + 0.5
        L_pytorch[:, 2, 2] = torch.abs(L_pytorch[:, 2, 2]) + 0.5

        # Correct transformation via covariance
        Sigma = L_pytorch @ L_pytorch.transpose(-2, -1)
        Sigma_reordered = Sigma[:, [2, 1, 0], :][:, :, [2, 1, 0]]
        L_metal = torch.linalg.cholesky(Sigma_reordered)

        # Check it's still lower triangular
        for i in range(5):
            upper = torch.triu(L_metal[i], diagonal=1)
            assert torch.allclose(upper, torch.zeros(3, 3), atol=1e-6), (
                f"Reordered L[{i}] is not lower triangular:\n{L_metal[i]}"
            )


class TestConicReordering:
    """Test conic (Σ⁻¹ upper triangle) reordering."""

    def test_conic_diagonal(self):
        """Test conic reordering for diagonal covariance."""
        # Conic in [Z,Y,X]: [Σ⁻¹_zz, 0, 0, Σ⁻¹_yy, 0, Σ⁻¹_xx]
        # Indices:          [  0,    1, 2,    3,    4,    5   ]
        conic_pytorch = torch.tensor(
            [[2.0, 0.0, 0.0, 1.5, 0.0, 1.0]], dtype=torch.float32
        )

        # Reorder to [X,Y,Z]: [Σ⁻¹_xx, 0, 0, Σ⁻¹_yy, 0, Σ⁻¹_zz]
        # Mapping: [zz,zy,zx,yy,yx,xx] → [xx,xy,xz,yy,yz,zz]
        #          [ 5, 4, 2, 3, 1, 0] → [0, 1, 2, 3, 4, 5]
        conic_metal = conic_pytorch[:, [5, 4, 2, 3, 1, 0]]

        # Should be [1.0, 0, 0, 1.5, 0, 2.0]
        expected = torch.tensor([[1.0, 0.0, 0.0, 1.5, 0.0, 2.0]], dtype=torch.float32)

        assert torch.allclose(conic_metal, expected), (
            f"Conic reordering failed: got {conic_metal}, expected {expected}"
        )

    def test_conic_from_L(self):
        """Document that permuting L then computing conic is NOT equivalent to
        computing conic then permuting the packed elements.

        This is because ``cholesky_to_conic`` assumes its input is
        lower-triangular.  Permuting both rows and columns of a
        lower-triangular matrix (``L[:, perm, :][:, :, perm]``) generally
        breaks the lower-triangular structure, so feeding the permuted matrix
        into ``cholesky_to_conic`` produces an incorrect result.

        The correct approach (tested in ``test_lower_triangular_L``) is:
          1. Compute Σ = L @ Lᵀ in the original coordinate order.
          2. Permute Σ to the new coordinate order.
          3. Re-factorise with ``torch.linalg.cholesky``.
        """
        from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import (
            cholesky_to_conic,
        )

        # L in [Z,Y,X] – a non-trivial lower-triangular matrix
        L_pytorch = torch.tensor(
            [[[2.0, 0.0, 0.0], [1.0, 1.5, 0.0], [0.5, 0.3, 1.0]]], dtype=torch.float32
        )

        # Path A: compute conic in [Z,Y,X], then reorder packed elements
        conic_pytorch = cholesky_to_conic(L_pytorch)
        conic_reordered = conic_pytorch[:, [5, 4, 2, 3, 1, 0]]

        # Path B: naively permute L to [X,Y,Z], then compute conic
        L_permuted = L_pytorch[:, [2, 1, 0], :][:, :, [2, 1, 0]]
        conic_from_permuted_L = cholesky_to_conic(L_permuted)

        # The two paths must NOT agree – permuting L breaks the
        # lower-triangular invariant that cholesky_to_conic relies on.
        assert not torch.allclose(conic_from_permuted_L, conic_reordered, atol=1e-5), (
            "Expected the two paths to disagree because permuting L breaks "
            "lower-triangular structure, but they unexpectedly matched:\n"
            f"  From permuted L:        {conic_from_permuted_L}\n"
            f"  From reordered conic:   {conic_reordered}"
        )


class TestGradientsReordering:
    """Test that gradient reordering is inverse of forward reordering."""

    def test_centers_gradient_roundtrip(self):
        """Test centers: forward reorder → backward reorder → original."""
        d_centers_pytorch_orig = torch.rand(10, 3)

        # Forward: [Z,Y,X] → [X,Y,Z]
        d_centers_metal = d_centers_pytorch_orig[:, [2, 1, 0]]

        # Backward: [X,Y,Z] → [Z,Y,X]
        d_centers_pytorch_back = d_centers_metal[:, [2, 1, 0]]

        assert torch.allclose(d_centers_pytorch_back, d_centers_pytorch_orig)

    def test_conic_gradient_roundtrip(self):
        """Test conic: forward reorder → backward reorder → original."""
        d_conic_pytorch_orig = torch.rand(10, 6)

        # Forward: [zz,zy,zx,yy,yx,xx] → [xx,xy,xz,yy,yz,zz]
        d_conic_metal = d_conic_pytorch_orig[:, [5, 4, 2, 3, 1, 0]]

        # Backward: inverse mapping
        d_conic_pytorch_back = d_conic_metal[:, [5, 4, 2, 3, 1, 0]]

        assert torch.allclose(d_conic_pytorch_back, d_conic_pytorch_orig)


@pytest.mark.skipif(
    not _metal_available, reason="Metal backend only available on macOS with MPS"
)
class TestEndToEndCoordinates:
    """End-to-end test of coordinate transformations."""

    def test_asymmetric_splat_metal_vs_pytorch(self):
        """Test that Metal produces same result as PyTorch after proper reordering."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
        from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

        shape = (16, 16, 16)

        # Place splat off-center in Z dimension
        centers = np.array([[10.0, 8.0, 8.0]], dtype=np.float32)  # [Z,Y,X]
        L = np.array([np.eye(3) * 1.5], dtype=np.float32)
        amps = np.array([1.0], dtype=np.float32)

        # Metal model
        model_metal = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        # PyTorch model
        model_pytorch = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="cpu",
        )

        # Forward passes
        output_metal = model_metal().cpu()
        output_pytorch = model_pytorch()

        # Peak should be at [10, 8, 8] for both
        assert torch.argmax(output_metal) == torch.argmax(output_pytorch), (
            "Peaks should be at same location"
        )

        peak_val_metal = output_metal[10, 8, 8].item()
        peak_val_pytorch = output_pytorch[10, 8, 8].item()

        aprint("\nAsymmetric splat test:")
        aprint(f"  Metal peak at [10,8,8]: {peak_val_metal:.6f}")
        aprint(f"  PyTorch peak at [10,8,8]: {peak_val_pytorch:.6f}")
        aprint(f"  Difference: {abs(peak_val_metal - peak_val_pytorch):.6e}")

        assert abs(peak_val_metal - peak_val_pytorch) < 0.02, (
            "Peak values should match within 2%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
