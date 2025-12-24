"""
Unit tests for Metal L→Conic computation.

Validates that Metal conic matches PyTorch conic with high precision.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import cholesky_to_conic

pytestmark = pytest.mark.skipif(
    not is_metal_available() or not torch.backends.mps.is_available(),
    reason="Metal backend or MPS not available",
)


# Import the extension module after the skip check
# This avoids import errors when Metal is not available
def _get_metal_backend():
    """Get the Metal extension module (already loaded by is_metal_available)."""
    import metal_splatting_backend

    return metal_splatting_backend


class TestMetalConicAccuracy:
    """Test Metal L→Conic accuracy against PyTorch reference."""

    def test_diagonal_L(self):
        """Test Metal conic for diagonal L matrices."""
        metal_splatting_backend = _get_metal_backend()

        # Diagonal L in [Z,Y,X] order
        L = torch.tensor(
            [[[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
        )

        # Compute in PyTorch (outputs in [Z,Y,X] order)
        conic_pytorch_zyx = cholesky_to_conic(L)

        # Compute in Metal (outputs in [X,Y,Z] order)
        L_mps = L.to("mps")
        conic_metal_xyz = metal_splatting_backend.compute_conic_metal(L_mps).cpu()

        # Reorder PyTorch to [X,Y,Z] for comparison
        conic_pytorch_xyz = conic_pytorch_zyx[:, [5, 4, 2, 3, 1, 0]]

        # Should be identical
        assert torch.allclose(conic_metal_xyz, conic_pytorch_xyz, atol=1e-6), (
            f"Metal conic differs:\nMetal [X,Y,Z]:   {conic_metal_xyz}\nPyTorch [X,Y,Z]: {conic_pytorch_xyz}"
        )

    def test_non_diagonal_L(self):
        """Test Metal conic for non-diagonal L matrices."""
        metal_splatting_backend = _get_metal_backend()

        # Non-diagonal L in [Z,Y,X] order
        L = torch.tensor(
            [[[2.0, 0.0, 0.0], [1.0, 1.5, 0.0], [0.5, 0.3, 1.0]]], dtype=torch.float32
        )

        # Compute in PyTorch (outputs [Z,Y,X])
        conic_pytorch_zyx = cholesky_to_conic(L)

        # Compute in Metal (outputs [X,Y,Z])
        L_mps = L.to("mps")
        conic_metal_xyz = metal_splatting_backend.compute_conic_metal(L_mps).cpu()

        # Reorder PyTorch to [X,Y,Z] for comparison
        conic_pytorch_xyz = conic_pytorch_zyx[:, [5, 4, 2, 3, 1, 0]]

        # Should match within floating-point precision
        max_diff = (conic_metal_xyz - conic_pytorch_xyz).abs().max().item()
        assert max_diff < 1e-6, f"Metal conic error too large: {max_diff}"

    def test_batch_processing(self):
        """Test Metal conic with multiple splats."""
        metal_splatting_backend = _get_metal_backend()

        # Multiple random L matrices
        N = 100
        L = torch.tril(torch.randn(N, 3, 3))
        L[:, 0, 0] = torch.abs(L[:, 0, 0]) + 0.5
        L[:, 1, 1] = torch.abs(L[:, 1, 1]) + 0.5
        L[:, 2, 2] = torch.abs(L[:, 2, 2]) + 0.5

        # Compute both ways
        conic_pytorch_zyx = cholesky_to_conic(L)
        conic_pytorch_xyz = conic_pytorch_zyx[
            :, [5, 4, 2, 3, 1, 0]
        ]  # Reorder to [X,Y,Z]

        conic_metal_xyz = metal_splatting_backend.compute_conic_metal(L.to("mps")).cpu()

        # Check all match
        max_diff = (conic_metal_xyz - conic_pytorch_xyz).abs().max().item()
        mean_diff = (conic_metal_xyz - conic_pytorch_xyz).abs().mean().item()

        # float32 precision limits: 1e-4 is reasonable for batch processing
        assert max_diff < 1e-4, f"Max difference too large: {max_diff}"
        assert mean_diff < 1e-5, f"Mean difference too large: {mean_diff}"

    def test_coordinate_ordering(self):
        """Verify Metal conic outputs in correct [X,Y,Z] order."""
        metal_splatting_backend = _get_metal_backend()

        # L with different values to check ordering
        L = torch.tensor(
            [
                [
                    [3.0, 0.0, 0.0],  # L_zz = 3
                    [0.0, 2.0, 0.0],  # L_yy = 2
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )  # L_xx = 1

        conic = metal_splatting_backend.compute_conic_metal(L.to("mps")).cpu()

        # Conic should be in [X,Y,Z] order: [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
        # Expected: [1/1²=1.0, 0, 0, 1/2²=0.25, 0, 1/3²=0.111]
        assert torch.allclose(conic[0, 0], torch.tensor(1.0), atol=1e-6), (
            "c_xx should be 1.0"
        )
        assert torch.allclose(conic[0, 3], torch.tensor(0.25), atol=1e-6), (
            "c_yy should be 0.25"
        )
        assert torch.allclose(conic[0, 5], torch.tensor(1 / 9), atol=1e-6), (
            "c_zz should be 1/9"
        )


class TestMetalConicIntegration:
    """Test Metal conic in full forward/backward pipeline."""

    def test_forward_with_metal_conic(self):
        """Test that forward works with Metal conic."""
        from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

        model = GaussianSplatModelMetal(
            shape=(16, 16, 16),
            centers0=np.array([[8, 8, 8]], dtype=np.float32),
            L0=np.array([np.eye(3) * 1.5], dtype=np.float32),
            amps0=np.array([1.0], dtype=np.float32),
            sigma_min_diag=[0.5, 0.5, 0.5],
            use_metal_conic=True,  # Enable Metal conic
            device="mps",
        )

        output = model()
        assert output.max() > 0, "Output should have non-zero values"
        assert output.shape == (16, 16, 16)

    def test_backward_with_metal_conic(self):
        """Test that backward works correctly with Metal conic."""
        from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

        model = GaussianSplatModelMetal(
            shape=(16, 16, 16),
            centers0=np.random.rand(5, 3) * 8 + 4,
            L0=np.tile(np.eye(3) * 1.5, (5, 1, 1)).astype(np.float32),
            amps0=np.ones(5, dtype=np.float32),
            sigma_min_diag=[0.5, 0.5, 0.5],
            use_metal_conic=True,
            device="mps",
        )

        output = model()
        loss = output.sum()
        loss.backward()

        # Check gradients computed
        has_grad = any(
            p.grad is not None and p.grad.norm() > 0 for p in model.parameters()
        )
        assert has_grad, "Gradients should be computed"

    def test_matches_pytorch_conic(self):
        """Test that Metal conic gives same results as PyTorch conic."""
        from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

        np.random.seed(42)
        centers = np.random.rand(10, 3) * 12 + 2
        L = np.tile(np.eye(3) * 1.5, (10, 1, 1)).astype(np.float32)
        amps = np.ones(10, dtype=np.float32)

        # Model with PyTorch conic
        model_pytorch = GaussianSplatModelMetal(
            shape=(16, 16, 16),
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            use_metal_conic=False,  # PyTorch
            device="mps",
        )

        # Model with Metal conic
        model_metal = GaussianSplatModelMetal(
            shape=(16, 16, 16),
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            use_metal_conic=True,  # Metal
            device="mps",
        )

        # Forward passes
        out_pytorch = model_pytorch().cpu()
        out_metal = model_metal().cpu()

        # Should be essentially identical
        max_diff = (out_pytorch - out_metal).abs().max().item()
        assert max_diff < 1e-5, f"Outputs differ: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
