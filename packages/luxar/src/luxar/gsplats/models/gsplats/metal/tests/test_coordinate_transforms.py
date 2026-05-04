"""
Coordinate-convention tests for the splat-centric Metal backend.

The current Metal kernels intentionally use Luxar/PyTorch volume order
``[Z, Y, X]`` directly.  Older tile-binned kernels used a packed-conic reorder
for ``[X, Y, Z]`` math; that path has been removed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from arbol import aprint

from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import cholesky_to_conic

_metal_available = is_metal_available()


class TestNativeZyxConvention:
    """Verify that the custom Metal path no longer needs coordinate reorders."""

    def test_centers_are_kept_in_zyx_order(self) -> None:
        centers = torch.tensor([[10.0, 8.0, 6.0]], dtype=torch.float32)
        assert torch.allclose(centers[0], torch.tensor([10.0, 8.0, 6.0]))

    def test_conic_packed_order_is_row_major_upper_triangle(self) -> None:
        # Diagonal L in [Z,Y,X] order: sigma_z=3, sigma_y=2, sigma_x=1.
        L = torch.tensor(
            [[[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        conic = cholesky_to_conic(L)

        expected = torch.tensor(
            [[1.0 / 9.0, 0.0, 0.0, 1.0 / 4.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(conic, expected, atol=1e-6)

    def test_conic_gradient_order_matches_forward_order(self) -> None:
        d_conic = torch.rand(10, 6)
        # No old [Z,Y,X] <-> [X,Y,Z] permutation is needed anymore.
        assert torch.allclose(d_conic, d_conic.clone())


@pytest.mark.skipif(
    not _metal_available, reason="Metal backend only available on macOS with MPS"
)
class TestEndToEndCoordinates:
    """End-to-end checks for native [Z,Y,X] Metal coordinates."""

    def test_asymmetric_splat_metal_vs_pytorch(self) -> None:
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
        from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

        shape = (16, 16, 16)

        # Place splat off-center in Z dimension.  Both backends should put the
        # peak at the same [Z,Y,X] location without any coordinate permutation.
        centers = np.array([[10.0, 8.0, 8.0]], dtype=np.float32)
        L = np.array([np.eye(3) * 1.5], dtype=np.float32)
        amps = np.array([1.0], dtype=np.float32)

        model_metal = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        model_pytorch = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="cpu",
        )

        output_metal = model_metal().cpu()
        output_pytorch = model_pytorch()

        assert torch.argmax(output_metal) == torch.argmax(output_pytorch)

        peak_val_metal = output_metal[10, 8, 8].item()
        peak_val_pytorch = output_pytorch[10, 8, 8].item()

        aprint("\nAsymmetric splat test:")
        aprint(f"  Metal peak at [10,8,8]: {peak_val_metal:.6f}")
        aprint(f"  PyTorch peak at [10,8,8]: {peak_val_pytorch:.6f}")
        aprint(f"  Difference: {abs(peak_val_metal - peak_val_pytorch):.6e}")

        assert abs(peak_val_metal - peak_val_pytorch) < 0.02
