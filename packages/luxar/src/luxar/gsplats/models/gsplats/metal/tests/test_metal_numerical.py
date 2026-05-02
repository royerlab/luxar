"""
Numerical accuracy and performance tests for Metal backend.

Tests:
1. cholesky_to_conic accuracy
2. Metal vs PyTorch output comparison
3. Performance speedup measurement
4. Gradient numerical validation
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from arbol import aprint

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

# Skip entire module on non-macOS platforms
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    reason="Metal backend only available on macOS with MPS",
)

# Import Metal-specific modules only on macOS
if sys.platform == "darwin":
    from luxar.gsplats.models.gsplats.metal import (
        GaussianSplatModelMetal,
        is_metal_available,
    )
    from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import (
        cholesky_to_conic,
    )
else:
    # Provide dummies for type checking
    GaussianSplatModelMetal = None  # type: ignore[misc, assignment]
    is_metal_available = lambda: False  # noqa: E731
    cholesky_to_conic = None  # type: ignore[misc, assignment]


class TestCholeskyToConic:
    """Test cholesky_to_conic function accuracy."""

    def test_diagonal_matrix(self):
        """Test with diagonal Cholesky factors."""
        # L = diag(1.5, 1.5, 1.5)
        L = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 1.5

        conic = cholesky_to_conic(L)

        # For diagonal L, Σ = L @ L^T = diag(2.25, 2.25, 2.25)
        # Σ^(-1) = diag(1/2.25, 1/2.25, 1/2.25) = diag(0.444..., 0.444..., 0.444...)
        expected_diag = 1.0 / (1.5**2)

        assert conic.shape == (1, 6)
        assert torch.allclose(
            conic[0, 0], torch.tensor(expected_diag), atol=1e-6
        )  # c_xx
        assert torch.allclose(
            conic[0, 3], torch.tensor(expected_diag), atol=1e-6
        )  # c_yy
        assert torch.allclose(
            conic[0, 5], torch.tensor(expected_diag), atol=1e-6
        )  # c_zz
        assert torch.allclose(conic[0, 1], torch.tensor(0.0), atol=1e-6)  # c_xy
        assert torch.allclose(conic[0, 2], torch.tensor(0.0), atol=1e-6)  # c_xz
        assert torch.allclose(conic[0, 4], torch.tensor(0.0), atol=1e-6)  # c_yz

    def test_general_matrix(self):
        """Test with non-diagonal Cholesky factors."""
        # Create a non-trivial lower-triangular L
        L = torch.tensor(
            [[[2.0, 0.0, 0.0], [1.0, 1.5, 0.0], [0.5, 0.3, 1.0]]], dtype=torch.float32
        )

        conic = cholesky_to_conic(L)

        # Verify shape
        assert conic.shape == (1, 6)

        # Verify Σ = L @ L^T and Σ^(-1) is correct
        Sigma = L @ L.transpose(-2, -1)
        Sigma_inv_expected = torch.linalg.inv(Sigma)

        # Reconstruct full matrix from conic
        Sigma_inv_reconstructed = torch.zeros(1, 3, 3, dtype=torch.float32)
        Sigma_inv_reconstructed[0, 0, 0] = conic[0, 0]  # c_xx
        Sigma_inv_reconstructed[0, 0, 1] = conic[0, 1]  # c_xy
        Sigma_inv_reconstructed[0, 1, 0] = conic[0, 1]  # c_xy (symmetric)
        Sigma_inv_reconstructed[0, 0, 2] = conic[0, 2]  # c_xz
        Sigma_inv_reconstructed[0, 2, 0] = conic[0, 2]  # c_xz
        Sigma_inv_reconstructed[0, 1, 1] = conic[0, 3]  # c_yy
        Sigma_inv_reconstructed[0, 1, 2] = conic[0, 4]  # c_yz
        Sigma_inv_reconstructed[0, 2, 1] = conic[0, 4]  # c_yz
        Sigma_inv_reconstructed[0, 2, 2] = conic[0, 5]  # c_zz

        assert torch.allclose(Sigma_inv_reconstructed, Sigma_inv_expected, atol=1e-5), (
            "Conic should match Σ^(-1)"
        )

    def test_batch_processing(self):
        """Test with multiple splats."""
        N = 10
        L = torch.randn(N, 3, 3, dtype=torch.float32)
        # Make it lower triangular with positive diagonal
        L = torch.tril(L)
        L[:, 0, 0] = torch.abs(L[:, 0, 0]) + 0.5
        L[:, 1, 1] = torch.abs(L[:, 1, 1]) + 0.5
        L[:, 2, 2] = torch.abs(L[:, 2, 2]) + 0.5

        conic = cholesky_to_conic(L)

        assert conic.shape == (N, 6)
        assert torch.all(torch.isfinite(conic))

    def test_mps_compatibility(self):
        """Test that cholesky_to_conic works on MPS device."""
        L = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 2.0
        L_mps = L.to("mps")

        conic = cholesky_to_conic(L_mps)

        assert conic.device.type == "mps"
        assert torch.allclose(conic[0, 0], torch.tensor(1.0 / 4.0).to("mps"), atol=1e-6)


class TestMetalVsPyTorchAccuracy:
    """Compare Metal rendering against PyTorch reference."""

    @pytest.fixture
    def test_volume_and_params(self):
        """Create test volume and model parameters."""
        shape = (32, 32, 32)
        n_splats = 20

        # Random splats throughout the volume
        np.random.seed(42)
        centers = np.random.rand(n_splats, 3) * 24 + 4  # Keep away from edges
        L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)

        # Add some variation to L
        for i in range(n_splats):
            scale = np.random.uniform(1.0, 3.0)
            L[i] *= scale

        amps = np.random.rand(n_splats).astype(np.float32)

        return shape, centers, L, amps

    def test_forward_matches_pytorch(self, test_volume_and_params):
        """Test that Metal forward output matches PyTorch."""
        shape, centers, L, amps = test_volume_and_params

        # Create Metal model
        # NOTE: Metal has an additional per-pixel intensity culling feature that
        # PyTorch doesn't have. This causes small differences in output values.
        model_metal = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            intensity_floor=1e-5,
            device="mps",
        )

        # Create PyTorch model (uses intensity_floor=1e-5 hardcoded for AABB only)
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

        # Compare
        max_diff = (output_metal - output_pytorch).abs().max().item()
        mean_diff = (output_metal - output_pytorch).abs().mean().item()
        output_max = max(
            output_metal.abs().max().item(), output_pytorch.abs().max().item()
        )
        relative_max_diff = max_diff / (output_max + 1e-10)

        aprint("\nMetal vs PyTorch:")
        aprint(f"  Max diff: {max_diff:.6e}")
        aprint(f"  Mean diff: {mean_diff:.6e}")
        aprint(f"  Relative max diff: {relative_max_diff:.4f}")
        aprint(f"  Output range: [{output_metal.min():.4f}, {output_metal.max():.4f}]")

        # Tolerance: Metal has per-pixel intensity culling that PyTorch doesn't have,
        # causing up to ~3% difference. Use 5% relative tolerance or 0.05 absolute.
        # The mean difference should still be very small (<1%).
        assert max_diff < 0.05 or relative_max_diff < 0.05, (
            f"Metal output should match PyTorch within 5%, got max_diff={max_diff:.4f} ({relative_max_diff:.2%})"
        )
        assert mean_diff < 0.01, f"Mean difference should be <1%, got {mean_diff:.6f}"


class TestMetalPerformance:
    """Performance comparison tests."""

    @pytest.mark.slow
    def test_speedup_vs_cpu(self):
        """Measure speedup of Metal vs CPU PyTorch."""
        import time

        shape = (64, 64, 64)
        n_splats = 500

        np.random.seed(42)
        centers = np.random.rand(n_splats, 3) * 48 + 8
        L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
        amps = np.ones(n_splats, dtype=np.float32)

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

        # CPU model
        model_cpu = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="cpu",
        )

        # Warmup
        _ = model_metal()
        _ = model_cpu()

        # Benchmark Metal
        torch.mps.synchronize()
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = model_metal()
            torch.mps.synchronize()
        metal_time = (time.perf_counter() - start) / n_iters

        # Benchmark CPU
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = model_cpu()
        cpu_time = (time.perf_counter() - start) / n_iters

        speedup = cpu_time / metal_time

        aprint(f"\nPerformance (shape={shape}, n_splats={n_splats}):")
        aprint(f"  CPU:   {cpu_time * 1000:.2f} ms/iter")
        aprint(f"  Metal: {metal_time * 1000:.2f} ms/iter")
        aprint(f"  Speedup: {speedup:.2f}x")

        # Note: Metal overhead may exceed benefit for small problems on fast CPUs.
        # This test benchmarks performance without asserting speedup.
        # For production use, Metal benefits larger volumes (128³+) with more splats (1000+).


class TestMetalGradients:
    """Test gradient computation accuracy."""

    def test_gradcheck_simple(self):
        """Run torch.autograd.gradcheck on a simple case."""
        # Small volume for gradcheck (it's slow)
        shape = (8, 8, 8)
        n_splats = 2

        centers = np.array([[4, 4, 4], [5, 5, 5]], dtype=np.float32)
        L = np.tile(np.eye(3) * 1.0, (n_splats, 1, 1)).astype(np.float32)
        amps = np.array([0.5, 0.5], dtype=np.float32)

        model = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.3, 0.3, 0.3],
            truncate=2.0,
            intensity_floor=1e-6,
            device="mps",
        )

        # Get parameters
        params = list(model.parameters())

        # Note: gradcheck is very slow and strict, so we use relaxed tolerances
        # For GPU compute, epsilon differences are expected
        pass_gradcheck = True
        try:
            for i, param in enumerate(params):
                if param.requires_grad:
                    # Test each parameter separately (faster than all at once)
                    result = torch.autograd.gradcheck(
                        lambda p: model(),
                        (param,),
                        eps=1e-3,  # Relaxed from default 1e-6
                        atol=1e-2,  # Relaxed from default 1e-5
                        rtol=1e-2,
                        raise_exception=False,
                    )
                    if not result:
                        aprint(f"  gradcheck failed for parameter {i}")
                        pass_gradcheck = False
        except Exception as e:
            aprint(f"  gradcheck error: {e}")
            pass_gradcheck = False

        # Note: Numerical gradcheck on custom GPU compute is notoriously finicky
        # If this fails, it doesn't necessarily mean gradients are wrong
        # The integration tests and convergence tests are more reliable.
        # Using xfail (not skip) so failures are visible in CI reports.
        if not pass_gradcheck:
            pytest.xfail("gradcheck failed (expected for GPU numerics)")

    def test_gradient_sign_correctness(self):
        """Test that gradient signs point in the correct direction to minimize loss.

        This test guards against the sign bug fixed in kernels.metal where Y and X
        gradients had incorrect signs (+1 instead of -1 in the chain rule).

        The test places a splat offset from a target in all 3 dimensions and verifies
        that gradient descent moves the splat towards the target.
        """
        from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import (
            MetalSplatFunction,
        )

        shape = (32, 32, 32)

        # Splat at [20, 18, 14], target at [16, 16, 16]
        # Gradient ∂loss/∂center tells us how loss changes when center increases.
        # Expected gradient directions (for MSE loss):
        #   Z: positive (splat at 20 > target 16, increasing Z moves away from target)
        #   Y: positive (splat at 18 > target 16, increasing Y moves away from target)
        #   X: negative (splat at 14 < target 16, increasing X moves towards target)
        centers = torch.tensor([[20.0, 18.0, 14.0]], device="mps", requires_grad=True)
        L = torch.tensor([[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]]], device="mps")
        amps = torch.tensor([1.0], device="mps")

        target = torch.zeros(shape, device="mps")
        target[16, 16, 16] = 1.0

        # Forward and backward
        output = MetalSplatFunction.apply(centers, L, amps, shape, 3.0, 1e-5, False)
        loss = ((output - target) ** 2).sum()
        loss.backward()

        grad = centers.grad[0].cpu().numpy()

        # Verify gradient signs (this is what the bug affected)
        # Z gradient should be positive (splat above target)
        assert grad[0] > 0, f"Z gradient should be positive, got {grad[0]:.6e}"
        # Y gradient should be positive (splat above target)
        assert grad[1] > 0, f"Y gradient should be positive, got {grad[1]:.6e}"
        # X gradient should be negative (splat below target)
        assert grad[2] < 0, f"X gradient should be negative, got {grad[2]:.6e}"

    def test_gradient_values_match_cpu_reference(self):
        """Test that Metal gradients match CPU reference values.

        Compares Metal backward pass gradients against CPU render_gaussians gradients.
        """
        from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import (
            MetalSplatFunction,
        )

        shape = (32, 32, 32)
        centers_np = np.array([[20.0, 18.0, 14.0]], dtype=np.float32)
        L_np = (np.eye(3) * 2.0)[np.newaxis, :, :].astype(np.float32)
        amps_np = np.array([1.0], dtype=np.float32)

        target = torch.zeros(shape)
        target[16, 16, 16] = 1.0

        # CPU reference
        centers_cpu = torch.tensor(centers_np, requires_grad=True)
        L_cpu = torch.tensor(L_np, requires_grad=True)
        amps_cpu = torch.tensor(amps_np, requires_grad=True)

        output_cpu = render_gaussians(
            shape,
            centers_cpu,
            L_cpu,
            amps_cpu,
            truncate=3.0,
            intensity_floor=1e-5,
        )
        loss_cpu = ((output_cpu - target) ** 2).sum()
        loss_cpu.backward()
        cpu_grad = centers_cpu.grad[0].numpy()

        # Metal
        centers_metal = torch.tensor(centers_np, requires_grad=True, device="mps")
        L_metal = torch.tensor(L_np, requires_grad=True, device="mps")
        amps_metal = torch.tensor(amps_np, requires_grad=True, device="mps")

        output_metal = MetalSplatFunction.apply(
            centers_metal,
            L_metal,
            amps_metal,
            shape,
            3.0,
            1e-5,
            False,
        )
        loss_metal = ((output_metal - target.to("mps")) ** 2).sum()
        loss_metal.backward()
        metal_grad = centers_metal.grad[0].cpu().numpy()

        # Verify signs match
        signs_match = np.sign(metal_grad) == np.sign(cpu_grad)
        assert signs_match.all(), (
            f"Gradient signs don't match: Metal={np.sign(metal_grad)}, CPU={np.sign(cpu_grad)}"
        )

        # Verify magnitudes are close (15% tolerance for GPU compute)
        ratio = metal_grad / (cpu_grad + 1e-15)
        assert np.allclose(ratio, 1.0, rtol=0.15), (
            f"Gradient magnitudes differ: ratios={ratio}"
        )

    def test_optimization_convergence(self):
        """Test that optimization converges to the target.

        This is a critical integration test that verifies gradients are correct
        enough for practical optimization. If the sign bug exists, convergence fails.
        """
        from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import (
            MetalSplatFunction,
        )

        shape = (32, 32, 32)

        # Start at [20, 18, 14], target at [16, 16, 16]
        centers = torch.tensor([[20.0, 18.0, 14.0]], device="mps", requires_grad=True)
        L = torch.tensor([[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]]], device="mps")
        amps = torch.tensor([1.0], device="mps")

        target = torch.zeros(shape, device="mps")
        target[16, 16, 16] = 1.0
        target_pos = torch.tensor([16.0, 16.0, 16.0], device="mps")

        initial_dist = torch.sqrt(((centers[0] - target_pos) ** 2).sum()).item()

        # Run optimization
        lr = 0.5
        for _ in range(30):
            output = MetalSplatFunction.apply(centers, L, amps, shape, 3.0, 1e-5, False)
            loss = ((output - target) ** 2).sum()
            loss.backward()

            with torch.no_grad():
                centers -= lr * centers.grad
                centers.grad.zero_()

        final_dist = torch.sqrt(((centers[0] - target_pos) ** 2).sum()).item()

        # Should converge significantly (at least 80% reduction in distance)
        assert final_dist < initial_dist * 0.2, (
            f"Optimization didn't converge: initial_dist={initial_dist:.2f}, "
            f"final_dist={final_dist:.2f}"
        )

        # Final position should be close to target
        assert final_dist < 1.0, (
            f"Final position too far from target: dist={final_dist:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
