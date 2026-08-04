"""
Tests for GaussianSplatModel class.
"""

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


@pytest.fixture
def simple_2d_setup():
    """Create simple 2D setup for testing."""
    shape = (10, 10)
    N = 3  # Number of splats

    # Centers: one at each corner and one in middle
    centers0 = np.array(
        [
            [2.0, 2.0],  # Top-left region
            [7.0, 7.0],  # Bottom-right region
            [5.0, 5.0],  # Center
        ],
        dtype=np.float32,
    )

    # Initial isotropic covariances
    L0 = np.zeros((N, 2, 2), dtype=np.float32)
    L0[:, 0, 0] = 1.0  # σx = 1.0
    L0[:, 1, 1] = 1.0  # σy = 1.0

    # Amplitudes
    amps0 = np.array([0.8, 0.6, 1.0], dtype=np.float32)

    # Constraints
    sigma_min_diag = [0.3, 0.3]
    sigma_max_diag = [3.0, 3.0]

    return {
        "shape": shape,
        "centers0": centers0,
        "L0": L0,
        "amps0": amps0,
        "sigma_min_diag": sigma_min_diag,
        "sigma_max_diag": sigma_max_diag,
    }


@pytest.fixture
def simple_3d_setup():
    """Create simple 3D setup for testing."""
    shape = (8, 8, 8)
    N = 2

    centers0 = np.array(
        [
            [2.0, 2.0, 2.0],
            [6.0, 6.0, 6.0],
        ],
        dtype=np.float32,
    )

    L0 = np.zeros((N, 3, 3), dtype=np.float32)
    L0[:, 0, 0] = 1.2
    L0[:, 1, 1] = 1.0
    L0[:, 2, 2] = 0.8

    amps0 = np.array([1.0, 0.7], dtype=np.float32)
    sigma_min_diag = [0.4, 0.4, 0.4]

    return {
        "shape": shape,
        "centers0": centers0,
        "L0": L0,
        "amps0": amps0,
        "sigma_min_diag": sigma_min_diag,
        "sigma_max_diag": None,
    }


class TestGaussianSplatModelInitialization:
    """Test model initialization and parameter setup."""

    def test_default_device_is_mps_on_macos(self, simple_2d_setup) -> None:
        """On macOS the default device auto-resolves to MPS (use_metal=True).

        This test guards the deliberate default behavior provided by the
        centralized device.resolve_torch_device helper. On systems without
        MPS the test skips so non-Mac CI stays clean.
        """
        from luxar.gsplats.utils.device import is_mps_available

        if not is_mps_available():
            pytest.skip("MPS backend not available")

        # use_cuda=False so a CUDA-enabled mac (rare but possible) doesn't beat MPS
        model = GaussianSplatModel(**simple_2d_setup, truncate=2.0, use_cuda=False)
        assert model.raw_mu.device.type == "mps"

    def test_model_creation_2d(self, simple_2d_setup) -> None:
        """Test basic model creation in 2D."""
        setup = simple_2d_setup

        model = GaussianSplatModel(**setup, truncate=2.0)

        # Check basic properties
        assert model.shape == setup["shape"]
        assert model.dim == 2
        assert model.truncate == 2.0

        # Check parameter shapes
        assert model.raw_mu.shape == (3, 2)  # 3 splats, 2D
        assert model.raw_L_diag.shape == (3, 2)  # 3 splats, 2D diagonal
        assert model.raw_a.shape == (3,)  # 3 amplitudes

    def test_model_creation_3d(self, simple_3d_setup) -> None:
        """Test basic model creation in 3D."""
        setup = simple_3d_setup

        model = GaussianSplatModel(**setup, truncate=3.0)

        assert model.shape == setup["shape"]
        assert model.dim == 3
        # Explicitly supplied above — this pins round-tripping, not the default.
        assert model.truncate == 3.0

        # Check parameter shapes for 3D
        assert model.raw_mu.shape == (2, 3)  # 2 splats, 3D
        assert model.raw_L_diag.shape == (2, 3)  # 2 splats, 3D diagonal
        assert model.raw_a.shape == (2,)  # 2 amplitudes

        # Check off-diagonal elements: 3D has 3 off-diagonal elements per matrix
        expected_off_diag = 3  # (1,0), (2,0), (2,1) per matrix
        assert model.L_off.shape == (2, expected_off_diag)

    def test_empty_model(self) -> None:
        """Test model creation with no splats."""
        shape = (5, 5)
        centers0 = np.zeros((0, 2), dtype=np.float32)
        L0 = np.zeros((0, 2, 2), dtype=np.float32)
        amps0 = np.zeros((0,), dtype=np.float32)
        sigma_min_diag = [0.5, 0.5]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
        )

        assert model.raw_mu.shape == (0, 2)
        assert model.raw_L_diag.shape == (0, 2)
        assert model.raw_a.shape == (0,)

    def test_single_splat_1d(self) -> None:
        """Test model with single 1D splat."""
        shape = (20,)
        centers0 = np.array([[10.0]], dtype=np.float32)
        L0 = np.array([[[1.5]]], dtype=np.float32)
        amps0 = np.array([0.8], dtype=np.float32)
        sigma_min_diag = [0.3]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
        )

        assert model.dim == 1
        assert model.raw_mu.shape == (1, 1)
        assert model.raw_L_diag.shape == (1, 1)
        assert model.L_off.shape == (1, 0)  # No off-diagonals in 1D

    def test_device_placement(self, simple_2d_setup) -> None:
        """Test model creation on different devices."""
        setup = simple_2d_setup

        # Test CPU
        model_cpu = GaussianSplatModel(**setup, device=torch.device("cpu"))
        assert model_cpu.raw_mu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = GaussianSplatModel(**setup, device=torch.device("cuda"))
            assert model_cuda.raw_mu.device.type == "cuda"

    def test_auto_device_honors_accelerator_opt_outs(
        self, simple_2d_setup, monkeypatch
    ) -> None:
        """Auto device selection should respect use_cuda/use_metal flags."""
        setup = simple_2d_setup

        class FakeMPSBackend:
            def is_available(self) -> bool:
                return True

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(), raising=False)

        model = GaussianSplatModel(**setup, use_cuda=False, use_metal=False)

        assert model.raw_mu.device.type == "cpu"

    def test_parameter_validation(self) -> None:
        """Test parameter validation during initialization."""
        shape = (10, 10)
        centers0 = np.array([[5.0, 5.0]], dtype=np.float32)
        L0 = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        amps0 = np.array([1.0], dtype=np.float32)

        # Test mismatched sigma_min_diag dimension
        with pytest.raises(AssertionError, match="sigma_min_diag must be length d"):
            GaussianSplatModel(
                shape=shape,
                centers0=centers0,
                L0=L0,
                amps0=amps0,
                sigma_min_diag=[0.5, 0.5, 0.5],  # Wrong dimension
            )

        # Test mismatched sigma_max_diag dimension
        with pytest.raises(AssertionError, match="sigma_max_diag must be length d"):
            GaussianSplatModel(
                shape=shape,
                centers0=centers0,
                L0=L0,
                amps0=amps0,
                sigma_min_diag=[0.5, 0.5],
                sigma_max_diag=[2.0],  # Wrong dimension
            )


class TestParameterRetrieval:
    """Test current parameter retrieval and transformations."""

    def test_current_params_2d(self, simple_2d_setup) -> None:
        """Test parameter retrieval in 2D."""
        setup = simple_2d_setup
        model = GaussianSplatModel(**setup)

        centers, L, amps = model.current_params()

        # Check shapes
        assert centers.shape == (3, 2)
        assert L.shape == (3, 2, 2)
        assert amps.shape == (3,)

        # Check data types
        assert centers.dtype == torch.float32
        assert L.dtype == torch.float32
        assert amps.dtype == torch.float32

        # Check that centers are within bounds
        centers_np = centers.detach().cpu().numpy()
        assert np.all(centers_np >= 0)
        assert np.all(centers_np[:, 0] <= setup["shape"][0] - 1)
        assert np.all(centers_np[:, 1] <= setup["shape"][1] - 1)

        # Check that L matrices are lower triangular
        L_np = L.detach().cpu().numpy()
        for i in range(3):
            # Upper triangle should be zero
            assert L_np[i, 0, 1] == 0.0  # (0,1) element

        # Check that amplitudes are non-negative
        amps_np = amps.detach().cpu().numpy()
        assert np.all(amps_np >= 0)

    def test_current_params_bounds_enforcement(self, simple_2d_setup) -> None:
        """Test that parameters respect bounds."""
        setup = simple_2d_setup
        model = GaussianSplatModel(**setup)

        centers, L, amps = model.current_params()

        # Check diagonal constraints
        L_np = L.detach().cpu().numpy()
        for i in range(3):
            for j in range(2):
                assert L_np[i, j, j] >= setup["sigma_min_diag"][j]
                if setup["sigma_max_diag"] is not None:
                    assert L_np[i, j, j] <= setup["sigma_max_diag"][j]

    def test_sigma_diag_calculation(self, simple_2d_setup) -> None:
        """Test diagonal covariance calculation."""
        setup = simple_2d_setup
        model = GaussianSplatModel(**setup)

        centers, L, amps = model.current_params()

        # Test static method
        sigma_diag = model._sigma_diag_from_L(L[0])  # Single matrix
        assert sigma_diag.shape == (2,)

        # Test batch method
        sigma_diag_batch = model._sigma_diag_from_L(L)  # Batch of matrices
        assert sigma_diag_batch.shape == (3, 2)

        # Verify calculation: Σ_ii = sum_j L[i,j]^2
        L_np = L.detach().cpu().numpy()
        sigma_expected = np.sum(L_np * L_np, axis=2)
        sigma_actual = sigma_diag_batch.detach().cpu().numpy()
        np.testing.assert_allclose(sigma_actual, sigma_expected, rtol=1e-5)

    def test_cholesky_reconstruction(self, simple_2d_setup) -> None:
        """Test Cholesky factor reconstruction from parameters."""
        setup = simple_2d_setup
        model = GaussianSplatModel(**setup)

        L = model._build_L()

        # Check that result is lower triangular
        L_np = L.detach().cpu().numpy()
        for i in range(3):
            # Upper triangle should be zero
            assert L_np[i, 0, 1] == 0.0
            # Diagonal should be positive
            assert L_np[i, 0, 0] > 0.0
            assert L_np[i, 1, 1] > 0.0


class TestRendering:
    """Test rendering functionality."""

    def test_forward_pass_2d(self, simple_2d_setup) -> None:
        """Test forward pass (rendering) in 2D."""
        setup = simple_2d_setup
        model = GaussianSplatModel(**setup, truncate=2.0)

        with torch.no_grad():
            output = model()

        # Check output shape and type
        assert output.shape == setup["shape"]
        assert output.dtype == torch.float32

        # Check that output is non-negative (since we're adding Gaussians)
        assert torch.all(output >= 0)

        # Check that output has meaningful values (not all zeros)
        assert torch.sum(output) > 0

        # Check that max value is reasonable
        assert torch.max(output) <= sum(setup["amps0"]) * 1.1  # Allow some margin

    def test_forward_pass_3d(self, simple_3d_setup) -> None:
        """Test forward pass in 3D."""
        setup = simple_3d_setup
        model = GaussianSplatModel(**setup, truncate=2.0)

        with torch.no_grad():
            output = model()

        assert output.shape == setup["shape"]
        assert output.dtype == torch.float32
        assert torch.all(output >= 0)
        assert torch.sum(output) > 0

    def test_model_consistency_after_state_copy(self, simple_2d_setup) -> None:
        """Test that models give identical results after state copying."""
        setup = simple_2d_setup

        # Create two identical models
        model1 = GaussianSplatModel(**setup)
        model2 = GaussianSplatModel(**setup)

        # Force same parameters by copying state
        model2.load_state_dict(model1.state_dict())

        with torch.no_grad():
            output1 = model1()
            output2 = model2()

        # Results should be identical after state copying
        torch.testing.assert_close(output1, output2, atol=1e-7, rtol=1e-6)

    def test_empty_model_rendering(self) -> None:
        """Test rendering with no splats."""
        shape = (5, 5)
        centers0 = np.zeros((0, 2), dtype=np.float32)
        L0 = np.zeros((0, 2, 2), dtype=np.float32)
        amps0 = np.zeros((0,), dtype=np.float32)
        sigma_min_diag = [0.5, 0.5]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
        )

        with torch.no_grad():
            output = model()

        # Should be all zeros
        assert output.shape == shape
        assert torch.all(output == 0)

    def test_single_gaussian_properties(self) -> None:
        """Test properties of a single well-defined Gaussian."""
        shape = (21, 21)  # Odd size for clear center
        center = np.array([[10.0, 10.0]], dtype=np.float32)  # Exactly in center
        L = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)  # Isotropic
        amp = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModel(
            shape=shape,
            centers0=center,
            L0=L,
            amps0=amp,
            sigma_min_diag=[0.1, 0.1],
            truncate=3.0,
        )

        with torch.no_grad():
            output = model()

        output_np = output.cpu().numpy()

        # Maximum should be at the center (or very close)
        max_idx = np.unravel_index(np.argmax(output_np), output_np.shape)
        assert abs(max_idx[0] - 10) <= 1  # Within 1 pixel of center
        assert abs(max_idx[1] - 10) <= 1

        # Should be symmetric around center (approximately)
        assert output_np[9, 10] == pytest.approx(output_np[11, 10], rel=0.1)
        assert output_np[10, 9] == pytest.approx(output_np[10, 11], rel=0.1)


class TestGradientFlow:
    """Test gradient computation and backpropagation."""

    def test_gradient_computation(self, simple_2d_setup) -> None:
        """Test that gradients flow through the model correctly."""
        setup = simple_2d_setup
        model = GaussianSplatModel(**setup)

        # Create a simple target (Gaussian at center) on the same device as model
        device = model.raw_mu.device
        target = torch.zeros(setup["shape"], dtype=torch.float32, device=device)
        target[5, 5] = 1.0

        # Forward pass
        output = model()
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()  # type: ignore[no-untyped-call]

        # Check that all parameters have gradients
        assert model.raw_mu.grad is not None
        assert model.raw_L_diag.grad is not None
        assert model.raw_a.grad is not None

        # Check that gradients are not all zero
        assert not torch.allclose(
            model.raw_mu.grad, torch.zeros_like(model.raw_mu.grad)
        )
        assert not torch.allclose(
            model.raw_L_diag.grad, torch.zeros_like(model.raw_L_diag.grad)
        )
        assert not torch.allclose(model.raw_a.grad, torch.zeros_like(model.raw_a.grad))

    def test_optimization_step(self, simple_2d_setup) -> None:
        """Test that model parameters can be optimized."""
        setup = simple_2d_setup
        model = GaussianSplatModel(**setup)

        # Simple target on the same device as model
        device = model.raw_mu.device
        target = torch.ones(setup["shape"], dtype=torch.float32, device=device) * 0.1

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Initial loss
        initial_output = model()

        # Optimization step
        optimizer.zero_grad()
        output = model()
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        # New loss
        new_output = model()

        # Loss should generally decrease (allow some tolerance for stochastic effects)
        # At minimum, parameters should have changed
        assert not torch.allclose(initial_output, new_output, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_splats(self) -> None:
        """Test with very small splat sizes."""
        shape = (10, 10)
        centers0 = np.array([[5.0, 5.0]], dtype=np.float32)
        L0 = np.array([[[0.1, 0.0], [0.0, 0.1]]], dtype=np.float32)  # Very small
        amps0 = np.array([1.0], dtype=np.float32)
        sigma_min_diag = [0.05, 0.05]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            truncate=3.0,
        )

        with torch.no_grad():
            output = model()

        assert torch.all(torch.isfinite(output))
        assert torch.sum(output) > 0  # Should still produce some output

    def test_very_large_splats(self) -> None:
        """Test with very large splat sizes."""
        shape = (10, 10)
        centers0 = np.array([[5.0, 5.0]], dtype=np.float32)
        L0 = np.array([[[5.0, 0.0], [0.0, 5.0]]], dtype=np.float32)  # Very large
        amps0 = np.array([0.1], dtype=np.float32)  # Small amplitude to compensate
        sigma_min_diag = [0.1, 0.1]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            truncate=2.0,  # Small truncation for efficiency
        )

        with torch.no_grad():
            output = model()

        assert torch.all(torch.isfinite(output))
        # Large splats should affect most of the image
        assert torch.sum(output > 1e-6) > shape[0] * shape[1] * 0.5

    def test_boundary_centers(self) -> None:
        """Test with centers at image boundaries."""
        shape = (10, 10)
        centers0 = np.array(
            [
                [0.0, 0.0],  # Corner
                [0.0, 9.0],  # Corner
                [9.0, 0.0],  # Corner
                [9.0, 9.0],  # Corner
                [5.0, 0.0],  # Edge
            ],
            dtype=np.float32,
        )

        N = len(centers0)
        L0 = np.tile(np.eye(2)[None, :, :], (N, 1, 1)).astype(np.float32)
        amps0 = np.ones(N, dtype=np.float32)
        sigma_min_diag = [0.3, 0.3]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
        )

        with torch.no_grad():
            output = model()

        assert torch.all(torch.isfinite(output))
        assert torch.sum(output) > 0


class TestVoxelSizeEccentricity:
    """Tests for voxel_size-aware eccentricity constraints in _build_L()."""

    def test_voxel_size_none_same_as_before(self, simple_3d_setup):
        """voxel_size=None produces same result as omitting it."""
        setup = simple_3d_setup
        model_none = GaussianSplatModel(
            **setup,
            max_eccentricity=4.0,
            device=torch.device("cpu"),
        )
        model_explicit = GaussianSplatModel(
            **setup,
            max_eccentricity=4.0,
            voxel_size=None,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            L_none = model_none._build_L()
            L_explicit = model_explicit._build_L()
        assert torch.allclose(L_none, L_explicit)

    def test_isotropic_voxel_size_same_as_none(self, simple_3d_setup):
        """voxel_size=(1,1,1) produces identical _build_L() as None."""
        setup = simple_3d_setup
        model_none = GaussianSplatModel(
            **setup,
            max_eccentricity=4.0,
            device=torch.device("cpu"),
        )
        model_iso = GaussianSplatModel(
            **setup,
            max_eccentricity=4.0,
            voxel_size=np.array([1.0, 1.0, 1.0]),
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            L_none = model_none._build_L()
            L_iso = model_iso._build_L()
        assert torch.allclose(L_none, L_iso, atol=1e-6)

    def test_anisotropic_allows_physical_isotropy(self):
        """Physically isotropic splat is NOT clamped with anisotropic voxels."""
        shape = (10, 50, 50)
        N = 1
        d = 3
        voxel_size = np.array([5.0, 1.0, 1.0])

        centers0 = np.array([[5.0, 25.0, 25.0]], dtype=np.float32)
        # L_diag in voxel space: (1, 5, 5) → physical: (5, 5, 5) = isotropic
        L0 = np.zeros((N, d, d), dtype=np.float32)
        L0[0, 0, 0] = 1.0
        L0[0, 1, 1] = 5.0
        L0[0, 2, 2] = 5.0
        amps0 = np.array([1.0], dtype=np.float32)
        sigma_min_diag = [0.1, 0.1, 0.1]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            max_eccentricity=2.0,
            voxel_size=voxel_size,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            L = model._build_L()
        diag = torch.diagonal(L[0])
        # Physical diags: 1*5=5, 5*1=5, 5*1=5 — isotropic, no clamping
        assert diag[0].item() == pytest.approx(1.0, abs=0.1)
        assert diag[1].item() == pytest.approx(5.0, abs=0.1)
        assert diag[2].item() == pytest.approx(5.0, abs=0.1)

    def test_anisotropic_clamps_physical_elongation(self):
        """Physically elongated splat IS clamped."""
        shape = (10, 50, 50)
        N = 1
        d = 3
        voxel_size = np.array([5.0, 1.0, 1.0])

        centers0 = np.array([[5.0, 25.0, 25.0]], dtype=np.float32)
        # L_diag voxel: (1, 50, 50) → physical: (5, 50, 50) → ratio 10x
        L0 = np.zeros((N, d, d), dtype=np.float32)
        L0[0, 0, 0] = 1.0
        L0[0, 1, 1] = 50.0
        L0[0, 2, 2] = 50.0
        amps0 = np.array([1.0], dtype=np.float32)
        sigma_min_diag = [0.1, 0.1, 0.1]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            max_eccentricity=4.0,
            voxel_size=voxel_size,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            L = model._build_L()
        diag = torch.diagonal(L[0])
        # Physical diags after clamping: min_phys ≈ 5, max_allowed = 5*2 = 10
        # So physical ratio should be <= sqrt(4) = 2
        diag_phys = diag * torch.tensor(voxel_size)
        ratio = diag_phys.max() / diag_phys.min()
        assert ratio.item() <= 2.0 + 0.01  # sqrt(4) with tolerance

    def test_sigma_min_diag_preserved_after_eccentricity(self):
        """sigma_min_diag is respected even after eccentricity clamping."""
        shape = (10, 50, 50)
        N = 1
        d = 3
        # Extreme anisotropy: Z is 100x physical size per voxel
        voxel_size = np.array([100.0, 1.0, 1.0])
        sigma_min_diag = [0.3, 0.3, 0.3]

        centers0 = np.array([[5.0, 25.0, 25.0]], dtype=np.float32)
        # L_diag voxel: (3, 0.5, 0.5) → physical: (300, 0.5, 0.5) → ratio 600
        # Eccentricity will try to shrink axis 0, but sigma_min must hold
        L0 = np.zeros((N, d, d), dtype=np.float32)
        L0[0, 0, 0] = 3.0
        L0[0, 1, 1] = 0.5
        L0[0, 2, 2] = 0.5
        amps0 = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            max_eccentricity=4.0,
            voxel_size=voxel_size,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            L = model._build_L()
        diag = torch.diagonal(L[0])
        # All diagonals must be >= sigma_min_diag
        for i in range(d):
            assert diag[i].item() >= sigma_min_diag[i] - 1e-6

    def test_without_voxel_size_same_diags_clamped(self):
        """Without voxel_size, voxel-space-elongated splat IS clamped."""
        shape = (10, 50, 50)
        N = 1
        d = 3
        centers0 = np.array([[5.0, 25.0, 25.0]], dtype=np.float32)
        # L_diag: (1, 5, 5) → voxel ratio = 5, eccentricity = 25 > 4 → clamped
        L0 = np.zeros((N, d, d), dtype=np.float32)
        L0[0, 0, 0] = 1.0
        L0[0, 1, 1] = 5.0
        L0[0, 2, 2] = 5.0
        amps0 = np.array([1.0], dtype=np.float32)
        sigma_min_diag = [0.1, 0.1, 0.1]

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            max_eccentricity=4.0,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            L = model._build_L()
        diag = torch.diagonal(L[0])
        ratio = diag.max() / diag.min()
        assert ratio.item() <= 2.0 + 0.01  # sqrt(4)
