# test_decomposition_basic.py

"""
Basic tests for multi-scale decomposition functionality.

Tests non-negativity, reconstruction accuracy, output shapes, and device handling.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.multiscale import (
    MultiScaleDecomposer,
    decompose_image,
    decomposition_loss,
)


@pytest.fixture
def simple_2d_image():
    """Create a simple 2D test image."""
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    image = np.exp(-(X**2 + Y**2) / 0.5)
    return image.astype(np.float32)


@pytest.fixture
def simple_3d_volume():
    """Create a simple 3D test volume."""
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    z = np.linspace(-1, 1, 32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    volume = np.exp(-(X**2 + Y**2 + Z**2) / 0.5)
    return volume.astype(np.float32)


class TestMultiScaleDecomposer:
    """Tests for MultiScaleDecomposer class."""

    def test_initialization(self):
        """Test model initialization with different shapes."""
        # 2D
        model = MultiScaleDecomposer((64, 64), scales=[1, 2, 4])
        assert len(model.raw_images) == 3
        assert model.raw_images[0].shape == (64, 64)
        assert model.raw_images[1].shape == (32, 32)
        assert model.raw_images[2].shape == (16, 16)

        # 3D
        model = MultiScaleDecomposer((32, 32, 32), scales=[1, 2])
        assert len(model.raw_images) == 2
        assert model.raw_images[0].shape == (32, 32, 32)
        assert model.raw_images[1].shape == (16, 16, 16)

    def test_forward_pass_shapes(self):
        """Test forward pass returns correct shapes."""
        model = MultiScaleDecomposer((64, 64), scales=[1, 2, 4])

        scales_list, upsampled_list, reconstruction = model()

        # Check list lengths
        assert len(scales_list) == 3
        assert len(upsampled_list) == 3

        # Check scale shapes
        assert scales_list[0].shape == (64, 64)
        assert scales_list[1].shape == (32, 32)
        assert scales_list[2].shape == (16, 16)

        # Check upsampled shapes (all full resolution)
        for img in upsampled_list:
            assert img.shape == (64, 64)

        # Check reconstruction shape
        assert reconstruction.shape == (64, 64)

    def test_non_negativity(self):
        """Test that all outputs are non-negative."""
        model = MultiScaleDecomposer((64, 64), scales=[1, 2, 4])

        # Initialize with random negative values
        for param in model.raw_images:
            param.data = torch.randn_like(param.data)

        scales_list, upsampled_list, reconstruction = model()

        # All scales should be non-negative
        for img in scales_list:
            assert torch.all(img >= 0), "Scale component has negative values"

        # All upsampled should be non-negative
        for img in upsampled_list:
            assert torch.all(img >= 0), "Upsampled component has negative values"

        # Reconstruction should be non-negative
        assert torch.all(
            reconstruction >= 0
        ), "Reconstruction has negative values"

    def test_pyramid_initialization(self, simple_2d_image):
        """Test Gaussian pyramid initialization."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2, 4])

        # Before initialization, reconstruction should be poor
        _, _, recon_before = model()
        error_before = torch.mean(torch.abs(recon_before - target)).item()

        # Initialize from pyramid
        model.initialize_from_pyramid(target)

        # After initialization, reconstruction should be better
        _, _, recon_after = model()
        error_after = torch.mean(torch.abs(recon_after - target)).item()

        assert error_after < error_before, \
            "Pyramid initialization should improve reconstruction"
        assert error_after < 0.1, \
            f"Pyramid initialization error too high: {error_after}"

    def test_finest_scale_initialization(self, simple_2d_image):
        """Test finest scale initialization puts all energy in finest scale."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2, 4])

        # Initialize with all energy in finest scale
        model.initialize_finest_scale(target)

        scales_list, _, reconstruction = model()

        # Check that finest scale (scale factor 1) has essentially all energy
        finest_idx = 0  # First scale is scale factor 1
        finest_energy = torch.sum(scales_list[finest_idx]).item()
        total_energy = sum(torch.sum(s).item() for s in scales_list)

        # Avoid division by zero
        if total_energy > 1e-12:
            finest_fraction = finest_energy / total_energy

            # Finest scale should have >98% of energy initially
            assert finest_fraction > 0.98, \
                f"Finest scale should have >98% energy, got {finest_fraction:.2%}"

            # Other scales should have negligible energy
            for i, scale_img in enumerate(scales_list):
                if i != finest_idx:
                    scale_energy = torch.sum(scale_img).item()
                    scale_fraction = scale_energy / total_energy
                    assert scale_fraction < 0.02, \
                        f"Scale {i} should have <2% energy, got {scale_fraction:.2%}"

        # Check that reconstruction has roughly the right energy
        # (not necessarily perfect due to softplus nonlinearity)
        target_energy = torch.sum(target).item()
        recon_energy = torch.sum(reconstruction).item()
        energy_ratio = recon_energy / (target_energy + 1e-12)
        assert 0.1 < energy_ratio < 10.0, \
            f"Reconstruction energy ratio unreasonable: {energy_ratio:.2f}"

    def test_uniform_initialization(self, simple_2d_image):
        """Test uniform initialization splits energy equally across scales when upsampled."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2, 4])

        # Initialize with uniform energy distribution
        model.initialize_uniform(target)

        scales_list, upsampled_list, reconstruction = model()

        # Calculate energy per scale (from upsampled versions)
        # This is what matters for the reconstruction
        total_upsampled_energy = sum(torch.sum(s).item() for s in upsampled_list)

        # Each UPSAMPLED scale should have approximately 1/K of the total energy
        for i, upsampled_scale in enumerate(upsampled_list):
            scale_energy = torch.sum(upsampled_scale).item()
            energy_fraction = scale_energy / total_upsampled_energy
            expected_fraction = 1.0 / len(upsampled_list)

            # Check within reasonable tolerance (25% to 42% for 3 scales = 33% ± 8%)
            # Allow wider tolerance due to softplus nonlinearity
            assert 0.25 < energy_fraction < 0.42, \
                f"Scale {i} (upsampled) should have ~{expected_fraction:.1%} energy, got {energy_fraction:.1%}"

        # Check that reconstruction has roughly the right total energy
        target_energy = torch.sum(target).item()
        recon_energy = torch.sum(reconstruction).item()
        energy_ratio = recon_energy / (target_energy + 1e-12)
        assert 0.8 < energy_ratio < 1.2, \
            f"Reconstruction energy ratio should be ~1.0, got {energy_ratio:.2f}"

    def test_coarse_initialization(self, simple_2d_image):
        """Test coarse initialization weights energy toward coarse scales."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        scales = [1, 2, 4]
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=scales)

        # Initialize with coarse-weighted energy distribution
        model.initialize_coarse(target)

        scales_list, upsampled_list, reconstruction = model()

        # Calculate energy per scale (from upsampled versions)
        total_upsampled_energy = sum(torch.sum(s).item() for s in upsampled_list)

        # For scales [1, 2, 4], expected weights are [1, 2, 4] -> normalized [1/7, 2/7, 4/7]
        # That's approximately [14%, 29%, 57%]
        expected_fractions = [s / sum(scales) for s in scales]

        for i, (upsampled_scale, expected_frac) in enumerate(zip(upsampled_list, expected_fractions)):
            scale_energy = torch.sum(upsampled_scale).item()
            energy_fraction = scale_energy / total_upsampled_energy

            # Check that coarser scales have progressively more energy
            # Allow 15% tolerance due to softplus nonlinearity
            lower_bound = expected_frac * 0.85
            upper_bound = expected_frac * 1.15

            assert lower_bound < energy_fraction < upper_bound, \
                f"Scale {i} should have ~{expected_frac:.1%} energy, got {energy_fraction:.1%}"

        # Verify that coarsest scale has most energy
        energies = [torch.sum(s).item() for s in upsampled_list]
        assert energies[-1] > energies[0], \
            "Coarsest scale should have more energy than finest scale"
        assert energies[-1] > energies[1], \
            "Coarsest scale should have more energy than middle scale"

        # Check that reconstruction has roughly the right total energy
        target_energy = torch.sum(target).item()
        recon_energy = torch.sum(reconstruction).item()
        energy_ratio = recon_energy / (target_energy + 1e-12)
        assert 0.8 < energy_ratio < 1.2, \
            f"Reconstruction energy ratio should be ~1.0, got {energy_ratio:.2f}"


class TestDecompositionLoss:
    """Tests for decomposition_loss function."""

    def test_loss_computation(self, simple_2d_image):
        """Test that loss is computed correctly."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])
        model.initialize_from_pyramid(target)

        loss, stats = decomposition_loss(model, target)

        # Check loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Check stats dictionary
        assert 'recon_loss' in stats
        assert 'energy_loss' in stats
        assert 'total_loss' in stats
        assert 'energy_scale_0' in stats
        assert 'energy_scale_1' in stats

        # Check all stats are finite numbers
        for key, value in stats.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_gradients_flow(self, simple_2d_image):
        """Test that gradients flow through the loss."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])

        loss, _ = decomposition_loss(model, target)
        loss.backward()

        # Check that gradients exist and are non-zero for at least some parameters
        has_nonzero_grad = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.any(param.grad != 0):
                    has_nonzero_grad = True
                    break

        assert has_nonzero_grad, "No parameters received non-zero gradients"


class TestDecomposeImage:
    """Tests for decompose_image function."""

    def test_basic_decomposition_2d(self, simple_2d_image):
        """Test basic 2D decomposition."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=50,  # Few iterations for fast test
            verbose=False
        )

        # Check output structure
        assert len(scales_list) == 2
        assert scales_list[0].shape == simple_2d_image.shape
        assert scales_list[1].shape == tuple(s // 2 for s in simple_2d_image.shape)

        # Check stats
        assert 'final_error' in stats
        assert 'energy_distribution' in stats
        assert len(stats['energy_distribution']) == 2

        # Check non-negativity
        for img in scales_list:
            assert np.all(img >= 0), "Output contains negative values"

    def test_basic_decomposition_3d(self, simple_3d_volume):
        """Test basic 3D decomposition."""
        scales_list, stats = decompose_image(
            simple_3d_volume,
            scales=[1, 2],
            n_iters=50,
            verbose=False
        )

        assert len(scales_list) == 2
        assert scales_list[0].shape == simple_3d_volume.shape
        assert scales_list[1].shape == tuple(s // 2 for s in simple_3d_volume.shape)

    def test_reconstruction_quality(self, simple_2d_image):
        """Test that reconstruction is close to original."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2, 4],
            n_iters=200,
            verbose=False
        )

        # Compute reconstruction
        from scipy.ndimage import zoom
        reconstruction = np.zeros_like(simple_2d_image)
        for i, img in enumerate(scales_list):
            if img.shape != simple_2d_image.shape:
                zoom_factors = [o / s for o, s in zip(simple_2d_image.shape, img.shape)]
                img_upsampled = zoom(img, zoom_factors, order=1)
            else:
                img_upsampled = img
            reconstruction += img_upsampled

        # Check reconstruction error
        mse = np.mean((reconstruction - simple_2d_image) ** 2)
        assert mse < 0.01, f"Reconstruction MSE too high: {mse}"

        # Check relative error
        relative_error = np.linalg.norm(reconstruction - simple_2d_image) / \
                        (np.linalg.norm(simple_2d_image) + 1e-12)
        assert relative_error < 0.1, \
            f"Relative reconstruction error too high: {relative_error}"

    def test_device_cpu(self, simple_2d_image):
        """Test decomposition on CPU."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=10,
            device='cpu',
            verbose=False
        )
        assert len(scales_list) == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self, simple_2d_image):
        """Test decomposition on CUDA."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=10,
            device='cuda',
            verbose=False
        )
        assert len(scales_list) == 2

    def test_convergence_tracking(self, simple_2d_image):
        """Test that optimization improves over iterations and convergence tracking works."""
        # Test with auto-convergence disabled (very low threshold) to ensure full iteration count
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=100,
            max_abs_error_threshold=1e-10,  # Very low threshold to prevent early convergence
            verbose=False
        )

        history = stats['history']
        actual_iters = stats['actual_iters']

        # With very low threshold, should run all iterations
        assert actual_iters == 100
        assert len(history) == 100

        # Check that reconstruction loss generally decreases
        initial_loss = history[0]['recon_loss']
        final_loss = history[-1]['recon_loss']
        assert final_loss < initial_loss, \
            "Reconstruction loss should decrease during optimization"

        # Test convergence with reasonable threshold
        scales_list2, stats2 = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=100,
            # Use default auto-convergence
            verbose=False
        )

        # With auto-convergence, should converge early
        assert stats2['converged'], "Should converge with auto-convergence threshold"
        assert stats2['actual_iters'] < 100, "Should converge before max iterations"
        assert stats2['best_max_abs_error'] < 1e-2, "Should achieve good convergence"

    def test_initialization_methods(self, simple_2d_image):
        """Test all initialization methods produce valid results."""
        for init_method in ['pyramid', 'finest', 'uniform', 'coarse', 'zero']:
            # Zero initialization needs more iterations to converge
            n_iters = 100 if init_method == 'zero' else 50

            scales_list, stats = decompose_image(
                simple_2d_image,
                scales=[1, 2, 4],
                n_iters=n_iters,
                init_method=init_method,
                verbose=False
            )

            # Check basic output structure
            assert len(scales_list) == 3
            assert 'final_error' in stats
            assert 'energy_distribution' in stats

            # Check reconstruction quality
            from scipy.ndimage import zoom
            reconstruction = np.zeros_like(simple_2d_image)
            for img in scales_list:
                if img.shape != simple_2d_image.shape:
                    zoom_factors = [o / s for o, s in zip(simple_2d_image.shape, img.shape)]
                    img_upsampled = zoom(img, zoom_factors, order=1)
                else:
                    img_upsampled = img
                reconstruction += img_upsampled

            mse = np.mean((reconstruction - simple_2d_image) ** 2)
            # Zero initialization starts from scratch, so be more lenient
            mse_threshold = 0.2 if init_method == 'zero' else 0.1
            assert mse < mse_threshold, f"Reconstruction MSE too high for {init_method}: {mse}"

            # Check non-negativity
            for img in scales_list:
                assert np.all(img >= 0), f"Output contains negative values for {init_method}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_scale(self, simple_2d_image):
        """Test decomposition with single scale."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1],
            n_iters=50,
            verbose=False
        )
        assert len(scales_list) == 1
        # Should reconstruct perfectly with single scale
        assert stats['final_error'] < 0.01

    def test_many_scales(self, simple_2d_image):
        """Test decomposition with many scales."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2, 4, 8, 16],
            n_iters=50,
            verbose=False
        )
        assert len(scales_list) == 5

    def test_small_image(self):
        """Test with very small image."""
        small_image = np.random.rand(16, 16).astype(np.float32)
        scales_list, stats = decompose_image(
            small_image,
            scales=[1, 2],
            n_iters=10,
            verbose=False
        )
        assert len(scales_list) == 2

    def test_different_alpha_values(self, simple_2d_image):
        """Test with different alpha values."""
        for alpha in [1.0, 1.5, 2.0, 3.0]:
            scales_list, stats = decompose_image(
                simple_2d_image,
                scales=[1, 2],
                n_iters=20,
                alpha=alpha,
                verbose=False
            )
            assert len(scales_list) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
