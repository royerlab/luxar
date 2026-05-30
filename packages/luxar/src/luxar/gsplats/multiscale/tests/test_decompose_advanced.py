"""
Advanced tests for multiscale decomposition.

Tests loss types, asymmetric penalties, edge cases, movie recording,
and other advanced features to achieve comprehensive coverage.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.multiscale import decompose_image, upsample_for_visualization
from luxar.gsplats.multiscale.decompose import (
    MultiScaleDecomposer,
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
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    volume = np.exp(-(X**2 + Y**2 + Z**2) / 0.5)
    return volume.astype(np.float32)


class TestLossTypes:
    """Test different loss functions (MSE, Poisson, L1)."""

    def test_mse_loss(self, simple_2d_image) -> None:
        """Test decomposition with MSE loss."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2, 4],
            n_iters=50,
            loss_type="mse",
            verbose=False,
        )

        assert len(scales_list) == 3
        assert stats["final_error"] < 0.05
        # Verify all outputs are non-negative
        for img in scales_list:
            assert np.all(img >= 0)

    def test_poisson_loss(self, simple_2d_image) -> None:
        """Test decomposition with Poisson loss."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2, 4],
            n_iters=50,
            loss_type="poisson",
            verbose=False,
        )

        assert len(scales_list) == 3
        # Poisson loss should still achieve good reconstruction
        assert stats["final_error"] < 0.1
        for img in scales_list:
            assert np.all(img >= 0)

    def test_l1_loss_explicit(self, simple_2d_image) -> None:
        """Test decomposition with explicit L1 loss."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=50,
            loss_type="l1",
            verbose=False,
        )

        assert len(scales_list) == 2
        assert stats["final_error"] < 0.05

    def test_different_losses_produce_different_results(self, simple_2d_image) -> None:
        """Test that different loss types produce different decompositions."""
        results = {}

        for loss_type in ["l1", "mse", "poisson"]:
            scales_list, stats = decompose_image(
                simple_2d_image,
                scales=[1, 2],
                n_iters=100,
                loss_type=loss_type,
                verbose=False,
            )
            # Store the finest scale for comparison
            results[loss_type] = scales_list[0]

        # Results should be different (not identical decompositions)
        assert not np.allclose(results["l1"], results["mse"], atol=1e-4)


class TestAsymmetricPenalty:
    """Test asymmetric penalty for over-prediction."""

    def test_asymmetric_penalty_with_mse(self, simple_2d_image) -> None:
        """Test MSE loss with asymmetric penalty."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=50,
            loss_type="mse",
            asymmetric_penalty=10.0,
            verbose=False,
        )

        assert len(scales_list) == 2
        # Should still achieve good quality
        assert stats["final_error"] < 0.1

    def test_asymmetric_penalty_with_poisson(self, simple_2d_image) -> None:
        """Test Poisson loss with asymmetric penalty."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=50,
            loss_type="poisson",
            asymmetric_penalty=10.0,
            verbose=False,
        )

        assert len(scales_list) == 2
        assert stats["final_error"] < 0.1

    def test_no_asymmetric_penalty(self, simple_2d_image) -> None:
        """Test with asymmetric_penalty=None (symmetric loss)."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=50,
            loss_type="l1",
            asymmetric_penalty=None,  # Disable asymmetric penalty
            verbose=False,
        )

        assert len(scales_list) == 2
        # Should still work with symmetric loss
        assert stats["final_error"] < 0.1


class TestUniformImageHandling:
    """Test handling of uniform images (edge case)."""

    def test_uniform_image(self) -> None:
        """Test decomposition of completely uniform image."""
        uniform_image = np.ones((64, 64), dtype=np.float32) * 5.0

        # Should handle gracefully without division by zero
        scales_list, stats = decompose_image(
            uniform_image, scales=[1, 2], n_iters=10, verbose=False
        )

        assert len(scales_list) == 2
        # Should have converged (not much to optimize)
        assert stats["converged"]

    def test_nearly_uniform_image(self) -> None:
        """Test decomposition of nearly uniform image."""
        nearly_uniform = np.ones((64, 64), dtype=np.float32) * 5.0
        nearly_uniform[32, 32] = 5.001  # Tiny variation

        scales_list, stats = decompose_image(
            nearly_uniform, scales=[1, 2], n_iters=20, verbose=False
        )

        assert len(scales_list) == 2
        # Should handle without numerical issues
        for img in scales_list:
            assert np.all(np.isfinite(img))


class TestDeviceHandling:
    """Test device-specific code paths."""

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_device_2d(self, simple_2d_image) -> None:
        """Test decomposition on MPS device with 2D data."""
        scales_list, stats = decompose_image(
            simple_2d_image, scales=[1, 2], n_iters=20, device="mps", verbose=False
        )

        assert len(scales_list) == 2
        assert stats["final_error"] < 0.1

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_device_3d(self, simple_3d_volume) -> None:
        """Test decomposition on MPS device with 3D data (uses fallback)."""
        scales_list, stats = decompose_image(
            simple_3d_volume, scales=[1, 2], n_iters=20, device="mps", verbose=False
        )

        assert len(scales_list) == 2
        # MPS uses interpolate fallback for 3D, should still work
        assert stats["final_error"] < 0.2

    def test_auto_device_detection(self, simple_2d_image) -> None:
        """Test automatic device detection."""
        # Call without device parameter - should auto-detect
        scales_list, stats = decompose_image(
            simple_2d_image, scales=[1, 2], n_iters=10, verbose=False
        )

        assert len(scales_list) == 2


class TestMovieRecording:
    """Test napari movie recording feature."""

    def test_movie_recording_enabled(self, simple_2d_image) -> None:
        """Test that movie frames are recorded when enabled."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=20,
            napari_movie=True,
            movie_every=5,
            movie_max_frames=100,
            max_abs_error_threshold=1e-10,  # Prevent early convergence
            verbose=False,
        )

        # Should have movie_frames in stats
        assert "movie_frames" in stats
        assert stats["movie_frames"] is not None

        movie_frames = stats["movie_frames"]
        assert "target" in movie_frames
        assert "reconstruction" in movie_frames
        assert "residual" in movie_frames
        assert "scales" in movie_frames
        assert "iterations" in movie_frames

        # Should have recorded frames (every 5 iterations: 5, 10, 15, 20)
        assert len(movie_frames["target"]) == 4
        assert len(movie_frames["iterations"]) == 4
        assert movie_frames["iterations"] == [5, 10, 15, 20]

    def test_movie_max_frames_limit(self, simple_2d_image) -> None:
        """Test that movie frame limit is respected."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=50,
            napari_movie=True,
            movie_every=1,
            movie_max_frames=10,  # Limit to 10 frames
            verbose=False,
        )

        movie_frames = stats["movie_frames"]
        # Should not exceed max_frames
        assert len(movie_frames["target"]) <= 10

    def test_movie_disabled_by_default(self, simple_2d_image) -> None:
        """Test that movie recording is disabled by default."""
        scales_list, stats = decompose_image(
            simple_2d_image, scales=[1, 2], n_iters=10, verbose=False
        )

        assert stats["movie_frames"] is None


class TestConvergenceEdgeCases:
    """Test convergence edge cases and best state tracking."""

    def test_best_state_restored(self, simple_2d_image) -> None:
        """Test that best state is restored if quality degrades."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=100,
            max_abs_error_threshold=1e-10,  # Won't converge, tests best state restore
            verbose=False,
        )

        # Should have best_iteration tracked
        assert "best_iteration" in stats
        assert stats["best_iteration"] > 0
        assert stats["best_iteration"] <= stats["actual_iters"]

        # Best max abs error should be finite
        assert np.isfinite(stats["best_max_abs_error"])

    def test_convergence_early_termination(self, simple_2d_image) -> None:
        """Test that optimization stops early when converged."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=1000,  # Many iterations
            max_abs_error_threshold=0.01,  # Reasonable threshold
            verbose=False,
        )

        # Should converge early
        assert stats["converged"]
        assert stats["actual_iters"] < 1000

    # [P5][P1] Data-adaptive convergence: the auto-threshold is
    # `0.01 * image_range`. Achieved best_max_abs_error must respect that
    # adaptive bound on inputs of widely different amplitude. An absolute
    # (non-adaptive) threshold would either falsely flag tiny-amplitude inputs
    # as converged at initialization or fail to converge on large-amplitude
    # inputs within reasonable iteration budgets.
    def test_threshold_auto_scales_with_amplitude(self, simple_2d_image) -> None:
        """Auto-threshold bound (1% of image range) is honoured at both
        small and large amplitudes."""
        for scale in (0.1, 1.0, 10.0):
            V = simple_2d_image * scale
            _, stats = decompose_image(V, scales=[1, 2], n_iters=500, verbose=False)
            image_range = float(V.max() - V.min())
            adaptive_threshold = 0.01 * image_range
            assert stats["converged"], (
                f"failed to converge at amplitude scale={scale}: "
                f"err={stats['best_max_abs_error']:.4e} thresh={adaptive_threshold:.4e}"
            )
            # Achieved error must respect the adaptive (range-dependent) bound.
            assert stats["best_max_abs_error"] < adaptive_threshold, (
                f"scale={scale}: err={stats['best_max_abs_error']:.4e} "
                f">= adaptive threshold {adaptive_threshold:.4e}"
            )

    # [P5] Parameter sensitivity: a tighter user-supplied threshold needs at
    # least as many iterations as a looser one to satisfy the stricter bound.
    def test_threshold_sensitivity_monotonic(self, simple_2d_image) -> None:
        """Tightening max_abs_error_threshold does not decrease iteration count."""
        loose_iters = []
        for threshold in (0.05, 0.01, 0.001):
            _, stats = decompose_image(
                simple_2d_image,
                scales=[1, 2],
                n_iters=1000,
                max_abs_error_threshold=threshold,
                verbose=False,
            )
            loose_iters.append(stats["actual_iters"])
        # Iteration counts should be monotonically non-decreasing as threshold
        # tightens.
        assert loose_iters[0] <= loose_iters[1] <= loose_iters[2], (
            f"tighter threshold should need >= iterations: {loose_iters}"
        )

    # [P5] Edge case: a flat (constant) volume reconstructs trivially under
    # the auto-threshold (default `None` → for zero-range data: 1% of mean ≈
    # 0.005 for V=0.5). Should hit threshold within a fraction of the
    # iteration budget.
    def test_flat_volume_converges_quickly(self) -> None:
        """Constant-valued input converges in well under the iteration
        budget using the auto-threshold."""
        flat = np.full((32, 32), 0.5, dtype=np.float32)
        _, stats = decompose_image(
            flat,
            scales=[1, 2, 4],
            n_iters=500,
            # Use default auto-threshold (≈ 0.005 for flat V=0.5) — that
            # is what production callers actually do.
            verbose=False,
        )
        assert stats["converged"]
        # A flat input is the easiest possible decomposition — should not need
        # anywhere near the full iteration budget.
        assert stats["actual_iters"] < 250, (
            f"flat input should converge fast, took {stats['actual_iters']} iters"
        )


class TestUpsampleForVisualization:
    """Test upsample_for_visualization helper function."""

    def test_upsample_2d_nearest(self) -> None:
        """Test upsampling with nearest interpolation."""
        img = np.random.rand(16, 16).astype(np.float32)
        target_shape = (64, 64)

        upsampled = upsample_for_visualization(
            img, target_shape, interpolation="nearest"
        )

        assert upsampled.shape == target_shape
        assert np.all(upsampled >= 0)
        # Nearest should preserve min/max values
        assert upsampled.min() >= img.min() - 1e-6
        assert upsampled.max() <= img.max() + 1e-6

    def test_upsample_2d_linear(self) -> None:
        """Test upsampling with linear interpolation."""
        img = np.random.rand(16, 16).astype(np.float32)
        target_shape = (64, 64)

        upsampled = upsample_for_visualization(
            img, target_shape, interpolation="linear"
        )

        assert upsampled.shape == target_shape
        assert np.all(upsampled >= 0)

    def test_upsample_2d_cubic(self) -> None:
        """Test upsampling with cubic interpolation."""
        img = np.random.rand(16, 16).astype(np.float32)
        target_shape = (64, 64)

        upsampled = upsample_for_visualization(img, target_shape, interpolation="cubic")

        assert upsampled.shape == target_shape
        # Cubic should clamp negatives to zero
        assert np.all(upsampled >= 0)

    def test_upsample_3d_cubic(self) -> None:
        """Test upsampling 3D with cubic (uses Keys cubic)."""
        img = np.random.rand(8, 8, 8).astype(np.float32)
        target_shape = (32, 32, 32)

        upsampled = upsample_for_visualization(img, target_shape, interpolation="cubic")

        assert upsampled.shape == target_shape
        assert np.all(upsampled >= 0)

    def test_upsample_no_op_same_shape(self) -> None:
        """Test that upsampling with same shape returns input."""
        img = np.random.rand(32, 32).astype(np.float32)

        upsampled = upsample_for_visualization(img, img.shape, interpolation="cubic")

        assert upsampled.shape == img.shape
        assert np.allclose(upsampled, img)


class TestScaleFiltering:
    """Test automatic scale filtering for small images."""

    def test_scale_filtering_warning(self) -> None:
        """Test that oversized scales are filtered with warning."""
        small_image = np.random.rand(24, 24).astype(np.float32)

        # Request scales larger than image
        scales_list, stats = decompose_image(
            small_image,
            scales=[1, 2, 4, 8, 16, 32, 64],  # 32 and 64 are too large
            n_iters=10,
            verbose=True,  # Enable to trigger warning
        )

        # Should have filtered scales
        actual_scales = stats["scales"]
        assert 64 not in actual_scales
        assert 32 not in actual_scales
        assert 16 in actual_scales  # 24/16 = 1.5, rounds to 1
        assert len(scales_list) == len(actual_scales)

    def test_all_scales_too_large(self) -> None:
        """Test when all scales are too large (fallback to scale=1)."""
        tiny_image = np.random.rand(8, 8).astype(np.float32)

        scales_list, stats = decompose_image(
            tiny_image,
            scales=[16, 32, 64],  # All too large
            n_iters=10,
            verbose=True,
        )

        # Should fallback to [1]
        assert stats["scales"] == [1]
        assert len(scales_list) == 1


class TestInitializationEdgeCases:
    """Test edge cases in initialization methods."""

    def test_zero_initialization_coverage(self, simple_2d_image) -> None:
        """Test zero initialization (all scales start at zero)."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])

        model.initialize_zero(target)

        scales_list, _, _ = model()

        # All scales should have very low energy initially
        # softplus(-10) ≈ 4.5e-5 per element, but with 64*64 elements, sums to ~0.18
        for scale_img in scales_list:
            total_energy = torch.sum(scale_img).item()
            # Each scale should have very small total energy
            assert total_energy < 1.0  # Much less than typical image energy

    def test_invalid_target_shape_raises_error(self, simple_2d_image) -> None:
        """Test that mismatched target shape raises error."""
        model = MultiScaleDecomposer((64, 64), scales=[1, 2])

        wrong_target = torch.randn(32, 32)

        with pytest.raises(ValueError, match="does not match model shape"):
            model.initialize_from_pyramid(wrong_target)

        with pytest.raises(ValueError, match="does not match model shape"):
            model.initialize_finest_scale(wrong_target)

        with pytest.raises(ValueError, match="does not match model shape"):
            model.initialize_uniform(wrong_target)

        with pytest.raises(ValueError, match="does not match model shape"):
            model.initialize_coarse(wrong_target)

        with pytest.raises(ValueError, match="does not match model shape"):
            model.initialize_zero(wrong_target)


class TestDecompositionLossEdgeCases:
    """Test decomposition_loss function edge cases."""

    def test_loss_with_mse_no_asymmetric(self, simple_2d_image) -> None:
        """Test MSE loss without asymmetric penalty."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])
        model.initialize_from_pyramid(target)

        loss, stats = decomposition_loss(
            model, target, loss_type="mse", asymmetric_penalty=None
        )

        assert torch.isfinite(loss)
        assert stats["recon_loss"] >= 0

    def test_loss_with_poisson_no_asymmetric(self, simple_2d_image) -> None:
        """Test Poisson loss without asymmetric penalty."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])
        model.initialize_from_pyramid(target)

        loss, stats = decomposition_loss(
            model, target, loss_type="poisson", asymmetric_penalty=None
        )

        assert torch.isfinite(loss)
        assert stats["recon_loss"] >= 0


class TestVerboseOutput:
    """Test verbose mode (covers logging branches)."""

    def test_verbose_mode(self, simple_2d_image) -> None:
        """Test with verbose=True to cover logging branches."""
        # This test ensures verbose logging paths are executed
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2, 4],
            n_iters=200,  # Enough iterations to converge
            verbose=True,  # Enable verbose
            max_abs_error_threshold=0.01,  # Should converge
        )

        assert len(scales_list) == 3
        # With enough iterations, should converge and trigger convergence logging
        assert stats["converged"]

    def test_verbose_no_convergence(self, simple_2d_image) -> None:
        """Test verbose output when convergence not achieved."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=5,  # Very few iterations
            max_abs_error_threshold=1e-8,  # Very strict threshold
            verbose=True,
        )

        # Should not converge
        assert not stats["converged"]


class TestBestStateTracking:
    """Test best state tracking and restoration."""

    def test_best_state_tracking_with_oscillation(self, simple_2d_image) -> None:
        """Test that best state is properly tracked even if quality oscillates."""
        # Use aggressive settings that might cause oscillation
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=50,
            lr=0.1,  # High LR can cause oscillation
            max_abs_error_threshold=1e-10,  # Won't converge
            verbose=False,
        )

        # Should track best iteration
        assert stats["best_iteration"] > 0
        assert stats["best_iteration"] <= stats["actual_iters"]

        # Best error should be finite and reasonable
        assert np.isfinite(stats["best_max_abs_error"])
        assert stats["best_max_abs_error"] >= 0


class Test3DInterpolationFallbacks:
    """Test 3D interpolation mode selection."""

    def test_3d_cubic_uses_keys_cubic(self, simple_3d_volume) -> None:
        """Test that 3D with cubic uses Keys cubic convolution."""
        scales_list, stats = decompose_image(
            simple_3d_volume,
            scales=[1, 2],
            n_iters=30,
            interpolation="cubic",
            verbose=False,
        )

        assert len(scales_list) == 2
        # Verify interpolation mode stored
        assert stats["interpolation"] == "cubic"

        # Should achieve good quality with cubic
        assert stats["final_error"] < 0.1

    def test_3d_nearest_mode(self, simple_3d_volume) -> None:
        """Test 3D with nearest neighbor interpolation."""
        scales_list, stats = decompose_image(
            simple_3d_volume,
            scales=[1, 2],
            n_iters=50,
            interpolation="nearest",
            verbose=False,
        )

        assert len(scales_list) == 2
        assert stats["interpolation"] == "nearest"


class TestNormPercentile:
    """Test configurable normalization percentile."""

    def test_norm_percentile_zero(self, simple_2d_image) -> None:
        """Test full range normalization (percentile=0)."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=20,
            # norm_percentile=0.0 is default
            verbose=False,
        )

        assert len(scales_list) == 2

    def test_norm_percentile_robust(self) -> None:
        """Test robust percentile normalization with outliers."""
        image = np.random.rand(64, 64).astype(np.float32)
        image[0, 0] = 100.0  # Extreme outlier

        scales_list, stats = decompose_image(
            image,
            scales=[1, 2],
            n_iters=20,
            # Uses default norm_percentile in decompose_image
            verbose=False,
        )

        assert len(scales_list) == 2
        # Should handle outlier gracefully
        for img in scales_list:
            assert np.all(np.isfinite(img))


class TestCubicInterpolationEdgeCases:
    """Test cubic interpolation edge cases and fallbacks."""

    def test_cubic_upsample_fractional_scaling(self) -> None:
        """Test cubic upsampling with non-power-of-2 scale factors."""
        # Create small image and upsample to non-power-of-2 size
        img = np.random.rand(8, 8).astype(np.float32)
        # Target size that requires fractional upsampling (not exact power of 2)
        target_shape = (20, 20)

        upsampled = upsample_for_visualization(img, target_shape, interpolation="cubic")

        assert upsampled.shape == target_shape
        assert np.all(upsampled >= 0)
        assert np.all(np.isfinite(upsampled))


class TestInitializationMethodSelection:
    """Test that all initialization method branches are covered."""

    def test_init_method_pyramid_explicit(self, simple_2d_image) -> None:
        """Test explicitly selecting pyramid init method."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=20,
            init_method="pyramid",
            verbose=True,  # Triggers logging branch
        )

        assert len(scales_list) == 2

    def test_init_method_finest_explicit(self, simple_2d_image) -> None:
        """Test explicitly selecting finest init method."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=20,
            init_method="finest",
            verbose=True,
        )

        assert len(scales_list) == 2

    def test_init_method_uniform_explicit(self, simple_2d_image) -> None:
        """Test explicitly selecting uniform init method."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=20,
            init_method="uniform",
            verbose=True,
        )

        assert len(scales_list) == 2

    def test_init_method_zero_explicit(self, simple_2d_image) -> None:
        """Test explicitly selecting zero init method."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=20,
            init_method="zero",
            verbose=True,
        )

        assert len(scales_list) == 2


class TestMovieFrameFIFO:
    """Test movie frame FIFO buffer management."""

    def test_movie_fifo_buffer_overflow(self, simple_2d_image) -> None:
        """Test that old frames are discarded when max_frames exceeded."""
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=30,
            napari_movie=True,
            movie_every=1,
            movie_max_frames=5,  # Small limit to trigger FIFO
            max_abs_error_threshold=1e-10,  # Prevent early convergence
            verbose=False,
        )

        movie_frames = stats["movie_frames"]
        # Should not exceed max_frames
        assert len(movie_frames["iterations"]) <= 5
        # Should have kept most recent frames
        if len(movie_frames["iterations"]) == 5:
            # Most recent iterations should be 26-30
            assert movie_frames["iterations"][-1] == 30


class TestDeviceDetection:
    """Test device auto-detection branches."""

    @pytest.mark.skipif(
        torch.cuda.is_available(), reason="CUDA available, won't test fallback"
    )
    def test_device_detection_no_cuda(self, simple_2d_image) -> None:
        """Test device detection when CUDA not available."""
        # When CUDA not available, should fall back to MPS or CPU
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=10,
            device=None,  # Auto-detect
            verbose=False,
        )

        assert len(scales_list) == 2


class TestBestStateFallback:
    """Test best state fallback scenarios."""

    def test_best_state_fallback_when_no_best_saved(self, simple_2d_image) -> None:
        """Test fallback to final state when no best state was saved."""
        # This should be rare, but tests the fallback branch
        scales_list, stats = decompose_image(
            simple_2d_image,
            scales=[1, 2],
            n_iters=10,
            max_abs_error_threshold=1e-20,  # Impossible threshold
            verbose=False,
        )

        # Should still return valid results using final state
        assert len(scales_list) == 2
        assert np.isfinite(stats["best_max_abs_error"])


class TestLossComputationHelpers:
    """Test individual loss computation functions via decomposition_loss."""

    def test_decomposition_loss_l1_symmetric(self, simple_2d_image) -> None:
        """Test decomposition loss with L1 and no asymmetric penalty."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])
        model.initialize_from_pyramid(target)

        loss, stats = decomposition_loss(
            model,
            target,
            energy_weight=0.001,
            alpha=1.5,
            loss_type="l1",
            asymmetric_penalty=None,  # Symmetric
        )

        assert torch.isfinite(loss)
        assert "recon_loss" in stats
        assert "energy_loss" in stats

    def test_decomposition_loss_mse_symmetric(self, simple_2d_image) -> None:
        """Test decomposition loss with MSE and no asymmetric penalty."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])
        model.initialize_from_pyramid(target)

        loss, stats = decomposition_loss(
            model,
            target,
            energy_weight=0.001,
            alpha=1.5,
            loss_type="mse",
            asymmetric_penalty=None,
        )

        assert torch.isfinite(loss)

    def test_decomposition_loss_poisson_symmetric(self, simple_2d_image) -> None:
        """Test decomposition loss with Poisson and no asymmetric penalty."""
        target = torch.tensor(simple_2d_image, dtype=torch.float32)
        model = MultiScaleDecomposer(simple_2d_image.shape, scales=[1, 2])
        model.initialize_from_pyramid(target)

        loss, stats = decomposition_loss(
            model,
            target,
            energy_weight=0.001,
            alpha=1.5,
            loss_type="poisson",
            asymmetric_penalty=None,
        )

        assert torch.isfinite(loss)
