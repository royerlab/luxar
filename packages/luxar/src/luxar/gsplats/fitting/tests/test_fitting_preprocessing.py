"""
Tests for fitting preprocessing module.
"""

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.fitting.config import FitConfig
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.fitting.preprocessing import (
        _compression_ratio_to_target_count,
        _compute_floats_per_splat,
        preprocess_data,
    )


@pytest.fixture
def mock_config_2d():
    """Create a mock 2D configuration for testing."""
    V = np.random.rand(32, 32).astype(np.float32)
    centers = np.random.rand(10, 2).astype(np.float32) * 32  # Scale to image size

    return FitConfig(
        V=V,
        seeds=centers,
        norm_percentile=0.0,
        floor="none",  # tests opt in to floor explicitly; default here = old behavior
        init_sigma_vox=1.5,
        sigma_min_diag=[0.1, 0.1],
        sigma_max_diag=None,
        truncate=3.0,
        n_iters=100,
        lr=0.01,
        max_abs_error=None,
        rel_l2_target=None,
        gradient_clip=1.0,
        loss_type="l1",
        asymmetric_penalty=10.0,
        l1_amp=0.001,
        l1_diag=0.0001,
        scheduler_type="plateau",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,
        enable_dynamic_ops=True,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=100,
        device=torch.device("cpu"),
        verbose=False,
        seed_method="auto",  # Use new default
        seed_kwargs={},
    )


class TestPreprocessDataValidation:
    """Test input data validation in preprocessing."""

    def test_nan_input_raises_error(self, mock_config_2d) -> None:
        """Test that NaN in input volume raises ValueError."""
        mock_config_2d.V[5, 5] = np.nan
        with pytest.raises(ValueError, match="NaN value"):
            preprocess_data(mock_config_2d)

    def test_inf_input_raises_error(self, mock_config_2d) -> None:
        """Test that Inf in input volume raises ValueError."""
        mock_config_2d.V[5, 5] = np.inf
        with pytest.raises(ValueError, match="Inf value"):
            preprocess_data(mock_config_2d)

    def test_negative_inf_input_raises_error(self, mock_config_2d) -> None:
        """Test that -Inf in input volume raises ValueError."""
        mock_config_2d.V[5, 5] = -np.inf
        with pytest.raises(ValueError, match="Inf value"):
            preprocess_data(mock_config_2d)


class TestPreprocessData:
    """Test data preprocessing functionality."""

    def test_basic_preprocessing(self, mock_config_2d) -> None:
        """Test basic data preprocessing."""
        result = preprocess_data(mock_config_2d)

        assert result.V_normalized.shape == mock_config_2d.V.shape
        assert result.seed_centers.shape == (10, 2)
        assert result.d == 2
        assert result.N == 10
        assert result.intensity_range > 0
        assert result.max_abs_error > 0

    def test_normalization_full_range(self, mock_config_2d) -> None:
        """Test full range normalization."""
        # Create data with known range
        V = np.array([[0.1, 0.5], [0.3, 0.9]], dtype=np.float32)
        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0

        result = preprocess_data(mock_config_2d)

        assert result.image_min == pytest.approx(0.1, abs=1e-6)
        assert result.image_max == pytest.approx(0.9, abs=1e-6)
        assert result.intensity_range == pytest.approx(0.8, abs=1e-6)
        # Check normalization
        assert np.min(result.V_normalized) >= 0.0
        assert np.max(result.V_normalized) <= 1.0

    def test_normalization_percentile(self, mock_config_2d) -> None:
        """Test percentile-based normalization."""
        # Create data with outliers
        V = np.array([[0.0, 0.5], [0.3, 10.0]], dtype=np.float32)  # 10.0 is outlier
        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 25.0  # Should ignore extremes

        result = preprocess_data(mock_config_2d)

        # Should not use the extreme values
        assert result.image_min != 0.0
        assert result.image_max != 10.0

    def test_floor_auto_subtracts_pedestal(self, mock_config_2d) -> None:
        """`floor='auto'` maps a background pedestal to 0 and keeps the peak."""
        rng = np.random.default_rng(0)
        V = np.full((40, 40), 110.0, np.float32)
        V += rng.normal(0, 1.0, V.shape).astype(np.float32)
        V[18:22, 18:22] += 400.0  # bright signal blob
        mock_config_2d.V = V
        mock_config_2d.floor = "auto"

        result = preprocess_data(mock_config_2d)

        # image_min lands on the pedestal (~110), not the hard min.
        assert result.image_min == pytest.approx(110.0, abs=3.0)
        assert result.floor == pytest.approx(result.image_min, abs=1e-6)
        # Most of the background maps to 0; the peak is preserved at ~1.
        assert float((result.V_normalized < 1e-6).mean()) > 0.3
        assert result.V_normalized.max() == pytest.approx(1.0, abs=1e-6)

    def test_floor_none_reproduces_hard_min(self, mock_config_2d) -> None:
        """`floor='none'` reproduces the historical hard-min normalization."""
        V = np.array([[0.1, 0.5], [0.3, 0.9]], dtype=np.float32)
        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = "none"

        result = preprocess_data(mock_config_2d)

        assert result.image_min == pytest.approx(0.1, abs=1e-6)
        assert result.floor is None

    def test_floor_fixed_value(self, mock_config_2d) -> None:
        """A fixed float floor sets image_min directly."""
        V = np.linspace(0.0, 1.0, 400, dtype=np.float32).reshape(20, 20)
        mock_config_2d.V = V
        mock_config_2d.floor = 0.25

        result = preprocess_data(mock_config_2d)

        assert result.image_min == pytest.approx(0.25, abs=1e-6)
        assert result.floor == pytest.approx(0.25, abs=1e-6)

    def test_floor_above_max_is_ignored(self, mock_config_2d) -> None:
        """A floor >= image max would erase all signal — it is refused, not
        applied (no degenerate uniform-0.5 volume)."""
        V = np.linspace(0.0, 100.0, 400, dtype=np.float32).reshape(20, 20)
        mock_config_2d.V = V
        mock_config_2d.floor = 500.0  # above the data max

        result = preprocess_data(mock_config_2d)

        assert result.floor is None  # floor not applied
        assert result.image_min == pytest.approx(0.0, abs=1e-6)  # default hard-min
        # Real dynamic range preserved (not collapsed to a constant 0.5).
        assert result.V_normalized.max() == pytest.approx(1.0, abs=1e-6)
        assert result.V_normalized.min() == pytest.approx(0.0, abs=1e-6)

    def test_uniform_image_handling(self, mock_config_2d) -> None:
        """Test handling of nearly uniform images."""
        # Create nearly uniform image
        V = np.full((16, 16), 0.5, dtype=np.float32)
        mock_config_2d.V = V

        result = preprocess_data(mock_config_2d)

        assert result.intensity_range == 1.0  # Should be set to avoid division by zero
        assert np.allclose(result.V_normalized, 0.5)

    def test_auto_candidate_generation(self, mock_config_2d) -> None:
        """Test automatic candidate generation."""
        mock_config_2d.seed_centers = None  # Trigger auto-generation

        result = preprocess_data(mock_config_2d)

        assert result.seed_centers is not None
        assert result.seed_centers.shape[1] == 2  # 2D centers
        assert result.N > 0

    def test_convergence_threshold_auto(self, mock_config_2d) -> None:
        """Test automatic convergence threshold setting."""
        mock_config_2d.max_abs_error = None

        result = preprocess_data(mock_config_2d)

        assert result.max_abs_error == 0.01  # 1% of normalized range

    def test_convergence_threshold_custom(self, mock_config_2d) -> None:
        """Test custom convergence threshold."""
        mock_config_2d.max_abs_error = 0.005

        result = preprocess_data(mock_config_2d)

        assert result.max_abs_error == 0.005

    def test_tensor_conversion(self, mock_config_2d) -> None:
        """Test conversion to PyTorch tensor."""
        result = preprocess_data(mock_config_2d)

        assert isinstance(result.V_tensor, torch.Tensor)
        assert result.V_tensor.device == mock_config_2d.device
        assert result.V_tensor.dtype == torch.float32
        assert result.V_tensor.shape == mock_config_2d.V.shape

    def test_seeds_as_int_count(self, mock_config_2d) -> None:
        """Test seeds parameter as integer count."""
        # Create volume with known peaks
        V = np.zeros((40, 40), dtype=np.float32)
        # Add Gaussian peaks
        for cx, cy in [(10, 10), (20, 20), (30, 30)]:
            x, y = np.meshgrid(np.arange(40) - cx, np.arange(40) - cy, indexing="ij")
            r2 = x**2 + y**2
            V += 100 * np.exp(-r2 / (2 * 3**2))
        V += np.random.rand(40, 40) * 5  # Add noise

        mock_config_2d.V = V
        mock_config_2d.seeds = 5  # Request exactly 5 seeds
        mock_config_2d.seed_method = "grid"  # Use grid method
        mock_config_2d.seed_kwargs = {"spacing": 8.0}

        result = preprocess_data(mock_config_2d)

        # Should get exactly 5 seeds (or fewer if less were detected)
        assert result.N <= 5
        assert result.seed_centers.shape[1] == 2

    def test_seeds_as_int_subsample(self, mock_config_2d) -> None:
        """Test that int seeds correctly subsamples when more are detected."""
        # Create volume with many peaks
        V = np.zeros((50, 50), dtype=np.float32)
        # Add many Gaussian peaks
        centers = [(i * 10 + 5, j * 10 + 5) for i in range(4) for j in range(4)]
        for cx, cy in centers:
            x, y = np.meshgrid(np.arange(50) - cx, np.arange(50) - cy, indexing="ij")
            r2 = x**2 + y**2
            V += 50 * np.exp(-r2 / (2 * 2**2))

        mock_config_2d.V = V
        mock_config_2d.seeds = 8  # Request 8 seeds (should subsample from many)
        mock_config_2d.seed_method = "grid"  # Use grid method
        mock_config_2d.seed_kwargs = {"spacing": 5.0}  # Dense grid

        result = preprocess_data(mock_config_2d)

        # Should get at most 8 seeds
        assert result.N <= 8
        # If subsampling occurred, should be exactly 8
        # (unless fewer than 8 were detected, but with this setup we should get more)
        if result.N < 8:
            # This is the case where fewer seeds were detected
            pass
        else:
            assert result.N == 8

    def test_seeds_as_compression_ratio(self, mock_config_2d) -> None:
        """Test seeds parameter as compression ratio."""
        # Create a 64x64 image (4096 voxels)
        V = np.random.rand(64, 64).astype(np.float32) * 0.5
        # Add some structure
        for cx, cy in [(16, 16), (32, 32), (48, 48)]:
            x, y = np.meshgrid(np.arange(64) - cx, np.arange(64) - cy, indexing="ij")
            V += np.exp(-(x**2 + y**2) / (2 * 5**2))

        mock_config_2d.V = V
        mock_config_2d.seeds = 0.1  # 10% compression ratio
        mock_config_2d.seed_method = "grid"
        mock_config_2d.seed_kwargs = {"spacing": 8.0}

        result = preprocess_data(mock_config_2d)

        # For 64x64 = 4096 voxels, 2D (7 floats/splat)
        # Expected target: 0.1 * 4096 / 7 = 58.5 → 58 seeds
        expected_target = int(0.1 * 4096 / 7)

        # Verify we get approximately the expected number of seeds
        # Allow some tolerance since seed generation may not hit exact target
        assert result.seed_centers is not None
        assert result.N > 0
        # The actual count should be within reasonable range of expected
        assert abs(result.N - expected_target) <= max(10, expected_target * 0.5), (
            f"Expected ~{expected_target} seeds, got {result.N}"
        )

    def test_seeds_explicit_array(self, mock_config_2d) -> None:
        """Test seeds parameter as explicit array."""
        explicit_seeds = np.array([[5.0, 5.0], [15.0, 15.0], [25.0, 25.0]])
        mock_config_2d.seeds = explicit_seeds

        result = preprocess_data(mock_config_2d)

        # Should use exact seeds provided
        assert result.N == 3
        assert np.allclose(result.seed_centers, explicit_seeds)

    def test_init_arrays_subsampled_with_seeds(self, mock_config_2d) -> None:
        """Test that init_L, init_amps are sliced when seeds are subsampled.

        This guards against a regression where subsampling seeds would leave the
        init arrays at their original size, causing shape mismatches during model init.
        """
        # Setup: create more seeds than target with pre-initialized arrays
        n_original = 20
        target_count = 5
        d = 2

        # Create init arrays for all original seeds
        mock_config_2d.init_L = np.eye(d, dtype=np.float32)[None, :, :].repeat(
            n_original, axis=0
        )
        mock_config_2d.init_L *= np.random.uniform(0.5, 2.0, size=(n_original, 1, 1))
        mock_config_2d.init_amps = np.random.rand(n_original).astype(np.float32)

        # Create many explicit seeds that will be subsampled
        V = np.zeros((50, 50), dtype=np.float32)
        centers = []
        for i in range(n_original):
            cx = 5 + (i % 5) * 10
            cy = 5 + (i // 5) * 10
            centers.append([cx, cy])
            x, y = np.meshgrid(np.arange(50) - cx, np.arange(50) - cy, indexing="ij")
            V += 50 * np.exp(-(x**2 + y**2) / (2 * 2**2))

        mock_config_2d.V = V
        mock_config_2d.seeds = target_count  # Will trigger subsampling
        mock_config_2d.seed_method = "grid"  # Use grid method
        mock_config_2d.seed_kwargs = {"spacing": 5.0}  # Dense grid

        # Store original config values to verify they aren't mutated
        original_init_L_shape = mock_config_2d.init_L.shape[0]

        result = preprocess_data(mock_config_2d)

        # Verify config is NOT mutated (new behavior)
        assert mock_config_2d.init_L.shape[0] == original_init_L_shape, (
            "Config should not be mutated by preprocess_data"
        )

        # After preprocessing, init arrays in PreprocessedData should match seed count
        if result.N < n_original:
            # Subsampling occurred - check PreprocessedData has correctly sliced arrays
            if result.init_L is not None:
                assert result.init_L.shape[0] == result.N, (
                    f"result.init_L not sliced: got {result.init_L.shape[0]}, expected {result.N}"
                )
            if result.init_amps is not None:
                assert result.init_amps.shape[0] == result.N, (
                    f"result.init_amps not sliced: got {result.init_amps.shape[0]}, expected {result.N}"
                )

    def test_init_arrays_extended_when_more_seeds_needed(self, mock_config_2d) -> None:
        """Test that init arrays are extended when more seeds need to be generated.

        When fewer seeds are detected than requested, more seeds are added via
        _ensure_minimum_seeds. The init arrays should be extended to include
        the new grid fallback seeds while preserving the original seeds' values.
        """
        # Setup: create sparse volume with few detectable peaks
        V = np.zeros((50, 50), dtype=np.float32)
        # Add only 2 clear peaks
        for cx, cy in [(15, 15), (35, 35)]:
            x, y = np.meshgrid(np.arange(50) - cx, np.arange(50) - cy, indexing="ij")
            V += 80 * np.exp(-(x**2 + y**2) / (2 * 3**2))

        # Add small amount of noise
        V += np.random.rand(50, 50).astype(np.float32) * 5

        mock_config_2d.V = V
        mock_config_2d.seeds = 10  # Request more seeds than detectable peaks
        mock_config_2d.seed_method = "decomposition"  # Use decomposition method
        # Force CPU to avoid CUDA availability issues in tests
        mock_config_2d.seed_kwargs = {
            "scales": [2, 4],
            "percentile_thresh": 95,
            "device": "cpu",
        }

        # Set init arrays - these should be preserved and extended
        mock_config_2d.init_L = np.eye(2, dtype=np.float32)[None, :, :].repeat(
            2, axis=0
        )
        mock_config_2d.init_amps = np.array([1.0, 1.0], dtype=np.float32)

        # Store original config values to verify they aren't mutated
        original_init_L_shape = mock_config_2d.init_L.shape[0]

        result = preprocess_data(mock_config_2d)

        # Verify config is NOT mutated (new behavior)
        assert mock_config_2d.init_L.shape[0] == original_init_L_shape, (
            "Config should not be mutated by preprocess_data"
        )

        # When more seeds are added, init arrays in PreprocessedData should be
        # EXTENDED (not cleared) to include values for the new grid fallback seeds
        if result.N > 2:  # More seeds were generated
            assert result.init_L is not None, (
                "result.init_L should be extended when more seeds are generated"
            )
            assert result.init_L.shape[0] == result.N, (
                f"result.init_L should have {result.N} entries, got {result.init_L.shape[0]}"
            )
            assert result.init_amps is not None, (
                "result.init_amps should be extended when more seeds are generated"
            )
            assert len(result.init_amps) == result.N, (
                f"result.init_amps should have {result.N} entries, got {len(result.init_amps)}"
            )


class TestCompressionRatio:
    """Tests for compression ratio helper functions."""

    def test_floats_per_splat_2d(self) -> None:
        """Test floats per splat calculation for 2D."""
        # 2D: center (2) + cholesky (2*3/2=3) + amp (1) = 6
        assert _compute_floats_per_splat(2) == 6

    def test_floats_per_splat_3d(self) -> None:
        """Test floats per splat calculation for 3D."""
        # 3D: center (3) + cholesky (3*4/2=6) + amp (1) = 10
        assert _compute_floats_per_splat(3) == 10

    def test_floats_per_splat_4d(self) -> None:
        """Test floats per splat calculation for 4D."""
        # 4D: center (4) + cholesky (4*5/2=10) + amp (1) = 15
        assert _compute_floats_per_splat(4) == 15

    def test_floats_per_splat_5d(self) -> None:
        """Test floats per splat calculation for 5D."""
        # 5D: center (5) + cholesky (5*6/2=15) + amp (1) = 21
        assert _compute_floats_per_splat(5) == 21

    def test_compression_ratio_to_target_count_2d(self) -> None:
        """Test compression ratio calculation for 2D."""
        # 2D: 6 floats per splat
        # 100x100 image = 10,000 voxels
        # ratio=0.1 → 0.1 * 10000 / 6 = 166.6 → 166
        target = _compression_ratio_to_target_count(0.1, (100, 100))
        assert target == 166

    def test_compression_ratio_to_target_count_3d(self) -> None:
        """Test compression ratio calculation for 3D."""
        # 3D: 10 floats per splat
        # 64³ = 262,144 voxels
        # ratio=0.05 → 0.05 * 262144 / 10 = 1310.72 → 1310
        target = _compression_ratio_to_target_count(0.05, (64, 64, 64))
        assert target == 1310

    def test_compression_ratio_to_target_count_3d_high_ratio(self) -> None:
        """Test compression ratio calculation for 3D with high ratio."""
        # 3D: 10 floats per splat
        # 64³ = 262,144 voxels
        # ratio=0.1 → 0.1 * 262144 / 10 = 2621.44 → 2621
        target = _compression_ratio_to_target_count(0.1, (64, 64, 64))
        assert target == 2621

    def test_compression_ratio_minimum_one_seed(self) -> None:
        """Test that compression ratio returns at least 1 seed."""
        # Very small ratio on small image should clamp to 1
        # 4x4 = 16 voxels, ratio=0.001, 2D (6 floats)
        # 0.001 * 16 / 6 = 0.0027 → 0, but should clamp to 1
        target = _compression_ratio_to_target_count(0.001, (4, 4))
        assert target == 1

    def test_compression_ratio_large_volume(self) -> None:
        """Test compression ratio calculation for large 3D volume."""
        # 256³ = 16,777,216 voxels
        # ratio=0.1 → 0.1 * 16777216 / 10 = 167,772
        target = _compression_ratio_to_target_count(0.1, (256, 256, 256))
        assert target == 167772

    def test_compression_ratio_4d(self) -> None:
        """Test compression ratio calculation for 4D."""
        # 4D: 15 floats per splat
        # 32^4 = 1,048,576 voxels
        # ratio=0.05 → 0.05 * 1048576 / 15 = 3495.25 → 3495
        target = _compression_ratio_to_target_count(0.05, (32, 32, 32, 32))
        assert target == 3495
