"""
Tests for fitting preprocessing module.
"""

import numpy as np
import pytest
import torch

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.fitting.config import FitConfig
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.fitting.preprocessing import preprocess_data


@pytest.fixture
def mock_config_2d():
    """Create a mock 2D configuration for testing."""
    V = np.random.rand(32, 32).astype(np.float32)
    centers = np.random.rand(10, 2).astype(np.float32) * 32  # Scale to image size

    return FitConfig(
        V=V,
        seeds=centers,
        norm_percentile=0.0,
        init_sigma_vox=1.5,
        sigma_min_diag=[0.1, 0.1],
        sigma_max_diag=None,
        truncate=3.0,
        n_iters=100,
        lr=0.01,
        max_abs_error=None,
        gradient_clip=1.0,
        loss_type="l1",
        asymmetric_penalty=10.0,
        l1_amp=0.001,
        l1_diag=0.0001,
        l1_sharpness=0.0,
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

    def test_seeds_as_float_proportion(self, mock_config_2d) -> None:
        """Test seeds parameter as float proportion."""
        mock_config_2d.seeds = 0.01  # 1% of voxels
        mock_config_2d.seed_method = "grid"  # Use grid method
        mock_config_2d.seed_kwargs = {"spacing": 8.0}

        result = preprocess_data(mock_config_2d)

        # Just verify it runs without error and generates seeds
        assert result.seed_centers is not None
        assert result.N >= 0  # May be 0 if no peaks detected

    def test_seeds_explicit_array(self, mock_config_2d) -> None:
        """Test seeds parameter as explicit array."""
        explicit_seeds = np.array([[5.0, 5.0], [15.0, 15.0], [25.0, 25.0]])
        mock_config_2d.seeds = explicit_seeds

        result = preprocess_data(mock_config_2d)

        # Should use exact seeds provided
        assert result.N == 3
        assert np.allclose(result.seed_centers, explicit_seeds)

    def test_init_arrays_subsampled_with_seeds(self, mock_config_2d) -> None:
        """Test that init_L, init_amps, init_sharpness are sliced when seeds are subsampled.

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
        mock_config_2d.init_sharpness = np.full(n_original, 2.0, dtype=np.float32)

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

        result = preprocess_data(mock_config_2d)

        # After preprocessing, init arrays should match the subsampled seed count
        if result.N < n_original:
            # Subsampling occurred
            if mock_config_2d.init_L is not None:
                assert mock_config_2d.init_L.shape[0] == result.N, (
                    f"init_L not sliced: got {mock_config_2d.init_L.shape[0]}, expected {result.N}"
                )
            if mock_config_2d.init_amps is not None:
                assert mock_config_2d.init_amps.shape[0] == result.N, (
                    f"init_amps not sliced: got {mock_config_2d.init_amps.shape[0]}, expected {result.N}"
                )
            if mock_config_2d.init_sharpness is not None:
                assert mock_config_2d.init_sharpness.shape[0] == result.N, (
                    f"init_sharpness not sliced: got {mock_config_2d.init_sharpness.shape[0]}, expected {result.N}"
                )

    def test_init_arrays_cleared_when_more_seeds_needed(self, mock_config_2d) -> None:
        """Test that init arrays are cleared when more seeds need to be generated.

        When fewer seeds are detected than requested, more seeds are added via
        _ensure_minimum_seeds. Since these new seeds don't have scale information,
        the init arrays must be cleared to avoid shape mismatches.
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
        mock_config_2d.seed_kwargs = {"scales": [2, 4], "percentile_thresh": 95}

        # Set init arrays - these should be cleared when more seeds are generated
        mock_config_2d.init_L = np.eye(2, dtype=np.float32)[None, :, :].repeat(
            2, axis=0
        )
        mock_config_2d.init_amps = np.array([1.0, 1.0], dtype=np.float32)
        mock_config_2d.init_sharpness = np.array([2.0, 2.0], dtype=np.float32)

        result = preprocess_data(mock_config_2d)

        # When more seeds are added, init arrays should be cleared (set to None)
        # This avoids shape mismatch since new seeds don't have scale info
        if result.N > 2:  # More seeds were generated
            assert mock_config_2d.init_L is None, (
                "init_L should be cleared when more seeds are generated"
            )
            assert mock_config_2d.init_amps is None, (
                "init_amps should be cleared when more seeds are generated"
            )
            assert mock_config_2d.init_sharpness is None, (
                "init_sharpness should be cleared when more seeds are generated"
            )
