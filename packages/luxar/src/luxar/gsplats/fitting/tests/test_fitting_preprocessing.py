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
        _normalize_data,
        _resolve_floor,
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

    def test_floor_guard_uses_true_max_with_percentile_normalization(
        self, mock_config_2d, capsys
    ) -> None:
        V = np.concatenate(
            [
                np.linspace(0.0, 10.0, 99, dtype=np.float32),
                np.array([100.0], dtype=np.float32),
            ]
        ).reshape(10, 10)
        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 10.0
        mock_config_2d.floor = 20.0
        mock_config_2d.verbose = True

        result = preprocess_data(mock_config_2d)

        assert result.floor == pytest.approx(20.0)
        assert result.image_min == pytest.approx(20.0)
        assert result.image_max == pytest.approx(100.0)
        assert result.V_normalized.max() == pytest.approx(1.0)
        assert "expanding high endpoint to data max 100" in capsys.readouterr().out

    def test_supplied_norm_range_remains_authoritative_for_floor_guard(self) -> None:
        V = np.array([0.0, 5.0], dtype=np.float32)

        normalized, image_min, image_max, intensity_range, applied_floor = (
            _normalize_data(
                V,
                norm_percentile=0.0,
                verbose=False,
                floor=8.0,
                norm_range=(0.0, 10.0),
            )
        )

        assert image_min == pytest.approx(8.0)
        assert image_max == pytest.approx(10.0)
        assert intensity_range == pytest.approx(2.0)
        assert applied_floor == pytest.approx(8.0)
        assert normalized.tolist() == pytest.approx([0.0, 0.0])

    def test_percentile_floor_excludes_exact_zero_padding(self) -> None:
        V = np.concatenate(
            [np.zeros(100, dtype=np.float32), np.linspace(100.0, 120.0, 100)]
        )

        assert _resolve_floor(V, "p10") == pytest.approx(102.0, abs=0.1)

    def test_percentile_floor_materializes_array_like_input(self) -> None:
        class ArrayLike:
            def __init__(self, values: np.ndarray) -> None:
                self.values = values

            def __array__(self, dtype=None) -> np.ndarray:
                return np.asarray(self.values, dtype=dtype)

        V = ArrayLike(np.array([0.0, 10.0, 20.0], dtype=np.float32))

        assert _resolve_floor(V, "p10") == pytest.approx(11.0)

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


def _pedestal_volume() -> np.ndarray:
    """A 40x40 volume with a constant background pedestal of 100 plus a blob."""
    V = np.full((40, 40), 100.0, dtype=np.float32)
    V[10:30, 10:30] += 400.0
    return V


def _gsplatdata_seeds(centers: np.ndarray, amps: np.ndarray):
    """Build a bare GSplatData the way both seed producers build one.

    ``generate_seeds()`` (raw amplitudes) and ``finalize_results`` (background-
    relative amplitudes) both emit exactly this — centers + Cholesky +
    amplitudes, no stats, no provenance. That is precisely why the amplitude
    convention has to be declared by the caller.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.utils.trils import pack_tril

    n, d = centers.shape
    L = np.zeros((n, d, d), dtype=np.float32)
    for i in range(d):
        L[:, i, i] = 1.5
    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=amps.astype(np.float32),
        cholesky_factors=pack_tril(L),
    )


class TestInitAmpsProvenance:
    """Amplitude-scale bookkeeping for pre-initialized seeds (#1172).

    Two conventions reach ``preprocess_data``: raw-image-sampled amplitudes
    (what the seeding methods produce) and background-relative ones (what a
    previous fit returns). They differ by exactly the background floor, so the
    rescaling into the optimizer's [0, 1] scale must differ too — and since the
    ``seeds=GSplatData`` door carries both, the caller declares which via
    ``FitConfig.seed_amps_background_relative``.
    """

    def test_gsplatdata_warm_start_keeps_subfloor_amplitudes(
        self, mock_config_2d
    ) -> None:
        """A declared background-relative warm start must NOT have the floor
        subtracted a second time.

        Before the fix, ``(a - image_min) / intensity_range`` clipped every
        sub-floor amplitude to exactly 0, silently erasing the warm start's dim
        splats.
        """
        V = _pedestal_volume()
        seed_amps = np.array([1.0, 5.0, 30.0], dtype=np.float32)  # all < floor 100
        centers = np.array([[12.0, 12.0], [20.0, 20.0], [26.0, 26.0]], np.float32)

        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = 100.0  # active floor == image_min
        mock_config_2d.seeds = _gsplatdata_seeds(centers, seed_amps)
        mock_config_2d.seed_amps_background_relative = True  # a fit's output

        result = preprocess_data(mock_config_2d)

        assert result.floor == pytest.approx(100.0, abs=1e-6)
        assert result.image_min == pytest.approx(100.0, abs=1e-6)
        assert result.intensity_range == pytest.approx(400.0, abs=1e-6)
        assert result.init_amps is not None
        # No second subtraction: pure division by the intensity range.
        expected = np.clip(seed_amps / result.intensity_range, 0.0, 1.0)
        assert np.allclose(result.init_amps, expected, atol=1e-6)
        # And specifically: nothing was zeroed.
        assert float(result.init_amps.min()) > 0.0

    def test_gsplatdata_warm_start_with_auto_floor(self, mock_config_2d) -> None:
        """Same guarantee under ``floor='auto'`` on a volume with a real pedestal
        (the setting `volume_refit` inherits from ``fit_gaussian_splats``)."""
        rng = np.random.default_rng(1)
        V = np.full((40, 40), 100.0, np.float32)
        V += rng.normal(0, 1.0, V.shape).astype(np.float32)
        V[16:24, 16:24] += 400.0
        seed_amps = np.array([2.0, 8.0, 40.0], dtype=np.float32)
        centers = np.array([[17.0, 17.0], [20.0, 20.0], [22.0, 22.0]], np.float32)

        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = "auto"
        mock_config_2d.seeds = _gsplatdata_seeds(centers, seed_amps)
        mock_config_2d.seed_amps_background_relative = True

        result = preprocess_data(mock_config_2d)

        assert result.floor is not None  # the floor really is active
        assert result.floor == pytest.approx(100.0, abs=3.0)
        assert result.init_amps is not None
        expected = np.clip(seed_amps / result.intensity_range, 0.0, 1.0)
        assert np.allclose(result.init_amps, expected, atol=1e-6)
        assert float(result.init_amps.min()) > 0.0

    def test_gsplatdata_seeds_default_to_raw_sampled(self, mock_config_2d) -> None:
        """Negative control for the OTHER user of the same door: a GSplatData
        straight out of ``generate_seeds()`` carries RAW amplitudes, so without
        the declaration the ``- image_min`` term must still apply.

        This is the documented explicit-seeding workflow
        (``fit_gaussian_splats(V, seeds=generate_seeds(V))`` — see
        ``seeds/generate.py`` and ``demo_splats_mitosis_explicit_seeding``), and
        a bare GSplatData records no provenance, so treating every GSplatData as
        a fit's output would start these seeds too bright by
        ``floor / intensity_range``.
        """
        V = _pedestal_volume()
        centers = np.array([[12.0, 12.0], [20.0, 20.0], [26.0, 26.0]], np.float32)
        # Sampled off the raw volume: pedestal level, and two blob-level values.
        raw_amps = np.array([100.0, 300.0, 500.0], dtype=np.float32)

        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = 100.0
        mock_config_2d.seeds = _gsplatdata_seeds(centers, raw_amps)
        # No declaration -> the default, raw-image-sampled.
        assert mock_config_2d.seed_amps_background_relative is False

        result = preprocess_data(mock_config_2d)

        assert result.init_amps is not None
        expected = np.clip(
            (raw_amps - result.image_min) / result.intensity_range, 0.0, 1.0
        )
        assert np.allclose(result.init_amps, expected, atol=1e-6)
        # The pedestal-level seed rescales to exactly 0 under this convention
        # (it would be 0.25 if the floor were skipped).
        assert float(result.init_amps[0]) == pytest.approx(0.0, abs=1e-6)

    def test_config_init_amps_stay_raw_sampled(self, mock_config_2d) -> None:
        """Negative control: ``config.init_amps`` keeps the raw-image-sampled
        convention, so ``(a - image_min) / intensity_range`` still applies.

        This is the path the seeding methods' scale is expressed in; the fix must
        not flip it.
        """
        V = _pedestal_volume()
        centers = np.array([[12.0, 12.0], [20.0, 20.0], [26.0, 26.0]], np.float32)
        raw_amps = np.array([100.0, 300.0, 500.0], dtype=np.float32)

        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = 100.0
        mock_config_2d.seeds = centers  # explicit centers, not a GSplatData
        mock_config_2d.init_amps = raw_amps

        result = preprocess_data(mock_config_2d)

        assert result.init_amps is not None
        expected = np.clip(
            (raw_amps - result.image_min) / result.intensity_range, 0.0, 1.0
        )
        assert np.allclose(result.init_amps, expected, atol=1e-6)
        # The pedestal-level amplitude maps to 0 under this convention.
        assert float(result.init_amps[0]) == pytest.approx(0.0, abs=1e-6)

    def test_generated_seeds_stay_raw_sampled(self, mock_config_2d) -> None:
        """Negative control for the seeding path: amplitudes sampled by
        ``generate_seeds`` off the still-raw volume keep the ``- image_min`` term.

        Background seeds sit at the pedestal, so under the raw convention they
        rescale to 0; under the background-relative one they would come out
        clearly positive (~0.22 here). Asserting the zeros pins the convention.
        """
        V = _pedestal_volume()
        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = 100.0
        mock_config_2d.seeds = None
        mock_config_2d.seed_method = "grid"
        mock_config_2d.seed_kwargs = {"spacing": 6.0, "device": "cpu"}

        result = preprocess_data(mock_config_2d)

        assert result.init_amps is not None
        assert result.init_amps.shape[0] == result.N
        # Some seeds land on the background pedestal -> exactly 0 after rescaling.
        assert float(result.init_amps.min()) == pytest.approx(0.0, abs=1e-6)
        # ...and some land on the blob, so the array is not all zeros.
        assert float(result.init_amps.max()) > 0.5

    def test_warm_start_round_trip_through_a_real_fit(self, mock_config_2d) -> None:
        """End-to-end: fit a tiny volume with an active floor, then feed the
        RESULT back as ``seeds=`` and check the amplitudes that reach the
        optimizer.

        What this proves: amplitudes produced by ``finalize_results``, DECLARED
        background-relative the way ``volume_refit`` declares them, survive
        re-preprocessing as ``seed_amps / intensity_range`` — i.e. the warm start
        is not zeroed. What it does NOT prove: bit-exact round-tripping. The
        re-fit resolves its own ``intensity_range``, which need not equal the
        one the seed was fitted under (here both fits see the same volume, so
        they do).
        """
        from luxar.gsplats.fit_gsplats import fit_gaussian_splats

        V = _pedestal_volume()
        fitted = fit_gaussian_splats(
            V,
            seeds=8,
            floor=100.0,
            n_iters=60,
            device="cpu",
            verbose=False,
            seed_method="grid",
            enable_dynamic_ops=False,
            cull_retention=None,
            sort_splats_enabled=False,
            spacing=8.0,
        )
        seed_amps = fitted.amplitudes.copy()
        assert seed_amps.size > 0

        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = 100.0
        mock_config_2d.seeds = fitted
        mock_config_2d.seed_amps_background_relative = True

        result = preprocess_data(mock_config_2d)

        assert result.init_amps is not None
        expected = np.clip(seed_amps / result.intensity_range, 0.0, 1.0)
        assert np.allclose(result.init_amps, expected, atol=1e-6)
        # Non-vacuous: the fit really does leave amplitudes below the floor.
        sub_floor = seed_amps < result.image_min
        assert int(sub_floor.sum()) > 0
        # Every positive warm-start amplitude stays positive — including the
        # sub-floor ones, which the double subtraction collapsed to exactly 0.
        positive = seed_amps > 0.0
        assert np.all(result.init_amps[positive] > 0.0), (
            f"{int(np.sum(result.init_amps[positive] == 0.0))} of "
            f"{int(positive.sum())} positive warm-start amplitudes were zeroed"
        )

    def test_public_api_threads_the_declaration_into_fitconfig(self) -> None:
        """The declaration must survive the WHOLE plumbing chain —
        ``fit_gaussian_splats`` -> ``GaussianSplatFitter.fit`` ->
        ``prepare_fit_config`` -> ``FitConfig`` -> ``preprocess_data`` — not just
        the last hop.

        Every other test in this class either sets the flag on a ``FitConfig``
        instance directly or intercepts ``fit_gaussian_splats`` itself, so all of
        them stay green if an intermediate hop silently drops the kwarg (which
        would put #1172 fully back for ``lod --refine volume``). This one drives
        the real public entry point with a single iteration and a near-zero
        learning rate, so the fit is effectively frozen at its initialization and
        the output amplitudes must come back as the seed amplitudes (through
        softplus/inverse-softplus and an ``intensity_range`` round trip, hence a
        tolerance rather than exact equality).
        """
        from luxar.gsplats.fit_gsplats import fit_gaussian_splats

        V = _pedestal_volume()
        seed_amps = np.array([1.0, 5.0, 30.0], dtype=np.float32)  # all < floor 100
        centers = np.array([[12.0, 12.0], [20.0, 20.0], [26.0, 26.0]], np.float32)

        declared = fit_gaussian_splats(
            V,
            seeds=_gsplatdata_seeds(centers, seed_amps),
            seed_amps_background_relative=True,
            floor=100.0,
            n_iters=1,
            lr=1e-8,  # effectively frozen: the output IS the initialization
            device="cpu",
            verbose=False,
            enable_dynamic_ops=False,
            cull_retention=None,
            sort_splats_enabled=False,
        )

        np.testing.assert_allclose(declared.amplitudes, seed_amps, rtol=1e-6)

        # Negative control, with the argument OMITTED so the PUBLIC signature's
        # default is pinned too — the complement of
        # test_gsplatdata_seeds_default_to_raw_sampled, which pins the FitConfig
        # dataclass default: the amplitudes are taken as raw-image-sampled,
        # the pedestal is subtracted from already-background-relative values and
        # all three sub-floor seeds collapse toward 0 (~4e-4 as measured).
        default = fit_gaussian_splats(
            V,
            seeds=_gsplatdata_seeds(centers, seed_amps),
            floor=100.0,
            n_iters=1,
            lr=1e-8,
            device="cpu",
            verbose=False,
            enable_dynamic_ops=False,
            cull_retention=None,
            sort_splats_enabled=False,
        )

        assert np.all(default.amplitudes < 0.01), (
            f"expected the default convention to collapse sub-floor seeds, "
            f"got {default.amplitudes}"
        )

    def test_verbose_amplitude_report_handles_empty_seeds(
        self, mock_config_2d, capsys
    ) -> None:
        """The ``verbose`` rescaling report must run on both of its branches.

        Every other config in this module uses ``verbose=False``, so neither is
        otherwise exercised. Without the ``init_amps.size == 0`` guard the report
        raises ``ValueError: zero-size array to reduction operation minimum which
        has no identity`` on an empty seed set — and nothing upstream rejects one
        (``prepare_fit_config`` only shape-checks a NON-empty ``GSplatData``).
        """
        V = _pedestal_volume()
        centers = np.array([[12.0, 12.0], [20.0, 20.0], [26.0, 26.0]], np.float32)
        seed_amps = np.array([1.0, 5.0, 30.0], dtype=np.float32)

        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0
        mock_config_2d.floor = 100.0
        mock_config_2d.verbose = True
        mock_config_2d.seeds = _gsplatdata_seeds(centers, seed_amps)
        mock_config_2d.seed_amps_background_relative = True

        result = preprocess_data(mock_config_2d)

        assert result.init_amps is not None
        expected = np.clip(seed_amps / result.intensity_range, 0.0, 1.0)
        assert np.allclose(result.init_amps, expected, atol=1e-6)
        reported = capsys.readouterr().out
        assert "Rescaled init_amps to normalized range" in reported
        assert f"{float(expected.max()):.4f}" in reported

        # Empty seed set: the guard reports a count instead of reducing over an
        # empty array.
        mock_config_2d.seeds = _gsplatdata_seeds(
            np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        )

        empty_result = preprocess_data(mock_config_2d)

        assert empty_result.N == 0
        assert empty_result.init_amps is not None
        assert empty_result.init_amps.size == 0
        assert "Rescaled init_amps to normalized range: 0 seeds" in (
            capsys.readouterr().out
        )

    def test_volume_refit_declares_background_relative_seeds(self) -> None:
        """``volume_refine_splats`` re-fits a previous fit's output, so it must
        declare ``seed_amps_background_relative=True``.

        It is the only production caller that passes a ``GSplatData`` as
        ``seeds=``, and it passes no ``floor``, so it inherits the ``"auto"``
        default — exactly the combination that double-subtracts the pedestal.
        Lives with the convention tests rather than in ``lod/tests`` because what
        is being pinned is this module's contract, not the re-fit engine's.
        """
        import luxar.gsplats.fit_gsplats as fit_gsplats_mod
        from luxar.gsplats.lod.volume_refit import (
            VolumeRefitConfig,
            volume_refine_splats,
        )

        recorded: dict[str, object] = {}

        class _Stop(RuntimeError):
            pass

        def _spy(*args, **kwargs):
            recorded.update(kwargs)
            raise _Stop

        V = _pedestal_volume()
        # Centers inside the volume's voxel range so the frame guard passes and
        # the re-fit is actually attempted.
        seed = _gsplatdata_seeds(
            np.array([[12.0, 12.0], [20.0, 20.0], [26.0, 26.0]], np.float32),
            np.array([1.0, 5.0, 30.0], np.float32),
        )

        original = fit_gsplats_mod.fit_gaussian_splats
        fit_gsplats_mod.fit_gaussian_splats = _spy
        try:
            with pytest.raises(_Stop):
                volume_refine_splats(
                    seed, V, config=VolumeRefitConfig(iters=1), device="cpu"
                )
        finally:
            fit_gsplats_mod.fit_gaussian_splats = original

        assert recorded.get("seed_amps_background_relative") is True
        # ...and it really does leave the floor at its "auto" default.
        assert "floor" not in recorded

    def test_progressive_fitting_forces_floor_none(self) -> None:
        """``fit_progressive_gaussian_splats`` subtracts the floor from the volume
        itself and must therefore run every per-pass fit with ``floor='none'``.

        Re-estimating a floor on the already background-relative volume (or on its
        residuals) would eat signal, so this is a real invariant of the
        progressive path; ``test_progressive_floor_suppresses_background`` covers
        it behaviourally, and this pins it at the call site. It is NOT, however,
        what
        makes that path immune to the double subtraction this class guards:
        progressive pops any caller-supplied ``seeds`` and passes an int count per
        pass, so ``isinstance(seeds, GSplatData)`` is never true there and the
        warm-start door is unreachable whatever the floor.
        """
        import luxar.gsplats.fit_gsplats as fit_gsplats_mod
        from luxar.gsplats.fit_progressive_gsplats import (
            fit_progressive_gaussian_splats,
        )

        recorded: dict[str, object] = {}

        class _Stop(RuntimeError):
            pass

        def _spy(*args, **kwargs):
            recorded.update(kwargs)
            raise _Stop

        original = fit_gsplats_mod.fit_gaussian_splats
        fit_gsplats_mod.fit_gaussian_splats = _spy
        try:
            with pytest.raises(_Stop):
                fit_progressive_gaussian_splats(
                    _pedestal_volume(),
                    max_splats=16,
                    max_splats_per_pass=8,
                    iters_per_pass=2,
                    device="cpu",
                    verbose=False,
                    floor=100.0,
                )
        finally:
            fit_gsplats_mod.fit_gaussian_splats = original

        assert recorded.get("floor") == "none"

    def test_empty_seed_set_survives_the_verbose_summary(self) -> None:
        """An empty seed set must come back as an empty result, not a KeyError.

        ``GaussianSplatFitter.fit`` deliberately short-circuits ``N == 0`` and
        returns an empty ``GSplatData`` with an empty ``stats`` dict, so the
        ``verbose`` summary in ``fit_gaussian_splats`` (verbose defaults to True)
        has no timing/iteration record to print. It used to index them anyway and
        died with ``KeyError: 'time_seconds'`` — right after the rescaling report
        the empty-array guard above keeps alive.
        """
        from luxar.gsplats.fit_gsplats import fit_gaussian_splats

        empty = _gsplatdata_seeds(
            np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        )

        result = fit_gaussian_splats(
            _pedestal_volume(),
            seeds=empty,
            seed_amps_background_relative=True,
            floor=100.0,
            n_iters=1,
            device="cpu",
            verbose=True,
            enable_dynamic_ops=False,
            cull_retention=None,
            sort_splats_enabled=False,
        )

        assert result.n_splats == 0
        assert result.centers.shape == (0, 2)
        assert result.stats == {}


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
