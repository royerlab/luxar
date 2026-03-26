"""Tests for denoise_pipeline module."""

import numpy as np
import pytest

from luxar.gsplats.preprocessing.denoise_pipeline import (
    _pick_sample_timepoints,
    denoise_volume_array,
    denormalize_volume,
    normalize_volume,
)


class TestNormalization:
    def test_roundtrip(self):
        vol = np.random.rand(10, 20, 20).astype(np.float32) * 1000 + 50
        norm, vmin, vmax = normalize_volume(vol)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0
        recovered = denormalize_volume(norm, vmin, vmax)
        np.testing.assert_allclose(recovered, vol, atol=1e-3)

    def test_constant_volume(self):
        vol = np.full((5, 10, 10), 42.0, dtype=np.float32)
        norm, vmin, vmax = normalize_volume(vol)
        assert np.all(norm == 0.0)
        recovered = denormalize_volume(norm, vmin, vmax)
        np.testing.assert_allclose(recovered, 42.0)

    def test_already_01(self):
        vol = np.random.rand(5, 10, 10).astype(np.float32)
        norm, vmin, vmax = normalize_volume(vol)
        # Roundtrip should recover original, not necessarily identity
        recovered = denormalize_volume(norm, vmin, vmax)
        np.testing.assert_allclose(recovered, vol, atol=1e-3)

    def test_negative_values(self):
        vol = np.random.rand(5, 10, 10).astype(np.float32) * 200 - 100
        norm, vmin, vmax = normalize_volume(vol)
        assert norm.min() >= -1e-6
        assert norm.max() <= 1.0 + 1e-6


class TestPickSampleTimepoints:
    def test_more_samples_than_available(self):
        result = _pick_sample_timepoints(3, 10)
        assert len(result) == 3

    def test_equidistant(self):
        result = _pick_sample_timepoints(100, 5)
        assert len(result) == 5
        assert result[0] == 0
        assert result[-1] == 99

    def test_single_timepoint(self):
        result = _pick_sample_timepoints(1, 5)
        assert len(result) == 1
        assert result[0] == 0

    def test_with_indices(self):
        result = _pick_sample_timepoints(100, 3, timepoint_indices=[10, 20, 30, 40, 50])
        assert len(result) == 3
        assert all(t in [10, 20, 30, 40, 50] for t in result)

    def test_five_from_ten(self):
        result = _pick_sample_timepoints(10, 5)
        assert len(result) == 5
        # Should be roughly equidistant
        assert result[0] == 0
        assert result[-1] == 9


class TestDenoiseVolumeArray:
    @pytest.fixture
    def noisy_volume(self):
        rng = np.random.RandomState(42)
        clean = np.zeros((16, 32, 32), dtype=np.float32)
        # Add some structure
        clean[4:12, 8:24, 8:24] = 1.0
        noisy = clean + rng.normal(0, 0.1, clean.shape).astype(np.float32)
        return noisy, clean

    def test_3d_default(self, noisy_volume):
        noisy, clean = noisy_volume
        denoised = denoise_volume_array(noisy, h=0.05, backend="skimage")
        assert denoised.shape == noisy.shape
        assert denoised.dtype == np.float32

    def test_2d_mode(self, noisy_volume):
        noisy, clean = noisy_volume
        denoised = denoise_volume_array(noisy, h=0.05, use_2d=True, backend="skimage")
        assert denoised.shape == noisy.shape

    def test_preserves_scale(self, noisy_volume):
        noisy, _ = noisy_volume
        # Scale to uint16-like range
        scaled = noisy * 10000 + 5000
        denoised = denoise_volume_array(scaled, h=0.05, backend="skimage")
        # Output should be in similar range (not [0,1])
        assert denoised.min() > 1000
        assert denoised.max() > 5000

    def test_2d_input(self):
        rng = np.random.RandomState(42)
        noisy = rng.rand(32, 32).astype(np.float32)
        denoised = denoise_volume_array(noisy, h=0.05, backend="skimage")
        assert denoised.shape == (32, 32)
