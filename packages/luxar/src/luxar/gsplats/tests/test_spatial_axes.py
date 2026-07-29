"""Unit tests for the shared spatial-axis helpers."""

import numpy as np

from luxar.gsplats.utils.spatial_axes import (
    SPATIAL_SIGMA_EPS,
    spatial_axes_from_max_sigma,
    spatial_only_shift,
)


class TestSpatialAxesFromMaxSigma:
    def test_excludes_zero_variance_axis(self):
        # spatial x,y,z + a zero-variance time axis.
        out = spatial_axes_from_max_sigma(np.array([2.0, 3.0, 4.0, 0.0]))
        assert list(out) == [0, 1, 2]

    def test_all_spatial(self):
        out = spatial_axes_from_max_sigma(np.array([1.0, 1.0, 1.0]))
        assert list(out) == [0, 1, 2]

    def test_fallback_all_when_none_qualify(self):
        # All-degenerate → fall back to every axis (never an empty selection).
        out = spatial_axes_from_max_sigma(np.zeros(4))
        assert list(out) == [0, 1, 2, 3]

    def test_eps_boundary(self):
        # Strictly-greater-than: exactly-eps does not qualify.
        out = spatial_axes_from_max_sigma(
            np.array([SPATIAL_SIGMA_EPS, 2 * SPATIAL_SIGMA_EPS])
        )
        assert list(out) == [1]

    def test_no_fallback_returns_empty_when_none_qualify(self):
        # fallback=False keeps the selection empty so a caller can tell an
        # all-degenerate store apart from a genuinely all-spatial one (the
        # fallback makes both return every axis).
        out = spatial_axes_from_max_sigma(np.zeros(4), fallback=False)
        assert out.size == 0

    def test_no_fallback_keeps_qualifying_axes(self):
        # fallback=False only changes the all-degenerate case; a real selection
        # is returned unchanged.
        out = spatial_axes_from_max_sigma(np.array([2.0, 0.0, 4.0]), fallback=False)
        assert list(out) == [0, 2]


class TestSpatialOnlyShift:
    def test_zeros_categorical_axis(self):
        shift = spatial_only_shift(np.array([5.0, 6.0, 7.0, 42.0]), np.array([0, 1, 2]))
        assert np.allclose(shift, [5.0, 6.0, 7.0, 0.0])

    def test_all_axes(self):
        shift = spatial_only_shift(np.array([1.0, 2.0]), np.array([0, 1]))
        assert np.allclose(shift, [1.0, 2.0])

    def test_length_matches_centroid(self):
        shift = spatial_only_shift(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1]))
        assert shift.shape == (4,)
        assert np.allclose(shift, [0.0, 2.0, 0.0, 0.0])
