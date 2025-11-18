"""
Tests for candidate detection functions in gsplats.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.candidates import (
    find_candidates_multiscale_gaussian,
)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


class TestFindCandidatesOvercompleteNd:
    """Test find_candidates_multiscale_gaussian main function."""

    def test_find_candidates_2d_basic(self) -> None:
        """Test basic candidate finding in 2D."""
        # Create simple 2D image with known structures
        x, y = np.meshgrid(np.linspace(-5, 5, 31), np.linspace(-5, 5, 31))
        V = np.exp(-(x**2 + y**2) / 4) + 0.5 * np.exp(
            -((x - 3) ** 2 + (y - 3) ** 2) / 2
        )

        candidates = find_candidates_multiscale_gaussian(V)

        # Should find some candidates
        assert len(candidates) > 0
        assert candidates.shape[1] == 2  # 2D coordinates

        # Coordinates should be within image bounds
        assert np.all(candidates >= 0)
        assert np.all(candidates < np.array(V.shape))

    def test_find_candidates_3d_basic(self) -> None:
        """Test basic candidate finding in 3D."""
        # Simple 3D blob
        x, y, z = np.meshgrid(
            np.linspace(-2, 2, 15), np.linspace(-2, 2, 15), np.linspace(-2, 2, 15)
        )
        V = np.exp(-(x**2 + y**2 + z**2))

        candidates = find_candidates_multiscale_gaussian(V)

        assert len(candidates) > 0
        assert candidates.shape[1] == 3  # 3D coordinates

        """Test effects of various parameters."""
        V = np.random.randn(25, 25) + 1

        # Test percentile threshold effect
        candidates_strict = find_candidates_multiscale_gaussian(
            V, percentile_thresh=90.0
        )
        candidates_loose = find_candidates_multiscale_gaussian(
            V, percentile_thresh=50.0
        )

        # Looser threshold should generally find more candidates
        assert len(candidates_loose) >= len(candidates_strict)

        # Test min_dist effect
        candidates_close = find_candidates_multiscale_gaussian(V, min_distance=0.5)
        candidates_far = find_candidates_multiscale_gaussian(V, min_distance=5.0)

        # Larger min_dist should reduce number of candidates
        assert len(candidates_far) <= len(candidates_close)

    def test_find_candidates_1d(self) -> None:
        """Test candidate finding in 1D."""
        x = np.linspace(-3, 3, 101)
        V = np.exp(-(x**2)) + 0.3 * np.exp(-((x - 1) ** 2))  # Two Gaussians

        candidates = find_candidates_multiscale_gaussian(V)

        assert len(candidates) > 0
        assert candidates.shape[1] == 1  # 1D coordinates

    def test_find_candidates_uniform_image(self) -> None:
        """Test candidate finding on uniform image."""
        V = np.ones((10, 10)) * 5.0  # Completely uniform

        # Should handle uniform image gracefully
        candidates = find_candidates_multiscale_gaussian(V)

        # May find few or no candidates, but shouldn't crash
        assert candidates.shape[1] == 2

    def test_find_candidates_empty_result(self) -> None:
        """Test handling when no candidates are found."""
        # Very noisy image with high threshold
        V = 0.01 * np.random.randn(10, 10)

        candidates = find_candidates_multiscale_gaussian(
            V, percentile_thresh=99.9, peaks_per_scale=1
        )

        # Should return empty array with correct shape
        assert candidates.shape[1] == 2
        # Length may be 0


if __name__ == "__main__":
    pytest.main([__file__])
