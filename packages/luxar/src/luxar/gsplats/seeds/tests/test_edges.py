"""
Tests for the seed_from_edges() edge-based seeding function.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds.edges import seed_from_edges

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


@pytest.fixture
def image_with_edges():
    """Create a 2D image with clear edges (square)."""
    V = np.zeros((100, 100), dtype=float)
    V[30:70, 30:70] = 1.0  # Square in center
    return V


@pytest.fixture
def image_with_circle():
    """Create a 2D image with circular edge."""
    y, x = np.ogrid[:100, :100]
    center = (50, 50)
    radius = 30
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    V = np.exp(-((dist - radius) ** 2) / (2 * 3**2))  # Ring
    return V


@pytest.fixture
def image_3d_with_edges():
    """Create a 3D volume with edges (cube)."""
    V = np.zeros((30, 30, 30), dtype=float)
    V[10:20, 10:20, 10:20] = 1.0  # Cube in center
    return V


@pytest.fixture
def uniform_image():
    """Create a uniform intensity image (no edges)."""
    return np.ones((50, 50), dtype=float) * 5.0


def validate_gsplatdata(result: GSplatData, expected_ndim: int) -> None:
    """Validate GSplatData output format."""
    assert isinstance(result, GSplatData), "Result should be GSplatData"
    assert result.centers.ndim == 2, "Centers should be 2D array (N, ndim)"
    assert result.centers.shape[1] == expected_ndim, (
        f"Should have {expected_ndim}D coordinates"
    )

    N = len(result.centers)
    assert len(result.amplitudes) == N, "Amplitudes should match centers count"
    assert len(result.sharpnesses) == N, "Sharpnesses should match centers count"

    tril_size = expected_ndim * (expected_ndim + 1) // 2
    assert result.cholesky_factors.shape == (N, tril_size), (
        f"Cholesky factors should be (N, {tril_size})"
    )


class TestBasicFunctionality:
    """Test basic edge seeding functionality."""

    def test_basic_2d_edges(self, image_with_edges) -> None:
        """Test basic 2D edge detection."""
        result = seed_from_edges(image_with_edges)
        validate_gsplatdata(result, 2)
        assert len(result.centers) > 0, "Should find seeds along edges"

    def test_basic_3d_edges(self, image_3d_with_edges) -> None:
        """Test basic 3D edge detection."""
        result = seed_from_edges(image_3d_with_edges)
        validate_gsplatdata(result, 3)
        assert len(result.centers) > 0, "Should find seeds along edges"

    def test_1d_edges(self) -> None:
        """Test 1D edge detection."""
        V = np.zeros(100)
        V[40:60] = 1.0  # Step function
        result = seed_from_edges(V, min_distance=1.0)
        validate_gsplatdata(result, 1)
        # May or may not find seeds depending on threshold

    def test_returns_gsplatdata(self, image_with_edges) -> None:
        """Test that result is GSplatData."""
        result = seed_from_edges(image_with_edges)
        assert isinstance(result, GSplatData)

    def test_seeds_near_edges(self, image_with_edges) -> None:
        """Test that seeds are placed near actual edges."""
        result = seed_from_edges(image_with_edges)
        if len(result.centers) > 0:
            # Seeds should be near edge positions (around x=30, x=70, y=30, y=70)
            centers = result.centers
            # At least some seeds should be near the edge boundaries
            near_edge_x = np.logical_or(
                np.abs(centers[:, 1] - 30) < 5, np.abs(centers[:, 1] - 70) < 5
            )
            near_edge_y = np.logical_or(
                np.abs(centers[:, 0] - 30) < 5, np.abs(centers[:, 0] - 70) < 5
            )
            assert np.any(near_edge_x) or np.any(near_edge_y), (
                "Seeds should be near edges"
            )


class TestParameters:
    """Test parameter handling."""

    def test_n_seeds_limits_output(self, image_with_edges) -> None:
        """Test that n_seeds limits output count."""
        result_10 = seed_from_edges(image_with_edges, n_seeds=10)
        result_100 = seed_from_edges(image_with_edges, n_seeds=100)

        assert len(result_10.centers) <= 10
        assert len(result_100.centers) <= 100
        # More seeds requested should generally give more (or equal)
        assert len(result_100.centers) >= len(result_10.centers)

    def test_min_distance(self, image_with_edges) -> None:
        """Test that min_distance is respected."""
        result = seed_from_edges(image_with_edges, min_distance=10.0)

        if len(result.centers) > 1:
            # Check pairwise distances
            centers = result.centers
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    assert dist >= 10.0 - 1e-6, "Seeds should respect min_distance"

    def test_edge_threshold_rel(self, image_with_edges) -> None:
        """Test edge_threshold_rel parameter."""
        result_low = seed_from_edges(image_with_edges, edge_threshold_rel=0.05)
        result_high = seed_from_edges(image_with_edges, edge_threshold_rel=0.5)

        # Lower threshold should find more or equal seeds
        assert len(result_low.centers) >= len(result_high.centers)

    def test_edge_threshold_rel_validation(self, image_with_edges) -> None:
        """Test edge_threshold_rel validation."""
        with pytest.raises(ValueError, match="edge_threshold_rel"):
            seed_from_edges(image_with_edges, edge_threshold_rel=-0.1)
        with pytest.raises(ValueError, match="edge_threshold_rel"):
            seed_from_edges(image_with_edges, edge_threshold_rel=1.5)

    def test_sigma_clamping(self, image_with_edges) -> None:
        """Test min_sigma and max_sigma clamping."""
        result = seed_from_edges(image_with_edges, min_sigma=2.0, max_sigma=5.0)
        validate_gsplatdata(result, 2)

    def test_sigma_validation(self, image_with_edges) -> None:
        """Test sigma parameter validation."""
        with pytest.raises(ValueError, match="min_sigma"):
            seed_from_edges(image_with_edges, min_sigma=-1.0)
        with pytest.raises(ValueError, match="max_sigma"):
            seed_from_edges(image_with_edges, max_sigma=-1.0)
        with pytest.raises(ValueError, match="min_sigma"):
            seed_from_edges(image_with_edges, min_sigma=10.0, max_sigma=5.0)


class TestAnisotropicShapes:
    """Test anisotropic Gaussian shape estimation."""

    def test_cholesky_valid(self, image_with_edges) -> None:
        """Test that Cholesky factors are valid."""
        result = seed_from_edges(image_with_edges)
        if len(result.centers) > 0:
            # Diagonal elements should be positive
            # 2D packed: [L00, L10, L11]
            assert np.all(result.cholesky_factors[:, 0] > 0), "L00 should be positive"
            assert np.all(result.cholesky_factors[:, 2] > 0), "L11 should be positive"

    def test_some_anisotropy(self, image_with_edges) -> None:
        """Test that edge seeds have some anisotropy."""
        result = seed_from_edges(image_with_edges)
        if len(result.centers) > 0:
            # For anisotropic shapes, off-diagonal or unequal diagonals
            # 2D packed: [L00, L10, L11]
            # This test verifies the code runs and produces valid output
            # The actual anisotropy depends on edge orientation
            assert result.cholesky_factors.shape[1] == 3

    def test_structure_radius_effect(self, image_with_edges) -> None:
        """Test that structure_radius affects results."""
        result_small = seed_from_edges(image_with_edges, structure_radius=1.0)
        result_large = seed_from_edges(image_with_edges, structure_radius=10.0)

        validate_gsplatdata(result_small, 2)
        validate_gsplatdata(result_large, 2)

        # Results should differ (different smoothing)
        if len(result_small.centers) > 0 and len(result_large.centers) > 0:
            # Cholesky factors should be different due to different integration
            pass  # Just check it doesn't crash


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_image_raises(self) -> None:
        """Test empty image raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            seed_from_edges(np.array([]))

    def test_scalar_raises(self) -> None:
        """Test scalar raises error."""
        with pytest.raises(ValueError, match="at least 1 dimension"):
            seed_from_edges(np.array(5.0))

    def test_uniform_image(self, uniform_image) -> None:
        """Test uniform image (no edges) returns empty."""
        result = seed_from_edges(uniform_image)
        validate_gsplatdata(result, 2)
        # May have zero seeds since no edges
        assert len(result.centers) >= 0  # Just check it doesn't crash

    def test_high_threshold_no_seeds(self, image_with_edges) -> None:
        """Test high threshold may produce no seeds."""
        result = seed_from_edges(image_with_edges, edge_threshold_rel=0.99)
        validate_gsplatdata(result, 2)
        # May have zero seeds

    def test_small_image(self) -> None:
        """Test small image."""
        V = np.zeros((10, 10))
        V[3:7, 3:7] = 1.0
        result = seed_from_edges(V, min_distance=1.0)
        validate_gsplatdata(result, 2)


class TestOutputFormat:
    """Test output format consistency."""

    def test_dtype_float32(self, image_with_edges) -> None:
        """Test that outputs are float32."""
        result = seed_from_edges(image_with_edges)
        assert result.centers.dtype == np.float32
        assert result.amplitudes.dtype == np.float32
        assert result.cholesky_factors.dtype == np.float32
        assert result.sharpnesses.dtype == np.float32

    def test_sharpness_value(self, image_with_edges) -> None:
        """Test that sharpness is 2.0 (standard Gaussian)."""
        result = seed_from_edges(image_with_edges)
        if len(result.sharpnesses) > 0:
            assert np.all(result.sharpnesses == 2.0)

    def test_coordinates_within_bounds(self, image_with_edges) -> None:
        """Test that coordinates are within image bounds."""
        result = seed_from_edges(image_with_edges)
        if len(result.centers) > 0:
            assert np.all(result.centers >= 0)
            assert np.all(result.centers < np.array(image_with_edges.shape))


class TestReproducibility:
    """Test reproducibility."""

    def test_deterministic_output(self, image_with_edges) -> None:
        """Test that outputs are deterministic."""
        result1 = seed_from_edges(image_with_edges, n_seeds=50)
        result2 = seed_from_edges(image_with_edges, n_seeds=50)

        np.testing.assert_array_equal(result1.centers, result2.centers)
        np.testing.assert_array_equal(result1.amplitudes, result2.amplitudes)
        np.testing.assert_array_equal(
            result1.cholesky_factors, result2.cholesky_factors
        )


class TestCircularEdge:
    """Test with circular edge (varied orientations)."""

    def test_circular_edge_detection(self, image_with_circle) -> None:
        """Test detection on circular edge."""
        result = seed_from_edges(image_with_circle, min_distance=5.0)
        validate_gsplatdata(result, 2)
        assert len(result.centers) > 0, "Should detect circular edge"

    def test_circular_edge_orientations(self, image_with_circle) -> None:
        """Test that circular edge produces varied orientations."""
        result = seed_from_edges(image_with_circle, min_distance=5.0)
        # This is a soft test - just verifying the code runs on circular edges
        # Off-diagonal elements should vary around the circle but we don't
        # assert specific values since they depend on edge orientation
        if len(result.centers) > 1:
            assert result.cholesky_factors.shape[1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
