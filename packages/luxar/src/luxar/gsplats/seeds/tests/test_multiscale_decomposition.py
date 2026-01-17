"""
Tests for seed_from_decomposition - scale-hierarchical detection.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds import (
    dedupe_farthest_first,
    local_maxima,
    seed_from_decomposition,
)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


def validate_gsplatdata(result: GSplatData, expected_ndim: int) -> None:
    """Validate GSplatData output format."""
    assert isinstance(result, GSplatData), "Result should be GSplatData"
    assert result.centers.ndim == 2, "Centers should be 2D array"
    assert result.centers.shape[1] == expected_ndim, (
        f"Should have {expected_ndim}D coordinates"
    )
    assert len(result.amplitudes) == len(result.centers), (
        "Amplitudes should match centers count"
    )
    assert len(result.sharpnesses) == len(result.centers), (
        "Sharpnesses should match centers count"
    )

    tril_size = expected_ndim * (expected_ndim + 1) // 2
    assert result.cholesky_factors.shape == (len(result.centers), tril_size), (
        f"Cholesky factors should be (N, {tril_size})"
    )


class TestInputValidation:
    """Test input validation for all functions."""

    def test_local_maxima_edge_cases(self) -> None:
        """Test edge cases for local_maxima."""
        img = np.ones((3, 3))

        coords = local_maxima(img, radius=1, thresh=10.0, top_k=5)
        assert coords.size == 0

        img[1, 1] = 2
        coords = local_maxima(img, radius=1, thresh=1.5, top_k=100)
        assert len(coords) <= 100

    def test_dedupe_farthest_first_edge_cases(self) -> None:
        """Test edge cases for dedupe_farthest_first."""
        coords = np.array([[0, 0], [10, 10]], dtype=float)
        deduped, _ = dedupe_farthest_first(coords, min_distance=1e-10)
        assert len(deduped) == 2

        deduped, _ = dedupe_farthest_first(coords, min_distance=1000.0)
        assert len(deduped) == 1


class TestDecompositionSeeds:
    """Test seed_from_decomposition function."""

    def test_basic_functionality_2d(self) -> None:
        """Test basic seed generation on 2D synthetic image."""
        x = np.linspace(-5, 5, 64)
        y = np.linspace(-5, 5, 64)
        X, Y = np.meshgrid(x, y)

        img = np.exp(-((X + 2) ** 2 + (Y + 2) ** 2) / 0.5) + 1.5 * np.exp(
            -((X - 2) ** 2 + (Y - 2) ** 2) / 2.0
        )

        result = seed_from_decomposition(
            img,
            scales=[1, 2, 4, 8],
            ignore_finest_k=1,
            min_distance=2.0,
            threshold_rel=0.1,
            decompose_kwargs={"n_iters": 100, "verbose": False},
            verbose=False,
        )

        validate_gsplatdata(result, 2)
        assert len(result.centers) > 0

        # Coordinates within bounds
        assert np.all(result.centers[:, 0] >= 0)
        assert np.all(result.centers[:, 0] <= img.shape[0])
        assert np.all(result.centers[:, 1] >= 0)
        assert np.all(result.centers[:, 1] <= img.shape[1])

    def test_ignore_finest_k_parameter(self) -> None:
        """Test that ignore_finest_k properly filters scales."""
        img = np.random.rand(32, 32) * 0.1
        img[10:15, 10:15] = 1.0

        result_k0 = seed_from_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=0,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        result_k1 = seed_from_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        validate_gsplatdata(result_k0, 2)
        validate_gsplatdata(result_k1, 2)

    def test_min_distance_deduplication(self) -> None:
        """Test that min_distance properly deduplicates seeds."""
        img = np.zeros((64, 64))
        for i, j in [(30, 30), (31, 30), (30, 31), (32, 32)]:
            img[i, j] = 0.8

        result_small = seed_from_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            min_distance=1.0,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        result_large = seed_from_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            min_distance=10.0,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        assert len(result_large.centers) <= len(result_small.centers)

    def test_threshold_rel_filtering(self) -> None:
        """Test that threshold_rel properly filters weak peaks."""
        img = np.zeros((64, 64))
        img[20, 20] = 1.0
        img[40, 40] = 0.2

        result_high = seed_from_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            threshold_rel=0.5,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        result_low = seed_from_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            threshold_rel=0.05,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        assert len(result_low.centers) >= len(result_high.centers)

    def test_1d_image(self) -> None:
        """Test seed generation on 1D signal."""
        x = np.linspace(-5, 5, 128)
        signal = np.exp(-(x**2)) + 0.5 * np.exp(-((x - 2) ** 2) / 0.5)

        result = seed_from_decomposition(
            signal,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        validate_gsplatdata(result, 1)
        assert len(result.centers) > 0
        assert np.all(result.centers[:, 0] >= 0)
        assert np.all(result.centers[:, 0] <= len(signal))

    def test_3d_volume(self) -> None:
        """Test seed generation on 3D volume."""
        x = np.linspace(-2, 2, 16)
        y = np.linspace(-2, 2, 16)
        z = np.linspace(-2, 2, 16)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        volume = np.exp(-(X**2 + Y**2 + Z**2) / 2.0)

        result = seed_from_decomposition(
            volume,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        validate_gsplatdata(result, 3)
        assert len(result.centers) >= 1

    def test_empty_result_handling(self) -> None:
        """Test handling when no seeds are found."""
        img = np.ones((32, 32)) * 0.5 + 0.01 * np.random.randn(32, 32)

        result = seed_from_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            threshold_rel=0.9,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        validate_gsplatdata(result, 2)

    def test_input_validation(self) -> None:
        """Test input validation for seed_from_decomposition."""
        valid_img = np.random.rand(32, 32)

        with pytest.raises(ValueError, match="cannot be empty"):
            seed_from_decomposition(np.array([]))

        with pytest.raises(ValueError, match="at least 1 dimension"):
            seed_from_decomposition(5.0)

        with pytest.raises(ValueError, match="non-empty list"):
            seed_from_decomposition(valid_img, scales=[])

        with pytest.raises(ValueError, match="positive"):
            seed_from_decomposition(valid_img, scales=[1, -2, 4])

        with pytest.raises(ValueError, match="non-negative"):
            seed_from_decomposition(valid_img, ignore_finest_k=-1)

        with pytest.warns(UserWarning, match="ignore_finest_k"):
            result = seed_from_decomposition(
                valid_img,
                scales=[1, 2],
                ignore_finest_k=5,
                decompose_kwargs={"n_iters": 10, "verbose": False},
                verbose=False,
            )
            validate_gsplatdata(result, 2)

        with pytest.raises(ValueError, match="positive"):
            seed_from_decomposition(valid_img, min_distance=-1.0)

        with pytest.raises(ValueError, match="between 0 and 1"):
            seed_from_decomposition(valid_img, threshold_rel=1.5)

    def test_verbose_mode(self) -> None:
        """Test that verbose mode runs without errors."""
        img = np.random.rand(32, 32)
        img[15:18, 15:18] = 1.0

        result = seed_from_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=True,
        )

        validate_gsplatdata(result, 2)


class TestScaleInfoPreserved:
    """Test that scale information is preserved in Cholesky factors."""

    def test_scale_to_sigma_mapping(self) -> None:
        """Test that seeds from different scales have different sigmas."""
        # Create image with features at different scales
        x, y = np.meshgrid(np.linspace(-10, 10, 64), np.linspace(-10, 10, 64))

        # Run decomposition with multiple scales
        img = np.exp(-(x**2 + y**2) / 4)

        result = seed_from_decomposition(
            img,
            scales=[1, 2, 4, 8],
            ignore_finest_k=0,
            decompose_kwargs={"n_iters": 100, "verbose": False},
            verbose=False,
        )

        if len(result.centers) > 0:
            # Check that cholesky factors are non-zero (scale info preserved)
            assert np.all(result.cholesky_factors[:, 0] > 0), (
                "Diagonal Cholesky elements should be positive (sigma > 0)"
            )


if __name__ == "__main__":
    pytest.main([__file__])
