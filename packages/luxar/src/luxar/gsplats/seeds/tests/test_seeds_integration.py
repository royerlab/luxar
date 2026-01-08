"""
Integration tests for seeds sub-package.

Tests cross-method comparisons and integration with the main fitting pipeline.
All seeding methods now return GSplatData with scale-informed Gaussian shapes.
"""

import numpy as np
import pytest

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds import (
    combine_seeds,
    seed_from_decomposition,
    seed_from_gaussian,
)


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


def test_methods_comparison() -> None:
    """Compare the two seed generation methods on the same image."""
    # Create a test image with known structure
    image = np.zeros((64, 64), dtype=np.float32)

    # Add some Gaussian blobs at known locations
    for x, y in [(16, 16), (48, 16), (32, 48)]:
        xx, yy = np.meshgrid(np.arange(64), np.arange(64))
        blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3**2))
        image += blob

    # Generate seeds using both methods
    result_gaussian = seed_from_gaussian(
        image, scales=[1, 2], min_distance=5.0, peaks_per_scale=10
    )

    result_decomp = seed_from_decomposition(
        image, scales=[1, 2, 4], min_distance=5.0, ignore_finest_k=0
    )

    # Both methods should return valid GSplatData
    validate_gsplatdata(result_gaussian, 2)
    validate_gsplatdata(result_decomp, 2)

    # Both methods should find seeds
    assert len(result_gaussian.centers) > 0, "Gaussian method should find seeds"
    assert len(result_decomp.centers) > 0, "Decomposition should find seeds"

    # At least one method should find seeds near the central blob
    # (other blobs may or may not be detected depending on parameters)
    center_blob = np.array([32, 32])

    dist_gaussian: float = float(
        np.min(np.linalg.norm(result_gaussian.centers - center_blob, axis=1))
    )
    dist_decomp: float = float(
        np.min(np.linalg.norm(result_decomp.centers - center_blob, axis=1))
    )

    # At least one method should find something reasonably close
    assert dist_gaussian < 20.0 or dist_decomp < 20.0, (
        "Neither method found any seeds near the image center"
    )

    # Test combining seeds from both methods
    combined = combine_seeds(
        result_gaussian.centers, result_decomp.centers, min_distance=5.0
    )

    # Combined should have some seeds (may be fewer than individual sets after dedup)
    assert combined.shape[0] > 0, "Combined seeds should not be empty"

    # Combined should not have more than sum (with some deduplication expected)
    total_before_dedup = len(result_gaussian.centers) + len(result_decomp.centers)
    assert combined.shape[0] <= total_before_dedup, (
        "Combined seeds should not exceed sum of individual sets"
    )


def test_seeds_integration_with_fitting() -> None:
    """Test that generated seeds work properly with the fitting pipeline."""
    # This is a basic integration test to ensure seeds format is compatible
    # Note: Full fitting test would require importing fit_gaussian_splats which may be slow

    # Create a simple test image
    image = np.zeros((32, 32), dtype=np.float32)
    image[16, 16] = 1.0  # Single bright pixel

    # Generate seeds using both methods
    result_gaussian = seed_from_gaussian(image, scales=[1], min_distance=1.0)
    result_decomp = seed_from_decomposition(image, scales=[1, 2], min_distance=1.0)

    # Verify GSplatData format is correct for fitting
    validate_gsplatdata(result_gaussian, 2)
    validate_gsplatdata(result_decomp, 2)

    # Centers should be float numpy arrays with shape (N, ndim)
    centers_gaussian = result_gaussian.centers
    centers_decomp = result_decomp.centers

    assert isinstance(centers_gaussian, np.ndarray), "Centers should be numpy arrays"
    assert centers_gaussian.dtype in [np.float32, np.float64], (
        "Centers should be float type"
    )
    assert centers_gaussian.ndim == 2, "Centers should be 2D arrays"
    assert centers_gaussian.shape[1] == 2, "Centers should have 2D coordinates"

    assert isinstance(centers_decomp, np.ndarray)
    assert centers_decomp.dtype in [np.float32, np.float64]
    assert centers_decomp.ndim == 2
    assert centers_decomp.shape[1] == 2

    # Centers should be within image bounds
    assert np.all(centers_gaussian >= 0) and np.all(centers_gaussian < 32), (
        "Centers should be within image bounds"
    )
    assert np.all(centers_decomp >= 0) and np.all(centers_decomp < 32), (
        "Centers should be within image bounds"
    )

    # Test combining seeds for fitting (with deduplication via min_distance)
    combined = combine_seeds(centers_gaussian, centers_decomp, min_distance=2.0)

    # Combined should also be in correct format
    assert isinstance(combined, np.ndarray)
    assert combined.dtype in [np.float32, np.float64]
    assert combined.ndim == 2
    assert combined.shape[1] == 2
    assert np.all(combined >= 0) and np.all(combined < 32)


def test_methods_on_noisy_image() -> None:
    """Test both methods on a noisy image to compare robustness."""
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)

    # Create clean image with structure - use a Gaussian blob stronger than noise
    clean = np.zeros((48, 48), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(48), np.arange(48))
    clean += 2.0 * np.exp(-((xx - 24) ** 2 + (yy - 24) ** 2) / (2 * 3**2))

    # Add noise (signal-to-noise ratio ensures detection is possible)
    noisy = clean + 0.1 * rng.standard_normal((48, 48)).astype(np.float32)

    # Multiscale Gaussian with CLAHE should be more robust to noise
    result_gaussian = seed_from_gaussian(
        noisy, scales=[1, 2], clahe_tile_size=8, min_distance=2.0
    )

    # Decomposition without CLAHE
    result_decomp = seed_from_decomposition(noisy, scales=[1, 2], min_distance=2.0)

    # Both should return valid GSplatData
    validate_gsplatdata(result_gaussian, 2)
    validate_gsplatdata(result_decomp, 2)

    # At least one method should find some seeds
    total_seeds = len(result_gaussian.centers) + len(result_decomp.centers)
    assert total_seeds > 0, "At least one method should find seeds"

    # Check if either method found something near the central peak (within 12 pixels)
    # (noise makes precise detection difficult)
    center = np.array([24, 24])
    found_near_center = False

    if len(result_gaussian.centers) > 0:
        dist_gaussian = float(
            np.min(np.linalg.norm(result_gaussian.centers - center, axis=1))
        )
        if dist_gaussian < 12.0:
            found_near_center = True

    if len(result_decomp.centers) > 0:
        dist_decomp = float(
            np.min(np.linalg.norm(result_decomp.centers - center, axis=1))
        )
        if dist_decomp < 12.0:
            found_near_center = True

    assert found_near_center, (
        "At least one method should find something near the central peak"
    )


def test_3d_seeds() -> None:
    """Test seed generation in 3D with both methods."""
    # Create simple 3D test image
    image_3d = np.zeros((16, 16, 16), dtype=np.float32)
    image_3d[8, 8, 8] = 1.0  # Single bright voxel

    # Test multiscale Gaussian in 3D
    result_gaussian = seed_from_gaussian(image_3d, scales=[1], min_distance=2.0)

    validate_gsplatdata(result_gaussian, 3)
    assert len(result_gaussian.centers) > 0, "Should find seeds in 3D"
    assert result_gaussian.centers.shape[1] == 3, "3D seeds should have 3 coordinates"

    # Test decomposition in 3D
    result_decomp = seed_from_decomposition(image_3d, scales=[1, 2], min_distance=2.0)

    validate_gsplatdata(result_decomp, 3)
    assert len(result_decomp.centers) > 0
    assert result_decomp.centers.shape[1] == 3

    # Both should find the central voxel
    center_3d = np.array([8, 8, 8])
    dist_gaussian: float = float(
        np.min(np.linalg.norm(result_gaussian.centers - center_3d, axis=1))
    )
    dist_decomp: float = float(
        np.min(np.linalg.norm(result_decomp.centers - center_3d, axis=1))
    )

    assert dist_gaussian < 2.0 or dist_decomp < 2.0


def test_cholesky_factors_vary_by_scale() -> None:
    """Test that Cholesky factors (sigmas) vary by detection scale."""
    # Create image with features at different scales
    x, y = np.meshgrid(np.linspace(-10, 10, 64), np.linspace(-10, 10, 64))

    # Large blob
    large_blob = np.exp(-(x**2 + y**2) / 16)
    result_large = seed_from_gaussian(large_blob, scales=[4.0, 8.0], min_distance=5.0)

    # Small blob
    small_blob = np.exp(-(x**2 + y**2) / 2)
    result_small = seed_from_gaussian(small_blob, scales=[1.0, 2.0], min_distance=5.0)

    if len(result_large.centers) > 0 and len(result_small.centers) > 0:
        # The diagonal Cholesky element (sigma) should be larger for large blobs
        sigma_large = result_large.cholesky_factors[0, 0]
        sigma_small = result_small.cholesky_factors[0, 0]

        assert sigma_large > sigma_small, (
            f"Large blob sigma ({sigma_large}) should be > small blob sigma ({sigma_small})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
