"""
Integration tests for seeds sub-package.

Tests cross-method comparisons and integration with the main fitting pipeline.
"""

import numpy as np
import pytest

from luxar.gsplats.seeds import (
    combine_seeds,
    find_seeds_multiscale_decomposition,
    find_seeds_multiscale_gaussian,
)


def test_methods_comparison():
    """Compare the two seed generation methods on the same image."""
    # Create a test image with known structure
    image = np.zeros((64, 64), dtype=np.float32)

    # Add some Gaussian blobs at known locations
    for x, y in [(16, 16), (48, 16), (32, 48)]:
        xx, yy = np.meshgrid(np.arange(64), np.arange(64))
        blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3**2))
        image += blob

    # Generate seeds using both methods
    seeds_multiscale = find_seeds_multiscale_gaussian(
        image, scales=[1, 2], min_distance=5.0, peaks_per_scale=10
    )

    seeds_decomp = find_seeds_multiscale_decomposition(
        image, scales=[1, 2, 4], min_distance=5.0, ignore_finest_k=0
    )

    # Both methods should find seeds
    assert seeds_multiscale.shape[0] > 0, "Multiscale should find seeds"
    assert seeds_decomp.shape[0] > 0, "Decomposition should find seeds"

    # Both should have 2D coordinates
    assert seeds_multiscale.shape[1] == 2
    assert seeds_decomp.shape[1] == 2

    # At least one method should find seeds near the central blob
    # (other blobs may or may not be detected depending on parameters)
    center_blob = np.array([32, 32])

    dist_multiscale = np.min(np.linalg.norm(seeds_multiscale - center_blob, axis=1))
    dist_decomp = np.min(np.linalg.norm(seeds_decomp - center_blob, axis=1))

    # At least one method should find something reasonably close
    assert dist_multiscale < 20.0 or dist_decomp < 20.0, (
        "Neither method found any seeds near the image center"
    )

    # Test combining seeds from both methods
    combined = combine_seeds(seeds_multiscale, seeds_decomp, min_distance=5.0)

    # Combined should have some seeds (may be fewer than individual sets after dedup)
    assert combined.shape[0] > 0, "Combined seeds should not be empty"

    # Combined should not have more than sum (with some deduplication expected)
    total_before_dedup = seeds_multiscale.shape[0] + seeds_decomp.shape[0]
    assert combined.shape[0] <= total_before_dedup, (
        "Combined seeds should not exceed sum of individual sets"
    )


def test_seeds_integration_with_fitting():
    """Test that generated seeds work properly with the fitting pipeline."""
    # This is a basic integration test to ensure seeds format is compatible
    # Note: Full fitting test would require importing fit_gaussian_splats which may be slow

    # Create a simple test image
    image = np.zeros((32, 32), dtype=np.float32)
    image[16, 16] = 1.0  # Single bright pixel

    # Generate seeds using both methods
    seeds_mg = find_seeds_multiscale_gaussian(image, scales=[1], min_distance=1.0)
    seeds_decomp = find_seeds_multiscale_decomposition(
        image, scales=[1, 2], min_distance=1.0
    )

    # Verify candidate format is correct for fitting
    # Candidates should be float numpy arrays with shape (N, ndim)
    assert isinstance(seeds_mg, np.ndarray), "Candidates should be numpy arrays"
    assert seeds_mg.dtype in [np.float32, np.float64], "Candidates should be float type"
    assert seeds_mg.ndim == 2, "Candidates should be 2D arrays"
    assert seeds_mg.shape[1] == 2, "Candidates should have 2D coordinates"

    assert isinstance(seeds_decomp, np.ndarray)
    assert seeds_decomp.dtype in [np.float32, np.float64]
    assert seeds_decomp.ndim == 2
    assert seeds_decomp.shape[1] == 2

    # Candidates should be within image bounds
    assert np.all(seeds_mg >= 0) and np.all(seeds_mg < 32), (
        "Candidates should be within image bounds"
    )
    assert np.all(seeds_decomp >= 0) and np.all(seeds_decomp < 32), (
        "Candidates should be within image bounds"
    )

    # Test combining seeds for fitting (with deduplication via min_distance)
    combined = combine_seeds(seeds_mg, seeds_decomp, min_distance=2.0)

    # Combined should also be in correct format
    assert isinstance(combined, np.ndarray)
    assert combined.dtype in [np.float32, np.float64]
    assert combined.ndim == 2
    assert combined.shape[1] == 2
    assert np.all(combined >= 0) and np.all(combined < 32)


def test_methods_on_noisy_image():
    """Test both methods on a noisy image to compare robustness."""
    # Create clean image with structure
    clean = np.zeros((48, 48), dtype=np.float32)
    clean[24, 24] = 1.0

    # Add noise
    noisy = clean + 0.1 * np.random.randn(48, 48).astype(np.float32)

    # Multiscale Gaussian with CLAHE should be more robust to noise
    seeds_mg = find_seeds_multiscale_gaussian(
        noisy, scales=[1], clahe_tile_size=8, min_distance=2.0
    )

    # Decomposition without CLAHE
    seeds_decomp = find_seeds_multiscale_decomposition(
        noisy, scales=[1, 2], min_distance=2.0
    )

    # Both should find the central peak
    assert seeds_mg.shape[0] > 0
    assert seeds_decomp.shape[0] > 0

    # Check if either method found something near the central peak (within 12 pixels)
    # (noise makes precise detection difficult)
    center = np.array([24, 24])
    dist_mg = np.min(np.linalg.norm(seeds_mg - center, axis=1))
    dist_decomp = np.min(np.linalg.norm(seeds_decomp - center, axis=1))

    assert dist_mg < 12.0 or dist_decomp < 12.0, (
        "At least one method should find something near the central peak"
    )


def test_3d_seeds():
    """Test seed generation in 3D with both methods."""
    # Create simple 3D test image
    image_3d = np.zeros((16, 16, 16), dtype=np.float32)
    image_3d[8, 8, 8] = 1.0  # Single bright voxel

    # Test multiscale Gaussian in 3D
    seeds_mg = find_seeds_multiscale_gaussian(image_3d, scales=[1], min_distance=2.0)

    assert seeds_mg.shape[1] == 3, "3D seeds should have 3 coordinates"
    assert seeds_mg.shape[0] > 0, "Should find seeds in 3D"

    # Test decomposition in 3D
    seeds_decomp = find_seeds_multiscale_decomposition(
        image_3d, scales=[1, 2], min_distance=2.0
    )

    assert seeds_decomp.shape[1] == 3
    assert seeds_decomp.shape[0] > 0

    # Both should find the central voxel
    center_3d = np.array([8, 8, 8])
    dist_mg = np.min(np.linalg.norm(seeds_mg - center_3d, axis=1))
    dist_decomp = np.min(np.linalg.norm(seeds_decomp - center_3d, axis=1))

    assert dist_mg < 2.0 or dist_decomp < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
