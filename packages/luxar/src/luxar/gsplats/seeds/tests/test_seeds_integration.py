"""
Integration tests for seeds sub-package.

Tests cross-method comparisons and integration with the main fitting pipeline.
All seeding methods return GSplatData with simplified σ=1.0 isotropic initialization
and 90% amplitude scaling (to avoid overlap and divergence during optimization).
"""

import numpy as np

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds import (
    combine_seeds,
    seed_from_decomposition,
    seed_from_edges,
    seed_from_grid,
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

    tril_size = expected_ndim * (expected_ndim + 1) // 2
    assert result.cholesky_factors.shape == (len(result.centers), tril_size), (
        f"Cholesky factors should be (N, {tril_size})"
    )


def test_methods_comparison() -> None:
    """Compare the three seed generation methods on the same image."""
    # Create a test image with known structure
    image = np.zeros((64, 64), dtype=np.float32)

    # Add some Gaussian blobs at known locations
    for x, y in [(16, 16), (48, 16), (32, 48)]:
        xx, yy = np.meshgrid(np.arange(64), np.arange(64))
        blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3**2))
        image += blob

    # Generate seeds using all three methods
    result_grid = seed_from_grid(image, spacing=10.0)
    result_decomp = seed_from_decomposition(
        image, scales=[1, 2, 4], min_distance=5.0, ignore_finest_k=0
    )
    result_edges = seed_from_edges(image, min_distance=5.0)

    # All methods should return valid GSplatData
    validate_gsplatdata(result_grid, 2)
    validate_gsplatdata(result_decomp, 2)
    validate_gsplatdata(result_edges, 2)

    # Grid method always produces seeds in a grid pattern
    assert len(result_grid.centers) > 0, "Grid method should produce seeds"

    # Decomposition should find seeds on blob features
    assert len(result_decomp.centers) > 0, "Decomposition should find seeds"

    # Edges may or may not find seeds depending on threshold
    # (Gaussian blobs have edges, but they may not be strong enough)

    # At least one method should find seeds near the central blob
    center_blob = np.array([32, 48])  # The third blob location

    distances = []
    if len(result_grid.centers) > 0:
        distances.append(
            float(np.min(np.linalg.norm(result_grid.centers - center_blob, axis=1)))
        )
    if len(result_decomp.centers) > 0:
        distances.append(
            float(np.min(np.linalg.norm(result_decomp.centers - center_blob, axis=1)))
        )
    if len(result_edges.centers) > 0:
        distances.append(
            float(np.min(np.linalg.norm(result_edges.centers - center_blob, axis=1)))
        )

    assert len(distances) > 0, "At least one method should find seeds"
    assert min(distances) < 20.0, "At least one method should find seeds near the blob"

    # Test combining seeds from multiple methods
    combined = combine_seeds(
        result_grid.centers, result_decomp.centers, min_distance=5.0
    )

    # Combined should have some seeds (may be fewer than individual sets after dedup)
    assert combined.shape[0] > 0, "Combined seeds should not be empty"

    # Combined should not have more than sum (with some deduplication expected)
    total_before_dedup = len(result_grid.centers) + len(result_decomp.centers)
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

    # Generate seeds using grid and decomposition methods
    result_grid = seed_from_grid(image, spacing=8.0)
    result_decomp = seed_from_decomposition(image, scales=[1, 2], min_distance=1.0)

    # Verify GSplatData format is correct for fitting
    validate_gsplatdata(result_grid, 2)
    validate_gsplatdata(result_decomp, 2)

    # Centers should be float numpy arrays with shape (N, ndim)
    centers_grid = result_grid.centers
    centers_decomp = result_decomp.centers

    assert isinstance(centers_grid, np.ndarray), "Centers should be numpy arrays"
    assert centers_grid.dtype in [np.float32, np.float64], (
        "Centers should be float type"
    )
    assert centers_grid.ndim == 2, "Centers should be 2D arrays"
    assert centers_grid.shape[1] == 2, "Centers should have 2D coordinates"

    assert isinstance(centers_decomp, np.ndarray)
    assert centers_decomp.dtype in [np.float32, np.float64]
    assert centers_decomp.ndim == 2
    assert centers_decomp.shape[1] == 2

    # Centers should be within image bounds
    assert np.all(centers_grid >= 0) and np.all(centers_grid < 32), (
        "Centers should be within image bounds"
    )
    assert np.all(centers_decomp >= 0) and np.all(centers_decomp < 32), (
        "Centers should be within image bounds"
    )

    # Test combining seeds for fitting (with deduplication via min_distance)
    combined = combine_seeds(centers_grid, centers_decomp, min_distance=2.0)

    # Combined should also be in correct format
    assert isinstance(combined, np.ndarray)
    assert combined.dtype in [np.float32, np.float64]
    assert combined.ndim == 2
    assert combined.shape[1] == 2
    assert np.all(combined >= 0) and np.all(combined < 32)


def test_methods_on_noisy_image() -> None:
    """Test methods on a noisy image to compare robustness."""
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)

    # Create clean image with structure - use a Gaussian blob stronger than noise
    clean = np.zeros((48, 48), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(48), np.arange(48))
    clean += 2.0 * np.exp(-((xx - 24) ** 2 + (yy - 24) ** 2) / (2 * 3**2))

    # Add noise (signal-to-noise ratio ensures detection is possible)
    noisy = clean + 0.1 * rng.standard_normal((48, 48)).astype(np.float32)

    # Grid seeding provides uniform coverage regardless of noise
    result_grid = seed_from_grid(noisy, spacing=8.0)

    # Decomposition may be affected by noise
    result_decomp = seed_from_decomposition(noisy, scales=[1, 2], min_distance=2.0)

    # All should return valid GSplatData
    validate_gsplatdata(result_grid, 2)
    validate_gsplatdata(result_decomp, 2)

    # Grid always produces seeds
    assert len(result_grid.centers) > 0, "Grid should always produce seeds"

    # At least one method should find some seeds
    total_seeds = len(result_grid.centers) + len(result_decomp.centers)
    assert total_seeds > 0, "At least one method should find seeds"

    # Check if methods found something near the central peak (within 12 pixels)
    # (noise makes precise detection difficult)
    center = np.array([24, 24])
    found_near_center = False

    if len(result_grid.centers) > 0:
        dist_grid = float(np.min(np.linalg.norm(result_grid.centers - center, axis=1)))
        if dist_grid < 12.0:
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
    """Test seed generation in 3D with multiple methods."""
    # Create simple 3D test image
    image_3d = np.zeros((16, 16, 16), dtype=np.float32)
    image_3d[8, 8, 8] = 1.0  # Single bright voxel

    # Test grid in 3D
    result_grid = seed_from_grid(image_3d, spacing=4.0)

    validate_gsplatdata(result_grid, 3)
    assert len(result_grid.centers) > 0, "Should find seeds in 3D"
    assert result_grid.centers.shape[1] == 3, "3D seeds should have 3 coordinates"

    # Test decomposition in 3D
    result_decomp = seed_from_decomposition(image_3d, scales=[1, 2], min_distance=2.0)

    validate_gsplatdata(result_decomp, 3)
    assert len(result_decomp.centers) > 0
    assert result_decomp.centers.shape[1] == 3

    # Test edges in 3D (should work but may not find seeds on single voxel)
    result_edges = seed_from_edges(image_3d, min_distance=2.0, edge_threshold_rel=0.05)
    validate_gsplatdata(result_edges, 3)

    # At least one method should find the central voxel
    center_3d = np.array([8, 8, 8])
    dist_grid: float = float(
        np.min(np.linalg.norm(result_grid.centers - center_3d, axis=1))
    )
    dist_decomp: float = float(
        np.min(np.linalg.norm(result_decomp.centers - center_3d, axis=1))
    )

    assert dist_grid < 5.0 or dist_decomp < 2.0, (
        "At least one method should find the central voxel"
    )


def test_decomposition_uses_scale_based_sigma() -> None:
    """Test that decomposition seeding uses scale-based sigma initialization.

    NOTE: seed_from_decomposition uses the detection scale as σ, because the
    multi-scale decomposition provides meaningful scale information.
    Other methods (edges, grid) use σ=1.0 since they lack scale info.
    """
    # Create image with features at different scales
    x, y = np.meshgrid(np.linspace(-10, 10, 64), np.linspace(-10, 10, 64))

    # Large blob - use larger scales for decomposition
    large_blob = np.exp(-(x**2 + y**2) / 16)
    result_large = seed_from_decomposition(large_blob, scales=[4, 8], min_distance=5.0)

    # Small blob - use smaller scales for decomposition
    small_blob = np.exp(-(x**2 + y**2) / 2)
    result_small = seed_from_decomposition(small_blob, scales=[1, 2], min_distance=5.0)

    if len(result_large.centers) > 0 and len(result_small.centers) > 0:
        # Decomposition uses scale-based σ (detection scale from decomposition)
        sigma_large = result_large.cholesky_factors[0, 0]
        sigma_small = result_small.cholesky_factors[0, 0]

        # Large blob detected at larger scale should have larger σ
        # Small blob detected at smaller scale should have smaller σ
        assert sigma_large > sigma_small, (
            f"Large blob sigma ({sigma_large}) should be > small blob sigma ({sigma_small})"
        )
        # Decomposition uses the scale factor as σ, so expect σ >= 1.0
        assert sigma_large >= 1.0, (
            f"Large blob sigma should be >= 1.0, got {sigma_large}"
        )
        assert sigma_small >= 1.0, (
            f"Small blob sigma should be >= 1.0, got {sigma_small}"
        )


def test_edges_produces_isotropic_shapes() -> None:
    """Test that edge seeding produces isotropic Gaussian shapes with σ=1.0.

    NOTE: As of 2024, seeding methods always use σ=1.0 isotropic initialization
    (the structure tensor anisotropic shapes were not effective in practice).
    """
    # Create image with clear edges (rectangle)
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:44, 20:44] = 1.0  # Rectangle

    result_edges = seed_from_edges(image, min_distance=5.0)

    validate_gsplatdata(result_edges, 2)

    if len(result_edges.centers) > 0:
        # Edge seeds should have valid Cholesky factors with σ=1.0
        # 2D packed: [L00, L10, L11]
        L00 = result_edges.cholesky_factors[:, 0]
        L10 = result_edges.cholesky_factors[:, 1]
        L11 = result_edges.cholesky_factors[:, 2]

        # Should be isotropic with σ=1.0
        assert np.allclose(L00, 1.0), "L00 should be 1.0"
        assert np.allclose(L11, 1.0), "L11 should be 1.0"
        assert np.allclose(L10, 0.0), "L10 (off-diagonal) should be 0"


def test_grid_produces_isotropic_shapes() -> None:
    """Test that grid seeding produces isotropic Gaussian shapes with spacing-based σ.

    Grid seeding uses σ = spacing/2 so that splats cover the image with
    ~60% overlap at midpoints between grid points.
    """
    x, y = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
    image = np.exp(-(x**2 + y**2) / 4)

    spacing = 10.0
    result_grid = seed_from_grid(image, spacing=spacing)

    validate_gsplatdata(result_grid, 2)
    assert len(result_grid.centers) > 0

    # Grid seeds should be isotropic: L00 == L11 and L10 == 0
    # 2D packed: [L00, L10, L11]
    L00 = result_grid.cholesky_factors[:, 0]
    L10 = result_grid.cholesky_factors[:, 1]
    L11 = result_grid.cholesky_factors[:, 2]

    # All seeds should have the same sigma (isotropic)
    np.testing.assert_array_almost_equal(L00, L11, decimal=5)
    np.testing.assert_array_almost_equal(L10, np.zeros_like(L10), decimal=5)

    # Sigma should be spacing / 2 for coverage
    expected_sigma = spacing / 2.0
    assert np.allclose(L00, expected_sigma), (
        f"Expected σ={expected_sigma}, got {L00[0]}"
    )
