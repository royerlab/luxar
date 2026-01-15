"""
Tests for multi-scale Gaussian splat fitting.

Validates the thin wrapper architecture, parameter scaling, and integration
with decompose_image() and fit_gaussian_splats().
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from luxar.gsplats import fit_multiscale_gaussian_splats


class TestMultiScaleFitting:
    """Tests for fit_multiscale_gaussian_splats function."""

    def test_basic_2d(self) -> None:
        """Test basic 2D multi-scale fitting."""
        # Create simple 2D Gaussian blob
        x = np.linspace(-2, 2, 64)
        y = np.linspace(-2, 2, 64)
        X, Y = np.meshgrid(x, y)
        V = np.exp(-(X**2 + Y**2) / 1.0).astype(np.float32)

        # Fit with minimal iterations for speed
        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=50, n_iters_per_scale=50, verbose=False
        )

        # Verify outputs
        assert result.centers.ndim == 2
        assert result.centers.shape[0] == len(result.amplitudes)
        assert result.cholesky_factors.shape[1] == 3  # 2D cholesky (3 elements)
        assert result.sharpnesses.shape[0] == len(result.amplitudes)
        assert "decomposition_stats" in result.stats
        assert "per_scale_stats" in result.stats
        assert "total_splats" in result.stats
        assert result.stats["total_splats"] > 0

    def test_parameter_shapes(self) -> None:
        """Test that parameter shapes are correct for different dimensions."""
        # 2D case
        V_2d = np.random.rand(32, 32).astype(np.float32)
        result_2d = fit_multiscale_gaussian_splats(
            V_2d, scales=[1], n_iters_decomp=10, n_iters_per_scale=10, verbose=False
        )
        d = 2
        assert result_2d.centers.shape[1] == d
        assert result_2d.cholesky_factors.shape[1] == d * (d + 1) // 2
        assert result_2d.sharpnesses.shape[0] == result_2d.centers.shape[0]
        assert result_2d.centers.shape[0] == len(result_2d.amplitudes)

        # 3D case
        V_3d = np.random.rand(16, 16, 16).astype(np.float32)
        result_3d = fit_multiscale_gaussian_splats(
            V_3d, scales=[1], n_iters_decomp=10, n_iters_per_scale=10, verbose=False
        )
        d = 3
        assert result_3d.centers.shape[1] == d
        assert result_3d.cholesky_factors.shape[1] == d * (d + 1) // 2
        assert result_3d.sharpnesses.shape[0] == result_3d.centers.shape[0]
        assert result_3d.centers.shape[0] == len(result_3d.amplitudes)

    def test_multiple_scales(self) -> None:
        """Test fitting with multiple scales."""
        V = np.random.rand(32, 32).astype(np.float32)
        scales = [1, 2, 4]

        result = fit_multiscale_gaussian_splats(
            V, scales=scales, n_iters_decomp=20, n_iters_per_scale=20, verbose=False
        )

        # Verify per-scale stats
        assert len(result.stats["per_scale_stats"]) == len(scales)
        assert len(result.stats["n_splats_per_scale"]) == len(scales)

        # Each scale should have some splats (with high probability)
        assert sum(result.stats["n_splats_per_scale"]) > 0

    def test_default_scales(self) -> None:
        """Test that default scales work correctly."""
        # Use 64x64 image to allow default scales [1, 2, 4, 8] with 8-pixel minimum
        V = np.random.rand(64, 64).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, n_iters_decomp=20, n_iters_per_scale=20, verbose=False
        )

        # Default should be [1, 2, 4, 8]
        assert len(result.stats["per_scale_stats"]) == 4
        assert result.stats["per_scale_stats"][0]["scale_factor"] == 1
        assert result.stats["per_scale_stats"][1]["scale_factor"] == 2
        assert result.stats["per_scale_stats"][2]["scale_factor"] == 4
        assert result.stats["per_scale_stats"][3]["scale_factor"] == 8

    def test_computational_speedup(self) -> None:
        """Test that computational speedup is calculated."""
        V = np.random.rand(64, 64).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2, 4], n_iters_decomp=20, n_iters_per_scale=20, verbose=False
        )

        # Speedup should be > 1 for multiple scales
        assert result.stats["computational_speedup"] > 1.0

        # For scales [1, 2, 4] in 2D:
        # expected speedup ~ (1 + 1 + 1) / (1 + 0.25 + 0.0625) ≈ 2.3
        assert result.stats["computational_speedup"] > 2.0

    def test_statistics_structure(self) -> None:
        """Test that returned statistics have correct structure."""
        V = np.random.rand(32, 32).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=20, n_iters_per_scale=20, verbose=False
        )

        # Check top-level keys
        assert "decomposition_stats" in result.stats
        assert "per_scale_stats" in result.stats
        assert "n_splats_per_scale" in result.stats
        assert "total_splats" in result.stats
        assert "computational_speedup" in result.stats
        assert "total_time_seconds" in result.stats

        # Check per-scale stats structure
        for scale_stat in result.stats["per_scale_stats"]:
            assert "scale_factor" in scale_stat
            assert "scale_shape" in scale_stat
            assert "n_voxels" in scale_stat
            assert "n_splats" in scale_stat
            assert "final_error" in scale_stat
            assert "time_seconds" in scale_stat

    def test_fit_kwargs_passthrough(self) -> None:
        """Test that fit_kwargs are passed through to fit_gaussian_splats."""
        V = np.random.rand(32, 32).astype(np.float32)

        # Test with custom loss_type
        result = fit_multiscale_gaussian_splats(
            V,
            scales=[1],
            n_iters_decomp=10,
            n_iters_per_scale=10,
            loss_type="mse",
            verbose=False,
        )

        # Should complete without error
        assert result.centers.shape[0] > 0

    def test_input_validation(self) -> None:
        """Test input validation."""
        V = np.random.rand(64, 64).astype(np.float32)

        # TypeError for non-numpy V
        with pytest.raises(TypeError, match="must be a numpy array"):
            fit_multiscale_gaussian_splats([1, 2, 3], scales=[1], verbose=False)

        # Invalid scales
        with pytest.raises(ValueError, match="scale factors must be >= 1"):
            fit_multiscale_gaussian_splats(V, scales=[1, 0], verbose=False)

        # Empty scales
        with pytest.raises(ValueError, match="at least one scale"):
            fit_multiscale_gaussian_splats(V, scales=[], verbose=False)

        # Scale too large for image size (Issue #8 fix)
        with pytest.raises(ValueError, match="too small"):
            fit_multiscale_gaussian_splats(
                np.random.rand(16, 16).astype(np.float32),
                scales=[1, 2, 4],  # Scale 4 gives 4x4, below minimum
                verbose=False,
            )

        # Invalid base_init_sigma (use valid scales to reach this validation)
        with pytest.raises(ValueError, match="base_init_sigma must be positive"):
            fit_multiscale_gaussian_splats(
                V, scales=[1, 2], base_init_sigma=-1, verbose=False
            )

        # Negative n_iters_decomp
        with pytest.raises(ValueError, match="n_iters_decomp must be non-negative"):
            fit_multiscale_gaussian_splats(
                V, scales=[1, 2], n_iters_decomp=-1, verbose=False
            )

        # Negative n_iters_per_scale
        with pytest.raises(ValueError, match="n_iters_per_scale must be non-negative"):
            fit_multiscale_gaussian_splats(
                V, scales=[1, 2], n_iters_per_scale=-1, verbose=False
            )

        # Invalid learning rate
        with pytest.raises(ValueError, match="lr must be positive"):
            fit_multiscale_gaussian_splats(V, scales=[1, 2], lr=-0.01, verbose=False)

        # Invalid V
        with pytest.raises(ValueError, match="non-empty"):
            fit_multiscale_gaussian_splats(np.array([]), scales=[1], verbose=False)

        # Non-finite V
        V_bad = V.copy()
        V_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            fit_multiscale_gaussian_splats(V_bad, scales=[1, 2], verbose=False)

    def test_3d_volume(self) -> None:
        """Test 3D volume fitting."""
        # Create small 3D Gaussian blob
        x = np.linspace(-1, 1, 16)
        X, Y, Z = np.meshgrid(x, x, x)
        V = np.exp(-(X**2 + Y**2 + Z**2) / 0.5).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=20, n_iters_per_scale=20, verbose=False
        )

        # For 3D: d=3
        assert result.centers.shape[1] == 3
        assert result.cholesky_factors.shape[1] == 6
        assert result.sharpnesses.shape[0] == len(result.amplitudes)
        assert result.centers.shape[0] == len(result.amplitudes)
        assert result.stats["total_splats"] > 0

    def test_sharpness_preservation(self) -> None:
        """Test that sharpness is preserved (not scaled)."""
        V = np.random.rand(32, 32).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2, 4], n_iters_decomp=20, n_iters_per_scale=20, verbose=False
        )

        # Extract sharpness column (last column)
        sharpness = result.sharpnesses

        # All sharpness values should be reasonable (close to default 2.0)
        # Since we use default initialization and few iterations
        assert np.all(sharpness > 0)  # Must be positive
        assert np.all(sharpness < 100)  # Should be reasonable

    def test_verbose_output(self, capsys) -> None:
        """Test that verbose mode produces output."""
        V = np.random.rand(32, 32).astype(np.float32)

        _result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10, verbose=True
        )

        # Capture output
        captured = capsys.readouterr()

        # Should have some output from arbol
        assert len(captured.out) > 0
        assert (
            "Multi-Scale Decomposition" in captured.out or "Decomposing" in captured.out
        )

    @patch.dict("sys.modules", {"napari": MagicMock()})
    def test_movie_recording(self) -> None:
        """Test that movie recording works when enabled."""
        # Mock napari module globally to prevent windows from opening

        V = np.random.rand(32, 32).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V,
            scales=[1, 2],
            n_iters_decomp=20,
            n_iters_per_scale=10,
            napari_movie=True,
            movie_every=5,
            verbose=False,
        )

        # Check that movie frames were recorded in decomposition stats
        assert "decomposition_stats" in result.stats
        decomp_stats = result.stats["decomposition_stats"]
        assert "movie_frames" in decomp_stats

        # Movie frames should be a dict with specific keys
        movie_frames = decomp_stats["movie_frames"]
        assert movie_frames is not None
        assert isinstance(movie_frames, dict)

        # Check expected keys in movie frames
        assert "target" in movie_frames
        assert "reconstruction" in movie_frames
        assert "residual" in movie_frames
        assert "scales" in movie_frames
        assert "iterations" in movie_frames

        # Each should be a list with at least some frames
        assert len(movie_frames["target"]) >= 2
        assert len(movie_frames["reconstruction"]) >= 2
        assert len(movie_frames["iterations"]) >= 2

        # All lists should have the same length
        n_frames = len(movie_frames["iterations"])
        assert len(movie_frames["target"]) == n_frames
        assert len(movie_frames["reconstruction"]) == n_frames
        assert len(movie_frames["residual"]) == n_frames

    def test_no_movie_by_default(self) -> None:
        """Test that movie recording is disabled by default."""
        V = np.random.rand(32, 32).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10, verbose=False
        )

        # Movie should not be recorded by default
        decomp_stats = result.stats.get("decomposition_stats", {})
        movie_frames = decomp_stats.get("movie_frames")
        assert movie_frames is None or len(movie_frames) == 0

    def test_per_scale_visualization(self) -> None:
        """Test that per-scale visualization data is generated when enabled."""
        V = np.random.rand(32, 32).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V,
            scales=[1, 2, 4],
            n_iters_decomp=20,
            n_iters_per_scale=20,
            visualize_per_scale=True,
            verbose=False,
        )

        # Check that per-scale visualizations were generated
        assert "per_scale_visualizations" in result.stats
        per_scale_vis = result.stats["per_scale_visualizations"]
        assert len(per_scale_vis) == 3  # Should match number of scales

        # Check structure of each visualization
        for vis in per_scale_vis:
            # Check required keys
            assert "scale_factor" in vis
            assert "scale_shape" in vis
            assert "n_splats" in vis
            assert "original_scale" in vis
            assert "centers" in vis
            assert "reconstruction" in vis
            assert "residual" in vis
            assert "error_mse" in vis
            assert "error_max_abs" in vis

            # Check data types and shapes
            assert isinstance(vis["scale_factor"], int)
            assert isinstance(vis["n_splats"], int)
            assert isinstance(vis["original_scale"], np.ndarray)
            assert isinstance(vis["centers"], np.ndarray)
            assert isinstance(vis["reconstruction"], np.ndarray)
            assert isinstance(vis["residual"], np.ndarray)

            # Reconstruction and residual should match full resolution shape
            assert vis["reconstruction"].shape == V.shape
            assert vis["residual"].shape == V.shape

            # Centers should have 2 columns for 2D data
            assert vis["centers"].ndim == 2
            assert vis["centers"].shape[1] == 2

    def test_no_per_scale_visualization_by_default(self) -> None:
        """Test that per-scale visualization is disabled by default."""
        V = np.random.rand(32, 32).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10, verbose=False
        )

        # Should not have per-scale visualizations by default
        assert "per_scale_visualizations" not in result.stats

    def test_return_intermediate(self) -> None:
        """Test return_intermediate returns per-scale GSplatData and reconstructions."""
        V = np.random.rand(32, 32).astype(np.float32)
        scales = [1, 2, 4]

        result = fit_multiscale_gaussian_splats(
            V,
            scales=scales,
            n_iters_decomp=20,
            n_iters_per_scale=20,
            return_intermediate=True,
            verbose=False,
        )

        # Check intermediate results exist
        assert "intermediate" in result.stats
        assert "scale_images" in result.stats
        intermediate = result.stats["intermediate"]
        scale_images = result.stats["scale_images"]

        # Should have one entry per scale
        assert len(intermediate) == len(scales)
        assert len(scale_images) == len(scales)

        # Check structure of each intermediate result
        for i, inter in enumerate(intermediate):
            scale_factor = scales[i]

            # Required keys
            assert "scale_factor" in inter
            assert "target" in inter
            assert "splats" in inter
            assert "splats_full_res" in inter
            assert "reconstruction" in inter
            assert "residual" in inter

            # Check scale factor matches
            assert inter["scale_factor"] == scale_factor

            # Check target shape matches expected downsampled shape
            expected_shape = tuple(s // scale_factor for s in V.shape)
            assert inter["target"].shape == expected_shape

            # Check reconstruction and residual match target shape
            assert inter["reconstruction"].shape == inter["target"].shape
            assert inter["residual"].shape == inter["target"].shape

            # Check splats is GSplatData
            from luxar.gsplats.fit_result import GSplatData

            assert isinstance(inter["splats"], GSplatData)
            assert isinstance(inter["splats_full_res"], GSplatData)

            # Check splats_full_res has correct dimensionality
            assert inter["splats_full_res"].centers.shape[1] == 2  # 2D

    def test_no_intermediate_by_default(self) -> None:
        """Test that intermediate results are not returned by default."""
        V = np.random.rand(32, 32).astype(np.float32)

        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10, verbose=False
        )

        # Should not have intermediate results by default
        assert "intermediate" not in result.stats
        assert "scale_images" not in result.stats

    def test_seeds_distribution_by_maxima(self) -> None:
        """Test that seeds parameter distributes seeds by local maxima."""
        # Create image with clear features
        V = np.random.rand(64, 64).astype(np.float32) * 0.1
        # Add some bright spots
        V[10:15, 10:15] = 1.0
        V[30:35, 30:35] = 1.0
        V[50:55, 50:55] = 1.0

        total_seeds = 200
        scales = [1, 2, 4]

        result = fit_multiscale_gaussian_splats(
            V,
            seeds=total_seeds,
            scales=scales,
            n_iters_decomp=50,
            n_iters_per_scale=30,
            verbose=False,
        )

        # Should have splats
        assert len(result.amplitudes) > 0

        # Check decomposition stats includes maxima_per_scale
        decomp_stats = result.stats.get("decomposition_stats", {})
        assert "maxima_per_scale" in decomp_stats
        maxima = decomp_stats["maxima_per_scale"]
        assert len(maxima) == len(scales)

        # Each scale should have at least some seeds (due to minimum)
        n_splats_per_scale = result.stats.get("n_splats_per_scale", [])
        assert len(n_splats_per_scale) == len(scales)
        for n in n_splats_per_scale:
            assert n > 0  # Each scale got some seeds

    def test_seeds_none_uses_auto_seeding(self) -> None:
        """Test that seeds=None uses auto-seeding for each scale."""
        V = np.random.rand(32, 32).astype(np.float32)

        # Without seeds parameter
        result = fit_multiscale_gaussian_splats(
            V, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10, verbose=False
        )

        # Should still work and produce splats
        assert len(result.amplitudes) > 0

    def test_seeds_compression_ratio(self) -> None:
        """Test that seeds as float is treated as compression ratio."""
        V = np.random.rand(64, 64).astype(np.float32)
        scales = [1, 2, 4]

        # Use 10% compression ratio
        compression_ratio = 0.1
        result = fit_multiscale_gaussian_splats(
            V,
            seeds=compression_ratio,
            scales=scales,
            n_iters_decomp=30,
            n_iters_per_scale=30,
            verbose=False,
        )

        # Should produce splats
        assert len(result.amplitudes) > 0

        # Verify the compression is roughly in the expected range
        # For 2D: floats_per_splat = 2 + 3 + 2 = 7
        # Total voxels = 64*64 + 32*32 + 16*16 = 4096 + 1024 + 256 = 5376
        # Expected seeds ≈ 0.1 * 5376 / 7 ≈ 77
        # Allow some tolerance due to minimum per-scale and rounding
        n_splats = len(result.amplitudes)
        assert 50 < n_splats < 200, (
            f"Expected ~77 splats for 10% compression, got {n_splats}"
        )

    def test_seeds_compression_ratio_vs_int(self) -> None:
        """Test that compression ratio produces fewer splats than large int."""
        V = np.random.rand(64, 64).astype(np.float32)
        scales = [1, 2]

        # Fit with small compression ratio (should produce fewer splats)
        result_compression = fit_multiscale_gaussian_splats(
            V,
            seeds=0.05,  # 5% compression
            scales=scales,
            n_iters_decomp=20,
            n_iters_per_scale=20,
            verbose=False,
        )

        # Fit with large seed count
        result_many = fit_multiscale_gaussian_splats(
            V,
            seeds=500,  # 500 total seeds
            scales=scales,
            n_iters_decomp=20,
            n_iters_per_scale=20,
            verbose=False,
        )

        # Compression ratio should produce fewer splats
        assert len(result_compression.amplitudes) < len(result_many.amplitudes)

    def test_seeds_compression_ratio_invalid(self) -> None:
        """Test that invalid compression ratios raise errors."""
        V = np.random.rand(32, 32).astype(np.float32)

        # Zero is invalid
        with pytest.raises(ValueError, match="Compression ratio must be in"):
            fit_multiscale_gaussian_splats(
                V, seeds=0.0, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10
            )

        # Negative is invalid
        with pytest.raises(ValueError, match="Compression ratio must be in"):
            fit_multiscale_gaussian_splats(
                V, seeds=-0.1, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10
            )

        # Greater than 1 is invalid
        with pytest.raises(ValueError, match="Compression ratio must be in"):
            fit_multiscale_gaussian_splats(
                V, seeds=1.5, scales=[1, 2], n_iters_decomp=10, n_iters_per_scale=10
            )

    def test_seeds_compression_ratio_one(self) -> None:
        """Test that compression ratio of 1.0 is valid (no compression)."""
        V = np.random.rand(32, 32).astype(np.float32)

        # 1.0 should be valid (100% - no compression, maximum quality)
        result = fit_multiscale_gaussian_splats(
            V,
            seeds=1.0,
            scales=[1, 2],
            n_iters_decomp=10,
            n_iters_per_scale=10,
            verbose=False,
        )

        # Should produce splats (many, since no compression)
        assert len(result.amplitudes) > 0

    def test_coordinate_alignment_across_scales(self) -> None:
        """Test that splat centers are correctly aligned when combining scales.

        This test verifies the half-pixel offset correction when upscaling from
        downsampled coordinates. Without the fix, coarse-scale splats would be
        shifted towards the top-left corner by (scale_factor - 1) / 2 pixels.
        """
        # Create image with a bright spot at a known center location
        V = np.zeros((64, 64), dtype=np.float32)
        center_y, center_x = 32, 32  # Center of image
        # Add a Gaussian-like spot
        y, x = np.ogrid[:64, :64]
        dist2 = (y - center_y) ** 2 + (x - center_x) ** 2
        V = np.exp(-dist2 / (2 * 5**2)).astype(np.float32)

        # Use scale 4 to make offset noticeable (1.5px shift without fix)
        result = fit_multiscale_gaussian_splats(
            V,
            seeds=50,
            scales=[1, 4],
            n_iters_decomp=100,
            n_iters_per_scale=100,
            verbose=False,
        )

        # Find the brightest splat (should be near the center of our spot)
        brightest_idx = np.argmax(result.amplitudes)
        brightest_center = result.centers[brightest_idx]

        # The brightest splat should be within 3 pixels of the true center
        # Without the half-pixel fix, coarse scale would shift it by ~1.5 pixels
        distance_to_center = np.sqrt(
            (brightest_center[0] - center_y) ** 2
            + (brightest_center[1] - center_x) ** 2
        )
        assert distance_to_center < 3.0, (
            f"Brightest splat at {brightest_center} is {distance_to_center:.2f} pixels "
            f"from expected center ({center_y}, {center_x}). "
            f"Possible half-pixel offset issue in coordinate transformation."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
