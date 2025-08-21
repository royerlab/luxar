"""Edge case tests for HDR color support."""

import numpy as np
import pytest
import zarr

from luxar import LuxarZarrCompiler


class TestHDRColorSupport:
    """Test HDR color support with edge cases."""

    def test_standard_sdr_colors(self, tmp_path):
        """Test standard SDR colors (0-1 range)."""
        with LuxarZarrCompiler(tmp_path / "sdr.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)  # 0-1 range

            scene.add_points("sdr_points", positions, colors=colors)

        # Verify colors are stored correctly
        store = zarr.open_group(tmp_path / "sdr.zarr", mode="r")
        stored_colors = store["sdr_points/colors"][:]
        assert stored_colors.dtype == np.float32
        assert np.all(stored_colors >= 0)
        assert np.all(stored_colors <= 1)
        np.testing.assert_array_almost_equal(stored_colors, colors)

    def test_hdr_colors_moderate(self, tmp_path):
        """Test moderate HDR colors (1-5 range)."""
        with LuxarZarrCompiler(tmp_path / "hdr_moderate.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32) * 5.0  # 0-5 range

            scene.add_points("hdr_points", positions, colors=colors)

        # Verify HDR colors are preserved
        store = zarr.open_group(tmp_path / "hdr_moderate.zarr", mode="r")
        stored_colors = store["hdr_points/colors"][:]
        assert stored_colors.dtype == np.float32
        assert np.max(stored_colors) > 1.0  # HDR values
        assert np.max(stored_colors) <= 5.0
        np.testing.assert_array_almost_equal(stored_colors, colors)

    def test_hdr_colors_extreme(self, tmp_path):
        """Test extreme HDR colors with warnings."""
        with LuxarZarrCompiler(tmp_path / "hdr_extreme.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32) * 100.0  # Very bright

            # Should trigger warning but still work
            with pytest.warns(UserWarning, match="HDR colors.*maximum value"):
                compiler.write_points("extreme_hdr", positions, colors=colors)

        # Verify extreme values are preserved
        store = zarr.open_group(tmp_path / "hdr_extreme.zarr", mode="r")
        stored_colors = store["extreme_hdr/colors"][:]
        assert np.max(stored_colors) > 10.0  # Extreme HDR
        np.testing.assert_array_almost_equal(stored_colors, colors, decimal=2)

    def test_mixed_hdr_sdr_colors(self, tmp_path):
        """Test mixed HDR and SDR values in same array."""
        with LuxarZarrCompiler(tmp_path / "mixed.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)
            # Make some points HDR
            colors[::2] *= 10.0  # Every other point is HDR

            scene.add_points("mixed_points", positions, colors=colors)

        # Verify mixed values are preserved
        store = zarr.open_group(tmp_path / "mixed.zarr", mode="r")
        stored_colors = store["mixed_points/colors"][:]

        # Check SDR points
        sdr_mask = np.arange(100) % 2 == 1
        assert np.all(stored_colors[sdr_mask] <= 1.0)

        # Check HDR points
        hdr_mask = np.arange(100) % 2 == 0
        assert np.any(stored_colors[hdr_mask] > 1.0)

        np.testing.assert_array_almost_equal(stored_colors, colors)

    def test_zero_colors(self, tmp_path):
        """Test all-zero colors (black points)."""
        with LuxarZarrCompiler(tmp_path / "black.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.zeros((100, 3), dtype=np.float32)  # All black

            scene.add_points("black_points", positions, colors=colors)

        # Verify zeros are preserved
        store = zarr.open_group(tmp_path / "black.zarr", mode="r")
        stored_colors = store["black_points/colors"][:]
        assert np.all(stored_colors == 0)

    def test_single_hdr_color_broadcast(self, tmp_path):
        """Test broadcasting a single HDR color to all points."""
        with LuxarZarrCompiler(tmp_path / "broadcast_hdr.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            single_hdr_color = [2.0, 3.0, 1.5]  # Single HDR color

            scene.add_points("broadcast_points", positions, colors=single_hdr_color)

        # Verify broadcast HDR color
        store = zarr.open_group(tmp_path / "broadcast_hdr.zarr", mode="r")
        stored_colors = store["broadcast_points/colors"][:]
        assert stored_colors.shape == (100, 3)
        assert np.all(stored_colors[0] == [2.0, 3.0, 1.5])
        assert np.all(stored_colors == stored_colors[0])  # All same

    def test_color_precision(self, tmp_path):
        """Test that float32 precision is maintained."""
        with LuxarZarrCompiler(tmp_path / "precision.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(10, 3).astype(np.float32)
            # Create colors with specific precision requirements
            colors = np.array([
                [0.123456789, 0.987654321, 0.555555555],
                [1.111111111, 2.222222222, 3.333333333],
                [np.pi, np.e, np.sqrt(2)],
                [1e-6, 1e-5, 1e-4],  # Small values
                [1e3, 1e4, 1e5],     # Large HDR values
            ] * 2, dtype=np.float32)[:10]  # Repeat to get 10 colors

            scene.add_points("precision_points", positions, colors=colors)

        # Verify precision is maintained to float32 limits
        store = zarr.open_group(tmp_path / "precision.zarr", mode="r")
        stored_colors = store["precision_points/colors"][:]

        # Float32 has ~7 decimal digits of precision
        np.testing.assert_array_almost_equal(stored_colors, colors, decimal=5)

    def test_negative_color_rejection(self, tmp_path):
        """Test that negative colors are properly rejected."""
        with LuxarZarrCompiler(tmp_path / "negative.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.randn(100, 3).astype(np.float32)  # Can be negative
            colors[0, 0] = -1.0  # Ensure at least one negative

            from luxar.validation import ValidationError
            with pytest.raises(ValidationError, match="Colors cannot be negative"):
                compiler.write_points("negative_colors", positions, colors=colors)

    def test_color_channel_count(self, tmp_path):
        """Test that only RGB (3 channels) is accepted."""
        with LuxarZarrCompiler(tmp_path / "channels.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)

            # Test RGBA (4 channels) - should fail
            rgba_colors = np.random.rand(100, 4).astype(np.float32)
            from luxar.validation import ValidationError
            with pytest.raises(ValidationError, match="must have 3 channels"):
                compiler.write_points("rgba", positions, colors=rgba_colors)

            # Test grayscale (1 channel) - should fail
            gray_colors = np.random.rand(100, 1).astype(np.float32)
            with pytest.raises(ValidationError, match="must have 3 channels"):
                compiler.write_points("gray", positions, colors=gray_colors)

    def test_no_colors_allowed(self, tmp_path):
        """Test that points without colors are allowed."""
        with LuxarZarrCompiler(tmp_path / "no_colors.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.random.randn(100, 3).astype(np.float32)
            # No colors specified
            scene.add_points("no_color_points", positions)

        # Verify no colors dataset was created
        store = zarr.open_group(tmp_path / "no_colors.zarr", mode="r")
        assert "no_color_points/positions" in store
        assert "no_color_points/colors" not in store
