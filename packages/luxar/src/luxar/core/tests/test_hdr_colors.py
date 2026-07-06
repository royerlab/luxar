"""Edge case tests for HDR color support."""

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding.decoder import ArrayDecoder


class TestHDRColorSupport:
    """Test HDR color support with edge cases."""

    def test_standard_sdr_colors(self, tmp_path) -> None:
        """Test standard SDR colors (0-1 range)."""
        with LuxarZarrCompiler(
            tmp_path / "sdr.luxar.zarr", enable_spatial_index=False
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)  # 0-1 range

            scene.add_points("sdr_points", positions, colors=colors)

        # Verify colors are stored correctly
        # With AUTO mode, SDR colors should be converted to uint8 for efficiency
        store = zarr.open_group(tmp_path / "sdr.luxar.zarr", mode="r")
        stored_colors = store["sdr_points/colors"][:]
        assert stored_colors.dtype == np.uint8  # AUTO mode converts SDR to uint8
        assert np.all(stored_colors >= 0)
        assert np.all(stored_colors <= 255)
        # Check values are correctly normalized
        np.testing.assert_array_almost_equal(stored_colors / 255.0, colors, decimal=2)

    def test_hdr_colors_moderate(self, tmp_path) -> None:
        """Test moderate HDR colors (1-5 range)."""
        with LuxarZarrCompiler(
            tmp_path / "hdr_moderate.luxar.zarr", enable_spatial_index=False
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32) * 5.0  # 0-5 range

            scene.add_points("hdr_points", positions, colors=colors)

        # HDR colors quantize to per-channel true-log uint16 under AUTO
        # (2026-07 policy); preservation is verified through DECODE.
        store = zarr.open_group(tmp_path / "hdr_moderate.luxar.zarr", mode="r")
        arr = store["hdr_points/colors"]
        assert arr.dtype == np.uint16
        assert arr.attrs["encoding"]["name"] == "geolog_perchannel_u16"
        decoded = np.asarray(ArrayDecoder().decode(arr, store["hdr_points"]))
        assert np.max(decoded) > 1.0  # HDR values survive the round-trip
        assert np.max(decoded) <= 5.0 * (1 + 1e-3)
        np.testing.assert_allclose(decoded, colors, rtol=1e-3)

    def test_hdr_colors_extreme(self, tmp_path) -> None:
        """Test extreme HDR colors with warnings."""
        with LuxarZarrCompiler(
            tmp_path / "hdr_extreme.luxar.zarr", enable_spatial_index=False
        ) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32) * 100.0  # Very bright

            # Should trigger warning but still work
            with pytest.warns(UserWarning, match="HDR colors.*maximum value"):
                compiler.write_points("extreme_hdr", positions, colors=colors)

        # Verify extreme values survive the geolog u16 round-trip (uniform
        # RELATIVE precision — exactly the regime true-log exists for).
        store = zarr.open_group(tmp_path / "hdr_extreme.luxar.zarr", mode="r")
        arr = store["extreme_hdr/colors"]
        decoded = np.asarray(ArrayDecoder().decode(arr, store["extreme_hdr"]))
        assert np.max(decoded) > 10.0  # Extreme HDR
        np.testing.assert_allclose(decoded, colors, rtol=1e-3)

    def test_mixed_hdr_sdr_colors(self, tmp_path) -> None:
        """Test mixed HDR and SDR values in same array."""
        with LuxarZarrCompiler(
            tmp_path / "mixed.luxar.zarr", enable_spatial_index=False
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)
            # Make some points HDR
            colors[::2] *= 10.0  # Every other point is HDR

            scene.add_points("mixed_points", positions, colors=colors)

        # A mixed array with any value > 1 is HDR -> geolog u16; verify both
        # populations through decode.
        store = zarr.open_group(tmp_path / "mixed.luxar.zarr", mode="r")
        arr = store["mixed_points/colors"]
        decoded = np.asarray(ArrayDecoder().decode(arr, store["mixed_points"]))

        # Check SDR points
        sdr_mask = np.arange(100) % 2 == 1
        assert np.all(decoded[sdr_mask] <= 1.0 * (1 + 1e-3))

        # Check HDR points
        hdr_mask = np.arange(100) % 2 == 0
        assert np.any(decoded[hdr_mask] > 1.0)

        np.testing.assert_allclose(decoded, colors, rtol=1e-3)

    def test_zero_colors(self, tmp_path) -> None:
        """Test all-zero colors (black points)."""
        with LuxarZarrCompiler(tmp_path / "black.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.zeros((100, 3), dtype=np.float32)  # All black

            scene.add_points("black_points", positions, colors=colors)

        # Verify zeros are preserved
        store = zarr.open_group(tmp_path / "black.luxar.zarr", mode="r")
        stored_colors = store["black_points/colors"][:]
        assert np.all(stored_colors == 0)

    def test_single_hdr_color_broadcast(self, tmp_path) -> None:
        """Test broadcasting a single HDR color to all points."""
        with LuxarZarrCompiler(tmp_path / "broadcast_hdr.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            single_hdr_color = [2.0, 3.0, 1.5]  # Single HDR color

            scene.add_points("broadcast_points", positions, colors=single_hdr_color)

        # Verify broadcast HDR color (NEW: encoder detects uniform and uses broadcasting)
        store = zarr.open_group(tmp_path / "broadcast_hdr.luxar.zarr", mode="r")
        colors_arr = store["broadcast_points/colors"]
        stored_colors = colors_arr[:]

        # With broadcasting encoding, shape is (1, 3) with metadata
        assert stored_colors.shape == (1, 3), (
            f"Expected broadcasted shape (1, 3), got {stored_colors.shape}"
        )
        assert np.allclose(stored_colors[0], [2.0, 3.0, 1.5])

        # Check encoding metadata
        enc = colors_arr.attrs.get("encoding", {})
        assert enc["name"] == "broadcasted", (
            "Should use broadcasted encoding for uniform colors"
        )
        assert enc["n_elements"] == 100, (
            "Broadcasting metadata should indicate 100 elements"
        )

    def test_color_precision(self, tmp_path) -> None:
        """Test that float32 precision is maintained."""
        with LuxarZarrCompiler(
            tmp_path / "precision.luxar.zarr", enable_spatial_index=False
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(10, 3).astype(np.float32)
            # Create colors with specific precision requirements
            colors = np.array(
                [
                    [0.123456789, 0.987654321, 0.555555555],
                    [1.111111111, 2.222222222, 3.333333333],
                    [np.pi, np.e, np.sqrt(2)],
                    [1e-6, 1e-5, 1e-4],  # Small values
                    [1e3, 1e4, 1e5],  # Large HDR values
                ]
                * 2,
                dtype=np.float32,
            )[:10]  # Repeat to get 10 colors

            scene.add_points("precision_points", positions, colors=colors)

        # Verify precision is maintained to float32 limits
        # Use ArrayDecoder to properly decode (may be LUT/broadcasted/etc)
        from luxar.encoding import ArrayDecoder

        store = zarr.open_group(tmp_path / "precision.luxar.zarr", mode="r")
        decoder = ArrayDecoder()
        stored_colors = decoder.decode(store["precision_points/colors"], store)

        # Float32 has ~7 decimal digits of precision
        np.testing.assert_array_almost_equal(stored_colors, colors, decimal=5)

    def test_negative_color_rejection(self, tmp_path) -> None:
        """Test that negative colors are properly rejected."""
        with LuxarZarrCompiler(tmp_path / "negative.luxar.zarr") as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.randn(100, 3).astype(np.float32)  # Can be negative
            colors[0, 0] = -1.0  # Ensure at least one negative

            from luxar.validation import ValidationError

            with pytest.raises(ValidationError, match="Colors cannot be negative"):
                compiler.write_points("negative_colors", positions, colors=colors)

    def test_color_channel_count(self, tmp_path) -> None:
        """Test that only RGB (3 channels) is accepted."""
        with LuxarZarrCompiler(tmp_path / "channels.luxar.zarr") as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

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

    def test_no_colors_allowed(self, tmp_path) -> None:
        """Test that points without colors are allowed."""
        with LuxarZarrCompiler(tmp_path / "no_colors.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(100, 3).astype(np.float32)
            # No colors specified
            scene.add_points("no_color_points", positions)

        # Verify no colors dataset was created
        store = zarr.open_group(tmp_path / "no_colors.luxar.zarr", mode="r")
        assert "no_color_points/positions" in store
        assert "no_color_points/colors" not in store
