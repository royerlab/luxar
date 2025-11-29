"""Integration tests for the progressive writing API.

These tests demonstrate best practices using LuxarZarrCompiler
and ensure the API works correctly with all features.
"""

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler, transforms


class TestCompilerIntegration:
    """Test the LuxarZarrCompiler progressive writing API integration."""

    def test_simple_scene_creation(self, tmp_path) -> None:
        """Test basic scene creation with new API."""
        output_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene()

            # Add points - written immediately
            positions = np.random.randn(1000, 3).astype(np.float32)
            colors = np.random.rand(1000, 3).astype(np.float32)

            compiler.write_points("points1", positions, colors=colors)

        # Verify the scene was created correctly
        store = zarr.open_group(output_path, mode="r")
        assert store.attrs["type"] == "scene"
        assert "points1" in store
        assert store["points1/positions"].shape == (1000, 3)
        assert store["points1/colors"].shape == (1000, 3)

    def test_hierarchical_scene_with_transforms(self, tmp_path) -> None:
        """Test building hierarchical scenes with transforms."""
        output_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene()

            # Create groups with transforms
            transform1 = transforms.translate(10, 0, 0)
            group1 = scene.add_group("Group1", transform=transform1)

            transform2 = transforms.rotate_z(45)
            group1.add_group("Group2", transform=transform2)

            # Add points to nested group
            positions = np.random.randn(500, 3).astype(np.float32)
            compiler.write_points("Group1/Group2/points", positions)

        # Verify hierarchy
        store = zarr.open_group(output_path, mode="r")
        assert "Group1" in store
        assert "Group1/Group2" in store
        assert "Group1/Group2/points" in store

        # Verify transforms were stored
        assert "transform" in store["Group1"].attrs
        assert "transform" in store["Group1/Group2"].attrs

    def test_scene_with_dimensions(self, tmp_path) -> None:
        """Test scene with dimension specifications."""
        output_path = tmp_path / "test.zarr"

        # Create 5D dimensions
        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
                Dimension("time", unit="s", display=False, discrete=True),
                Dimension("channel", unit="ch", display=False, discrete=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)

            # Add 5D points
            positions = np.random.randn(1000, 5).astype(np.float32)
            compiler.write_points("points5d", positions)

        # Verify dimensions were stored
        store = zarr.open_group(output_path, mode="r")
        assert "scene_dimensions" in store.attrs
        stored_dims = store.attrs["scene_dimensions"]
        assert len(stored_dims["dimensions"]) == 5
        assert stored_dims["dimensions"][3]["name"] == "time"

    def test_hdr_colors_and_attributes(self, tmp_path) -> None:
        """Test HDR colors and rendering attributes."""
        output_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene()

            # Create points with HDR colors
            positions = np.random.randn(100, 3).astype(np.float32)

            # HDR colors with values > 1.0
            colors = np.random.rand(100, 3).astype(np.float32) * 10.0

            # Write with rendering attributes
            compiler.write_points(
                "hdr_points",
                positions,
                colors=colors,
                opacity=0.7,
                gamma=1.2,
                blending_mode="additive",
            )

        # Verify HDR colors and attributes
        store = zarr.open_group(output_path, mode="r")
        stored_colors = store["hdr_points/colors"][:]
        assert stored_colors.max() > 1.0  # HDR values
        assert store["hdr_points"].attrs["opacity"] == 0.7
        assert store["hdr_points"].attrs["gamma"] == 1.2
        assert store["hdr_points"].attrs["blending_mode"] == "additive"

    # Legacy API compatibility test removed - we no longer support the old API

    def test_memory_efficiency(self, tmp_path) -> None:
        """Test that large data doesn't accumulate in memory."""
        output_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene()

            # Write multiple large arrays
            for i in range(10):
                # Each array is ~12MB (1M points * 3 dims * 4 bytes)
                large_positions = np.random.randn(1_000_000, 3).astype(np.float32)

                # Write immediately, don't keep in memory
                metadata = compiler.write_points(f"cloud_{i}", large_positions)

                # Metadata should be small
                assert "positions" not in metadata  # Data not in metadata
                assert metadata["n_points"] == 1_000_000

                # Clear reference to allow garbage collection
                del large_positions

        # Verify all data was written
        store = zarr.open_group(output_path, mode="r")
        for i in range(10):
            assert f"cloud_{i}" in store
            assert store[f"cloud_{i}/positions"].shape == (1_000_000, 3)

    def test_error_handling_in_context(self, tmp_path) -> None:
        """Test error handling with context manager."""
        output_path = tmp_path / "test.zarr"

        with pytest.raises(ValueError):
            with LuxarZarrCompiler(output_path) as compiler:
                compiler.create_scene()

                # Try to write invalid data
                invalid_positions = np.random.randn(100)  # 1D instead of 2D
                compiler.write_points("bad_points", invalid_positions)

        # Even with error, context manager should clean up
        # Store should still be finalized (though incomplete)
        assert output_path.exists()
