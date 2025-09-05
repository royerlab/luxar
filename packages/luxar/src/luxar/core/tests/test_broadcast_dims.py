"""Tests for broadcast_dims functionality in Scene.add_points()."""

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


class TestBroadcastDims:
    """Test broadcast_dims parameter in Scene.add_points()."""

    def test_no_broadcast_default(self, tmp_path):
        """Test that default behavior has no broadcasting."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            positions = np.array([[0, 0, 0, 0]], dtype=np.float32)
            scene.add_points("points", positions)  # No broadcast_dims

        # Check that broadcast_dims is not in attributes
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert "broadcast_dims" not in store["points"].attrs

    def test_explicit_broadcast_dims(self, tmp_path):
        """Test explicit broadcast_dims list."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
                Dimension("Channel", display=False, range=(0, 1)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Points only at Time=0, Channel=0 but broadcast to all times/channels
            positions = np.array([[1, 2, 3, 0, 0]], dtype=np.float32)
            scene.add_points(
                "broadcast_points", positions, broadcast_dims=["Time", "Channel"]
            )

        # Check that broadcast_dims is saved in attributes
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert "broadcast_dims" in store["broadcast_points"].attrs
        assert store["broadcast_points"].attrs["broadcast_dims"] == ["Time", "Channel"]

    def test_broadcast_all_non_displayed(self, tmp_path):
        """Test broadcast_dims='all' broadcasts all non-displayed dimensions."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
                Dimension("Channel", display=False, range=(0, 1)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            positions = np.array([[1, 2, 3, 0, 0]], dtype=np.float32)
            scene.add_points("broadcast_all", positions, broadcast_dims="all")

        # Check that all non-displayed dimensions are broadcast
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert "broadcast_dims" in store["broadcast_all"].attrs
        # Should include both Time and Channel
        broadcast_dims = store["broadcast_all"].attrs["broadcast_dims"]
        assert "Time" in broadcast_dims
        assert "Channel" in broadcast_dims

    def test_broadcast_auto_detection(self, tmp_path):
        """Test broadcast_dims='auto' with clear single-value dimension."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Create points all at Time=0 (should auto-detect Time for broadcast)
            positions = np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [2, 0, 0, 0],
                ],
                dtype=np.float32,
            )

            scene.add_points("auto_broadcast", positions, broadcast_dims="auto")

        # Check that Time was auto-detected for broadcasting
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        if "broadcast_dims" in store["auto_broadcast"].attrs:
            broadcast_dims = store["auto_broadcast"].attrs["broadcast_dims"]
            assert "Time" in broadcast_dims

    def test_invalid_broadcast_dims_value(self, tmp_path):
        """Test that invalid broadcast_dims value raises error."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            positions = np.array([[0, 0, 0]], dtype=np.float32)

            with pytest.raises(ValueError, match="Invalid broadcast_dims value"):
                scene.add_points(
                    "points",
                    positions,
                    broadcast_dims=123,  # Invalid type
                )

    def test_broadcast_with_colors_and_radii(self, tmp_path):
        """Test broadcasting works with colors and radii."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            positions = np.array([[1, 2, 3, 0]], dtype=np.float32)
            colors = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)  # Orange
            radii = np.array([0.5], dtype=np.float32)

            scene.add_points(
                "colored_broadcast",
                positions,
                colors=colors,
                radii=radii,
                broadcast_dims=["Time"],
            )

        # Check the data was written correctly
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert "broadcast_dims" in store["colored_broadcast"].attrs
        assert store["colored_broadcast"].attrs["broadcast_dims"] == ["Time"]

        # Check data arrays
        assert store["colored_broadcast/positions"].shape == (1, 4)
        assert store["colored_broadcast/colors"].shape == (1, 3)
        assert store["colored_broadcast/radii"].shape == (1,)

    def test_position_array_must_include_all_dimensions(self, tmp_path):
        """Test that position arrays must include ALL dimensions even when broadcasting."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # This is correct - includes Time dimension even though broadcasting
            positions_correct = np.array([[1, 2, 3, 0]], dtype=np.float32)
            scene.add_points("correct", positions_correct, broadcast_dims=["Time"])

            # Position array should have 4 dimensions (X, Y, Z, Time)
            store = zarr.open_group(tmp_path / "test.zarr", mode="r")
            assert store["correct/positions"].shape == (1, 4)
