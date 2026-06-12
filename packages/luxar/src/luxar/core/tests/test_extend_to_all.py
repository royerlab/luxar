"""Tests for extend_to_all functionality in Scene.add_points()."""

import warnings
from typing import Any, cast

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


class TestExtendToAll:
    """Test extend_to_all parameter in Scene.add_points()."""

    def test_no_extension_with_empty_list(self, tmp_path) -> None:
        """Test that extend_to_all=[] has no extension and no warning."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 10)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # All points at Time=0 - would normally trigger warning
            positions = np.array([[0, 0, 0, 0]], dtype=np.float32)

            # Using empty list explicitly silences the warning
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                scene.add_points("points", positions, extend_to_all=[])

        # Check that extend_to_all is not in attributes
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" not in store["points"].attrs

    def test_warning_when_none_and_candidates_detected(self, tmp_path) -> None:
        """Test that a warning is issued when extend_to_all=None and candidates exist."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension(
                    "Time", display=False, range=(0, 10)
                ),  # Range larger than data
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # All points at Time=0, but Time range is [0, 10]
            positions = np.array(
                [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 0]], dtype=np.float32
            )

            # Should warn about Time being a candidate
            with pytest.warns(UserWarning, match="Time"):
                scene.add_points("points", positions)  # extend_to_all=None (default)

        # Should still write without extension
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" not in store["points"].attrs

    def test_no_warning_when_multiple_values_in_dimension(self, tmp_path) -> None:
        """Test no warning when dimension has multiple values (not a candidate)."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 10)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Points at different Time values - not a candidate for extension
            positions = np.array(
                [[0, 0, 0, 0], [1, 1, 1, 5], [2, 2, 2, 10]], dtype=np.float32
            )

            # Should NOT warn because Time has multiple values
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                scene.add_points("points", positions)  # extend_to_all=None

    def test_explicit_extend_to_all_list(self, tmp_path) -> None:
        """Test explicit extend_to_all list."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
                Dimension("Channel", display=False, range=(0, 1)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Points only at Time=0, Channel=0 but extend to all times/channels
            positions = np.array([[1, 2, 3, 0, 0]], dtype=np.float32)
            scene.add_points(
                "extended_points", positions, extend_to_all=["Time", "Channel"]
            )

        # Check that extend_to_all is saved in attributes
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" in store["extended_points"].attrs
        assert store["extended_points"].attrs["extend_to_all"] == ["Time", "Channel"]

    def test_extend_to_all_non_displayed(self, tmp_path) -> None:
        """Test extend_to_all='all' extends all non-displayed dimensions."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
                Dimension("Channel", display=False, range=(0, 1)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            positions = np.array([[1, 2, 3, 0, 0]], dtype=np.float32)
            scene.add_points("extend_all", positions, extend_to_all="all")

        # Check that all non-displayed dimensions are extended
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" in store["extend_all"].attrs
        # Should include both Time and Channel
        extend_dims = store["extend_all"].attrs["extend_to_all"]
        assert "Time" in extend_dims
        assert "Channel" in extend_dims

    def test_invalid_extend_to_all_value(self, tmp_path) -> None:
        """Test that invalid extend_to_all value raises error."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.array([[0, 0, 0]], dtype=np.float32)

            with pytest.raises(ValueError, match="Invalid extend_to_all value"):
                scene.add_points(
                    "points",
                    positions,
                    extend_to_all=cast(Any, 123),  # Invalid type
                )

    def test_extend_with_colors_and_radii(self, tmp_path) -> None:
        """Test extension works with colors and radii."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            positions = np.array([[1, 2, 3, 0]], dtype=np.float32)
            colors = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)  # Orange
            radii = np.array([0.5], dtype=np.float32)

            scene.add_points(
                "colored_extended",
                positions,
                colors=colors,
                radii=radii,
                extend_to_all=["Time"],
            )

        # Check the data was written correctly
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" in store["colored_extended"].attrs
        assert store["colored_extended"].attrs["extend_to_all"] == ["Time"]

        # Check data arrays
        assert store["colored_extended/positions"].shape == (1, 4)
        assert store["colored_extended/colors"].shape == (1, 3)
        assert store["colored_extended/radii"].shape == (1,)

    def test_position_array_must_include_all_dimensions(self, tmp_path) -> None:
        """Test that position arrays must include ALL dimensions even when extending."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # This is correct - includes Time dimension even though extending
            positions_correct = np.array([[1, 2, 3, 0]], dtype=np.float32)
            scene.add_points("correct", positions_correct, extend_to_all=["Time"])

            # Position array should have 4 dimensions (X, Y, Z, Time)
            store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
            assert store["correct/positions"].shape == (1, 4)

    def test_no_warning_without_range(self, tmp_path) -> None:
        """Test no warning when dimension has no defined range."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False),  # No range defined
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # All at Time=0, but no range defined so no warning
            positions = np.array([[0, 0, 0, 0]], dtype=np.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                scene.add_points("points", positions)  # extend_to_all=None
