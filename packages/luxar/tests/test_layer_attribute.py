"""Tests for the `layer` node attribute and `color_data_range` encoding."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimension, Dimensions
from luxar.io.compiler import LuxarZarrCompiler
from luxar.validation.types import validate_layer

# ─── validate_layer ─────────────────────────────────────


class TestValidateLayer:
    def test_true(self) -> None:
        assert validate_layer(True) is True

    def test_false(self) -> None:
        assert validate_layer(False) is False

    def test_int_truthy(self) -> None:
        assert validate_layer(1) is True

    def test_int_falsy(self) -> None:
        assert validate_layer(0) is False

    def test_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="boolean"):
            validate_layer("yes")

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="boolean"):
            validate_layer(None)

    def test_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="boolean"):
            validate_layer([])


# ─── Helper ─────────────────────────────────────────────

_DIMS = Dimensions(
    [Dimension(name="x", range=(0, 10)), Dimension(name="y", range=(0, 10))]
)


# ─── layer flag on Node via add_points ──────────────────


class TestLayerOnNode:
    def test_layer_true_persisted(self) -> None:
        """layer=True flows through add_points and is stored in attrs."""
        with tempfile.TemporaryDirectory() as d:
            with LuxarZarrCompiler(f"{d}/scene.zarr") as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                pts = scene.add_points(
                    "ch0",
                    positions=np.random.rand(10, 2).astype(np.float32),
                    layer=True,
                )
                assert pts.layer is True
                assert pts.attrs["layer"] is True

    def test_layer_false_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with LuxarZarrCompiler(f"{d}/scene.zarr") as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                pts = scene.add_points(
                    "ch0",
                    positions=np.random.rand(10, 2).astype(np.float32),
                    layer=False,
                )
                assert pts.layer is False

    def test_layer_default_false(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with LuxarZarrCompiler(f"{d}/scene.zarr") as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                pts = scene.add_points(
                    "ch0",
                    positions=np.random.rand(10, 2).astype(np.float32),
                )
                assert pts.layer is False

    def test_layer_on_group(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with LuxarZarrCompiler(f"{d}/scene.zarr") as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                grp = scene.add_group("overlay", layer=True)
                assert grp.layer is True

    def test_layer_on_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with LuxarZarrCompiler(f"{d}/scene.zarr") as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                verts = np.array([[0, 0], [5, 5], [10, 10]], dtype=np.float32)
                widths = np.array([0.1, 0.1, 0.1], dtype=np.float32)
                lines = scene.add_lines(
                    "mylines", vertices=verts, widths=widths, layer=True
                )
                assert lines.layer is True

    def test_layer_invalid_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with LuxarZarrCompiler(f"{d}/scene.zarr") as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                with pytest.raises((TypeError, ValueError)):
                    scene.add_points(
                        "ch0",
                        positions=np.random.rand(10, 2).astype(np.float32),
                        layer="yes",
                    )


# ─── color_data_range in zarr ───────────────────────────


class TestColorDataRange:
    def test_color_data_range_written(self) -> None:
        """color_data_range should be written to zarr attrs when colors are provided."""
        with tempfile.TemporaryDirectory() as d:
            zarr_path = f"{d}/scene.zarr"
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                colors = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0]], dtype=np.float32)
                scene.add_points(
                    "ch0",
                    positions=np.array([[1, 2], [3, 4]], dtype=np.float32),
                    colors=colors,
                )

            # Read back from zarr
            store = zarr.open(zarr_path, mode="r")
            data_range = store["ch0"].attrs.get("color_data_range")
            assert data_range is not None
            assert len(data_range) == 2
            assert abs(data_range[0] - 0.1) < 1e-5
            assert abs(data_range[1] - 1.0) < 1e-5

    def test_no_color_data_range_without_colors(self) -> None:
        """color_data_range should not be present when no colors are provided."""
        with tempfile.TemporaryDirectory() as d:
            zarr_path = f"{d}/scene.zarr"
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=_DIMS)
                scene.add_points(
                    "ch0",
                    positions=np.array([[1, 2], [3, 4]], dtype=np.float32),
                )

            store = zarr.open(zarr_path, mode="r")
            data_range = store["ch0"].attrs.get("color_data_range")
            assert data_range is None
