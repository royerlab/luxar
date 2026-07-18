"""Integration tests for the progressive writing API.

These tests demonstrate best practices using LuxarZarrCompiler
and ensure the API works correctly with all features.
"""

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler, transforms
from luxar.encoding import ArrayDecoder


class TestCompilerIntegration:
    """Test the LuxarZarrCompiler progressive writing API integration."""

    def test_simple_scene_creation(self, tmp_path) -> None:
        """Test basic scene creation."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

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
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

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
        output_path = tmp_path / "test.luxar.zarr"

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

    def test_array_ref_positions_keep_logical_broadcast_counts(self, tmp_path) -> None:
        """Scalar attrs must broadcast to logical count when positions are array_ref."""
        output_path = tmp_path / "test.luxar.zarr"
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        with LuxarZarrCompiler(output_path, enable_spatial_index=False) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "source",
                positions,
                colors=colors,
                radii=np.arange(1, 5, dtype=np.float32),
            )
            compiler.write_points(
                "ref_with_scalars",
                positions,
                colors=(0.25, 0.5, 0.75),
                radii=0.5,
                sharpness=0.5,
            )
            compiler.write_points(
                "ref_with_colormap",
                positions,
                radii=0.5,
                scalars=1.25,
                colormap="viridis",
            )

        store = zarr.open_group(output_path, mode="r")
        assert store["ref_with_scalars/positions"].shape == (0, 3)
        assert (
            store["ref_with_scalars/positions"].attrs["encoding"]["name"] == "array_ref"
        )

        for path in (
            "ref_with_scalars/colors",
            "ref_with_scalars/radii",
            "ref_with_scalars/sharpnesses",
            "ref_with_colormap/radii",
            "ref_with_colormap/scalars",
        ):
            enc = store[path].attrs["encoding"]
            assert enc["name"] == "broadcasted"
            assert enc["n_elements"] == len(positions)
            decoded = ArrayDecoder().decode(store[path], store)
            assert decoded.shape[0] == len(positions)

    def test_grid_line_vertices_never_lut_encode(self, tmp_path) -> None:
        """Grid-snapped line vertices must not LUT-encode (raw-read loader).

        The viewer's lines spatial-index loader reads ``vertices`` as raw
        chunked zarr with no structural-encoding dispatch, so a lut_uint8
        store (triggered by few unique coordinate values, e.g. a lattice)
        would decode as garbage geometry. Regression test for the
        allow_lut=False carve-out (sibling of deduplicate=False).
        """
        output_path = tmp_path / "test.luxar.zarr"
        rng = np.random.default_rng(0)
        lattice = np.arange(8, dtype=np.float32)
        vertices = lattice[rng.integers(0, 8, size=(500, 3))].astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "grid_lines", vertices, widths=np.full(500, 0.1, np.float32)
            )

        store = zarr.open(str(output_path), mode="r")
        enc = store["grid_lines/vertices"].attrs["encoding"]
        assert not enc["name"].startswith("lut"), enc
        assert enc["name"] == "linear_perchannel_u16"
        decoded = ArrayDecoder().decode(store["grid_lines/vertices"], store)
        # Sort-invariant comparison (spatial ordering permutes rows).
        np.testing.assert_allclose(
            np.sort(decoded, axis=0), np.sort(vertices, axis=0), atol=1e-3
        )

    def test_hdr_colors_and_attributes(self, tmp_path) -> None:
        """Test HDR colors and rendering attributes."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

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

    def test_write_lines_rendering_attribute_defaults(self, tmp_path) -> None:
        """Test that write_lines sets identity-valued rendering defaults but
        deliberately does NOT stamp blending_mode (no identity value — a
        stamped default would shadow ancestor-set modes in the viewer)."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

            vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0]], dtype=np.float32)
            # Write lines WITHOUT explicit rendering attributes
            compiler.write_lines("test_lines", vertices, widths=0.1)

        store = zarr.open_group(output_path, mode="r")
        attrs = dict(store["test_lines"].attrs)

        # Verify defaults are set (matching write_points/write_gsplats behavior)
        assert attrs["opacity"] == 1.0
        assert attrs["gamma"] == 1.0
        assert "blending_mode" not in attrs

    def test_group_blending_mode_not_shadowed_by_leaf_default(self, tmp_path) -> None:
        """A group-authored blending_mode must reach modeless leaves.

        The viewer composes blending_mode nearest-setter-wins, so a leaf that
        does not author a mode must OMIT the attr on disk — a stamped default
        would silently override the ancestor (the historical bug: partition
        parts of a `normal` import rendered as additive glow).
        """
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("surface", blending_mode="normal")

            positions = np.random.randn(50, 3).astype(np.float32)
            group.add_points("pts", positions)
            group.add_lines(
                "lns",
                np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                widths=0.1,
            )

        store = zarr.open_group(output_path, mode="r")
        assert store["surface"].attrs["blending_mode"] == "normal"
        for leaf in ("surface/pts", "surface/lns"):
            assert "blending_mode" not in dict(store[leaf].attrs), (
                f"{leaf} carries a stamped blending_mode that shadows the "
                "group's 'normal' under nearest-setter-wins composition"
            )

    def test_gsplat_leaf_omits_blending_mode_when_unset(self, tmp_path) -> None:
        """GSplat leaves mirror points/lines: no stamped blending_mode."""
        output_path = tmp_path / "test.luxar.zarr"

        rng = np.random.default_rng(0)
        n = 16
        centers = rng.standard_normal((n, 3)).astype(np.float32)
        amplitudes = np.abs(rng.standard_normal(n)).astype(np.float32)
        cholesky = np.tile(np.array([1, 0, 0, 1, 0, 1], dtype=np.float32), (n, 1))

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_gsplats("splats", centers, amplitudes, cholesky)

        store = zarr.open_group(output_path, mode="r")
        assert "blending_mode" not in dict(store["splats"].attrs)

    def test_invalid_blending_mode_rejected_before_write(self, tmp_path) -> None:
        """An invalid blending_mode fails BEFORE any group lands on disk.

        Historically the writers wrote the group first and Node validation
        raised afterwards, leaving a partial leaf on disk. All three geometry
        writers must now fail fast (three-geometry symmetry).
        """
        output_path = tmp_path / "test.luxar.zarr"

        rng = np.random.default_rng(0)
        positions = rng.standard_normal((10, 3)).astype(np.float32)
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        amplitudes = np.abs(rng.standard_normal(10)).astype(np.float32)
        cholesky = np.tile(np.array([1, 0, 0, 1, 0, 1], dtype=np.float32), (10, 1))

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

            with pytest.raises(ValueError, match="Invalid blending mode"):
                compiler.write_points("bad_pts", positions, blending_mode="bogus")
            with pytest.raises(ValueError, match="Invalid blending mode"):
                compiler.write_lines(
                    "bad_lns", vertices, widths=0.1, blending_mode="bogus"
                )
            with pytest.raises(ValueError, match="Invalid blending mode"):
                compiler.write_gsplats(
                    "bad_gs", positions, amplitudes, cholesky, blending_mode="bogus"
                )

        store = zarr.open_group(output_path, mode="r")
        for leaf in ("bad_pts", "bad_lns", "bad_gs"):
            assert leaf not in store, f"partial node {leaf} left on disk"

    def test_memory_efficiency(self, tmp_path) -> None:
        """Test that large data doesn't accumulate in memory."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

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
        output_path = tmp_path / "test.luxar.zarr"

        with pytest.raises(ValueError):
            with LuxarZarrCompiler(output_path) as compiler:
                compiler.create_scene(dimensions=Dimensions.default_3d())

                # Try to write invalid data
                invalid_positions = np.random.randn(100)  # 1D instead of 2D
                compiler.write_points("bad_points", invalid_positions)

        # Even with error, context manager should clean up
        # Store should still be finalized (though incomplete)
        assert output_path.exists()


# ─── Layer flag on Node via add_points ──────────────────


_DIMS_2D = Dimensions(
    [Dimension(name="x", range=(0, 10)), Dimension(name="y", range=(0, 10))]
)


class TestLayerOnNode:
    def test_layer_true_persisted(self, tmp_path) -> None:
        """layer=True flows through add_points and is stored in attrs."""
        with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            pts = scene.add_points(
                "ch0",
                positions=np.random.rand(10, 2).astype(np.float32),
                layer=True,
            )
            assert pts.layer is True
            assert pts.attrs["layer"] is True

    def test_layer_false_persisted(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            pts = scene.add_points(
                "ch0",
                positions=np.random.rand(10, 2).astype(np.float32),
                layer=False,
            )
            assert pts.layer is False

    def test_layer_default_false(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            pts = scene.add_points(
                "ch0",
                positions=np.random.rand(10, 2).astype(np.float32),
            )
            assert pts.layer is False

    def test_layer_on_group(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            grp = scene.add_group("overlay", layer=True)
            assert grp.layer is True

    def test_layer_on_lines(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            verts = np.array([[0, 0], [5, 5], [10, 10]], dtype=np.float32)
            widths = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            lines = scene.add_lines(
                "mylines", vertices=verts, widths=widths, layer=True
            )
            assert lines.layer is True

    def test_layer_invalid_type_raises(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            with pytest.raises((TypeError, ValueError)):
                scene.add_points(
                    "ch0",
                    positions=np.random.rand(10, 2).astype(np.float32),
                    layer="yes",
                )


# ─── color_data_range in zarr ───────────────────────────


class TestColorDataRange:
    def test_color_data_range_written(self, tmp_path) -> None:
        """color_data_range should be written to zarr attrs when colors are provided."""
        zarr_path = tmp_path / "scene.luxar.zarr"
        with LuxarZarrCompiler(zarr_path) as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            colors = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0]], dtype=np.float32)
            scene.add_points(
                "ch0",
                positions=np.array([[1, 2], [3, 4]], dtype=np.float32),
                colors=colors,
            )

        # Read back from zarr
        store = zarr.open(str(zarr_path), mode="r")
        data_range = store["ch0"].attrs.get("color_data_range")
        assert data_range is not None
        assert len(data_range) == 2
        assert abs(data_range[0] - 0.1) < 1e-5
        assert abs(data_range[1] - 1.0) < 1e-5

    def test_no_color_data_range_without_colors(self, tmp_path) -> None:
        """color_data_range should not be present when no colors are provided."""
        zarr_path = tmp_path / "scene.luxar.zarr"
        with LuxarZarrCompiler(zarr_path) as compiler:
            scene = compiler.create_scene(dimensions=_DIMS_2D)
            scene.add_points(
                "ch0",
                positions=np.array([[1, 2], [3, 4]], dtype=np.float32),
            )

        store = zarr.open(str(zarr_path), mode="r")
        data_range = store["ch0"].attrs.get("color_data_range")
        assert data_range is None
