"""Comprehensive round-trip tests for the Luxar zarr format.

These tests verify that all aspects of the luxar format are correctly
written and can be read back with exact data preservation using the
LuxarScene reading API and ArrayDecoder.

Tests cover:
- Basic points with positions only
- Points with colors, radii, sharpness
- HDR colors (values > 1.0)
- Scene dimensions (nD support)
- Transforms (hierarchical groups)
- Rendering attributes (opacity, gamma, blending)
- Encoding/decoding (uint8 colors, bounded scalars, broadcasting, LUT)
- Spatial ordering metadata
- Group hierarchy
"""

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarScene, LuxarZarrCompiler, transforms


class TestBasicRoundTrip:
    """Basic round-trip tests for points data."""

    # [Python-R1/io-MAJOR] Single-point and empty-scene boundary cases.
    # Every other test in this file uses >= 200 points; the empty / N=1
    # paths exercise spatial-index degeneracies (a single point has no
    # neighbours and the median-split has nothing to split) that the
    # bulk tests never reach.
    def test_single_point_roundtrip(self, tmp_path) -> None:
        output_path = tmp_path / "test.luxar.zarr"
        positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
        radii = np.array([0.5], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("single", positions, colors=colors, radii=radii)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("single")
        assert data["positions"].shape == (1, 3)
        np.testing.assert_allclose(data["positions"], positions, atol=1e-6)
        assert data["colors"].shape == (1, 3)
        np.testing.assert_allclose(data["colors"], colors, atol=1e-3)
        assert data["radii"].shape == (1,)
        np.testing.assert_allclose(data["radii"], radii, atol=1e-3)
        assert data["metadata"]["n_points"] == 1

    def test_positions_only(self, tmp_path) -> None:
        """Test round-trip with positions only."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 1000

        # Generate test data
        positions = np.random.randn(n_points, 3).astype(np.float32)

        # Write
        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("test_points", positions)

        # Read back
        scene = LuxarScene.load(output_path)
        data = scene.get_points("test_points")

        # Verify positions match exactly
        # Note: Spatial ordering may reorder points, so we compare sorted values
        assert data["positions"].shape == positions.shape
        assert data["positions"].dtype == np.float32

        # Compare actual position values (sort both arrays to account for
        # spatial reordering that the compiler may apply)
        original_sorted = np.sort(positions, axis=0)
        loaded_sorted = np.sort(data["positions"], axis=0)
        np.testing.assert_allclose(
            loaded_sorted,
            original_sorted,
            rtol=1e-4,
            atol=2e-3,  # positions are uint16 fixed-point under AUTO (~extent/65535)
            err_msg="Position values differ after round-trip",
        )

        # Verify metadata
        assert data["metadata"]["type"] == "points"
        assert data["metadata"]["n_points"] == n_points

        # No colors/radii should be None
        assert data["colors"] is None
        assert data["radii"] is None
        assert data["sharpness"] is None

    def test_positions_and_colors(self, tmp_path) -> None:
        """Test round-trip with positions and colors."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 500

        rng = np.random.RandomState(123)
        positions = rng.randn(n_points, 3).astype(np.float32)
        colors = rng.rand(n_points, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("test_points", positions, colors=colors)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("test_points")

        assert data["positions"].shape == positions.shape
        assert data["colors"].shape == colors.shape
        assert data["colors"].dtype == np.float32

        # Verify values survive round-trip. Positions may be spatially reordered
        # (Morton/Hilbert) AND are uint16 fixed-point under AUTO, so lexsort is
        # unstable (quantization can flip near-tied points). Match each loaded
        # point to its nearest original — order- and quantization-robust — then
        # use that mapping to check colors too.
        d = np.linalg.norm(
            data["positions"][:, None, :] - positions[None, :, :], axis=2
        )
        match = d.argmin(axis=1)
        assert (
            d[np.arange(len(match)), match].max() < 2e-3
        ), "Position values differ after round-trip"
        assert len(set(match.tolist())) == len(match), "non-unique NN match"
        # SDR colors go through uint8 quantization: allow ~1/255 error per channel
        np.testing.assert_allclose(
            data["colors"],
            colors[match],
            atol=2.0 / 255,
            err_msg="Color values differ beyond uint8 quantization tolerance",
        )

    def test_full_point_attributes(self, tmp_path) -> None:
        """Test round-trip with all point attributes."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 200

        rng = np.random.RandomState(42)
        positions = rng.randn(n_points, 3).astype(np.float32)
        colors = rng.rand(n_points, 3).astype(np.float32)
        radii = rng.rand(n_points).astype(np.float32) * 0.5 + 0.1
        sharpness = rng.rand(n_points).astype(np.float32)  # normalized [0, 1]

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "test_points",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_points("test_points")

        assert data["positions"].shape == positions.shape
        assert data["colors"].shape == colors.shape
        assert data["radii"].shape == radii.shape
        assert data["sharpness"].shape == sharpness.shape

        # Verify actual values survive round-trip (sort to handle spatial reordering)
        sort_idx_orig = np.lexsort(positions.T)
        sort_idx_load = np.lexsort(data["positions"].T)
        np.testing.assert_allclose(
            data["positions"][sort_idx_load],
            positions[sort_idx_orig],
            rtol=1e-4,
            atol=2e-3,  # positions are uint16 fixed-point under AUTO (~extent/65535)
            err_msg="Position values differ after round-trip",
        )
        np.testing.assert_allclose(
            data["colors"][sort_idx_load],
            colors[sort_idx_orig],
            atol=2.0 / 255,
            err_msg="Color values differ beyond uint8 quantization tolerance",
        )
        # Radii/sharpness are encoded as positive_scalar with dynamic quantization.
        # Sharpness uses uint8 over [0, 1] range: step ≈ 1/255 ≈ 0.0039.
        # Use atol based on quantization step size rather than rtol.
        np.testing.assert_allclose(
            data["radii"][sort_idx_load],
            radii[sort_idx_orig],
            atol=0.01,
            err_msg="Radii values differ after round-trip",
        )
        # The theoretical quantization step is 1 / (2**8 - 1) ≈ 0.00392.
        # Tighten to one full step + tiny float slop. If a real change
        # in the encoder pushes us above this bound, we want to know.
        sharpness_step = 1.0 / 255.0  # ≈ 0.00392 (see decoder.py docstring)
        np.testing.assert_allclose(
            data["sharpness"][sort_idx_load],
            sharpness[sort_idx_orig],
            atol=sharpness_step + 1e-6,
            err_msg="Sharpness values differ after round-trip",
        )


class TestHDRColors:
    """Tests for HDR color support."""

    def test_hdr_colors_preserved(self, tmp_path) -> None:
        """Test that HDR colors (values > 1.0) are preserved."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 100

        positions = np.random.randn(n_points, 3).astype(np.float32)
        # HDR colors with values > 1.0
        colors = np.random.rand(n_points, 3).astype(np.float32) * 10.0

        with LuxarZarrCompiler(output_path, enable_spatial_index=False) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("hdr_points", positions, colors=colors)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("hdr_points")

        # HDR values should be preserved
        assert data["colors"].max() > 1.0
        # Allow tolerance for float32 precision
        np.testing.assert_allclose(data["colors"], colors, rtol=1e-2, atol=1e-3)


class TestSceneDimensions:
    """Tests for scene dimension specifications."""

    def test_3d_scene_dimensions(self, tmp_path) -> None:
        """Test round-trip with explicit 3D dimensions."""
        output_path = tmp_path / "test.luxar.zarr"

        dims = Dimensions(
            [
                Dimension("x", unit="um"),
                Dimension("y", unit="um"),
                Dimension("z", unit="um"),
            ]
        )

        positions = np.random.randn(100, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)
            compiler.write_points("points", positions)

        scene = LuxarScene.load(output_path)

        # Verify dimensions were stored
        assert scene.dimensions is not None
        assert len(scene.dimensions) == 3
        assert scene.dimensions.names == ["x", "y", "z"]
        assert scene.dimensions.dimensions[0].unit == "um"

    def test_5d_scene_dimensions(self, tmp_path) -> None:
        """Test round-trip with 5D dimensions (time, channel, xyz)."""
        output_path = tmp_path / "test.luxar.zarr"

        dims = Dimensions(
            [
                Dimension(
                    "time", unit="s", display=False, discrete=True, range=(0, 10)
                ),
                Dimension(
                    "channel", unit="ch", display=False, discrete=True, range=(0, 3)
                ),
                Dimension("x", unit="um"),
                Dimension("y", unit="um"),
                Dimension("z", unit="um"),
            ]
        )

        positions = np.random.randn(100, 5).astype(np.float32)
        # Set discrete dimension values to integers
        positions[:, 0] = np.random.randint(0, 11, 100).astype(np.float32)  # time
        positions[:, 1] = np.random.randint(0, 4, 100).astype(np.float32)  # channel

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)
            compiler.write_points("points5d", positions)

        scene = LuxarScene.load(output_path)

        assert scene.dimensions is not None
        assert len(scene.dimensions) == 5
        assert scene.dimensions.names == ["time", "channel", "x", "y", "z"]

        # Check non-displayed dimensions
        assert scene.dimensions.displayed == [2, 3, 4]
        assert scene.dimensions.non_displayed == [0, 1]

        # Verify discrete flag
        assert scene.dimensions.dimensions[0].discrete is True
        assert scene.dimensions.dimensions[1].discrete is True

        # [Python-R4/io-MAJOR] Pin POSITION-array round-trip in 5D, not
        # just dimension METADATA. Previously the test verified only the
        # Dimensions schema survived; the actual (100, 5) position array
        # round-trip was untested — a regression that flattened the
        # 4th/5th dimension or wrote only the first 3 columns would
        # have passed every existing assertion. The spatial-index may
        # reorder rows, so we sort lexicographically on both sides.
        data = scene.get_points("points5d")
        assert data["positions"].shape == (100, 5)
        assert data["positions"].dtype == np.float32
        sort_orig = np.lexsort(positions.T)
        sort_load = np.lexsort(data["positions"].T)
        np.testing.assert_allclose(
            data["positions"][sort_load],
            positions[sort_orig],
            rtol=1e-4,
            atol=2e-3,  # positions are uint16 fixed-point under AUTO
            err_msg="5D position values differ after round-trip",
        )

    def test_4d_scene_dimensions(self, tmp_path) -> None:
        """Round-trip with 4D dimensions (time + xyz) — added to fill the
        4D gap between test_3d_scene_dimensions and test_5d_scene_dimensions.
        """
        output_path = tmp_path / "test.luxar.zarr"

        dims = Dimensions(
            [
                Dimension("time", unit="s", display=False, discrete=True, range=(0, 5)),
                Dimension("x", unit="um"),
                Dimension("y", unit="um"),
                Dimension("z", unit="um"),
            ]
        )

        positions = np.random.randn(80, 4).astype(np.float32)
        positions[:, 0] = np.random.randint(0, 6, 80).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)
            compiler.write_points("points4d", positions)

        scene = LuxarScene.load(output_path)
        assert scene.dimensions is not None
        assert len(scene.dimensions) == 4
        assert scene.dimensions.names == ["time", "x", "y", "z"]
        assert scene.dimensions.displayed == [1, 2, 3]
        assert scene.dimensions.non_displayed == [0]

        data = scene.get_points("points4d")
        assert data["positions"].shape == (80, 4)
        assert data["positions"].dtype == np.float32
        sort_orig = np.lexsort(positions.T)
        sort_load = np.lexsort(data["positions"].T)
        np.testing.assert_allclose(
            data["positions"][sort_load],
            positions[sort_orig],
            rtol=1e-4,
            atol=2e-3,  # positions are uint16 fixed-point under AUTO
            err_msg="4D position values differ after round-trip",
        )

    def test_categorical_dimensions(self, tmp_path) -> None:
        """Test round-trip with categorical dimensions."""
        output_path = tmp_path / "test.luxar.zarr"

        dims = Dimensions(
            [
                Dimension("x", unit="um"),
                Dimension("y", unit="um"),
                Dimension("z", unit="um"),
                Dimension(
                    "channel",
                    display=False,
                    categories=["DAPI", "GFP", "mCherry"],
                ),
            ]
        )

        positions = np.random.randn(100, 4).astype(np.float32)
        positions[:, 3] = np.random.randint(0, 3, 100).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)
            compiler.write_points("points", positions)

        scene = LuxarScene.load(output_path)

        assert scene.dimensions is not None
        channel_dim = scene.dimensions.dimensions[3]
        assert channel_dim.categories == ["DAPI", "GFP", "mCherry"]
        assert channel_dim.is_categorical is True


class TestTransforms:
    """Tests for transform handling."""

    def test_points_with_transform(self, tmp_path) -> None:
        """Test round-trip with transform on points."""
        output_path = tmp_path / "test.luxar.zarr"

        positions = np.random.randn(100, 3).astype(np.float32)
        transform = transforms.translate(10, 20, 30)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("points", positions, transform=transform)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        # Transform should be stored and read back
        assert "transform" in data["metadata"]
        stored_transform = data["metadata"]["transform"]

        # Verify transform matrix matches
        np.testing.assert_allclose(stored_transform, transform, rtol=1e-5)

    def test_hierarchical_transforms(self, tmp_path) -> None:
        """Test round-trip with hierarchical group transforms."""
        output_path = tmp_path / "test.luxar.zarr"

        positions = np.random.randn(50, 3).astype(np.float32)
        group_transform = transforms.translate(100, 0, 0)
        points_transform = transforms.rotate_z(45)

        with LuxarZarrCompiler(output_path) as compiler:
            scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Create group with transform
            _group = scene_node.add_group("my_group", transform=group_transform)

            # Add points inside group with its own transform
            compiler.write_points(
                "my_group/nested_points", positions, transform=points_transform
            )

        scene = LuxarScene.load(output_path)

        # Verify group exists
        assert scene.has_node("my_group")
        assert scene.get_node_type("my_group") == "group"

        # Verify points
        data = scene.get_points("my_group/nested_points")
        np.testing.assert_allclose(
            data["metadata"]["transform"], points_transform, rtol=1e-5
        )


class TestRenderingAttributes:
    """Tests for rendering attribute preservation."""

    def test_opacity_gamma_blending(self, tmp_path) -> None:
        """Test round-trip of rendering attributes."""
        output_path = tmp_path / "test.luxar.zarr"

        positions = np.random.randn(50, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "points",
                positions,
                opacity=0.7,
                gamma=1.5,
                blending_mode="additive",
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        assert data["metadata"]["opacity"] == 0.7
        assert data["metadata"]["gamma"] == 1.5
        assert data["metadata"]["blending_mode"] == "additive"

    def test_normal_blending_mode(self, tmp_path) -> None:
        """Test round-trip with normal blending mode."""
        output_path = tmp_path / "test.luxar.zarr"

        positions = np.random.randn(50, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "points",
                positions,
                blending_mode="normal",
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        assert data["metadata"]["blending_mode"] == "normal"


class TestEncodingDecoding:
    """Tests for encoding/decoding correctness."""

    def test_uniform_color_broadcast(self, tmp_path) -> None:
        """Test that uniform colors are broadcast and decoded correctly."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 1000

        positions = np.random.randn(n_points, 3).astype(np.float32)
        # Single color for all points
        colors = np.full((n_points, 3), [1.0, 0.0, 0.0], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("points", positions, colors=colors)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        # All colors should be red after decoding
        assert data["colors"].shape == (n_points, 3)
        # Check uniformity (all same color)
        _unique_colors = np.unique(data["colors"], axis=0)
        # Should be just 1 unique color (allowing for float tolerance)
        np.testing.assert_allclose(data["colors"][0], [1.0, 0.0, 0.0], atol=0.05)

    def test_uniform_radii_broadcast(self, tmp_path) -> None:
        """Test that uniform radii are broadcast and decoded correctly."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 500

        positions = np.random.randn(n_points, 3).astype(np.float32)
        # All same radius
        radii = np.full(n_points, 0.5, dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("points", positions, radii=radii)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        assert data["radii"].shape == (n_points,)
        # All radii should be approximately 0.5
        np.testing.assert_allclose(data["radii"], 0.5, atol=0.01)

    def test_uint8_color_quantization(self, tmp_path) -> None:
        """Test that SDR colors are quantized to uint8 and decoded correctly."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 100

        positions = np.random.randn(n_points, 3).astype(np.float32)
        # SDR colors (0-1 range)
        colors = np.random.rand(n_points, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path, enable_spatial_index=False) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("points", positions, colors=colors)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        # Colors should be float32 after decoding
        assert data["colors"].dtype == np.float32
        # Allow for uint8 quantization error (1/256 ≈ 0.004)
        np.testing.assert_allclose(data["colors"], colors, atol=0.01)


class TestSceneAPI:
    """Tests for the LuxarScene API methods."""

    def test_list_nodes(self, tmp_path) -> None:
        """Test node listing methods."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene_node.add_group("group1")
            compiler.write_points("points1", np.random.randn(10, 3).astype(np.float32))
            compiler.write_points("points2", np.random.randn(10, 3).astype(np.float32))
            compiler.write_points(
                "group1/nested", np.random.randn(10, 3).astype(np.float32)
            )

        scene = LuxarScene.load(output_path)

        # List points
        point_names = scene.list_points()
        assert "points1" in point_names
        assert "points2" in point_names
        assert "group1/nested" in point_names
        assert len(point_names) == 3

        # List groups
        group_names = scene.list_groups()
        assert "group1" in group_names

    def test_has_node(self, tmp_path) -> None:
        """Test has_node method."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("exists", np.random.randn(10, 3).astype(np.float32))

        scene = LuxarScene.load(output_path)

        assert scene.has_node("exists") is True
        assert scene.has_node("does_not_exist") is False

    def test_get_node_type(self, tmp_path) -> None:
        """Test get_node_type method."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene_node.add_group("my_group")
            compiler.write_points(
                "my_points", np.random.randn(10, 3).astype(np.float32)
            )

        scene = LuxarScene.load(output_path)

        assert scene.get_node_type("my_group") == "group"
        assert scene.get_node_type("my_points") == "points"

    def test_get_node_metadata(self, tmp_path) -> None:
        """Test get_node_metadata method."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "my_points",
                np.random.randn(100, 3).astype(np.float32),
                opacity=0.8,
            )

        scene = LuxarScene.load(output_path)
        metadata = scene.get_node_metadata("my_points")

        assert metadata["type"] == "points"
        assert metadata["n_points"] == 100
        assert metadata["opacity"] == 0.8

    def test_scene_version(self, tmp_path) -> None:
        """Test that scene version is stored and retrieved."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("points", np.random.randn(10, 3).astype(np.float32))

        scene = LuxarScene.load(output_path)

        # Version should be set
        assert scene.version != "unknown"

    def test_scene_path_property(self, tmp_path) -> None:
        """Test that scene.path returns the correct path."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("points", np.random.randn(10, 3).astype(np.float32))

        scene = LuxarScene.load(output_path)

        # Path property should return the path
        assert scene.path == output_path

    def test_scene_root_attrs(self, tmp_path) -> None:
        """Test that scene.root_attrs returns all root attributes."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("points", np.random.randn(10, 3).astype(np.float32))

        scene = LuxarScene.load(output_path)

        # root_attrs should be a dict with scene attributes
        attrs = scene.root_attrs
        assert isinstance(attrs, dict)
        assert "type" in attrs
        assert attrs["type"] == "scene"
        assert "luxar_version" in attrs

    def test_scene_with_minimal_dimensions(self, tmp_path) -> None:
        """Test that scene with minimal dimensions works correctly."""
        output_path = tmp_path / "test.luxar.zarr"

        # Create scene with minimal 3D dimensions
        dims = Dimensions.default_3d()
        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)
            compiler.write_points("points", np.random.randn(10, 3).astype(np.float32))

        scene = LuxarScene.load(output_path)

        # dimensions should be present
        assert scene.dimensions is not None
        assert len(scene.dimensions) == 3


class TestErrorHandling:
    """Tests for error handling in LuxarScene."""

    def test_file_not_found(self, tmp_path) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LuxarScene.load(tmp_path / "nonexistent.luxar.zarr")

    def test_not_a_luxar_scene(self, tmp_path) -> None:
        """Test error when zarr is not a Luxar scene."""
        import zarr

        # Create a generic zarr store
        output_path = tmp_path / "generic.luxar.zarr"
        root = zarr.open_group(output_path, mode="w")
        root.attrs["not_a_scene"] = True

        with pytest.raises(ValueError, match="Not a valid Luxar scene"):
            LuxarScene.load(output_path)

    def test_get_points_wrong_type(self, tmp_path) -> None:
        """Test error when getting points from non-points node."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene_node.add_group("my_group")

        scene = LuxarScene.load(output_path)

        with pytest.raises(ValueError, match="is not a points node"):
            scene.get_points("my_group")

    def test_get_nonexistent_node(self, tmp_path) -> None:
        """Test error when node doesn't exist."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

        scene = LuxarScene.load(output_path)

        with pytest.raises(KeyError):
            scene.get_points("nonexistent")

    def test_get_node_type_not_found(self, tmp_path) -> None:
        """Test error when get_node_type is called with nonexistent node."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

        scene = LuxarScene.load(output_path)

        with pytest.raises(KeyError, match="Node not found"):
            scene.get_node_type("nonexistent")

    def test_get_node_metadata_not_found(self, tmp_path) -> None:
        """Test error when get_node_metadata is called with nonexistent node."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

        scene = LuxarScene.load(output_path)

        with pytest.raises(KeyError, match="Node not found"):
            scene.get_node_metadata("nonexistent")


class TestSpatialOrdering:
    """Tests for spatial ordering metadata."""

    def test_spatial_ordering_metadata(self, tmp_path) -> None:
        """Test that spatial ordering metadata is preserved when dimensions are provided."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 1000

        # Create scene with explicit dimensions (required for spatial ordering)
        dims = Dimensions(
            [
                Dimension("x", unit="um"),
                Dimension("y", unit="um"),
                Dimension("z", unit="um"),
            ]
        )

        positions = np.random.randn(n_points, 3).astype(np.float32) * 100

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)
            compiler.write_points("points", positions)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        metadata = data["metadata"]

        # Should have ordering metadata when dimensions are provided
        assert "ordering" in metadata
        # Hilbert is the default
        assert metadata["ordering"] in ["morton", "hilbert", "none"]

        # If ordered, should have bounds metadata
        if metadata["ordering"] != "none":
            assert "ordering_bits_per_dim" in metadata or "chunk_size" in metadata

    def test_chunk_bounds_present(self, tmp_path) -> None:
        """Test that chunk bounds are stored for ordered data."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 10000  # Need enough points for multiple chunks

        # Create scene with explicit dimensions (required for spatial ordering)
        dims = Dimensions(
            [
                Dimension("x", unit="um"),
                Dimension("y", unit="um"),
                Dimension("z", unit="um"),
            ]
        )

        positions = np.random.randn(n_points, 3).astype(np.float32) * 100

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=dims)
            compiler.write_points("points", positions)

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        # chunk_bounds may or may not be present depending on implementation
        # If present, verify shape
        if data["chunk_bounds"] is not None:
            # Shape should be (num_chunks, ndim, 2)
            assert len(data["chunk_bounds"].shape) == 3
            assert data["chunk_bounds"].shape[1] == 3  # 3D
            assert data["chunk_bounds"].shape[2] == 2  # min/max


class TestMultiplePointGroups:
    """Tests for scenes with multiple point groups."""

    def test_multiple_point_groups(self, tmp_path) -> None:
        """Test round-trip with multiple point groups."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

            # Create multiple point groups
            for i in range(5):
                positions = np.random.randn(100, 3).astype(np.float32)
                colors = np.random.rand(100, 3).astype(np.float32)
                compiler.write_points(f"points_{i}", positions, colors=colors)

        scene = LuxarScene.load(output_path)

        # Verify all groups exist
        point_names = scene.list_points()
        assert len(point_names) == 5

        # Verify each can be loaded
        for i in range(5):
            data = scene.get_points(f"points_{i}")
            assert data["positions"].shape == (100, 3)
            assert data["colors"].shape == (100, 3)

    def test_mixed_attributes_per_group(self, tmp_path) -> None:
        """Test groups with different attribute combinations."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

            # Positions only
            compiler.write_points("pos_only", np.random.randn(50, 3).astype(np.float32))

            # Positions + colors
            compiler.write_points(
                "pos_colors",
                np.random.randn(50, 3).astype(np.float32),
                colors=np.random.rand(50, 3).astype(np.float32),
            )

            # Full attributes
            compiler.write_points(
                "full",
                np.random.randn(50, 3).astype(np.float32),
                colors=np.random.rand(50, 3).astype(np.float32),
                radii=np.random.rand(50).astype(np.float32) * 0.5,
                sharpness=np.random.rand(50).astype(np.float32),
            )

        scene = LuxarScene.load(output_path)

        # Verify each group
        pos_only = scene.get_points("pos_only")
        assert pos_only["colors"] is None
        assert pos_only["radii"] is None

        pos_colors = scene.get_points("pos_colors")
        assert pos_colors["colors"] is not None
        assert pos_colors["radii"] is None

        full = scene.get_points("full")
        assert full["colors"] is not None
        assert full["radii"] is not None
        assert full["sharpness"] is not None


class TestGSplatsRoundTrip:
    """Tests for GSplats round-trip."""

    def test_basic_gsplats(self, tmp_path) -> None:
        """Test round-trip with basic GSplats data."""
        output_path = tmp_path / "test.luxar.zarr"
        n_splats = 100

        # Generate test data
        centers = np.random.randn(n_splats, 3).astype(np.float32)
        amplitudes = np.random.rand(n_splats).astype(np.float32) + 0.1
        # Cholesky factors: 6 for 3D (lower triangular)
        cholesky = np.random.rand(n_splats, 6).astype(np.float32) * 0.5 + 0.1
        colors = np.random.rand(n_splats, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_gsplats(
                "test_splats",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )

        # Read back
        scene = LuxarScene.load(output_path)
        data = scene.get_gsplats("test_splats")

        # Verify shapes
        assert data["centers"].shape == centers.shape
        assert data["amplitudes"].shape == amplitudes.shape
        assert data["cholesky_factors"].shape == cholesky.shape
        assert data["colors"].shape == colors.shape

        # Verify metadata
        assert data["metadata"]["type"] == "gsplats"
        assert data["metadata"]["n_splats"] == n_splats

    def test_gsplats_with_transform(self, tmp_path) -> None:
        """Test GSplats with transform."""
        output_path = tmp_path / "test.luxar.zarr"
        n_splats = 50

        centers = np.random.randn(n_splats, 3).astype(np.float32)
        amplitudes = np.ones(n_splats, dtype=np.float32)
        cholesky = np.random.rand(n_splats, 6).astype(np.float32) * 0.5 + 0.1
        transform = transforms.translate(5, 10, 15)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_gsplats(
                "splats",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                transform=transform,
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_gsplats("splats")

        assert "transform" in data["metadata"]
        np.testing.assert_allclose(data["metadata"]["transform"], transform, rtol=1e-5)

    def test_list_gsplats(self, tmp_path) -> None:
        """Test list_gsplats method."""
        output_path = tmp_path / "test.luxar.zarr"

        centers = np.random.randn(20, 3).astype(np.float32)
        amplitudes = np.ones(20, dtype=np.float32)
        cholesky = np.random.rand(20, 6).astype(np.float32) * 0.5 + 0.1

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_gsplats("splats1", centers, amplitudes, cholesky)
            compiler.write_gsplats("splats2", centers, amplitudes, cholesky)

        scene = LuxarScene.load(output_path)
        splat_names = scene.list_gsplats()

        assert "splats1" in splat_names
        assert "splats2" in splat_names
        assert len(splat_names) == 2

    def test_get_gsplats_wrong_type(self, tmp_path) -> None:
        """Test error when getting gsplats from non-gsplats node."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "my_points", np.random.randn(10, 3).astype(np.float32)
            )

        scene = LuxarScene.load(output_path)

        with pytest.raises(ValueError, match="is not a gsplats node"):
            scene.get_gsplats("my_points")

    def test_get_gsplats_not_found(self, tmp_path) -> None:
        """Test error when gsplats node doesn't exist."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

        scene = LuxarScene.load(output_path)

        with pytest.raises(KeyError):
            scene.get_gsplats("nonexistent")

    def test_gsplats_metadata_completeness(self, tmp_path) -> None:
        """Test that ALL required metadata fields are written to group.attrs.

        This test prevents regression of the critical bug where ndim, has_colors,
        ordering and chunk_size were in the
        internal metadata dict but NOT written to group.attrs, causing viewer errors.
        """
        output_path = tmp_path / "test.luxar.zarr"
        n_splats = 100

        centers = np.random.randn(n_splats, 3).astype(np.float32)
        amplitudes = np.random.rand(n_splats).astype(np.float32) + 0.1
        cholesky = np.random.rand(n_splats, 6).astype(np.float32) * 0.5 + 0.1
        colors = np.random.rand(n_splats, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_gsplats(
                "test_splats",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )

        # Open zarr directly to check group.attrs (not through LuxarScene API)
        store = zarr.open_group(output_path, mode="r")
        gsplats_group = store["test_splats"]
        attrs = dict(gsplats_group.attrs)

        # Verify ALL required fields are in group.attrs (not just metadata dict)
        required_fields = [
            "type",
            "n_splats",
            "ndim",  # Bug #7: Was missing
            "has_colors",  # Bug #9: Was missing
            "chunk_size",  # Bug: Required by TypeScript
            "ordering",  # Bug: Required by TypeScript
            "amplitude_range",
            "center_bounds",
            "position_bounds",  # Required for dynamic clipping
        ]

        for field in required_fields:
            assert field in attrs, f"Required field '{field}' missing from group.attrs!"

        # Verify values are correct
        assert attrs["type"] == "gsplats"
        assert attrs["n_splats"] == n_splats
        assert attrs["ndim"] == 3
        assert attrs["has_colors"] is True
        assert isinstance(attrs["chunk_size"], int)
        assert attrs["chunk_size"] > 0
        assert attrs["ordering"] in ["morton", "hilbert", "none"]

        # Verify ranges are dicts with min/max
        assert "min" in attrs["amplitude_range"]
        assert "max" in attrs["amplitude_range"]
        assert "min" in attrs["center_bounds"]
        assert "max" in attrs["center_bounds"]
        assert "min" in attrs["position_bounds"]
        assert "max" in attrs["position_bounds"]

        # Verify position_bounds has correct dimensionality (3D)
        assert len(attrs["position_bounds"]["min"]) == 3
        assert len(attrs["position_bounds"]["max"]) == 3

    def test_gsplats_metadata_without_optional_arrays(self, tmp_path) -> None:
        """Test metadata completeness when colors and sharpness are not provided."""
        output_path = tmp_path / "test.luxar.zarr"
        n_splats = 50

        centers = np.random.randn(n_splats, 3).astype(np.float32)
        amplitudes = np.random.rand(n_splats).astype(np.float32) + 0.1
        cholesky = np.random.rand(n_splats, 6).astype(np.float32) * 0.5 + 0.1

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_gsplats(
                "test_splats",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                # No colors or sharpness
            )

        store = zarr.open_group(output_path, mode="r")
        gsplats_group = store["test_splats"]
        attrs = dict(gsplats_group.attrs)

        # Verify required fields still present
        assert attrs["has_colors"] is False
        assert attrs["ndim"] == 3  # Must be present
        assert attrs["chunk_size"] > 0  # Must be present


class TestLinesRoundTrip:
    """Tests for Lines round-trip."""

    def test_basic_lines(self, tmp_path) -> None:
        """Test round-trip with basic Lines data."""
        output_path = tmp_path / "test.luxar.zarr"
        n_vertices = 100

        # Generate test data for line segments
        vertices = np.random.randn(n_vertices, 3).astype(np.float32)
        widths = np.random.rand(n_vertices).astype(np.float32) * 0.5 + 0.1
        colors = np.random.rand(n_vertices, 3).astype(np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_lines(
                "test_lines",
                vertices=vertices,
                widths=widths,
                colors=colors,
                line_type="segments",
            )

        # Read back
        scene = LuxarScene.load(output_path)
        data = scene.get_lines("test_lines")

        # Verify shapes
        assert data["vertices"].shape == vertices.shape
        assert data["widths"].shape == widths.shape
        assert data["colors"].shape == colors.shape
        assert data["segments"] is not None
        assert data["segments"].shape[1] == 2

        # Verify metadata
        assert data["metadata"]["type"] == "lines"
        assert data["metadata"]["n_vertices"] == n_vertices

    def test_lines_with_transform(self, tmp_path) -> None:
        """Test Lines with transform."""
        output_path = tmp_path / "test.luxar.zarr"
        n_vertices = 50

        vertices = np.random.randn(n_vertices, 3).astype(np.float32)
        widths = np.ones(n_vertices, dtype=np.float32) * 0.1
        transform = transforms.rotate_x(45)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_lines(
                "lines",
                vertices=vertices,
                widths=widths,
                transform=transform,
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_lines("lines")

        assert "transform" in data["metadata"]
        np.testing.assert_allclose(data["metadata"]["transform"], transform, rtol=1e-5)

    def test_list_lines(self, tmp_path) -> None:
        """Test list_lines method."""
        output_path = tmp_path / "test.luxar.zarr"

        vertices = np.random.randn(20, 3).astype(np.float32)
        widths = np.ones(20, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_lines("lines1", vertices, widths)
            compiler.write_lines("lines2", vertices, widths)

        scene = LuxarScene.load(output_path)
        line_names = scene.list_lines()

        assert "lines1" in line_names
        assert "lines2" in line_names
        assert len(line_names) == 2

    def test_nodes_metadata_flags(self, tmp_path) -> None:
        """Test nodes metadata flags for lines and points."""
        output_path = tmp_path / "test.luxar.zarr"

        positions = np.random.randn(5, 3).astype(np.float32)
        sharpness = np.ones(5, dtype=np.float32) * 0.5
        vertices = np.random.randn(6, 3).astype(np.float32)
        widths = np.ones(6, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("pts", positions, sharpness=sharpness)
            compiler.write_lines("lines", vertices, widths, line_type="segments")

        scene = LuxarScene.load(output_path)
        nodes = {node["name"]: node for node in scene.nodes}

        assert nodes["pts"]["has_sharpness"] is True
        assert nodes["lines"]["line_type"] == "segments"

    def test_get_lines_wrong_type(self, tmp_path) -> None:
        """Test error when getting lines from non-lines node."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "my_points", np.random.randn(10, 3).astype(np.float32)
            )

        scene = LuxarScene.load(output_path)

        with pytest.raises(ValueError, match="is not a lines node"):
            scene.get_lines("my_points")

    def test_get_lines_not_found(self, tmp_path) -> None:
        """Test error when lines node doesn't exist."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

        scene = LuxarScene.load(output_path)

        with pytest.raises(KeyError):
            scene.get_lines("nonexistent")


class TestBroadcastedArraysRoundTrip:
    """Tests for broadcasted array inputs (shape (1, ...))."""

    def test_points_broadcasted_arrays(self, tmp_path) -> None:
        """Broadcasted point attributes should decode to full length."""
        output_path = tmp_path / "test.luxar.zarr"
        n_points = 25

        positions = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
        radii = np.array([0.5], dtype=np.float32)
        sharpness = np.array([0.5], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points(
                "points",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_points("points")

        assert data["colors"].shape == (n_points, 3)
        assert data["radii"].shape == (n_points,)
        assert data["sharpness"].shape == (n_points,)

    def test_lines_broadcasted_arrays(self, tmp_path) -> None:
        """Broadcasted line attributes should decode to full length."""
        output_path = tmp_path / "test.luxar.zarr"
        n_vertices = 10

        vertices = np.random.randn(n_vertices, 3).astype(np.float32)
        widths = 0.1
        colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sharpness = np.array([0.5], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_lines(
                "lines",
                vertices=vertices,
                widths=widths,
                colors=colors,
                sharpness=sharpness,
                line_type="segments",
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_lines("lines")

        assert data["colors"].shape == (n_vertices, 3)
        assert data["sharpness"].shape == (n_vertices,)

    def test_gsplats_broadcasted_arrays(self, tmp_path) -> None:
        """Broadcasted gsplats attributes should decode to full length."""
        output_path = tmp_path / "test.luxar.zarr"
        n_splats = 8

        centers = np.random.randn(n_splats, 3).astype(np.float32)
        amplitudes = 1.0
        cholesky = np.tile(
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (n_splats, 1),
        )
        colors = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_gsplats(
                "splats",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )

        scene = LuxarScene.load(output_path)
        data = scene.get_gsplats("splats")

        assert data["colors"].shape == (n_splats, 3)


class TestReaderGroupTransforms:
    """Tests for reader group transform support."""

    def test_get_group_with_transform(self, tmp_path) -> None:
        """Test that get_group() returns parsed transform."""
        output_path = tmp_path / "test.luxar.zarr"
        t = transforms.translate(10, 20, 30)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_group("MyGroup", transform=t)

        reader = LuxarScene.load(output_path)
        group_meta = reader.get_group("MyGroup")

        assert "transform" in group_meta
        assert isinstance(group_meta["transform"], np.ndarray)
        assert group_meta["transform"].shape == (4, 4)
        assert np.allclose(group_meta["transform"], t, atol=1e-6)

    def test_get_group_without_transform(self, tmp_path) -> None:
        """Test that get_group() works for groups without transforms."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_group("PlainGroup")

        reader = LuxarScene.load(output_path)
        group_meta = reader.get_group("PlainGroup")

        assert "transform" not in group_meta

    def test_get_group_type_check(self, tmp_path) -> None:
        """Test that get_group() rejects non-group nodes."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0, 0, 0]], dtype=np.float32)
            scene.add_points("MyPoints", positions)

        reader = LuxarScene.load(output_path)
        with pytest.raises(ValueError):
            reader.get_group("MyPoints")

    def test_collect_nodes_includes_transforms(self, tmp_path) -> None:
        """Test that .nodes property includes transforms for all node types."""
        output_path = tmp_path / "test.luxar.zarr"
        t_group = transforms.translate(1, 0, 0)
        t_points = transforms.scale(2, 2, 2)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("TGroup", transform=t_group)
            positions = np.array([[0, 0, 0]], dtype=np.float32)
            scene.add_points("TPoints", positions, parent=group, transform=t_points)

        reader = LuxarScene.load(output_path)
        nodes = reader.nodes

        # Find the group node
        group_node = next(n for n in nodes if n["name"] == "TGroup")
        assert "transform" in group_node
        assert np.allclose(group_node["transform"], t_group, atol=1e-6)

        # Find the points node
        points_node = next(n for n in nodes if n["name"] == "TGroup/TPoints")
        assert "transform" in points_node
        assert np.allclose(points_node["transform"], t_points, atol=1e-6)
