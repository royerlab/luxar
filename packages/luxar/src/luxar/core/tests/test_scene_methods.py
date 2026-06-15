"""Tests for Scene class methods not covered elsewhere."""

import warnings

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


class TestSceneMethods:
    """Test Scene class methods."""

    def test_scene_str_representation(self, tmp_path) -> None:
        """Test Scene __str__ method."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            str_repr = str(scene)
            assert "Scene" in str_repr
            assert "0 children" in str_repr

            # Add some groups
            scene.add_group("group1")
            scene.add_group("group2")
            str_repr = str(scene)
            assert "2 children" in str_repr

    def test_scene_finalize_methods(self, tmp_path) -> None:
        """Test Scene finalize and related methods."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add some points
            positions = np.random.randn(100, 3).astype(np.float32)
            scene.add_points("points", positions)

            # Finalize is called automatically by context manager

        # Check the zarr was properly finalized
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert ".zmetadata" in store.store
        assert "points" in store

    def test_scene_dimensions_always_set(self, tmp_path) -> None:
        """Test that Scene always has dimensions from creation."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Scene must have dimensions from creation
            assert scene.dimensions is not None
            assert len(scene.dimensions.dimensions) == 3

            # Add matching points - should work
            positions = np.random.randn(100, 3).astype(np.float32)
            scene.add_points("points", positions)

            # Dimensions unchanged - still 3D
            assert scene.dimensions is not None
            assert len(scene.dimensions.dimensions) == 3

    def test_dimension_mismatch_error_add_points(self, tmp_path) -> None:
        """Test that adding points with wrong dimensionality raises error."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # 5D positions to 3D scene should fail
            positions_5d = np.random.randn(100, 5).astype(np.float32)
            with pytest.raises(ValueError, match="Dimension mismatch"):
                scene.add_points("bad_points", positions_5d)

            # 2D positions to 3D scene should also fail
            positions_2d = np.random.randn(100, 2).astype(np.float32)
            with pytest.raises(ValueError, match="Dimension mismatch"):
                scene.add_points("bad_points_2d", positions_2d)

    def test_dimension_mismatch_error_add_lines(self, tmp_path) -> None:
        """Test that adding lines with wrong dimensionality raises error."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # 4D vertices to 3D scene should fail
            vertices_4d = np.random.randn(100, 4).astype(np.float32)
            with pytest.raises(ValueError, match="Dimension mismatch"):
                scene.add_lines("bad_lines", vertices_4d, widths=0.1)

    def test_dimension_mismatch_error_add_gsplats(self, tmp_path) -> None:
        """Test that adding gsplats with wrong dimensionality raises error."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # 4D centers to 3D scene should fail
            centers_4d = np.random.randn(100, 4).astype(np.float32)
            # Cholesky factors for 4D: k = 4*(4+1)/2 = 10
            cholesky_4d = np.random.randn(100, 10).astype(np.float32)
            with pytest.raises(ValueError, match="Dimension mismatch"):
                scene.add_gsplats(
                    "bad_gsplats",
                    centers_4d,
                    amplitudes=1.0,
                    cholesky_factors=cholesky_4d,
                )

    def test_dimension_range_warning(self, tmp_path) -> None:
        """Test that values outside declared range produce a warning."""
        dims = Dimensions(
            [
                Dimension("x", range=(0, 100)),
                Dimension("y", range=(0, 100)),
                Dimension("z", range=(0, 100)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Positions outside declared range should produce warning
            positions = np.array(
                [
                    [50, 50, 50],  # Within range
                    [150, 50, 50],  # x outside range
                    [-10, 50, 50],  # x outside range (negative)
                ],
                dtype=np.float32,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                scene.add_points("out_of_range", positions)

                # Should have warning about x dimension
                range_warnings = [
                    warning
                    for warning in w
                    if "outside declared range" in str(warning.message)
                ]
                assert len(range_warnings) >= 1
                assert "x" in str(range_warnings[0].message)

    def test_dimension_range_no_warning_when_within(self, tmp_path) -> None:
        """Test that values within declared range produce no warning."""
        dims = Dimensions(
            [
                Dimension("x", range=(0, 100)),
                Dimension("y", range=(0, 100)),
                Dimension("z", range=(0, 100)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Positions within declared range should not produce warning
            positions = np.array(
                [
                    [0, 0, 0],
                    [50, 50, 50],
                    [100, 100, 100],
                ],
                dtype=np.float32,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                scene.add_points("in_range", positions)

                # Should have no warnings about range
                range_warnings = [
                    warning
                    for warning in w
                    if "outside declared range" in str(warning.message)
                ]
                assert len(range_warnings) == 0

    def test_dimension_validation_helpful_error_message(self, tmp_path) -> None:
        """Test that dimension mismatch error has helpful message."""
        dims = Dimensions(
            [
                Dimension("time", display=False),
                Dimension("x"),
                Dimension("y"),
                Dimension("z"),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # 3D positions to 4D scene should fail with helpful message
            positions_3d = np.random.randn(100, 3).astype(np.float32)
            with pytest.raises(ValueError) as exc_info:
                scene.add_points("bad", positions_3d)

            # Check error message contains useful info
            error_msg = str(exc_info.value)
            assert "3 columns" in error_msg  # What we got
            assert "4 dimensions" in error_msg  # What we expected
            assert "time" in error_msg  # Dimension names
            assert "(N, 4)" in error_msg  # Expected shape

    # Test removed: _apply_dimension_metadata is internal implementation
    # Metadata application is now handled automatically during scene creation

    def test_scene_writer_access(self, tmp_path) -> None:
        """Test Scene has access to writer."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Scene should have writer
            assert scene._writer is not None
            assert scene._writer is compiler

    def test_scene_add_points_error_handling(self, tmp_path) -> None:
        """Test Scene.add_points error handling."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Test with invalid positions (1D array)
            with pytest.raises(ValueError, match="Could not add points"):
                scene.add_points("bad", np.array([1, 2, 3]))

            # Test with 3D positions array
            with pytest.raises(ValueError, match="Positions must have shape"):
                scene.add_points("bad", np.zeros((10, 10, 3)))

    def test_scene_add_group_with_transform(self, tmp_path) -> None:
        """Test Scene.add_group with transform."""
        from luxar import transforms

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Create transform
            transform = transforms.translate(1, 2, 3)

            # Add group with transform
            group = scene.add_group("transformed", transform=transform)

            # Check transform was stored
            assert group.transform is not None
            np.testing.assert_array_almost_equal(group.transform, transform)

    def test_scene_add_group_with_rendering_attrs(self, tmp_path) -> None:
        """Test Scene.add_group with rendering attributes."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add group with rendering attributes
            group = scene.add_group(
                "rendered", opacity=0.5, gamma=1.5, blending_mode="additive"
            )

            # Check attributes were set
            assert group.opacity == 0.5
            assert group.gamma == 1.5
            assert group.blending_mode == "additive"

    def test_scene_context_manager_exception(self, tmp_path):
        """Test Scene handles exceptions in context manager."""
        try:
            with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("points", np.random.randn(10, 3).astype(np.float32))
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected

        # Scene should still be finalized
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "points" in store

    def test_points_metadata_preservation(self, tmp_path) -> None:
        """Test that Points object preserves metadata from write_points().

        This test catches the bug where Node.__init__() would overwrite
        Points._metadata by initializing it after Points had set it.

        Critical metadata to verify:
        - has_colors, has_radii, has_sharpness (boolean flags)
        - max_radius (float value)
        - n_points, dims (shape info)
        """
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Create points with all attributes
            n_points = 100
            positions = np.random.randn(n_points, 3).astype(np.float32)
            colors = np.random.rand(n_points, 3).astype(np.float32)
            radii = np.ones(n_points, dtype=np.float32) * 0.15
            sharpness = np.ones(n_points, dtype=np.float32) * 0.5

            points = scene.add_points(
                "test_points",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
            )

            # Verify all metadata is preserved
            assert points.has_colors, "has_colors should be True"
            assert points.has_radii, "has_radii should be True"
            assert points.has_sharpness, "has_sharpness should be True"

            # Verify metadata dict has all expected keys
            assert "n_points" in points.metadata, "n_points missing from metadata"
            assert "ndim" in points.metadata, "ndim missing from metadata"
            assert "has_colors" in points.metadata, "has_colors missing"
            assert "has_radii" in points.metadata, "has_radii missing"
            assert "has_sharpness" in points.metadata, "has_sharpness missing"
            assert "max_radius" in points.metadata, "max_radius missing"

            # Verify values
            assert points.n_points == n_points
            assert points.metadata["ndim"] == 3
            assert 0.14 < points.metadata["max_radius"] < 0.16, "max_radius incorrect"

    def test_points_metadata_without_optional_attributes(self, tmp_path) -> None:
        """Test Points metadata when only positions are provided.

        Note: Radii now have a default value of 0.5, so has_radii is always True.
        """
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            positions = np.random.randn(50, 3).astype(np.float32)
            points = scene.add_points("minimal", positions)

            # Should have basic metadata
            assert points.n_points == 50
            assert points.metadata["ndim"] == 3

            # Colors and sharpness are not provided
            assert not points.has_colors
            assert not points.has_sharpness

            # Radii are auto-assigned with default value 0.5
            assert points.has_radii
            assert points.metadata["max_radius"] == 0.5

            # Metadata dict should still exist and have required keys
            assert "n_points" in points.metadata
            assert "has_colors" in points.metadata
            assert not points.metadata["has_colors"]
