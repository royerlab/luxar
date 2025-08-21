"""Tests for Scene class methods not covered elsewhere."""

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


class TestSceneMethods:
    """Test Scene class methods."""

    def test_scene_str_representation(self, tmp_path):
        """Test Scene __str__ method."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()
            str_repr = str(scene)
            assert "Scene" in str_repr
            assert "0 children" in str_repr

            # Add some groups
            scene.add_group("group1")
            scene.add_group("group2")
            str_repr = str(scene)
            assert "2 children" in str_repr

    def test_scene_finalize_methods(self, tmp_path):
        """Test Scene finalize and related methods."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Add some points
            positions = np.random.randn(100, 3).astype(np.float32)
            scene.add_points("points", positions)

            # Finalize is called automatically by context manager

        # Check the zarr was properly finalized
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert ".zmetadata" in store.store
        assert "points" in store

    @pytest.mark.skip(reason="groups property removed in new API")
    def test_scene_groups_property(self, tmp_path):
        """Test Scene.groups property."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Initially empty
            assert len(scene.groups) == 0

            # Add groups
            g1 = scene.add_group("group1")
            g2 = scene.add_group("group2")

            # Check groups property
            assert len(scene.groups) == 2
            assert "group1" in scene.groups
            assert "group2" in scene.groups
            assert scene.groups["group1"] is g1
            assert scene.groups["group2"] is g2

    @pytest.mark.skip(reason="_validate_scene_dimensions removed in new API")
    def test_scene_validate_scene_dimensions(self, tmp_path):
        """Test Scene._validate_scene_dimensions method."""
        dims = Dimensions(
            [
                Dimension("x", unit="um"),
                Dimension("y", unit="um"),
                Dimension("z", unit="um"),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Access internal method
            scene._validate_scene_dimensions()  # Should not raise

            # Test with invalid dimensions attribute
            scene.dimensions = None
            scene._validate_scene_dimensions()  # Should handle gracefully

    def test_scene_infer_dimensions_from_points(self, tmp_path):
        """Test Scene._infer_dimensions_from_points method."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Initially no dimensions
            assert scene.dimensions is None

            # Add points - dimensions should be inferred (but we disabled this)
            positions = np.random.randn(100, 5).astype(np.float32)
            scene.add_points("points", positions)

            # Since we disabled inference, dimensions should still be None
            assert scene.dimensions is None

    @pytest.mark.skip(reason="_apply_dimension_metadata is internal implementation")
    def test_scene_apply_dimension_metadata(self, tmp_path):
        """Test Scene._apply_dimension_metadata method."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Create attrs dict
            attrs = {}

            # Test with None metadata
            scene._apply_dimension_metadata(attrs, None, 3)
            # Should not add anything
            assert "dimension_metadata" not in attrs

            # Test with metadata (though this is disabled in new API)
            from luxar.types import DimensionMetadata

            metadata = [
                DimensionMetadata(name="x"),
                DimensionMetadata(name="y"),
                DimensionMetadata(name="z"),
            ]
            scene._apply_dimension_metadata(attrs, metadata, 3)
            # This is disabled, so still shouldn't add
            assert "dimension_metadata" not in attrs

    def test_scene_writer_access(self, tmp_path):
        """Test Scene has access to writer."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Scene should have writer
            assert scene._writer is not None
            assert scene._writer is compiler

    def test_scene_add_points_error_handling(self, tmp_path):
        """Test Scene.add_points error handling."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Test with invalid positions (1D array)
            with pytest.raises(ValueError, match="Could not add points"):
                scene.add_points("bad", np.array([1, 2, 3]))

            # Test with 3D positions array
            with pytest.raises(ValueError, match="Positions must have shape"):
                scene.add_points("bad", np.zeros((10, 10, 3)))

    def test_scene_add_group_with_transform(self, tmp_path):
        """Test Scene.add_group with transform."""
        from luxar import transforms

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Create transform
            transform = transforms.translate(1, 2, 3)

            # Add group with transform
            group = scene.add_group("transformed", transform=transform)

            # Check transform was stored
            assert group.transform is not None
            np.testing.assert_array_almost_equal(group.transform, transform)

    def test_scene_add_group_with_rendering_attrs(self, tmp_path):
        """Test Scene.add_group with rendering attributes."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Add group with rendering attributes
            group = scene.add_group(
                "rendered", opacity=0.5, gamma=1.5, blending_mode="additive"
            )

            # Check attributes were set
            assert group.opacity == 0.5
            assert group.gamma == 1.5
            assert group.blending_mode == "additive"

    def test_scene_finalize_already_finalized(self, tmp_path):
        """Test calling finalize multiple times."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()
            scene.add_points("points", np.random.randn(10, 3).astype(np.float32))

            # Finalize once
            scene.finalize()

            # Finalize again - should be safe
            scene.finalize()

    def test_scene_context_manager_exception(self, tmp_path):
        """Test Scene handles exceptions in context manager."""
        try:
            with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
                scene = compiler.create_scene()
                scene.add_points("points", np.random.randn(10, 3).astype(np.float32))
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected

        # Scene should still be finalized
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert "points" in store
