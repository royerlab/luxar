"""Tests for Group class with add_* methods."""

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.node import Node
from luxar.io.compiler import LuxarZarrCompiler


class TestGroupAddData:
    """Test that Group nodes can add data children."""

    def test_group_add_points(self, tmp_path) -> None:
        """Test group.add_points() writes data under group path."""
        output_path = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            pts = group.add_points("pts", positions)

            assert pts.n_elements == 2
            assert pts.parent == group

        # Verify zarr hierarchy
        store = zarr.open(str(output_path), mode="r")
        assert "grp" in store
        assert "pts" in store["grp"]
        assert store["grp"]["pts"].attrs.get("type") == "points"

    def test_group_add_lines(self, tmp_path) -> None:
        """Test group.add_lines() writes data under group path."""
        output_path = tmp_path / "test.zarr"
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            lines = group.add_lines("lines", vertices, widths=0.1)

            assert lines.n_elements == 2

        store = zarr.open(str(output_path), mode="r")
        assert "lines" in store["grp"]

    def test_group_add_gsplats(self, tmp_path) -> None:
        """Test group.add_gsplats() writes data under group path."""
        output_path = tmp_path / "test.zarr"
        centers = np.array([[1, 2, 3]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            gsplats = group.add_gsplats("splats", centers, amplitudes, cholesky)

            assert gsplats.n_splats == 1

        store = zarr.open(str(output_path), mode="r")
        assert "splats" in store["grp"]
        assert store["grp"]["splats"].attrs.get("type") == "gsplats"

    def test_nested_groups(self, tmp_path) -> None:
        """Test nested groups: group.add_group('sub').add_points(...)."""
        output_path = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("level1")
            sub = group.add_group("level2")
            sub.add_points("pts", positions)

        store = zarr.open(str(output_path), mode="r")
        assert "pts" in store["level1"]["level2"]

    def test_dimension_validation_through_group(self, tmp_path) -> None:
        """Test that dimension validation works when adding data to a group."""
        output_path = tmp_path / "test.zarr"
        positions_2d = np.array([[1, 2]], dtype=np.float32)  # 2D, but scene is 3D

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")

            with pytest.raises(ValueError, match="Dimension mismatch"):
                group.add_points("bad", positions_2d)


class TestGroupNotAttachedToScene:
    """Test error when Group is not part of a Scene hierarchy."""

    def test_detached_group_raises(self) -> None:
        """Group not attached to Scene should raise on add_*."""
        group = Group("orphan")
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with pytest.raises(ValueError, match="not attached to a Scene"):
            group.add_points("pts", positions)


class TestBackwardCompatibility:
    """Test that scene.add_points(..., parent=group) still works."""

    def test_scene_add_points_with_parent(self, tmp_path) -> None:
        """Old pattern: scene.add_points('name', data, parent=group)."""
        output_path = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            pts = scene.add_points("pts", positions, parent=group)

            assert pts.parent == group

        store = zarr.open(str(output_path), mode="r")
        assert "pts" in store["grp"]

    def test_scene_add_gsplats_with_parent(self, tmp_path) -> None:
        """Old pattern: scene.add_gsplats('name', ..., parent=group)."""
        output_path = tmp_path / "test.zarr"
        centers = np.array([[1, 2, 3]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            gsplats = scene.add_gsplats(
                "splats", centers, amplitudes, cholesky, parent=group
            )

            assert gsplats.parent == group


class TestGroupIsGroup:
    """Test that add_group() returns Group instances."""

    def test_add_group_returns_group(self, tmp_path) -> None:
        output_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")

            assert isinstance(group, Group)
            assert isinstance(group, Node)  # Group is also a Node
            assert hasattr(group, "add_points")
            assert hasattr(group, "add_lines")
            assert hasattr(group, "add_gsplats")
            assert hasattr(group, "add_gsplats_from_data")
