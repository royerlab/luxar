"""Tests for Node properties and method chaining."""

from pathlib import Path

import numpy as np
import pytest

from luxar import Dimension, Dimensions
from luxar.io.compiler import LuxarZarrCompiler


class TestNodeProperties:
    """Test Node hierarchy properties."""

    def test_num_children_property(self, tmp_path: Path) -> None:
        """Test num_children property."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # Root has no children initially
            assert scene.num_children == 0

            # Add one child
            group1 = scene.add_group("group1")
            assert scene.num_children == 1
            assert group1.num_children == 0

            # Add more children to root
            scene.add_group("group2")
            assert scene.num_children == 2

            # Add children to group1
            group1.add_group("subgroup1")
            assert group1.num_children == 1

            group1.add_group("subgroup2")
            assert group1.num_children == 2

            # Add points as children
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            scene.add_points("points1", positions)
            assert scene.num_children == 3  # group1, group2, points1

    def test_is_leaf_property(self, tmp_path: Path) -> None:
        """Test is_leaf property."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # Root with no children is a leaf
            assert scene.is_leaf

            # Add a group - root is no longer a leaf
            group1 = scene.add_group("group1")
            assert not scene.is_leaf
            assert group1.is_leaf  # Empty group is a leaf

            # Add children to group1
            group1.add_group("subgroup")
            assert not group1.is_leaf

            # Points are always leaves
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("points1", positions)
            assert points.is_leaf  # Points node is always a leaf

    def test_is_root_property(self, tmp_path: Path) -> None:
        """Test is_root property."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # Scene is root
            assert scene.is_root is True

            # Child groups are not root
            group1 = scene.add_group("group1")
            assert group1.is_root is False

            # Nested groups are not root
            subgroup = group1.add_group("subgroup")
            assert subgroup.is_root is False

            # Points are not root
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("points1", positions)
            assert points.is_root is False

    def test_properties_with_complex_hierarchy(self, tmp_path: Path) -> None:
        """Test properties with a complex hierarchy."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # Build hierarchy: scene -> group1 -> subgroup1
            #                        -> group2
            #                        -> points1
            #                        -> points2
            group1 = scene.add_group("group1")
            group2 = scene.add_group("group2")

            subgroup1 = group1.add_group("subgroup1")

            positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
            points1 = scene.add_points("points1", positions)  # add_points only on Scene
            points2 = scene.add_points("points2", positions)

            # Verify scene (root)
            assert scene.is_root is True
            assert scene.is_leaf is False
            assert scene.num_children == 4  # group1, group2, points1, points2

            # Verify group1
            assert group1.is_root is False
            assert group1.is_leaf is False  # Has subgroup1
            assert group1.num_children == 1  # subgroup1

            # Verify group2
            assert group2.is_root is False
            assert (
                group2.is_leaf is True
            )  # No children (points added to scene, not group2)
            assert group2.num_children == 0

            # Verify subgroup1
            assert subgroup1.is_root is False
            assert (
                subgroup1.is_leaf is True
            )  # No children (points added to scene, not subgroup1)
            assert subgroup1.num_children == 0

            # Verify points (leaves)
            assert points1.is_root is False
            assert points1.is_leaf is True
            assert points1.num_children == 0

            assert points2.is_root is False
            assert points2.is_leaf is True
            assert points2.num_children == 0


class TestNodeMethodChaining:
    """Test Node method chaining for rendering attributes."""

    def test_set_opacity_returns_self(self, tmp_path: Path) -> None:
        """Test that set_opacity() returns self for chaining."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # set_opacity should return the node itself
            result = group.set_opacity(0.5)
            assert result is group

            # Verify opacity was set
            assert group.attrs["opacity"] == 0.5

    def test_set_absorption_returns_self(self, tmp_path: Path) -> None:
        """Test that set_absorption() returns self for chaining."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # set_absorption should return the node itself
            result = group.set_absorption(2.5)
            assert result is group

            # Verify absorption was set
            assert group.attrs["absorption"] == 2.5

    def test_set_gamma_returns_self(self, tmp_path: Path) -> None:
        """Test that set_gamma() returns self for chaining."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # set_gamma should return the node itself
            result = group.set_gamma(1.5)
            assert result is group

            # Verify gamma was set
            assert group.attrs["gamma"] == 1.5

    def test_set_blending_mode_returns_self(self, tmp_path: Path) -> None:
        """Test that set_blending_mode() returns self for chaining."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # set_blending_mode should return the node itself
            result = group.set_blending_mode("additive")
            assert result is group

            # Verify blending mode was set
            assert group.attrs["blending_mode"] == "additive"

    def test_chain_two_methods(self, tmp_path: Path) -> None:
        """Test chaining two setter methods."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # Chain opacity and gamma
            result = group.set_opacity(0.7).set_gamma(1.2)

            # Result should still be the group
            assert result is group

            # Verify both attributes were set
            assert group.attrs["opacity"] == 0.7
            assert group.attrs["gamma"] == 1.2

    def test_chain_three_methods(self, tmp_path: Path) -> None:
        """Test chaining all three setter methods."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # Chain all three setters
            result = group.set_opacity(0.8).set_gamma(1.5).set_blending_mode("additive")

            # Result should still be the group
            assert result is group

            # Verify all attributes were set
            assert group.attrs["opacity"] == 0.8
            assert group.attrs["gamma"] == 1.5
            assert group.attrs["blending_mode"] == "additive"

    def test_chain_methods_on_multiple_nodes(self, tmp_path: Path) -> None:
        """Test chaining methods on multiple nodes independently."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group1 = scene.add_group("group1")
            group2 = scene.add_group("group2")

            # Chain on group1
            group1.set_opacity(0.5).set_gamma(1.0)

            # Chain on group2 with different values
            group2.set_opacity(0.9).set_blending_mode("normal")

            # Verify group1 attributes
            assert group1.attrs["opacity"] == 0.5
            assert group1.attrs["gamma"] == 1.0
            assert "blending_mode" not in group1.attrs

            # Verify group2 attributes
            assert group2.attrs["opacity"] == 0.9
            assert group2.attrs["blending_mode"] == "normal"
            assert "gamma" not in group2.attrs

    def test_chain_after_add_group(self, tmp_path: Path) -> None:
        """Test chaining setters immediately after add_group."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # Chain setters right after creating group
            group = scene.add_group("group1").set_opacity(0.6).set_gamma(1.8)

            # Verify attributes were set
            assert group.attrs["opacity"] == 0.6
            assert group.attrs["gamma"] == 1.8

    def test_chain_methods_on_points(self, tmp_path: Path) -> None:
        """Test that setter methods also work on Points nodes."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

            # Chain setters on points node
            points = (
                scene.add_points("points1", positions)
                .set_opacity(0.5)
                .set_blending_mode("additive")
            )

            # Verify attributes were set
            assert points.attrs["opacity"] == 0.5
            assert points.attrs["blending_mode"] == "additive"

    def test_chain_with_invalid_values_raises_error(self, tmp_path: Path) -> None:
        """Test that chaining with invalid values raises errors."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # Invalid opacity should raise ValueError
            with pytest.raises(ValueError, match="Opacity must be between"):
                group.set_opacity(1.5)  # > 1.0

            # Invalid gamma should raise ValueError
            with pytest.raises(ValueError, match="Gamma must be between"):
                group.set_gamma(0.05)  # < 0.1 (new minimum per spec)

            # Invalid blending mode should raise ValueError
            with pytest.raises(ValueError, match="Invalid blending mode"):
                group.set_blending_mode("invalid")

    def test_multiple_chains_on_same_node(self, tmp_path: Path) -> None:
        """Test multiple separate chains on the same node (updating values)."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("group1")

            # First chain
            group.set_opacity(0.5).set_gamma(1.0)
            assert group.attrs["opacity"] == 0.5
            assert group.attrs["gamma"] == 1.0

            # Second chain (updates values)
            group.set_opacity(0.8).set_blending_mode("additive")
            assert group.attrs["opacity"] == 0.8  # Updated
            assert group.attrs["gamma"] == 1.0  # Unchanged
            assert group.attrs["blending_mode"] == "additive"  # New


class TestNodeEqualityAndHashing:
    """Test Node __eq__ and __hash__."""

    def test_same_path_equal(self, tmp_path: Path) -> None:
        """Nodes with the same path should be equal."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("alpha")
            # Same object is equal to itself
            assert group == group

    def test_different_path_not_equal(self, tmp_path: Path) -> None:
        """Nodes with different paths should not be equal."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            g1 = scene.add_group("alpha")
            g2 = scene.add_group("beta")
            assert g1 != g2

    def test_node_in_set(self, tmp_path: Path) -> None:
        """Nodes should be usable in sets."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            g1 = scene.add_group("a")
            g2 = scene.add_group("b")
            node_set = {g1, g2, g1}  # duplicate g1
            assert len(node_set) == 2

    def test_not_equal_to_non_node(self, tmp_path: Path) -> None:
        """Node should not be equal to non-Node objects."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            assert scene != "Scene"
            assert scene != 42


class TestDuplicateChildNames:
    """Test that duplicate child names are rejected."""

    def test_duplicate_child_name_raises(self, tmp_path: Path) -> None:
        """Adding two children with the same name should raise ValueError."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_group("alpha")

            with pytest.raises(ValueError, match="Duplicate child name 'alpha'"):
                scene.add_group("alpha")

    def test_same_name_under_different_parents_ok(self, tmp_path: Path) -> None:
        """Same name under different parents should be allowed."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group_a = scene.add_group("A")
            group_b = scene.add_group("B")

            # Both parents can have a child named "data"
            child_a = group_a.add_group("data")
            child_b = group_b.add_group("data")

            assert child_a.name == "data"
            assert child_b.name == "data"
            assert child_a.path != child_b.path


class TestNodeWalkDirect:
    """Test Node.walk() method directly."""

    def test_walk_single_node(self, tmp_path: Path) -> None:
        """Test walk on a single node (no children)."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # Walk scene with no children
            hierarchy = list(
                scene.walk()
            )  # walk() returns generator of (depth, node) tuples

            # Should return list with just the root
            assert isinstance(hierarchy, list)
            assert len(hierarchy) == 1
            depth, node = hierarchy[0]
            assert depth == 0
            assert node.name == "Scene"  # Root is Scene node

    def test_walk_with_children(self, tmp_path: Path) -> None:
        """Test walk with children."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_group("group1")
            scene.add_group("group2")

            hierarchy = list(
                scene.walk()
            )  # walk() returns generator of (depth, node) tuples

            # Should have 3 nodes (scene + 2 groups)
            assert len(hierarchy) == 3

            # Check structure
            depth0, node0 = hierarchy[0]
            assert node0.name == "Scene"  # Root is Scene node
            assert depth0 == 0
            depth1, node1 = hierarchy[1]
            assert node1.name == "group1"
            assert depth1 == 1
            depth2, node2 = hierarchy[2]
            assert node2.name == "group2"
            assert depth2 == 1

    def test_walk_nested_hierarchy(self, tmp_path: Path) -> None:
        """Test walk with nested hierarchy."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group1 = scene.add_group("group1")
            _subgroup = group1.add_group("subgroup")  # Create hierarchy depth
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            scene.add_points("points", positions)  # add_points only on Scene

            hierarchy = list(
                scene.walk()
            )  # walk() returns generator of (depth, node) tuples

            # Should have 4 nodes (scene -> group1 -> subgroup, and points as separate child)
            assert len(hierarchy) == 4

            # Check depths - points is added to scene (depth 1), not nested
            assert hierarchy[0][0] == 0  # scene depth
            assert hierarchy[1][0] == 1  # group1 depth
            assert hierarchy[2][0] == 2  # subgroup depth
            assert hierarchy[3][0] == 1  # points depth (direct child of scene)

    def test_walk_from_non_root_node(self, tmp_path: Path) -> None:
        """Test walk starting from a non-root node."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group1 = scene.add_group("group1")
            group1.add_group("subgroup1")
            group1.add_group("subgroup2")

            # Walk from group1 instead of root
            hierarchy = list(
                group1.walk()
            )  # walk() returns generator of (depth, node) tuples

            # Should have 3 nodes (group1 + 2 subgroups)
            assert len(hierarchy) == 3
            depth0, node0 = hierarchy[0]
            assert node0.name == "group1"
            assert depth0 == 0  # Depth starts at 0 from walk start
            depth1, node1 = hierarchy[1]
            assert node1.name == "subgroup1"
            assert depth1 == 1
            depth2, node2 = hierarchy[2]
            assert node2.name == "subgroup2"
            assert depth2 == 1

    def test_walk_returns_nodes(self, tmp_path: Path) -> None:
        """Test that walk returns actual node objects."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group1 = scene.add_group("group1")
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points1 = scene.add_points("points1", positions)

            hierarchy = list(
                scene.walk()
            )  # walk() returns generator of (depth, node) tuples

            # Check that actual node objects are returned
            _, scene_node = hierarchy[0]
            assert scene_node is scene

            _, group_node = hierarchy[1]
            assert group_node is group1

            _, points_node = hierarchy[2]
            assert points_node is points1


class TestNodeNdTransform:
    """Test Node nd_transform property."""

    def test_nd_transform_set_affine(self, tmp_path: Path) -> None:
        """Test setting an affine nd_transform and reading it back."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("time", unit="s", display=False, range=(0, 100)),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)
            positions = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            nd = {"time": {"scale": 2.0, "offset": 1.0}}
            points.nd_transform = nd

            assert points.nd_transform is not None
            assert points.nd_transform["time"]["scale"] == 2.0
            assert points.nd_transform["time"]["offset"] == 1.0

    def test_nd_transform_delete(self, tmp_path: Path) -> None:
        """Test setting nd_transform then removing it with None."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            points.nd_transform = {"dim0": {"scale": 1.0, "offset": 0.0}}
            assert points.nd_transform is not None

            points.nd_transform = None
            assert points.nd_transform is None

    def test_nd_transform_invalid_raises(self, tmp_path: Path) -> None:
        """Test that invalid nd_transform raises an error."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            # Empty dict entry (no scale/offset or permutation)
            with pytest.raises(ValueError):
                points.nd_transform = {"time": {}}

            # Non-dict value
            with pytest.raises(TypeError):
                points.nd_transform = "not_a_dict"  # type: ignore[assignment]


class TestNodeIntensity:
    """Test Node intensity property."""

    def test_intensity_default(self, tmp_path: Path) -> None:
        """Test that default intensity is 1.0."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            assert points.intensity == 1.0

    def test_intensity_set_and_get(self, tmp_path: Path) -> None:
        """Test setting and getting intensity."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            points.intensity = 50.0
            assert points.intensity == 50.0

    def test_intensity_boundaries(self, tmp_path: Path) -> None:
        """Test intensity boundary values."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            # Valid boundaries
            points.intensity = 0.0
            assert points.intensity == 0.0

            points.intensity = 100.0
            assert points.intensity == 100.0

            # Invalid: below minimum
            with pytest.raises(ValueError):
                points.intensity = -1.0

            # Invalid: above maximum
            with pytest.raises(ValueError):
                points.intensity = 101.0

    def test_intensity_chaining(self, tmp_path: Path) -> None:
        """Test that set_intensity returns self for chaining."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            result = points.set_intensity(50.0).set_opacity(0.5)
            assert result is points
            assert points.intensity == 50.0
            assert points.opacity == 0.5


class TestNodeOffset:
    """Test Node offset property."""

    def test_offset_default(self, tmp_path: Path) -> None:
        """Test that default offset is 0.0."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            assert points.offset == 0.0

    def test_offset_set_and_get(self, tmp_path: Path) -> None:
        """Test setting and getting offset."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            points.offset = -5.0
            assert points.offset == -5.0

    def test_offset_boundaries(self, tmp_path: Path) -> None:
        """Test offset boundary values."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            # Valid boundaries
            points.offset = -10.0
            assert points.offset == -10.0

            points.offset = 10.0
            assert points.offset == 10.0

            # Invalid: below minimum
            with pytest.raises(ValueError):
                points.offset = -11.0

            # Invalid: above maximum
            with pytest.raises(ValueError):
                points.offset = 11.0

    def test_offset_chaining(self, tmp_path: Path) -> None:
        """Test that set_offset returns self for chaining."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            result = points.set_offset(-5.0).set_opacity(0.5)
            assert result is points
            assert points.offset == -5.0
            assert points.opacity == 0.5


class TestNodeLayer:
    """Test Node layer property."""

    def test_layer_default(self, tmp_path: Path) -> None:
        """Test that default layer is False."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            assert points.layer is False

    def test_layer_set_at_creation(self, tmp_path: Path) -> None:
        """Test creating points with layer=True."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions, layer=True)

            assert points.layer is True

    def test_layer_setter_persists(self, tmp_path: Path) -> None:
        """Setting node.layer after creation updates attrs and zarr store."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            assert points.layer is False
            points.layer = True
            assert points.layer is True
            assert points.attrs["layer"] is True

            points.layer = False
            assert points.layer is False

    def test_layer_setter_rejects_invalid(self, tmp_path: Path) -> None:
        """Non-boolean-compatible values are rejected by the setter."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            with pytest.raises(TypeError):
                points.layer = "yes"


class TestNodeVisible:
    """Test Node visible authoring-time property."""

    def test_visible_default_true(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)
            assert points.visible is True

    def test_visible_false_at_creation(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions, layer=True, visible=False)
            assert points.visible is False
            assert points.attrs["visible"] is False

    def test_visible_setter_persists(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            points.visible = False
            assert points.visible is False
            points.visible = True
            assert points.visible is True

    def test_visible_rejects_invalid(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            with pytest.raises((TypeError, ValueError)):
                scene.add_points("test", positions, visible="yes")


class TestNodeColormap:
    """Test Node colormap property."""

    def test_colormap_set_string(self, tmp_path: Path) -> None:
        """Test setting colormap to a string name."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            points.colormap = "viridis"
            assert points.colormap == "viridis"

    def test_colormap_setter_rejects_array(self, tmp_path: Path) -> None:
        """Test that setting a numpy array via the setter raises TypeError."""
        store_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            points = scene.add_points("test", positions)

            lut = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            with pytest.raises(
                TypeError, match="Colormap property setter only accepts string"
            ):
                points.colormap = lut


class TestGeometryMetadataParity:
    """Points/Lines/GSplats expose the same metadata surface for what they share.

    Each property must read a value the writer actually produces — a property
    backed by a metadata key no writer sets silently reports its default, which
    is worse than not offering the property at all.
    """

    def test_points_and_lines_expose_the_shared_metadata_properties(
        self, tmp_path: Path
    ) -> None:
        rng = np.random.RandomState(0)
        n = 40
        with LuxarZarrCompiler(tmp_path / "parity.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            points = scene.add_points(
                "pts",
                rng.rand(n, 3).astype(np.float32),
                radii=np.full(n, 0.25, dtype=np.float32),
            )
            lines = scene.add_lines(
                "lns",
                rng.rand(n, 3).astype(np.float32),
                widths=np.full(n, 0.1, dtype=np.float32),
            )

            # Shared by both: dimensionality, spatial index, ordering.
            for node in (points, lines):
                assert node.ndim == 3
                assert node.has_spatial_index is True
                # Not merely "some string" — the real method the writer chose.
                assert node.ordering in {"morton", "hilbert"}

            # Per-type extent property (Points has no width, Lines no radius).
            assert points.max_radius == pytest.approx(0.25)
            assert lines.max_width == pytest.approx(0.1)

    def test_ordering_property_matches_what_was_written_to_disk(
        self, tmp_path: Path
    ) -> None:
        """The property must report the writer's real choice, not its default.

        ``Points.ordering`` reads the metadata the writer returns. When the
        writer omitted the key the property silently answered "none" for a node
        that was in fact hilbert-ordered on disk, so pin the two together.
        """
        import zarr

        store = tmp_path / "ordering.luxar.zarr"
        rng = np.random.RandomState(1)
        n = 40
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            nodes = {
                "pts": scene.add_points("pts", rng.rand(n, 3).astype(np.float32)),
                "lns": scene.add_lines(
                    "lns",
                    rng.rand(n, 3).astype(np.float32),
                    widths=np.full(n, 0.1, dtype=np.float32),
                ),
            }
            in_memory = {name: node.ordering for name, node in nodes.items()}

        root = zarr.open_group(store, "r")
        for name, reported in in_memory.items():
            # Absent on disk is how an unordered node is spelled; the viewer
            # treats a missing attr as "none" (chunk-index-loader.ts).
            on_disk = dict(root[name].attrs).get("ordering", "none")
            assert reported == on_disk, f"{name}: property={reported} disk={on_disk}"
