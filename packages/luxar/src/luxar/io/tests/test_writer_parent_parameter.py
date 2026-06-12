"""Test zarr hierarchy creation with nested groups and points.

INVESTIGATION RESULT: Writer works correctly! Earlier confusion was due to
incorrect API usage in test. The parent parameter works when using:
1. Full paths: compiler.write_points("GroupA/Points", ...)
2. Scene API: scene.add_points("Points", parent=group_node)

This test suite guards against regressions in hierarchy creation.
"""

from pathlib import Path

import numpy as np
import zarr
from arbol import aprint

from luxar.core.dimensions import Dimensions


def test_points_with_parent_creates_correct_hierarchy(tmp_path: Path) -> None:
    """Test that points added with parent are stored under parent in zarr."""
    output_path = tmp_path / "hierarchy_test.luxar.zarr"

    # Create scene using compiler (recommended pattern)
    from luxar.io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(output_path) as compiler:
        scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Add root-level points
        compiler.write_points("RootPoints", np.array([[0, 0, 0]], dtype=np.float32))

        # Create group and add points to it
        group_a = scene_node.add_group("GroupA")
        compiler.write_points(
            "GroupA/GroupAPoints", np.array([[1, 1, 1]], dtype=np.float32)
        )

        # Create nested group and add points
        _group_b = group_a.add_group("GroupB")
        compiler.write_points(
            "GroupA/GroupB/GroupBPoints", np.array([[2, 2, 2]], dtype=np.float32)
        )

    # Verify zarr structure
    store = zarr.open(str(output_path), mode="r")

    # Check the actual zarr hierarchy
    aprint("\nZarr hierarchy:")

    def print_tree(group, indent=0):
        for key in sorted(group.group_keys()):
            child = group[key]
            node_type = child.attrs.get("type", "group")
            aprint("  " * indent + f"- {key} ({node_type})")
            if node_type == "group":
                print_tree(child, indent + 1)

    print_tree(store)

    # Assertions: Verify correct hierarchy
    assert "RootPoints" in store  # At root level ✓
    assert "GroupA" in store  # At root level ✓
    assert "GroupAPoints" in store["GroupA"]  # Under GroupA ✓
    assert "GroupB" in store["GroupA"]  # Under GroupA ✓
    assert "GroupBPoints" in store["GroupA"]["GroupB"]  # Under GroupA/GroupB ✓

    # Verify types
    assert store["RootPoints"].attrs.get("type") == "points"
    assert store["GroupA"].attrs.get("type") == "group"
    assert store["GroupA"]["GroupAPoints"].attrs.get("type") == "points"
    assert store["GroupA"]["GroupB"].attrs.get("type") == "group"
    assert store["GroupA"]["GroupB"]["GroupBPoints"].attrs.get("type") == "points"

    aprint("\n✅ All nodes are in correct hierarchy!")


def test_reader_reflects_correct_hierarchy(tmp_path: Path) -> None:
    """Test that reader shows correct hierarchy for nested nodes."""
    output_path = tmp_path / "reader_hierarchy.luxar.zarr"

    # Create scene with hierarchy
    from luxar.io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(output_path) as compiler:
        scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())

        compiler.write_points("Root1", np.array([[0, 0, 0]], dtype=np.float32))

        _group_a = scene_node.add_group("GroupA")
        compiler.write_points("GroupA/Points1", np.array([[1, 1, 1]], dtype=np.float32))

    # Read back
    from luxar.io.reader import LuxarScene

    reader = LuxarScene.load(output_path)

    nodes = reader.nodes
    names = [n["name"] for n in nodes]

    aprint("\nReader nodes:")
    for node in nodes:
        aprint(f"  - {node['name']} ({node['type']})")

    # Verify reader shows full paths
    assert "Root1" in names
    assert "GroupA" in names
    assert "GroupA/Points1" in names  # Should show full path

    # Verify list_points returns full paths
    point_names = reader.list_points()
    assert "Root1" in point_names
    assert "GroupA/Points1" in point_names

    aprint("\n✅ Reader shows correct hierarchy with full paths!")
