"""Test reader node collection for potential duplicates.

INVESTIGATION RESULT: Reader works correctly with NO duplicates in any scenario.
The original bug report was a misunderstanding of the code behavior.

Further investigation of writer hierarchy also found NO bugs - the API works correctly
when used properly (see test_writer_parent_parameter.py).

This test suite guards against future regressions in node collection.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr
from arbol import aprint

from luxar.core.dimensions import Dimensions
from luxar.io.compiler import LuxarZarrCompiler
from luxar.io.reader import LuxarScene
from luxar.typing_utils._format_contract import SCENE_FORMAT_VERSION
from luxar.typing_utils.format_version import UnsupportedFormatVersionError


class TestReaderNodeCollection:
    """Test that reader correctly collects nodes without duplicates.

    VERDICT: Reader works correctly. No duplicates found in any scenario.
    """

    def test_nested_groups_no_duplicates(self, tmp_path: Path) -> None:
        """Test that nested groups don't cause duplicate node entries."""
        output_path = tmp_path / "nested_groups.luxar.zarr"

        # Create nested structure: Root -> GroupA -> GroupB -> Points
        # Use full paths (not parent= kwarg which doesn't exist on write_points)
        with LuxarZarrCompiler(output_path) as compiler:
            scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add root level points
            compiler.write_points("RootPoints", np.array([[0, 0, 0]], dtype=np.float32))

            # Create nested groups
            group_a = scene_node.add_group("GroupA")
            compiler.write_points(
                "GroupA/GroupAPoints",
                np.array([[1, 1, 1]], dtype=np.float32),
            )

            # Create nested group under GroupA
            _group_b = group_a.add_group("GroupB")
            compiler.write_points(
                "GroupA/GroupB/GroupBPoints",
                np.array([[2, 2, 2]], dtype=np.float32),
            )

        # Load and verify
        scene = LuxarScene.load(output_path)
        nodes = scene.nodes

        # Check total count
        aprint(f"\nTotal nodes found: {len(nodes)}")
        for node in nodes:
            aprint(f"  - {node['name']} ({node['type']})")

        # Check for duplicates
        names = [n["name"] for n in nodes]
        duplicates = [name for name in set(names) if names.count(name) > 1]

        # Assertion: No duplicates should exist
        assert len(duplicates) == 0, f"Found duplicate nodes: {duplicates}"

        # Verify expected nodes are present with correct hierarchy
        assert "RootPoints" in names
        assert "GroupA" in names
        assert "GroupA/GroupAPoints" in names
        assert "GroupA/GroupB" in names
        assert "GroupA/GroupB/GroupBPoints" in names

        # Verify each node appears exactly once
        for name in names:
            assert names.count(name) == 1, (
                f"Node '{name}' appears {names.count(name)} times"
            )

        # Verify list methods return correct results
        assert len(scene.list_points()) == 3  # RootPoints, GroupAPoints, GroupBPoints
        assert len(scene.list_groups()) == 2  # GroupA, GroupB

    def test_flat_structure_no_duplicates(self, tmp_path: Path) -> None:
        """Test that flat structure (no nesting) works correctly."""
        output_path = tmp_path / "flat_structure.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add multiple points at root level
            for i in range(5):
                compiler.write_points(
                    f"Points{i}", np.random.randn(10, 3).astype(np.float32)
                )

        scene = LuxarScene.load(output_path)
        nodes = scene.nodes

        # Check for duplicates
        names = [n["name"] for n in nodes]
        duplicates = [name for name in set(names) if names.count(name) > 1]

        assert len(duplicates) == 0, f"Found duplicate nodes: {duplicates}"
        assert len(scene.list_points()) == 5

    def test_mixed_types_no_duplicates(self, tmp_path: Path) -> None:
        """Test mixed node types (groups, points, lines) without duplicates."""
        output_path = tmp_path / "mixed_types.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene_node = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Root level: points, lines, group
            compiler.write_points("RootPoints", np.array([[0, 0, 0]], dtype=np.float32))

            # Add lines
            vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
            widths = np.array([0.1, 0.1], dtype=np.float32)
            compiler.write_lines("RootLines", vertices, widths, line_type="segments")

            # Add group with nested content
            _group = scene_node.add_group("MyGroup")
            compiler.write_points(
                "MyGroup/GroupPoints",
                np.array([[2, 2, 2]], dtype=np.float32),
            )

        scene = LuxarScene.load(output_path)
        nodes = scene.nodes

        # Check for duplicates
        names = [n["name"] for n in nodes]
        duplicates = [name for name in set(names) if names.count(name) > 1]

        assert len(duplicates) == 0, f"Found duplicate nodes: {duplicates}"

        # Verify counts
        assert len(scene.list_points()) == 2
        assert len(scene.list_lines()) == 1
        assert len(scene.list_groups()) == 1


class TestReaderVersionCheck:
    """``LuxarScene.load()`` applies the shared format-version policy.

    The arm table below is the one in ``typing_utils/tests/test_format_version.py``
    (and the viewer's ``format-version.test.ts``), exercised through the real
    entry point on a real on-disk root so the enforcement seam itself — not just
    the policy function — is what is pinned.
    """

    @staticmethod
    def _root(path: Path, attrs: dict) -> None:
        store = zarr.open_group(path, mode="w")
        store.attrs.update({"type": "scene", **attrs})

    def _load(self, path: Path) -> list[warnings.WarningMessage]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            LuxarScene.load(path)
        return [w for w in caught if issubclass(w.category, UserWarning)]

    @pytest.mark.parametrize(
        "attrs",
        [
            pytest.param(
                {"format_version": "0.2", "format_type": "luxar_zarr"}, id="supported"
            ),
            pytest.param({"luxar_version": "0.1"}, id="legacy-key-fallback"),
            pytest.param({}, id="missing-without-format-type"),
        ],
    )
    def test_supported_arms_load_silently(self, tmp_path: Path, attrs: dict) -> None:
        path = tmp_path / "scene.luxar.zarr"
        self._root(path, attrs)
        assert self._load(path) == []

    def test_newer_minor_warns_and_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "newer.luxar.zarr"
        self._root(path, {"format_version": "0.3", "format_type": "luxar_zarr"})
        caught = self._load(path)
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "0.3" in message and SCENE_FORMAT_VERSION in message
        assert "Loading anyway" in message

    @pytest.mark.parametrize(
        ("attrs", "needle"),
        [
            pytest.param({"format_version": "0.0"}, "0.0", id="older-unsupported"),
            pytest.param({"format_version": "9.9"}, "9.9", id="newer-major"),
            pytest.param({"format_version": "abc"}, "abc", id="unparsable"),
            pytest.param(
                {"format_type": "luxar_zarr"},
                "format_type",
                id="missing-with-format-type",
            ),
        ],
    )
    def test_refused_arms_raise(self, tmp_path: Path, attrs: dict, needle: str) -> None:
        path = tmp_path / "bad.luxar.zarr"
        self._root(path, attrs)
        with pytest.raises(UnsupportedFormatVersionError, match=needle):
            LuxarScene.load(path)

    def test_version_property_reads_both_spellings(self, tmp_path: Path) -> None:
        new = tmp_path / "new.luxar.zarr"
        self._root(new, {"format_version": "0.2", "format_type": "luxar_zarr"})
        assert LuxarScene.load(new).version == "0.2"
        legacy = tmp_path / "legacy.luxar.zarr"
        self._root(legacy, {"luxar_version": "0.1"})
        assert LuxarScene.load(legacy).version == "0.1"
        bare = tmp_path / "bare.luxar.zarr"
        self._root(bare, {})
        assert LuxarScene.load(bare).version == "unknown"
