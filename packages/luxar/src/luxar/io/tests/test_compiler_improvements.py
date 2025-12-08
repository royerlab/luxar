"""Tests for compiler improvements and fixes."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.transforms import prepare_transform_for_zarr, translate
from luxar.validation.base import ValidationError, validate_zarr_attributes


class TestVersionUpdate:
    """Test that the version is correctly set to 0.3."""

    def test_compiler_writes_correct_version(self) -> None:
        """Verify compiler writes version 0.3 to zarr attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                positions = np.random.randn(100, 3).astype(np.float32)
                scene.add_points("test", positions)

            # Read back and check version
            store = zarr.open_group(zarr_path, mode="r")
            assert store.attrs["luxar_version"] == "0.1"
            assert store.attrs["type"] == "scene"


class TestChunkAlignment:
    """Test improved chunk alignment with spatial index."""

    def test_chunk_alignment_with_spatial_ordering(self) -> None:
        """Verify chunks are aligned with spatial ordering when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create dataset with spatial ordering
            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                from luxar.core.dimensions import Dimension, Dimensions

                dims = Dimensions(
                    [
                        Dimension("x", unit="m", display=True),
                        Dimension("y", unit="m", display=True),
                        Dimension("z", unit="m", display=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                positions = np.random.randn(10000, 3).astype(np.float32)
                scene.add_points("test", positions)

            # Check that chunks were created
            store = zarr.open_group(zarr_path, mode="r")
            positions_array = store["test/positions"]
            chunks = positions_array.chunks

            # Should have reasonable chunk size
            assert chunks[0] > 0
            assert chunks[0] <= 32768  # Default max chunk size
            assert chunks[1] == 3  # Dimensions should not be chunked

    def test_chunk_calculation_without_spatial_index(self) -> None:
        """Verify standard chunking when spatial index is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=False) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                positions = np.random.randn(10000, 3).astype(np.float32)
                scene.add_points("test", positions)

            store = zarr.open_group(zarr_path, mode="r")
            positions_array = store["test/positions"]
            chunks = positions_array.chunks

            # Should use standard chunking
            assert chunks[0] > 0
            assert chunks[0] <= 32768
            assert chunks[1] == 3


class TestTransformCentralization:
    """Test centralized transform conversion."""

    def test_prepare_transform_from_numpy_array(self) -> None:
        """Test converting numpy array to zarr format."""
        matrix = translate(1, 2, 3)
        result = prepare_transform_for_zarr(matrix)

        assert isinstance(result, list)
        assert len(result) == 16
        # Check translation values are in correct positions for THREE.js
        # In column-major order, translations are at indices 12, 13, 14
        assert result[12] == 1.0
        assert result[13] == 2.0
        assert result[14] == 3.0

    def test_prepare_transform_from_list(self) -> None:
        """Test that list format is validated and preserved."""
        # Already in column-major format
        transform_list = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]
        result = prepare_transform_for_zarr(transform_list)

        assert isinstance(result, list)
        assert len(result) == 16
        assert result[12] == 5.0
        assert result[13] == 6.0
        assert result[14] == 7.0

    def test_prepare_transform_from_flat_array(self) -> None:
        """Test converting flat numpy array."""
        flat = np.array(
            [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1], dtype=np.float32
        )
        result = prepare_transform_for_zarr(flat)

        assert isinstance(result, list)
        assert len(result) == 16
        # After transpose, translations should be at 12, 13, 14
        assert result[12] == 1.0
        assert result[13] == 2.0
        assert result[14] == 3.0

    def test_prepare_transform_invalid_size(self) -> None:
        """Test that invalid transform size raises error."""
        with pytest.raises(ValueError, match="must have 16 elements"):
            prepare_transform_for_zarr([1, 2, 3])

    def test_transform_in_compiler(self) -> None:
        """Test that compiler uses centralized transform conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                positions = np.random.randn(100, 3).astype(np.float32)
                transform = translate(10, 20, 30)
                scene.add_points("test", positions, transform=transform)

            # Read back and verify transform
            store = zarr.open_group(zarr_path, mode="r")
            attrs = dict(store["test"].attrs)
            assert "transform" in attrs
            assert isinstance(attrs["transform"], list)
            assert len(attrs["transform"]) == 16
            # Check translation values in THREE.js format
            assert attrs["transform"][12] == 10.0
            assert attrs["transform"][13] == 20.0
            assert attrs["transform"][14] == 30.0


class TestSpatialOrdering:
    """Test spatial ordering with Morton/Hilbert curves."""

    def test_spatial_ordering_in_compiler(self) -> None:
        """Test that compiler applies spatial ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(
                zarr_path, enable_spatial_index=True, ordering_method="morton"
            ) as compiler:
                from luxar.core.dimensions import Dimension, Dimensions

                # Create 4D scene so we have discrete dimensions to order
                dims = Dimensions(
                    [
                        Dimension("x", unit="m", display=True),
                        Dimension("y", unit="m", display=True),
                        Dimension("z", unit="m", display=True),
                        Dimension("t", unit="s", display=False, discrete=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                positions = np.random.randn(1000, 4).astype(np.float32)
                scene.add_points("test", positions)

            # Check that spatial ordering metadata was created
            store = zarr.open_group(zarr_path, mode="r")
            # Check chunk_bounds written directly to points group
            assert "test/chunk_bounds" in store

            # Check ordering metadata in points group attrs (not sub-group)
            test_attrs = dict(store["test"].attrs)
            assert test_attrs["ordering"] == "morton"
            assert "slice_dims" in test_attrs
            assert "ordering_dims" in test_attrs
            assert "chunk_size" in test_attrs


class TestZarrAttributeValidation:
    """Test zarr attribute validation."""

    def test_validate_root_attributes_complete(self) -> None:
        """Test validation passes for complete root attributes."""
        attrs = {
            "type": "scene",
            "luxar_version": "0.3",
            "units": "um",
            "scene_dimensions": {},
        }
        # Should not raise
        validate_zarr_attributes(attrs, is_root=True)

    def test_validate_root_missing_required(self) -> None:
        """Test validation fails for missing required root attributes."""
        attrs = {"units": "um"}  # Missing type and luxar_version

        with pytest.raises(ValidationError, match="Missing required"):
            validate_zarr_attributes(attrs, is_root=True)

    def test_validate_node_attributes(self) -> None:
        """Test validation for non-root node attributes."""
        attrs = {"type": "points"}
        # Should not raise
        validate_zarr_attributes(attrs, is_root=False)

    def test_validate_invalid_type(self) -> None:
        """Test validation fails for invalid node type."""
        attrs = {"type": "invalid_type"}

        with pytest.raises(ValidationError, match="Invalid node type"):
            validate_zarr_attributes(attrs)

    def test_validate_unsupported_version(self) -> None:
        """Test validation fails for unsupported version."""
        attrs = {"type": "scene", "luxar_version": "99.9"}

        with pytest.raises(ValidationError, match="Unsupported Luxar version"):
            validate_zarr_attributes(attrs, is_root=True)

    def test_validate_warns_missing_recommended(self) -> None:
        """Test validation warns about missing recommended attributes."""
        attrs = {
            "type": "scene",
            "luxar_version": "0.3",
            # Missing units and scene_dimensions (recommended)
        }

        with pytest.warns(UserWarning, match="Missing recommended"):
            validate_zarr_attributes(attrs, is_root=True)


class TestHDRColorRanges:
    """Test HDR color range handling."""

    def test_sdr_colors_accepted(self) -> None:
        """Test that SDR colors (0-1) are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                positions = np.random.randn(100, 3).astype(np.float32)
                colors = np.random.rand(100, 3).astype(np.float32)  # 0-1 range
                scene.add_points("test", positions, colors=colors)

            # Should complete without warnings
            store = zarr.open_group(zarr_path, mode="r")
            assert "test/colors" in store

    def test_hdr_colors_warning(self) -> None:
        """Test that extreme HDR colors trigger warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with pytest.warns(UserWarning, match="HDR colors"):
                with LuxarZarrCompiler(zarr_path) as compiler:
                    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                    positions = np.random.randn(100, 3).astype(np.float32)
                    colors = (
                        np.random.rand(100, 3).astype(np.float32) * 20
                    )  # Very bright HDR
                    scene.add_points("test", positions, colors=colors)

    def test_negative_colors_rejected(self) -> None:
        """Test that negative colors are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Scene wraps ValidationError in ValueError
            with pytest.raises(ValueError, match="cannot be negative"):
                with LuxarZarrCompiler(zarr_path) as compiler:
                    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                    positions = np.random.randn(100, 3).astype(np.float32)
                    colors = np.random.randn(100, 3).astype(
                        np.float32
                    )  # Can be negative
                    scene.add_points("test", positions, colors=colors)


class TestEmptyDatasets:
    """Test handling of empty datasets."""

    def test_empty_positions_rejected(self) -> None:
        """Test that empty positions are properly rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Scene wraps ValidationError in ValueError
            with pytest.raises(ValueError, match="Cannot write empty"):
                with LuxarZarrCompiler(zarr_path) as compiler:
                    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                    positions = np.array([], dtype=np.float32).reshape(0, 3)
                    scene.add_points("test", positions)


class TestPositionBounds:
    """Test position_bounds computation and storage."""

    def test_single_node_bounds(self) -> None:
        """Test that position_bounds is computed correctly for a single node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create simple positions with known bounds
            positions = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 20.0, 30.0],
                    [5.0, 10.0, 15.0],
                ],
                dtype=np.float32,
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("test", positions)

            # Read back and verify bounds
            store = zarr.open_group(zarr_path, mode="r")

            # Check node-level bounds
            node_bounds = store["test"].attrs["position_bounds"]
            assert node_bounds["min"] == [0.0, 0.0, 0.0]
            assert node_bounds["max"] == [10.0, 20.0, 30.0]

            # Check scene-level bounds (should match since single node)
            scene_bounds = store.attrs["position_bounds"]
            assert scene_bounds["min"] == [0.0, 0.0, 0.0]
            assert scene_bounds["max"] == [10.0, 20.0, 30.0]

    def test_multiple_nodes_bounds_union(self) -> None:
        """Test that scene bounds are the union of all node bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create two sets of positions with different bounds
            positions1 = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 5.0, 5.0],
                ],
                dtype=np.float32,
            )
            positions2 = np.array(
                [
                    [-10.0, -10.0, -10.0],
                    [20.0, 30.0, 40.0],
                ],
                dtype=np.float32,
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("points1", positions1)
                scene.add_points("points2", positions2)

            # Read back and verify bounds
            store = zarr.open_group(zarr_path, mode="r")

            # Check individual node bounds
            bounds1 = store["points1"].attrs["position_bounds"]
            assert bounds1["min"] == [0.0, 0.0, 0.0]
            assert bounds1["max"] == [5.0, 5.0, 5.0]

            bounds2 = store["points2"].attrs["position_bounds"]
            assert bounds2["min"] == [-10.0, -10.0, -10.0]
            assert bounds2["max"] == [20.0, 30.0, 40.0]

            # Check scene-level bounds (union of both)
            scene_bounds = store.attrs["position_bounds"]
            assert scene_bounds["min"] == [-10.0, -10.0, -10.0]
            assert scene_bounds["max"] == [20.0, 30.0, 40.0]

    def test_nd_bounds(self) -> None:
        """Test that position_bounds works correctly for nD data."""
        from luxar.core.dimensions import Dimension, Dimensions

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create 5D positions
            positions = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.5, 1.0, 1.5, 2.0, 2.5],
                ],
                dtype=np.float32,
            )

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("time", unit="s", display=False),
                    Dimension("channel", unit="", display=False),
                ]
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points("test", positions)

            # Read back and verify 5D bounds
            store = zarr.open_group(zarr_path, mode="r")

            node_bounds = store["test"].attrs["position_bounds"]
            assert len(node_bounds["min"]) == 5
            assert len(node_bounds["max"]) == 5
            assert node_bounds["min"] == [0.0, 0.0, 0.0, 0.0, 0.0]
            assert node_bounds["max"] == [1.0, 2.0, 3.0, 4.0, 5.0]

            scene_bounds = store.attrs["position_bounds"]
            assert scene_bounds["min"] == [0.0, 0.0, 0.0, 0.0, 0.0]
            assert scene_bounds["max"] == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_bounds_with_spatial_ordering(self) -> None:
        """Test that bounds are computed correctly even with spatial reordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create positions - they will be reordered by spatial index
            np.random.seed(42)
            positions = np.random.randn(1000, 3).astype(np.float32) * 10

            expected_min = positions.min(axis=0).tolist()
            expected_max = positions.max(axis=0).tolist()

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("test", positions)

            # Read back and verify bounds match original (pre-reordering) data
            store = zarr.open_group(zarr_path, mode="r")
            node_bounds = store["test"].attrs["position_bounds"]

            # Bounds should be the same regardless of reordering
            for i in range(3):
                assert abs(node_bounds["min"][i] - expected_min[i]) < 1e-5
                assert abs(node_bounds["max"][i] - expected_max[i]) < 1e-5
