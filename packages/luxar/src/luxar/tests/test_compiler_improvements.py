"""Tests for compiler improvements and fixes."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import LuxarZarrCompiler
from luxar.core.transforms import prepare_transform_for_zarr, translate
from luxar.io.point_spatial_index import build_spatial_index, validate_spatial_index
from luxar.validation.base import ValidationError, validate_zarr_attributes


class TestVersionUpdate:
    """Test that the version is correctly set to 0.3."""

    def test_compiler_writes_correct_version(self):
        """Verify compiler writes version 0.3 to zarr attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene()
                positions = np.random.randn(100, 3).astype(np.float32)
                scene.add_points("test", positions)

            # Read back and check version
            store = zarr.open_group(zarr_path, mode="r")
            assert store.attrs["luxar_version"] == "0.1"
            assert store.attrs["type"] == "scene"


class TestChunkAlignment:
    """Test improved chunk alignment with spatial index."""

    def test_chunk_alignment_with_spatial_index(self):
        """Verify chunks are aligned with spatial index when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create dataset with spatial index
            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                scene = compiler.create_scene()
                positions = np.random.randn(10000, 3).astype(np.float32)
                scene.add_points("test", positions, grid_shape=(5, 5, 5))

            # Check that chunks were created
            store = zarr.open_group(zarr_path, mode="r")
            positions_array = store["test/positions"]
            chunks = positions_array.chunks

            # Should have reasonable chunk size aligned with spatial cells
            assert chunks[0] > 0
            assert chunks[0] <= 32768  # Default max chunk size
            assert chunks[1] == 3  # Dimensions should not be chunked

    def test_chunk_calculation_without_spatial_index(self):
        """Verify standard chunking when spatial index is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=False) as compiler:
                scene = compiler.create_scene()
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

    def test_prepare_transform_from_numpy_array(self):
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

    def test_prepare_transform_from_list(self):
        """Test that list format is validated and preserved."""
        # Already in column-major format
        transform_list = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]
        result = prepare_transform_for_zarr(transform_list)

        assert isinstance(result, list)
        assert len(result) == 16
        assert result[12] == 5.0
        assert result[13] == 6.0
        assert result[14] == 7.0

    def test_prepare_transform_from_flat_array(self):
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

    def test_prepare_transform_invalid_size(self):
        """Test that invalid transform size raises error."""
        with pytest.raises(ValueError, match="must have 16 elements"):
            prepare_transform_for_zarr([1, 2, 3])

    def test_transform_in_compiler(self):
        """Test that compiler uses centralized transform conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene()
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


class TestSpatialIndexValidation:
    """Test spatial index validation."""

    def test_validate_spatial_index_correct(self):
        """Test validation passes for correct spatial index."""
        positions = np.random.randn(1000, 3).astype(np.float32)
        index_data = build_spatial_index(positions)

        # Should not raise
        validate_spatial_index(index_data, 1000, 3)

    def test_validate_spatial_index_wrong_dimensions(self):
        """Test validation fails for dimension mismatch."""
        positions = np.random.randn(1000, 3).astype(np.float32)
        index_data = build_spatial_index(positions)

        with pytest.raises(ValueError, match="doesn't match expected"):
            validate_spatial_index(index_data, 1000, 4)  # Wrong dimensions

    def test_validate_spatial_index_wrong_points(self):
        """Test validation fails for point count mismatch."""
        positions = np.random.randn(1000, 3).astype(np.float32)
        index_data = build_spatial_index(positions)

        with pytest.raises(ValueError, match="doesn't match expected"):
            validate_spatial_index(index_data, 500, 3)  # Wrong point count

    def test_validate_spatial_index_missing_keys(self):
        """Test validation fails for missing keys."""
        incomplete_data = {
            "occupied_cells": np.array([]),
            "cell_ranges": np.array([]),
            # Missing other required keys
        }

        with pytest.raises(ValueError, match="missing required keys"):
            validate_spatial_index(incomplete_data, 0, 3)

    def test_spatial_index_in_compiler(self):
        """Test that compiler validates spatial index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                scene = compiler.create_scene()
                # Use 4D positions so that the 4th dimension is not displayed and gets indexed
                positions = np.random.randn(1000, 4).astype(np.float32)
                # This should build and validate spatial index internally
                scene.add_points("test", positions)

            # Check that spatial index was created
            store = zarr.open_group(zarr_path, mode="r")
            assert "test/spatial_index" in store
            assert "occupied_cells" in store["test/spatial_index"]
            assert "cell_ranges" in store["test/spatial_index"]


class TestZarrAttributeValidation:
    """Test zarr attribute validation."""

    def test_validate_root_attributes_complete(self):
        """Test validation passes for complete root attributes."""
        attrs = {
            "type": "scene",
            "luxar_version": "0.3",
            "units": "um",
            "scene_dimensions": {},
        }
        # Should not raise
        validate_zarr_attributes(attrs, is_root=True)

    def test_validate_root_missing_required(self):
        """Test validation fails for missing required root attributes."""
        attrs = {"units": "um"}  # Missing type and luxar_version

        with pytest.raises(ValidationError, match="Missing required"):
            validate_zarr_attributes(attrs, is_root=True)

    def test_validate_node_attributes(self):
        """Test validation for non-root node attributes."""
        attrs = {"type": "points"}
        # Should not raise
        validate_zarr_attributes(attrs, is_root=False)

    def test_validate_invalid_type(self):
        """Test validation fails for invalid node type."""
        attrs = {"type": "invalid_type"}

        with pytest.raises(ValidationError, match="Invalid node type"):
            validate_zarr_attributes(attrs)

    def test_validate_unsupported_version(self):
        """Test validation fails for unsupported version."""
        attrs = {"type": "scene", "luxar_version": "99.9"}

        with pytest.raises(ValidationError, match="Unsupported Luxar version"):
            validate_zarr_attributes(attrs, is_root=True)

    def test_validate_warns_missing_recommended(self):
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

    def test_sdr_colors_accepted(self):
        """Test that SDR colors (0-1) are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene()
                positions = np.random.randn(100, 3).astype(np.float32)
                colors = np.random.rand(100, 3).astype(np.float32)  # 0-1 range
                scene.add_points("test", positions, colors=colors)

            # Should complete without warnings
            store = zarr.open_group(zarr_path, mode="r")
            assert "test/colors" in store

    def test_hdr_colors_warning(self):
        """Test that extreme HDR colors trigger warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with pytest.warns(UserWarning, match="HDR colors"):
                with LuxarZarrCompiler(zarr_path) as compiler:
                    scene = compiler.create_scene()
                    positions = np.random.randn(100, 3).astype(np.float32)
                    colors = (
                        np.random.rand(100, 3).astype(np.float32) * 20
                    )  # Very bright HDR
                    scene.add_points("test", positions, colors=colors)

    def test_negative_colors_rejected(self):
        """Test that negative colors are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Scene wraps ValidationError in ValueError
            with pytest.raises(ValueError, match="cannot be negative"):
                with LuxarZarrCompiler(zarr_path) as compiler:
                    scene = compiler.create_scene()
                    positions = np.random.randn(100, 3).astype(np.float32)
                    colors = np.random.randn(100, 3).astype(
                        np.float32
                    )  # Can be negative
                    scene.add_points("test", positions, colors=colors)


class TestEmptyDatasets:
    """Test handling of empty datasets."""

    def test_empty_positions_rejected(self):
        """Test that empty positions are properly rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Scene wraps ValidationError in ValueError
            with pytest.raises(ValueError, match="Cannot write empty"):
                with LuxarZarrCompiler(zarr_path) as compiler:
                    scene = compiler.create_scene()
                    positions = np.array([], dtype=np.float32).reshape(0, 3)
                    scene.add_points("test", positions)

    def test_spatial_index_empty_data(self):
        """Test spatial index handles empty data gracefully."""
        positions = np.array([], dtype=np.float32).reshape(0, 3)
        index_data = build_spatial_index(positions)

        # Should return valid but empty index
        # For 3D data where all dimensions are displayed, occupied_cells has shape (0, 0)
        assert index_data["occupied_cells"].shape == (0, 0)
        assert index_data["cell_ranges"].shape == (0, 2)
        assert index_data["sorted_positions"].shape == (0, 3)
        assert len(index_data["sort_order"]) == 0
