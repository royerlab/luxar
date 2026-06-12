"""Test progressive writing architecture with LuxarZarrCompiler."""

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar.compiler import LuxarZarrCompiler
from luxar.dimensions import Dimensions


class TestProgressiveWriting:
    """Test the new progressive writing architecture."""

    def test_compiler_context_manager(self) -> None:
        """Test LuxarZarrCompiler as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                # Compiler should be initialized
                assert compiler.store_path == str(output_path)
                assert compiler.store is not None

                # Create scene
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                assert scene is not None
                assert scene._writer == compiler

            # After context exit, should be finalized
            assert compiler._is_finalized

            # Zarr store should exist with consolidated metadata
            assert output_path.exists()
            store = zarr.open_group(output_path, mode="r")
            assert store.attrs["type"] == "scene"
            assert ".zmetadata" in store.store

    def test_progressive_points_writing(self) -> None:
        """Test that points are written immediately without keeping in memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            # Create large array (but not too large for CI)
            n_points = 10000
            positions = np.random.randn(n_points, 3).astype(np.float32)
            colors = np.random.rand(n_points, 3).astype(np.float32)

            with LuxarZarrCompiler(output_path, enable_spatial_index=False) as compiler:
                compiler.create_scene(dimensions=Dimensions.default_3d())

                # Write points - should go directly to disk
                metadata = compiler.write_points(
                    "test_points", positions, colors=colors
                )

                # Check metadata
                assert metadata["n_points"] == n_points
                assert metadata["ndim"] == 3
                assert metadata["has_colors"] is True

                # Verify data is in Zarr store
                store = zarr.open_group(output_path, mode="r")
                assert "test_points" in store
                assert "test_points/positions" in store
                assert "test_points/colors" in store

                # Verify data matches
                stored_positions = store["test_points/positions"][:]
                np.testing.assert_array_almost_equal(stored_positions, positions)

    def test_scene_with_dimensions(self) -> None:
        """Test scene creation with dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            from luxar import Dimension

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("time", unit="s", display=False),
                ]
            )

            with LuxarZarrCompiler(output_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                assert scene._dimensions == dims

                # Dimensions should be stored in Zarr
                store = zarr.open_group(output_path, mode="r")
                assert "scene_dimensions" in store.attrs
                dims_dict = store.attrs["scene_dimensions"]
                assert len(dims_dict["dimensions"]) == 4

    def test_hierarchical_structure(self) -> None:
        """Test creating hierarchical structure with groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Add a group
                group1 = scene.add_group("group1", opacity=0.8)
                assert group1.name == "group1"
                assert group1._writer == compiler

                # Add nested group
                group2 = group1.add_group("group2")
                assert group2.path == "group1/group2"

                # Verify structure in Zarr
                store = zarr.open_group(output_path, mode="r")
                assert "group1" in store
                assert "group1/group2" in store

    def test_no_memory_accumulation(self) -> None:
        """Test that data is not kept in memory after writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Create data
                positions = np.random.randn(1000, 3).astype(np.float32)

                # Write points
                metadata = compiler.write_points("points", positions)

                # Metadata should not contain actual data
                assert "positions" not in metadata
                assert metadata["n_points"] == 1000

                # Scene should not have the data
                assert not hasattr(scene, "_positions")

    def test_compiler_without_context_manager(self) -> None:
        """Test using compiler without context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(output_path)
            compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add some data
            positions = np.random.randn(100, 3).astype(np.float32)
            compiler.write_points("points", positions)

            # Manual finalization
            compiler.finalize()
            assert compiler._is_finalized

            # Verify store exists
            assert output_path.exists()

    def test_resizable_dataset_creation(self) -> None:
        """Test creating resizable datasets for streaming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(output_path) as compiler:
                # Create resizable dataset
                dataset = compiler.create_resizable_dataset(
                    "streaming_points/positions",
                    dtype=np.float32,
                    shape=(0, 3),
                    maxshape=(None, 3),
                )

                assert dataset is not None
                assert dataset.shape == (0, 3)

                # Append data
                batch1 = np.random.randn(100, 3).astype(np.float32)
                dataset.resize((100, 3))
                dataset[:100] = batch1

                # Append more data
                batch2 = np.random.randn(50, 3).astype(np.float32)
                dataset.resize((150, 3))
                dataset[100:150] = batch2

                # Verify
                assert dataset.shape == (150, 3)
                np.testing.assert_array_almost_equal(dataset[:100], batch1)
                np.testing.assert_array_almost_equal(dataset[100:150], batch2)
