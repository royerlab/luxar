"""Tests for nD zarr data handling and chunking optimization."""

import numpy as np
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import ArrayDecoder
from luxar.typing_utils.constants import MAX_CHUNK_BYTES, MIN_CHUNK_BYTES


def _atol(a):
    # AUTO positions are uint16 per-axis fixed-point (~extent/65535).
    return float(np.ptp(a, axis=0).max()) / 65535 * 2


class TestZarrNDChunking:
    """Test suite for nD zarr data chunking and loading."""

    def test_4d_data_chunking_optimization(self, tmp_path) -> None:
        """Test that 4D data is chunked appropriately for temporal slicing."""
        store = tmp_path / "4d_chunked.luxar.zarr"

        # Create 4D data: 1000 time steps, 1000 points per step
        n_timesteps = 1000
        n_points_per_step = 1000
        total_points = n_timesteps * n_points_per_step

        # Generate 4D positions (x, y, z, t)
        positions = np.zeros((total_points, 4), dtype=np.float32)
        for t in range(n_timesteps):
            start_idx = t * n_points_per_step
            end_idx = (t + 1) * n_points_per_step
            # Create points in a sphere that moves over time
            theta = np.random.uniform(0, 2 * np.pi, n_points_per_step)
            phi = np.random.uniform(0, np.pi, n_points_per_step)
            r = np.random.uniform(0.5, 1.0, n_points_per_step)

            positions[start_idx:end_idx, 0] = r * np.sin(phi) * np.cos(theta)
            positions[start_idx:end_idx, 1] = r * np.sin(phi) * np.sin(theta)
            positions[start_idx:end_idx, 2] = r * np.cos(phi)
            positions[start_idx:end_idx, 3] = t  # Time coordinate

        # Create scene at the specified store location and add points
        # Disable spatial index to preserve order for this test
        with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("Points4D", positions)

        # Verify chunking by opening the zarr store
        root = zarr.open_group(store, "r")
        positions_array = root["Points4D"]["positions"]

        # Check that chunks are reasonable for temporal slicing
        # First dimension (points) should have moderate chunk size
        # Second dimension (coordinates) should be fully included in each chunk
        assert positions_array.chunks[1] == 4  # All coordinates in one chunk
        assert positions_array.chunks[0] <= 1000000  # Reasonable chunk size for points

        # Verify data integrity
        loaded_positions = ArrayDecoder().decode(positions_array, root)
        np.testing.assert_allclose(loaded_positions, positions, atol=_atol(positions))

    def test_nd_generic_handling(self, tmp_path) -> None:
        """Test that the system handles arbitrary nD data generically."""
        store = tmp_path / "nd_generic.luxar.zarr"

        # Test with 5D data (x, y, z, t, channel)
        n_points = 10000
        positions_5d = np.random.randn(n_points, 5).astype(np.float32)

        # Create scene and add 5D points
        # Disable spatial index to preserve order for this test
        with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("Points5D", positions_5d)

        # Verify it saved correctly
        root = zarr.open_group(store, "r")
        loaded_positions = ArrayDecoder().decode(root["Points5D"]["positions"], root)

        assert loaded_positions.shape == (n_points, 5)
        np.testing.assert_allclose(
            loaded_positions, positions_5d, atol=_atol(positions_5d)
        )

    def test_chunk_boundary_alignment(self, tmp_path) -> None:
        """Test that chunking aligns well with typical access patterns."""
        store = tmp_path / "chunk_aligned.luxar.zarr"

        # Create data that doesn't align perfectly with default chunks
        n_points = 123456  # Not a nice round number
        positions = np.random.randn(n_points, 3).astype(np.float32)

        with LuxarZarrCompiler(store) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("Points", positions)

        root = zarr.open_group(store, "r")
        positions_array = root["Points"]["positions"]

        # Verify chunks exist and cover all data
        n_chunks_0 = int(np.ceil(n_points / positions_array.chunks[0]))
        _ = int(np.ceil(3 / positions_array.chunks[1]))  # Verify chunk calculation

        # Check we can access boundary chunks without error
        last_chunk_start = (n_chunks_0 - 1) * positions_array.chunks[0]
        last_chunk_data = positions_array[last_chunk_start:]
        assert len(last_chunk_data) == n_points - last_chunk_start

    def test_memory_efficient_slicing(self, tmp_path) -> None:
        """Test that slicing large datasets is memory efficient."""
        store = tmp_path / "memory_efficient.luxar.zarr"

        # Create large dataset
        n_slices = 100
        points_per_slice = 10000
        total_points = n_slices * points_per_slice

        # Generate data slice by slice to avoid memory issues
        positions = np.zeros((total_points, 4), dtype=np.float32)

        for s in range(n_slices):
            start = s * points_per_slice
            end = (s + 1) * points_per_slice
            positions[start:end, 0] = np.random.randn(points_per_slice)
            positions[start:end, 1] = np.random.randn(points_per_slice)
            positions[start:end, 2] = np.random.randn(points_per_slice)
            positions[start:end, 3] = s  # Slice index as 4th dimension

        with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("Points", positions)

        # Test that we can efficiently load a single slice
        root = zarr.open_group(store, "r")
        positions_array = root["Points"]["positions"]

        # Load just one slice
        slice_50_start = 50 * points_per_slice
        slice_50_end = 51 * points_per_slice
        single_slice = positions_array[slice_50_start:slice_50_end]

        assert single_slice.shape == (points_per_slice, 4)
        # positions are uint16 fixed-point (AUTO); decode axis-3 to check the slice index
        enc = dict(positions_array.attrs["encoding"])
        lo3, hi3 = enc["col_lo"][3], enc["col_hi"][3]
        decoded_ax3 = lo3 + single_slice[:, 3].astype(np.float64) / 65535 * (hi3 - lo3)
        np.testing.assert_allclose(decoded_ax3, 50, atol=0.01)

    def test_no_hardcoded_dimensions(self, tmp_path) -> None:
        """Ensure the system doesn't assume specific dimension meanings."""
        store = tmp_path / "no_hardcoded.luxar.zarr"

        # Create data with unusual dimension count
        for n_dims in [2, 3, 4, 7, 10]:
            n_points = 1000
            positions = np.random.randn(n_points, n_dims).astype(np.float32)

            # Save with unique name
            dim_store = store / f"dims_{n_dims}.luxar.zarr"
            # Disable spatial index to preserve order for this test
            with LuxarZarrCompiler(dim_store, enable_spatial_index=False) as compiler:
                compiler.create_scene(dimensions=Dimensions.default_3d())
                compiler.write_points("Points", positions)

            # Verify it loads correctly
            root = zarr.open_group(dim_store, "r")
            loaded = ArrayDecoder().decode(root["Points"]["positions"], root)

            assert loaded.shape == (n_points, n_dims)
            np.testing.assert_allclose(loaded, positions, atol=_atol(positions))

    def test_optimal_chunk_cache_interaction(self, tmp_path) -> None:
        """Test that chunk sizes work well with typical cache sizes."""
        store = tmp_path / "cache_optimized.luxar.zarr"

        # Create dataset sized to test cache behavior
        n_points = 1_000_000
        positions = np.random.randn(n_points, 3).astype(np.float32)

        # Disable spatial indexing to test basic chunking strategy
        with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("Points", positions)

        root = zarr.open_group(store, "r")
        positions_array = root["Points"]["positions"]

        # Calculate chunk size in bytes
        chunk_shape = positions_array.chunks
        bytes_per_element = 4  # float32
        chunk_size_bytes = chunk_shape[0] * chunk_shape[1] * bytes_per_element

        # Chunk size should sit within the byte-target band defined by the
        # single source of truth in typing_utils.constants.
        assert MIN_CHUNK_BYTES <= chunk_size_bytes <= MAX_CHUNK_BYTES

        # Verify chunks are not too small (inefficient) or too large (memory issues)
        assert chunk_shape[0] >= 1000  # At least 1000 points per chunk
        assert chunk_shape[0] <= 1_000_000  # At most 1M points per chunk

    def test_sparse_data_efficiency(self, tmp_path) -> None:
        """Test that sparse nD data is handled efficiently."""
        store = tmp_path / "sparse.luxar.zarr"

        # Create sparse 4D data where most time slices are empty
        _ = 1000  # Total timesteps (most are empty)
        active_timesteps = [10, 50, 100, 500, 900]  # Only 5 active timesteps
        points_per_active = 1000

        positions_list = []
        for t in active_timesteps:
            # Create points only at specific timesteps
            points = np.random.randn(points_per_active, 4).astype(np.float32)
            points[:, 3] = t  # Set time coordinate
            positions_list.append(points)

        positions = np.vstack(positions_list)

        with LuxarZarrCompiler(store) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("Points", positions)

        root = zarr.open_group(store, "r")
        positions_array = root["Points"]["positions"]

        # Verify sparse data is stored efficiently
        assert positions_array.shape[0] == len(active_timesteps) * points_per_active

        # Check that we can query specific timesteps efficiently
        loaded = ArrayDecoder().decode(positions_array, root)
        for i, t in enumerate(active_timesteps):
            slice_data = loaded[np.isclose(loaded[:, 3], t, atol=0.01)]
            assert len(slice_data) == points_per_active
