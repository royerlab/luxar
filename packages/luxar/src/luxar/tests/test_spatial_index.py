"""Tests for spatial index functionality."""

import numpy as np
import zarr

from luxar.io.spatial_index import (
    apply_sort_order,
    build_spatial_index,
    decode_cell_id,
    query_spatial_index,
)


class TestDecoding:
    """Test cell ID decoding functions."""

    def test_decode_cell_id_2d(self):
        """Test decoding cell IDs for 2D grids."""
        grid_shape = np.array([3, 4], dtype=np.uint32)

        # Test corner cases
        assert decode_cell_id(0, grid_shape) == [0, 0]
        assert decode_cell_id(3, grid_shape) == [0, 3]
        assert decode_cell_id(4, grid_shape) == [1, 0]
        assert decode_cell_id(11, grid_shape) == [2, 3]

    def test_decode_cell_id_3d(self):
        """Test decoding cell IDs for 3D grids."""
        grid_shape = np.array([2, 3, 4], dtype=np.uint32)

        # Cell ID = z * (3*4) + y * 4 + x
        assert decode_cell_id(0, grid_shape) == [0, 0, 0]
        assert decode_cell_id(1, grid_shape) == [0, 0, 1]
        assert decode_cell_id(4, grid_shape) == [0, 1, 0]
        assert decode_cell_id(12, grid_shape) == [1, 0, 0]
        assert decode_cell_id(23, grid_shape) == [1, 2, 3]


class TestSpatialIndexBuilding:
    """Test spatial index construction."""

    def test_build_index_2d(self):
        """Test building spatial index for 2D points."""
        np.random.seed(42)
        positions = np.random.randn(100, 2).astype(np.float32) * 10
        grid_shape = np.array([5], dtype=np.uint32)  # Only for non-displayed dim

        # For 2D, both dimensions would be displayed by default, so specify only first is displayed
        result = build_spatial_index(positions, grid_shape, displayed_dims=[0])

        # Check output structure
        assert "occupied_cells" in result
        assert "cell_ranges" in result
        assert "sorted_positions" in result
        assert "sort_order" in result
        assert "grid_origin" in result
        assert "cell_size" in result
        assert "grid_shape" in result

        # Check shapes
        assert result["sorted_positions"].shape == positions.shape
        assert result["sort_order"].shape == (100,)
        assert len(result["occupied_cells"]) == len(result["cell_ranges"])

        # Check that all points are accounted for
        total_points = 0
        for start, end in result["cell_ranges"]:
            total_points += end - start
        assert total_points == 100

        # Check ranges are non-overlapping and sorted
        last_end = 0
        for start, end in result["cell_ranges"]:
            assert start == last_end
            assert end > start
            last_end = end

    def test_build_index_auto_grid(self):
        """Test automatic grid resolution determination."""
        np.random.seed(42)
        # Make 4D data where first 3 are displayed, last one is indexed
        positions = np.random.randn(1000, 4).astype(np.float32)

        result = build_spatial_index(
            positions
        )  # No grid_shape provided, defaults to first 3 displayed

        # Should auto-determine reasonable grid shape for the one non-displayed dimension
        assert result["grid_shape"] is not None
        assert len(result["grid_shape"]) == 1  # Only 1 non-displayed dimension
        assert all(2 <= dim <= 20 for dim in result["grid_shape"])

    def test_build_index_single_point(self):
        """Test index building with single point."""
        positions = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)  # 4D point
        grid_shape = np.array(
            [2], dtype=np.uint32
        )  # Grid for the non-displayed dimension

        result = build_spatial_index(
            positions, grid_shape
        )  # First 3 dims are displayed by default

        assert len(result["occupied_cells"]) == 1
        assert len(result["cell_ranges"]) == 1
        assert result["cell_ranges"][0][0] == 0
        assert result["cell_ranges"][0][1] == 1

    def test_build_index_empty(self):
        """Test index building with no points."""
        positions = np.array([], dtype=np.float32).reshape(0, 4)  # 4D for testing
        grid_shape = np.array([2], dtype=np.uint32)  # Grid for non-displayed dim

        result = build_spatial_index(positions, grid_shape)

        assert len(result["occupied_cells"]) == 0
        assert len(result["cell_ranges"]) == 0
        assert result["sorted_positions"].shape == (0, 4)

    def test_spatial_locality(self):
        """Test that nearby points end up in the same cell."""
        # Create clustered points in 3D (2 spatial + 1 indexed dimension)
        cluster1 = np.random.randn(50, 3).astype(np.float32) * 0.1 + [5, 5, 0]
        cluster2 = np.random.randn(50, 3).astype(np.float32) * 0.1 + [-5, -5, 0]
        positions = np.vstack([cluster1, cluster2])

        grid_shape = np.array(
            [3], dtype=np.uint32
        )  # Grid for the non-displayed dimension
        result = build_spatial_index(
            positions, grid_shape, displayed_dims=[0, 1]
        )  # Display first 2 dims

        # Should have at most 2 main occupied cells (one per cluster)
        # Plus maybe a few boundary cells
        assert len(result["occupied_cells"]) <= 4


class TestSpatialIndexQuery:
    """Test spatial index querying."""

    def test_query_index_basic(self):
        """Test basic query functionality."""
        # Build a simple index with 3D points (2 displayed, 1 indexed)
        positions = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [5, 5, 5],
                [5, 6, 5],
                [6, 5, 5],
                [6, 6, 5],
            ],
            dtype=np.float32,
        )

        grid_shape = np.array([3], dtype=np.uint32)  # Grid for the indexed dimension
        index_data = build_spatial_index(positions, grid_shape, displayed_dims=[0, 1])

        # Query around origin
        index_metadata = {
            "grid_shape": index_data["grid_shape"],
            "grid_origin": index_data["grid_origin"],
            "cell_size": index_data["cell_size"],
        }

        ranges = query_spatial_index(
            index_metadata,
            index_data["occupied_cells"],
            index_data["cell_ranges"],
            np.array([0.5], dtype=np.float32),  # Query position for indexed dim only
            np.array([2.0], dtype=np.float32),  # Tolerance for indexed dim only
        )

        # Should find points near origin
        total_points = sum(end - start for start, end in ranges)
        assert total_points >= 4  # At least the first 4 points

    def test_query_index_no_results(self):
        """Test query with no matching cells."""
        positions = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
        grid_shape = np.array([10], dtype=np.uint32)  # Grid for indexed dimension
        index_data = build_spatial_index(positions, grid_shape, displayed_dims=[0, 1])

        index_metadata = {
            "grid_shape": index_data["grid_shape"],
            "grid_origin": index_data["grid_origin"],
            "cell_size": index_data["cell_size"],
        }

        # Query far from any points in the indexed dimension
        ranges = query_spatial_index(
            index_metadata,
            index_data["occupied_cells"],
            index_data["cell_ranges"],
            np.array([100], dtype=np.float32),  # Far from z=0
            np.array([1], dtype=np.float32),
        )

        assert len(ranges) == 0

    def test_query_index_all_cells(self):
        """Test query that covers entire grid."""
        positions = np.random.randn(100, 3).astype(np.float32) * 10  # 3D points
        grid_shape = np.array([3], dtype=np.uint32)  # Grid for indexed dimension
        index_data = build_spatial_index(positions, grid_shape, displayed_dims=[0, 1])

        index_metadata = {
            "grid_shape": index_data["grid_shape"],
            "grid_origin": index_data["grid_origin"],
            "cell_size": index_data["cell_size"],
        }

        # Query with huge tolerance in the indexed dimension
        ranges = query_spatial_index(
            index_metadata,
            index_data["occupied_cells"],
            index_data["cell_ranges"],
            np.array([0], dtype=np.float32),  # Query position for indexed dim
            np.array([1000], dtype=np.float32),  # Huge tolerance
        )

        # Should return all points
        total_points = sum(end - start for start, end in ranges)
        assert total_points == 100


class TestSortOrder:
    """Test array reordering functionality."""

    def test_apply_sort_order(self):
        """Test applying sort order to arrays."""
        original = np.array([10, 20, 30, 40, 50])
        sort_order = np.array([4, 2, 0, 1, 3])

        sorted_array = apply_sort_order(original, sort_order)
        expected = np.array([50, 30, 10, 20, 40])

        np.testing.assert_array_equal(sorted_array, expected)

    def test_apply_sort_order_2d(self):
        """Test applying sort order to 2D arrays."""
        original = np.array([[1, 2], [3, 4], [5, 6]])
        sort_order = np.array([2, 0, 1])

        sorted_array = apply_sort_order(original, sort_order)
        expected = np.array([[5, 6], [1, 2], [3, 4]])

        np.testing.assert_array_equal(sorted_array, expected)

    def test_apply_sort_order_none(self):
        """Test that None arrays remain None."""
        result = apply_sort_order(None, np.array([0, 1, 2]))
        assert result is None


class TestIntegration:
    """Integration tests with zarr storage."""

    def test_round_trip_with_zarr(self, tmp_path):
        """Test building index, storing in zarr, and loading back."""
        from luxar import Dimension, Dimensions, LuxarZarrCompiler

        # Create test data
        np.random.seed(42)
        positions = np.random.randn(500, 4).astype(np.float32) * 10
        colors = np.random.randint(0, 255, (500, 3)).astype(np.float32)
        radii = np.random.uniform(0.1, 2.0, 500).astype(np.float32)

        # Create scene with spatial index
        output_path = tmp_path / "test_spatial_index.zarr"

        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
                Dimension("time", unit="ms", display=False),
            ]
        )

        with LuxarZarrCompiler(str(output_path), enable_spatial_index=True) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "test_points",
                positions,
                colors=colors,
                radii=radii,
                grid_shape=(3,),  # Only for the non-displayed time dimension
            )

        # Load back and verify
        store = zarr.open_group(str(output_path), mode="r")
        points_group = store["test_points"]

        # Check spatial index exists
        assert "spatial_index" in points_group
        index_group = points_group["spatial_index"]

        # Check metadata
        assert "grid_shape" in index_group.attrs
        assert index_group.attrs["grid_shape"] == [3]  # Only time dimension indexed
        assert index_group.attrs["dimensions"] == 1  # Only 1 indexed dimension
        assert index_group.attrs["indexed_dimensions"] == [3]  # Time is dimension 3
        assert index_group.attrs["displayed_dimensions"] == [
            0,
            1,
            2,
        ]  # x, y, z displayed
        assert index_group.attrs["build_version"] == "0.5"  # Updated version

        # Check arrays
        assert "occupied_cells" in index_group
        assert "cell_ranges" in index_group

        # Check that occupied_cells exists (we just need to verify it's there)
        assert index_group["occupied_cells"].shape[0] > 0
        cell_ranges = index_group["cell_ranges"][:]

        # Verify all points are covered
        total_points = 0
        for start, end in cell_ranges:
            total_points += end - start
        assert total_points == 500

        # Verify data is sorted
        loaded_positions = points_group["positions"][:]
        loaded_colors = points_group["colors"][:]
        loaded_radii = points_group["radii"][:]

        # Data should be reordered but have same statistics
        assert loaded_positions.shape == positions.shape
        assert loaded_colors.shape == colors.shape
        assert loaded_radii.shape == radii.shape

        # Same data, different order
        assert np.allclose(
            np.sort(loaded_positions.flatten()), np.sort(positions.flatten()), rtol=1e-5
        )
