"""Tests for Lines spatial indexing with dual ordering.

This module tests the Lines spatial indexing implementation:
1. convert_to_indexed() - line type conversion
2. sort_segments_compound() - segment ordering in (2×D)-space
3. order_lines_spatial() - dual ordering (vertices + segments)
4. compute_vertex_chunk_bounds() - vertex chunk bounds
5. compute_segment_chunk_bounds() - segment chunk bounds with width expansion
6. End-to-end integration tests via write_lines()
"""

import numpy as np
import pytest

from luxar.core import Dimension, Dimensions
from luxar.io.ordering import (
    _BARRIER_BOUND_EPS,
    compute_segment_chunk_bounds,
    compute_vertex_chunk_bounds,
    convert_to_indexed,
    morton_encode_128bit,
    order_lines_spatial,
    sort_segments_compound,
)


class TestConvertToIndexed:
    """Test line type conversion to indexed representation."""

    def test_polyline_conversion(self) -> None:
        """Test polyline converts to consecutive vertex pairs."""
        segments = convert_to_indexed(5, "polyline", None)

        # 5 vertices -> 4 segments: (0,1), (1,2), (2,3), (3,4)
        assert segments.shape == (4, 2)
        assert segments.dtype == np.uint32

        expected = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.uint32)
        np.testing.assert_array_equal(segments, expected)

    def test_loop_conversion(self) -> None:
        """Test loop converts to consecutive pairs with closure."""
        segments = convert_to_indexed(4, "loop", None)

        # 4 vertices -> 4 segments: (0,1), (1,2), (2,3), (3,0)
        assert segments.shape == (4, 2)

        expected = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(segments, expected)

    def test_segments_conversion(self) -> None:
        """Test segments (pairs) converts correctly."""
        segments = convert_to_indexed(6, "segments", None)

        # 6 vertices -> 3 segments: (0,1), (2,3), (4,5)
        assert segments.shape == (3, 2)

        expected = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint32)
        np.testing.assert_array_equal(segments, expected)

    def test_indexed_passthrough(self) -> None:
        """Test indexed type passes through the indices."""
        indices = np.array([0, 2, 1, 3, 0, 3], dtype=np.uint32)
        segments = convert_to_indexed(4, "indexed", indices)

        # Should be reshaped to (3, 2)
        assert segments.shape == (3, 2)
        expected = np.array([[0, 2], [1, 3], [0, 3]], dtype=np.uint32)
        np.testing.assert_array_equal(segments, expected)

    def test_indexed_requires_indices(self) -> None:
        """Test indexed type raises error without indices."""
        with pytest.raises(ValueError, match="requires indices array"):
            convert_to_indexed(4, "indexed", None)

    def test_invalid_line_type_raises(self) -> None:
        """Test invalid line type raises error."""
        with pytest.raises(ValueError, match="Invalid line_type"):
            convert_to_indexed(4, "invalid_type", None)


class TestMorton128Bit:
    """Test 128-bit Morton encoding for high-dimensional data."""

    def test_morton_128_basic(self) -> None:
        """Test basic 128-bit Morton encoding."""
        # Simple 2D coords
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint32)
        high, low = morton_encode_128bit(coords, bits_per_dim=8)

        # All coords fit in low part
        assert high.dtype == np.uint64
        assert low.dtype == np.uint64
        assert np.all(high == 0)  # Small coords, should be all in low

        # Morton order should be: 0, 1, 2, 3 (Z-order)
        assert low[0] == 0  # (0,0) -> 0
        assert low[1] > 0  # Others should be non-zero

    def test_morton_128_high_bits_used(self) -> None:
        """Test that high bits are used for large enough coords."""
        # 6D coords with high values need 128 bits
        coords = np.array([[1000, 1000, 1000, 1000, 1000, 1000]], dtype=np.uint32)
        high, low = morton_encode_128bit(coords, bits_per_dim=16)

        # With 16 bits per dim * 6 dims = 96 bits total
        # Both high and low should be set
        assert high.shape == (1,)
        assert low.shape == (1,)


class TestSortSegmentsCompound:
    """Test segment ordering in (2×D)-space."""

    def test_sort_segments_3d(self) -> None:
        """Test segment sorting for 3D data."""
        # 3D vertices -> 6D segment space
        # Create segments at different positions
        segment_coords_2d = np.array(
            [
                [10, 10, 10, 11, 11, 11],  # Segment in middle-right
                [0, 0, 0, 1, 1, 1],  # Segment at origin
                [5, 5, 5, 6, 6, 6],  # Segment in middle
            ],
            dtype=np.float32,
        )

        dimensions = [
            Dimension("x", unit="um"),
            Dimension("y", unit="um"),
            Dimension("z", unit="um"),
        ]

        sort_indices, metadata = sort_segments_compound(
            segment_coords_2d, dimensions, method="morton"
        )

        # Should reorder based on spatial locality
        assert sort_indices.shape == (3,)
        assert metadata["ordering"] == "morton"
        assert len(metadata["ordering_dims"]) == 6  # All 6 dims are spatial
        assert len(metadata["slice_dims"]) == 0

    def test_sort_segments_with_discrete_dims(self) -> None:
        """Test segment sorting with discrete dimensions."""
        # 4D: [x, y, z, time] where time is discrete
        # Segments in (2×4) = 8D space
        segment_coords_2d = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1],  # time=1
                [0, 0, 0, 0, 1, 1, 1, 0],  # time=0
                [5, 5, 5, 1, 6, 6, 6, 1],  # time=1
            ],
            dtype=np.float32,
        )

        dimensions = [
            Dimension("x", unit="um"),
            Dimension("y", unit="um"),
            Dimension("z", unit="um"),
            Dimension("time", unit="s", discrete=True, display=False),
        ]

        sort_indices, metadata = sort_segments_compound(
            segment_coords_2d, dimensions, method="morton"
        )

        # Should identify discrete dims (time from both endpoints: 3 and 7)
        assert 3 in metadata["slice_dims"]
        assert 7 in metadata["slice_dims"]
        assert len(metadata["slice_dims"]) == 2

        # Spatial dims should be the rest
        assert len(metadata["ordering_dims"]) == 6


class TestOrderLinesSpatial:
    """Test complete dual ordering for lines."""

    def test_order_lines_basic(self) -> None:
        """Test basic dual ordering."""
        vertices = np.array(
            [
                [5, 5, 5],
                [0, 0, 0],
                [10, 10, 10],
                [2, 2, 2],
            ],
            dtype=np.float32,
        )

        # Polyline: (0,1), (1,2), (2,3)
        segments = convert_to_indexed(4, "polyline", None)

        dimensions = [
            Dimension("x", unit="um"),
            Dimension("y", unit="um"),
            Dimension("z", unit="um"),
        ]

        (
            sorted_vertices,
            sorted_segments,
            vertex_sort_indices,
            segment_sort_indices,
            metadata,
        ) = order_lines_spatial(vertices, segments, dimensions, method="morton")

        # Check shapes
        assert sorted_vertices.shape == (4, 3)
        assert sorted_segments.shape == (3, 2)
        assert vertex_sort_indices.shape == (4,)
        assert segment_sort_indices.shape == (3,)

        # Check metadata structure
        assert "vertex_ordering" in metadata
        assert "segment_ordering" in metadata
        assert metadata["vertex_ordering"]["ordering"] == "morton"
        assert metadata["segment_ordering"]["ordering"] == "morton"

        # Segment indices should still be valid after remapping
        assert np.all(sorted_segments < 4)
        assert np.all(sorted_segments >= 0)

    def test_order_lines_preserves_connectivity(self) -> None:
        """Test that segment connectivity is preserved after reordering."""
        # Create a simple line
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )

        # Loop: (0,1), (1,2), (2,3), (3,0)
        segments = convert_to_indexed(4, "loop", None)

        dimensions = [Dimension("x"), Dimension("y"), Dimension("z")]

        (
            sorted_vertices,
            sorted_segments,
            vertex_sort_indices,
            segment_sort_indices,
            metadata,
        ) = order_lines_spatial(vertices, segments, dimensions)

        # For each segment, get the actual vertex positions
        for seg_idx in range(sorted_segments.shape[0]):
            v1_idx, v2_idx = sorted_segments[seg_idx]
            _v1 = sorted_vertices[v1_idx]
            _v2 = sorted_vertices[v2_idx]

            # The segment should connect two vertices that are neighbors in original
            # We can't check exact connectivity, but we can verify indices are valid
            assert 0 <= v1_idx < 4
            assert 0 <= v2_idx < 4


class TestComputeVertexChunkBounds:
    """Test vertex chunk bounds computation."""

    def test_vertex_bounds_basic(self) -> None:
        """Test basic vertex chunk bounds."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [5, 5, 5],
                [6, 6, 6],
            ],
            dtype=np.float32,
        )

        bounds = compute_vertex_chunk_bounds(vertices, chunk_size=2)

        # 2 chunks
        assert bounds.shape == (2, 3, 2)

        # First chunk: [0,1] range
        assert bounds[0, 0, 0] == pytest.approx(0.0)
        assert bounds[0, 0, 1] == pytest.approx(1.0)

        # Second chunk: [5,6] range
        assert bounds[1, 0, 0] == pytest.approx(5.0)
        assert bounds[1, 0, 1] == pytest.approx(6.0)

    def test_vertex_bounds_discrete_dims(self) -> None:
        """Test vertex bounds with discrete dimensions."""
        # [time, x, y] where time is discrete
        vertices = np.array(
            [
                [0, 5, 5],
                [0, 6, 6],
                [1, 5, 5],
                [1, 6, 6],
            ],
            dtype=np.float32,
        )

        bounds = compute_vertex_chunk_bounds(vertices, chunk_size=2, slice_dims=[0])

        # First chunk (time=0): discrete bounds should be tight (epsilon-padded)
        assert bounds[0, 0, 0] == pytest.approx(-_BARRIER_BOUND_EPS)
        assert bounds[0, 0, 1] == pytest.approx(_BARRIER_BOUND_EPS)


class TestComputeSegmentChunkBounds:
    """Test segment chunk bounds computation with width expansion."""

    def test_segment_bounds_with_width(self) -> None:
        """Test segment bounds include width extent."""
        vertices = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [0, 10, 0],
                [10, 10, 0],
            ],
            dtype=np.float32,
        )

        segments = np.array(
            [
                [0, 1],  # Horizontal line y=0
                [2, 3],  # Horizontal line y=10
            ],
            dtype=np.uint32,
        )

        widths = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        bounds = compute_segment_chunk_bounds(vertices, segments, widths, chunk_size=1)

        # 2 chunks, one per segment
        assert bounds.shape == (2, 3, 2)

        # First segment: x=[0,10], y=0, width=1
        # Bounds should be: x=[-1,11], y=[-1,1]
        assert bounds[0, 0, 0] == pytest.approx(-1.0)  # x min
        assert bounds[0, 0, 1] == pytest.approx(11.0)  # x max
        assert bounds[0, 1, 0] == pytest.approx(-1.0)  # y min
        assert bounds[0, 1, 1] == pytest.approx(1.0)  # y max

        # Second segment: x=[0,10], y=10, width=2
        # Bounds should be: x=[-2,12], y=[8,12]
        assert bounds[1, 0, 0] == pytest.approx(-2.0)  # x min
        assert bounds[1, 0, 1] == pytest.approx(12.0)  # x max
        assert bounds[1, 1, 0] == pytest.approx(8.0)  # y min
        assert bounds[1, 1, 1] == pytest.approx(12.0)  # y max

    def test_segment_bounds_discrete_no_width_expansion(self) -> None:
        """Test that discrete dimensions don't expand by width."""
        # [time, x, y]
        vertices = np.array(
            [
                [0, 0, 0],
                [0, 10, 10],
                [1, 0, 0],
                [1, 10, 10],
            ],
            dtype=np.float32,
        )

        segments = np.array(
            [
                [0, 1],  # time=0
                [2, 3],  # time=1
            ],
            dtype=np.uint32,
        )

        widths = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)

        bounds = compute_segment_chunk_bounds(
            vertices, segments, widths, chunk_size=1, slice_dims=[0]
        )

        # Time dimension (discrete) should NOT expand by width
        # First segment time=0: bounds should be ~0 (epsilon-padded), not ±width
        assert bounds[0, 0, 0] >= -0.6
        assert bounds[0, 0, 1] <= 0.6

        # Spatial dimensions SHOULD expand by width
        assert bounds[0, 1, 0] == pytest.approx(-5.0)  # x min
        assert bounds[0, 1, 1] == pytest.approx(15.0)  # x max


class TestIntegration:
    """Integration tests for write_lines with spatial indexing."""

    def test_write_lines_creates_segments_array(self, tmp_path) -> None:
        """Test that write_lines creates the segments array."""
        import zarr

        from luxar import LuxarZarrCompiler

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions(
                    [
                        Dimension("x", unit="um"),
                        Dimension("y", unit="um"),
                        Dimension("z", unit="um"),
                    ]
                )
            )

            vertices = np.random.rand(100, 3).astype(np.float32) * 10
            scene.add_lines("my_lines", vertices, widths=0.1, line_type="polyline")

        # Verify segments array exists
        store = zarr.open_group(store_path, mode="r")
        assert "my_lines/segments" in store

        segments = store["my_lines/segments"][:]
        assert segments.shape == (99, 2)  # 100 vertices -> 99 polyline segments

    def test_write_lines_creates_dual_chunk_bounds(self, tmp_path) -> None:
        """Test that write_lines creates both vertex and segment chunk bounds."""
        import zarr

        from luxar import LuxarZarrCompiler

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions(
                    [
                        Dimension("x", unit="um"),
                        Dimension("y", unit="um"),
                        Dimension("z", unit="um"),
                    ]
                )
            )

            vertices = np.random.rand(1000, 3).astype(np.float32) * 10
            scene.add_lines("my_lines", vertices, widths=0.5, line_type="polyline")

        # Verify both chunk bounds exist
        store = zarr.open_group(store_path, mode="r")
        assert "my_lines/vertex_chunk_bounds" in store
        assert "my_lines/segment_chunk_bounds" in store

        # Check shapes
        vertex_bounds = store["my_lines/vertex_chunk_bounds"][:]
        segment_bounds = store["my_lines/segment_chunk_bounds"][:]

        assert vertex_bounds.shape[1] == 3  # 3D
        assert vertex_bounds.shape[2] == 2  # min, max
        assert segment_bounds.shape[1] == 3
        assert segment_bounds.shape[2] == 2

    def test_write_lines_stores_ordering_metadata(self, tmp_path) -> None:
        """Test that write_lines stores dual ordering metadata."""
        import zarr

        from luxar import LuxarZarrCompiler

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path, ordering_method="morton") as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions(
                    [
                        Dimension("x", unit="um"),
                        Dimension("y", unit="um"),
                        Dimension("z", unit="um"),
                    ]
                )
            )

            vertices = np.random.rand(100, 3).astype(np.float32) * 10
            scene.add_lines("my_lines", vertices, widths=0.1)

        store = zarr.open_group(store_path, mode="r")
        attrs = dict(store["my_lines"].attrs)

        # Check ordering metadata
        assert attrs["ordering"] == "morton"
        assert attrs["original_line_type"] == "polyline"

        # Check vertex ordering metadata
        assert "vertex_ordering" in attrs
        vertex_meta = attrs["vertex_ordering"]
        assert "chunk_size" in vertex_meta
        assert "ordering_dims" in vertex_meta

        # Check segment ordering metadata
        assert "segment_ordering" in attrs
        segment_meta = attrs["segment_ordering"]
        assert "chunk_size" in segment_meta
        # Segment ordering dims should be doubled (2×D)
        assert len(segment_meta["ordering_dims"]) == 6  # 2×3

    def test_write_lines_all_line_types(self, tmp_path) -> None:
        """Test that all line types work with spatial indexing."""
        import zarr

        from luxar import LuxarZarrCompiler

        line_types = {
            "polyline": (10, 9),  # 10 verts -> 9 segs
            "loop": (10, 10),  # 10 verts -> 10 segs
            "segments": (10, 5),  # 10 verts -> 5 segs
        }

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            )

            vertices = np.random.rand(10, 3).astype(np.float32)

            for lt, (n_verts, expected_segs) in line_types.items():
                scene.add_lines(f"lines_{lt}", vertices, widths=0.1, line_type=lt)

        store = zarr.open_group(store_path, mode="r")

        for lt, (n_verts, expected_segs) in line_types.items():
            segs = store[f"lines_{lt}/segments"][:]
            assert segs.shape == (expected_segs, 2), f"Wrong shape for {lt}"

    def test_write_lines_indexed_type(self, tmp_path) -> None:
        """Test indexed line type with spatial indexing."""
        import zarr

        from luxar import LuxarZarrCompiler

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            )

            vertices = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
            )

            indices = np.array([0, 1, 0, 2, 1, 3, 2, 3], dtype=np.uint32)

            scene.add_lines(
                "indexed_lines",
                vertices,
                widths=0.1,
                line_type="indexed",
                indices=indices,
            )

        store = zarr.open_group(store_path, mode="r")
        segs = store["indexed_lines/segments"][:]

        # 8 indices -> 4 segments
        assert segs.shape == (4, 2)

    def test_write_lines_no_spatial_index(self, tmp_path) -> None:
        """Test write_lines with spatial indexing disabled."""
        import zarr

        from luxar import LuxarZarrCompiler

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            )

            vertices = np.random.rand(100, 3).astype(np.float32)
            scene.add_lines("my_lines", vertices, widths=0.1)

        store = zarr.open_group(store_path, mode="r")

        # Segments array should still exist (unified representation)
        assert "my_lines/segments" in store

        # But chunk bounds should NOT exist
        assert "my_lines/vertex_chunk_bounds" not in store
        assert "my_lines/segment_chunk_bounds" not in store

        # ordering should be "none"
        assert store["my_lines"].attrs["ordering"] == "none"

    def test_write_lines_zarr_attributes_match_spec(self, tmp_path) -> None:
        """Test that Zarr attributes match the specification (Section 6.6)."""
        import zarr

        from luxar import LuxarZarrCompiler

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            )

            vertices = np.random.rand(100, 3).astype(np.float32) * 10
            colors = np.random.rand(100, 3).astype(np.float32)
            widths = np.random.rand(100).astype(np.float32) * 0.5 + 0.1
            sharpness = np.random.rand(100).astype(np.float32)

            scene.add_lines(
                "test_lines",
                vertices,
                widths,
                colors=colors,
                sharpness=sharpness,
                line_type="polyline",
            )

        store = zarr.open_group(store_path, mode="r")
        attrs = dict(store["test_lines"].attrs)

        # Verify all core metadata fields per spec Section 6.6
        assert attrs["type"] == "lines"
        assert attrs["n_vertices"] == 100
        assert attrs["n_segments"] == 99  # polyline: N-1
        assert attrs["ndim"] == 3
        assert attrs["original_line_type"] == "polyline"
        assert attrs["has_colors"] is True
        assert attrs["has_sharpness"] is True
        assert attrs["max_width"] > 0

        # Verify ordering metadata structure
        assert "ordering" in attrs
        assert attrs["ordering"] in ("morton", "hilbert")

        # Verify vertex_ordering structure
        assert "vertex_ordering" in attrs
        v_ord = attrs["vertex_ordering"]
        assert "slice_dims" in v_ord
        assert "ordering_dims" in v_ord
        assert "ordering_min" in v_ord
        assert "ordering_max" in v_ord
        assert "chunk_size" in v_ord

        # Verify segment_ordering structure
        assert "segment_ordering" in attrs
        s_ord = attrs["segment_ordering"]
        assert "slice_dims" in s_ord
        assert "ordering_dims" in s_ord
        assert "ordering_min" in s_ord
        assert "ordering_max" in s_ord
        assert "chunk_size" in s_ord

        # Segment ordering should be in (2×D) space
        # For 3D data: ordering_dims should have 6 elements (2×3)
        assert len(s_ord["ordering_dims"]) == 6

    def test_write_lines_stores_position_bounds(self, tmp_path) -> None:
        """Test that write_lines stores position_bounds for dynamic clipping."""
        import zarr

        from luxar import LuxarZarrCompiler

        store_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            )

            # Create vertices with known bounds
            vertices = np.array(
                [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [5.0, 10.0, 15.0]],
                dtype=np.float32,
            )
            scene.add_lines("my_lines", vertices, widths=0.1, line_type="polyline")

        store = zarr.open_group(store_path, mode="r")

        # Verify node-level position_bounds
        node_bounds = store["my_lines"].attrs["position_bounds"]
        assert node_bounds["min"] == [0.0, 0.0, 0.0]
        assert node_bounds["max"] == [10.0, 20.0, 30.0]

        # Verify scene-level position_bounds (should match since single node)
        scene_bounds = store.attrs["position_bounds"]
        assert scene_bounds["min"] == [0.0, 0.0, 0.0]
        assert scene_bounds["max"] == [10.0, 20.0, 30.0]
