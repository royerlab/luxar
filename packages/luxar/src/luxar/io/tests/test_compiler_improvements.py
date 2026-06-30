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
    """Test that the version is correctly set to 0.1."""

    def test_compiler_writes_correct_version(self) -> None:
        """Verify compiler writes version 0.1 to zarr attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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


class TestChunkBoundsZarrAlignment:
    """Test that chunk_bounds spatial partitions align with zarr chunk boundaries.

    This guards against the bug where the compiler computed spatial index
    partitions with one chunk_size but wrote zarr arrays with a different
    chunk_size, causing the viewer to fetch misaligned data.
    """

    # -- GSplats alignment ---------------------------------------------------

    def test_gsplats_all_arrays_aligned(self) -> None:
        """All gsplats zarr arrays must have chunks[0] == spatial chunk_size."""
        from math import ceil

        from luxar.core.dimensions import Dimension, Dimensions
        from luxar.gsplats.utils.trils import tril_size

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            n_splats = 2500
            ndim = 4
            k = tril_size(ndim)

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", display=True),
                        Dimension("y", display=True),
                        Dimension("z", display=True),
                        Dimension("t", display=False, discrete=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                centers = np.random.randn(n_splats, ndim).astype(np.float32)
                amplitudes = np.random.rand(n_splats).astype(np.float32)
                cholesky = np.random.randn(n_splats, k).astype(np.float32)
                colors = np.random.rand(n_splats, 3).astype(np.float32)

                scene.add_gsplats(
                    "splats",
                    centers=centers,
                    amplitudes=amplitudes,
                    cholesky_factors=cholesky,
                    colors=colors,
                )

            store = zarr.open_group(zarr_path, mode="r")
            g = store["splats"]
            chunk_size = g.attrs["chunk_size"]
            assert chunk_size > 0, "chunk_size metadata must be positive"

            # Every array's first chunk dimension must match chunk_size
            assert g["centers"].chunks[0] == chunk_size, (
                f"centers chunks[0]={g['centers'].chunks[0]} != chunk_size={chunk_size}"
            )
            assert g["amplitudes"].chunks[0] == chunk_size, (
                f"amplitudes chunks[0]={g['amplitudes'].chunks[0]} != chunk_size={chunk_size}"
            )
            # v3.1: Cholesky stored as a diagonal + off-diagonal split; both
            # halves share the same row-chunk size as the other arrays.
            assert g["cholesky_factors_diag"].chunks[0] == chunk_size, (
                f"cholesky diag chunks[0]={g['cholesky_factors_diag'].chunks[0]} "
                f"!= chunk_size={chunk_size}"
            )
            assert g["cholesky_factors_offdiag"].chunks[0] == chunk_size, (
                f"cholesky offdiag chunks[0]={g['cholesky_factors_offdiag'].chunks[0]} "
                f"!= chunk_size={chunk_size}"
            )
            assert g["colors"].chunks[0] == chunk_size, (
                f"colors chunks[0]={g['colors'].chunks[0]} != chunk_size={chunk_size}"
            )

            # chunk_bounds partitions must match number of zarr chunks
            cb = np.array(g["chunk_bounds"])
            expected_partitions = ceil(n_splats / chunk_size)
            assert cb.shape[0] == expected_partitions, (
                f"chunk_bounds has {cb.shape[0]} partitions but expected "
                f"{expected_partitions} (ceil({n_splats}/{chunk_size}))"
            )

            # Zarr chunk count along first axis must equal partition count
            zarr_n_chunks = ceil(g["centers"].shape[0] / g["centers"].chunks[0])
            assert zarr_n_chunks == cb.shape[0], (
                f"zarr has {zarr_n_chunks} chunks but chunk_bounds has {cb.shape[0]} partitions"
            )

    def test_gsplats_small_dataset_single_chunk(self) -> None:
        """GSplats with fewer splats than chunk_size should produce 1 partition."""
        from luxar.core.dimensions import Dimension, Dimensions
        from luxar.gsplats.utils.trils import tril_size

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            n_splats = 100
            ndim = 3
            k = tril_size(ndim)

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", display=True),
                        Dimension("y", display=True),
                        Dimension("z", display=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                centers = np.random.randn(n_splats, ndim).astype(np.float32)
                amplitudes = np.random.rand(n_splats).astype(np.float32)
                cholesky = np.random.randn(n_splats, k).astype(np.float32)

                scene.add_gsplats(
                    "small",
                    centers=centers,
                    amplitudes=amplitudes,
                    cholesky_factors=cholesky,
                )

            store = zarr.open_group(zarr_path, mode="r")
            g = store["small"]
            chunk_size = g.attrs["chunk_size"]

            # chunk_size should be clamped to n_splats
            assert chunk_size >= n_splats, (
                f"chunk_size={chunk_size} should be >= n_splats={n_splats}"
            )

            # Exactly 1 chunk_bounds partition and 1 zarr chunk
            cb = np.array(g["chunk_bounds"])
            assert cb.shape[0] == 1, f"Expected 1 partition, got {cb.shape[0]}"
            assert g["centers"].chunks[0] >= n_splats, (
                f"centers chunks[0]={g['centers'].chunks[0]} should contain all {n_splats} splats"
            )

    # -- Points alignment ----------------------------------------------------

    def test_points_all_attributes_aligned(self) -> None:
        """All point attribute arrays must share the same chunk_size as positions."""
        from math import ceil

        from luxar.core.dimensions import Dimension, Dimensions

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            n_points = 10000

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", display=True),
                        Dimension("y", display=True),
                        Dimension("z", display=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                positions = np.random.randn(n_points, 3).astype(np.float32)
                colors = np.random.rand(n_points, 3).astype(np.float32)
                radii = np.random.rand(n_points).astype(np.float32) + 0.1
                # Use non-uniform sharpness so the encoder doesn't broadcast it
                sharpness = np.random.uniform(0.2, 0.9, n_points).astype(np.float32)

                scene.add_points(
                    "pts",
                    positions=positions,
                    colors=colors,
                    radii=radii,
                    sharpness=sharpness,
                )

            store = zarr.open_group(zarr_path, mode="r")
            g = store["pts"]
            chunk_size = g.attrs["chunk_size"]
            pos_chunk0 = g["positions"].chunks[0]

            # Positions must match chunk_size
            assert pos_chunk0 == chunk_size, (
                f"positions chunks[0]={pos_chunk0} != chunk_size={chunk_size}"
            )

            # All attributes must match positions chunk[0]
            assert g["colors"].chunks[0] == pos_chunk0, (
                f"colors chunks[0]={g['colors'].chunks[0]} != positions chunks[0]={pos_chunk0}"
            )
            assert g["radii"].chunks[0] == pos_chunk0, (
                f"radii chunks[0]={g['radii'].chunks[0]} != positions chunks[0]={pos_chunk0}"
            )
            assert g["sharpnesses"].chunks[0] == pos_chunk0, (
                f"sharpnesses chunks[0]={g['sharpnesses'].chunks[0]} != positions chunks[0]={pos_chunk0}"
            )

            # chunk_bounds partitions == zarr chunk count
            cb = np.array(g["chunk_bounds"])
            expected = ceil(n_points / chunk_size)
            assert cb.shape[0] == expected, (
                f"chunk_bounds has {cb.shape[0]} partitions, expected {expected}"
            )

    def test_points_4d_with_discrete_dim(self) -> None:
        """4D points with discrete time dim must also align all attributes."""
        from luxar.core.dimensions import Dimension, Dimensions

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            n_points = 5000

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", display=True),
                        Dimension("y", display=True),
                        Dimension("z", display=True),
                        Dimension("t", display=False, discrete=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                positions = np.random.randn(n_points, 4).astype(np.float32)
                # Assign discrete time values (0, 1, 2, ...)
                positions[:, 3] = np.random.randint(0, 10, n_points).astype(np.float32)
                colors = np.random.rand(n_points, 3).astype(np.float32)
                radii = np.random.rand(n_points).astype(np.float32) + 0.1

                scene.add_points(
                    "pts4d",
                    positions=positions,
                    colors=colors,
                    radii=radii,
                )

            store = zarr.open_group(zarr_path, mode="r")
            g = store["pts4d"]
            chunk_size = g.attrs["chunk_size"]
            pos_chunk0 = g["positions"].chunks[0]

            assert pos_chunk0 == chunk_size
            assert g["colors"].chunks[0] == pos_chunk0
            assert g["radii"].chunks[0] == pos_chunk0

    # -- Lines alignment -----------------------------------------------------

    def test_lines_all_attributes_aligned(self) -> None:
        """All line attribute arrays must share the vertex chunk_size."""
        from luxar.core.dimensions import Dimension, Dimensions

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            n_vertices = 5000

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", display=True),
                        Dimension("y", display=True),
                        Dimension("z", display=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                # Need even number for segments (pairs of vertices)
                vertices = np.random.randn(n_vertices, 3).astype(np.float32)
                widths = np.random.rand(n_vertices).astype(np.float32) + 0.01
                colors = np.random.rand(n_vertices, 3).astype(np.float32)
                # Use non-uniform sharpness so the encoder doesn't broadcast it
                sharpness_arr = np.random.uniform(0.2, 0.9, n_vertices).astype(
                    np.float32
                )

                scene.add_lines(
                    "lines",
                    vertices=vertices,
                    widths=widths,
                    line_type="segments",
                    colors=colors,
                    sharpness=sharpness_arr,
                )

            store = zarr.open_group(zarr_path, mode="r")
            g = store["lines"]

            # Get vertex chunk_size from ordering metadata
            vtx_chunk0 = g["vertices"].chunks[0]

            # All vertex-indexed attributes must match
            assert g["widths"].chunks[0] == vtx_chunk0, (
                f"widths chunks[0]={g['widths'].chunks[0]} != vertices chunks[0]={vtx_chunk0}"
            )
            assert g["colors"].chunks[0] == vtx_chunk0, (
                f"colors chunks[0]={g['colors'].chunks[0]} != vertices chunks[0]={vtx_chunk0}"
            )
            assert g["sharpnesses"].chunks[0] == vtx_chunk0, (
                f"sharpnesses chunks[0]={g['sharpnesses'].chunks[0]} != vertices chunks[0]={vtx_chunk0}"
            )

    # -- No spatial index (regression guard) ---------------------------------

    def test_no_spatial_index_still_works(self) -> None:
        """Without spatial ordering, standard chunking should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=False) as compiler:
                dims = Dimensions.default_3d()
                scene = compiler.create_scene(dimensions=dims)
                positions = np.random.randn(5000, 3).astype(np.float32)
                colors = np.random.rand(5000, 3).astype(np.float32)
                radii = np.random.rand(5000).astype(np.float32) + 0.1

                scene.add_points("pts", positions, colors=colors, radii=radii)

            store = zarr.open_group(zarr_path, mode="r")
            g = store["pts"]

            # Should have valid chunks (no assertion on exact values, just sanity)
            assert g["positions"].chunks[0] > 0
            assert g["positions"].chunks[1] == 3
            assert g["colors"].chunks[0] > 0
            assert g["radii"].chunks[0] > 0

            # Should NOT have chunk_bounds
            assert "chunk_bounds" not in g

    # -- Unit test for calculate_intelligent_chunks -------------------------

    def testcalculate_intelligent_chunks_1d_uses_spatial_data(self) -> None:
        """calculate_intelligent_chunks must use spatial chunk_size for 1D arrays."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        spatial = {"chunk_size": 512}

        # Without spatial data: uses byte-based default (64 KiB / float32)
        result = calculate_intelligent_chunks((10000,))
        assert result == (10000,)

        # With spatial data: uses chunk_size
        result = calculate_intelligent_chunks((10000,), spatial_index_data=spatial)
        assert result == (512,)

        # Small array clamped to actual size
        result = calculate_intelligent_chunks((100,), spatial_index_data=spatial)
        assert result == (100,)

    def testcalculate_intelligent_chunks_2d_uses_spatial_data(self) -> None:
        """calculate_intelligent_chunks must use spatial chunk_size for 2D arrays."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        spatial = {"chunk_size": 1024}

        # Without spatial data: uses byte-based default (64 KiB / float32 / 4 dims)
        result = calculate_intelligent_chunks((5000, 4))
        assert result[1] == 4
        assert result[0] == min(5000, (65536 // 4) // 4)

        # With spatial data: uses chunk_size
        result = calculate_intelligent_chunks((5000, 4), spatial_index_data=spatial)
        assert result == (1024, 4)

    # [Python-R6 / io-MAJOR] Dtype awareness — the byte-target heuristic
    # MUST scale chunk size by element size. A uint8 array gets 4x as
    # many elements per chunk as a float32 array of the same byte target
    # (1 byte vs 4 bytes per element). A regression that hard-coded
    # itemsize=4 would silently under-chunk uint8 colors / uint16 LUTs.
    def testcalculate_intelligent_chunks_scales_with_dtype_itemsize(self) -> None:
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        # Float32: 65536 / 4 = 16384 elements per chunk
        f32 = calculate_intelligent_chunks((100_000,), dtype=np.dtype(np.float32))
        # Uint8: 65536 / 1 = 65536 elements per chunk (4x more)
        u8 = calculate_intelligent_chunks((100_000,), dtype=np.dtype(np.uint8))
        # Uint16: 65536 / 2 = 32768 elements per chunk (2x more than f32)
        u16 = calculate_intelligent_chunks((100_000,), dtype=np.dtype(np.uint16))

        assert f32 == (16384,)
        assert u8 == (65536,)
        assert u16 == (32768,)
        # Ratio invariant: u8 / f32 == 4, u16 / f32 == 2 (catches a
        # regression that broke the formula without touching values).
        assert u8[0] == 4 * f32[0]
        assert u16[0] == 2 * f32[0]

    def testcalculate_intelligent_chunks_clamps_small_arrays(self) -> None:
        """If the dataset is smaller than the byte-target derived chunk,
        the chunk shape matches the dataset shape exactly. A regression
        that returned a chunk LARGER than the array would crash zarr."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        # 50 elements * 4 bytes = 200 bytes << 64 KiB target.
        result = calculate_intelligent_chunks((50,), dtype=np.dtype(np.float32))
        assert result == (50,)  # clamped to actual array size

        # 2D: shape smaller than target → match shape exactly.
        result = calculate_intelligent_chunks((50, 4), dtype=np.dtype(np.float32))
        assert result == (50, 4)

    def testcalculate_intelligent_chunks_handles_4d_shape(self) -> None:
        """4D+ shapes use byte-based defaults per-dimension. Pin the
        contract: every dim is clamped to min(shape_dim, target_elements)."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        # 4D shape with large dims; expect every chunk dim equal to
        # min(shape_dim, target_elements=16384 for float32).
        result = calculate_intelligent_chunks(
            (1000, 200, 100, 50), dtype=np.dtype(np.float32)
        )
        assert len(result) == 4
        target = 65536 // 4  # 16384 for float32
        assert result == tuple(min(s, target) for s in (1000, 200, 100, 50))


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
        """Test that list format is treated as row-major (NumPy convention)."""
        # Row-major format: translation at indices [3, 7, 11] (last column)
        # This is translate(5, 6, 7) flattened in row-major order
        transform_list = [1, 0, 0, 5, 0, 1, 0, 6, 0, 0, 1, 7, 0, 0, 0, 1]
        result = prepare_transform_for_zarr(transform_list)

        assert isinstance(result, list)
        assert len(result) == 16
        # After row-major → column-major conversion, translations at indices 12, 13, 14
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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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

    def test_delete_group_attr_missing_path_no_group_created(self) -> None:
        """Deleting attributes from missing groups should not create groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                compiler.delete_group_attr("missing/group", "transform")

            store = zarr.open_group(zarr_path, mode="r")
            with pytest.raises(KeyError):
                _ = store["missing"]


class TestSpatialOrdering:
    """Test spatial ordering with Morton/Hilbert curves."""

    def test_spatial_ordering_in_compiler(self) -> None:
        """Test that compiler applies spatial ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

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


class TestCalculateIntelligentChunksDtype:
    """CC-1-r: chunk-size heuristic must scale with the array dtype itemsize.

    Pre-fix the helper accepted ``itemsize: int = 4`` which silently
    under-chunked any non-float32 caller. The current API takes ``dtype=``
    so the contract is explicit at the call site.
    """

    def test_chunk_size_scales_with_dtype_itemsize(self) -> None:
        from luxar.io._compiler.chunking import calculate_intelligent_chunks
        from luxar.typing_utils.constants import (
            MAX_CHUNK_BYTES,
            MIN_CHUNK_BYTES,
            TARGET_CHUNK_BYTES,
        )

        shape = (1_000_000, 3)

        for dtype_str in ("uint8", "uint16", "float32", "float64"):
            dtype = np.dtype(dtype_str)
            chunks = calculate_intelligent_chunks(shape, dtype=dtype)
            chunk_rows = chunks[0]
            chunk_bytes = chunk_rows * shape[1] * dtype.itemsize

            # Every dtype should land within the byte-target band.
            assert MIN_CHUNK_BYTES <= chunk_bytes <= MAX_CHUNK_BYTES, (
                f"{dtype_str}: {chunk_bytes} bytes outside "
                f"[{MIN_CHUNK_BYTES}, {MAX_CHUNK_BYTES}]"
            )
            # And close to the target — the heuristic is byte-targeted, not
            # element-targeted, so smaller dtypes get more rows per chunk.
            assert chunk_bytes <= TARGET_CHUNK_BYTES, (
                f"{dtype_str}: {chunk_bytes} > target {TARGET_CHUNK_BYTES}"
            )

    def test_default_dtype_is_float32(self) -> None:
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        # Without an explicit dtype, behaviour matches the explicit float32
        # default. This pins the default contract so a future signature
        # change is caught.
        shape = (10_000, 3)
        implicit = calculate_intelligent_chunks(shape)
        explicit = calculate_intelligent_chunks(shape, dtype=np.dtype(np.float32))
        assert implicit == explicit


class TestFinalizeGuards:
    """CL-2: writes after finalize() must raise rather than silently no-op or
    corrupt the consolidated metadata."""

    def test_write_points_after_finalize_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(zarr_path)
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("a", np.random.randn(5, 3).astype(np.float32))
            compiler.finalize()

            with pytest.raises(RuntimeError, match="finalized"):
                compiler.write_points("b", np.random.randn(5, 3).astype(np.float32))

    def test_create_scene_after_finalize_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(zarr_path)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.finalize()

            with pytest.raises(RuntimeError, match="finalized"):
                compiler.create_scene(dimensions=Dimensions.default_3d())

    def test_write_group_after_finalize_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(zarr_path)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.finalize()

            with pytest.raises(RuntimeError, match="finalized"):
                compiler.write_group("/group", attr="value")

    def test_writes_inside_context_still_work(self) -> None:
        """Sanity check: the guard only fires after finalize, not at context entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("a", np.random.randn(5, 3).astype(np.float32))
                # No exception expected here.
                scene.add_points("b", np.random.randn(5, 3).astype(np.float32))
