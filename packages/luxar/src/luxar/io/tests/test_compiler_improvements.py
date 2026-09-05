"""Tests for compiler improvements and fixes."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.transforms import prepare_transform_for_zarr, translate
from luxar.typing_utils.constants import MAX_COPY_CHARS, MAX_LINK_CHARS
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

    Every geometry's per-element arrays are sized per-array to their own dtype
    byte budget (``per_array_bytes=True``): each chunk[0] is a multiple of the
    spatial-index ``chunk_size`` atom (or the full array length), which keeps it
    on the viewer's row-range query grid while issuing far fewer requests on
    large scenes. Correctness depends only on the ``chunk_bounds``/``chunk_size``
    partition grid, which these arrays never change.
    """

    # -- GSplats alignment ---------------------------------------------------

    def test_gsplats_all_arrays_aligned(self) -> None:
        """Gsplats arrays land on the chunk_size atom grid (multiples of it)."""
        from math import ceil

        from luxar.core.dimensions import Dimension, Dimensions
        from luxar.gsplats.utils.trils import tril_size

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            # Large enough that EVERY per-splat array spans several atoms. At
            # the previous n=2,500 three of the five arrays came out as one
            # full-array chunk, which the loop below skips as trivially aligned
            # — so the alignment assertion only really covered the Cholesky
            # pair. The `no array is a single chunk` assertion right after the
            # loop keeps it that way if the budgets ever shift.
            n_splats = 20_000
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
                # Positive diagonal required by the writer's Cholesky gate;
                # values are irrelevant to this chunk-alignment test.
                cholesky = (np.abs(np.random.randn(n_splats, k)) + 0.1).astype(
                    np.float32
                )
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

            # Each per-splat array is sized to its own dtype byte budget, so
            # chunk[0] is never below the atom and always lands ON the atom grid
            # (a multiple of it) unless it is one full-array chunk. That is what
            # keeps the viewer's row-range reads whole-chunk aligned.
            # v3.1: Cholesky is stored as a diagonal + off-diagonal split, and
            # each half is sized to its own row width like every other array.
            names = (
                "centers",
                "amplitudes",
                "cholesky_factors_diag",
                "cholesky_factors_offdiag",
                "colors",
            )
            for name in names:
                c0 = g[name].chunks[0]
                n_rows = g[name].shape[0]
                if c0 == n_rows:
                    continue  # single full-array chunk is always aligned
                assert c0 >= chunk_size, (
                    f"{name} chunks[0]={c0} < chunk_size atom={chunk_size}"
                )
                assert c0 % chunk_size == 0, (
                    f"{name} chunks[0]={c0} is neither a multiple of the atom "
                    f"{chunk_size} nor the full array length {n_rows}"
                )

            # Guard against the loop going vacuous: at this size every array
            # must genuinely span several chunks, or the divisibility assertion
            # above is skipped for all of them.
            single_chunk = [n for n in names if g[n].chunks[0] == g[n].shape[0]]
            assert not single_chunk, (
                f"{single_chunk} came out as one full-array chunk, so the "
                f"alignment assertions above never ran — raise n_splats"
            )

            # Each Cholesky half is sized to its OWN row width, not to the
            # packed (N, k) width: at 4D the 4-column diagonal affords a bigger
            # row chunk than the 6-column off-diagonal. A single row count
            # derived from the packed shape would make these equal (and give
            # both halves half their byte budget).
            diag_c0 = g["cholesky_factors_diag"].chunks[0]
            off_c0 = g["cholesky_factors_offdiag"].chunks[0]
            assert diag_c0 > off_c0, (
                f"cholesky diag chunks[0]={diag_c0} should exceed offdiag "
                f"{off_c0} at {ndim}D — both halves look sized from the packed "
                f"({n_splats}, {k}) shape instead of their own row widths"
            )

            # chunk_bounds partitions must match number of zarr chunks
            cb = np.array(g["chunk_bounds"])
            expected_partitions = ceil(n_splats / chunk_size)
            assert cb.shape[0] == expected_partitions, (
                f"chunk_bounds has {cb.shape[0]} partitions but expected "
                f"{expected_partitions} (ceil({n_splats}/{chunk_size}))"
            )

            # Per-array sizing means a zarr chunk may now SPAN several
            # partitions, so the counts are no longer equal. The property that
            # matters for the viewer is that the partition grid SUBDIVIDES the
            # zarr chunk grid: every partition's row range falls inside a single
            # zarr chunk, so a row-range read never straddles a chunk boundary.
            # Divisibility (asserted above) is what guarantees that, and it
            # implies there can never be more zarr chunks than partitions.
            zarr_n_chunks = ceil(g["centers"].shape[0] / g["centers"].chunks[0])
            assert zarr_n_chunks <= cb.shape[0], (
                f"zarr has {zarr_n_chunks} chunks, more than the {cb.shape[0]} "
                f"chunk_bounds partitions — the partition grid must subdivide "
                f"the chunk grid, not the other way round"
            )
            centers_c0 = g["centers"].chunks[0]
            assert (
                centers_c0 == g["centers"].shape[0] or centers_c0 % chunk_size == 0
            ), (
                f"centers chunks[0]={centers_c0} must be an atom multiple so each "
                f"partition lies within one zarr chunk"
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
                # Positive diagonal required by the writer's Cholesky gate;
                # values are irrelevant to this chunk-alignment test.
                cholesky = (np.abs(np.random.randn(n_splats, k)) + 0.1).astype(
                    np.float32
                )

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
        """Points arrays are per-array sized (multiples of the chunk_size atom)."""
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

            # chunk_bounds partition count still uses the atom (unchanged).
            cb = np.array(g["chunk_bounds"])
            expected = ceil(n_points / chunk_size)
            assert cb.shape[0] == expected, (
                f"chunk_bounds has {cb.shape[0]} partitions, expected {expected}"
            )

            # Each per-point array is per-array-sized: chunk[0] is >= the atom
            # and lands on the atom grid (a multiple of it) OR is a single
            # full-array chunk (trivially aligned).
            for name in ("positions", "colors", "radii", "sharpnesses"):
                arr = g[name]
                c0 = arr.chunks[0]
                assert c0 >= chunk_size, (
                    f"{name} chunks[0]={c0} < chunk_size atom={chunk_size}"
                )
                assert c0 % chunk_size == 0 or c0 == arr.shape[0], (
                    f"{name} chunks[0]={c0} is neither a multiple of the atom "
                    f"{chunk_size} nor the full array length {arr.shape[0]}"
                )

            # float32 positions enlarge past the atom (proves the per-array
            # byte budget is active). Relate it to the atom, don't hard-code.
            pos_chunk0 = g["positions"].chunks[0]
            assert pos_chunk0 > chunk_size, (
                f"positions chunks[0]={pos_chunk0} should exceed the atom "
                f"{chunk_size} under per-array byte-budget chunking"
            )
            assert pos_chunk0 % chunk_size == 0, (
                f"positions chunks[0]={pos_chunk0} must stay on the atom grid "
                f"(multiple of {chunk_size})"
            )

    def test_points_4d_with_discrete_dim(self) -> None:
        """4D points arrays stay on the atom grid under per-array sizing."""
        from math import ceil

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

            # Partition count still uses the atom (unchanged).
            cb = np.array(g["chunk_bounds"])
            assert cb.shape[0] == ceil(n_points / chunk_size)

            # Every per-point array lands on the atom grid or is a single
            # full-array chunk, and is never smaller than the atom.
            for name in ("positions", "colors", "radii"):
                arr = g[name]
                c0 = arr.chunks[0]
                assert c0 >= chunk_size, (
                    f"{name} chunks[0]={c0} < chunk_size atom={chunk_size}"
                )
                assert c0 % chunk_size == 0 or c0 == arr.shape[0], (
                    f"{name} chunks[0]={c0} is neither a multiple of the atom "
                    f"{chunk_size} nor the full array length {arr.shape[0]}"
                )

            # Discriminating check (fails under the pre-change atom-sized
            # writer): float32 positions in this 4D scene (atom=2048) enlarge
            # to 2*2048=4096. Relate it to the atom, don't hard-code.
            pos_chunk0 = g["positions"].chunks[0]
            assert pos_chunk0 > chunk_size, (
                f"positions chunks[0]={pos_chunk0} should exceed the atom "
                f"{chunk_size} under per-array byte-budget chunking"
            )
            assert pos_chunk0 % chunk_size == 0

    def test_points_1d_scalars_enlarge_past_atom(self) -> None:
        """1D per-point arrays (radii, colormap scalars) genuinely ENLARGE to a
        multiple of the atom — not just clamp to the full array.

        Covers (a) the opted-in ``write_scalars`` path and (b) real 1D
        enlargement: with n_points=40000 (3D → atom=2340) a 1D float32 array
        sizes to 7*2340 = 16380 rows, which is a proper multiple of the atom
        AND strictly smaller than the array length (so it doesn't slip through
        the single-full-array-chunk escape hatch the other tests allow).
        """
        from luxar.core.dimensions import Dimension, Dimensions

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            n_points = 40000

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
                radii = np.random.rand(n_points).astype(np.float32) + 0.1
                # Per-point scalars (float32 array) so write_scalars runs.
                scalars = np.random.rand(n_points).astype(np.float32)

                scene.add_points(
                    "pts1d",
                    positions=positions,
                    radii=radii,
                    scalars=scalars,
                    colormap="viridis",  # scalars require a colormap to map
                )

            store = zarr.open_group(zarr_path, mode="r")
            g = store["pts1d"]
            chunk_size = g.attrs["chunk_size"]

            # Both 1D arrays enlarge to a proper multiple of the atom that is
            # strictly between one atom and the full array length.
            for name in ("radii", "scalars"):
                arr = g[name]
                c0 = arr.chunks[0]
                assert c0 % chunk_size == 0, (
                    f"{name} chunks[0]={c0} not a multiple of atom {chunk_size}"
                )
                assert c0 > chunk_size, (
                    f"{name} chunks[0]={c0} did not enlarge past atom {chunk_size}"
                )
                assert c0 < arr.shape[0], (
                    f"{name} chunks[0]={c0} is a single full-array chunk "
                    f"(len={arr.shape[0]}), not genuine 1D enlargement"
                )

    # -- Lines alignment -----------------------------------------------------

    def test_lines_all_attributes_aligned(self) -> None:
        """Line attribute arrays land on the vertex chunk_size atom grid."""
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

            # The atom is the vertex-ordering chunk_size: the grid the viewer
            # resolves matched partitions to row ranges against.
            atom = g.attrs["vertex_ordering"]["chunk_size"]
            assert atom > 0, "vertex_ordering chunk_size must be positive"

            # Each vertex-indexed array is sized to its own dtype byte budget,
            # so they no longer share one row count. What must hold is that each
            # lands ON the atom grid (or is a single full-array chunk), which is
            # what keeps a row-range read whole-chunk aligned.
            for name in ("vertices", "widths", "colors", "sharpnesses"):
                c0 = g[name].chunks[0]
                n_rows = g[name].shape[0]
                if c0 == n_rows:
                    continue  # single full-array chunk is always aligned
                assert c0 >= atom, f"{name} chunks[0]={c0} < atom={atom}"
                assert c0 % atom == 0, (
                    f"{name} chunks[0]={c0} is neither a multiple of the atom "
                    f"{atom} nor the full array length {n_rows}"
                )

    def test_lines_scalars_land_on_the_atom_grid(self) -> None:
        """Lines ``scalars`` is atom-aligned like every other per-vertex array.

        Regression guard. Lines' ``ordering_data`` is NESTED
        (``vertex_ordering`` / ``segment_ordering``) where Points' is flat, and
        ``write_lines`` used to hand the OUTER dict to ``write_scalars``. The
        chunk calculator looks for a top-level ``chunk_size``, found none, and
        fell back to a pure byte budget — so ``scalars`` was the one per-vertex
        array off the grid (16,384 rows against a 3,276 atom;
        ``16384 % 3276 == 4``). No test wrote lines WITH scalars and checked
        alignment, so it stayed invisible.

        Needs enough vertices that the byte budget exceeds one atom, or the
        misaligned and aligned answers coincide and this cannot fail.
        """
        from luxar.core.dimensions import Dimension, Dimensions

        n_vertices = 60_000

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"
            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", display=True),
                        Dimension("y", display=True),
                        Dimension("z", display=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                rng = np.random.default_rng(5)
                scene.add_lines(
                    "lines",
                    vertices=rng.standard_normal((n_vertices, 3)).astype(np.float32),
                    widths=rng.random(n_vertices).astype(np.float32) + 0.01,
                    line_type="segments",
                    sharpness=rng.uniform(0.2, 0.9, n_vertices).astype(np.float32),
                    # Non-uniform so the encoder cannot broadcast it to one row.
                    scalars=rng.random(n_vertices).astype(np.float32),
                    colormap="viridis",
                )

            g = zarr.open_group(zarr_path, mode="r")["lines"]
            atom = g.attrs["vertex_ordering"]["chunk_size"]
            assert atom > 0

            c0 = g["scalars"].chunks[0]
            n_rows = g["scalars"].shape[0]
            assert c0 > atom, (
                f"scalars chunks[0]={c0} did not exceed the atom {atom}; pick a "
                f"larger n_vertices or this test cannot detect misalignment"
            )
            assert c0 % atom == 0, (
                f"scalars chunks[0]={c0} is not a multiple of the vertex atom "
                f"{atom} (nor the full length {n_rows}) — write_lines is likely "
                f"passing the outer ordering_data instead of its vertex_ordering"
            )

    def test_lines_segments_use_their_own_byte_budget(self) -> None:
        """``segments`` is sized to its byte budget on the SEGMENT atom grid.

        The segments array has its own ordering grid (``segment_ordering``), and
        was the last per-element array pinned to exactly one atom — 4,096 uint32
        pairs = 32 KB against the 64 KB target, so twice the requests it needs.
        The atom is what the viewer resolves matched segment partitions to row
        ranges against, so the chunk must stay a whole multiple of it.
        """
        from luxar.core.dimensions import Dimension, Dimensions

        n_vertices = 60_000

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"
            with LuxarZarrCompiler(zarr_path, enable_spatial_index=True) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", display=True),
                        Dimension("y", display=True),
                        Dimension("z", display=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)
                rng = np.random.default_rng(7)
                scene.add_lines(
                    "lines",
                    vertices=rng.standard_normal((n_vertices, 3)).astype(np.float32),
                    widths=rng.random(n_vertices).astype(np.float32) + 0.01,
                    line_type="segments",
                )

            g = zarr.open_group(zarr_path, mode="r")["lines"]
            atom = g.attrs["segment_ordering"]["chunk_size"]
            assert atom > 0, "segment_ordering chunk_size must be positive"

            c0 = g["segments"].chunks[0]
            n_rows = g["segments"].shape[0]
            assert c0 > atom, (
                f"segments chunks[0]={c0} did not exceed the segment atom "
                f"{atom} — it is still pinned to one atom (or n_vertices is too "
                f"small for the byte budget to clear one atom)"
            )
            assert c0 % atom == 0, (
                f"segments chunks[0]={c0} is not a multiple of the segment atom "
                f"{atom} (nor the full length {n_rows}), so a matched segment "
                f"partition's row range can straddle a zarr chunk"
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
        result = calculate_intelligent_chunks((10000,), dtype=np.dtype(np.float32))
        assert result == (10000,)

        # With spatial data: uses chunk_size
        result = calculate_intelligent_chunks(
            (10000,), spatial_index_data=spatial, dtype=np.dtype(np.float32)
        )
        assert result == (512,)

        # Small array clamped to actual size
        result = calculate_intelligent_chunks(
            (100,), spatial_index_data=spatial, dtype=np.dtype(np.float32)
        )
        assert result == (100,)

    def testcalculate_intelligent_chunks_2d_uses_spatial_data(self) -> None:
        """calculate_intelligent_chunks must use spatial chunk_size for 2D arrays."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        spatial = {"chunk_size": 1024}

        # Without spatial data: uses byte-based default (64 KiB / float32 / 4 dims)
        result = calculate_intelligent_chunks((5000, 4), dtype=np.dtype(np.float32))
        assert result[1] == 4
        assert result[0] == min(5000, (65536 // 4) // 4)

        # With spatial data: uses chunk_size
        result = calculate_intelligent_chunks(
            (5000, 4), spatial_index_data=spatial, dtype=np.dtype(np.float32)
        )
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

    # -- per_array_bytes opt-in (issue #808 cause #3) -----------------------

    def test_per_array_bytes_default_is_exact_atom_2d(self) -> None:
        """Default (per_array_bytes=False) returns EXACTLY the atom for a 2D
        array — proving the new opt-in leaves gsplats/lines byte-identical."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        atom = 1024
        spatial = {"chunk_size": atom}
        result = calculate_intelligent_chunks(
            (50000, 4), spatial_index_data=spatial, dtype=np.dtype(np.float32)
        )
        assert result == (atom, 4)

    def test_per_array_bytes_enlarges_small_itemsize(self) -> None:
        """per_array_bytes=True with uint8 colors (N, 3) returns a chunk that
        is > atom and a multiple of the atom (its own byte budget is large)."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        atom = 1000
        spatial = {"chunk_size": atom}
        # uint8, 3 cols: target_elements = 65536 // 1 = 65536;
        # ideal rows = 65536 // 3 = 21845; atom-aligned = 21 * 1000 = 21000.
        result = calculate_intelligent_chunks(
            (100000, 3),
            spatial_index_data=spatial,
            dtype=np.dtype(np.uint8),
            per_array_bytes=True,
        )
        assert result == (21000, 3)
        assert result[0] > atom
        assert result[0] % atom == 0

    def test_per_array_bytes_floors_at_one_atom(self) -> None:
        """When the per-array byte budget is smaller than one atom, the result
        floors at exactly the atom (never smaller than the query grid)."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        atom = 20000  # deliberately larger than one byte-budget of rows
        spatial = {"chunk_size": atom}
        # float32, 8 cols: target_elements = 16384; ideal = 16384 // 8 = 2048,
        # far below the atom → floored to a single atom.
        result = calculate_intelligent_chunks(
            (100000, 8),
            spatial_index_data=spatial,
            dtype=np.dtype(np.float32),
            per_array_bytes=True,
        )
        assert result == (atom, 8)

    def test_per_array_bytes_alignment_invariant(self) -> None:
        """The first-axis chunk is always a multiple of the atom OR equals the
        array length, across dtypes."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        atom = 1024
        spatial = {"chunk_size": atom}
        n_points = 50000
        for dt in (np.float32, np.uint8, np.uint16):
            result = calculate_intelligent_chunks(
                (n_points, 3),
                spatial_index_data=spatial,
                dtype=np.dtype(dt),
                per_array_bytes=True,
            )
            c0 = result[0]
            assert c0 >= atom
            assert c0 % atom == 0 or c0 == n_points, (
                f"dtype={dt}: chunk[0]={c0} not atom-aligned nor full length"
            )

    def test_per_array_bytes_ignored_without_spatial_index(self) -> None:
        """With no chunk_size atom, per_array_bytes changes nothing — the plain
        byte-based path is used regardless of the flag."""
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        shape = (100000, 3)
        dtype = np.dtype(np.float32)
        default = calculate_intelligent_chunks(shape, dtype=dtype)
        opted = calculate_intelligent_chunks(shape, dtype=dtype, per_array_bytes=True)
        assert default == opted


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

    def test_node_rollback_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"
            compiler = LuxarZarrCompiler(zarr_path)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_group("parent/child", opacity=0.5)

            assert compiler.node_exists("parent/child")
            assert not compiler.node_exists("missing")
            compiler.delete_node("missing")
            compiler.delete_node("parent")

            assert not compiler.node_exists("parent")
            with pytest.raises(ValueError, match="Cannot delete the scene root"):
                compiler.delete_node("")


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

    def test_dtype_is_required(self) -> None:
        from luxar.io._compiler.chunking import calculate_intelligent_chunks

        # dtype is a required keyword-only argument: the byte-target heuristic
        # cannot pick chunks without knowing the element size, and an implicit
        # float32 default silently under-chunked non-float32 callers.
        shape = (10_000, 3)
        with pytest.raises(TypeError, match="dtype"):
            calculate_intelligent_chunks(shape)  # type: ignore[call-arg]


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

    def test_delete_group_attr_after_finalize_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(zarr_path)
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", np.random.randn(5, 3).astype(np.float32))
            compiler.finalize()

            with pytest.raises(RuntimeError, match="finalized"):
                compiler.delete_group_attr("pts", "transform")

    @pytest.mark.parametrize("method_name", ["node_exists", "delete_node"])
    def test_node_rollback_primitives_after_finalize_raise(
        self, method_name: str
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"
            compiler = LuxarZarrCompiler(zarr_path)
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.finalize()

            with pytest.raises(RuntimeError, match="finalized"):
                getattr(compiler, method_name)("missing")

    def test_writes_inside_context_still_work(self) -> None:
        """Sanity check: the guard only fires after finalize, not at context entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("a", np.random.randn(5, 3).astype(np.float32))
                # No exception expected here.
                scene.add_points("b", np.random.randn(5, 3).astype(np.float32))


class TestPostFinalizeAttrDeletion:
    """Issue #677: clearing ``transform`` / ``nd_transform`` after finalize()
    must warn and leave raw ``.zattrs`` and consolidated ``.zmetadata`` in
    agreement rather than silently desynchronizing them."""

    def test_transform_delete_after_finalize_warns_and_stays_consistent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(zarr_path)
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            pts = scene.add_points("pts", np.random.randn(5, 3).astype(np.float32))
            pts.transform = np.eye(4, dtype=np.float32)
            compiler.finalize()

            with pytest.warns(UserWarning, match="finalized") as record:
                pts.transform = None

            assert len(record) == 1
            # In-memory cache reflects the removal.
            assert pts.transform is None

            # Raw and consolidated views must AGREE: disk untouched, so both
            # still contain "transform".
            consolidated = zarr.open_consolidated(zarr_path, mode="r")
            raw = zarr.open_group(zarr_path, mode="r")
            assert ("transform" in consolidated["pts"].attrs) == (
                "transform" in raw["pts"].attrs
            )
            assert "transform" in raw["pts"].attrs

    def test_nd_transform_delete_after_finalize_warns_and_stays_consistent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            compiler = LuxarZarrCompiler(zarr_path)
            # A (t, x, y, z) scene: the setter validates the key against the
            # scene dimensions (issue #1418), so it must name a real
            # non-displayed dimension.
            scene = compiler.create_scene(dimensions=Dimensions.default_timeseries())
            positions = np.random.randn(5, 4).astype(np.float32)
            positions[:, 0] = np.arange(5, dtype=np.float32)
            pts = scene.add_points("pts", positions)
            pts.nd_transform = {"t": {"scale": 1.0, "offset": 0.0}}
            compiler.finalize()

            with pytest.warns(UserWarning, match="finalized") as record:
                pts.nd_transform = None

            assert len(record) == 1
            assert pts.nd_transform is None

            consolidated = zarr.open_consolidated(zarr_path, mode="r")
            raw = zarr.open_group(zarr_path, mode="r")
            assert ("nd_transform" in consolidated["pts"].attrs) == (
                "nd_transform" in raw["pts"].attrs
            )
            assert "nd_transform" in raw["pts"].attrs

    def test_transform_delete_before_finalize_removes_from_disk_no_warning(
        self,
    ) -> None:
        """Pre-finalize behavior is preserved: clearing removes the attr from
        disk with no warning."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                pts = scene.add_points("pts", np.random.randn(5, 3).astype(np.float32))
                pts.transform = np.eye(4, dtype=np.float32)

                with warnings.catch_warnings(record=True) as record:
                    warnings.simplefilter("always")
                    pts.transform = None
                assert len(record) == 0
                assert pts.transform is None

            store = zarr.open_group(zarr_path, mode="r")
            assert "transform" not in store["pts"].attrs


class TestWriterFuzzRegressions:
    """Regressions for the compiler-fuzz findings (campaign-4 iter 12).

    Every test here failed before the fail-fast pre-write gate landed:
    F1 empty name clobbered the scene root; F2 optional-array lengths were
    validated only AFTER the spatial reorder (silent truncation / raw
    IndexError); F3 negative line indices wrapped to uint32; F4 non-str
    labels AttributeError'd after the node was written; F5 zarr-reserved
    names died deep in zarr storage; F6 cheap-attr validation ran after the
    node's arrays were on disk.
    """

    @staticmethod
    def _scene(tmpdir: str):
        zarr_path = Path(tmpdir) / "test.luxar.zarr"
        compiler = LuxarZarrCompiler(zarr_path)
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        return zarr_path, compiler, scene

    POS = np.random.RandomState(0).rand(50, 3).astype(np.float32) * 10

    # ---- review follow-ups: numpy scalars + gsplat reserved attrs ------

    def test_numpy_scalar_broadcast_components_accepted(self) -> None:
        """np.float32 does NOT subclass Python float — tuple components
        unpacked from a float32 array (a legitimate caller pattern) must
        pass the pre-write gate like plain floats do."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, scene = self._scene(tmpdir)
            scene.add_points(
                "np_scalars",
                self.POS,
                colors=(np.float32(1.0), np.float32(0.5), np.float32(0.2)),
            )
            # numpy-scalar radii/sharpness stay CLEANLY REJECTED (the
            # downstream broadcast writers only handle Python floats —
            # on main they crashed with a deep IndexError; the pre-write
            # gate converts that to a typed error).
            with pytest.raises((ValueError, TypeError)):
                scene.add_points("np_rad", self.POS, radii=np.float32(0.5))

    def test_numpy_scalar_colormap_scalars_hint_is_actionable(self) -> None:
        """The scalars preflight rejects numpy scalars with the same
        one-step float(...) hint as radii/widths/sharpness (#752) — not
        the dead-end np.array(scalars) suggestion that fails again on 0D."""
        from luxar.validation.writing import (
            validate_scalars_preflight,
        )

        with pytest.raises(ValueError) as exc_info:
            validate_scalars_preflight(np.float32(0.5), 50)
        msg = str(exc_info.value)
        assert "float(" in msg
        assert "np.array(scalars)" not in msg

    def test_gsplat_position_bounds_is_reserved(self) -> None:
        """The gsplat writer unconditionally stamps position_bounds; a
        user-supplied value must be rejected up front, not silently
        stamped over (the same rule points/lines already enforce)."""
        from luxar.validation.writing import (
            GSPLATS_RESERVED_ATTRS,
            POINTS_RESERVED_ATTRS,
        )

        assert "position_bounds" in GSPLATS_RESERVED_ATTRS
        assert {"has_label_ids", "label_vocabulary"} <= GSPLATS_RESERVED_ATTRS
        assert not {"has_label_ids", "label_vocabulary"} & POINTS_RESERVED_ATTRS

    # ---- F1: empty node name must not clobber the scene root -----------

    def test_empty_node_name_rejected_and_root_intact(self) -> None:
        from luxar.io.reader import LuxarScene

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="empty"):
                scene.add_points("", self.POS)
            with pytest.raises(ValueError, match="empty"):
                scene.add_lines("", self.POS, 0.5)
            with pytest.raises(ValueError, match="empty"):
                scene.add_group("")
            compiler.finalize()

            root = zarr.open_group(str(zarr_path), mode="r")
            assert root.attrs["type"] == "scene"  # NOT clobbered to 'points'
            LuxarScene.load(zarr_path)  # store still loadable

    def test_empty_path_rejected_at_writer_level(self) -> None:
        """The raw compiler API is covered too (require_group('') == root)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, _scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="empty|ROOT"):
                compiler.write_points("", self.POS)
            with pytest.raises(ValueError, match="empty|ROOT"):
                compiler.write_points("/", self.POS)
            compiler.finalize()

    # ---- F5: zarr-reserved (dot-prefixed) names -------------------------

    def test_zarr_reserved_names_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            for bad in (".zgroup", ".zattrs", ".zmetadata", ".zarray"):
                with pytest.raises(ValueError, match="cannot start with"):
                    scene.add_points(bad, self.POS)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert list(root.group_keys()) == []

    # ---- F2: optional-array lengths validated BEFORE spatial reorder ----

    def test_too_long_optional_array_rejected_not_truncated(self) -> None:
        """A radii array of n+5 used to be silently TRUNCATED by the
        spatial-ordering fancy-indexing and accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, compiler, scene = self._scene(tmpdir)
            radii = np.full(55, 2.0, np.float32)  # 55 != 50
            with pytest.raises(ValueError, match="radii"):
                scene.add_points("pts", self.POS, radii=radii)
            compiler.finalize()

    def test_too_short_optional_array_rejected_cleanly(self) -> None:
        """A sharpness array of n-1 used to raise a raw IndexError inside
        build_points_ordering (before any validator ran)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, compiler, scene = self._scene(tmpdir)
            sharpness = np.full(49, 0.5, np.float32)
            with pytest.raises(ValueError, match="sharpness"):
                scene.add_points("pts", self.POS, sharpness=sharpness)
            compiler.finalize()

    def test_lines_wrong_length_colors_rejected_cleanly(self) -> None:
        """Lines colors of the wrong length used to IndexError during the
        vertex reorder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, compiler, scene = self._scene(tmpdir)
            colors = np.random.RandomState(1).rand(51, 3).astype(np.float32)
            with pytest.raises(ValueError, match="colors"):
                scene.add_lines("lns", self.POS, 0.5, colors=colors)
            compiler.finalize()

    # ---- F3: negative line indices must not wrap to uint32 --------------

    def test_negative_line_indices_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, compiler, scene = self._scene(tmpdir)
            idx = np.array([[-1, 0], [1, 2]], dtype=np.int64)
            with pytest.raises(ValueError, match="< 0"):
                scene.add_lines("lns", self.POS, 0.5, indices=idx, line_type="indexed")
            compiler.finalize()

    def test_list_indices_accepted(self) -> None:
        """A Python-list `indices` (a legitimate adder input) used to
        AttributeError on `.size`; it must be arrayed and accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_lines(
                "lns", self.POS, 0.5, indices=[0, 1, 1, 2], line_type="indexed"
            )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["lns"].attrs["n_segments"] == 2

    def test_wrong_width_indices_rejected(self) -> None:
        """An even-size but wrong-width (E, 3) index array used to pass the
        element-count checks and reshape into bogus edges; it must be
        rejected up front."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, compiler, scene = self._scene(tmpdir)
            bad = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)  # (E, 3)
            with pytest.raises(ValueError, match="shape"):
                scene.add_lines("lns", self.POS, 0.5, indices=bad, line_type="indexed")
            compiler.finalize()

    def test_float_indices_rejected(self) -> None:
        """A float index array used to pass the layout/parity/bounds checks
        and then get silently truncated by convert_to_indexed's
        `.astype(np.uint32)` (1.7 -> 1), producing unauthored edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, compiler, scene = self._scene(tmpdir)
            bad = np.array([0.5, 1.7, 1.0, 2.0])  # float dtype
            with pytest.raises(ValueError, match="integer"):
                scene.add_lines("lns", self.POS, 0.5, indices=bad, line_type="indexed")
            compiler.finalize()

    # ---- F4: non-str labels fail fast, BEFORE any zarr write ------------

    def test_non_str_labels_rejected_without_partial_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="expected str or None"):
                scene.add_points("pts", self.POS, labels=[42] * 50)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root  # nothing leaked

    # ---- F6 (cheap half): validate BEFORE any array lands on disk -------

    def test_invalid_gamma_leaves_no_node_behind(self) -> None:
        """gamma=-1 used to be rejected only AFTER the node was fully
        written (node persisted with the invalid attr)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Gamma"):
                scene.add_points("pts", self.POS, gamma=-1.0)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_reserved_attr_collision_rejected_pre_write(self) -> None:
        """type=/n_points= junk attrs used to raise an accidental TypeError
        in the Node constructor AFTER the node was written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="reserved"):
                scene.add_points("pts", self.POS, type="banana", n_points=-1)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_duplicate_name_rejected_before_overwriting_first_node(self) -> None:
        """A duplicate add used to overwrite the first node's arrays on disk
        before the (post-write) duplicate check raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_points("pts", self.POS, radii=np.full(50, 1.5, np.float32))
            other = np.random.RandomState(7).rand(20, 3).astype(np.float32)
            with pytest.raises(ValueError, match="Duplicate"):
                scene.add_points("pts", other)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            # First node intact: still 50 points, radii untouched.
            assert root["pts"].attrs["n_points"] == 50

    def test_invalid_transform_leaves_no_node_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError):
                scene.add_points("pts", self.POS, transform="banana")
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_scalar_sharpness_out_of_range_rejected_by_writer(self) -> None:
        """Scalar sharpness > 1.0 was accepted while the equivalent array
        was rejected (scalar/array asymmetry)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="[Ss]harpness"):
                scene.add_points("pts", self.POS, sharpness=5.0)
            with pytest.raises(ValueError, match="[Ss]harpness"):
                scene.add_lines("lns", self.POS, 0.5, sharpness=5.0)
            compiler.finalize()


class TestUnknownRenderAttrRejected:
    """Issue #787: a misspelled render attr (e.g. ``blending="max"`` instead of
    ``blending_mode="max"``) used to be written into the zarr and silently
    ignored by the viewer. It must now fail fast, BEFORE any zarr is written,
    with a "Did you mean ...?" hint. Legitimate render attrs still write and a
    made-up key is rejected too."""

    @staticmethod
    def _scene(tmpdir: str):
        zarr_path = Path(tmpdir) / "test.luxar.zarr"
        compiler = LuxarZarrCompiler(zarr_path)
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        return zarr_path, compiler, scene

    POS = np.random.RandomState(0).rand(50, 3).astype(np.float32) * 10

    def test_real_render_attrs_still_write_and_apply(self) -> None:
        """blending_mode / opacity / layer / visible are accepted and persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_points(
                "pts",
                self.POS,
                blending_mode="max",
                opacity=0.5,
                layer=True,
                visible=True,
            )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["pts"].attrs["blending_mode"] == "max"
            assert root["pts"].attrs["opacity"] == 0.5

    def test_lod_quality_attrs_are_allowed(self) -> None:
        """Internal Points/Lines LOD quality stamps pass the strict attr gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_points(
                "pts",
                self.POS,
                additive_lod=dict(n_lods=3, method="random", seed=0),
            )
            line_pos = np.random.RandomState(1).rand(100, 3).astype(np.float32)
            scene.add_lines(
                "lns",
                line_pos,
                0.5,
                line_type="segments",
                additive_lod=dict(n_lods=3, method="random", seed=0),
            )
            compiler.finalize()

            root = zarr.open_group(str(zarr_path), mode="r")
            for name in ("pts", "lns"):
                group = root[name]
                assert "level_stats" in group.attrs
                assert "lod_stats" in group["additive_0"].attrs

    def test_near_miss_typo_rejected_with_hint_before_write(self) -> None:
        """``blending=`` (typo of ``blending_mode=``) fails fast with a hint and
        leaves no node on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Did you mean 'blending_mode'"):
                scene.add_points("pts", self.POS, blending="max")
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root  # nothing leaked

    def test_totally_made_up_attr_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Unknown node attribute") as excinfo:
                scene.add_points("pts", self.POS, totally_made_up_attr=42)
            # No close match => no bogus "Did you mean ...?" suggestion.
            assert "Did you mean" not in str(excinfo.value)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_retired_grid_shape_keyword_rejected(self) -> None:
        """``grid_shape=`` used to be a real ``add_points`` parameter whose value
        nothing ever read — it was persisted as a dead attr and ignored. Now that
        the parameter is gone, it arrives through ``**attrs`` and must fail fast
        (on the flat path and on the additive-ladder path alike) instead of
        quietly landing back on disk.
        """
        for kwargs in ({}, {"additive_lod": True}):
            with tempfile.TemporaryDirectory() as tmpdir:
                zarr_path, compiler, scene = self._scene(tmpdir)
                with pytest.raises(
                    ValueError, match="Unknown node attribute 'grid_shape'"
                ):
                    scene.add_points("pts", self.POS, grid_shape=(8, 8, 8), **kwargs)
                compiler.finalize()
                root = zarr.open_group(str(zarr_path), mode="r")
                assert "pts" not in root

    def test_unknown_attr_rejected_on_lines_and_gsplats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Unknown node attribute"):
                scene.add_lines("lns", self.POS, 0.5, blending="max")
            n_splats = 20
            centers = np.random.randn(n_splats, 3).astype(np.float32)
            amplitudes = np.random.rand(n_splats).astype(np.float32)
            cholesky = np.random.randn(n_splats, 6).astype(np.float32)
            with pytest.raises(ValueError, match="Did you mean 'colormap'"):
                scene.add_gsplats(
                    "splats",
                    centers=centers,
                    amplitudes=amplitudes,
                    cholesky_factors=cholesky,
                    colormapp="gray",  # typo of colormap
                )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            # Neither the lines nor the gsplats node leaked to disk (each
            # writer fails BEFORE creating its group — gsplats especially,
            # which is a different writer path from points/lines).
            assert "lns" not in root
            assert "splats" not in root

    def test_line_join_style_round_trips_and_rejects_a_typo(self) -> None:
        """``join=`` (issue #790) persists verbatim, stays absent when
        unauthored, and rejects an unrecognised VALUE.

        The key allowlist and the value check are separate gates and both
        matter: ``jion=`` is caught by the former, ``join="mitre"`` only by the
        latter. Without the value check a typo would write cleanly and render
        with the default join, giving the author nothing to go on.
        """
        for style in ("none", "miter"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zarr_path, compiler, scene = self._scene(tmpdir)
                scene.add_lines("lns", self.POS, 0.5, join=style)
                compiler.finalize()
                root = zarr.open_group(str(zarr_path), mode="r")
                assert root["lns"].attrs["join"] == style

        # Unauthored: the writer must NOT bake a default into the file — the
        # default belongs to the viewer, where a ?lineJoin= override can still
        # win over it.
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_lines("lns", self.POS, 0.5)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "join" not in root["lns"].attrs

        # A misspelled STYLE fails fast and leaves nothing on disk.
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Unknown line join style"):
                scene.add_lines("lns", self.POS, 0.5, join="mitre")
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "lns" not in root

    def test_line_join_setter_persists_like_its_compositing_siblings(self) -> None:
        """``node.join = "none"`` after construction must reach disk.

        Every other member of COMPOSITING_ATTRS has a validating property +
        ``_persist_attr`` setter. Without one, the assignment landed in the
        instance ``__dict__``, the node reported the new style, and the file kept
        the old one — the silent half of a divergence that only shows up when
        somebody opens the scene.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            lns = scene.add_lines("lns", self.POS, 0.5)
            # Unauthored reads as None, NOT as the default: the writer must not
            # bake today's default into the file.
            assert lns.join is None
            lns.join = "none"
            assert lns.join == "none"
            # Chainable sibling of set_blending_mode.
            assert lns.set_join("miter") is lns

            with pytest.raises(ValueError, match="Unknown line join style"):
                lns.join = "mitre"
            with pytest.raises(TypeError, match="Line join style must be a string"):
                lns.join = 1
            # The failed assignments left the last good value in place.
            assert lns.join == "miter"

            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["lns"].attrs["join"] == "miter"

    def test_line_join_rides_the_partition_wrapper_not_the_parts(self) -> None:
        """``join`` is a COMPOSITING attr, so a partitioned lines node writes it
        ONCE on the ``kind=partition`` wrapper and the parts inherit it.

        This is the routing the viewer's attrs composer exists to follow. A
        regression that copied it onto each part instead would still render
        correctly, so only the wrapper's own ``.zattrs`` can pin the contract.
        """
        # Polylines are atomic — a single one never splits — so the fixture needs
        # genuinely separable geometry: 8 disjoint segments, well spread out.
        starts = np.arange(8, dtype=np.float32)[:, None] * 100.0
        verts = np.repeat(starts, 2, axis=0) * np.ones((1, 3), dtype=np.float32)
        verts[1::2, 0] += 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_lines(
                "tracks",
                verts,
                0.5,
                line_type="segments",
                join="none",
                partition={"max_elements": 2},
            )
            compiler.finalize()

            root = zarr.open_group(str(zarr_path), mode="r")
            wrapper = root["tracks"]
            assert wrapper.attrs["kind"] == "partition"
            assert wrapper.attrs["join"] == "none"

            parts = [key for key in wrapper.group_keys() if key.startswith("part_")]
            # Verified, not assumed: with one part the wrapper/leaf distinction
            # this test is about would be untestable.
            assert len(parts) > 1, f"partition produced {len(parts)} part(s)"
            for part in parts:
                assert "join" not in wrapper[part].attrs

    @pytest.mark.parametrize("geometry_type", ["points", "gsplats", "mesh"])
    def test_line_join_refused_on_a_non_lines_leaf(self, geometry_type: str) -> None:
        """``join`` on a points / gsplats / mesh LEAF is dead metadata, so it raises.

        ``KNOWN_RENDER_ATTRS`` is one set shared by all four geometry writers, so
        before this guard ``add_points(..., join="none")`` wrote a ``join`` into a
        points ``.zattrs`` that nothing will ever read — contradicting the format
        spec's "Optional, LINES ONLY". Refused per type at the adder, mirroring the
        mesh ``volumetric`` refusal.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="lines-only attribute"):
                if geometry_type == "points":
                    scene.add_points("node", self.POS, join="none")
                elif geometry_type == "gsplats":
                    scene.add_gsplats(
                        "node",
                        centers=self.POS,
                        amplitudes=np.ones(len(self.POS), dtype=np.float32),
                        cholesky_factors=np.ones((len(self.POS), 6), dtype=np.float32),
                        join="none",
                    )
                else:
                    scene.add_mesh(
                        "node",
                        np.array(
                            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                            dtype=np.float32,
                        ),
                        np.array([[0, 1, 2]], dtype=np.uint32),
                        join="none",
                    )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "node" not in root

    def test_line_join_still_allowed_on_a_group(self) -> None:
        """The other half: a ``join`` on a GROUP is correct and must keep working.

        It is a compositing attr precisely so it can be authored once above a
        lines node — including on a wrapper whose own children are the parts of a
        partitioned lines leaf. A refusal that keyed on "not a lines node" rather
        than "a non-lines LEAF" would break that.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            grp = scene.add_group("styled", join="none")
            grp.add_lines("lns", self.POS, 0.5)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["styled"].attrs["join"] == "none"
            assert "join" not in root["styled"]["lns"].attrs

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("ambient", 0.3),
            ("shade_exponent", 1.5),
            ("specular", 0.5),
            ("shininess", 24.0),
            ("alpha_cutoff", 0.2),
            ("texture_filter", "nearest"),
            ("texture_wrap", "clamp"),
        ],
    )
    @pytest.mark.parametrize("node_type", ["points", "lines", "gsplats", "group"])
    def test_mesh_appearance_refused_on_a_non_mesh_node(
        self, node_type: str, attr: str, value: float
    ) -> None:
        """Mesh appearance attrs must not persist as dead non-mesh metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="mesh-only attribute") as exc_info:
                attrs = {attr: value}
                if node_type == "points":
                    scene.add_points("node", self.POS, **attrs)
                elif node_type == "lines":
                    scene.add_lines("node", self.POS, 0.5, **attrs)
                elif node_type == "gsplats":
                    scene.add_gsplats(
                        "node",
                        centers=self.POS,
                        amplitudes=np.ones(len(self.POS), dtype=np.float32),
                        cholesky_factors=np.ones((len(self.POS), 6), dtype=np.float32),
                        **attrs,
                    )
                else:
                    scene.add_group("node", **attrs)
            error = str(exc_info.value)
            assert "set them on each mesh leaf (part_<i> / child_<i>)" in error
            assert "pass them to add_mesh(...)" in error
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "node" not in root

    def test_mesh_appearance_guard_covers_every_validated_key(self) -> None:
        """Every mesh appearance validator must have a non-mesh refusal."""
        from luxar.core.group.compositing import MESH_ONLY_APPEARANCE_ATTRS
        from luxar.validation.writing import _MESH_APPEARANCE_VALIDATORS

        assert MESH_ONLY_APPEARANCE_ATTRS == _MESH_APPEARANCE_VALIDATORS.keys()

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("ambient", 0.3),
            ("shade_exponent", 1.5),
            ("specular", 0.5),
            ("shininess", 24.0),
            ("alpha_cutoff", 0.2),
        ],
    )
    @pytest.mark.parametrize("node_type", ["points", "lines", "gsplats", "group"])
    def test_mesh_appearance_refused_via_write_through_attrs(
        self, node_type: str, attr: str, value: float
    ) -> None:
        """The adder's refusal is worthless if the write-through mapping re-opens
        the same door (#1782). ``node.attrs["specular"] = 0.5`` on a non-mesh node
        persists straight to the store via the #1764 mapping, so it must be
        refused there too — never left as dead metadata the viewer ignores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            if node_type == "points":
                node = scene.add_points("node", self.POS)
            elif node_type == "lines":
                node = scene.add_lines("node", self.POS, 0.5)
            elif node_type == "gsplats":
                node = scene.add_gsplats(
                    "node",
                    centers=self.POS,
                    amplitudes=np.ones(len(self.POS), dtype=np.float32),
                    cholesky_factors=np.ones((len(self.POS), 6), dtype=np.float32),
                )
            else:
                node = scene.add_group("node")
            with pytest.raises(ValueError, match="mesh-only attribute") as exc_info:
                node.attrs[attr] = value
            error = str(exc_info.value)
            assert "set them on each mesh leaf (part_<i> / child_<i>)" in error
            assert "pass them to add_mesh(...)" in error
            assert attr not in node._attrs_cache
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert attr not in root["node"].attrs

    def test_mesh_appearance_allowed_via_write_through_on_a_mesh_leaf(self) -> None:
        """The write-through guard must not block a genuine mesh leaf — its
        construction fills the attr cache directly (never through the setter), and
        a post-hoc set is how an author tweaks a mesh after ``add_mesh`` returns."""
        verts = np.random.RandomState(1).rand(12, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint32)
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            mesh = scene.add_mesh("m", verts, faces)
            mesh.attrs["specular"] = 0.4
            assert mesh._attrs_cache["specular"] == 0.4
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["m"].attrs["specular"] == 0.4

    @pytest.mark.parametrize("geometry_type", ["points", "gsplats", "mesh"])
    @pytest.mark.parametrize("via", ["property", "set_join"])
    def test_line_join_setter_refused_on_a_non_lines_leaf(
        self, geometry_type: str, via: str
    ) -> None:
        """The adder's refusal is worthless if the SETTER re-opens the same door.

        ``join`` is declared on :class:`Node`, so ``pts.join = "none"`` — and
        ``pts.set_join("none")``, which assigns through that same property — would
        persist the dead attr one line after ``add_points(join=...)`` refused it.
        Each non-lines geometry class overrides the setter to refuse, exactly as
        ``Mesh`` overrides ``blending_mode`` to refuse ``volumetric``. Both
        surfaces raise with the same explanation, so the check below is the one
        the adder test uses.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            if geometry_type == "points":
                node = scene.add_points("node", self.POS)
            elif geometry_type == "gsplats":
                node = scene.add_gsplats(
                    "node",
                    centers=self.POS,
                    amplitudes=np.ones(len(self.POS), dtype=np.float32),
                    cholesky_factors=np.ones((len(self.POS), 6), dtype=np.float32),
                )
            else:
                node = scene.add_mesh(
                    "node",
                    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                    np.array([[0, 1, 2]], dtype=np.uint32),
                )

            with pytest.raises(ValueError, match="lines-only attribute"):
                if via == "property":
                    node.join = "none"
                else:
                    node.set_join("none")

            # The getter still reads as unset (it does NOT substitute a default),
            # and nothing reached disk.
            assert node.join is None
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "join" not in root["node"].attrs

    def test_line_join_setter_still_allowed_on_a_group(self) -> None:
        """A GROUP's ``join`` setter keeps working, like its ``add_group`` sibling.

        ``Group`` derives from ``Node`` directly and gets no override, so the
        compositing attr can still be authored after construction on a wrapper
        above a lines node. (The Lines-leaf half of this is already pinned by
        ``test_line_join_setter_persists_like_its_compositing_siblings``.)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            grp = scene.add_group("styled")
            grp.add_lines("lns", self.POS, 0.5)
            grp.join = "miter"
            assert grp.join == "miter"
            # Chainable form routes through the same property.
            assert grp.set_join("none") is grp
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["styled"].attrs["join"] == "none"

    def test_typo_of_structural_key_rejected_without_structural_suggestion(
        self,
    ) -> None:
        """A typo near an internal structural key (e.g. ``typ=`` / ``kinds=``) is
        still rejected, and the hint never advertises a structural key
        (``type`` / ``kind`` / ``child_index``) — only render attrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, _compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Unknown node attribute") as excinfo:
                scene.add_group("grp", kinds="lod")
            msg = str(excinfo.value)
            assert "Did you mean 'type'" not in msg
            assert "Did you mean 'kind'" not in msg
            assert "Did you mean 'child_index'" not in msg

    def test_add_group_rejects_unknown_attr(self) -> None:
        """add_group routes through the same guard (write_group)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _zarr_path, _compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Did you mean 'blending_mode'"):
                scene.add_group("grp", blending="max")

    def test_add_group_accepts_real_render_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_group("grp", blending_mode="additive", opacity=0.8)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["grp"].attrs["blending_mode"] == "additive"

    def test_scene_root_and_overlay_writes_are_exempt(self) -> None:
        """The unknown-key guard is scoped to real geometry/group nodes: the
        scene root (scene_dimensions / viewer_config) and the ``overlays/``
        namespace carry their own internal attr schemas and must still round-
        trip. Proves the exemption didn't break scene creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"
            with LuxarZarrCompiler(zarr_path) as compiler:
                # create_scene writes scene_dimensions (+ viewer_config) to "/".
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("pts", self.POS)
                # An overlay writes a whole non-render attr schema to
                # overlays/<name> via the same write_group entry point.
                scene.add_text("hello", position=(0.5, 0.5))
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "scene_dimensions" in root.attrs
            assert "pts" in root
            assert "overlays" in root


class TestInteractionTemplateAttrs:
    """``link`` / ``copy`` / ``link_target`` node attrs (issue #1917).

    These are plain node attrs — they ride ``**attrs`` and need no adder
    signature — so the ONLY thing standing between an authoring mistake and a
    click that silently does nothing is the shared attr gate. Every rejection
    below therefore also asserts that no node leaked onto disk.
    """

    @staticmethod
    def _scene(tmpdir: str):
        zarr_path = Path(tmpdir) / "test.luxar.zarr"
        compiler = LuxarZarrCompiler(zarr_path)
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        return zarr_path, compiler, scene

    POS = np.random.RandomState(0).rand(50, 3).astype(np.float32) * 10

    def test_templates_round_trip_onto_the_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_points(
                "pts",
                self.POS,
                labels=[f"L{i}" for i in range(len(self.POS))],
                link="https://example.org/x/{hover_label}",
                copy="{hover_label}",
                link_target="_self",
            )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["pts"].attrs["link"] == "https://example.org/x/{hover_label}"
            assert root["pts"].attrs["copy"] == "{hover_label}"
            assert root["pts"].attrs["link_target"] == "_self"

    def test_link_without_copy_or_target_is_fine(self) -> None:
        """Each template is independent; only `link_target` needs a companion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_points("pts", self.POS, link="https://example.org/{hover_index}")
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "link" in root["pts"].attrs
            assert "link_target" not in root["pts"].attrs

    @pytest.mark.parametrize(
        "bad_link",
        [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "blob:https://example.org/abc",
            "file:///etc/passwd",
            "vbscript:msgbox(1)",
        ],
    )
    def test_non_http_schemes_rejected_before_write(self, bad_link: str) -> None:
        """`.zattrs` is untrusted and the viewer NAVIGATES to this value, so the
        check is an allowlist. Each of these would otherwise be a live XSS or
        local-file vector wired to an ordinary-looking left-click."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="scheme"):
                scene.add_points("pts", self.POS, link=bad_link)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_relative_link_rejected_before_write(self) -> None:
        """A relative template resolves against whatever origin the VIEWER is
        served from, so a third-party store could aim a click at the embedder's
        own site."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="absolute"):
                scene.add_points("pts", self.POS, link="/admin/{hover_label}")
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    @pytest.mark.parametrize(
        "bad_link",
        [
            "https://good.example@evil.example/",
            "https://user:pass@evil.example/",
            "https://:pass@evil.example/",
        ],
    )
    def test_embedded_credentials_rejected(self, bad_link: str) -> None:
        """`https://good.example@evil.example/` navigates to evil.example while
        READING as good.example — including in the viewer's own
        'Copy link address'. Nothing legitimate needs userinfo in a scene link."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="credentials"):
                scene.add_points("pts", self.POS, link=bad_link)
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_hostless_link_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="no host"):
                scene.add_points("pts", self.POS, link="https:///nowhere")
            compiler.finalize()

    def test_placeholders_do_not_break_url_parsing(self) -> None:
        """The template is validated with `{...}` runs still in place — the
        point of validating at write time. A parser that choked on them would
        make the whole gate unusable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_points(
                "pts",
                self.POS,
                link="https://ex.org/{hover_node}/{hover_index}?q={hover_label}#f",
            )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "link" in root["pts"].attrs

    @pytest.mark.parametrize(
        "bad_target", ["_parent", "_top", "_blank ", "myframe", ""]
    )
    def test_bad_link_target_rejected(self, bad_target: str) -> None:
        """Only the two keywords that imply `noopener`. A near-miss like
        `"_blank "` is a NAMED target, which the browser opens with a live
        `window.opener` (reverse tabnabbing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="link_target"):
                scene.add_points(
                    "pts", self.POS, link="https://example.org/", link_target=bad_target
                )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_link_target_without_link_rejected(self) -> None:
        """Inert on its own, and far more often the typo `link_taget=` than a
        deliberate choice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="without link"):
                scene.add_points("pts", self.POS, link_target="_blank")
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "pts" not in root

    def test_non_string_templates_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, compiler, scene = self._scene(tmpdir)
            # The adder rewraps the validator's TypeError as a ValueError
            # ("Could not add points 'pts': ..."), so match the message rather
            # than the class.
            with pytest.raises((TypeError, ValueError), match="link must be a string"):
                scene.add_points("pts", self.POS, link=42)
            with pytest.raises((TypeError, ValueError), match="copy must be a string"):
                scene.add_points("pts2", self.POS, copy=["a"])
            compiler.finalize()

    def test_oversized_templates_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="exceeding"):
                scene.add_points(
                    "pts", self.POS, link="https://e.org/" + "x" * MAX_LINK_CHARS
                )
            with pytest.raises(ValueError, match="exceeding"):
                scene.add_points("pts2", self.POS, copy="y" * (MAX_COPY_CHARS + 1))
            compiler.finalize()

    def test_typo_of_link_gets_a_hint(self) -> None:
        """The attrs are in KNOWN_RENDER_ATTRS rather than the silent
        `_ALLOWED_NODE_ATTRS`, so a near-miss is advertised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, compiler, scene = self._scene(tmpdir)
            with pytest.raises(ValueError, match="Did you mean 'link'"):
                scene.add_points("pts", self.POS, lnk="https://example.org/")
            compiler.finalize()

    def test_templates_accepted_on_mesh(self) -> None:
        """Mesh runs the same gate but through MESH_RESERVED_ATTRS and its own
        mesh-only appearance guard, so it needs its own coverage — the other
        three passing says nothing about it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            verts = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
            )
            faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
            scene.add_mesh(
                "surf",
                vertices=verts,
                faces=faces,
                labels=["a", "b", "c", "d"],
                link="https://example.org/{hover_label}",
                copy="{hover_label}",
                link_target="_self",
            )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["surf"].attrs["link"] == "https://example.org/{hover_label}"
            assert root["surf"].attrs["copy"] == "{hover_label}"
            assert root["surf"].attrs["link_target"] == "_self"

    def test_templates_rejected_on_mesh_too(self) -> None:
        """The gate must be as strict on mesh as everywhere else."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            verts = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
            )
            faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
            with pytest.raises(ValueError, match="scheme"):
                scene.add_mesh(
                    "surf", vertices=verts, faces=faces, link="javascript:alert(1)"
                )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "surf" not in root

    def test_templates_on_a_partitioned_layer_reach_every_leaf(self) -> None:
        """`link` is not a COMPOSITING attr, so the adders route it to each
        `part_<i>` leaf rather than to the wrapper. That is what the viewer
        relies on: a pick hits a leaf. If this ever flips to wrapper-only, the
        viewer's ancestor walk still saves it — but the placement is worth
        pinning, because only one of the two is O(1) at click time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            n = 200
            rng = np.random.RandomState(1)
            pos = (rng.rand(n, 3) * 10).astype(np.float32)
            scene.add_points(
                "tiled",
                pos,
                labels=[f"p{i}" for i in range(n)],
                link="https://example.org/{hover_label}",
                partition={"max_elements": 50},
            )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            wrapper = root["tiled"]
            parts = [k for k in wrapper.keys() if k.startswith("part_")]
            assert len(parts) > 1, f"expected a real partition, got {parts}"
            for part in parts:
                assert (
                    wrapper[part].attrs["link"] == "https://example.org/{hover_label}"
                ), f"leaf {part} did not receive the link template"

    def test_templates_on_a_group_node(self) -> None:
        """Groups go through `write_group`, a different gate call with no
        reserved set. An author labelling a whole group is plausible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            scene.add_group("grp", link="https://example.org/g")
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["grp"].attrs["link"] == "https://example.org/g"

    def test_templates_accepted_on_every_geometry_type(self) -> None:
        """All four leaf adders run the same gate, so all four must accept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path, compiler, scene = self._scene(tmpdir)
            link = "https://example.org/{hover_label}"
            scene.add_points("pts", self.POS, link=link)
            scene.add_lines(
                "lns",
                np.array(
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32
                ),
                widths=np.full(4, 0.1, dtype=np.float32),
                link=link,
            )
            scene.add_gsplats(
                "gs",
                centers=self.POS[:5],
                amplitudes=np.ones(5, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1)
                ),
                link=link,
            )
            scene.add_mesh(
                "mesh",
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.uint32),
                link=link,
            )
            compiler.finalize()
            root = zarr.open_group(str(zarr_path), mode="r")
            for name in ("pts", "lns", "gs", "mesh"):
                assert root[name].attrs["link"] == link
