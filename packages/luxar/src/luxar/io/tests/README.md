# luxar.io.tests

Tests for the Luxar I/O package (compiler, reader, ordering).

## Test Files

- `test_roundtrip.py` - Write/read round-trip tests for all geometry types
- `test_compiler_integration.py` - Integration tests for the LuxarZarrCompiler
- `test_compiler_improvements.py` - Tests for compiler features (scalar convenience, encoding modes)
- `test_compiler_colormap.py` - Tests for colormap encoding in the compiler
- `test_compiler_nd_bounds.py` - Tests for nD bounds computation and nd_transform integration
- `test_reader_nodes.py` - Tests for LuxarScene reader (node listing, metadata, data loading)
- `test_ordering_points.py` - Tests for Morton/Hilbert ordering of points
- `test_ordering_lines.py` - Tests for line segment ordering
- `test_zarr_nd_chunking.py` - Tests for nD-aware chunk size calculation
- `test_optimise.py` - Tests for `luxar.io.optimise` / `luxar optimise`: bit-identical round-trips across flat leaves, additive ladders, `kind=lod` groups, partitions and Lines-with-scalars; attrs (including Lines' nested `vertex_ordering`) and consolidated metadata preserved; the atom-alignment invariant; zarr format 2 and 3 both preserved; degenerate shapes (`n=1`, `(0, D)` refs, `(1,)` broadcasts, no spatial index, ±1 around the atom boundary); never-shrink; and the `content_hash` change that stops a warm viewer cache serving chunks whose keys have moved
- `test_progressive_writing.py` - Tests for progressive (streaming) writing and deterministic cleanup of compiler-owned temporary scenes on successful or failed finalization
- `test_writer_parent_parameter.py` - Tests for parent parameter in writer
- `test_io_metadata.py` - Tests for metadata consolidation and storage
- `test_volume_lazy.py` - Tests for lazy volume access (`open_volume_lazy` array selection, `pin_volume_axes`, splat-dim → volume-axis maps)

## Running

```bash
hatch run pytest packages/luxar/src/luxar/io/tests/
```
