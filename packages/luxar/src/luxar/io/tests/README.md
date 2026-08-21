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
- `test_optimise.py` - Tests for `luxar.io.optimise` / `luxar optimise`: bit-identical round-trips across flat leaves, additive ladders, `kind=lod` groups, partitions, Lines-with-scalars and standalone `.gsplats.zarr` trees; attrs (including Lines' nested `vertex_ordering`), filters, `fill_value`, memory order, `dimension_names`, the chunk key layout and consolidated metadata preserved; the atom-alignment invariant and the vestigial-`chunk_size` rejection; zarr format 2 and 3 both preserved; a sharded array keeping its shard grid (and being counted as shards, not inner chunks); degenerate shapes (`n=1`, 0-d, `(0, D)` refs, `(1,)` broadcasts, no spatial index, ±1 around the atom boundary, and a structured / variable-width dtype that `str(np.dtype(...))` cannot round-trip — the pair that turned `info --stats` into exit 1 and `--dry-run` into a traceback on the third-party stores `--generic` exists for); never-shrink; a skip reason that does not overclaim (a trailing-axis-chunked array is neither a "broadcast" nor a "single chunk"); destination guards (in-place, containment either way, a non-store destination, staged output so a failure leaves nothing behind, and a `--generic` gate that tests recognized marker VALUES rather than attr presence — both directions, including unhashable attr values); a `--verify` that actually discriminates (flipped byte, missing array, shape/dtype/attr mismatch, `-0.0` and NaN payloads, and a variable-width `StringDType` array whose `tobytes()` compares arena descriptors rather than characters); the bounded-memory slab loops actually iterating, in both the copy and the verify; the `content_hash` change that stops a warm viewer cache serving chunks whose keys have moved, including under `--generic`; and the proof that the slab-wise scene hash is byte-identical to `compute_content_hashes` (over an F-order store and a sharded one, the term #1719 is in flight to fold in). The compiled scenes are session-scoped fixtures — measured, a per-test compile costs ~13 s on every run of this file (63.7 s vs 50.5 s back to back on one machine), not the "~5 min" an earlier note here claimed
- `test_lod_restamp.py` - Tests for `luxar.io.lod_restamp` / `luxar restamp-lod`: legacy `coverage` ladders re-derived under the `screen-area` selector, per anchor (whole-object, the fills-screen one only under a REAL multi-part partition, and the one-part-partition and one-part-inside-a-tiling shapes that pin the writers' `under_partition or len(children) > 1` rule); `child_index` ordering on a twelve-level ladder, where a name sort diverges; a wrapper level summed over its leaves; a single-level ladder; loud skips that never write (an out-of-vocabulary `selector`, an unresolvable or zero finest element count) versus a missing COARSER count, which is harmless and reported as `None`; a mixed store where the already-`screen-area` group is untouched; the `content_hash` moving on an attrs-only change and NOT moving on a no-op or a dry run; idempotency down to the hash; `--group` restriction, leading-slash tolerance, and an unmatched path failing before anything is written; refusal of a `.zarr.zip` and of a non-Luxar store; a real `gsplat lod --recipe levels` store whose root IS the ladder, landing back on the digest the current writer stamps; and a direct test that the re-verification is not vacuous (both the per-node documents and the consolidated index object to a ladder that did not land)
- `test_progressive_writing.py` - Tests for progressive (streaming) writing and deterministic cleanup of compiler-owned temporary scenes on successful or failed finalization
- `test_writer_parent_parameter.py` - Tests for parent parameter in writer
- `test_io_metadata.py` - Tests for metadata consolidation and storage
- `test_volume_lazy.py` - Tests for lazy volume access (`open_volume_lazy` array selection, `pin_volume_axes`, splat-dim → volume-axis maps)

## Running

```bash
hatch run pytest packages/luxar/src/luxar/io/tests/
```
