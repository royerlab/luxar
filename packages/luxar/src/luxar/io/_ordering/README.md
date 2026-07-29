# luxar.io._ordering

**Internal ordering primitives** for the spatial-indexing layer. The modules here implement the space-filling-curve encoders (Morton, Hilbert), grid normalization, compound ordering (categorical barrier → spatial curve within), and per-geometry ordering glue (Points, Lines, GSplats). Application code uses the high-level `LuxarZarrCompiler` with `enable_spatial_index=True`; these primitives are never imported directly.

## Purpose

Provide the **shared spatial-ordering infrastructure** that drives all three geometry types:

- **Space-filling curves** (Morton, Hilbert) map nD integer grid coordinates to scalar codes whose sort order is spatially local
- **Compound ordering** lexsorts categorical/barrier axes first, then a space-filling curve within each barrier value (time-series and nD slicing optimization)
- **Chunk-bounds computation** extends chunks by geometric extents (radii, ellipsoidal σ) so a query never misses an element straddling a chunk boundary
- **Per-geometry glue** sequences the primitives for each geometry type (Points, Lines, GSplats)

## Module Ownership

| Module | Responsibility | Key Entry Points |
|--------|---------------|------------------|
| **curves/morton.py** | Morton (Z-order) encoding | `morton_encode_nd(coords, bits_per_dim=16)`, `morton_encode_128bit(coords, bits_per_dim)` |
| **curves/hilbert.py** | Hilbert curve encoding | `hilbert_encode_nd(coords, bits_per_dim=16)` |
| **grid.py** | Grid normalization | `normalize_coords_to_grid(coords, min_coords, max_coords, resolution)`, `compute_auto_resolution(coords, max_resolution=2**16)` |
| **compound.py** | Compound ordering core | `_compound_sort(coords, slice_dims, ordering_dims, method="hilbert")`, `detect_barrier_dims(centers, max_cardinality=1024)` |
| **bounds.py** | Shared chunk-bounds constants | `_BARRIER_BOUND_EPS` |
| **points.py** | Points-specific glue | `sort_points_compound(positions, dimensions, method="hilbert")`, `compute_chunk_bounds_points(positions, radii, chunk_size, slice_dims=None)` |
| **lines.py** | Lines-specific glue | `convert_to_indexed(n_vertices, line_type, indices)`, `order_lines_spatial(vertices, segments, dimensions, method="hilbert")`, `sort_segments_compound(segment_coords_2d, dimensions, method="hilbert")`, `compute_vertex_chunk_bounds(vertices, chunk_size, slice_dims=None)`, `compute_segment_chunk_bounds(vertices, segments, widths, chunk_size, slice_dims=None)` |
| **gsplats.py** | GSplats-specific glue | `sort_splats_spatial(centers, method="hilbert", resolution=None, slice_dims=None)`, `compute_chunk_bounds_gsplats(centers, cholesky_factors, chunk_size, coverage_sigma=3.0, slice_dims=None)` |

## Call Flow

### Space-Filling Curves (`curves/`)

**Morton (Z-order)** — bit interleaving:

1. `morton_encode_nd(coords, bits_per_dim)`:
   - Tries Numba JIT-compiled kernel (`_get_morton_numba_kernel`) for fast single-threaded encoding (`@numba.njit(cache=True)`, no `parallel`/`prange`)
   - Falls back to vectorized NumPy if Numba is unavailable
   - Returns uint64 codes, shape `(N,)`

2. `morton_encode_128bit(coords, bits_per_dim)`:
   - For high-dimensional data (> 6 dims), 64-bit codes have insufficient precision
   - Returns `(high, low)` paired uint64 values

**Hilbert** — space-filling curve with better locality than Morton:

1. `hilbert_encode_nd(coords, bits_per_dim)`:
   - Tries Numba JIT-compiled kernel (`_get_hilbert_numba_kernel`) — implements Skilling's "Programming the Hilbert curve" algorithm
   - Falls back to `hilbertcurve` library (pure Python, slow for large N) if Numba is unavailable
   - Returns uint64 codes, shape `(N,)`

Both encoders are **permutation-equivariant**: reordering the input rows reorders the output codes identically, so a spatial sort is independent of input row order (verified by property-based tests in `io/tests/test_ordering_properties.py`).

**Numba vs fallback parity**: The JIT kernels and their fallbacks are kept in 1:1 sync via deterministic tests that force the fallback and assert byte-identical codes (verified by `test_morton_numba_numpy_parity`, `test_hilbert_numba_numpy_parity`). Morton's fallback is pure NumPy; Hilbert's fallback is the external `hilbertcurve` library (not NumPy).

### Grid Normalization (`grid.py`)

`normalize_coords_to_grid(coords, min_coords, max_coords, resolution)`:
- Maps float coordinates `[min_coords, max_coords]` to integer grid `[0, resolution - 1]`
- Per-axis linear rescaling with truncation: `grid_coord = ((coord - min) / (max - min) * (resolution - 1)).astype(uint32)` (integer cast truncates toward zero, not round)
- Handles degenerate axes (min == max) by mapping everything to grid coordinate `0` (the range is forced to `1.0`, so the normalized value is `0`)
- Returns `(N, d)` integer array, clipped to `[0, resolution - 1]`

### Compound Ordering (`compound.py`)

**`_compound_sort(coords, slice_dims, ordering_dims, method="hilbert")`** — the single source of truth shared by Points, Lines, and GSplats:

1. **Barrier/categorical axes** (`slice_dims`): lexsort first (e.g., time, channel)
2. **Spatial axes** (`ordering_dims`): Morton/Hilbert code within each barrier value

**Invariant**: A chunk never straddles a categorical value (a splat at time=0 never extends into time=1's chunk bounds).

**Bits budget**: Split across ONLY the spatial (ordering) dims, so excluding a barrier axis gives spatial axes more resolution.

**Degenerate cases**:
- `ordering_dims == []` (all discrete): spatial codes are all zero, sort is pure lexsort on barrier dims
- `slice_dims == []` (pure spatial): reduces to pure spatial ordering over all of `ordering_dims`

**Returns**: `(sort_indices, metadata)` where metadata includes `{"ordering": "morton"/"hilbert", "slice_dims": [...], "ordering_dims": [...], "ordering_min": [...], "ordering_max": [...], "ordering_bits_per_dim": int}`

**`detect_barrier_dims(centers, max_cardinality=1024)`** — heuristically identify categorical/barrier axes for provenance-less standalone `.gsplats.zarr`:

1. An axis qualifies iff:
   - Values are integers (`np.allclose(col, np.round(col), rtol=0.0, atol=1e-3)`) — rtol=0 ensures large-magnitude continuous floats are never "close enough" to integers
   - Few distinct values (`<= max_cardinality` AND `n_unique * 4 <= n`) — rejects a fine integer spatial grid

2. **Conservative in the safe direction**: A false NEGATIVE (missing a barrier) only causes over-fetch (no worse than pure spatial ordering). A false POSITIVE (flagging a spatial axis) gives it tight epsilon-padded chunk bounds with no σ expansion, so a spatially-extended splat can fall outside its chunk bounds and be *dropped* from a query — a correctness bug. Both guards err toward NOT flagging.

3. **Subordinate by design**: Callers apply explicit `barrier_dims` and `coarsen_dims` complements first (the scene compiler passes scene `Dimension.discrete` dims; the batch merge passes the stacked-time axis), using this only as the last resort for provenance-less standalone files.

### Barrier Bounds (`bounds.py`)

`_BARRIER_BOUND_EPS = 1e-3` — absolute padding added to barrier/discrete-dimension chunk bounds. This is ONLY a float-boundary safety margin (the query "reach" lives entirely in the reader's per-dimension tolerance). It used to be 0.5 (half a step); combined with the reader's own half-step tolerance that summed to a full step and made a single-category query (e.g., one timepoint) pull in the entire neighbouring category. Keep this tiny.

**KNOWN LIMIT**: The pad is absolute while the reader's reach is step-scaled (`0.25 × step`), so for pathological discrete steps below ~1.3e-3 the pad reaches past the neighbour category's quarter-step boundary and the over-fetch returns. Step metadata is not plumbed into these bound builders; discrete/categorical dims with milli-scale steps are not a supported layout (rescale the axis instead).

### Per-Geometry Glue

**Points** (`points.py`):
- `sort_points_compound(positions, dimensions, method="hilbert")` → `(sort_indices, metadata)`
  - Splits `dimensions` into `slice_dims` (`d.discrete and not d.display`) and `ordering_dims` (`not d.discrete or d.display` — i.e. spatial dims AND any displayed dim, so continuous non-display dims count as ordering dims too)
  - Delegates to `_compound_sort`
- `compute_chunk_bounds_points(positions, radii, chunk_size, slice_dims=None)` → `(num_chunks, d, 2)` bounds array
  - Radius expansion is applied to all NON-`slice_dims` axes (displayed dims AND continuous non-display dims). For a broadcast scalar radius the box is `[min - r, max + r]`; for per-point radii the code takes the per-point envelope `(p - r).min()` / `(p + r).max()` (not a single chunk-wide `r_max`)
  - Tight `[min - eps, max + eps]` for discrete (`slice_dims`) axes (no radius expansion on categorical axes)
  - `radii` is `Optional`: a per-point array OR a broadcast scalar OR `None`. When `None`, spatial axes get a safety margin of 1%-of-chunk-range or `0.01` (whichever is larger) so points aren't missed at chunk boundaries once a default render radius is applied

**Lines** (`lines.py`):
- `order_lines_spatial(vertices, segments, dimensions, method="hilbert")` → `(sorted_vertices, sorted_segments, vertex_sort_indices, segment_sort_indices, metadata)` (5-tuple)
  - Dual-indexed: reorders both vertices (in D-space, via `sort_points_compound`) AND segments (in 2×D-space, via `sort_segments_compound`)
  - Returns two sort orders (one for vertices, one for segments)
- `sort_segments_compound(segment_coords_2d, dimensions, method="hilbert")` → `(sort_indices, metadata)`
  - Takes pre-built `(S, 2·D)` segment coordinates (both endpoints concatenated), NOT midpoints — this captures position, orientation, and length
  - Returns a single sort order over the segments
  - Has its OWN bit budget (distinct from `_compound_sort`'s): the endpoint concatenation doubles the ordering-dim count, so when `64 // n_ordering_dims < 10` it escapes to a 128-bit code (`min(21, 128 // n_ordering_dims)`); Hilbert has no 128-bit kernel, so the 128-bit path silently falls back to Morton
- `compute_vertex_chunk_bounds(vertices, chunk_size, slice_dims=None)` → `(num_chunks, D, 2)` bounds array
  - Exact spatial bounds (no size expansion for vertices); tight epsilon-padded bounds for discrete dims
- `compute_segment_chunk_bounds(vertices, segments, widths, chunk_size, slice_dims=None)` → `(num_chunks, D, 2)` bounds array
  - Per-chunk box extends each spatial axis by the full per-segment max endpoint width (`p ± max_w`, no `/2`)
  - Tight epsilon-padded bounds for discrete dims

**GSplats** (`gsplats.py`):
- `sort_splats_spatial(centers, method="hilbert", resolution=None, slice_dims=None)` → `(sort_indices, metadata)`
  - `slice_dims` is explicit (from scene `Dimension.discrete` or a stacked-time axis); this function does NOT auto-detect — the caller (`_compiler/gsplat_assembly.py`) falls back to `detect_barrier_dims` when no explicit dims are supplied
  - Splits center columns into `slice_dims` (barrier) and `ordering_dims` (complement)
  - Delegates to `_compound_sort`
- `compute_chunk_bounds_gsplats(centers, cholesky_factors, chunk_size, coverage_sigma=3.0, slice_dims=None)` → `(num_chunks, d, 2)` bounds array
  - Per-chunk box extends by the requested Gaussian coverage on spatial axes
  - Tight epsilon-padded bounds for barrier dims

## Key Invariants

1. **Permutation equivariance**: `curve_encode(coords[perm]) == curve_encode(coords)[perm]` for any permutation `perm` — verified by hypothesis property tests (`test_ordering_properties.py`).

2. **Numba vs fallback parity**: The JIT kernels and their fallbacks (pure NumPy for Morton, the `hilbertcurve` library for Hilbert) produce byte-identical codes — verified by forcing the fallback and comparing (`test_morton_numba_numpy_parity`, `test_hilbert_numba_numpy_parity`).

3. **Compound sort invariant**: A chunk never straddles a categorical value. The barrier dims are lexsorted first, then the spatial curve codes are sorted within each barrier value.

4. **Tight barrier bounds**: Discrete/barrier dimensions get `[min - eps, max + eps]` bounds with no geometric extent expansion (no radius, no σ). A splat at time=0 never extends into time=1's chunk bounds.

5. **Conservative barrier detection**: `detect_barrier_dims` errs toward NOT flagging an axis as a barrier (a false positive drops splats; a false negative only causes over-fetch).

6. **Bits budget**: The bit budget is split across ONLY the spatial (ordering) dims, so excluding a barrier axis gives the spatial axes more resolution (e.g., 3D + time: barrier=time, ordering=xyz gets 21 bits/dim vs 16 if all 4 dims were spatial).

## Testing

The ordering primitives are exercised by multiple test suites:

- **io/tests/test_ordering_properties.py** — Hypothesis property tests (permutation equivariance, Numba/NumPy parity)
- **io/tests/test_ordering_points.py** / **io/tests/test_ordering_lines.py** / **io/tests/test_ordering_gsplats.py** — Integration tests for each geometry (barrier dims, chunk-straddling invariant)
- **gsplats/io/tests/test_ordering.py** — GSplat-specific ordering tests (barrier detection, chunk bounds)

## See Also

- `../ordering.py` — Public `luxar.io.ordering` API (re-exports the high-level functions)
- `../_compiler/spatial_ordering/` — Compiler glue to these primitives (Points/Lines)
- `../_compiler/gsplat_assembly.py` — GSplat ordering integration
- `../../encoding/README.md` — Encoding layer (delta filter benefits from spatial ordering)
- `curves/README.md` — Space-filling-curve encoder details
