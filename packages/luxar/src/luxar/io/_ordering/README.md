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
| **grid.py** | Grid normalization | `normalize_coords_to_grid(coords, min_vals, max_vals, grid_size)` |
| **compound.py** | Compound ordering core | `_compound_sort(coords, slice_dims, ordering_dims, method="hilbert")`, `detect_barrier_dims(centers, max_cardinality=1024)` |
| **bounds.py** | Shared chunk-bounds constants | `_BARRIER_BOUND_EPS` |
| **points.py** | Points-specific glue | `sort_points_compound(positions, dimensions, method="hilbert")`, `compute_chunk_bounds_points(sorted_positions, radii, chunk_size)` |
| **lines.py** | Lines-specific glue | `sort_segments_compound(vertices, segments, dimensions, widths, method="hilbert")`, `compute_chunk_bounds_segments(sorted_vertices, sorted_segments, widths, chunk_size)` |
| **gsplats.py** | GSplats-specific glue | `sort_splats_spatial(centers, method="hilbert", resolution=None, slice_dims=None)`, `compute_chunk_bounds_gsplats(centers, cholesky_factors, chunk_size, coverage_sigma=3.0, slice_dims=None)` |

## Call Flow

### Space-Filling Curves (`curves/`)

**Morton (Z-order)** — bit interleaving:

1. `morton_encode_nd(coords, bits_per_dim)`:
   - Tries Numba JIT-compiled kernel (`_get_morton_numba_kernel`) for fast parallel encoding
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

**Numba vs NumPy fallback parity**: The JIT kernels and fallbacks are kept in 1:1 sync via deterministic tests that force the fallback and assert byte-identical codes (verified by `test_morton_numba_numpy_parity`, `test_hilbert_numba_numpy_parity`).

### Grid Normalization (`grid.py`)

`normalize_coords_to_grid(coords, min_vals, max_vals, grid_size)`:
- Maps float coordinates `[min_vals, max_vals]` to integer grid `[0, grid_size - 1]`
- Per-axis linear rescaling: `grid_coord = round((coord - min) / (max - min) * (grid_size - 1))`
- Handles degenerate axes (min == max) by mapping everything to grid center
- Returns `(N, d)` integer array, clipped to `[0, grid_size - 1]`

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
  - Splits `dimensions` into `slice_dims` (discrete non-display) and `ordering_dims` (display)
  - Delegates to `_compound_sort`
- `compute_chunk_bounds_points(sorted_positions, radii, chunk_size)` → `(n_chunks, d)` bounds array
  - Per-chunk axis-aligned box `[min - r_max, max + r_max]` for display dims
  - Tight `[min - eps, max + eps]` for discrete dims (no radius expansion on categorical axes)
  - `radii` can be a per-point array OR a broadcast scalar

**Lines** (`lines.py`):
- `sort_segments_compound(vertices, segments, dimensions, widths, method="hilbert")` → `(vertex_order, segment_order, metadata)`
  - Dual-indexed: reorders both vertices AND segments
  - Computes segment midpoints from vertices for spatial ordering
  - Returns two sort orders (one for vertices, one for segments)
- `compute_chunk_bounds_segments(sorted_vertices, sorted_segments, widths, chunk_size)` → `(n_chunks, d)` bounds array
  - Per-chunk box extends by `max_width / 2` along each segment (both endpoints)
  - Tight epsilon-padded bounds for discrete dims

**GSplats** (`gsplats.py`):
- `sort_splats_spatial(centers, method="hilbert", resolution=None, slice_dims=None)` → `(sort_indices, metadata)`
  - `slice_dims` is explicit (from scene `Dimension.discrete` or a stacked-time axis), or falls back to `detect_barrier_dims`
  - Splits center columns into `slice_dims` (barrier) and `ordering_dims` (complement)
  - Delegates to `_compound_sort`
- `compute_chunk_bounds_gsplats(centers, cholesky_factors, chunk_size, coverage_sigma=3.0, slice_dims=None)` → `(n_chunks, d)` bounds array
  - Per-chunk box extends by the requested Gaussian coverage on spatial axes
  - Tight epsilon-padded bounds for barrier dims

## Key Invariants

1. **Permutation equivariance**: `curve_encode(coords[perm]) == curve_encode(coords)[perm]` for any permutation `perm` — verified by hypothesis property tests (`test_ordering_properties.py`).

2. **Numba vs NumPy parity**: The JIT kernels and fallbacks produce byte-identical codes — verified by forcing the fallback and comparing (`test_morton_numba_numpy_parity`, `test_hilbert_numba_numpy_parity`).

3. **Compound sort invariant**: A chunk never straddles a categorical value. The barrier dims are lexsorted first, then the spatial curve codes are sorted within each barrier value.

4. **Tight barrier bounds**: Discrete/barrier dimensions get `[min - eps, max + eps]` bounds with no geometric extent expansion (no radius, no σ). A splat at time=0 never extends into time=1's chunk bounds.

5. **Conservative barrier detection**: `detect_barrier_dims` errs toward NOT flagging an axis as a barrier (a false positive drops splats; a false negative only causes over-fetch).

6. **Bits budget**: The bit budget is split across ONLY the spatial (ordering) dims, so excluding a barrier axis gives the spatial axes more resolution (e.g., 3D + time: barrier=time, ordering=xyz gets 21 bits/dim vs 16 if all 4 dims were spatial).

## Testing

The ordering primitives are exercised by multiple test suites:

- **test_ordering_properties.py** — Hypothesis property tests (permutation equivariance, Numba/NumPy parity)
- **test_ordering_points.py** / **test_ordering_lines.py** / **test_ordering_gsplats.py** — Integration tests for each geometry
- **test_ordering_compound.py** — Compound ordering correctness (barrier dims, chunk-straddling invariant)
- **gsplats/tests/test_ordering_gsplats.py** — GSplat-specific ordering tests (barrier detection, chunk bounds)

## See Also

- `../ordering.py` — Public `luxar.io.ordering` API (re-exports the high-level functions)
- `../_compiler/spatial_ordering/` — Compiler glue to these primitives (Points/Lines)
- `../_compiler/gsplat_assembly.py` — GSplat ordering integration
- `../../encoding/README.md` — Encoding layer (delta filter benefits from spatial ordering)
- `curves/README.md` — Space-filling-curve encoder details
