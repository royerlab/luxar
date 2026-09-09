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
| **bounds.py** | Shared chunk-bounds constants, `slice_dims` / slack sanitising, and the float32 outward store | `_BARRIER_BOUND_EPS`, `_normalise_slice_dims(slice_dims, ndim)`, `_normalise_coord_slack(coord_slack, ndim)`, `_normalise_scalar_slack(scalar_slack)`, `_store_outward_f32(lo, hi)`, `_store_outward_f32_array(lo, hi)` |
| **points.py** | Points-specific glue | `sort_points_compound(positions, dimensions, method="hilbert")`, `compute_chunk_bounds_points(positions, radii, chunk_size, slice_dims=None, *, coord_slack=None, scalar_slack=None)` |
| **lines.py** | Lines-specific glue | `convert_to_indexed(n_vertices, line_type, indices)`, `order_lines_spatial(vertices, segments, dimensions, method="hilbert")`, `sort_segments_compound(segment_coords_2d, dimensions, method="hilbert")`, `compute_vertex_chunk_bounds(vertices, chunk_size, slice_dims=None, *, coord_slack=None)`, `compute_segment_chunk_bounds(vertices, segments, widths, chunk_size, slice_dims=None, *, coord_slack=None, scalar_slack=None)` |
| **gsplats.py** | GSplats-specific glue | `sort_splats_spatial(centers, method="hilbert", resolution=None, slice_dims=None)`, `compute_chunk_bounds_gsplats(centers, cholesky_factors, chunk_size, coverage_sigma=2.75, slice_dims=None, *, coord_slack=None)` |

## Call Flow

### Space-Filling Curves (`curves/`)

**Morton (Z-order)** — bit interleaving:

1. `morton_encode_nd(coords, bits_per_dim)`:
   - Tries a Numba JIT-compiled kernel (`_get_morton_numba_kernel`) for fast single-threaded encoding (explicit typed signature, no `parallel`/`prange`); the signature eagerly compiles the kernel, requires C-contiguous input/output arrays, and accepts read-only input
   - Falls back to vectorized NumPy if Numba is unavailable or the kernel fails to compile (warned once)
   - Returns uint64 codes, shape `(N,)`

2. `morton_encode_128bit(coords, bits_per_dim)`:
   - For high-dimensional data (> 6 dims), 64-bit codes have insufficient precision
   - Returns `(high, low)` paired uint64 values

**Hilbert** — space-filling curve with better locality than Morton:

1. `hilbert_encode_nd(coords, bits_per_dim)`:
   - Tries a Numba JIT-compiled kernel (`_get_hilbert_numba_kernel`) — implements Skilling's "Programming the Hilbert curve" algorithm and deliberately keeps an `A`-layout signature (`int64[:,:]` / `uint64[:]`)
   - Falls back to `hilbertcurve` library (pure Python, slow for large N) if Numba is unavailable or the kernel fails to compile (warned once)
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

**KNOWN LIMIT (small steps)**: The pad is absolute while the reader's reach is step-scaled (`0.25 × step`), so a chunk at category `c` is pulled into a query for `c + 1` as soon as `c + pad >= (c + step) - 0.25 × step`, i.e. as soon as `pad >= 0.75 × step` — equivalently `step <= pad / 0.75`. At the bare epsilon that is `1e-3 / 0.75 = ~1.3e-3`: below that the over-fetch returns. Step metadata is not plumbed into these bound builders; discrete/categorical dims with milli-scale steps are not a supported layout (rescale the axis instead).

Since the quantisation pad (#1655) the threshold is worse on a **non-gridded** barrier axis of a points, lines or gsplats node, because the compiler adds `coord_slack` on top of this epsilon: the effective pad is `1e-3 + extent/131070`, i.e. 8.63e-3 at an axis extent of 1000, so by the same `pad / 0.75` rule steps below `8.63e-3 / 0.75 = ~1.2e-2` over-fetch there. The ordinary stacked integer time/channel axis is unaffected — it is GRIDDED, its slack is exactly 0, and it keeps the plain 1e-3.

**KNOWN LIMIT (large coordinates)**: The outward float32 store (below) never lets a pad vanish, so the pad a barrier axis EFFECTIVELY gets is `max(_BARRIER_BOUND_EPS, up to one float32 ULP at |x|)` — 1e-3 near the origin, but 2.0 at 2e7 and 8.0 at 1e8. The two ends are asymmetric at an exact power of two, where the ULP below the binade boundary is half the one above: at `|x| = 2**23` the low end moves 0.5 and the high end 1.0. Integers stay exactly float32-representable through `2**24`, so a unit-step categorical axis with large absolute values (a millisecond timestamp, an acquisition index offset into an experiment) is a legitimate layout there and will over-fetch a whole neighbouring category. That is the deliberate trade — over-fetching a neighbour beats dropping the chunk at its own category value — but re-base such an axis near the origin if the extra traffic matters.

### Float32 Outward Store (`bounds.py`)

`_store_outward_f32(lo, hi)` (scalar, used by Points) and `_store_outward_f32_array(lo, hi)` (per-chunk `(d,)` vectors, used by GSplats and Lines) narrow a float64 interval to the float32 `chunk_bounds` array by rounding each end AWAY from the interval. Every pad the builders add is a small **absolute** quantity — a point radius, a gsplat's `coverage_sigma·σ`, a line endpoint width, `_BARRIER_BOUND_EPS` — so past `|x| ~ 2**23` it falls under half a float32 ULP and a round-to-nearest store discards it, leaving a bound TIGHTER than the footprint (elements silently dropped from queries at their own edge). Every builder therefore accumulates its interval in **float64** first and stores it through these helpers: an extent already lost to float32 arithmetic is invisible to the outward step, so the array form rejects a float32 input outright rather than casting it up.

A bound is stepped one ULP outward **only** when the cast moved it the wrong way, never unconditionally — a padless spatial vertex dim comes out exactly equal to its float32 input (pinned by `test_padless_vertex_bounds_are_exact_at_every_magnitude`).

The array form vectorises only the outward-store **branching** over an already-reduced `(d,)` vector — it is not a way to vectorise the min/max **reduce** as well. Both lines builders reduce per dimension on purpose: NumPy's outer-axis reduce over a 3-or-4-element inner row is several times slower than one reduce per column, and these run over every chunk of every dataset (measured end-to-end at 1M vertices/segments and repo-default chunk sizes: ~4-6x for `compute_vertex_chunk_bounds`, ~2.5-3.5x for `compute_segment_chunk_bounds`, whose per-chunk gather dominates more of the work; bitwise-identical output either way). `compute_chunk_bounds_gsplats` keeps its `(chunk_centers ± extents).min(axis=0)` because it materialises that padded `(n, d)` array anyway.

`_normalise_slice_dims(slice_dims, ndim)` is the single sanitiser all four builders use: it coerces each entry with `int()` and raises `ValueError` for anything outside `[0, ndim)`. Ignoring a bad index would silently cost that categorical axis its barrier treatment (it would take the geometric-extent expansion instead and bleed into the neighbouring category), so it fails loudly. A NEGATIVE index is rejected too — not because the old bounds were wrong (they were bitwise identical to `[ndim - 1]`) but because `_compound_sort` lands it in the barrier set AND in `ordering_dims`, and the raw `-1` is persisted into the store's `slice_dims` attr, where the viewer rejects the whole attr; the full argument is in the `_normalise_slice_dims` docstring. `sort_splats_spatial` runs the same check, so a bad index fails at the FIRST door — `apply_gsplat_spatial_ordering` hands one list to the sort and the bounds builder. No in-tree caller can produce one today (they either `enumerate` the actual dims or complement over `range(ndim)`; the batch merge orchestrator derives its single entry from the manifest's own axis count), but `GSplatData.save(barrier_dims=...)` is public API a user can drive directly. (The Points/Lines sorts derive their own `slice_dims` by `enumerate`-ing `Dimension` objects and take no such argument.)

### Per-Geometry Glue

**Points** (`points.py`):
- `sort_points_compound(positions, dimensions, method="hilbert")` → `(sort_indices, metadata)`
  - Splits `dimensions` into `slice_dims` (`d.discrete and not d.display`) and `ordering_dims` (`not d.discrete or d.display` — i.e. spatial dims AND any displayed dim, so continuous non-display dims count as ordering dims too)
  - Delegates to `_compound_sort`
- `compute_chunk_bounds_points(positions, radii, chunk_size, slice_dims=None, *, coord_slack=None, scalar_slack=None)` → `(num_chunks, d, 2)` bounds array
  - Radius expansion is applied to all NON-`slice_dims` axes (displayed dims AND continuous non-display dims). For a broadcast scalar radius the box is `[min - r, max + r]`; for per-point radii the code takes the per-point envelope `(p - r).min()` / `(p + r).max()` (not a single chunk-wide `r_max`)
  - Tight `[min - eps, max + eps]` for discrete (`slice_dims`) axes (no radius expansion on categorical axes)
  - `radii` is `Optional`: a per-point array OR a broadcast scalar OR `None`. `None` does NOT mean "no extent" — a points node that stores no radii array is still drawn with the renderer's default radius, so spatial axes are expanded by `DEFAULT_POINT_RADIUS` (`luxar.typing_utils.constants`, mirrored in `packages/luxar-viewer/src/config/constants.ts`), exactly as if that scalar had been passed. The pad IS the footprint, so `[min - r, max + r]` is exactly the set of query positions for which some point in the chunk can be visible — correct in both directions, not merely wide enough. (It used to be a 1%-of-chunk-range/`0.01` fudge, unrelated to the footprint: tighter than it below ~50 units of chunk range, looser above, and different for the same scene authored in different units.)

**Lines** (`lines.py`):
- `order_lines_spatial(vertices, segments, dimensions, method="hilbert")` → `(sorted_vertices, sorted_segments, vertex_sort_indices, segment_sort_indices, metadata)` (5-tuple)
  - Dual-indexed: reorders both vertices (in D-space, via `sort_points_compound`) AND segments (in 2×D-space, via `sort_segments_compound`)
  - Returns two sort orders (one for vertices, one for segments)
- `sort_segments_compound(segment_coords_2d, dimensions, method="hilbert")` → `(sort_indices, metadata)`
  - Takes pre-built `(S, 2·D)` segment coordinates (both endpoints concatenated), NOT midpoints — this captures position, orientation, and length
  - Returns a single sort order over the segments
  - Has its OWN bit budget (distinct from `_compound_sort`'s): the endpoint concatenation doubles the ordering-dim count, so when `64 // n_ordering_dims < 10` it escapes to a 128-bit code (`min(21, 128 // n_ordering_dims)`); Hilbert has no 128-bit kernel, so the 128-bit path silently falls back to Morton
- `compute_vertex_chunk_bounds(vertices, chunk_size, slice_dims=None, *, coord_slack=None)` → `(num_chunks, D, 2)` bounds array
  - Exact spatial bounds (no size expansion for vertices); tight epsilon-padded bounds for discrete dims
- `compute_segment_chunk_bounds(vertices, segments, widths, chunk_size, slice_dims=None, *, coord_slack=None, scalar_slack=None)` → `(num_chunks, D, 2)` bounds array
  - Per-chunk box extends each spatial axis by the full per-segment max endpoint width (`p ± max_w`, no `/2`)
  - Tight epsilon-padded bounds for discrete dims

**GSplats** (`gsplats.py`):
- `sort_splats_spatial(centers, method="hilbert", resolution=None, slice_dims=None)` → `(sort_indices, metadata)`
  - `slice_dims` is explicit (from scene `Dimension.discrete` or a stacked-time axis); this function does NOT auto-detect — the caller (`_compiler/gsplat_assembly.py`) falls back to `detect_barrier_dims` when no explicit dims are supplied
  - Splits center columns into `slice_dims` (barrier) and `ordering_dims` (complement)
  - Delegates to `_compound_sort`
- `compute_chunk_bounds_gsplats(centers, cholesky_factors, chunk_size, coverage_sigma=2.75, slice_dims=None, *, coord_slack=None)` → `(num_chunks, d, 2)` bounds array
  - Per-chunk box extends by the requested Gaussian coverage on spatial axes
  - Tight epsilon-padded bounds for barrier dims

## Key Invariants

1. **Permutation equivariance**: `curve_encode(coords[perm]) == curve_encode(coords)[perm]` for any permutation `perm` — verified by hypothesis property tests (`test_ordering_properties.py`).

2. **Numba vs fallback parity**: The JIT kernels and their fallbacks (pure NumPy for Morton, the `hilbertcurve` library for Hilbert) produce byte-identical codes — verified by forcing the fallback and comparing (`test_morton_numba_numpy_parity`, `test_hilbert_numba_numpy_parity`).

3. **Compound sort invariant**: A chunk never straddles a categorical value. The barrier dims are lexsorted first, then the spatial curve codes are sorted within each barrier value.

4. **Tight barrier bounds**: Discrete/barrier dimensions get `[min - eps, max + eps]` bounds with no geometric extent expansion (no radius, no σ). A splat at time=0 never extends into time=1's chunk bounds.

5. **Conservative barrier detection**: `detect_barrier_dims` errs toward NOT flagging an axis as a barrier (a false positive drops splats; a false negative only causes over-fetch).

6. **Never tighter than the footprint**: A stored `chunk_bounds` interval contains the chunk's geometric footprint at ANY coordinate magnitude. All four builders accumulate in float64 and narrow outward to float32. Their compiler callers supply the encoder's per-axis coordinate round-trip slack, so bounds contain DECODED coordinates after uint16 fixed-point storage on spatial and barrier axes. Exact grids and float32 stores add zero; a LUT adds zero only when the writer actually permits that array to use one. LUT eligibility alone is NOT exemption: the lines glue passes `allow_lut=False` because the lines writer blocks LUT on `vertices` (the spatial-index loader reads it raw), while points and gsplats keep `allow_lut=True`, matching writers that permit `positions` and `centers` to store a LUT. GSplats resolve their sigma rail first and ask for slack using that actual centers mode, so an escalated array is not padded again; a continuous barrier axis that stays uint16 gets the half-quantum pad on top of `_BARRIER_BOUND_EPS`. Points and lines additionally take one positive-scalar round-trip slack for decoded `radii`/`widths` on spatial dimensions only; direct callers may omit either slack to retain authored-value bounds. The remaining scalar/extent gap is decoded σ derived from gsplat `cholesky_factors`. Swept over magnitudes `{0, 1e-6, 1, 100, 2**23, 2**24, 2**24+1, 2**25, 1e9, 1e15, 1e20, 1e30, 3e38}` × sign × pad by `io/tests/test_ordering_properties.py::test_chunk_bounds_contain_the_footprint_at_every_magnitude`, with end-to-end decoded containment and exemption tightness in `_compiler/test_spatial_ordering.py`.

7. **Bits budget**: The bit budget is split across ONLY the spatial (ordering) dims, so excluding a barrier axis gives the spatial axes more resolution (e.g., 3D + time: barrier=time, ordering=xyz gets 21 bits/dim vs 16 if all 4 dims were spatial).

## Testing

The ordering primitives are exercised by multiple test suites:

- **io/tests/test_ordering_properties.py** — Hypothesis property tests (permutation equivariance, Numba/NumPy parity) plus the cross-builder chunk-bounds sweeps (Key Invariant 6 across magnitudes, padless exactness, `slice_dims` range checking)
- **io/tests/test_ordering_points.py** / **io/tests/test_ordering_lines.py** / **io/tests/test_ordering_gsplats.py** — Integration tests for each geometry (barrier dims, chunk-straddling invariant)
- **gsplats/io/tests/test_ordering.py** — GSplat-specific ordering tests (barrier detection, chunk bounds)
- **io/tests/\_compiler/test_spatial_ordering.py** — End-to-end compiles that read the STORED bounds back and check them against DECODED coordinates and point/line footprint scalars

## See Also

- `../ordering.py` — Public `luxar.io.ordering` API (re-exports the high-level functions)
- `../_compiler/spatial_ordering/` — Compiler glue to these primitives (Points/Lines)
- `../_compiler/gsplat_assembly.py` — GSplat ordering integration
- `../../encoding/README.md` — Encoding layer (delta filter benefits from spatial ordering)
- `curves/README.md` — Space-filling-curve encoder details
