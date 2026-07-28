# luxar.io._ordering.curves

**Space-filling curve encoders** for the spatial-ordering layer. The modules here implement Morton (Z-order) and Hilbert curve encoding, mapping nD integer grid coordinates to scalar codes whose sort order is spatially local. Both use Numba JIT-compiled kernels for fast parallel encoding, with pure-NumPy fallbacks kept in 1:1 sync.

## Purpose

Provide **space-filling curve primitives** that enable spatial locality in chunked zarr stores:

- **Morton (Z-order)**: Bit interleaving — fastest to compute, good locality
- **Hilbert**: Better locality than Morton at the cost of a more involved transform
- **128-bit Morton**: For high-dimensional data (> 6 dims) where 64-bit codes have insufficient precision

## Module Ownership

| Module | Curve | Entry Points | Numba Kernel |
|--------|-------|--------------|--------------|
| **morton.py** | Morton (Z-order) | `morton_encode_nd(coords, bits_per_dim=16)`, `morton_encode_128bit(coords, bits_per_dim)` | `_morton_numba_kernel` (lazy-compiled) |
| **hilbert.py** | Hilbert | `hilbert_encode_nd(coords, bits_per_dim=16)` | `_hilbert_numba_kernel` (lazy-compiled) |

## Call Flow

### Morton Encoding (`morton.py`)

**`morton_encode_nd(coords, bits_per_dim=16)`** — 64-bit Morton codes:

1. **Input**: `coords` (N, d) integer array, `bits_per_dim` (bit budget per dimension)
2. **Kernel selection**:
   - Try Numba JIT-compiled kernel (`_get_morton_numba_kernel`) on first use (lazy-compile, cached in `_morton_numba_kernel` module global)
   - If Numba is unavailable or compilation fails, set `_morton_numba_kernel = False` (tried and failed)
   - Once loaded, the kernel is reused for all subsequent calls
3. **Numba path** (if available):
   - Allocate `out = np.empty(N, dtype=np.uint64)`
   - Cast `coords` to contiguous int64 array
   - Call `_morton_numba_kernel(coords, bits_per_dim, out)` — parallel bit interleaving
   - Return `out`
4. **NumPy fallback** (if Numba is unavailable):
   - Vectorized bit interleaving over all points:
     ```python
     morton = np.zeros(n_points, dtype=np.uint64)
     for bit in range(bits_per_dim):
         for dim in range(n_dims):
             coord_bit = (coords[:, dim] >> bit) & 1
             morton |= coord_bit.astype(np.uint64) << (bit * n_dims + dim)
     ```
   - Return `morton`

**Bit interleaving layout**: For each bit position `b` (LSB first), interleave bits from all dimensions `d` in order: `bit(b, dim=0), bit(b, dim=1), ..., bit(b, dim=d-1)`. This produces a Z-order (Morton) curve.

**`morton_encode_128bit(coords, bits_per_dim)`** — 128-bit Morton codes:

1. For high-dimensional data (> 6 dims), 64-bit codes overflow
2. Returns `(high, low)` paired uint64 values:
   - `low` holds bits `[0:64)`
   - `high` holds bits `[64:128)`
3. Same bit interleaving as `morton_encode_nd`, split across two uint64 arrays

### Hilbert Encoding (`hilbert.py`)

**`hilbert_encode_nd(coords, bits_per_dim=16)`** — 64-bit Hilbert codes:

1. **Input**: `coords` (N, d) integer array, `bits_per_dim`
2. **Kernel selection**:
   - Try Numba JIT-compiled kernel (`_get_hilbert_numba_kernel`) on first use (lazy-compile, cached in `_hilbert_numba_kernel` module global)
   - If Numba is unavailable or compilation fails, set `_hilbert_numba_kernel = False`
3. **Numba path** (if available):
   - Allocate `out = np.empty(N, dtype=np.uint64)`
   - Call `_hilbert_numba_kernel(coords.astype(np.int64), bits_per_dim, out)` — implements Skilling's "Programming the Hilbert curve" algorithm
   - Return `out`
4. **NumPy fallback** (if Numba is unavailable):
   - Import `hilbertcurve` library (pure Python, slow for large N)
   - Create `HilbertCurve(bits_per_dim, n_dims)`
   - Loop over points: `hilbert_indices[i] = hilbert.distance_from_point(coords[i])`
   - Return `hilbert_indices`
   - **Raises ImportError** if neither Numba nor `hilbertcurve` is available

**Hilbert algorithm** (Numba kernel):

1. **Inverse undo excess work** — removes redundant Gray-code bits
2. **Gray encode** — converts to Gray code
3. **Transpose to Hilbert integer** — MSB-first bit interleave (matches `hilbertcurve` library convention: MSB of dim 0 first)

### Kernel Caching

Both encoders cache the compiled Numba kernels as module globals:

- `_morton_numba_kernel` — `None` (not tried), `False` (tried and failed), or compiled callable
- `_hilbert_numba_kernel` — same

On first use:
1. If `kernel is None`, try `_get_*_numba_kernel()` (lazy-compile)
2. If compilation succeeds, cache the kernel callable
3. If compilation fails (ImportError or any exception), set `kernel = False` (never try again)
4. Subsequent calls reuse the cached kernel (fast) or go straight to the fallback (if `False`)

## Key Invariants

1. **Permutation equivariance**: `encode(coords[perm]) == encode(coords)[perm]` for any permutation `perm` — a pure function of the coordinates, so a spatial sort is independent of input row order. Verified by hypothesis property tests (`io/tests/test_ordering_properties.py`).

2. **Numba vs NumPy parity**: The JIT kernels and fallbacks produce byte-identical codes. The Numba kernel is the fast path; the NumPy fallback is the reference implementation. Verified by deterministic tests that force the fallback and compare (`test_morton_numba_numpy_parity`, `test_hilbert_numba_numpy_parity` in `io/tests/test_ordering_properties.py`).

3. **Lazy compilation**: Kernels are compiled on first use, not at import time. This avoids blocking at module load and isolates Numba import errors to the first call.

4. **Bit budget**: The `bits_per_dim` parameter controls the grid resolution. Default is 16 bits/dim (grid size `2^16 = 65536`). The total code width is `bits_per_dim * n_dims` (capped at 64 bits for `*_encode_nd`, 128 bits for `morton_encode_128bit`).

5. **Contiguous input**: The Numba kernels require contiguous int64 input. `morton_encode_nd` converts via `coords.astype(np.int64)` before calling the kernel; multi-dimensional non-C-contiguous input is normalized automatically.

6. **Hilbert convention**: The MSB of dimension 0 comes first in the bit interleave (matches the `hilbertcurve` library convention). This is the opposite order of Morton (which interleaves LSB-first).

## Testing

The curve encoders are exercised by multiple test suites:

- **test_ordering_properties.py** — Hypothesis property tests:
  - `test_morton_is_permutation_equivariant` — Morton equivariance
  - `test_hilbert_is_permutation_equivariant` — Hilbert equivariance
  - `test_morton_numba_numpy_parity` — Morton Numba vs NumPy byte-identical codes
  - `test_hilbert_numba_numpy_parity` — Hilbert Numba vs NumPy byte-identical codes

- **test_ordering_points.py** / **test_ordering_lines.py** / **test_ordering_gsplats.py** — Integration tests for each geometry type, verifying that sorted data has better spatial locality than unsorted

- **test_ordering_compound.py** — Compound ordering correctness (barrier dims, chunk-straddling invariant)

## Usage (Internal Only)

These encoders are NOT public API. They are called by the higher-level ordering functions:

- `_ordering/compound.py::_compound_sort` — Uses Morton or Hilbert to order spatial dimensions within barrier values
- `_ordering/points.py::sort_points_compound` — Points-specific wrapper
- `_ordering/lines.py::sort_segments_compound` — Lines-specific wrapper
- `_ordering/gsplats.py::sort_splats_spatial` — GSplats-specific wrapper

**Example call chain** (Points):

```
LuxarZarrCompiler.write_points(...)
  → geometry_writers/points.py::write_points(...)
  → spatial_ordering/points.py::build_points_ordering(...)
  → _ordering/points.py::sort_points_compound(positions, dimensions, method="hilbert")
  → _ordering/compound.py::_compound_sort(coords, slice_dims, ordering_dims, method="hilbert")
  → _ordering/curves/hilbert.py::hilbert_encode_nd(grid_coords, bits_per_dim=16)
  → _hilbert_numba_kernel(coords, bits_per_dim, out)  # or numpy fallback
```

## Performance Characteristics

**Numba JIT kernels** (when available):
- **Compilation overhead**: ~100-500 ms on first use (lazy, one-time)
- **Encoding speed**: ~1-10 million points/sec (depends on dimensionality and CPU)
- **Parallelization**: Numba auto-vectorizes and parallelizes over points

**NumPy fallback**:
- **Morton**: Vectorized over all points, reasonably fast (~500k-2M points/sec)
- **Hilbert**: Pure Python loop via `hilbertcurve` library, slow for large N (~1k-10k points/sec)

**Recommendation**: Install Numba for production workloads (`pip install numba`). The fallback is sufficient for small datasets (< 10k points) or testing.

## See Also

- `../compound.py` — Compound ordering core (uses these encoders)
- `../grid.py` — Grid normalization (prepares input for these encoders)
- `../README.md` — Ordering package overview
- `../../tests/test_ordering_properties.py` — Property-based tests
- External dependencies: `numba` (optional, recommended), `hilbertcurve` (optional, fallback)
