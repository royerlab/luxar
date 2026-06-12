# Bug: GSplats chunk_bounds partitions vs zarr chunk indices mismatch

## Summary

The viewer fails to display gsplats data for nodes where `chunk_bounds` has more spatial partitions than the zarr array has actual chunks. The viewer interprets `chunk_bounds` partition indices as zarr chunk indices and issues HTTP requests for chunks that don't exist (404 errors).

## Symptoms

- In a 4D scene with 64 gsplats nodes (one per timepoint), only the first ~2 timepoints are visible
- Moving the time slider shows "No visible gsplats" for most nodes, or partially loads data with 404 errors
- Console shows:
  ```
  [PointSpatialIndexLoader] No visible gsplats - returning empty data
  [SceneLoader] Clearing gsplats for /gsplats_t0005 (no visible splats at current slice)
  ```
- When data IS loaded, 404 errors appear for non-existent chunk files:
  ```
  Failed to load resource: 404 (Not Found) @ .../gsplats_t0032/centers/1.0
  Failed to load resource: 404 (Not Found) @ .../gsplats_t0032/centers/0.1
  Failed to load resource: 404 (Not Found) @ .../gsplats_t0032/amplitudes/1
  ```

## Root Cause

There is a mismatch between two concepts:

1. **`chunk_bounds`** — A spatial index stored in the zarr metadata. Created by the `LuxarZarrCompiler` during Morton/Hilbert ordering. It describes **logical spatial partitions** of the data (e.g., "splats 0-800 are in this bounding box, splats 801-1600 in that one").

2. **Zarr array chunks** — The actual on-disk chunk files. For small gsplats nodes (~1k-7k splats), the entire array fits in a single zarr chunk (the chunk shape equals the full array shape).

### Example: `gsplats_t0032` (2,508 splats)

| Property | Value |
|----------|-------|
| `chunk_bounds` shape | `(3, 4, 2)` — **3 spatial partitions** |
| `centers` zarr chunks | `[2508, 4]` — **1 chunk** (all data in chunk `0.0`) |
| `amplitudes` zarr chunks | `[2508]` — **1 chunk** (all data in chunk `0`) |

The viewer sees 3 partitions in `chunk_bounds` and requests zarr chunks `0.0`, `1.0`, and `0.1`. But only `0.0` exists on disk. The requests for `1.0` and `0.1` return 404.

### Contrast with `gsplats_t0000` (1,214 splats)

| Property | Value |
|----------|-------|
| `chunk_bounds` shape | `(2, 4, 2)` — **2 spatial partitions** |
| `centers` zarr chunks | `[1214, 4]` — **1 chunk** |

This partially works because the first partition loads from chunk `0.0`, and the 404 on chunk `1.0` just loses the second half of the data (the early timepoints have so little signal that even partial data looks OK).

### Late timepoints (e.g., `gsplats_t0063`, 7,059 splats)

| Property | Value |
|----------|-------|
| `chunk_bounds` shape | `(7, 4, 2)` — **7 spatial partitions** |
| `centers` zarr chunks | `[7059, 4]` — **1 chunk** |

The viewer tries to load 7 zarr chunks, but only chunk `0.0` exists. 6 out of 7 partitions get 404s, so most of the data is lost.

## Affected Code Paths

### Compiler side (Python — data generation)

- `packages/luxar/src/luxar/io/compiler.py` — The `LuxarZarrCompiler` writes:
  - Zarr arrays with chunk sizes that may differ from the spatial partition count
  - `chunk_bounds` metadata describing the spatial ordering partitions

- The compiler's Morton/Hilbert ordering creates N partitions based on the number of splats and the ordering algorithm, but writes the data into zarr with a single chunk (or a different chunk count).

### Viewer side (TypeScript — data loading)

- `packages/luxar-viewer/src/` — The `PointSpatialIndexLoader`:
  - Reads `chunk_bounds` to determine which partitions are visible at the current view state
  - Maps partition indices to zarr chunk fetch requests
  - Assumes partition index == zarr chunk index

## Reproduction

```bash
# Generate a 4D zebrafish scene (uses cached GSplats if available)
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_zebrafish_timelapse.py --no-serve

# Serve and open in browser
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_zebrafish_timelapse.py --serve-only
```

Then in the viewer:
1. Scene loads at time=0 — some faint data visible (first 1-2 timepoints partially loaded)
2. Move the time slider to time=32 — the viewer loads 3 chunks but gets 404 on 2 of them
3. Move to time=63 — the viewer tries to load 7 chunks but only gets 1

## Verification

```python
import zarr, numpy as np

store = zarr.open('datasets/demos/gsplats_4d_zebrafish_timelapse.luxar.zarr', 'r')

for name in ['gsplats_t0000', 'gsplats_t0032', 'gsplats_t0063']:
    g = store[name]
    cb = np.array(g['chunk_bounds'])
    print(f"{name}:")
    print(f"  chunk_bounds partitions: {cb.shape[0]}")
    print(f"  centers zarr chunks: {g['centers'].chunks} for shape {g['centers'].shape}")
    print(f"  → mismatch: {cb.shape[0]} partitions vs {g['centers'].shape[0] // g['centers'].chunks[0] + (1 if g['centers'].shape[0] % g['centers'].chunks[0] else 0)} zarr chunks")
```

## Possible Fixes

### Option A: Compiler — align zarr chunk size with spatial partitions

Make the compiler set the zarr chunk size so that each spatial partition maps to exactly one zarr chunk. When Morton ordering creates N partitions for M splats, set the zarr chunk size to `ceil(M / N)` along the first axis.

### Option B: Viewer — load by byte range within single chunk

Instead of mapping partition index → zarr chunk index, the viewer should:
1. Know the zarr chunk layout (from `.zarray` metadata)
2. Load the correct zarr chunk(s) that contain the byte range for each spatial partition
3. Extract the relevant rows from the loaded chunk

### Option C: Viewer — treat chunk_bounds as a spatial filter, not chunk lookup

Load all zarr chunks for a visible node, then use `chunk_bounds` client-side to filter which splats to render. This is simpler but loads more data than necessary.

## Context

This bug was discovered while testing the `demo_gsplats_4d_zebrafish_timelapse.py` demo, which creates a 4D scene with 64 gsplats nodes (one per timepoint of a zebrafish embryo confocal recording). The same bug would affect any scene with gsplats nodes that have more spatial partitions than zarr chunks — which happens whenever the splat count is small enough to fit in a single zarr chunk but the Morton/Hilbert ordering creates multiple partitions.

The existing `demo_gsplats_3d_cells3d_multichannel.py` works because it has only 2 channel nodes with ~15,000 splats each — likely producing chunk counts that match partition counts. The zebrafish demo has 64 nodes with 1,200–7,000 splats each, which consistently triggers the mismatch.

## Resolution

**Fixed** by passing `ordering_data` / `spatial_index_data` to `_calculate_intelligent_chunks()` at all 11 call sites in `compiler.py` that were missing it. Also fixed the function itself to handle 1D arrays (radii, amplitudes, sharpness, widths) with spatial index data — previously only 2D arrays used the `chunk_size` from spatial ordering.

**Note on symptoms**: The viewer uses `zarrita.get(array, sliceSpec)` with logical row ranges (not direct chunk file paths), so zarrita handles the physical chunk mapping internally. The primary consequence of misaligned chunks is **bandwidth waste** (the same physical zarr chunk fetched repeatedly for different partitions), not HTTP 404 errors. The 404 errors described above may have been observed with a different zarr client or HTTP serving configuration.

The fix was applied to:
- `write_gsplats()`: centers, amplitudes, cholesky_factors, colors, sharpness (5 sites)
- `write_lines()`: widths, colors, sharpness (3 sites)
- `_write_colors_dataset()`, `_write_radii_dataset()`, `_write_sharpness_dataset()` (3 sites, used by `write_points()`)
- `_calculate_intelligent_chunks()`: 1D branch now checks `spatial_index_data`

**Option A** (compiler-side alignment) was implemented. Existing data must be re-compiled to benefit.
