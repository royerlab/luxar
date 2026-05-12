# Data Loaders Specifications

Algorithms and contracts for `src/data/loaders/`. The
[README](./README.md) covers usage; this file documents the encoding
dispatch, spatial query algorithm, and zero-allocation buffer
contract.

## 1. Encoding dispatch (`RangeLoader.detectEncoding`)

Every zarr array under a Luxar node carries an `encoding` attribute
that the loader uses to decide how to materialize raw zarr bytes into
float32 attribute buffers. The dispatch in
`range-loader.ts:detectEncoding` consults the encoding name in a
**fixed priority order**, and the priority matters: several encoding
flags can co-occur and the earlier one wins.

```
priority 1: broadcasted      → name === 'broadcasted'
priority 2: array_ref         → name === 'array_ref' (with target)
priority 3: lut               → isLUTEncodingName(name) AND lut present
priority 4: quantized         → isQuantizedEncoding(attrs)
priority 5: direct            → isDirectEncodingName(name)
otherwise:                    → throw Unknown encoding
```

### Why this order

- **broadcasted** is a constant value replicated to every element.
  Range loading is trivial (load one value, no chunk math), so it
  bypasses every later check.
- **array_ref** points at another zarr array for deduplication.
  Range loading resolves the ref FIRST, then applies the target's
  encoding — encoded values may live behind a ref.
- **LUT** decodes through an in-memory lookup table from a low-bit
  index. The LUT must be carried in `encoding.lut`; if absent,
  fall through to quantized.
- **quantized** is dtype-tagged but no LUT — `uint8` / `uint16` /
  `log_scalar_uint8` / `log_scalar_uint16` paths.
- **direct** is the fallback when the dtype is already the desired
  numeric type (`float32`, `uint8`, etc. with no decoding step).

This priority is mirrored on the Python encoder side
(`packages/luxar/src/luxar/encoding/`), so a round-trip is contractual:
encoder picks the lowest-priority encoding that fits the data; the
decoder dispatches by the same rule.

### Range-loading semantics

`RangeLoader.loadRanges` accepts `[{start, end}, ...]` ranges (in the
target array's index space) and:

1. Resolves array refs once (the resolved target is cached in
   `ArrayRefRegistry` keyed by `(rootLoc, target, hash)`).
2. For broadcasted encoding, materializes the single value across the
   union range size.
3. For chunk-backed encodings (LUT / quantized / direct), batches
   chunk reads through zarr's `get(arr, [slice(start, end), …])` and
   decodes each chunk's contiguous bytes into the output buffer.
4. When the total range size exceeds `workerThreshold` (default
   1000), the decode step is **offloaded to a worker** via
   `WorkerPool.runWithTimeout(name, kind, fn)`. The threshold is
   conservative: worker round-trip cost dominates for small payloads.

## 2. Spatial query (`SpatialQueryBuilder`)

Chunk-bounds AABB algorithm shared by Points, Lines, and GSplats
spatial-index loaders.

### Inputs

- The geometry's `ChunkBoundsIndex` (per-chunk min/max in nD).
- A `ViewState`: `{ slicePosition: number[], tolerance: number[],
displayDims: number[], dimensions: Dim[] }`.
- A geometry-specific `extendDimsTolerance` callback that adjusts
  per-dim tolerance (Lines have segment-bound padding; GSplats have
  the σ-scaled tolerance; Points use `maxRadius`).

### Algorithm

```
visibleChunks = []
for each chunk c in index:
  for each non-display dim d:
    if c.min[d] > slicePos[d] + tolerance[d]: skip chunk
    if c.max[d] < slicePos[d] - tolerance[d]: skip chunk
  visibleChunks.push(c)
sort visibleChunks by chunkIndex
ranges = mergeContiguous(visibleChunks)  // emit [{start, end}, …]
```

Display dims are excluded from the slab test (those carry the
visible/displayed axes). The merge step coalesces adjacent chunk
indices into contiguous ranges so the loader fires one zarr read per
range instead of one per chunk.

### Complexity

- `O(numChunks × ndim)` in the AABB test. Even on million-chunk
  datasets this is fast (≤1ms on commodity CPU) because we do not
  touch elements, only chunk bounds.
- Range merge is `O(numChunks)` after the sort.

### Why chunk-bounds, not element-level

Element-level visibility (testing every point's nD position against
the slice) would be `O(N)`. The chunk-bounds approach is `O(C)` where
`C = N / chunkSize`, typically 100-1000× fewer ops. Visible chunks
still load their full chunk content (the worker culls invisible
elements element-wise post-decode), so the trade-off is overload-
fetch in exchange for fast query.

## 3. Zero-allocation accumulator (`TransferableAccumulator`)

Range-loading writes through pre-allocated buffers when the caller
supplies a `TransferableAccumulator`. This avoids per-update
allocations during nD navigation (which is the hot path).

### Contract

- `TransferableAccumulator` owns its `positions3D` / `colors` /
  `radii` / `sharpness` / `scalars` typed-array buffers and exposes
  them via getter properties.
- `ensureCapacity(numElements)` grows the buffers (×1.5 geometric
  growth, ArrayBuffer.transfer-friendly) only when needed.
- `fill(...)` writes per-element values into the buffers and tracks
  a "used" length separate from buffer capacity.
- `isDisposed()` is checked at the top of every mutator. `fill` /
  `ensureCapacity` throw `Error('[Accumulator] post-dispose use')`
  rather than corrupting a disposed buffer.

### Worker fallback path

`projectPointsTo3D` (worker side) accepts an `outputBuffers` option;
when present, the worker writes directly into the caller's buffers
and returns them as transferable arrays. When absent (worker pool
unavailable / cold start), the loader allocates fresh arrays. Both
paths return the same `ProcessedXData` shape — callers can't tell
which path ran.

### Memory invariants

- Capacity grows ×1.5 (smooth amortized cost, low fragmentation).
- The accumulator is reset (length to 0, capacity preserved) at the
  start of every range-load cycle — buffers are reused across nD
  scrubs without freeing memory.
- Lazy allocation: `scalars` buffer is only allocated on the first
  colormap-mode commit (avoids 4 bytes/point overhead for non-
  colormap datasets).

## 4. Loader registry + monitor wiring

`loader-registry.ts` exposes a `LoaderRegistry` that the scene loader
populates and `data-monitor-manager` reads. Each loader registers
under a unique path key (`/group/node`) and exposes a `LoaderMonitor`
interface (`addEventListener`, `getMetrics`, `getActiveQueries`).

The monitor wiring is **structural** (port pattern) — the loader
implements the interface, the monitor consumes it, and neither
imports the concrete other. See
`src/data/scene-loader-monitor-port.ts` for the formal port
declaration.

## 5. Color attribute utilities

`color-attribute-utils.ts` handles the Uint8 / Uint16 / Float32 dtype
matrix for color attributes. Materials expect normalized Float32
input in some paths (WASM batches) and dtype-tagged input in others
(GPU buffer pool). The helpers ensure the right dtype reaches each
boundary without redundant copies.

## 6. Label loaders

`label-loader.ts` and `image-label-loader.ts` handle the optional
`labels` array under a node. Labels are sparse string lookups indexed
by element ID; the loaders use `BigUint64Array` offsets to support
datasets with > 2^32 elements without precision loss.

Label loading is on-demand (triggered by picking) rather than
upfront, and uses its own LRU to keep memory bounded.
