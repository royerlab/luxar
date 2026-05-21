# Data accumulators

Per-geometry zero-allocation buffer pools for the Points/Lines/GSplats
loading hot path. The spatial-index loaders write directly into these
persistent typed-array buffers and hand zero-copy subarrays back to the
scene-loader commit phase — every nD scrub avoids per-update allocation
and the GC pauses that come with it.

## Files

| File          | Role                                                                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `types.ts`    | Shared `DataAccumulator<TData, TGetArgs, TFillArgs>` contract + `AccumulatorStats` shape. The three per-geometry files implement this interface.  |
| `points.ts`   | `LoadedPointsDataAccumulator` — positions / colors / radii / sharpness / scalars, with native Uint8/Uint16/Float32 color preservation.            |
| `lines.ts`    | `LinesDataAccumulator` — flat per-vertex buffers (positions, widths, colors, sharpness, scalars) + a separate `Uint32` segment-index buffer.      |
| `gsplats.ts`  | `GSplatsDataAccumulator` — centers, amplitudes, packed Cholesky factors (camelCase `choleskyFactors`), colors.                                    |

## The `DataAccumulator<T>` contract

All three implementations expose the same four methods (signatures
specialised by `TGetArgs` / `TFillArgs`):

| Method                  | Purpose                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `ensureCapacity(n)`     | Grow underlying buffers to at least `n` (1.5× growth). Returns `true` if a real allocation happened.                   |
| `getData(...counts)`    | Build a `Loaded{Points,Lines,GSplats}Data` payload of zero-copy `subarray()` views over the live prefix.               |
| `fill(...offsets, data)`| Write a `Partial<Loaded*Data>` chunk into the buffers at the given offset(s). First call pins per-attribute dtypes.    |
| `getStats()`            | `{ capacity, allocations, growthEvents, memoryMB }` — surfaced through the monitor port to the data-loading monitor.   |
| `dispose()`             | Drop buffers to zero-length sentinels and set the `_disposed` flag. Subsequent `fill`/`ensureCapacity` throw.          |

The variadic shape exists because Lines needs **two** counters
(vertices and segments grow independently), while Points and GSplats
need only an element count:

| Type    | `getData` signature                              | `fill` signature                                                |
| ------- | ------------------------------------------------ | --------------------------------------------------------------- |
| Points  | `getData(count)`                                 | `fill(offset, data)`                                            |
| Lines   | `getData(segmentCount, vertexCount)`             | `fill(segmentOffset, vertexOffset, data)`                       |
| GSplats | `getData(count)`                                 | `fill(offset, data)`                                            |

## Shared invariants

- **Type detection on first fill.** Each accumulator inspects the first
  `fill()` call to pin attribute dtypes (`Uint8Array` / `Uint16Array` /
  `Float32Array` for colors, `Uint8Array` / `Float32Array` for radii,
  sharpness, and scalars). Subsequent fills MUST use the same types.
  Growing buffers preserves the pinned dtype — no silent widening.
- **Live-prefix copy on growth.** `ensureCapacity` only copies the live
  prefix (`usedCount` / `usedVertexCount` / `usedSegmentCount`) into
  the new buffers, not the full previous capacity. On a fresh
  accumulator the difference can be a 5× reduction in copy cost.
- **Lazy scalar buffers.** Points and Lines start their scalar buffer
  at length 0 (`Float32Array(0)` sentinel). The first fill carrying
  `data.scalars` — or a direct `getScalarBuffer()` call from the
  spatial-index loader's zero-copy path — allocates it at the current
  capacity. Non-scalar datasets never pay the 4 B/element cost.
- **Optional attribute presence tracking.** `hasColors`, `hasRadii`,
  `hasSharpness`, `hasScalars` flip to `true` only when an actual
  fill writes the attribute. `getData()` returns `null`/`undefined` for
  attributes that were never filled, so commit code can skip uploading
  empty GPU buffers.
- **Disposed accumulators throw.** `assertNotDisposed(method)` raises a
  descriptive error if `fill`/`ensureCapacity`/`getData` is called
  after `dispose()`, surfacing caller-lifecycle bugs instead of
  silently operating on zero-length buffers.
- **GSplats camelCase Cholesky.** The accumulator exposes
  `choleskyFactors` (camelCase) — never `cholesky_factors`. Matches
  `LoadedGSplatsData` and the GPU buffer pool's attribute names.

## Direct buffer accessors

Each accumulator also exposes typed getters (`getPositionBuffer()`,
`getColorBuffer()`, `getVertexBuffer()`, `getSegmentBuffer()`,
`getCholeskyBuffer()`, ...) for the spatial-index loaders' direct-write
path. Loaders that bypass `fill()` to write into the raw buffers must
call `markScalarsLoaded()` (Lines) or arrange for `hasScalars` to be
set in some other way; otherwise `getData()` will return `undefined`
for that attribute.

## See also

- `../data-loader-types.ts` — `LoadedPointsData` shape consumed here
- `../../types/lines.ts`, `../../types/gsplats.ts` — Lines / GSplats
  `Loaded*Data` shapes
- `../loaders/README.md` — `RangeLoader` and `SpatialQueryBuilder` that
  drive these accumulators
- `../scene-loader/commit/commit-{points,lines,gsplats}-geometry.ts` —
  the commit-side consumers that upload the zero-copy subarrays to GPU
