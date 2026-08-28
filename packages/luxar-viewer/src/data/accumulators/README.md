# Data accumulators

Per-geometry zero-allocation buffer pools for the Points/Lines/GSplats
loading hot path. The spatial-index loaders write directly into these
persistent typed-array buffers and hand zero-copy subarrays back to the
scene-loader commit phase — every nD scrub avoids per-update allocation
and the GC pauses that come with it.

## Files

| File         | Role                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `types.ts`   | Shared `DataAccumulator<TData, TGetArgs, TFillArgs>` contract + `AccumulatorStats` shape. The three per-geometry files implement this interface.                                                  |
| `growth.ts`  | `nextCapacity(current, needed)` — the shared 1.5× growth policy, clamped so a single large jump lands exactly on the count the loader already knows rather than on a term of the growth sequence. |
| `points.ts`  | `LoadedPointsDataAccumulator` — positions / colors / radii / sharpness / scalars, with native Uint8/Uint16/Float32 color preservation.                                                            |
| `lines.ts`   | `LinesDataAccumulator` — flat per-vertex buffers (positions, widths, colors, sharpness, scalars) + a separate `Uint32` segment-index buffer.                                                      |
| `gsplats.ts` | `GSplatsDataAccumulator` — centers, amplitudes, packed Cholesky factors (camelCase `choleskyFactors`), colors.                                                                                    |

## The `DataAccumulator<T>` contract

All three implementations expose the same five methods (signatures
specialised by `TGetArgs` / `TFillArgs`):

| Method                   | Purpose                                                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `ensureCapacity(n)`      | Grow underlying buffers to at least `n` (1.5× growth, never past `n` — see `growth.ts`). Returns `true` if a real allocation happened. |
| `getData(...counts)`     | Build a `Loaded{Points,Lines,GSplats}Data` payload of zero-copy `subarray()` views over the live prefix.                               |
| `fill(...offsets, data)` | Write a `Partial<Loaded*Data>` chunk into the buffers at the given offset(s). First call pins per-attribute dtypes.                    |
| `getStats()`             | `{ capacity, allocations, growthEvents, memoryMB }` — surfaced through the monitor port to the data-loading monitor.                   |
| `dispose()`              | Drop buffers to zero-length sentinels and set the `_disposed` flag. Subsequent `fill`/`ensureCapacity` throw.                          |

The variadic shape exists because Lines needs **two** counters
(vertices and segments grow independently), while Points and GSplats
need only an element count:

| Type    | `getData` signature                  | `fill` signature                          |
| ------- | ------------------------------------ | ----------------------------------------- |
| Points  | `getData(count)`                     | `fill(offset, data)`                      |
| Lines   | `getData(segmentCount, vertexCount)` | `fill(segmentOffset, vertexOffset, data)` |
| GSplats | `getData(count)`                     | `fill(offset, data)`                      |

## Shared invariants

- **Type detection on first fill.** Each accumulator inspects the first
  `fill()` call to pin attribute dtypes (`Uint8Array` / `Uint16Array` /
  `Float32Array` for colors; `Uint8Array` / `Float32Array` for radii and
  sharpness; `Uint8Array` / `Float16Array` / `Float32Array` tracked for
  scalars, with Float16 widened to a Float32 buffer at fill time via the
  numeric `TypedArray.set()` conversion). Subsequent fills MUST use the
  same types. Growing buffers preserves the pinned dtype — no silent
  widening. Lines and GSplats only enter `initializeTypes` when the fill
  carries colors (or, for Lines, colors or scalars); Points enters on
  every fill but the call is a no-op once types are pinned.
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

Each accumulator also exposes typed getters for the spatial-index
loaders' direct-write path:

- Points: `getPositionBuffer()`, `getColorBuffer()`, `getRadiiBuffer()`,
  `getSharpnessBuffer()`, `getScalarBuffer()`. `getScalarBuffer()`
  lazy-allocates the scalar buffer at the current capacity on first
  call (the sentinel-empty path).
- Lines: `getVertexBuffer()`, `getSegmentBuffer()`, `getWidthBuffer()`,
  `getColorBuffer()`, `getSharpnessBuffer()`, `getScalarBuffer()`
  (same lazy allocation as Points).
- GSplats: `getCenterBuffer()`, `getAmplitudeBuffer()`,
  `getCholeskyBuffer()`, `getColorBuffer()`. GSplats additionally
  exposes `setColorBuffer()` so the loader can swap in a
  pre-allocated/quantised color buffer without going through `fill()`.

Loaders that bypass `fill()` to write into the raw buffers must arrange
for the corresponding `has*` flag to be set. Lines exposes
`markScalarsLoaded()` for the zero-copy scalar path; for other
attributes the loader is expected to perform at least one
`fill()`-style call so the presence flag flips. Otherwise `getData()`
will return `null`/`undefined` for that attribute even though the
buffer holds valid data.

Introspection helpers — `Points.hasTypes()`, and
`{Points,Lines,GSplats}.isDisposed()` — let tests and the data-loading
monitor inspect accumulator state without triggering allocations. The
Points accumulator also exposes `updateMetadata({ ndim, totalPoints,
usedSpatialIndex, ... })` for late metadata adjustments; the `bounds`
field is intentionally ignored because `getData()` recomputes bounds
from the live positions on every call.

## See also

- `../data-loader-types.ts` — `LoadedPointsData` shape consumed here
- `../../types/lines.ts`, `../../types/gsplats.ts` — Lines / GSplats
  `Loaded*Data` shapes
- `../loaders/README.md` — `RangeLoader` and `SpatialQueryBuilder` that
  drive these accumulators
- `../scene-loader/commit/commit-{points,lines,gsplats}-geometry.ts` —
  the commit-side consumers that upload the zero-copy subarrays to GPU
