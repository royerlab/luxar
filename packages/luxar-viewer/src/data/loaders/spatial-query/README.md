# Spatial Query

The chunk-bounds query path shared by the Points, Lines, and GSplats
spatial-index loaders: turn a view state into the minimal set of array ranges
that must be fetched, then load those ranges regardless of how the Python
`luxar.encoding` layer stored them.

## Overview

Each geometry loader probes its own `chunk_bounds` array (loader-private; the
names differ — `chunk_bounds` for points/gsplats, `vertex_chunk_bounds` +
`segment_chunk_bounds` for lines), then hands the resulting
[`ChunkSpatialIndex`](./spatial-query-builder.ts) to this folder for the
common work:

1. **Tolerance** — `tolerance-computer.ts` turns the view state into a
   per-dimension tolerance using geometry-aware semantics.
2. **Query** — `spatial-query-builder.ts` builds the query box, runs an AABB
   scan over the chunk bounds, converts matching chunks to ranges, and
   coalesces them.
3. **Fetch** — `range-loader.ts` loads each range from zarr, dispatching on
   the array's encoding (direct / quantized / LUT / broadcasted / array_ref /
   perchannel).

The query side runs on the **main thread** — an AABB scan is
O(numChunks × ndim) and finishes in microseconds, so worker dispatch would
only add structured-clone overhead. The fetch side offloads heavy decodes to
the worker pool (see `range-loader/`).

## File Structure

```
spatial-query/
├── spatial-query-builder.ts   # SpatialQueryBuilder + AABB scan / range helpers
├── tolerance-computer.ts      # Geometry-aware per-dimension tolerance
├── range-loader.ts            # Encoding-dispatch range loader (entry point)
└── range-loader/              # Per-encoding loader bodies (see its README)
    ├── detect-encoding.ts
    ├── direct.ts  quantized.ts  perchannel.ts  lut.ts  broadcasted.ts
    ├── array-ref.ts  ref-resolution.ts
    ├── encoding-types.ts  shared-instance.ts
    └── README.md
```

## SpatialQueryBuilder

`SpatialQueryBuilder` runs one chunk-bounds query end-to-end. The tolerance
source is selected at construction via a discriminated union — supply **either**
a `geometryType` (tolerance delegated to `computeTolerance`) **or** a
pre-computed `tolerance: number[]` (used by the points loader, which already
knows about `EffectiveRadiusConfig`). The points pre-computed path
(`calculateSpatialQueryTolerance` in `effective-radius-calculator.ts`) applies
the SAME shared `discreteDimTolerance` quarter-cell rule for discrete dims, so
both paths agree at runtime.

```typescript
import { SpatialQueryBuilder, type ChunkSpatialIndex } from './spatial-query-builder';

// geometry-aware path (gsplats / lines)
const ranges = await new SpatialQueryBuilder(index, viewState, {
  geometryType: 'gsplats',
  totalElements: attrs.n_splats,
  chunkSize: attrs.chunk_size,
  extendDims: attrs.extend_to_all,
  logModule: Modules.GSPLATS_SPATIAL_INDEX_LOADER,
}).execute();

// pre-computed tolerance path (points)
const tolerance = calculateSpatialQueryTolerance(viewState, config, ndim);
const ranges = await new SpatialQueryBuilder(index, viewState, {
  tolerance,
  totalElements: attrs.n_points,
  chunkSize: attrs.chunk_size,
}).execute();
```

`execute()` returns merged `LoadRange[]` for the visible elements, or a single
load-all range when an `extend_to_all` dimension is currently hidden (see
`shouldExtendVisibility`).

### Canonical index shape

All three geometry types produce indices of this shape — a flattened
`(numChunks, ndim, 2)` `[min, max]` array plus the chunk count and
dimensionality. For lines, only the **segment** side feeds the chunk query;
the vertex side is loaded via a separate sorted-indices path.

```typescript
interface ChunkSpatialIndex {
  chunkBounds: Float32Array; // (numChunks, ndim, 2) flattened row-major
  chunkCount: number;
  metadata: { ndim: number; chunk_size?: number };
}
```

### Standalone helpers

The builder is a thin shell over pure functions, all exported for callers that
don't need the full builder (e.g. `scene-loader`'s lines clipping path):

| Function                                          | Description                                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `buildQueryPosition(viewState, ndim)`             | Pad/truncate `slicePosition` to `ndim`.                                                        |
| `executeSpatialQuery(params, logModule?)`         | AABB scan; returns matching chunk indices. Early-exits on the first non-overlapping dimension. |
| `chunkIndicesToRanges(indices, chunkSize, total)` | Map chunk indices to `LoadRange[]`, clipped at `total`.                                        |
| `mergeRanges(ranges)`                             | Coalesce overlapping/adjacent ranges (sorted by `start`).                                      |
| `shouldExtendVisibility(extendDims, viewState)`   | True iff an `extend_to_all` dim is currently hidden.                                           |
| `createLoadAllRange(total)`                       | Single range covering the whole dataset.                                                       |

## Tolerance computer

`computeTolerance(geometryType, displayDims, ndim, dimensions?, options?)`
returns a length-`ndim` tolerance array. Displayed dimensions always get an
infinite sentinel (`1e10`). Hidden **discrete** dims share ONE rule across all
three geometries, split by ROLE: the default **query** role
(`discreteDimTolerance`) is a quarter-cell `0.25 × step` (fallback `0.25`) —
deliberately `< 0.5 × step` so write-side chunk-bound padding plus tolerance
can never sum to a full step and bleed the neighbouring category (the barrier
over-fetch fix) — while the **membership** role
(`options.discreteRole: 'membership'`, used by the lines projection-clipping
path via `discreteDimMembershipTolerance`) is the half-cell `0.5 × step`
matching the points/gsplats projection visibility gates. Hidden **spatial/continuous** dims stay
geometry-specific:

| Geometry  | Hidden spatial dim                                                            | Hidden discrete dim             |
| --------- | ----------------------------------------------------------------------------- | ------------------------------- |
| `points`  | `maxRadius` (so points whose radius intersects the slice load)                | `0.25 × step` (fallback `0.25`) |
| `lines`   | `0` (segment bounds already include line-width extent)                        | `0.25 × step` (fallback `0.25`) |
| `gsplats` | `step × gsplatsDefaultTolerance` (default 3σ), else `gsplatsDefaultTolerance` | `0.25 × step` (fallback `0.25`) |

Points decide "spatial vs discrete" from `options.spatialExtendDims` (the
per-dimension flag array carried by `EffectiveRadiusConfig`) rather than the
`discrete` flag; lines and gsplats read `DimensionInfo.discrete` / `step`.
Called from `SpatialQueryBuilder`'s geometry-aware path and directly from
`scene-loader.ts` for lines projection clipping.

## RangeLoader

`RangeLoader` is the single entry point for range-based array loading shared by
all three spatial-index loaders. It detects the encoding from the array's
metadata and dispatches to the matching `load*` body under `range-loader/`,
writing the decoded result into a caller-supplied `Float32Array`.

```typescript
import { getSharedRangeLoader } from './range-loader/shared-instance';

const loader = getSharedRangeLoader();
const written = await loader.loadRangesResolvingRef(
  array,
  attrs,
  ranges,
  output,
  totalElements,
  elementsPerItem,
  zarrStore,
  logPrefix
);
```

- `loadRanges(...)` — dispatch on encoding (`broadcasted` / `quantized` /
  `lut` / `array_ref` / `perchannel` / `direct`, defaulting to `direct`).
- `loadRangesResolvingRef(...)` — same, but first resolves `array_ref`
  encodings (opens the target array via `resolveArrayRef`) and delegates
  against the resolved array. The standard entry point for loaders.
- `static detectEncoding(attrs)` — expose the encoding classifier.
- `getDecoder()` — the underlying `ArrayDecoder` for full-array operations.

Per-encoding semantics, the worker-offload threshold, the `array_ref`
resolution rules, and the process-wide shared singletons all live in
[`range-loader/README.md`](./range-loader/README.md).

## See Also

- [`range-loader/`](./range-loader/README.md) — per-encoding loader bodies and
  the shared `RangeLoader` / `ArrayRefRegistry` singletons.
- [`../README.md`](../README.md) — unified loader infrastructure overview.
- [`../chunk-bounds-loader.ts`](../chunk-bounds-loader.ts) — shared `chunk_bounds`
  zarr probe (the layer below this folder).
- [`../../array-decoder/`](../../array-decoder/) — `ArrayDecoder` /
  `ArrayRefRegistry`; the main-thread decode fallback and metadata helpers.
