# Unified Loader Architecture

**Status**: All geometry loaders use RangeLoader + TransferableAccumulator.

## Overview

This module provides the foundation for the unified spatial index loader architecture. It extracts common code patterns from Points, Lines, and GSplats loaders into reusable components.

### Problem Solved

The shared loader components keep Points, Lines, and GSplats aligned on:

- Encoding dispatch logic
- Spatial query construction
- Tolerance calculation
- Worker dispatch patterns

### Solution

Shared components that all loaders can use:

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED LOADERS                          │
│  (Points, Lines, GSplats extend common patterns)            │
│                          │                                  │
│          ┌───────────────┼───────────────┐                  │
│          ▼               ▼               ▼                  │
│   SpatialQueryBuilder  RangeLoader   Accumulator            │
│   (query logic)     (encoding)    (buffer reuse)            │
│          │               │               │                  │
│          └───────────────┼───────────────┘                  │
│                          ▼                                  │
│                    WorkerPool                               │
│              (CPU offload + WASM)                           │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. base-types.ts

Common type definitions:

```typescript
// View state shared by all loaders
interface BaseViewState {
  displayDims: number[];
  slicePosition: number[];
  tolerance: number[];
  dimensions?: DimensionMetadata[];
}

// Generic range for array loading
interface LoadRange {
  start: number;
  end: number;
}

// Loader interface
interface SpatialDataLoader<TViewState, TLoadedData> {
  loadData(viewState: TViewState, session?: UpdateSession): Promise<TLoadedData>;
  updateView(viewState: TViewState, session?: UpdateSession): Promise<TLoadedData>;
  dispose(): void;
}
```

### 2. spatial-query/range-loader.ts

Unified encoding dispatch. The orchestrator (`spatial-query/range-loader.ts`)
is a thin dispatcher; each encoding's body lives in a sibling helper
under `spatial-query/range-loader/` (`broadcasted.ts`, `quantized.ts`,
`perchannel.ts`, `lut.ts`, `direct.ts`, `array-ref.ts`, plus
`detect-encoding.ts`, `ref-resolution.ts`, `shared-instance.ts`, and the
shared `encoding-types.ts`).

```typescript
import { RangeLoader, getSharedRangeLoader } from './loaders';

// Get shared instance
const rangeLoader = getSharedRangeLoader();

// Load with automatic encoding detection
const elementsWritten = await rangeLoader.loadRanges(
  array, // zarr.Array
  attrs, // ArrayMetadata with encoding info
  ranges, // LoadRange[]
  outputBuffer, // Float32Array (pre-allocated)
  totalElements,
  elementsPerItem // 3 for positions, 1 for radii, etc.
);
```

Supported encodings:

- **broadcasted**: Single value replicated to all elements
- **quantized**: uint8/uint16 → float32 dequantization
- **lut**: Index-based lookup table decoding
- **array_ref**: Reference to another array (resolved at init)
- **perchannel**: Per-column (`col_lo`/`col_hi`) log / signed-log / linear
  dequantization of `*_perchannel_*` uint8/uint16 arrays → float32
- **direct**: No encoding, pass-through

### 3. spatial-query/spatial-query-builder.ts

Canonical chunk-bounds query API consumed by Points, Lines, and GSplats loaders.
The constructor takes a discriminated-union `SpatialQueryOptions` — pass either
`geometryType` (delegates tolerance to `spatial-query/tolerance-computer.computeTolerance`)
or a pre-computed `tolerance: number[]` (used by points, which has bespoke
`EffectiveRadiusConfig` semantics).

Geometry-aware path (gsplats / lines):

```typescript
import { SpatialQueryBuilder } from './loaders';

const ranges = await new SpatialQueryBuilder(chunkIndex, viewState, {
  geometryType: 'gsplats',
  totalElements: attrs.n_splats,
  chunkSize: attrs.chunk_size,
  extendDims: attrs.extend_to_all,
  toleranceOptions: { gsplatsDefaultTolerance: 3.0 }, // optional tuning
}).execute();
```

Pre-computed tolerance path (points):

```typescript
const tolerance = calculateSpatialQueryTolerance(viewState, config, ndim);

const ranges = await new SpatialQueryBuilder(chunkIndex, viewState, {
  tolerance,
  totalElements: attrs.n_points,
  chunkSize: attrs.chunk_size,
  extendDims: attrs.extend_to_all,
}).execute();
```

Lower-level functional helpers (used internally by the builder, exported for
direct use when needed):

```typescript
import {
  buildQueryPosition,
  executeSpatialQuery,
  chunkIndicesToRanges,
  mergeRanges,
} from './loaders';

const position = buildQueryPosition(viewState, ndim);
const chunkIndices = executeSpatialQuery({
  chunkBounds,
  queryPosition: position,
  queryTolerance: tolerance,
  numChunks,
  ndim,
});
const ranges = mergeRanges(chunkIndicesToRanges(chunkIndices, chunkSize, totalElements));
```

## Tolerance calculation

Unified tolerance logic lives in `spatial-query/tolerance-computer.ts::computeTolerance`
and is selected by `geometryType`. Hidden DISCRETE dims distinguish two roles:
the **query** role (chunk-fetch reach — the default) uses the shared
quarter-cell `0.25 × step`, while the **membership** role
(`options.discreteRole: 'membership'`, the per-element visibility slab used by
the lines projection-clipping path) uses the half-cell `0.5 × step`, matching
the points/gsplats projection gates:

| Geometry  | Hidden spatial dim                                              | Hidden discrete dim (query)   | Discrete membership gate        |
| --------- | --------------------------------------------------------------- | ----------------------------- | ------------------------------- |
| `points`  | `maxRadius` (discrete rule if `spatialExtendDims[d]` false)     | `0.25 × step` (0.25 fallback) | 0.5 absolute (projection stage) |
| `lines`   | 0 (segment bounds already include line width)                   | `0.25 × step` (0.25 fallback) | `0.5 × step` via `discreteRole` |
| `gsplats` | `step × gsplatsDefaultTolerance` (default 3 σ; or 3.0 fallback) | `0.25 × step` (0.25 fallback) | `step × 0.5` (projection stage) |

The quarter-cell query reach sits deliberately below the half-cell membership
gates: chunk bounds are epsilon-padded on the write side
(`packages/luxar/src/luxar/io/ordering.py`),
and pad + reach must stay under one step or a single-category query bleeds in
the whole neighbouring category.

Displayed dimensions always get `1e10` (effectively infinite).

## Worker Integration

All components automatically use workers when enabled. Always go through
`runWithTimeout()` — calling `getWorker()` directly bypasses the timeout
guard and the hung-worker eviction logic. See
`src/workers/README.md#runwithtimeout` for the full contract.

```typescript
// Config check happens automatically inside the loader.
if (appConfig.dataLoading.performance.useWebWorkers) {
  // The pool maps the TimeoutKind ('projection' / 'decode')
  // to the corresponding config knob and evicts the worker on timeout.
  result = await getWorkerPool().runWithTimeout(
    'someMethod',
    'projection',
    (api) => api.someMethod(...)
  );
} else {
  // Main thread fallback
  result = mainThreadImplementation(...);
}
```

`runWithTimeout()` already rejects with a `WorkerTimeoutError` when the
operation exceeds the configured budget; catch it (or the more general
`Error`) at the loader boundary if you want a main-thread fallback:

```typescript
try {
  result = await getWorkerPool().runWithTimeout(
    'someMethod',
    'projection',
    (api) => api.someMethod(...)
  );
} catch (error) {
  log.warning('Worker failed, falling back to main thread');
  result = mainThreadImplementation(...);
}
```

## Range Loading

All three spatial index loaders use `RangeLoader`:

- `point-spatial-index-loader.ts` uses RangeLoader for all encoding types.
- `lines-spatial-index-loader.ts` uses RangeLoader for vertex attribute loading.
- `gsplats-spatial-index-loader.ts` uses RangeLoader for gsplat array loading.

```typescript
import { RangeLoader } from './loaders';

private rangeLoader = new RangeLoader(this.refRegistry);

await this.rangeLoader.loadRanges(array, attrs, ranges, output, total);
```

**Key behavior:**

1. RangeLoader outputs Float32Array for encoded arrays (broadcasted, quantized, lut)
2. Direct (unencoded) arrays preserve native type in points loader (critical for rendering!)
3. Lines and GSplats always use Float32Array (per their type definitions)
4. array_ref encoding handled specially (needs zarrStore access)
5. Worker dispatch integrated with main thread fallback

**Type Preservation (Points Only):**
The points loader preserves native types because the rendering pipeline depends
on actual array types:

- `Uint8Array` colors: THREE.js normalizes (0-255 → 0-1) with `normalized=true`
- `Float32Array` colors: Expected to be 0-1, no normalization

Converting `Uint8Array(255)` to `Float32Array(255.0)` would make colors ~255x too bright!

**Two cases handled:**

1. **Direct (unencoded) arrays**: Native type preserved during loading
2. **Encoded arrays with `original_dtype`**: Decoded to float, then converted back
   to original_dtype (e.g., uint8 colors restored as Uint8Array)
3. **Array references with `original_dtype`**: Same as encoded arrays

**Known Limitation - Lines & GSplats:**
Lines and GSplats loaders always output Float32Array for colors. While Python's
`ColorArray` type supports both uint8 and float32, the TypeScript types for
`LoadedLinesData.colors` and `LoadedGSplatsData.colors` only allow Float32Array.
This is a design decision - updating would require type changes across the codebase.

### TransferableAccumulator

Enable zero-allocation + CPU offload:

```typescript
import { TransferableAccumulator, createPointsAccumulator, type PointsBuffers } from './loaders';

// Create accumulator once (owns reusable buffers)
const accumulator = createPointsAccumulator(10000);

// For each update:
// 1. Detach buffers for transfer to worker
const buffers = accumulator.detach();
const transferables = accumulator.getTransferables(buffers);

// 2. Transfer to worker (zero-copy via Comlink)
const result = await worker.projectPointsTo3D(
  Comlink.transfer({ params, outputBuffers: buffers }, transferables)
);

// 3. Adopt returned buffers (zero-copy)
accumulator.adopt(result.outputBuffers);
```

**Key Benefits**:

- Zero-allocation in steady state (after initial warmup)
- Zero-copy buffer transfer via `Comlink.transfer()`
- Enables BOTH accumulator pattern AND worker CPU offload
- Buffers cycle between main thread and worker without copying

### Other shared helpers

Beyond the three top-level abstractions above, this folder also holds the
narrowly-scoped helpers each spatial-index loader composes:

- **`chunk-bounds-loader.ts`** —
  `fetchChunkBoundsArray(location, arrayName, logModule, notFoundMessage)`:
  open a `chunk_bounds` / `vertex_chunk_bounds` / `segment_chunk_bounds` zarr
  array and return `{ data: Float32Array, shape }`.
  Soft-falls-back to `null` on 404 / Not Found (datasets without
  spatial ordering legitimately omit the array) and on corrupt-zarr / network
  errors after a warning.
- **`color-loader.ts`** — shared color-range loader with native-dtype
  preservation. Exports `loadColorRanges()` (end-to-end load with direct /
  `rgb_uint8` shortcut / encoded-then-restored branches), `allocateColorBuffer`,
  `restoreOriginalDtype`, `getExpectedColorType`, `colorBufferTypeMatches`, and
  the `ColorBuffer` / `ColorBufferKind` / `ColorRange` types. Direct (unencoded)
  reads are delegated to `RangeLoader.loadDirectTyped` — the single
  dtype-preserving reader — so `color-loader.ts` keeps only the color-specific
  concerns (RGB layout, `original_dtype` restoration).
- **`spatial-query/prefetch-ranges.ts`** — `prefetchRangesIntoCache(arrays, ranges)`:
  shared cache-warming read for the three loaders' `prefetchChunks`.
  Fires a `get()` per (array × range) and discards the result. Deliberately
  separate from `RangeLoader.loadDirectTyped` — prefetch warms future frames,
  so it allocates no typed output and carries no per-update abort signal.
- **`extend-to-all-preflight.ts`** — `warnExtendToAllNoDimensions` (warns when
  `extend_to_all` is set but the view state has no resolved dimensions) +
  `announceExtendToAllOnce` (one-shot BROADCAST emoji log on first load).
  Splitting into two functions matches the existing call structure of the
  loaders (warning → chunkIndex early-return → broadcast).
- **`picking/image-label-loader.ts`** — `ImageLabelLoader`: lazy per-element
  image fetching from `image_label_offsets` + `image_label_bytes` arrays.
  Bulk-loads the offsets table (small), then fetches each image's byte range
  on demand. Decoded blobs are cached as blob URLs in a 50 MB-default LRU;
  eviction revokes the URL. Detects JPEG / PNG / WebP from magic bytes.
- **`picking/label-loader.ts`** — `LabelLoader`: lazy CSR-style string-label
  fetching from `label_offsets` + `label_bytes`. Bulk-loads the whole node's
  labels on first hover; concurrent requests for the same node share one
  in-flight promise.
- **`overlays/overlay-loader.ts`** — `loadOverlayConfigs(store, rootLoc)`:
  enumerates the `overlays/` group and parses each child's `.zattrs` into an
  `OverlayConfig` (text / image / html, with per-type fields). Results are
  z-index-sorted; missing `overlays/` group returns `[]` silently.
- **`loader-metrics.ts`** — `recordLoadEvent(counters, elements, bytes, loadTime)`
  (rolling-mean update of `loads` / `elementsLoaded` /
  `bytesLoaded` / `avgLoadTime`), `computeLoadLatency(startMs, nowMs?)`
  (latency, 0 when start is undefined / 0), and
  `finishQueryTracking(activeQueries, metrics, queryId, startTime, status)`
  (query close-out: stamps `status`/`endTime` on the tracked `QueryInfo`,
  drops it from the map, folds the elapsed time into the rolling
  `avgQueryTime` — called by the facades' `loadX` wrappers on BOTH the
  success and error paths so the active-query map never leaks), plus
  `makeInitialLoaderMetrics(type, path)` (the zeroed initial `LoaderMetrics`
  record every facade starts from), and
  `buildSpatialIndexMetrics(chunkCount, queries, totalQueryCells, elementsLoaded)`
  (the chunk-index
  telemetry snapshot all three facades attach as `metrics.spatialIndex` for
  the monitor advisor). Pure helpers, unit-tested without a zarr
  store, used by all three geometry facades. `elementsLoaded` is the
  geometry-neutral throughput counter (points / vertices / splats).
- **`spatial-facade.ts`** — shared facade-level orchestration for the three
  spatial-index loaders, driven by one per-loader `SpatialFacadeCtx` (stable
  references + `this`-bound accessors, built once in each constructor):
  `loadSliceWithCache(ctx, viewState, loadInternal)` (the `loadX` template —
  S-cache restore → internal load → query close-out → S-cache store, with the
  abort-aware error branch), `recordLoadMetrics(ctx, arrayName, elements, output)`
  (per-array load metrics + 'load' event), and
  `runWithActiveSignal` / `runWithResidencyProbe` (the `updateView` /
  `updateViewWithResidency` bodies: per-update abort-signal publication and
  cache-residency probing). Each used to exist as three byte-identical
  private methods.
- **`monitor-events.ts`** — `LoaderEventEmitter`: owns the listener `Set` for
  a `LoaderMonitor` implementation. Per-listener try/catch isolates one bad
  listener from the rest; `clear()` is called on dispose.
- **`once-init.ts`** — `OnceInit.ensure(initFn)`: concurrent callers await a
  shared in-flight promise; a rejected init clears the cache so the next call
  can retry from scratch. Centralizes the pattern previously duplicated four
  times across the three loaders.
- **`aggregate-loader-metrics.ts`** — `aggregateLoaderMetrics(inner, path)`:
  pure roll-up of N per-LOD `LoaderMetrics` into one snapshot for a progressive
  node. Counters are summed; `avgQueryTime` / `avgLoadTime` are query/load-weighted
  means; optional `spatialIndex` cell counts (`occupiedCells` / `totalCells`) are
  summed, its per-query rates are query-weighted means, and `avgElementsPerCell`
  is cell-weighted (a per-cell density); `optimization` is taken from
  the first reporter to avoid double-counting app-global singletons.
- **`progressive-monitor-adapter.ts`** — `ProgressiveMonitorAdapter`: makes a
  progressive node (N inner per-LOD loaders) look like a SINGLE loader to the
  data monitor. Re-stamps every inner monitor event / active query with the
  parent node path so per-`additive_<i>` sub-paths never reach the monitor
  (which would otherwise double-count throughput/memory). `getMetrics()`
  delegates to `aggregateLoaderMetrics`; `addEventListener` is idempotent.

## File Structure

External callers should import from the barrel (`from '../loaders'`) rather
than reach into subpackages. The internal layout is:

```
src/data/loaders/
├── index.ts                      # Module exports (barrel — only public surface)
├── README.md                     # This file
├── base-types.ts                 # Common type definitions (BaseViewState, LoadRange, BaseLoader, ...)
├── abort-error.ts                # isAbortError — realm-proof "superseded, not failed" classifier
├── chunk-bounds-loader.ts        # Shared chunk_bounds zarr probe (Points/Lines/GSplats)
├── color-loader.ts               # Shared color-range loader with native-dtype preservation
├── transferable-accumulator.ts   # Zero-allocation + worker offload buffer pattern
├── loader-metrics.ts             # Pure helpers for load/query metric bookkeeping
├── spatial-facade.ts             # Shared loadX/updateView/metrics facade orchestration
├── monitor-events.ts             # LoaderEventEmitter — listener fan-out with error isolation
├── once-init.ts                  # One-shot async initializer with retry-on-failure
├── extend-to-all-preflight.ts    # Shared extend_to_all warning + one-time announce
├── aggregate-loader-metrics.ts   # Pure roll-up of per-LOD metrics into one snapshot
├── progressive-monitor-adapter.ts # Re-paths inner-LOD monitor events to the parent node
│
├── spatial-query/                # Chunk-bounds → tolerance → AABB scan → range fetch
│   ├── spatial-query-builder.ts  # Canonical chunk-bounds AABB query + helpers
│   ├── tolerance-computer.ts     # Geometry-aware per-dimension tolerance
│   ├── prefetch-ranges.ts        # prefetchRangesIntoCache — shared cache-warming read
│   ├── range-loader.ts           # Encoding-dispatch orchestrator (thin dispatcher)
│   └── range-loader/             # Per-encoding helper bodies (private to range-loader.ts)
│       ├── encoding-types.ts     # EncodingType, RangeLoaderConfig, shared helpers
│       ├── detect-encoding.ts    # Encoding detection from ArrayMetadata
│       ├── broadcasted.ts        # Single value → all elements
│       ├── quantized.ts          # uint8/uint16 → float32 dequantization
│       ├── perchannel.ts         # Per-column (col_lo/col_hi) dequant → float32
│       ├── lut.ts                # Lookup-table decoding
│       ├── direct.ts             # Unencoded pass-through
│       ├── array-ref.ts          # Array ref diagnostic (should be pre-resolved)
│       ├── ref-resolution.ts     # Open target array on array_ref
│       └── shared-instance.ts    # getSharedRangeLoader singleton
│
├── picking/                      # Label loaders consumed by core/app/picking
│   ├── label-loader.ts           # Lazy CSR-style string-label fetching
│   └── image-label-loader.ts     # Lazy per-element image-label fetching (LRU blob URLs)
│
├── progressive/                  # Shared helpers for additive-LOD progressive loaders
│   ├── concat-helpers.ts         # Generic typed-array field concatenation across LOD parts
│   ├── slice-cache-helper.ts     # Shared SliceCache key/snapshot/lookup helpers (S-cache)
│   └── constants.ts              # CACHE_HIT_THRESHOLD_MS — shared streaming threshold
│
└── overlays/                     # Overlay metadata loader
    └── overlay-loader.ts         # Reads overlay configurations from the zarr store
```

## Testing

```bash
# Run all tests
pnpm test src/tests/unit/data/loaders/

# Run specific tests
pnpm test src/tests/unit/data/loaders/spatial-query/spatial-query-builder.test.ts
pnpm test src/tests/unit/data/loaders/transferable-accumulator.test.ts
```

### Test coverage

- **spatial-query-builder.test.ts**
  - `buildQueryPosition`: position-array padding/truncation
  - `chunkIndicesToRanges`: chunk → range conversion
  - `mergeRanges`: range coalescing
  - `executeSpatialQuery`: AABB scan
  - `shouldExtendVisibility`, `createLoadAllRange`: extend_to_all helpers
  - `SpatialQueryBuilder` geometry-aware path (delegates to `computeTolerance`)
  - `SpatialQueryBuilder` pre-computed-tolerance path (used by points)
  - `SpatialQueryBuilder` extend-to-all short-circuit and `chunkSize` fallback

- **transferable-accumulator.test.ts**
  - Basic operations, capacity management
  - Detach/adopt cycle for worker transfer
  - Optional buffer enabling
  - Points, Lines, GSplats factory functions
  - Memory tracking statistics

- **tolerance-computer.test.ts**
  - `computeTolerance` for points / lines / gsplats with displayed/hidden,
    discrete/spatial, with/without step, and option overrides

- **base-types.test.ts** — `hasDimensionMetadata`, `getDisplayDimCount`,
  `isHiddenDimension` type guards.
- **chunk-bounds-loader.test.ts** — `fetchChunkBoundsArray` happy-path,
  404 soft-fallback, and corrupt-zarr warning behavior.
- **color-loader.test.ts** — native-dtype allocation, direct vs encoded vs
  array_ref branches, `original_dtype` restoration with clamping.
- **extend-to-all-preflight.test.ts** — warning and one-time announce
  predicates, silence when `extendDims` is empty.
- **loader-metrics.test.ts** — `recordLoadEvent` rolling-mean math,
  `computeLoadLatency` undefined/zero start fallback, `finishQueryTracking`
  close-out (complete/error stamping, map removal, rolling `avgQueryTime`,
  `queries === 0` guard).
- **monitor-events.test.ts** — add/remove idempotency, per-listener
  try/catch isolation, `clear()` on dispose.
- **once-init.test.ts** — concurrent callers share in-flight promise;
  rejected init clears the cache so the next call retries.
- **aggregate-loader-metrics.test.ts** — empty-array zeroed fallback,
  counter summing, query/load-weighted mean times (+ zero-when-no-queries/loads),
  `spatialIndex` roll-up (summed cell counts, query-weighted per-query rates,
  cell-weighted `avgElementsPerCell`), first-reporter `optimization`.
- **progressive-monitor-adapter.test.ts** — event/active-query re-pathing to
  the parent node, `addEventListener` idempotency, `getMetrics` aggregation.
- **concat-helpers.test.ts** (under `progressive/`) — required/optional
  typed-array concatenation, dtype preservation, all-or-nothing optional gate.

## Dependencies

- Internal: `data/zarr`, `data/array-decoder`, `workers/worker-pool`,
  `cache/lru-cache`, `utils/log`, `utils/clamp`, `config`,
  `types/dims`, `types/data-monitor-types`, `types/zarr`,
  `profiling/update-profiler`.
- External: `three`, `comlink` (via `workers/worker-pool`).

## See Also

- [../array-decoder/](../array-decoder/) — low-level encoding metadata and decoders consumed by `spatial-query/range-loader.ts`.
- [../../workers/README.md](../../workers/README.md) — worker pool API and the `runWithTimeout()` contract referenced above.
- [progressive/README.md](progressive/README.md) — shared concat/constants helpers for the additive-LOD progressive loaders that the metrics-aggregation and monitor-adapter helpers above also serve.
