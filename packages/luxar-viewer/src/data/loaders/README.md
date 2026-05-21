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

### 2. range-loader.ts

Unified encoding dispatch:

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
- **direct**: No encoding, pass-through

### 3. spatial-query-builder.ts

Canonical chunk-bounds query API consumed by Points, Lines, and GSplats loaders.
The constructor takes a discriminated-union `SpatialQueryOptions` — pass either
`geometryType` (delegates tolerance to `tolerance-computer.computeTolerance`) or
a pre-computed `tolerance: number[]` (used by points, which has bespoke
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

Unified tolerance logic lives in `tolerance-computer.ts::computeTolerance`
and is selected by `geometryType`:

| Geometry  | Hidden spatial dim                                              | Hidden discrete dim          |
| --------- | --------------------------------------------------------------- | ---------------------------- |
| `points`  | `maxRadius` (or 0.5 if `spatialExtendDims[d]` is false)         | 0.5                          |
| `lines`   | 0 (segment bounds already include line width)                   | `step / 2` (or 0.5 fallback) |
| `gsplats` | `step × gsplatsDefaultTolerance` (default 3 σ; or 3.0 fallback) | 0.5                          |

Displayed dimensions always get `1e10` (effectively infinite).

## Worker Integration

All components automatically use workers when enabled. Always go through
`runWithTimeout()` — calling `getWorker()` directly bypasses the timeout
guard and the hung-worker eviction logic. See
`src/workers/README.md#runwithtimeout` for the full contract.

```typescript
// Config check happens automatically inside the loader.
if (appConfig.dataLoading.performance.useWebWorkers) {
  // The pool maps the TimeoutKind ('projection' / 'decode' / 'visibility')
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

### (Future) Unified Base Class

All loaders extend base class:

```typescript
class PointSpatialIndexLoader extends BaseSpatialLoader<PointsViewState, LoadedPointsData> {
  protected async projectTo3D(raw, viewState): Promise<LoadedPointsData> {
    // Type-specific projection
  }
}
```

## File Structure

```
src/data/loaders/
├── index.ts                      # Module exports (barrel)
├── README.md                     # This file
├── base-types.ts                 # Common type definitions (BaseViewState, LoadRange, BaseLoader, ...)
├── range-loader.ts               # Encoding dispatch (broadcasted/quantized/LUT/array_ref/direct)
├── spatial-query-builder.ts      # Canonical chunk-bounds AABB query + helpers
├── tolerance-computer.ts         # Geometry-aware per-dimension tolerance
├── transferable-accumulator.ts   # Zero-allocation + worker offload buffer pattern
├── chunk-bounds-loader.ts        # Shared chunk_bounds zarr probe (Points/Lines/GSplats)
├── color-attribute-utils.ts      # Shared color-range loader with native-dtype preservation
├── extend-to-all-preflight.ts    # Shared extend_to_all warning + one-time announce
├── image-label-loader.ts         # Lazy per-element image-label fetching (LRU blob URLs)
├── label-loader.ts               # Lazy CSR-style string-label fetching
├── overlay-loader.ts             # Reads overlay configurations from the zarr store
├── loader-metrics.ts             # Pure helpers for moving-average load metrics
├── monitor-events.ts             # LoaderEventEmitter — listener fan-out with error isolation
└── once-init.ts                  # One-shot async initializer with retry-on-failure
```

## Testing

```bash
# Run all tests
pnpm test src/tests/unit/data/loaders/

# Run specific tests
pnpm test src/tests/unit/data/loaders/spatial-query-builder.test.ts
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
- **color-attribute-utils.test.ts** — native-dtype allocation, direct vs
  encoded vs array_ref branches, `original_dtype` restoration with clamping.
- **extend-to-all-preflight.test.ts** — warning and one-time announce
  predicates, silence when `extendDims` is empty.
- **loader-metrics.test.ts** — `recordLoadEvent` rolling-mean math,
  `computeLoadLatency` undefined/zero start fallback.
- **monitor-events.test.ts** — add/remove idempotency, per-listener
  try/catch isolation, `clear()` on dispose.
- **once-init.test.ts** — concurrent callers share in-flight promise;
  rejected init clears the cache so the next call retries.

## Dependencies

- Internal: `data/zarr`, `data/array-decoder`, `workers/worker-pool`,
  `cache/lru-cache`, `utils/log`, `utils/clamp`, `config`,
  `types/dims`, `types/data-monitor-types`, `types/zarr`,
  `profiling/update-profiler`.
- External: `three`, `comlink` (via `workers/worker-pool`).

## See Also

- [../array-decoder/](../array-decoder/) — low-level encoding metadata and decoders consumed by `range-loader.ts`.
- [../../workers/README.md](../../workers/README.md) — worker pool API and the `runWithTimeout()` contract referenced above.

