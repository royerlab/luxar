# Unified Loader Architecture

**Status**: PHASE 1 & 2 COMPLETE - All loaders migrated to RangeLoader + TransferableAccumulator
**Date**: 2025-12-27

## Overview

This module provides the foundation for the unified spatial index loader architecture. It extracts common code patterns from Points, Lines, and GSplats loaders into reusable components.

### Problem Solved

Before unification, each loader (Points, Lines, GSplats) had:

- Duplicated encoding dispatch logic (~600 lines total)
- Duplicated spatial query logic (~300 lines total)
- Inconsistent tolerance calculation
- Inconsistent worker dispatch patterns

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

Fluent API for spatial queries:

```typescript
import { SpatialQueryBuilder } from './loaders';

// Build and execute query
const builder = new SpatialQueryBuilder(chunkIndex, viewState, totalElements, chunkSize);

const ranges = await builder
  .withExtendToAll(['time']) // extend_to_all support
  .withMaxRadius(5.0) // tolerance fallback
  .withStepMultiplier(1.5) // tolerance = step * 1.5
  .execute();
```

Lower-level functions:

```typescript
import {
  computeQueryTolerance,
  buildQueryPosition,
  executeSpatialQuery,
  mergeRanges,
} from './loaders';

// Build tolerance array
const tolerance = computeQueryTolerance(viewState, ndim, {
  defaultTolerance: 3.0,
  stepMultiplier: 1.0,
  maxRadius: 5.0,
});

// Build position array
const position = buildQueryPosition(viewState, ndim);

// Execute query (uses workers if enabled)
const chunkIndices = await executeSpatialQuery({
  chunkBounds,
  queryPosition: position,
  queryTolerance: tolerance,
  numChunks,
  ndim,
});
```

## Tolerance Calculation

Unified tolerance logic across all loaders:

| Dimension Type                 | Tolerance Value                    |
| ------------------------------ | ---------------------------------- |
| Displayed (in displayDims)     | `1e10` (infinite - see all points) |
| Hidden with step               | `step * stepMultiplier`            |
| Hidden with explicit tolerance | `viewState.tolerance[d]`           |
| Hidden (fallback)              | `maxRadius` or `defaultTolerance`  |

## Worker Integration

All components automatically use workers when enabled:

```typescript
// Config check happens automatically
if (appConfig.dataLoading.performance.useWebWorkers) {
  // Use worker pool
  const worker = await getWorkerPool().getWorker();
  result = await worker.someMethod(...);
} else {
  // Main thread fallback
  result = mainThreadImplementation(...);
}
```

Worker fallback on error:

```typescript
try {
  result = await worker.someMethod(...);
} catch (error) {
  log.warning('Worker failed, falling back to main thread');
  result = mainThreadImplementation(...);
}
```

## Migration Guide

### Phase 1 (COMPLETE): All Loaders Use RangeLoader

All three spatial index loaders have been migrated to use RangeLoader:

**Loaders Migrated:**

- ✅ `point-spatial-index-loader.ts` - Uses RangeLoader for all encoding types
- ✅ `lines-spatial-index-loader.ts` - Uses RangeLoader for vertex attribute loading
- ✅ `gsplats-spatial-index-loader.ts` - Uses RangeLoader for gsplat array loading

**Code Reduction:** ~600 lines of duplicated encoding dispatch code eliminated

```typescript
// Before migration (in each loader)
private async loadQuantizedRanges(...) { ... }  // 100 lines
private async loadBroadcastedRanges(...) { ... }  // 50 lines
private async loadLUTRanges(...) { ... }  // 80 lines
private async loadDirectRanges(...) { ... }  // 40 lines
private normalizeDtype(...) { ... }  // 10 lines

// After migration
import { RangeLoader } from './loaders';

private rangeLoader = new RangeLoader(this.refRegistry);

// Single unified method handles all encodings:
await this.rangeLoader.loadRanges(array, attrs, ranges, output, total);
```

**Key Changes:**

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

### Phase 2 (COMPLETE): TransferableAccumulator

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

See `integration-example.ts` for a complete reference implementation.

### Phase 3 (FUTURE): Unified Base Class

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
├── index.ts                      # Module exports
├── README.md                     # This file
├── base-types.ts                 # Common type definitions
├── range-loader.ts               # Encoding dispatch
├── spatial-query-builder.ts      # Spatial query logic
├── transferable-accumulator.ts   # Zero-allocation + worker pattern
└── integration-example.ts        # Reference implementation
```

## Testing

```bash
# Run all tests (61 tests)
pnpm test src/tests/unit/data/loaders/

# Run specific tests
pnpm test src/tests/unit/data/loaders/spatial-query-builder.test.ts  # 24 tests
pnpm test src/tests/unit/data/loaders/transferable-accumulator.test.ts  # 27 tests
pnpm test src/tests/unit/data/loaders/integration-example.test.ts  # 10 tests
```

### Test Coverage

- **spatial-query-builder.test.ts** (24 tests)
  - `computeQueryTolerance`: Displayed dims, hidden dims with step/tolerance/fallback
  - `buildQueryPosition`: Position array building
  - `chunkIndicesToRanges`: Chunk → range conversion
  - `mergeRanges`: Range merging logic
  - `shouldExtendVisibility`: extend_to_all handling
  - `SpatialQueryBuilder`: Full query execution

- **transferable-accumulator.test.ts** (27 tests)
  - Basic operations, capacity management
  - Detach/adopt cycle for worker transfer
  - Optional buffer enabling
  - Points, Lines, GSplats factory functions
  - Memory tracking statistics

- **integration-example.test.ts** (10 tests)
  - PointsLoaderIntegrationExample lifecycle
  - 3D projection with colors and radii
  - Capacity growth and buffer reuse tracking
  - SpatialQueryBuilder integration

## See Also

- [SPECIFICATIONS.md](../SPECIFICATIONS.md) - Data loading specifications
- [array-decoder.ts](../array-decoder.ts) - Low-level array decoding
