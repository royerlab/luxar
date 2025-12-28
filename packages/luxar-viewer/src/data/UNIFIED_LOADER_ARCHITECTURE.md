# Unified Loader Architecture Proposal

**Status**: IMPLEMENTED (TransferableAccumulator in `loaders/transferable-accumulator.ts`)
**Date**: 2025-12-27
**Goal**: Achieve both zero-allocation AND CPU offload

---

## Problem Statement

Current architecture has two mutually exclusive optimization paths:

1. **Accumulators** (zero-allocation): Pre-allocated buffers on main thread
2. **Workers** (CPU offload): Computation in background thread

Points loader blocks workers when accumulators are enabled because workers can't write to main thread's accumulator buffers (separate memory spaces).

---

## Proposed Solution: TransferableAccumulator

### Core Idea

Use `Comlink.transfer()` for **bidirectional zero-copy buffer exchange**:

```
┌─────────────────┐                    ┌─────────────────┐
│   MAIN THREAD   │                    │     WORKER      │
│                 │                    │                 │
│  Accumulator    │ ── transfer() ──► │  Receives       │
│  (owns buffers) │                    │  buffers        │
│                 │                    │                 │
│                 │                    │  Fills buffers  │
│                 │                    │  (zero alloc!)  │
│                 │                    │                 │
│  Adopts         │ ◄── transfer() ── │  Returns        │
│  buffers back   │                    │  buffers        │
└─────────────────┘                    └─────────────────┘
```

### Key Properties

1. **Zero-copy transfer**: `transfer()` moves ownership without copying
2. **Buffer reuse**: Same buffers cycle between main thread and worker
3. **Zero allocation** (after warmup): No `new Float32Array()` in steady state
4. **CPU offload**: Heavy computation runs on worker thread

---

## Detailed Design

### 1. TransferableAccumulator Class

```typescript
/**
 * Accumulator that supports zero-copy transfer to/from workers.
 * Buffers can be "detached" for worker use and "adopted" back.
 */
class TransferableAccumulator<T extends AccumulatorData> {
  private buffers: AccumulatorBuffers | null = null;
  private capacity: number = 0;

  /**
   * Detach buffers for transfer to worker.
   * After this call, accumulator has no buffers until adopt() is called.
   */
  detach(): AccumulatorBuffers {
    if (!this.buffers) {
      // First call or after dispose - allocate new
      this.buffers = this.allocateBuffers(this.capacity);
    }
    const detached = this.buffers;
    this.buffers = null; // Ownership transferred
    return detached;
  }

  /**
   * Adopt buffers returned from worker.
   * Takes ownership of the transferred buffers.
   */
  adopt(buffers: AccumulatorBuffers): void {
    // Dispose old buffers if any (shouldn't happen in normal flow)
    this.buffers = buffers;
  }

  /**
   * Get transferable list for Comlink.transfer()
   */
  getTransferables(buffers: AccumulatorBuffers): ArrayBuffer[] {
    return [
      buffers.positions.buffer,
      buffers.colors?.buffer,
      buffers.radii?.buffer,
      buffers.sharpness?.buffer,
    ].filter(Boolean) as ArrayBuffer[];
  }
}
```

### 2. Unified Loader Base Class

```typescript
/**
 * Base class for all spatial index loaders.
 * Provides unified pattern for loading, projection, and worker dispatch.
 */
abstract class SpatialIndexLoader<TData, TViewState> {
  protected accumulator: TransferableAccumulator<TData>;
  protected arrayLoader: ArrayLoader;  // Shared encoding dispatch

  /**
   * Main entry point - unified pattern for all loaders
   */
  async loadData(viewState: TViewState): Promise<TData> {
    // 1. Query spatial index (worker if enabled)
    const ranges = await this.queryVisibleRanges(viewState);

    // 2. Load raw arrays (uses shared ArrayLoader)
    const rawData = await this.loadRanges(ranges);

    // 3. Project to 3D (worker with transferable buffers)
    const projected = await this.projectTo3D(rawData, viewState);

    return projected;
  }

  /**
   * Project using worker with transferable buffer reuse
   */
  protected async projectTo3D(
    rawData: RawData,
    viewState: TViewState
  ): Promise<TData> {
    const config = appConfig.dataLoading.performance;

    if (config.useWebWorkers && rawData.count > 1000) {
      // Detach accumulator buffers for transfer to worker
      const buffers = this.accumulator.detach();

      // Transfer buffers to worker (zero-copy)
      const result = await this.projectUsingWorker(rawData, viewState, buffers);

      // Adopt returned buffers (zero-copy back)
      this.accumulator.adopt(result.buffers);

      return result.data;
    } else {
      // Main thread fallback (small datasets)
      return this.projectOnMainThread(rawData, viewState);
    }
  }

  // Abstract methods for type-specific logic
  protected abstract queryVisibleRanges(viewState: TViewState): Promise<Range[]>;
  protected abstract projectUsingWorker(...): Promise<WorkerResult<TData>>;
  protected abstract projectOnMainThread(...): TData;
}
```

### 3. Shared ArrayLoader

```typescript
/**
 * Generic array loader with encoding dispatch.
 * Replaces duplicated loading logic in all three loaders.
 */
class ArrayLoader {
  /**
   * Load array with automatic encoding detection and worker dispatch.
   */
  async loadArray(
    array: zarr.Array,
    ranges: Range[],
    encoding?: EncodingMetadata
  ): Promise<TypedArray> {
    // Detect encoding type
    const encodingType = encoding?.type || 'direct';

    switch (encodingType) {
      case 'broadcasted':
        return this.loadBroadcasted(array, ranges, encoding);

      case 'quantized':
        return this.loadQuantized(array, ranges, encoding);

      case 'lut':
        return this.loadLUT(array, ranges, encoding);

      case 'array_ref':
        return this.loadArrayRef(array, ranges, encoding);

      default:
        return this.loadDirect(array, ranges);
    }
  }

  // Worker dispatch for each encoding type
  private async loadQuantized(...): Promise<TypedArray> {
    const config = appConfig.dataLoading.performance;

    if (config.useWebWorkers) {
      const worker = await getWorkerPool().getWorker();
      return worker.decodeQuantized({ ... });
    }

    // Main thread fallback
    return decodeQuantizedMainThread(...);
  }
}
```

### 4. Worker API Changes

```typescript
// data-worker.ts

/**
 * Project points to 3D with buffer reuse.
 * If inputBuffers provided, writes to them (zero-allocation).
 * Otherwise allocates new buffers.
 */
async function projectPointsTo3D(params: {
  rawData: RawPointsData;
  viewState: ProjectionViewState;
  // NEW: Optional pre-allocated buffers from accumulator
  inputBuffers?: {
    positions3D: Float32Array;
    colors: Float32Array | null;
    radii: Float32Array | null;
    sharpness: Float32Array | null;
  };
}): Promise<{
  data: ProjectedPointsData;
  // Return buffers for accumulator to adopt
  buffers: AccumulatorBuffers;
}> {
  const { rawData, viewState, inputBuffers } = params;

  // Use provided buffers or allocate new (first call only)
  const buffers = inputBuffers || allocateBuffers(rawData.count);

  // Fill buffers with projected data
  projectIntoBuffers(rawData, viewState, buffers);

  // Transfer buffers back (zero-copy)
  return transfer({ data: buildResult(buffers), buffers }, getTransferables(buffers));
}
```

---

## File Structure After Refactoring

```
src/data/
├── loaders/
│   ├── base-spatial-loader.ts      # NEW: Unified base class
│   ├── point-loader.ts             # Simplified, extends base
│   ├── lines-loader.ts             # Simplified, extends base
│   └── gsplats-loader.ts           # Simplified, extends base
│
├── accumulator/
│   ├── transferable-accumulator.ts # NEW: Worker-compatible accumulator
│   ├── points-accumulator.ts       # Type-specific buffer shapes
│   ├── lines-accumulator.ts
│   └── gsplats-accumulator.ts
│
├── encoding/
│   └── array-loader.ts             # NEW: Shared encoding dispatch
│
├── projection/
│   ├── project-points.ts           # Extracted from loader
│   ├── project-lines.ts            # Extracted from scene-loader
│   └── project-gsplats.ts          # Extracted from scene-loader
│
└── scene-loader.ts                 # Simplified: just orchestration
```

---

## Migration Strategy

### Phase 1: Extract Shared Logic (Non-breaking)

1. Create `ArrayLoader` class
2. Create `BaseSpatialLoader` class
3. Existing loaders extend base class, delegate to shared code
4. No behavior change, just code organization

### Phase 2: Implement TransferableAccumulator (Non-breaking)

1. Create `TransferableAccumulator` alongside existing accumulators
2. Add `inputBuffers` parameter to worker functions
3. Points loader uses new pattern first (proof of concept)
4. Verify performance parity

### Phase 3: Unify All Loaders

1. Lines loader adopts unified pattern
2. GSplats loader adopts unified pattern
3. Remove projection from scene-loader
4. Update documentation

### Phase 4: Cleanup

1. Remove old accumulator code
2. Remove duplicated functions
3. Update tests

---

## Performance Expectations

| Scenario               | Current                               | After Refactoring               |
| ---------------------- | ------------------------------------- | ------------------------------- |
| Small 3D dataset (<1K) | Main thread, zero-alloc               | Main thread, zero-alloc         |
| Large 3D dataset (>1K) | Main thread (blocked by accumulators) | Worker, zero-alloc after warmup |
| nD dataset navigation  | Main thread per-frame                 | Worker, zero-alloc after warmup |
| Memory churn           | Low (accumulators)                    | Low (transferable reuse)        |
| Main thread blocking   | Moderate-High                         | Low (worker offload)            |

---

## Open Questions

1. **SharedArrayBuffer alternative?**
   - Would allow true shared memory (no transfer needed)
   - Requires COOP/COEP headers
   - More complex synchronization

2. **Double-buffering?**
   - Accumulator holds 2 buffer sets
   - One for GPU, one for worker
   - Swap on completion
   - Higher memory, lower latency

3. **Threshold tuning?**
   - Current: >1000 elements → use worker
   - Should this be configurable?
   - Different thresholds per data type?

---

## Next Steps

1. [ ] Review and approve this proposal
2. [ ] Create `ArrayLoader` class (Phase 1)
3. [ ] Create `BaseSpatialLoader` class (Phase 1)
4. [ ] Implement `TransferableAccumulator` (Phase 2)
5. [ ] Migrate Points loader (Phase 2)
6. [ ] Migrate Lines/GSplats loaders (Phase 3)
7. [ ] Update documentation (Phase 4)
