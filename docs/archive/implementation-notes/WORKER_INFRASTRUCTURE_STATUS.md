# Worker Infrastructure Status

> **⚠️ Historical document (2025-12).** The standalone nD-visibility worker
> kernels (`computeNDVisibility{Points,Lines,GSplats}` / `compute_nd_visibility_*`)
> and the `querySpatialIndex` worker task described below never gained a
> production caller and were **deleted in 2026-07** (see CHANGELOG). Per-element
> nD visibility/culling lives inside the projection kernels
> (`clip_segments_batch`, effective radius, gsplats attenuation), and chunk-AABB
> spatial queries run on the main thread (`SpatialQueryBuilder`).

**Last Updated**: 2025-12-27
**Status**: ✅ **FULLY INTEGRATED AND ACTIVE**

---

## Overview

The WebWorker infrastructure is **complete and actively used** by all spatial index loaders. Workers offload CPU-intensive computations from the main thread, enabling smoother rendering during data loading.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN THREAD                             │
├─────────────────────────────────────────────────────────────────┤
│  SceneLoader                                                    │
│       ↓                                                         │
│  Point/Lines/GSplats Loaders                                    │
│       │                                                         │
│       ├──[useWebWorkers=true]──┐                                │
│       │                        ↓                                │
│       │              ┌─────────────────┐                        │
│       │              │  WORKER POOL    │                        │
│       │              │  (1-N workers)  │                        │
│       │              │                 │                        │
│       │              │ ┌─────────────┐ │                        │
│       │              │ │ DataWorker  │ │                        │
│       │              │ │             │ │                        │
│       │              │ │ WASM Module │ │                        │
│       │              │ │ (or TS      │ │                        │
│       │              │ │  fallback)  │ │                        │
│       │              │ └─────────────┘ │                        │
│       │              └─────────────────┘                        │
│       │                        │                                │
│       ├────────────────────────┘                                │
│       ↓                                                         │
│  ArrayDecoder (stays on main - needs zarr.Array)                │
│       ↓                                                         │
│  Data Accumulators → GPU Buffer Pool → Rendering                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

Workers are enabled by default in `src/config/index.ts`:

```typescript
dataLoading: {
  performance: {
    useWebWorkers: true,  // ✅ ENABLED
    workerCount: 0,       // 0 = auto (hardwareConcurrency - 1)
    useWASM: true,        // WASM with TypeScript fallback
  }
}
```

---

## What Workers Handle

### All Loaders (Points, Lines, GSplats)

| Operation           | Worker Function       | Fallback    |
| ------------------- | --------------------- | ----------- |
| Spatial Index Query | `querySpatialIndex()` | Main thread |
| Broadcast Decoding  | `decodeBroadcasted()` | Main thread |
| Quantized Decoding  | `decodeQuantized()`   | Main thread |
| Log-space Decoding  | `decodeLogScalar()`   | Main thread |
| LUT Decoding        | `decodeLUT()`         | Main thread |

### Points Loader Only

| Operation     | Worker Function       | Fallback    |
| ------------- | --------------------- | ----------- |
| 3D Projection | `projectPointsTo3D()` | Main thread |

### TransferableAccumulator Support (NEW)

The `projectPointsTo3D()` worker function now supports an optional `outputBuffers` parameter
for zero-allocation operation. This enables the **TransferableAccumulator pattern**:

```typescript
// Main thread: Detach buffers for zero-copy transfer to worker
const buffers = accumulator.detach();
const transferables = accumulator.getTransferables(buffers);

// Call worker with pre-allocated buffers
const result = await worker.projectPointsTo3D(
  Comlink.transfer({ params, outputBuffers: buffers }, transferables)
);

// Adopt buffers back (zero-copy)
accumulator.adopt(result.outputBuffers);
```

**Key Benefits**:

- Zero-allocation in steady state (after warmup)
- Zero-copy buffer transfer via `Comlink.transfer()`
- Enables BOTH accumulator pattern AND worker CPU offload
- Buffers cycle between main thread and worker without copying

---

## File Structure

```
src/workers/
├── data-worker.ts         # Worker implementation (~1166 lines)
├── worker-pool.ts         # Pool manager with load balancing (~251 lines)
└── WORKER_INFRASTRUCTURE_STATUS.md  # This file

src/data/loaders/          # NEW: Unified loader infrastructure
├── base-types.ts          # Common types (BaseViewState, LoadRange, etc.)
├── range-loader.ts        # Unified encoding dispatch
├── spatial-query-builder.ts  # Unified spatial query logic
├── transferable-accumulator.ts  # Zero-allocation + worker pattern
├── integration-example.ts # Reference implementation
├── index.ts               # Module exports
└── README.md              # Documentation

src/wasm/
├── index.ts               # WASM loader with fallback detection
├── types.ts               # WasmModule interface (525 lines)
├── typescript/            # TypeScript fallback implementations
│   ├── index.ts
│   ├── spatial.ts
│   ├── points.ts
│   ├── lines.ts
│   ├── gsplats.ts
│   ├── decode.ts
│   ├── projection.ts
│   └── ...
└── rust/                  # Rust WASM source
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── spatial.rs
        ├── points.rs
        ├── lines.rs
        ├── gsplats.rs
        └── ...

public/wasm/               # Built WASM output
├── luxar_wasm_bg.wasm    # ~44KB WASM binary
├── luxar_wasm.js         # JS bindings
└── luxar_wasm.d.ts       # TypeScript types
```

---

## Worker Pool Features

- **Multi-worker support**: Creates `hardwareConcurrency - 1` workers by default
- **Least-busy selection**: Routes queries to worker with fewest active tasks
- **Lazy initialization**: Workers created on first use
- **Error handling**: Graceful fallback to main thread on worker failure
- **Query tracking**: `getWorkerWithTracking()` for accurate load balancing

---

## WASM Module Status

| Component           | Status      | Notes                                   |
| ------------------- | ----------- | --------------------------------------- |
| Rust Source         | ✅ Complete | 39/39 unit tests passing                |
| Build Output        | ✅ Built    | `public/wasm/luxar_wasm_bg.wasm` (44KB) |
| TypeScript Fallback | ✅ Complete | Full parity with Rust implementation    |
| SIMD Optimization   | ✅ Enabled  | via wasm-opt                            |

---

## Performance Characteristics

### When Workers Help Most

- Large datasets (>10K points)
- nD datasets with visibility filtering
- Quantized/LUT encoded data requiring decoding
- Multiple concurrent nodes loading

### When Workers Add Overhead

- Small datasets (<1K points) - Comlink serialization overhead
- Simple 3D datasets without encoding - no decoding needed
- Single node scenes - no parallelization benefit

---

## Testing

```bash
# Unit tests (includes worker tests - skipped in JSDOM)
pnpm test --run

# Run Rust WASM tests
cd src/wasm/rust && cargo test

# E2E tests (workers active in browser)
pnpm test:e2e
```

---

## Disabling Workers

To disable workers and use main-thread processing:

```typescript
// In src/config/index.ts or via runtime config
config.dataLoading.performance.useWebWorkers = false;
```

---

## Integration Points

Workers are integrated in these loader methods:

### `point-spatial-index-loader.ts`

- `queryVisiblePointRanges()` - line 723
- `loadBroadcastedRanges()` - line 845
- `loadQuantizedRanges()` - line 925
- `loadLUTRanges()` - line 1017
- `projectTo3DUsingWorker()` - line 1705

### `lines-spatial-index-loader.ts`

- `queryVisibleRanges()` - line 528
- `loadBroadcastedRanges()` - line 670
- `loadQuantizedRanges()` - line 727
- `loadLUTRanges()` - line 796

### `gsplats-spatial-index-loader.ts`

- `queryVisibleRanges()` - line 372
- `loadBroadcastedRanges()` - line 482
- `loadQuantizedRanges()` - line 540
- `loadLUTRanges()` - line 610

---

## History

- **2025-12-24**: Initial worker infrastructure created (Phase 2)
- **2025-12-25**: WASM module built, TypeScript fallback completed
- **2025-12-26**: Loader integration completed for all three loaders
- **2025-12-27**: Documentation updated to reflect complete integration
- **2025-12-27**: Added TransferableAccumulator pattern and unified loaders module
  - New `src/data/loaders/` module with shared components
  - `projectPointsTo3D()` now supports `outputBuffers` for zero-allocation
  - 51 new unit tests for loaders infrastructure
