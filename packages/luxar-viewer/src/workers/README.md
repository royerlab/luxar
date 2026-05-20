# luxar-viewer/src/workers

Multi-threaded worker pool for offloading CPU-intensive spatial queries, nD visibility computations, projection, clipping, and array decoding from the main thread.

## Architecture

```
Main Thread (rendering, UI)
  │
  └─► WorkerPool (singleton, least-busy load balancing)
        ├── DataWorker 0  ──► WASM module instance
        ├── DataWorker 1  ──► WASM module instance
        └── DataWorker N  ──► WASM module instance
```

Communication between main thread and workers uses [Comlink](https://github.com/GoogleChromeLabs/comlink) for typed RPC.

## Data Flow

```mermaid
sequenceDiagram
    participant Loader as Main thread loader
    participant Cache as Chunk cache
    participant Pool as WorkerPool
    participant Worker as DataWorker
    participant WASM as WASM / TypeScript fallback
    participant GPU as WebGL buffers

    Loader->>Cache: Fetch raw Zarr chunk bytes
    Cache-->>Loader: Return ArrayBuffer / cached bytes
    Loader->>Pool: Request least-busy worker
    Pool-->>Loader: DataWorker API + tracking hooks
    Loader->>Worker: Decode arrays / compute nD visibility
    Worker->>WASM: Run accelerated kernel if available
    WASM-->>Worker: Decoded or filtered typed arrays
    Worker-->>Loader: Transfer result buffers
    Loader->>GPU: Update geometry/material buffers
```

The main thread owns network I/O, cache coordination, and GPU updates. Workers
own CPU-heavy decoding, projection, clipping, and visibility kernels. Result
buffers are transferred back to the main thread to avoid copying where possible.

## Usage

```typescript
import { getWorkerPool, disposeWorkerPool } from './workers';

// Get singleton pool (auto-initializes on first call)
const pool = getWorkerPool();
await pool.initialize();

// Recommended: route every call through `runWithTimeout()` so
// hung workers are detected via the per-call budget. The pool
// picks a worker via load-balanced selection, tracks the call,
// and applies the configured timeout (visibility / projection /
// decode kinds map to different default budgets).
const result = await pool.runWithTimeout('querySpatialIndex', 'visibility', (api) =>
  api.querySpatialIndex(/* ... */)
);

// Clean up
disposeWorkerPool();
```

**Note**: `pool.getWorker()` and `pool.getWorkerWithTracking()`
also exist for advanced scenarios but bypass the per-call
timeout guard. Production hot paths should use `runWithTimeout`
unless they have their own timeout strategy.

## Worker Pool

### Configuration

- **Worker count**: Defaults to `navigator.hardwareConcurrency - 1` (reserves one core for main thread)
- **Minimum**: 1 worker
- **Fallback**: If `hardwareConcurrency` is unavailable, assumes 4 cores

### Load Balancing

The pool tracks `activeQueries` per worker and selects the worker with the fewest active tasks. Callers using `getWorkerWithTracking()` receive `markQueryStart`/`markQueryEnd` callbacks to keep the counters accurate.

### Initialization

- Lazy: workers are created on first `getWorkerPool()` call
- Safe: promise deduplication ensures concurrent callers share a single initialization
- Resilient: uses `Promise.allSettled()` so partial worker failures don't block the pool

## Data Worker API

Each worker loads a WASM module on `initialize()` and exposes these operations:

| Category            | Methods                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| **Spatial queries** | `querySpatialIndex()` — find chunks intersecting nD slice                                   |
| **nD visibility**   | `computeNDVisibilityPoints()`, `computeNDVisibilityLines()`, `computeNDVisibilityGSplats()` |
| **Decoding**        | `decodeQuantized()`, `decodeLogScalar()`, `decodeLUT()`, `decodeBroadcasted()`              |
| **Projection**      | `projectPointsTo3D()`, `projectLinesTo3D()`, `projectGSplatsTo3D()`                         |

### What workers handle

- Spatial index queries (chunk bounding box tests)
- nD visibility computation (hypersphere intersection)
- Array decoding (LUT, quantization, log-space)

### What stays on main thread

- Zarr chunk fetching (network I/O)
- GPU buffer updates (WebGL)
- UI rendering and interaction

## Fallback Behavior

If WASM is unavailable, each worker falls back to a pure TypeScript implementation of the same API (slower but functionally identical). See `src/wasm/` for details.

## File Structure

```
workers/
├── worker-pool.ts   — Pool manager with load balancing
└── data-worker.ts   — Worker implementation (WASM + Comlink)
```
