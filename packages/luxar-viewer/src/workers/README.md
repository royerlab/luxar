# luxar-viewer/src/workers

Multi-threaded worker pool for offloading CPU-intensive spatial queries, nD visibility computations, and array decoding from the main thread.

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

## Usage

```typescript
import { getWorkerPool, disposeWorkerPool } from './workers';

// Get singleton pool (auto-initializes on first call)
const pool = getWorkerPool();
await pool.initialize();

// Simple: get a worker (least-busy selection, no tracking)
const worker = await pool.getWorker();
const result = await worker.querySpatialIndex(/* ... */);

// With tracking: enables accurate load balancing
const { api, markQueryStart, markQueryEnd } = await pool.getWorkerWithTracking();
markQueryStart();
try {
  const result = await api.querySpatialIndex(/* ... */);
} finally {
  markQueryEnd();
}

// Clean up
disposeWorkerPool();
```

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
