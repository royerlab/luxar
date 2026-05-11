# luxar-viewer.workers - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-10

## Purpose

The `luxar-viewer.workers` package offloads CPU-heavy viewer work from the main thread. It provides a singleton worker pool, Comlink-based typed RPC, per-call timeout handling, and WASM-backed kernels with TypeScript fallbacks.

---

## Core Concepts

### Worker Pool

`WorkerPool` owns a configurable set of `data-worker.ts` instances. The pool initializes lazily, deduplicates concurrent initialization, and selects the least-busy worker for each tracked request.

### Data Worker API

Each worker exposes methods for:

- Spatial index queries.
- nD visibility checks for Points, Lines, and GSplats.
- Array decoding for encoded zarr arrays.
- nD to 3D projection for Points, Lines, and GSplats.

The worker initializes the WASM module once, then routes supported calls through WASM. If WASM is unavailable or a kernel is unsupported, the worker uses the TypeScript fallback exposed by `src/wasm`.

### Timeout Guard

Production callers should use `WorkerPool.runWithTimeout(kind, category, callback)`. The pool chooses a worker, tracks active work for load balancing, and applies the configured timeout budget. Direct worker access exists for specialized callers that provide their own timeout handling.

### Transferable Buffers

Large typed arrays should be transferred, not copied, when crossing the worker boundary. Projection methods can write into provided output buffers to reduce allocation and transfer only the populated buffers back to the main thread.

---

## Data Structures

### WorkerPool State

```text
WorkerPool:
  workers: WorkerHandle[]
  activeQueries: Map<workerId, number>
  initializationPromise: Promise<void> | null
  disposed: boolean
```

**Invariants**:

- Initialization is idempotent; concurrent callers share the same promise.
- Disposing the pool terminates workers and clears active-query state.
- Every tracked call increments and decrements exactly one active-query counter.

### DataWorker Request

```text
DataWorkerRequest:
  method: string
  payload: structured-clone-compatible object
  transferables?: ArrayBuffer[]
```

**Invariants**:

- Inputs are validated at the worker boundary before WASM is called.
- Returned typed arrays have lengths consistent with their reported element counts.
- Output buffers, when supplied, are large enough for the requested maximum output size.

---

## Algorithms

### Worker Selection

**Purpose**: Spread independent CPU work across available workers.

**Inputs**:

- `workers`: initialized worker handles.
- `activeQueries`: current active call count per worker.

**Outputs**:

- Worker handle for the next request.

**Algorithm**:

```text
1. Ensure the pool is initialized.
2. Scan active query counts for every worker.
3. Select the worker with the lowest count.
4. Increment its count before dispatch.
5. Dispatch the Comlink call.
6. Decrement its count in finally.
```

**Complexity**: O(worker count) per dispatch.

**Edge Cases**:

- If a worker fails during initialization, the pool can still use workers that initialized successfully.
- If all workers fail, callers receive an error and may fall back to main-thread TypeScript paths.

### Timeout Dispatch

**Purpose**: Prevent hung worker calls from blocking the viewer indefinitely.

**Inputs**:

- Request category (`visibility`, `projection`, `decode`, or custom category).
- Configured timeout budget.
- Worker callback.

**Outputs**:

- Callback result or timeout error.

**Algorithm**:

```text
1. Pick the least-busy worker.
2. Start a timeout timer using the configured budget.
3. Race the worker callback against the timer.
4. Clear the timer when either side settles.
5. Decrement the worker's active-query count in finally.
6. Propagate the result or error to the caller.
```

**Complexity**: O(1) beyond worker selection.

**Edge Cases**:

- Timeout does not guarantee cancellation inside WASM; it protects the caller and load-balancing counters.
- Callers that retry should treat timeouts as recoverable worker failures and use the main-thread fallback when appropriate.

### Projection with Optional Output Buffers

**Purpose**: Reduce allocations for hot projection paths.

**Inputs**:

- Source arrays and view state.
- Optional preallocated output buffers.

**Outputs**:

- Compacted projected arrays and element count.

**Algorithm**:

```text
1. Validate source lengths and output-buffer capacity.
2. Iterate source elements and test visibility/clipping.
3. Write visible projected data into output buffers when provided; otherwise allocate result arrays.
4. Return populated typed arrays plus the visible element count.
5. Transfer result buffers back to the main thread.
```

**Complexity**: O(input element count) time; O(1) extra space when output buffers are provided, otherwise O(visible element count).

---

## Validation Rules

- Worker entry points validate array lengths, dimensionality, display dimensions, and tolerance arrays before calling WASM.
- `viewState.displayDims` must contain exactly the displayed dimensions needed by the projection method.
- `viewState.tolerance` must have at least `ndim` entries for effective-radius calculations.
- Projection output buffers must be large enough for the maximum possible output.
- Worker results must be treated as untrusted until element counts and array lengths agree.

---

## Cross-Language Compatibility

The worker package calls WASM kernels generated from Rust and TypeScript fallback kernels in `src/wasm`. The observable API must remain identical across backends:

- Same input validation semantics.
- Same output array layout.
- Same handling of unsupported dimensionalities.
- Same visibility and clipping rules as the main-thread fallback functions.

When a WASM kernel supports fewer dimensions than the TypeScript path, the worker must route unsupported inputs to the TypeScript fallback rather than returning partial results.

---

## Related Specifications

- `luxar-viewer.wasm` - WASM kernel behavior and fallback parity (see `../wasm/SPECIFICATIONS.md`).
- `luxar-viewer.data` - Loader orchestration and projection call sites (see `../data/SPECIFICATIONS.md`).
- `luxar-viewer.config` - Worker timeout and performance configuration (see `../config/SPECIFICATIONS.md`).

---

## Changelog

- **v1.0.0** (2026-05-10): Initial specification for worker pool and data-worker contracts.
