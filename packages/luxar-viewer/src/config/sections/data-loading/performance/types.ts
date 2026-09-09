/**
 * Performance optimization configuration: object pooling, web workers,
 * WASM acceleration, GPU buffer pool.
 */
export interface DataLoadingPerformanceConfig {
  // Object pooling
  useAccumulators: boolean;
  initialAccumulatorCapacity: number;

  // Web Workers
  useWebWorkers: boolean;
  workerCount: number;
  /**
   * Soft timeout for projection- and decode-class worker calls
   * (project*To3D, decode*). Reject the awaiting promise after this
   * many ms with a `WorkerTimeoutError` and remove the worker from the
   * pool. 0 disables.
   */
  workerProjectionTimeoutMs: number;
  /**
   * Hard timeout for the per-worker `api.initialize()` Comlink call
   * during pool startup. Without this guard, a blocked / unreachable
   * worker script (e.g. a dev environment that 404s the worker chunk)
   * leaves Comlink waiting forever — the worker's `onerror` fires but
   * pool init runs *before* the worker is in the pool, so the
   * standard handleWorkerFailure path can't evict it. The guard
   * rejects the init promise so the caller can fall back gracefully.
   */
  workerInitTimeoutMs: number;

  // GPU buffer pool
  useGPUBufferPool: boolean;
  gpuPoolMaxSize: number;
  gpuPoolEvictionFrames: number;
  /**
   * Per-call eviction-batch cap for the GPU buffer pool. When many
   * pooled buffers cross the eviction threshold in the same frame
   * (common after a long pause + viewport change), without this cap
   * `evictUnused` would dispose every qualifying buffer synchronously,
   * stuttering the frame. The cap defers excess evictions to the
   * next frame. The pool-over-limit path bypasses the cap so memory
   * still stays bounded.
   */
  gpuPoolEvictBatchSize: number;
  /**
   * Single GPU-geometry byte budget shared by the buffer pool (pooled-
   * buffer eviction) and the LOD-group registry (resident-level
   * eviction). When usage exceeds it, the largest/coldest buffers are
   * disposed until back under budget.
   *
   * - `null` (default): **auto-size** from the lower of 25% of
   *   `navigator.deviceMemory` and the heap model's non-cache remainder, with a
   *   2 GB ceiling and a 512 MB fallback only when no signal exists. An explicit
   *   `cacheBudgetMB` replaces the heap-derived remainder in either direction.
   * - `0`: disable byte-budget eviction entirely (count-only / unbounded
   *   resident geometry).
   * - a positive number: pin the budget to exactly that many bytes.
   *
   * The `?gpuBudgetMB=` URL param overrides this at runtime.
   */
  gpuPoolMaxBytes: number | null;

  // Debugging
  enablePerformanceMonitoring: boolean;
}
