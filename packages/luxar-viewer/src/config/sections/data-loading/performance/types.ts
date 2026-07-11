/**
 * Performance optimization configuration: object pooling, web workers,
 * WASM acceleration, GPU buffer pool.
 */
export interface DataLoadingPerformanceConfig {
  // Object pooling
  useAccumulators: boolean;
  initialAccumulatorCapacity: number;
  accumulatorGrowthFactor: number;

  // Web Workers
  useWebWorkers: boolean;
  workerCount: number;
  /**
   * Soft timeout for visibility-class worker calls (computeNDVisibility*).
   * Reject the awaiting promise after this many ms with a
   * `WorkerTimeoutError` and remove the worker from the pool. 0 disables.
   */
  workerVisibilityTimeoutMs: number;
  /**
   * Soft timeout for projection-class worker calls (project*To3D).
   * Same semantics as `workerVisibilityTimeoutMs` but typically larger
   * since projection over millions of items takes longer than visibility.
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
   * - `null` (default): **auto-size** from `navigator.deviceMemory`
   *   (clamped to [512 MB, 2 GB]) — see `rendering/gpu-byte-budget.ts`.
   * - `0`: disable byte-budget eviction entirely (count-only / unbounded
   *   resident geometry).
   * - a positive number: pin the budget to exactly that many bytes.
   *
   * The `?gpuBudgetMB=` URL param overrides this at runtime.
   */
  gpuPoolMaxBytes: number | null;

  /**
   * Maximum number of cached materials per type (point, line, gsplat).
   * Materials are bucketed by attribute (opacity / gamma / intensity / …);
   * an unbounded cache leaks GPU shader programs over long sessions when
   * users animate sliders. Set to 0 to disable LRU eviction.
   */
  materialCacheMaxSize: number;

  // Debugging
  enablePerformanceMonitoring: boolean;
}
