import type { DataLoadingNetworkConfig } from './network/types';
export type { DataLoadingNetworkConfig };

import type { DataLoadingMemoryConfig } from './memory/types';
export type { DataLoadingMemoryConfig };

/**
 * Data loading monitor timings
 */
export interface MonitorTimings {
  eventCleanupInterval: number;
  maxEventAge: number;
  ratesCacheTimeout: number;
  defaultUpdateInterval: number;
  minRenderInterval: number;
  timelinePointInterval: number;
  defaultTimeRange: number;
  queryCleanupCheckInterval: number;
  maxQueryAge: number;
}

/**
 * Data loading monitor thresholds
 */
export interface MonitorThresholds {
  lowCacheHitRate: number;
  highQueryTime: number;
  highLoadTime: number;
  highMemoryUsage: number;
  highErrorRate: number;
  lowQueryEfficiency: number;
}

/**
 * Data loading monitor limits
 */
export interface MonitorLimits {
  maxEvents: number;
  maxTimelinePoints: number;
  maxAdvisorHistory: number;
  defaultMemoryLimit: number;
  rateCalculationWindow: number;
  bandwidthCalculationWindow: number;
}

/**
 * Data loading monitor configuration
 */
export interface DataLoadingMonitorConfig {
  timings: MonitorTimings;
  thresholds: MonitorThresholds;
  limits: MonitorLimits;
}

import type { DataLoadingSpatialConfig } from './spatial/types';
export type { DataLoadingSpatialConfig };

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

  // WASM acceleration
  useWASM: boolean;
  wasmModulePath: string;

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
   * byte-budget for the GPU buffer pool. When `pooledBytes` exceeds
   * this value, `evictUnused()` disposes pooled buffers (largest first)
   * until under budget — independent of the count cap. `0` disables
   * the byte-budget pass (count-only behavior). Default ~512 MB.
   */
  gpuPoolMaxBytes: number;

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

/**
 * Data loading configuration
 */
export interface DataLoadingConfig {
  spatial: DataLoadingSpatialConfig;
  network: DataLoadingNetworkConfig;
  memory: DataLoadingMemoryConfig;
  monitor: DataLoadingMonitorConfig;
  performance: DataLoadingPerformanceConfig;
}
