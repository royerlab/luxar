import type { DataLoadingPerformanceConfig } from './types';

/**
 * Performance optimization settings.
 */
export const dataLoadingPerformanceConfig: DataLoadingPerformanceConfig = {
  // Object pooling — multi-type accumulator with in-place projection
  // and filtering. Eliminates allocations in projectTo3D (positions,
  // filtered arrays, return object) by writing directly into
  // accumulator buffers and compacting in place.
  useAccumulators: true,
  initialAccumulatorCapacity: 8192,

  // Web workers offload CPU-heavy operations: nD→3D projection
  // (with built-in per-element visibility/culling) and decoding. AABB
  // spatial queries always run on the main thread (faster than the
  // roundtrip).
  useWebWorkers: true,
  workerCount: 0, // 0 = auto (uses navigator.hardwareConcurrency - 1)
  // Per-call worker timeout (projection + decode). Projection over
  // millions of items is slow. 0 disables timeout enforcement.
  workerProjectionTimeoutMs: 60000,
  // Worker pool init timeout: protects against unreachable worker
  // scripts (404 on the chunk URL, blocked by route, dev-server
  // misconfig). 10s is generous for any healthy environment;
  // anything longer suggests a real load problem and the app should
  // fall back to main-thread execution rather than hang on boot.
  workerInitTimeoutMs: 10000,

  // GPU buffer pool — multi-type support (Float32Array, Uint8Array,
  // Uint16Array with auto normalization). Reuses geometries when
  // capacity AND types match (0ms allocation on reuse). Integrated
  // into the scene-loader geometry-update path.
  useGPUBufferPool: true,
  gpuPoolMaxSize: 20,
  gpuPoolEvictionFrames: 300,
  gpuPoolEvictBatchSize: 5,
  // Single GPU-geometry byte budget (pool + LOD retention). `null`
  // auto-sizes from device memory (see rendering/gpu-byte-budget.ts);
  // `0` disables byte-budget eviction; a positive number pins it.
  // Without a budget, a 10M-element Lines buffer (~760 MB at 1.5×
  // overallocation) or a stack of retained LOD levels can briefly hold
  // gigabytes. Overridable at runtime via `?gpuBudgetMB=`.
  gpuPoolMaxBytes: null, // auto-size from deviceMemory and/or one third of cacheBudgetMB
  refinementPassBudgetMs: 8, // half a 60 Hz frame of refinement work between paint yields

  // Debugging
  enablePerformanceMonitoring: false,
};
