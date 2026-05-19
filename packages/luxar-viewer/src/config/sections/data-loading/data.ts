import type { DataLoadingConfig } from './types';
import { dataLoadingSpatialConfig } from './spatial/data';
import { dataLoadingNetworkConfig } from './network/data';
import { dataLoadingMemoryConfig } from './memory/data';

/**
 * Data loading configuration
 */
export const dataLoadingConfig: DataLoadingConfig = {
  spatial: dataLoadingSpatialConfig,
  network: dataLoadingNetworkConfig,
  memory: dataLoadingMemoryConfig,
  monitor: {
    timings: {
      eventCleanupInterval: 30000,
      maxEventAge: 300000,
      ratesCacheTimeout: 1000,
      defaultUpdateInterval: 100,
      minRenderInterval: 100,
      timelinePointInterval: 200,
      defaultTimeRange: 60,
      queryCleanupCheckInterval: 10,
      maxQueryAge: 60000,
    },
    thresholds: {
      lowCacheHitRate: 30,
      highQueryTime: 100,
      highLoadTime: 500,
      highMemoryUsage: 0.8,
      highErrorRate: 0.05,
      lowQueryEfficiency: 0.5,
    },
    limits: {
      maxEvents: 1000,
      maxTimelinePoints: 300,
      maxAdvisorHistory: 100,
      defaultMemoryLimit: 1024 * 1024 * 1024,
      rateCalculationWindow: 5000,
      bandwidthCalculationWindow: 1000,
    },
  },
  // Performance optimization settings.
  performance: {
    // Object pooling — multi-type accumulator with in-place projection
    // and filtering. Eliminates allocations in projectTo3D (positions,
    // filtered arrays, return object) by writing directly into
    // accumulator buffers and compacting in place.
    useAccumulators: true,
    initialAccumulatorCapacity: 8192,
    accumulatorGrowthFactor: 1.5,

    // Web workers offload CPU-heavy operations: nD→3D projection,
    // visibility, decoding. AABB spatial queries always run on the
    // main thread (faster than the roundtrip).
    useWebWorkers: true,
    workerCount: 0, // 0 = auto (uses navigator.hardwareConcurrency - 1)
    // Per-call worker timeouts. Visibility is fast (chunk-bounding-box
    // test); projection over millions of items is slow. 0 disables
    // timeout enforcement.
    workerVisibilityTimeoutMs: 30000,
    workerProjectionTimeoutMs: 60000,
    // Worker pool init timeout: protects against unreachable worker
    // scripts (404 on the chunk URL, blocked by route, dev-server
    // misconfig). 10s is generous for any healthy environment;
    // anything longer suggests a real load problem and the app should
    // fall back to main-thread execution rather than hang on boot.
    workerInitTimeoutMs: 10000,
    // Material cache eviction: 200 entries × 3 types = 600 cached
    // materials max. Users animating sliders can blow through this
    // quickly so eviction keeps memory bounded.
    materialCacheMaxSize: 200,

    // WASM acceleration — module loads automatically via initWasm()
    // when workers are enabled.
    useWASM: true,
    wasmModulePath: 'wasm/luxar_wasm_bg.wasm', // Resolved relative to bundle via import.meta.url

    // GPU buffer pool — multi-type support (Float32Array, Uint8Array,
    // Uint16Array with auto normalization). Reuses geometries when
    // capacity AND types match (0ms allocation on reuse). Integrated
    // into the scene-loader geometry-update path.
    useGPUBufferPool: true,
    gpuPoolMaxSize: 20,
    gpuPoolEvictionFrames: 300,
    gpuPoolEvictBatchSize: 5,
    // byte-budget eviction. Pooled buffers above this many bytes
    // are evicted (largest-first) regardless of count budget. Without
    // this, a 10M-element Lines buffer (~760 MB at 1.5× overallocation)
    // counts the same as a 1K-point buffer (~32 KB) in
    // `gpuPoolMaxSize`, so a single dataset switch can briefly hold
    // gigabytes. `0` disables byte-budget (count-only).
    gpuPoolMaxBytes: 512_000_000, // 512 MB

    // Debugging
    enablePerformanceMonitoring: false,
  },
};
