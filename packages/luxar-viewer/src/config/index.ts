// Main configuration file for the Luxar scene player
// This module centralizes ALL configuration values to ensure consistency
// and make the application easy to customize.

import type { AppConfig } from './types';
import { cameraConfig } from './sections/camera/data';
import { animationConfig } from './sections/animation/data';
import { sceneConfig, shaderConfig } from './sections/scene/data';
import { adaptiveDPRConfig } from './sections/adaptive-dpr/data';
import { cacheConfig } from './sections/cache/data';
import { dimensionAnimationConfig } from './sections/dimension-animation/data';
import { webglConfig } from './sections/webgl/data';
import { inputConfig } from './sections/input/data';
import { controlsConfig } from './sections/controls/data';
import { renderingControlsConfig } from './sections/rendering-controls/data';
import { uiConfig } from './sections/ui/data';

/**
 * Main configuration object containing all application settings
 *
 * Organized by functional area for easy navigation and maintenance.
 * Uses 'as const' to ensure TypeScript treats these as literal values
 * rather than generic types, enabling better type checking.
 */
export const config: AppConfig = {
  camera: cameraConfig,

  animation: animationConfig,

  adaptiveDPR: adaptiveDPRConfig,

  scene: sceneConfig,

  shader: shaderConfig,

  ui: uiConfig,

  renderingControls: renderingControlsConfig,

  controls: controlsConfig,

  input: inputConfig,

  // Data loading configuration
  dataLoading: {
    spatial: {
      defaultTolerance: 0.1, // Default tolerance for nD slicing
      defaultMaxRadius: 0.1, // Default max radius for spatial queries
    },
    network: {
      timeoutMs: 30000,
      // Dedicated short budget for the L2 cache-validation HEAD probe
      // (MultiLevelCachingStore.getRemoteContentHash). On flaky networks this
      // path must NOT block scene loading for the full timeoutMs — failing
      // fast is better since we can render from cached data.
      validationTimeoutMs: 5000,
      maxConcurrent: 6,
      retryAttempts: 3,
    },
    memory: {
      targetHeapUsage: 0.8,
      minCacheMB: 128,
      checkIntervalMs: 10000,
      adjustmentThresholds: {
        critical: 0.85,
        high: 0.7,
      },
    },
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
  },

  webgl: webglConfig,

  cache: cacheConfig,

  dimensionAnimation: dimensionAnimationConfig,

  // Default path to demo Zarr data when no source is specified
  // Empty string = show dataset browser instead of attempting to load non-existent dataset
  defaultZarrPath: '',
} as const;

// Export types
export type { AppConfig, RenderingSettings } from './types';
