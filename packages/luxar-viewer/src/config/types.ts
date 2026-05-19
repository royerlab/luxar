// Type definitions for ALL configuration interfaces
// This file is the single source of truth for configuration types

import type { CameraConfig } from './sections/camera/types';
export type { CameraConfig };

import type { AnimationConfig } from './sections/animation/types';
export type { AnimationConfig };

import type { SceneConfig, ShaderConfig } from './sections/scene/types';
export type { SceneConfig, ShaderConfig };

import type { AdaptiveDPRConfig } from './sections/adaptive-dpr/types';
export type { AdaptiveDPRConfig };

import type { CacheConfig } from './sections/cache/types';
export type { CacheConfig };

/**
 * Configuration range with min/max/default/step
 */
export interface ConfigRange {
  min: number;
  max: number;
  default: number;
  step?: number;
}

/**
 * Configuration value with default
 */
export interface ConfigValue {
  default: number;
}

/**
 * Fly controls configuration
 */
export interface FlyControlsConfig {
  inertialMode: {
    default: boolean;
  };
  movement: {
    speed: ConfigRange;
    acceleration: ConfigRange;
    damping: ConfigRange;
  };
  rotation: {
    speed: ConfigRange;
    damping: ConfigRange;
  };
  look: {
    mouseSpeed: ConfigValue;
  };
  physics: {
    velocityThreshold: number;
    dampingPower: number;
    angularVelocityThreshold: number;
  };
}

/**
 * Orbit controls configuration
 */
export interface OrbitControlsConfig {
  autoRotate: {
    speed: ConfigRange;
  };
  zoom: {
    minDistance: number;
    maxDistance: number;
    speed: ConfigRange;
  };
  damping: {
    enabled: boolean;
    factor: ConfigRange;
  };
}

/**
 * Scale multipliers for adapting camera controls to scene size.
 * All factors are multiplied by the bounding box diagonal to produce
 * the actual control parameter value.
 */
export interface ScaleMultipliers {
  /** orbit minDistance = diagonal * factor (default: 0.01) */
  minDistanceFactor: number;
  /** orbit maxDistance = diagonal * factor (default: 100) */
  maxDistanceFactor: number;
  /** fly movementSpeed = diagonal * factor (default: 0.05) */
  flySpeedFactor: number;
}

/**
 * Control system configuration
 */
export interface ControlsConfig {
  fly: FlyControlsConfig;
  orbit: OrbitControlsConfig;
  scaleMultipliers: ScaleMultipliers;
}

/**
 * Input handling configuration
 */
export interface InputConfig {
  defaultSensitivity: number;
  keyboard: {
    shortcuts: {
      toggleFullscreen: string;
      toggleHelp: string;
      toggleDimensions: string;
      toggleDatasetBrowser: string;
      togglePerformance: string;
      toggleRendering: string;
      toggleScaleBar: string;
      toggleColormapLegend: string;
      toggleDebugConsole: string;
      toggleLayers: string;
      recenterCamera: string;
      toggleControlMode: string;
      toggleInertialMode: string;
      toggleCinematicMode: string;
      toggleOverlays: string;
    };
    flyModeKeys: string[];
    dimensionKeys: string[];
  };
  mouse: {
    doubleClickDelay: number;
  };
}

// Styling uses CSS variables and classes in src/styles/ (see the
// theming system in src/themes/).

/**
 * Debug console configuration
 */
export interface DebugConsoleConfig {
  panel: {
    defaultWidth: number;
    defaultHeight: number;
    minWidth: number;
    maxWidth: number;
    minHeight: number;
    maxHeight: number;
    bottomOffset: number;
    leftOffset: number;
  };
  interceptor: {
    maxBufferSize: number;
  };
  resize: {
    borderWidth: number;
  };
  style: {
    backgroundColor: string;
    borderColor: string;
    borderRadius: number;
    backdropBlur: number;
    boxShadow: string;
  };
}

/**
 * UI component configurations
 */
export interface UIComponentsConfig {
  datasetBrowser: {
    borderRadius: {
      panel: number;
      section: number;
      element: number;
    };
    padding: {
      panel: number;
      section: number;
      element: number;
    };
  };
  debugConsole: {
    borderRadius: {
      header: number;
      content: number;
      button: number;
    };
  };
  renderingControls: {
    borderRadius: {
      checkbox: number;
      section: number;
      header: number;
    };
  };
  dataMonitor: {
    borderRadius: {
      card: number;
      section: number;
    };
    padding: {
      default: number;
      compact: number;
    };
  };
}

/**
 * UI configuration for overlays and visual elements
 */
export interface UIConfig {
  zIndex: {
    // Base layer components
    dimensionSliders: number;
    performanceMonitor: number;
    debugConsole: number;
    // Mid-layer overlays
    datasetBrowser: number;
    loading: number;
    error: number;
    help: number;
    renderingControls: number;
    recordingPanel: number;
    layersPanel: number;
    // Top layer
    statsMonitor: number;
  };
  timings: {
    errorAutoDismissMs: number;
    helpClickDelayMs: number;
  };
  spinner: {
    size: number;
    borderWidth: number;
  };
  // Styling lives in CSS variables/classes rather than config objects.
  debugConsole: DebugConsoleConfig;
  components: UIComponentsConfig;
  scaleBar: {
    targetWidthPx: number;
    position: 'bottom-left' | 'bottom-right';
  };
}

/**
 * Data loading network configuration
 */
export interface DataLoadingNetworkConfig {
  timeoutMs: number;
  /**
   * Dedicated short budget for the L2 cache-validation HEAD probe. On flaky
   * networks the validation must NOT block scene loading for the full
   * `timeoutMs` — failing fast lets cached data render quickly.
   *
   * **Trade-off**: lower values fail faster (good — render from cached
   * data while the network is slow). Higher values tolerate slower
   * networks but block first-paint until the validation completes or
   * times out. Default is `5000` (5 s).
   *
   * **3G / Edge / high-latency**: real-world 3G round-trip + server
   * processing can exceed 5 s, which would cause spurious validation
   * timeouts and force re-fetches of otherwise-valid cached data. If you
   * target slow networks, raise this to `>=8000` (8 s).
   *
   * The validation layer logs a warning when this drops below 3 s (the
   * "almost certainly broken" floor); 5 s is the broadband-tuned
   * default and does not warn.
   */
  validationTimeoutMs: number;
  maxConcurrent: number;
  retryAttempts: number;
}

/**
 * Data loading memory configuration
 */
export interface DataLoadingMemoryConfig {
  targetHeapUsage: number;
  minCacheMB: number;
  checkIntervalMs: number;
  adjustmentThresholds: {
    critical: number;
    high: number;
  };
}

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

/**
 * Data loading spatial configuration
 */
export interface DataLoadingSpatialConfig {
  defaultTolerance: number;
  defaultMaxRadius: number;
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
 * User-adjustable rendering settings that can be persisted
 */
export interface RenderingSettings {
  // Camera settings
  fov: number;
  fovPreset: '28mm Wide' | '35mm' | '50mm Normal' | '85mm Portrait' | '135mm Tele' | 'Custom';
  near: number;
  far: number;
  // Dynamic clipping planes
  dynamicClippingEnabled: boolean;
  // Rendering effects (bloom is now the single source of truth)
  bloomEnabled: boolean;
  bloomThreshold: number;
  bloomStrength: number;
  bloomRadius: number;
  bloomLevels: number;
  exposure: number; // Log2 stops, default 0.0
  globalOffset: number; // Additive shift, default 0.0
  globalGamma: number; // Midtone curve, default 1.0
  fxaaEnabled: boolean;
  msaaEnabled: boolean;
  msaaSamples: number;
  ssaaEnabled: boolean;
  ssaaMultiplier: number;
  toneMapping: 'None' | 'Linear' | 'Reinhard' | 'Cineon' | 'ACES' | 'AgX' | 'Neutral';
  vignetteEnabled: boolean;
  vignetteDarkness: number;
  vignetteOffset: number;
  // Detector noise effect (physics-based: Poisson + Gaussian + FPN)
  detectorNoiseEnabled: boolean;
  detectorNoiseReadoutSigma: number;
  detectorNoisePhotonGain: number;
  detectorNoiseFpnSigma: number;
  // Chromatic lens distortion effect
  chromaticLensDistortionEnabled: boolean;
  chromaticLensDistortionX: number;
  chromaticLensDistortionY: number;
  chromaticLensDispersion: number;
  chromaticLensPrincipalPointX: number;
  chromaticLensPrincipalPointY: number;
  chromaticLensFocalLengthX: number;
  chromaticLensFocalLengthY: number;
  chromaticLensSkew: number;
  // Navigation controls
  controlType: 'orbit' | 'fly' | 'ortho';
  autoRotate: boolean;
  autoRotateSpeed: number;
  /**
   * "Natural drag" — swap LEFT ↔ RIGHT mouse buttons in orbit mode so a
   * one-finger touchpad drag rotates (and two-finger / right-drag pans).
   * Defaults to true on macOS. Orbit (3D) only; ortho and fly modes ignore.
   */
  naturalDrag: boolean;
  // Fly controls - these are added at runtime from config.controls.fly
  flyMovementSpeed?: number;
  flyRotationSpeed?: number;
  flyInertialMode?: boolean;
  flyDamping?: number;
  flyRotationDamping?: number;
  // Adaptive resolution
  adaptiveDPREnabled: boolean;
  // Cinematic mode toggle (for UI only, actual state determined by effects)
  cinematicMode: boolean;
}

/**
 * Rendering controls configuration
 */
export interface RenderingControlsConfig {
  defaults: RenderingSettings;
}

// Rendering settings are centralized in RenderingSettings.

/**
 * WebGL context attributes
 */
export interface WebGLContextAttributes {
  alpha: boolean;
  antialias: boolean;
  depth: boolean;
  stencil: boolean;
  powerPreference: 'high-performance' | 'low-power' | 'default';
  colorSpace: string;
  preserveDrawingBuffer: boolean;
  desynchronized: boolean;
  premultipliedAlpha: boolean;
  failIfMajorPerformanceCaveat: boolean;
}

/**
 * THREE.WebGLRenderer configuration (renderer-specific settings only).
 * Shared attributes (antialias, powerPreference, preserveDrawingBuffer,
 * premultipliedAlpha) live in WebGLContextAttributes and are spread
 * alongside these at renderer creation time.
 */
export interface WebGLRendererConfig {
  logarithmicDepthBuffer: boolean;
  precision: 'highp' | 'mediump' | 'lowp';
  shadowMap: {
    enabled: boolean;
    type: number;
  };
}

/**
 * WebGL render target configuration
 */
export interface WebGLRenderTargetConfig {
  depthBuffer: boolean;
  stencilBuffer: boolean;
  samples: number;
}

/**
 * WebGL configuration
 */
export interface WebGLConfig {
  context: WebGLContextAttributes;
  renderer: WebGLRendererConfig;
  renderTarget: WebGLRenderTargetConfig;
}

/**
 * Dimension animation configuration
 * Controls FPS-based animation through dimension ranges with various loop modes
 */
export interface DimensionAnimationConfig {
  defaults: {
    targetFPS: number;
    loop: 'once' | 'loop' | 'bounce';
    direction: 'forward' | 'backward';
  };
  presets: {
    fps: number[];
    customMin: number;
    customMax: number;
  };
  timing: {
    minFrameTimeMs: number;
    continuousTraverseSeconds: number;
  };
  ui: {
    showFPSFeedback: boolean;
    feedbackThreshold: number;
  };
}

/**
 * Complete application configuration structure
 */
export interface AppConfig {
  camera: CameraConfig;
  animation: AnimationConfig;
  adaptiveDPR: AdaptiveDPRConfig;
  scene: SceneConfig;
  shader: ShaderConfig;
  ui: UIConfig;
  renderingControls: RenderingControlsConfig;
  controls: ControlsConfig;
  input: InputConfig;
  dataLoading: DataLoadingConfig;
  webgl: WebGLConfig;
  cache: CacheConfig;
  dimensionAnimation: DimensionAnimationConfig;
  defaultZarrPath: string;
}
