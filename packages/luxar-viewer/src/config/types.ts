// Type definitions for ALL configuration interfaces
// This file is the single source of truth for configuration types

import type {
  HalfFloatType,
  LinearSRGBColorSpace,
  SRGBColorSpace,
  NoToneMapping,
  ACESFilmicToneMapping,
} from 'three';

/**
 * Camera configuration for 3D perspective and navigation
 */
export interface CameraConfig {
  fov: number;
  near: number;
  far: number;
  initialPosition: { x: number; y: number; z: number };
  fovMin: number;
  fovMax: number;
  fovSensitivity: number;
  fovPresets: Record<string, number>;
  lensDistortionPresets: Record<
    string,
    {
      distortionX: number;
      distortionY: number;
      principalPointX: number;
      principalPointY: number;
      focalLengthX: number;
      focalLengthY: number;
      skew: number;
      dispersion: number;
    }
  >;
}

/**
 * Animation loop and performance optimization settings
 */
export interface AnimationConfig {
  idleTimeoutMs: number;
  targetFPS: number;
  minFPS: number;
}

/**
 * Adaptive pixel ratio configuration for dynamic performance optimization
 *
 * This system dynamically adjusts the device pixel ratio based on real-time FPS
 * to maintain smooth frame rates during heavy rendering. Uses hysteresis to
 * prevent rapid toggling between quality levels.
 */
export interface AdaptiveDPRConfig {
  /** Enable adaptive DPR system (default: true) */
  enabled: boolean;
  /** Target FPS - slightly below 60 to prevent toggling (default: 55) */
  targetFPS: number;
  /** FPS threshold for scaling down resolution (default: 50) */
  minFPS: number;
  /** FPS threshold for scaling up resolution (default: 58) */
  maxFPS: number;
  /** Minimum allowed DPR - lower bound before image becomes too pixelated (default: 0.5) */
  minDPR: number;
  /** Factor to multiply DPR when scaling down (default: 0.9 = 10% reduction) */
  scaleDownFactor: number;
  /** Factor to multiply DPR when scaling up (default: 1.05 = 5% increase) */
  scaleUpFactor: number;
  /** Seconds FPS must stay above maxFPS before scaling up (default: 3) */
  hysteresisSeconds: number;
  /** How often to evaluate FPS and adjust DPR in milliseconds (default: 500) */
  evaluationIntervalMs: number;
}

/**
 * 3D scene visual configuration
 */
export interface SceneConfig {
  backgroundColor: number;
  defaultFitRatio: number;
}

/**
 * Shader configuration for point rendering
 */
export interface ShaderConfig {
  points: {
    hdrMultiplier: number;
    baseAlpha: number;
  };
}

/**
 * Post-processing pipeline configuration
 */
export interface PostProcessingConfig {
  hdr: {
    renderTargetType: typeof HalfFloatType;
  };
  // Note: bloom configuration moved to RenderingSettings for centralization
  toneMapping: {
    initial: {
      outputColorSpace: typeof LinearSRGBColorSpace;
      toneMapping: typeof NoToneMapping;
    };
    final: {
      outputColorSpace: typeof SRGBColorSpace;
      toneMapping: typeof ACESFilmicToneMapping;
    };
  };
}

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
 * Control system configuration
 */
export interface ControlsConfig {
  fly: FlyControlsConfig;
  orbit: OrbitControlsConfig;
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
      toggleDebugConsole: string;
      recenterCamera: string;
      toggleControlMode: string;
      toggleInertialMode: string;
      toggleCinematicMode: string;
    };
    flyModeKeys: string[];
    dimensionKeys: string[];
  };
  mouse: {
    doubleClickDelay: number;
  };
}

// NOTE: UIColors, UITypography, UISpacing, UIEffects, and UIStyles interfaces
// have been removed. All styling now uses CSS variables and classes in
// src/styles/ (see theming system in src/themes/)

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
    zIndex: number;
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
    zIndex: number;
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
  // NOTE: styles property removed - all styling now uses CSS variables
  debugConsole: DebugConsoleConfig;
  components: UIComponentsConfig;
}

/**
 * Data loading network configuration
 */
export interface DataLoadingNetworkConfig {
  timeoutMs: number;
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
 * Performance optimization configuration (Phases 1-4)
 */
export interface DataLoadingPerformanceConfig {
  // Object pooling (Phase 1)
  useAccumulators: boolean;
  initialAccumulatorCapacity: number;
  accumulatorGrowthFactor: number;

  // Web Workers (Phase 2)
  useWebWorkers: boolean;
  workerCount: number;

  // WASM acceleration (Phase 3)
  useWASM: boolean;
  wasmModulePath: string;

  // GPU buffer pool (Phase 4)
  useGPUBufferPool: boolean;
  gpuPoolMaxSize: number;
  gpuPoolEvictionFrames: number;

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
  clippingAdaptSpeed: number;
  // Rendering effects (bloom is now the single source of truth)
  bloomThreshold: number;
  bloomStrength: number;
  bloomRadius: number;
  bloomLevels: number;
  hdrMultiplier: number;
  fxaaEnabled: boolean;
  msaaEnabled: boolean;
  msaaSamples: number;
  smaaEnabled: boolean;
  smaaThreshold: number;
  smaaSearchSteps: number;
  ssaaEnabled: boolean;
  ssaaMultiplier: number;
  // Post-processing effects
  toneMapping: 'None' | 'Linear' | 'Reinhard' | 'Cineon' | 'ACES' | 'AgX' | 'Neutral';
  dofEnabled: boolean;
  dofFocus: number;
  dofStrength: number;
  // New pmndrs effects
  aoEnabled: boolean;
  aoQuality: 'low' | 'medium' | 'high' | 'ultra';
  vignetteEnabled: boolean;
  vignetteDarkness: number;
  vignetteOffset: number;
  // Detector noise effect (physics-based: Poisson + Gaussian + FPN)
  detectorNoiseEnabled: boolean;
  detectorNoiseReadoutSigma: number;
  detectorNoisePhotonGain: number;
  detectorNoiseFpnSigma: number;
  // Chromatic lens distortion effect (replaces old separate lens distortion + chromatic aberration)
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
  controlType: 'orbit' | 'arcball' | 'fly';
  autoRotate: boolean;
  autoRotateSpeed: number;
  // Fly controls - these are added at runtime from config.controls.fly
  flyMovementSpeed?: number;
  flyRotationSpeed?: number;
  flyInertialMode?: boolean;
  flyDamping?: number;
  flyRotationDamping?: number;
  // Adaptive resolution
  adaptiveDPREnabled: boolean;
  // Cinematic mode checkbox (multi-effect toggle)
  cinematicMode: boolean;
}

/**
 * Rendering controls configuration
 */
export interface RenderingControlsConfig {
  defaults: RenderingSettings;
}

// Note: RenderingConfig removed - all rendering settings moved to RenderingSettings for centralization

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
 * THREE.WebGLRenderer configuration
 */
export interface WebGLRendererConfig {
  antialias: boolean;
  powerPreference: 'high-performance' | 'low-power' | 'default';
  preserveDrawingBuffer: boolean;
  logarithmicDepthBuffer: boolean;
  precision: 'highp' | 'mediump' | 'lowp';
  premultipliedAlpha: boolean;
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
 * WebGL performance profile
 */
export interface WebGLPerformanceProfile {
  powerPreference: 'high-performance' | 'low-power' | 'default';
  antialias: boolean;
  precision: 'highp' | 'mediump' | 'lowp';
}

/**
 * WebGL configuration
 */
export interface WebGLConfig {
  context: WebGLContextAttributes;
  renderer: WebGLRendererConfig;
  renderTarget: WebGLRenderTargetConfig;
  profiles: {
    quality: WebGLPerformanceProfile;
    balanced: WebGLPerformanceProfile;
    performance: WebGLPerformanceProfile;
  };
}

/**
 * OPFS-based zarr cache configuration
 */
export interface CacheConfig {
  /** Enable OPFS caching (default: true) */
  enabled: boolean;
  /** L1 memory cache size in MB (default: 100) */
  l1MaxSizeMB: number;
  /** L2 OPFS cache size in MB (default: 2048) */
  l2MaxSizeMB: number;
  /** Enable cache debug logging (default: false) */
  debug: boolean;
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
  postProcessing: PostProcessingConfig;
  ui: UIConfig;
  renderingControls: RenderingControlsConfig;
  controls: ControlsConfig;
  input: InputConfig;
  dataLoading: DataLoadingConfig;
  webgl: WebGLConfig;
  cache: CacheConfig;
  dimensionAnimation: DimensionAnimationConfig;
  defaultZarrPath: string;
  canvasId: string;
}
