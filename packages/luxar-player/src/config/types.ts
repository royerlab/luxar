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
    size: number;
    hdrMultiplier: number;
    baseAlpha: number;
    falloffSteepness: number;
  };
}

/**
 * Post-processing pipeline configuration
 */
export interface PostProcessingConfig {
  hdr: {
    renderTargetType: typeof HalfFloatType;
  };
  // Note: bloom configuration moved to RenderingConfig.bloom for consolidation
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
    };
    flyModeKeys: string[];
    dimensionKeys: string[];
  };
  mouse: {
    doubleClickDelay: number;
  };
}

/**
 * UI style colors
 */
export interface UIColors {
  // Semantic colors
  success: string;
  warning: string;
  error: string;
  info: string;
  secondary: string;
  // Text colors
  primaryText: string;
  secondaryText: string;
  muted: string;
  dimmed: string;
  // Background colors
  panelBg: string;
  sectionBg: string;
  hoverBg: string;
  // Cache visualization
  cacheHot: string;
  cacheWarm: string;
  cacheCold: string;
  // Grid/separator colors
  separator: string;
  separatorStrong: string;
}

/**
 * UI typography settings
 */
export interface UITypography {
  fontFamily: string;
  fontFamilyMono: string;
  // Font sizes
  title: { fontSize: string; fontWeight: string | number };
  sectionHeader: { fontSize: string; fontWeight: number };
  body: { fontSize: string };
  small: { fontSize: string };
  tiny: { fontSize: string };
  // Line heights
  compact: { lineHeight: number };
  normal: { lineHeight: number };
  relaxed: { lineHeight: number };
}

/**
 * UI spacing settings
 */
export interface UISpacing {
  // Panel spacing
  panelPadding: number;
  panelMargin: number;
  // Section spacing
  sectionPadding: number;
  sectionGap: number;
  // Element spacing
  elementGap: number;
  compactGap: number;
  tinyGap: number;
  // Border and separator spacing
  borderPadding: number;
}

/**
 * UI visual effects
 */
export interface UIEffects {
  // Backdrop and shadows
  backdropBlur: string;
  boxShadow: string;
  boxShadowStrong: string;
  // Border radius
  borderRadius: number;
  borderRadiusSmall: number;
  borderRadiusLarge: number;
  // Transitions
  transition: string;
  transitionFast: string;
  transitionSlow: string;
}

/**
 * UI styles configuration
 */
export interface UIStyles {
  colors: UIColors;
  typography: UITypography;
  spacing: UISpacing;
  effects: UIEffects;
}

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
  styles: UIStyles;
  debugConsole: DebugConsoleConfig;
  components: UIComponentsConfig;
}

/**
 * Data loading cache configuration
 */
export interface DataLoadingCacheConfig {
  maxSizeMB: number;
  evictionStrategy: 'lru' | 'lfu' | 'fifo';
  ttlMs: number;
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
  cache: DataLoadingCacheConfig;
  network: DataLoadingNetworkConfig;
  memory: DataLoadingMemoryConfig;
  monitor: DataLoadingMonitorConfig;
}

/**
 * User-adjustable rendering settings that can be persisted
 */
export interface RenderingSettings {
  bloomThreshold: number;
  bloomStrength: number;
  bloomRadius: number;
  exposure: number;
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
  chromaticAberrationEnabled: boolean;
  chromaticAberrationStrength: number;
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
}

/**
 * Rendering controls configuration
 */
export interface RenderingControlsConfig {
  defaults: RenderingSettings;
}

/**
 * Bloom configuration - single source of truth
 */
export interface BloomConfig {
  strength: number;
  radius: number;
  threshold: number;
  resolutionScale: number;
}

/**
 * Main rendering configuration
 */
export interface RenderingConfig {
  hdrEnabled: boolean;
  bloom: BloomConfig;
  // Note: points configuration moved to ShaderConfig to avoid duplication
}

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
 * Complete application configuration structure
 */
export interface AppConfig {
  camera: CameraConfig;
  animation: AnimationConfig;
  scene: SceneConfig;
  rendering: RenderingConfig;
  shader: ShaderConfig;
  postProcessing: PostProcessingConfig;
  ui: UIConfig;
  renderingControls: RenderingControlsConfig;
  controls: ControlsConfig;
  input: InputConfig;
  dataLoading: DataLoadingConfig;
  webgl: WebGLConfig;
  defaultZarrPath: string;
  canvasId: string;
}
