// Main configuration file for the Luxar scene player
// This module centralizes ALL configuration values to ensure consistency
// and make the application easy to customize.

import type { AppConfig } from './types';
import { isMacPlatform } from '../utils/platform';
import { cameraConfig } from './sections/camera/data';
import { animationConfig } from './sections/animation/data';
import { sceneConfig, shaderConfig } from './sections/scene/data';
import { adaptiveDPRConfig } from './sections/adaptive-dpr/data';
import { cacheConfig } from './sections/cache/data';
import { dimensionAnimationConfig } from './sections/dimension-animation/data';
import { webglConfig } from './sections/webgl/data';

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

  // UI configuration for overlays and visual elements
  ui: {
    zIndex: {
      // Base layer components (100-199)
      dimensionSliders: 100, // Dimension sliders at bottom
      performanceMonitor: 100, // Performance stats panel
      debugConsole: 150, // Debug console (slightly above base)

      // Mid-layer overlays (1000-1999)
      datasetBrowser: 1000, // Dataset browser modal
      loading: 1000, // Loading indicator
      error: 1000, // Error messages
      help: 1001, // Help overlay (above errors)
      renderingControls: 1999, // Rendering controls panel (top of mid-layer)
      recordingPanel: 1500, // Recording panel (screenshot/video capture)
      layersPanel: 1500, // Layers panel (per-node controls)

      // Top layer (2000+)
      statsMonitor: 2000, // Three.js stats monitor (always on top)
    },
    timings: {
      errorAutoDismissMs: 10000, // Auto-dismiss error messages after 10s
      helpClickDelayMs: 100, // Delay before help can be closed by click
    },
    spinner: {
      size: 24, // Loading spinner size in pixels
      borderWidth: 3, // Spinner border width
    },
    // Styling is provided by CSS variables and classes in src/styles/
    // (see the theming system in src/themes/).

    // Debug console configuration (migrated from debug-console.ts)
    debugConsole: {
      panel: {
        defaultWidth: 600,
        defaultHeight: 400,
        minWidth: 400,
        maxWidth: 1200,
        minHeight: 200,
        maxHeight: 800,
        bottomOffset: 20,
        leftOffset: 20,
      },
      interceptor: {
        maxBufferSize: 10000,
      },
      resize: {
        borderWidth: 4,
      },
      style: {
        backgroundColor: 'rgba(30, 30, 30, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderRadius: 8,
        backdropBlur: 10,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
      },
    },
    scaleBar: {
      targetWidthPx: 150,
      position: 'bottom-left' as const,
    },
    // UI component-specific configuration for consistent styling
    components: {
      datasetBrowser: {
        borderRadius: {
          panel: 12,
          section: 6,
          element: 4,
        },
        padding: {
          panel: 20,
          section: 15,
          element: 10,
        },
      },
      debugConsole: {
        borderRadius: {
          header: 8,
          content: 4,
          button: 3,
        },
      },
      renderingControls: {
        borderRadius: {
          checkbox: 4,
          section: 4,
          header: 4,
        },
      },
      dataMonitor: {
        borderRadius: {
          card: 4,
          section: 6,
        },
        padding: {
          default: 10,
          compact: 8,
        },
      },
    },
  },

  // Rendering controls configuration with user-adjustable defaults
  renderingControls: {
    defaults: {
      // Camera settings
      fov: 47, // Field of view in degrees (50mm Normal, default matches camera.fov)
      fovPreset: '50mm Normal', // Default to normal lens equivalent
      near: 0.1, // Near clipping plane (default matches camera.near)
      far: 1000, // Far clipping plane (default matches camera.far)
      // Dynamic clipping planes
      dynamicClippingEnabled: true, // Auto-adjust clipping planes based on camera position
      // Bloom settings - single source of truth (moved from config.rendering.bloom)
      bloomEnabled: false, // Enable/disable bloom effect (opt-in via zarr viewer_config)
      bloomThreshold: 0.01, // Luminance threshold (0-1), lower = more bloom, higher = less bloom
      bloomStrength: 0.25, // Bloom intensity multiplier (moved from rendering.bloom)
      bloomRadius: 1.0, // Blur radius for bloom spread (moved from rendering.bloom)
      bloomLevels: 8, // Number of mipmap levels (1-12, lower = coarser/faster, higher = smoother)
      exposure: 0.0, // Global exposure in log2 stops (0 = neutral, +1 = 2x brighter)
      globalOffset: 0.0, // Global additive brightness shift
      globalGamma: 1.0, // Global gamma correction (1.0 = linear)
      fxaaEnabled: false, // FXAA disabled by default
      msaaEnabled: false, // MSAA disabled by default (enable for fast hardware-accelerated AA)
      msaaSamples: 4, // MSAA sample count (2, 4, 8)
      ssaaEnabled: false, // SSAA disabled by default (highest quality, heavy performance cost)
      ssaaMultiplier: 2.0, // SSAA resolution multiplier (1.5x, 2x, 4x)
      // Tone mapping (Neutral preserves hue fidelity for scientific data)
      toneMapping: 'Neutral' as const,
      vignetteEnabled: false, // Vignette disabled by default
      vignetteDarkness: 0.5, // Vignette darkness (0-1)
      vignetteOffset: 0.5, // Vignette offset from center (0-1)
      // Detector noise effect settings (physics-based: Poisson + Gaussian + FPN)
      detectorNoiseEnabled: false, // Detector noise disabled by default
      detectorNoiseReadoutSigma: 0.002, // Temporal readout noise sigma (0-0.1)
      detectorNoisePhotonGain: 0.002, // Photon gain for shot noise visibility (0.0001-0.1)
      detectorNoiseFpnSigma: 0.001, // Fixed pattern noise sigma (0-0.05)
      // Chromatic lens distortion effect settings
      chromaticLensDistortionEnabled: false, // Chromatic lens distortion disabled by default
      chromaticLensDistortionX: 0, // Radial distortion coefficient X (50mm Normal: no distortion)
      chromaticLensDistortionY: 0, // Radial distortion coefficient Y (50mm Normal: no distortion)
      chromaticLensDispersion: 0.02, // Chromatic dispersion strength (50mm Normal: minimal)
      chromaticLensPrincipalPointX: 0, // Principal point offset X
      chromaticLensPrincipalPointY: 0, // Principal point offset Y
      chromaticLensFocalLengthX: 1.0, // Focal length X (50mm Normal: neutral)
      chromaticLensFocalLengthY: 1.0, // Focal length Y (50mm Normal: neutral)
      chromaticLensSkew: 0, // Skew in radians
      // Navigation controls
      controlType: 'orbit' as const, // Default to orbit controls
      autoRotate: false, // Auto-rotation disabled by default
      autoRotateSpeed: 0.25, // Slow rotation speed for presentations
      // Touchpad-friendly orbit drag mapping (LEFT=rotate, RIGHT=pan).
      // Default-on for Mac users; off elsewhere. The rendering-controls
      // persistence layer overrides this with the user's stored choice.
      naturalDrag: isMacPlatform(),
      // Note: Fly control settings are referenced directly from controls.fly to avoid duplication
      // Adaptive resolution (runtime/UI toggle; overrides adaptiveDPR.enabled after init)
      adaptiveDPREnabled: true, // Persisted per-scene via localStorage
      // Cinematic mode (disabled by default)
      cinematicMode: false,
    },
  },

  // Control system configuration (migrated from control-config.ts)
  controls: {
    scaleMultipliers: {
      minDistanceFactor: 0.01,
      maxDistanceFactor: 100,
      flySpeedFactor: 0.05,
    },
    fly: {
      inertialMode: {
        default: true,
      },
      movement: {
        speed: { min: 0.01, max: 5.0, default: 0.5, step: 0.01 },
        acceleration: { min: 0.1, max: 2.0, default: 0.5, step: 0.1 },
        damping: { min: 0.9, max: 0.99999, default: 0.999, step: 0.0001 },
      },
      rotation: {
        speed: { min: 0.1, max: 5.0, default: 1.5, step: 0.1 },
        damping: { min: 0.9, max: 0.9999, default: 0.99, step: 0.0001 },
      },
      look: {
        mouseSpeed: { default: 0.002 },
      },
      physics: {
        velocityThreshold: 1e-4,
        dampingPower: 60,
        angularVelocityThreshold: 1e-4,
      },
    },
    orbit: {
      autoRotate: {
        speed: { min: 0.1, max: 5.0, default: 0.25, step: 0.1 },
      },
      zoom: {
        minDistance: 0.1,
        maxDistance: 1000,
        speed: { min: 0.5, max: 2.0, default: 1.0, step: 0.1 },
      },
      damping: {
        enabled: true,
        factor: { min: 0.01, max: 0.3, default: 0.05, step: 0.01 },
      },
    },
  },

  // Input handling configuration (migrated from control-config.ts)
  input: {
    defaultSensitivity: 0.1, // Default input sensitivity for adjustments
    keyboard: {
      shortcuts: {
        toggleFullscreen: ' ',
        toggleHelp: 'h',
        toggleDimensions: 'n',
        toggleDatasetBrowser: 'o',
        togglePerformance: 'p',
        toggleRendering: 'r',
        toggleScaleBar: 'b',
        toggleColormapLegend: 'j',
        toggleDebugConsole: 'ctrl+l',
        toggleLayers: 'l',
        recenterCamera: 'f',
        toggleControlMode: 'v',
        toggleInertialMode: 'i',
        toggleCinematicMode: 'c',
        toggleOverlays: 'u',
      },
      flyModeKeys: [
        'w',
        'a',
        's',
        'd',
        'q',
        'e',
        'W',
        'A',
        'S',
        'D',
        'Q',
        'E',
        'Shift',
        'ArrowUp',
        'ArrowDown',
        'ArrowLeft',
        'ArrowRight',
      ],
      dimensionKeys: ['[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    },
    mouse: {
      doubleClickDelay: 300,
    },
  },

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
