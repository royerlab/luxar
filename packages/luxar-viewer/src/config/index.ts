// Main configuration file for the Luxar scene player
// This module centralizes ALL configuration values to ensure consistency
// and make the application easy to customize.

import * as THREE from 'three';
import type { AppConfig } from './types';

/**
 * Main configuration object containing all application settings
 *
 * Organized by functional area for easy navigation and maintenance.
 * Uses 'as const' to ensure TypeScript treats these as literal values
 * rather than generic types, enabling better type checking.
 */
export const config: AppConfig = {
  // Camera configuration for 3D perspective and navigation
  // Note: fov, near, far live in renderingControls.defaults as the single source of truth
  camera: {
    initialPosition: { x: 0, y: 0, z: 8 }, // Initial camera position in 3D space (world coordinates)
    fovMin: 10, // Minimum field of view for zoom limits - prevents excessive zoom-in
    fovMax: 170, // Maximum field of view for zoom limits - must be <180° (fish-eye territory)
    fovSensitivity: 0.05, // FOV change sensitivity for Ctrl+wheel input - lower = finer control
    // FOV presets based on 35mm equivalent focal lengths (horizontal FOV - photography standard)
    fovPresets: {
      '28mm Wide': 75, // Wide angle - 75° horizontal FOV, good for large scenes and landscapes
      '35mm': 63, // Wide normal - 63° horizontal FOV, comfortable wide viewing
      '50mm Normal': 47, // Normal lens - 47° horizontal FOV, closest to human vision
      '85mm Portrait': 29, // Portrait lens - 29° horizontal FOV, good for isolating subjects
      '135mm Tele': 18, // Telephoto - 18° horizontal FOV, extreme subject isolation
      Custom: -1, // Custom value - preserves current FOV slider setting
    },
    // Lens distortion presets matching realistic lens characteristics for each focal length
    // Negative distortion = barrel (wide angle), positive = pincushion (telephoto)
    // Dispersion values simulate chromatic aberration (wavelength-dependent refraction)
    lensDistortionPresets: {
      '28mm Wide': {
        distortionX: -0.07,
        distortionY: -0.07, // Barrel distortion from your screenshot
        principalPointX: 0,
        principalPointY: 0, // Centered (standard)
        focalLengthX: 1.075,
        focalLengthY: 1.08, // Focal length values from your screenshot
        skew: 0, // No skew (perfect optics)
        dispersion: 0.05, // Wide angle = more chromatic aberration (higher light bending angles)
      },
      '35mm': {
        distortionX: -0.05,
        distortionY: -0.05, // Moderate barrel distortion from your screenshot
        principalPointX: 0,
        principalPointY: 0,
        focalLengthX: 1.054,
        focalLengthY: 1.055, // Focal length values from your screenshot
        skew: 0,
        dispersion: 0.035, // Moderate chromatic aberration
      },
      '50mm Normal': {
        distortionX: 0,
        distortionY: 0, // No distortion (ideal normal lens)
        principalPointX: 0,
        principalPointY: 0,
        focalLengthX: 1,
        focalLengthY: 1,
        skew: 0,
        dispersion: 0.02, // Minimal chromatic aberration (normal focal length)
      },
      '85mm Portrait': {
        distortionX: 0.05,
        distortionY: 0.05, // Pincushion (telephoto) - similar magnitude to 35mm barrel
        principalPointX: 0,
        principalPointY: 0,
        focalLengthX: 0.91,
        focalLengthY: 0.91, // Slight compression (opposite of wide angle expansion)
        skew: 0,
        dispersion: 0.025, // Low chromatic aberration (longer focal length = less bending)
      },
      '135mm Tele': {
        distortionX: 0.07,
        distortionY: 0.07, // Barrel distortion from your screenshot
        principalPointX: 0,
        principalPointY: 0,
        focalLengthX: 0.882,
        focalLengthY: 0.883, // Focal length values from your screenshot
        skew: 0,
        dispersion: 0.03, // Telephoto with some chromatic aberration at edges
      },
    },
  },

  // Animation loop settings
  animation: {
    idleTimeoutMs: 2000, // Time in milliseconds before pausing animation when idle - saves power
  },

  // Adaptive pixel ratio configuration for dynamic performance optimization
  adaptiveDPR: {
    enabled: true, // Construction-time default; runtime toggle is renderingControls.defaults.adaptiveDPREnabled
    minFPS: 50, // FPS threshold for scaling down resolution
    maxFPS: 58, // FPS threshold for scaling up resolution
    minDPR: 0.5, // Minimum DPR - lower bound before image becomes too pixelated
    scaleDownFactor: 0.9, // Reduce DPR by 10% when scaling down
    scaleUpFactor: 1.05, // Increase DPR by 5% when scaling up
    hysteresisSeconds: 3, // Wait 3 seconds of stable high FPS before scaling up
    evaluationIntervalMs: 500, // Evaluate FPS every 500ms
  },

  // 3D scene visual configuration
  scene: {
    backgroundColor: 0x111111, // Background color in hexadecimal - dark gray for good contrast with points
    defaultFitRatio: 0.75, // How much of view to fill when fitting to bounds (0-1)
  },

  // Shader configuration for point rendering
  // Note: global exposure/offset/gamma live in renderingControls.defaults (applied in post-processing)
  shader: {
    points: {},
  },

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
    // NOTE: ui.styles section has been removed - all styling now uses CSS variables
    // and classes in src/styles/ (see theming system in src/themes/)

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
      smaaEnabled: false, // SMAA disabled by default
      smaaThreshold: 0.1, // SMAA edge detection threshold (0.05-0.2)
      smaaSearchSteps: 8, // SMAA search steps for pattern detection (4-32)
      ssaaEnabled: false, // SSAA disabled by default (highest quality, heavy performance cost)
      ssaaMultiplier: 2.0, // SSAA resolution multiplier (1.5x, 2x, 4x)
      // New post-processing effects
      toneMapping: 'Neutral' as const, // Tone mapping method (Neutral preserves hue fidelity for scientific data)
      dofEnabled: false, // Depth of field disabled by default
      dofFocus: 10, // DOF focus distance
      dofStrength: 0.5, // DOF blur strength (0-1)
      // New pmndrs effects
      aoEnabled: false, // Ambient occlusion disabled by default
      aoQuality: 'medium' as const, // AO quality level
      vignetteEnabled: false, // Vignette disabled by default
      vignetteDarkness: 0.5, // Vignette darkness (0-1)
      vignetteOffset: 0.5, // Vignette offset from center (0-1)
      // Detector noise effect settings (physics-based: Poisson + Gaussian + FPN)
      detectorNoiseEnabled: false, // Detector noise disabled by default
      detectorNoiseReadoutSigma: 0.002, // Temporal readout noise sigma (0-0.1)
      detectorNoisePhotonGain: 0.002, // Photon gain for shot noise visibility (0.0001-0.1)
      detectorNoiseFpnSigma: 0.001, // Fixed pattern noise sigma (0-0.05)
      // Chromatic lens distortion effect settings (replaces old separate lens distortion + chromatic aberration)
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

      // Debugging
      enablePerformanceMonitoring: false,
    },
  },

  // WebGL context and renderer configuration
  webgl: {
    // WebGL2 context attributes for canvas
    context: {
      alpha: false, // No transparency in canvas background
      antialias: true, // Enable antialiasing for smoother edges
      depth: true, // Enable depth buffer for 3D rendering
      stencil: false, // No stencil buffer needed (saves memory)
      powerPreference: 'high-performance' as const, // Request high-performance GPU
      colorSpace: 'display-p3', // Wide color gamut for better colors
      preserveDrawingBuffer: false, // Don't preserve buffer (better performance)
      desynchronized: true, // Better performance with async updates
      premultipliedAlpha: true, // Standard alpha blending
      failIfMajorPerformanceCaveat: false, // Don't fail on slow GPUs
    },

    // THREE.WebGLRenderer specific settings (renderer-only; shared attributes
    // like antialias, powerPreference, preserveDrawingBuffer, premultipliedAlpha
    // are sourced from webgl.context and spread at renderer creation time)
    renderer: {
      logarithmicDepthBuffer: false, // Standard depth buffer (faster)
      precision: 'highp' as const, // High precision for better quality
      shadowMap: {
        enabled: false, // No shadows needed for points
        type: THREE.PCFSoftShadowMap, // Soft shadows if enabled
      },
    },

    // Render target configuration for post-processing
    renderTarget: {
      depthBuffer: true, // Needed for depth testing
      stencilBuffer: false, // Not needed, saves memory
      samples: 0, // MSAA samples (0 = disabled for additive blending compatibility)
    },
  },

  // OPFS-based zarr cache configuration
  cache: {
    enabled: true,
    l0Enabled: true, // L0 decompressed chunk cache - eliminates ~2ms Blosc decompression per chunk
    l0MaxSizeMB: 200, // 200MB for decompressed chunks (5x larger than compressed, but instant access)
    l1MaxSizeMB: 100,
    l2MaxSizeMB: 2048,
    opfsOperationTimeoutMs: 10_000,
    externalDatasetTtlMs: null,
    debug: false,
  },

  // Dimension animation configuration
  dimensionAnimation: {
    defaults: {
      targetFPS: 10,
      loop: 'loop' as const,
      direction: 'forward' as const,
    },
    presets: {
      fps: [1, 2, 5, 10, 15, 30, 60],
      customMin: 0.1,
      customMax: 120,
    },
    timing: {
      minFrameTimeMs: 16, // ~60fps absolute max
      continuousTraverseSeconds: 10, // Full range in 10s for continuous dims
    },
    ui: {
      showFPSFeedback: true, // Show "target vs actual" fps
      feedbackThreshold: 0.8, // Warn if actual < 80% of target
    },
  },

  // Default path to demo Zarr data when no source is specified
  // Empty string = show dataset browser instead of attempting to load non-existent dataset
  defaultZarrPath: '',
} as const;

// Export types
export type { AppConfig, RenderingSettings } from './types';
