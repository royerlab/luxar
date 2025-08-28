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
  camera: {
    fov: 60, // Field of view in degrees - 60° provides natural human-like viewing angle
    near: 0.1, // Near clipping plane distance - objects closer than this are not rendered
    far: 1000, // Far clipping plane distance - objects further than this are not rendered
    initialPosition: { x: 0, y: 0, z: 8 }, // Initial camera position in 3D space (world coordinates)
    fovMin: 10, // Minimum field of view for zoom limits - prevents excessive zoom-in
    fovMax: 200, // Maximum field of view for zoom limits - prevents excessive zoom-out
    fovSensitivity: 0.05, // FOV change sensitivity for Shift+wheel input - lower = finer control
  },

  // Animation loop and performance optimization settings
  animation: {
    idleTimeoutMs: 2000, // Time in milliseconds before pausing animation when idle - saves power
    targetFPS: 60, // Target frames per second
    minFPS: 30, // Minimum acceptable FPS before quality reduction
  },

  // 3D scene visual configuration
  scene: {
    backgroundColor: 0x111111, // Background color in hexadecimal - dark gray for good contrast with point clouds
    defaultFitRatio: 0.75, // How much of view to fill when fitting to bounds (0-1)
  },

  // HDR post-processing and rendering configuration
  rendering: {
    hdrEnabled: true, // Enable HDR post-processing pipeline with bloom effects

    // Unified bloom configuration - single source of truth
    bloom: {
      strength: 0.25, // Bloom intensity - how strong the glow effect appears
      radius: 1.0, // Bloom radius - how far the glow spreads from bright areas
      threshold: 0.01, // Bloom threshold - brightness level required to trigger bloom
      levels: 8, // Number of mipmap levels (1-12, lower = coarser/faster, higher = smoother)
    },
    // Note: Point rendering settings moved to shader.points to avoid duplication
  },

  // Shader configuration for point rendering
  shader: {
    points: {
      hdrMultiplier: 16.0, // HDR color multiplier for bloom effects
      baseAlpha: 0.01, // Base alpha intensity
    },
  },

  // Post-processing pipeline configuration
  postProcessing: {
    hdr: {
      renderTargetType: THREE.HalfFloatType, // Use 16-bit float for HDR precision without banding
    },
    // Note: bloom settings moved to rendering.bloom for consolidation

    toneMapping: {
      initial: {
        outputColorSpace: THREE.LinearSRGBColorSpace,
        toneMapping: THREE.NoToneMapping,
      },
      final: {
        outputColorSpace: THREE.SRGBColorSpace,
        toneMapping: THREE.ACESFilmicToneMapping,
      },
    },
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
    // Style system (migrated from data-loading-monitor-styles.ts)
    styles: {
      colors: {
        // Semantic colors
        success: '#4CAF50',
        warning: '#FFC107',
        error: '#f44336',
        info: '#2196F3',
        secondary: '#9C27B0',
        // Text colors
        primaryText: '#e0e0e0',
        secondaryText: '#888',
        muted: 'rgba(255, 255, 255, 0.6)',
        dimmed: 'rgba(255, 255, 255, 0.4)',
        // Background colors
        panelBg: 'rgba(30, 30, 30, 0.95)',
        sectionBg: 'rgba(0, 0, 0, 0.3)',
        hoverBg: 'rgba(40, 40, 40, 0.9)',
        // Cache visualization
        cacheHot: '#ff6b6b',
        cacheWarm: '#FFC107',
        cacheCold: '#4CAF50',
        // Grid/separator colors
        separator: 'rgba(255, 255, 255, 0.1)',
        separatorStrong: 'rgba(255, 255, 255, 0.2)',
      },
      typography: {
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif',
        fontFamilyMono: 'monospace',
        // Font sizes
        title: { fontSize: '14px', fontWeight: 'bold' },
        sectionHeader: { fontSize: '12px', fontWeight: 600 },
        body: { fontSize: '11px' },
        small: { fontSize: '10px' },
        tiny: { fontSize: '9px' },
        // Line heights
        compact: { lineHeight: 1.2 },
        normal: { lineHeight: 1.4 },
        relaxed: { lineHeight: 1.6 },
      },
      spacing: {
        // Panel spacing
        panelPadding: 15,
        panelMargin: 20,
        // Section spacing
        sectionPadding: 10,
        sectionGap: 15,
        // Element spacing
        elementGap: 8,
        compactGap: 5,
        tinyGap: 2,
        // Border and separator spacing
        borderPadding: 6,
      },
      effects: {
        // Backdrop and shadows
        backdropBlur: 'blur(10px)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        boxShadowStrong: '0 8px 32px rgba(0, 0, 0, 0.4)',
        // Border radius
        borderRadius: 8,
        borderRadiusSmall: 4,
        borderRadiusLarge: 12,
        // Transitions
        transition: 'all 0.2s ease',
        transitionFast: 'all 0.1s ease',
        transitionSlow: 'all 0.3s ease',
      },
    },
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
    // UI component-specific configuration for consistent styling
    components: {
      datasetBrowser: {
        zIndex: 1000,
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
        zIndex: 150,
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
      // Bloom settings
      bloomThreshold: 0.01, // Luminance threshold (0-1), lower = more bloom, higher = less bloom
      bloomStrength: 0.5, // Bloom intensity multiplier
      bloomRadius: 0.6, // Blur radius for bloom spread (in mipmap blur units)
      bloomLevels: 8, // Number of mipmap levels (1-12, lower = coarser/faster, higher = smoother)
      exposure: 1.0, // Tone mapping exposure value
      hdrMultiplier: 16.0, // HDR intensity multiplier
      fxaaEnabled: false, // FXAA disabled by default
      msaaEnabled: false, // MSAA disabled by default (incompatible with additive blending)
      msaaSamples: 4, // MSAA sample count (2, 4, 8)
      smaaEnabled: false, // SMAA disabled by default
      smaaThreshold: 0.1, // SMAA edge detection threshold (0.05-0.2)
      smaaSearchSteps: 8, // SMAA search steps for pattern detection (4-32)
      ssaaEnabled: false, // SSAA disabled by default (has brightness issues with additive blending)
      ssaaMultiplier: 2.0, // SSAA resolution multiplier (1.5x, 2x, 4x)
      // New post-processing effects
      toneMapping: 'ACES' as const, // Tone mapping method
      dofEnabled: false, // Depth of field disabled by default
      dofFocus: 10, // DOF focus distance
      dofStrength: 0.5, // DOF blur strength (0-1)
      chromaticAberrationEnabled: false, // Chromatic aberration disabled by default
      chromaticAberrationStrength: 0.15, // Chromatic aberration strength - default as shown
      // New pmndrs effects
      aoEnabled: false, // Ambient occlusion disabled by default
      aoQuality: 'medium' as const, // AO quality level
      vignetteEnabled: false, // Vignette disabled by default
      vignetteDarkness: 0.5, // Vignette darkness (0-1)
      vignetteOffset: 0.5, // Vignette offset from center (0-1)
      // Navigation controls
      controlType: 'orbit' as const, // Default to orbit controls
      autoRotate: false, // Auto-rotation disabled by default
      autoRotateSpeed: 0.25, // Slow rotation speed for presentations
      // Note: Fly control settings are referenced directly from controls.fly to avoid duplication
    },
  },

  // Control system configuration (migrated from control-config.ts)
  controls: {
    fly: {
      inertialMode: {
        default: true,
      },
      movement: {
        speed: { min: 0.5, max: 50.0, default: 5.0, step: 0.1 },
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
        toggleDebugConsole: 'ctrl+l',
        recenterCamera: 'f',
        toggleControlMode: 'v',
        toggleInertialMode: 'i',
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
    cache: {
      maxSizeMB: 512,
      evictionStrategy: 'lru' as const,
      ttlMs: 300000,
    },
    network: {
      timeoutMs: 30000,
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

    // THREE.WebGLRenderer specific settings
    renderer: {
      antialias: true, // MSAA for smoother rendering
      powerPreference: 'high-performance' as const, // High performance GPU
      preserveDrawingBuffer: false, // Better performance
      logarithmicDepthBuffer: false, // Standard depth buffer (faster)
      precision: 'highp' as const, // High precision for better quality
      premultipliedAlpha: true, // Standard alpha blending
      shadowMap: {
        enabled: false, // No shadows needed for point clouds
        type: THREE.PCFSoftShadowMap, // Soft shadows if enabled
      },
    },

    // Render target configuration for post-processing
    renderTarget: {
      depthBuffer: true, // Needed for depth testing
      stencilBuffer: false, // Not needed, saves memory
      samples: 0, // MSAA samples (0 = disabled for additive blending compatibility)
    },

    // Performance profiles for different hardware/use cases
    profiles: {
      quality: {
        powerPreference: 'high-performance' as const,
        antialias: true,
        precision: 'highp' as const,
      },
      balanced: {
        powerPreference: 'default' as const,
        antialias: true,
        precision: 'mediump' as const,
      },
      performance: {
        powerPreference: 'low-power' as const,
        antialias: false,
        precision: 'lowp' as const,
      },
    },
  },

  // Default path to demo Zarr data when no source is specified
  defaultZarrPath: '/data/demo.zarr',

  // HTML element ID for the canvas where 3D rendering occurs
  canvasId: 'app',
} as const;

// Export types
export type { AppConfig, RenderingSettings } from './types';
