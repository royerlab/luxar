// Main configuration file for the Luxar scene player
// This module centralizes all configuration values to ensure consistency
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
  },

  // 3D scene visual configuration
  scene: {
    backgroundColor: 0x111111, // Background color in hexadecimal - dark gray for good contrast with point clouds
  },

  // HDR post-processing and rendering configuration
  rendering: {
    hdrEnabled: true, // Enable HDR post-processing pipeline with bloom effects

    // Bloom effect settings
    bloom: {
      strength: 0.1, // Bloom intensity - how strong the glow effect appears
      radius: 0.5, // Bloom radius - how far the glow spreads from bright areas
      threshold: 0.0, // Bloom threshold - brightness level required to trigger bloom (0.0 = everything glows)
      resolutionScale: 4, // Resolution scale for bloom pass - higher = faster but lower quality
    },

    // Advanced point rendering settings
    points: {
      size: 4.0, // Base point size in screen pixels
      hdrMultiplier: 13.0, // HDR color multiplier for driving bloom effects
      baseAlpha: 0.01, // Base alpha intensity for point visibility
      falloffSteepness: 40.0, // Gaussian falloff steepness for smooth point edges
    },
  },

  // Shader configuration for point rendering
  shader: {
    points: {
      size: 8.0, // Base point size in screen pixels
      hdrMultiplier: 13.0, // HDR color multiplier for bloom effects
      baseAlpha: 0.01, // Base alpha intensity
      falloffSteepness: 20.0, // Gaussian falloff steepness
    },
  },

  // Post-processing pipeline configuration
  postProcessing: {
    hdr: {
      renderTargetType: THREE.HalfFloatType, // Use 16-bit float for HDR precision without banding
    },

    bloom: {
      threshold: 0.01, // Bloom threshold - 0.0 means everything blooms, higher = only bright areas
      strength: 0.1, // Bloom strength - controls intensity of glow effect
      radius: 0.5, // Bloom radius - controls how far the glow spreads
      resolutionScale: 4, // Resolution divisor for bloom pass - higher = faster but lower quality
    },

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
      loading: 1000, // Loading indicator z-index
      error: 1000, // Error message z-index
      help: 1001, // Help overlay z-index
    },
    timings: {
      errorAutoDismissMs: 10000, // Auto-dismiss error messages after 10s
      helpClickDelayMs: 100, // Delay before help can be closed by click
    },
    spinner: {
      size: 24, // Loading spinner size in pixels
      borderWidth: 3, // Spinner border width
    },
  },

  // Rendering controls configuration with user-adjustable defaults
  renderingControls: {
    defaults: {
      bloomThreshold: 0.01, // Bloom threshold for rendering controls
      bloomStrength: 0.1, // Bloom strength for rendering controls
      bloomRadius: 0.5, // Bloom radius for rendering controls
      exposure: 1.0, // Tone mapping exposure value
      hdrMultiplier: 13.0, // HDR intensity multiplier
      fxaaEnabled: true, // FXAA enabled by default (works well with additive blending)
      msaaEnabled: false, // MSAA disabled by default (incompatible with additive blending)
      msaaSamples: 4, // MSAA sample count (2, 4, 8)
      smaaEnabled: false, // SMAA disabled by default
      smaaThreshold: 0.1, // SMAA edge detection threshold (0.05-0.2)
      smaaSearchSteps: 8, // SMAA search steps for pattern detection (4-32)
      ssaaEnabled: false, // SSAA disabled by default (has brightness issues with additive blending)
      ssaaMultiplier: 2.0, // SSAA resolution multiplier (1.5x, 2x, 4x)
    },
  },

  // Default path to demo Zarr data when no source is specified
  defaultZarrPath: '/data/demo.zarr',

  // HTML element ID for the canvas where 3D rendering occurs
  canvasId: 'app',
} as const;

// Export types
export type { AppConfig, RenderingSettings } from './types';
