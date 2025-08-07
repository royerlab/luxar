// Type definitions for all configuration interfaces
// This file defines the structure of the application configuration

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
}

/**
 * 3D scene visual configuration
 */
export interface SceneConfig {
  backgroundColor: number;
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
  bloom: {
    threshold: number;
    strength: number;
    radius: number;
    resolutionScale: number;
  };
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
 * UI configuration for overlays and visual elements
 */
export interface UIConfig {
  zIndex: {
    loading: number;
    error: number;
    help: number;
  };
  timings: {
    errorAutoDismissMs: number;
    helpClickDelayMs: number;
  };
  spinner: {
    size: number;
    borderWidth: number;
  };
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
}

/**
 * Rendering controls configuration
 */
export interface RenderingControlsConfig {
  defaults: RenderingSettings;
}

/**
 * Main rendering configuration
 */
export interface RenderingConfig {
  hdrEnabled: boolean;
  bloom: {
    strength: number;
    radius: number;
    threshold: number;
    resolutionScale: number;
  };
  points: {
    size: number;
    hdrMultiplier: number;
    baseAlpha: number;
    falloffSteepness: number;
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
  defaultZarrPath: string;
  canvasId: string;
}
