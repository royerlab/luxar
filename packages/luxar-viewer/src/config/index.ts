// Main configuration file for the Luxar scene player
// This module centralizes ALL configuration values to ensure consistency
// and make the application easy to customize.

import type { AppConfig } from './types';
import { cameraConfig } from './sections/camera/data';
import { animationConfig } from './sections/animation/data';
import { sceneConfig, shaderConfig } from './sections/scene/data';
import { adaptiveDPRConfig } from './sections/adaptive-dpr/data';
import { depthSortConfig } from './sections/depth-sort/data';
import { cacheConfig } from './sections/cache/data';
import { dimensionAnimationConfig } from './sections/dimension-animation/data';
import { webglConfig } from './sections/webgl/data';
import { inputConfig } from './sections/input/data';
import { controlsConfig } from './sections/controls/data';
import { renderingControlsConfig } from './sections/rendering-controls/data';
import { uiConfig } from './sections/ui/data';
import { dataLoadingConfig } from './sections/data-loading/data';

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

  depthSort: depthSortConfig,

  scene: sceneConfig,

  shader: shaderConfig,

  ui: uiConfig,

  renderingControls: renderingControlsConfig,

  controls: controlsConfig,

  input: inputConfig,

  dataLoading: dataLoadingConfig,

  webgl: webglConfig,

  cache: cacheConfig,

  dimensionAnimation: dimensionAnimationConfig,

  // Default path to demo Zarr data when no source is specified
  // Empty string = show dataset browser instead of attempting to load non-existent dataset
  defaultZarrPath: '',
} as const;

/** Re-exported config types for consumers importing from `config`. */
export type { AppConfig, RenderingSettings } from './types';
