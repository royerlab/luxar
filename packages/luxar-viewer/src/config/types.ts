// Type definitions for ALL configuration interfaces
// This file is the single source of truth for configuration types

import type { CameraConfig } from './sections/camera/types';
/** Camera configuration type. */
export type { CameraConfig };

import type { AnimationConfig } from './sections/animation/types';
/** Animation configuration type. */
export type { AnimationConfig };

import type { SceneConfig, ShaderConfig } from './sections/scene/types';
/** Scene and shader configuration types. */
export type { SceneConfig, ShaderConfig };

import type { AdaptiveDPRConfig } from './sections/adaptive-dpr/types';
/** Adaptive device-pixel-ratio configuration type. */
export type { AdaptiveDPRConfig };

import type { CacheConfig } from './sections/cache/types';
/** Cache configuration type. */
export type { CacheConfig };

import type { DepthSortConfig } from './sections/depth-sort/types';
/** Depth-sort configuration type. */
export type { DepthSortConfig };

import type { DimensionAnimationConfig } from './sections/dimension-animation/types';
/** Dimension-animation configuration type. */
export type { DimensionAnimationConfig };

import type {
  WebGLConfig,
  WebGLContextAttributes,
  WebGLRendererConfig,
  WebGLRenderTargetConfig,
} from './sections/webgl/types';
/** WebGL renderer, context-attributes, and render-target configuration types. */
export type { WebGLConfig, WebGLContextAttributes, WebGLRendererConfig, WebGLRenderTargetConfig };

import type { InputConfig } from './sections/input/types';
/** Input (keyboard/mouse) configuration type. */
export type { InputConfig };

import type {
  ConfigRange,
  ControlsConfig,
  FlyControlsConfig,
  OrbitControlsConfig,
  ScaleMultipliers,
} from './sections/controls/types';
/** Camera-controls configuration types (orbit/fly controls, ranges, scale multipliers). */
export type {
  ConfigRange,
  ControlsConfig,
  FlyControlsConfig,
  OrbitControlsConfig,
  ScaleMultipliers,
};

import type {
  RenderingControlsConfig,
  RenderingSettings,
} from './sections/rendering-controls/types';
/** Rendering-controls configuration and per-scene rendering-settings types. */
export type { RenderingControlsConfig, RenderingSettings };

import type { DebugConsoleConfig, UIComponentsConfig, UIConfig } from './sections/ui/types';
/** UI, UI-components, and debug-console configuration types. */
export type { DebugConsoleConfig, UIComponentsConfig, UIConfig };

import type {
  DataLoadingConfig,
  DataLoadingMemoryConfig,
  DataLoadingMonitorConfig,
  DataLoadingNetworkConfig,
  DataLoadingPerformanceConfig,
  DataLoadingSpatialConfig,
  MonitorLimits,
  MonitorThresholds,
  MonitorTimings,
} from './sections/data-loading/types';
/** Data-loading configuration types (memory, network, spatial, monitor, performance). */
export type {
  DataLoadingConfig,
  DataLoadingMemoryConfig,
  DataLoadingMonitorConfig,
  DataLoadingNetworkConfig,
  DataLoadingPerformanceConfig,
  DataLoadingSpatialConfig,
  MonitorLimits,
  MonitorThresholds,
  MonitorTimings,
};

/**
 * Complete application configuration structure
 */
export interface AppConfig {
  camera: CameraConfig;
  animation: AnimationConfig;
  adaptiveDPR: AdaptiveDPRConfig;
  depthSort: DepthSortConfig;
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
