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

import type { DimensionAnimationConfig } from './sections/dimension-animation/types';
export type { DimensionAnimationConfig };

import type {
  WebGLConfig,
  WebGLContextAttributes,
  WebGLRendererConfig,
  WebGLRenderTargetConfig,
} from './sections/webgl/types';
export type { WebGLConfig, WebGLContextAttributes, WebGLRendererConfig, WebGLRenderTargetConfig };

import type { InputConfig } from './sections/input/types';
export type { InputConfig };

import type {
  ConfigRange,
  ControlsConfig,
  FlyControlsConfig,
  OrbitControlsConfig,
  ScaleMultipliers,
} from './sections/controls/types';
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
export type { RenderingControlsConfig, RenderingSettings };

import type { DebugConsoleConfig, UIComponentsConfig, UIConfig } from './sections/ui/types';
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
