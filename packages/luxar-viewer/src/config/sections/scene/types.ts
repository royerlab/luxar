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
  points: Record<string, never>;
}
