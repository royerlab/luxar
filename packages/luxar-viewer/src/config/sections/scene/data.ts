import type { SceneConfig, ShaderConfig } from './types';

/**
 * 3D scene visual configuration
 */
export const sceneConfig: SceneConfig = {
  backgroundColor: 0x111111, // Background color in hexadecimal - dark gray for good contrast with points
  defaultFitRatio: 0.75, // How much of view to fill when fitting to bounds (0-1)
};

/**
 * Shader configuration for point rendering
 *
 * Note: global exposure/offset/gamma live in renderingControls.defaults (applied in post-processing)
 */
export const shaderConfig: ShaderConfig = {
  points: {},
};
