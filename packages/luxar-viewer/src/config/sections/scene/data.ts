import type { SceneConfig, ShaderConfig } from './types';

/**
 * 3D scene visual configuration
 */
export const sceneConfig: SceneConfig = {
  // Pitch black = zero radiance in the HDR buffer, so the post-processing
  // exposure/bloom chain never lifts an empty background (a non-zero clear
  // color is scene light: it goes white at high exposure). Scenes can still
  // author a tinted background via viewer_config.background_color.
  backgroundColor: 0x000000,
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
