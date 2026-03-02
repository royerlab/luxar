/**
 * Captures the complete viewer state as a JSON-serializable object.
 *
 * This is used by the Ctrl+Shift+S key combo to export the current viewer
 * state to the clipboard, enabling Python → zarr → viewer → export → Python
 * round-trips.
 */

import type { ZarrViewerConfig } from '../types/zarr';
import { REVERSE_SETTINGS_MAP } from './viewer-config-utils';
import type { SceneManager } from '../scene/scene-manager';
import type { RenderingControls } from '../ui/rendering-controls';
import type { SceneDimsManager } from '../scene/scene-dims-manager';
import type { DimensionAnimationManager } from '../scene/dimension-animation-manager';
import { ThemeManager } from '../themes/theme-manager';

/**
 * Capture the complete viewer state as a ZarrViewerConfig object.
 *
 * This produces the same JSON format stored in zarr viewer_config,
 * so the output can be loaded directly with `ViewerConfig.from_json()` in Python.
 *
 * @param sceneManager - Scene manager with camera, scene, and controls
 * @param renderingControls - Rendering controls with all settings
 * @param sceneDimsManager - Scene dimensions manager with current navigation state
 * @param animationManager - Optional animation manager for playback state
 * @returns Complete ZarrViewerConfig snapshot
 */
export function captureViewerState(
  sceneManager: SceneManager,
  renderingControls: RenderingControls,
  sceneDimsManager: SceneDimsManager,
  animationManager?: DimensionAnimationManager,
  themeManager?: ThemeManager
): ZarrViewerConfig {
  const result: ZarrViewerConfig = {};
  const settings = renderingControls.settings;

  // --- Camera ---
  const camera = sceneManager.camera;
  const target = sceneManager.controls.getFocusTarget();

  result.camera = {
    position: [camera.position.x, camera.position.y, camera.position.z],
    target: [target.x, target.y, target.z],
    up: [camera.up.x, camera.up.y, camera.up.z],
    fov: settings.fov,
    fov_preset: settings.fovPreset,
    near: settings.near,
    far: settings.far,
  };

  // --- Background color ---
  if (sceneManager.scene.background) {
    const bg = sceneManager.scene.background;
    if ('isColor' in bg && (bg as any).isColor) {
      result.background_color = '#' + (bg as any).getHexString();
    }
  }

  // --- RenderingSettings → snake_case ---
  for (const [camelKey, value] of Object.entries(settings)) {
    if (value === undefined) continue;
    const snakeKey = REVERSE_SETTINGS_MAP[camelKey];
    if (snakeKey) {
      (result as Record<string, unknown>)[snakeKey] = value;
    }
  }

  // --- Theme ---
  try {
    const tm = themeManager ?? ThemeManager.getInstance();
    result.theme = tm.getCurrentTheme().id;
  } catch {
    // ThemeManager may not be initialized in tests
  }

  // --- Dimensions ---
  const dims = sceneDimsManager.getDims();
  if (dims) {
    result.dimensions = {
      current_step: [...dims.currentStep],
    };
  }

  // --- Animation (per-dimension) ---
  if (animationManager && dims) {
    const animEntries: ZarrViewerConfig['animation'] = [];
    let hasAnyState = false;

    for (let i = 0; i < dims.ndim; i++) {
      const state = animationManager.getState(i);
      if (state) {
        hasAnyState = true;
        animEntries.push({
          playing: state.isPlaying,
          target_fps: state.targetFPS,
          loop: state.loopMode,
          direction: state.direction,
        });
      } else {
        animEntries.push({});
      }
    }

    if (hasAnyState) {
      result.animation = animEntries;
    }
  }

  return result;
}
