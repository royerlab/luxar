/**
 * Captures the complete viewer state as a JSON-serializable object.
 *
 * This is used by the Ctrl+Shift+S key combo to export the current viewer
 * state to the clipboard, enabling Python → zarr → viewer → export → Python
 * round-trips.
 */

import type { ZarrViewerConfig } from '../../types/zarr';
import { REVERSE_SETTINGS_MAP, environmentConfigToZarr } from './viewer-config-utils';
import type { SceneManager } from '../../scene/scene-manager';
import type { RenderingSettings } from '../sections/rendering-controls/types';
import type { SceneDimsManager } from '../../scene/scene-dims-manager';
import type { DimensionAnimationManager } from '../../scene/animation/dimension-animation-manager';
import { ThemeManager } from '../../themes/theme-manager';
import { log, Modules } from '../../utils/log';

/**
 * Module-local set of camelCase keys we have already warned about, so that
 * each unknown field surfaces exactly once per page load instead of spamming
 * the console on every state capture.
 */
const _warnedUnknownStateKeys = new Set<string>();

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
  renderingControls: { readonly settings: RenderingSettings },
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
  };

  // `near` / `far` are AUTHORED here only when the user owns them. While dynamic
  // clipping is on, `settings.near` / `settings.far` are the live camera readouts
  // stamped in by `ClippingDisplay`'s RAF loop and `syncCurrentState` — the same
  // transient values `stripDynamicClippingPlanes` keeps out of localStorage.
  // Capturing them wrote a zoomed-in pose's planes (e.g. near 1.05e-4 / far 61)
  // into an exported viewer_config as if the author had chosen them, and since
  // the capture also emits `dynamic_clipping_enabled: true`, loading that file
  // tripped the "dynamic clipping will override these" warning — whose advice
  // (set it false) is exactly what would pin the pathological pair. Omitting
  // them lets the loader auto-adjust, which is what the authored scene means.
  if (!settings.dynamicClippingEnabled) {
    result.camera.near = settings.near;
    result.camera.far = settings.far;
  }

  // --- Background color ---
  if (sceneManager.scene.background) {
    const bg = sceneManager.scene.background;
    // Three.Color's isColor flag is a runtime tag — narrow on the
    // structural shape rather than importing the full type.
    if ('isColor' in bg && (bg as { isColor: boolean }).isColor) {
      result.background_color = '#' + (bg as { getHexString: () => string }).getHexString();
    }
  }

  // --- Scene environment (material="physical" lighting) ---
  // Only when the scene authored one: the default room is what an absent block
  // means, and writing it out would freeze today's default into every export.
  if (sceneManager.environment && sceneManager.getSceneViewerConfig()?.environment) {
    result.environment = environmentConfigToZarr(sceneManager.environment.getConfig());
  }

  // --- RenderingSettings → snake_case ---
  // We iterate the INPUT keys (not REVERSE_SETTINGS_MAP) so that future-added
  // RenderingSettings fields missing from the bridge map become visible. They
  // are still dropped (round-trip safety with older zarr files), but a single
  // warning per unknown key surfaces the silent loss for engineers to fix.
  for (const [camelKey, value] of Object.entries(settings)) {
    if (value === undefined) continue;
    const snakeKey = REVERSE_SETTINGS_MAP[camelKey];
    if (snakeKey) {
      (result as Record<string, unknown>)[snakeKey] = value;
    } else if (!_warnedUnknownStateKeys.has(camelKey)) {
      _warnedUnknownStateKeys.add(camelKey);
      log.warning(
        Modules.RENDERING_CONTROLS,
        `captureViewerState: dropping unknown RenderingSettings key "${camelKey}" — add it to RENDERING_SETTINGS_MAP to persist.`
      );
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
          // Conditional: Auto (null) stays absent, keeping the exported
          // shape identical for states that never touched the override.
          ...(state.stepSize != null ? { step_size: state.stepSize } : {}),
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
