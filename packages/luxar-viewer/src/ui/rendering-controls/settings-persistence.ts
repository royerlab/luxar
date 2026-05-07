/**
 * Pure helpers for rendering-controls settings persistence and defaults.
 *
 * Consolidates three previously inlined responsibilities:
 *  - Building the canonical default `RenderingSettings` (with optional
 *    zarr viewer-config overlay).
 *  - Reading a stored settings string from localStorage and merging it
 *    on top of base defaults.
 *  - Writing the current settings back to localStorage and clearing it.
 *
 * No DOM, no postprocessing-manager, no scene-manager calls — just
 * I/O + structured merging. The facade still owns the "after-load"
 * orchestration (apply to camera, refresh GUI, etc.).
 */

import { config, type RenderingSettings } from '../../config';
import { log, Modules } from '../../utils/log';
import { StorageKeys } from '../../utils/storage-keys';
import { serializeSettings, deserializeSettings } from './rendering-controls-utils';
import { extractRenderingOverrides } from '../../config/viewer-config-utils';
import type { ZarrViewerConfig } from '../../types/zarr';

/** RenderingSettings with the fly fields narrowed to non-optional concretes. */
export type FullySpecifiedRenderingSettings = RenderingSettings & {
  flyMovementSpeed: number;
  flyRotationSpeed: number;
  flyInertialMode: boolean;
  flyDamping: number;
  flyRotationDamping: number;
};

/** Build the hardcoded base defaults (config.renderingControls.defaults + fly defaults). */
export function buildBaseDefaults(): FullySpecifiedRenderingSettings {
  return {
    ...config.renderingControls.defaults,
    flyMovementSpeed: config.controls.fly.movement.speed.default,
    flyRotationSpeed: config.controls.fly.rotation.speed.default,
    flyInertialMode: config.controls.fly.inertialMode.default,
    flyDamping: config.controls.fly.movement.damping.default,
    flyRotationDamping: config.controls.fly.rotation.damping.default,
  };
}

/**
 * Build the "reset" defaults: base defaults overlaid with zarr viewer-config
 * overrides if available. Used by resetToDefaults.
 */
export function buildResetDefaults(
  zarrViewerConfig?: ZarrViewerConfig
): FullySpecifiedRenderingSettings {
  const defaults = buildBaseDefaults();
  if (zarrViewerConfig) {
    const zarrOverrides = extractRenderingOverrides(zarrViewerConfig);
    Object.assign(defaults, zarrOverrides);
  }
  return defaults;
}

/** Wipe stored rendering settings for the given scene id. Quota-safe. */
export function clearStoredSettings(sceneId: string): void {
  if (!sceneId) return;
  try {
    const key = StorageKeys.rendering(sceneId);
    localStorage.removeItem(key);
  } catch (err) {
    log.warning(Modules.RENDERING_CONTROLS, 'Failed to clear saved rendering settings', err);
  }
}

/** Persist current settings under the scene id. Quota-safe. */
export function saveSettingsToStorage(sceneId: string, settings: RenderingSettings): void {
  if (!sceneId) return;
  try {
    const key = StorageKeys.rendering(sceneId);
    localStorage.setItem(key, serializeSettings(settings));
  } catch (err) {
    log.warning(
      Modules.RENDERING_CONTROLS,
      'Failed to save rendering settings to localStorage',
      err
    );
  }
}

export interface LoadedSettings {
  /** Whether any settings string was found in localStorage (regardless of parseability). */
  stored: boolean;
  /** Parsed loaded settings, or null on parse failure / no value. */
  loaded: Partial<RenderingSettings> | null;
}

/** Read settings from localStorage. Quota-safe. */
export function loadSettingsFromStorage(sceneId: string): LoadedSettings {
  if (!sceneId) return { stored: false, loaded: null };

  let stored: string | null = null;
  try {
    const key = StorageKeys.rendering(sceneId);
    stored = localStorage.getItem(key);
  } catch (err) {
    log.warning(
      Modules.RENDERING_CONTROLS,
      'Failed to read rendering settings from localStorage',
      err
    );
  }

  if (!stored) return { stored: false, loaded: null };

  const loaded = deserializeSettings(stored);
  return { stored: true, loaded };
}
