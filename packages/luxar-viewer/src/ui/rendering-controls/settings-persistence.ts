/**
 * Pure helpers for rendering-controls settings persistence and defaults.
 *
 * Three responsibilities:
 *  - Building the canonical default `RenderingSettings` (with optional
 *    zarr viewer-config overlay).
 *  - Reading a stored settings string from localStorage and merging it
 *    on top of base defaults.
 *  - Writing the current settings back to localStorage and clearing it.
 *
 * No DOM, no post-processing-manager, no scene-manager calls — just
 * I/O + structured merging. The facade still owns the "after-load"
 * orchestration (apply to camera, refresh GUI, etc.).
 */

import { config, type RenderingSettings } from '../../config';
import { log, Modules } from '../../utils/log';
import { StorageKeys } from '../../utils/storage-keys';
import {
  serializeSettings,
  deserializeSettings,
  validateRenderingSettings,
} from './controls-utils';
import { extractRenderingOverrides } from '../../config/zarr-bridge/viewer-config-utils';
import type { ZarrViewerConfig } from '../../types/zarr';

/**
 * Version stamp of the per-scene rendering-settings document in localStorage.
 *
 * The stored value is an envelope `{ version, settings }` rather than the bare
 * settings object, so a future change to what the settings MEAN (a renamed
 * key, a rescaled range) can bump this number and have every older document
 * treated as absent — the zarr `viewer_config` defaults then apply, exactly as
 * on a first visit — instead of being merged as if nothing had changed. Same
 * reset-to-defaults policy as `SETTINGS_VERSION` (config/user-settings.ts)
 * and `OPFS_ENCODING_VERSION` (cache/types.ts).
 */
export const RENDERING_SETTINGS_VERSION = 1 as const;

/** On-disk shape of a per-scene rendering-settings document. */
interface RenderingSettingsEnvelope {
  version: number;
  settings: Partial<RenderingSettings>;
}

/** RenderingSettings with the orbit/fly feel fields narrowed to non-optional concretes. */
export type FullySpecifiedRenderingSettings = RenderingSettings & {
  orbitZoomSpeed: number;
  orbitDampingFactor: number;
  flyMovementSpeed: number;
  flyRotationSpeed: number;
  flyLookSpeed: number;
  flyInertialMode: boolean;
  flyDamping: number;
  flyRotationDamping: number;
};

/** Build the hardcoded base defaults (config.renderingControls.defaults + orbit/fly defaults). */
export function buildBaseDefaults(): FullySpecifiedRenderingSettings {
  return {
    ...config.renderingControls.defaults,
    orbitZoomSpeed: config.controls.orbit.zoom.speed.default,
    orbitDampingFactor: config.controls.orbit.damping.factor.default,
    flyMovementSpeed: config.controls.fly.movement.speed.default,
    flyRotationSpeed: config.controls.fly.rotation.speed.default,
    flyLookSpeed: config.controls.fly.look.mouseSpeed.default,
    flyInertialMode: config.controls.fly.inertialMode.default,
    flyDamping: config.controls.fly.movement.damping.default,
    flyRotationDamping: config.controls.fly.rotation.damping.default,
  };
}

/**
 * Build the "reset" defaults: base defaults overlaid with zarr viewer-config
 * overrides if available. Used by resetToDefaults.
 *
 * The zarr overrides are routed through `validateRenderingSettings`
 * to clamp non-finite or out-of-range values (corrupted or malicious
 * viewer_config can otherwise inject NaN/Infinity into runtime
 * rendering state).
 */
export function buildResetDefaults(
  zarrViewerConfig?: ZarrViewerConfig
): FullySpecifiedRenderingSettings {
  const defaults = buildBaseDefaults();
  if (zarrViewerConfig) {
    const zarrOverrides = extractRenderingOverrides(zarrViewerConfig);
    const merged = { ...defaults, ...zarrOverrides };
    return validateRenderingSettings(merged) as FullySpecifiedRenderingSettings;
  }
  return defaults;
}

/** Wipe stored rendering settings for the given scene id. Quota-safe. */
export function clearStoredSettings(sceneId: string): void {
  if (!sceneId) return;
  try {
    localStorage.removeItem(StorageKeys.rendering(sceneId));
  } catch (err) {
    log.warning(Modules.RENDERING_CONTROLS, 'Failed to clear saved rendering settings', err);
  }
}

/**
 * Strip `near` / `far` from a settings snapshot when dynamic clipping
 * owns them.
 *
 * TWO places push the LIVE camera near/far into `settings.near` /
 * `settings.far` so the (read-only, greyed-out) sliders show current
 * values: `ClippingDisplay`'s RAF loop, and `syncCurrentState` on every
 * panel-open. Those are transient camera-derived readouts, not user
 * intent — but `saveSettingsToStorage`
 * serializes the whole settings object, so ANY later control change
 * would persist whatever the camera happened to read at that instant
 * (e.g. `near = 1.05e-4`, `far = 61` from a zoomed-in pose). The next
 * load then re-applies them as FIXED manual planes via
 * `setSceneId` → `updateClippingPlanes`, pinning a pathological
 * near/far even though nothing was set by hand.
 *
 * Omitting the keys (rather than writing defaults) means the load path's
 * `{ ...buildBaseDefaults(), ...loaded }` spread in
 * `RenderingControls.loadSettings` leaves `config.renderingControls.defaults`
 * standing for `near` / `far`, after which `autoAdjustClippingPlanes` + the
 * per-frame dynamic update take over — which is the whole point of the mode.
 * (The `validateRenderingSettings` call wrapping that spread only clamps
 * non-finite / out-of-range values; it is the spread, not the validator,
 * that supplies the fallback.) When dynamic clipping is OFF the values ARE
 * user intent, and are persisted unchanged.
 */
export function stripDynamicClippingPlanes(
  settings: RenderingSettings
): Partial<RenderingSettings> {
  // Always a fresh object, both branches: an exported helper that sometimes
  // aliases its input and sometimes copies it is a footgun for the next
  // caller. The spread costs nothing next to the JSON.stringify +
  // localStorage write it feeds, and saves are user-action-triggered.
  const copy: Partial<RenderingSettings> = { ...settings };
  if (!settings.dynamicClippingEnabled) return copy;
  delete copy.near;
  delete copy.far;
  return copy;
}

/**
 * Persist current settings under the scene id, wrapped in the
 * `{ version: RENDERING_SETTINGS_VERSION, settings }` envelope. Quota-safe.
 */
export function saveSettingsToStorage(sceneId: string, settings: RenderingSettings): void {
  if (!sceneId) return;
  try {
    const envelope: RenderingSettingsEnvelope = {
      version: RENDERING_SETTINGS_VERSION,
      settings: JSON.parse(serializeSettings(stripDynamicClippingPlanes(settings))),
    };
    localStorage.setItem(StorageKeys.rendering(sceneId), JSON.stringify(envelope));
  } catch (err) {
    log.warning(
      Modules.RENDERING_CONTROLS,
      'Failed to save rendering settings to localStorage',
      err
    );
  }
}

export interface LoadedSettings {
  /**
   * Whether a CURRENT-version settings document was found in localStorage.
   * A missing document, a pre-envelope value, or a document from another
   * `RENDERING_SETTINGS_VERSION` all report `false` — the caller then applies
   * the zarr `viewer_config` defaults exactly as on a first visit.
   */
  stored: boolean;
  /** Parsed loaded settings, or null when `stored` is false or the settings member is malformed. */
  loaded: Partial<RenderingSettings> | null;
}

/**
 * Parse a stored document into its envelope, or `null` when it is not a
 * current-version envelope (unparsable, bare pre-envelope settings, or a
 * different `version`).
 */
function parseEnvelope(raw: string): RenderingSettingsEnvelope | null {
  const parsed = deserializeSettings(raw) as Partial<RenderingSettingsEnvelope> | null;
  if (!parsed || parsed.version !== RENDERING_SETTINGS_VERSION) return null;
  if (typeof parsed.settings !== 'object' || parsed.settings === null) return null;
  return parsed as RenderingSettingsEnvelope;
}

/**
 * Read settings from localStorage. Quota-safe.
 *
 * A document that is not a current-version envelope is REMOVED (one
 * `log.info` line names the key and the version found) and reported as
 * `{ stored: false, loaded: null }`, so stale documents cannot linger and
 * a later `saveSettingsToStorage` writes a fresh envelope.
 */
export function loadSettingsFromStorage(sceneId: string): LoadedSettings {
  if (!sceneId) return { stored: false, loaded: null };

  let stored: string | null = null;
  try {
    stored = localStorage.getItem(StorageKeys.rendering(sceneId));
  } catch (err) {
    log.warning(
      Modules.RENDERING_CONTROLS,
      'Failed to read rendering settings from localStorage',
      err
    );
  }

  if (!stored) return { stored: false, loaded: null };

  const envelope = parseEnvelope(stored);
  if (!envelope) {
    const found = describeStoredVersion(stored);
    log.info(
      Modules.CONFIG,
      `${StorageKeys.rendering(sceneId)} version ${found} != ${RENDERING_SETTINGS_VERSION}, using defaults`
    );
    try {
      localStorage.removeItem(StorageKeys.rendering(sceneId));
    } catch (err) {
      log.warning(Modules.RENDERING_CONTROLS, 'Failed to remove stale rendering settings', err);
    }
    return { stored: false, loaded: null };
  }

  return { stored: true, loaded: envelope.settings };
}

/** Human-readable version of a rejected document, for the one log line. */
function describeStoredVersion(raw: string): string {
  const parsed = deserializeSettings(raw) as { version?: unknown } | null;
  if (!parsed) return 'unparsable';
  if (parsed.version === undefined) return 'none (pre-envelope)';
  return JSON.stringify(parsed.version);
}
