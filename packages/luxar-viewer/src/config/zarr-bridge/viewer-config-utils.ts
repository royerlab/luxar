/**
 * Utilities for converting between zarr viewer_config (snake_case) and
 * TypeScript RenderingSettings (camelCase).
 *
 * The zarr format uses snake_case to match the Python/zarr convention.
 * The viewer uses camelCase for RenderingSettings.
 */

import type { ZarrViewerConfig } from '../../types/zarr';
import type { RenderingSettings } from '../types';
import {
  buildCinematicValues,
  CINEMATIC_SNAPSHOT_KEYS,
  type CinematicSnapshotKeys,
} from '../cinematic-preset';
import { cameraConfig } from '../sections/camera/data';
import { log, Modules } from '../../utils/log';

/**
 * Module-local sets of values we have already warned about, so each warning
 * surfaces exactly once instead of spamming the console on repeated calls.
 * Exported for tests to reset.
 */
export const _warnedUnknownRenderingKeys = new Set<string>();
export const _warnedFovPresetConflicts = new Set<string>();

/**
 * Mapping from snake_case zarr viewer_config keys to camelCase RenderingSettings keys.
 * Only includes fields that map directly to RenderingSettings properties.
 * Camera config, background_color, UI, theme, and dimensions are handled separately.
 */
export const RENDERING_SETTINGS_MAP: Record<string, keyof RenderingSettings> = {
  // Rendering pipeline
  tone_mapping: 'toneMapping',
  exposure: 'exposure',
  global_offset: 'globalOffset',
  global_gamma: 'globalGamma',

  // Bloom
  bloom_enabled: 'bloomEnabled',
  bloom_strength: 'bloomStrength',
  bloom_radius: 'bloomRadius',
  bloom_threshold: 'bloomThreshold',
  bloom_levels: 'bloomLevels',

  // Navigation
  control_type: 'controlType',
  auto_rotate: 'autoRotate',
  auto_rotate_speed: 'autoRotateSpeed',
  auto_rotate_axis: 'autoRotateAxis',
  natural_drag: 'naturalDrag',

  // Cinematic
  cinematic_mode: 'cinematicMode',

  // Vignette
  vignette_enabled: 'vignetteEnabled',
  vignette_darkness: 'vignetteDarkness',
  vignette_offset: 'vignetteOffset',

  // Detector noise
  detector_noise_enabled: 'detectorNoiseEnabled',
  detector_noise_readout_sigma: 'detectorNoiseReadoutSigma',
  detector_noise_photon_gain: 'detectorNoisePhotonGain',
  detector_noise_fpn_sigma: 'detectorNoiseFpnSigma',

  // Anti-aliasing
  fxaa_enabled: 'fxaaEnabled',
  msaa_enabled: 'msaaEnabled',
  msaa_samples: 'msaaSamples',
  ssaa_enabled: 'ssaaEnabled',
  ssaa_multiplier: 'ssaaMultiplier',

  // Chromatic lens distortion
  chromatic_lens_distortion_enabled: 'chromaticLensDistortionEnabled',
  chromatic_lens_distortion_x: 'chromaticLensDistortionX',
  chromatic_lens_distortion_y: 'chromaticLensDistortionY',
  chromatic_lens_dispersion: 'chromaticLensDispersion',
  chromatic_lens_principal_point_x: 'chromaticLensPrincipalPointX',
  chromatic_lens_principal_point_y: 'chromaticLensPrincipalPointY',
  chromatic_lens_focal_length_x: 'chromaticLensFocalLengthX',
  chromatic_lens_focal_length_y: 'chromaticLensFocalLengthY',
  chromatic_lens_skew: 'chromaticLensSkew',

  // Fly controls
  fly_movement_speed: 'flyMovementSpeed',
  fly_rotation_speed: 'flyRotationSpeed',
  fly_inertial_mode: 'flyInertialMode',
  fly_damping: 'flyDamping',
  fly_rotation_damping: 'flyRotationDamping',

  // Dynamic clipping
  dynamic_clipping_enabled: 'dynamicClippingEnabled',

  // Adaptive resolution
  adaptive_dpr_enabled: 'adaptiveDPREnabled',
};

/**
 * Reverse mapping: camelCase RenderingSettings key → snake_case zarr key.
 * Built from RENDERING_SETTINGS_MAP at module load time.
 */
export const REVERSE_SETTINGS_MAP: Record<string, string> = Object.fromEntries(
  Object.entries(RENDERING_SETTINGS_MAP).map(([snake, camel]) => [camel, snake])
);

/**
 * Extract RenderingSettings overrides from zarr viewer_config.
 * Returns only the fields that are set (partial object), plus — when the
 * scene asks for cinematic mode — the cinematic preset expanded into the
 * fields the scene left unset (see `expandCinematicPreset` below).
 * A recognized camera FOV preset supplies its numeric FOV when absent; a
 * conflicting numeric FOV wins and warns once per preset/value pair.
 *
 * @param zarrConfig - Viewer config from zarr root attributes
 * @returns Partial RenderingSettings with the fields set in zarr
 */
export function extractRenderingOverrides(
  zarrConfig: ZarrViewerConfig
): Partial<RenderingSettings> {
  const overrides: Partial<RenderingSettings> = {};

  for (const [snakeKey, camelKey] of Object.entries(RENDERING_SETTINGS_MAP)) {
    const value = (zarrConfig as Record<string, unknown>)[snakeKey];
    if (value !== undefined && value !== null) {
      (overrides as Record<string, unknown>)[camelKey] = value;
    }
  }

  // The camera block uses `!= null` deliberately (not `!== undefined`): it must
  // reject null exactly as the map walk above does, so "present in `overrides`"
  // stays a single uniform author-set test. A null that slipped through would
  // both suppress the cinematic preset for that key and reach consumers whose
  // own guards only test `!== undefined`.

  // camera.fov maps to RenderingSettings.fov
  if (zarrConfig.camera?.fov != null) {
    overrides.fov = zarrConfig.camera.fov;
  }

  // camera.fov_preset maps to RenderingSettings.fovPreset
  if (zarrConfig.camera?.fov_preset != null) {
    const fovPreset = zarrConfig.camera.fov_preset as RenderingSettings['fovPreset'];
    overrides.fovPreset = fovPreset;
    const presetFov = cameraConfig.fovPresets[fovPreset];
    if (overrides.fov === undefined) {
      // `> 0` rejects both the sentinel Custom value and unknown preset names.
      if (presetFov > 0) overrides.fov = presetFov;
    } else if (presetFov > 0 && Math.abs(presetFov - overrides.fov) >= 0.5) {
      const conflictKey = `${fovPreset}:${overrides.fov}`;
      if (!_warnedFovPresetConflicts.has(conflictKey)) {
        _warnedFovPresetConflicts.add(conflictKey);
        log.warning(
          Modules.CONFIG,
          `camera.fov (${overrides.fov}°) conflicts with camera.fov_preset "${fovPreset}" (${presetFov}°); the numeric FOV wins and the preset label will be re-derived from it when the panel first opens.`
        );
      }
    }
  }

  // camera.near/far map to RenderingSettings.near/far
  if (zarrConfig.camera?.near != null) {
    overrides.near = zarrConfig.camera.near;
  }
  if (zarrConfig.camera?.far != null) {
    overrides.far = zarrConfig.camera.far;
  }

  expandCinematicPreset(zarrConfig, overrides);

  return overrides;
}

/**
 * The two preset keys that describe the camera's framing. They are expanded as
 * ONE unit: if the author set EITHER `camera.fov` or `camera.fov_preset`,
 * NEITHER is filled from the cinematic preset.
 *
 * Why coupled — a framing is a unit, and half a pair is worse than neither
 * half. A recognized authored `fov_preset` resolves its own numeric FOV above;
 * this guard prevents the cinematic preset from replacing either half with
 * its 35 mm lens. An explicit numeric FOV remains authoritative without an
 * invented label, and an unknown or `Custom` preset remains unresolved.
 */
const CINEMATIC_FOV_PAIR: readonly CinematicSnapshotKeys[] = ['fov', 'fovPreset'];

/**
 * Expand `viewer_config.cinematic_mode = true` into the actual preset values
 * (ACES, subtle wide bloom, detector noise, vignette, 35 mm chromatic lens +
 * FOV).
 *
 * Why here: `cinematic_mode` used to map straight through to
 * `RenderingSettings.cinematicMode`, a flag nothing downstream acted on — the
 * preset was only ever applied by the C-key/rail toggle, so an authored scene
 * rendered with none of the effects. Expanding at the bridge means BOTH
 * consumers of `extractRenderingOverrides` get it for free:
 * `RenderingControls.applyZarrDefaults` (first-time scene load) and
 * `buildResetDefaults` (reset-to-defaults for a scene that ships a config).
 *
 * PRECEDENCE — an author-set key always wins over the preset. A key counts as
 * author-set exactly when it is already present in `overrides`, which is the
 * single uniform test for both routes into this object: the
 * `RENDERING_SETTINGS_MAP` walk above only adds a key when its snake_case
 * spelling was present and non-null in the zarr config, and the camera block
 * only adds `fov` / `fovPreset` when `camera.fov` / `camera.fov_preset` were
 * likewise present and non-null.
 * So `{cinematic_mode: true, bloom_strength: 0.9}` yields the full preset with
 * `bloomStrength = 0.9`.
 *
 * The one exception to the per-key rule is `CINEMATIC_FOV_PAIR` (declared just
 * above) — see there for why `fov` and `fovPreset` are expanded (or skipped)
 * together.
 *
 * Strictly `=== true`: a `false`, `null`, absent, or non-boolean truthy value
 * expands nothing (a corrupt config must not silently restyle the scene).
 *
 * `cinematicMode: true` itself stays in the overrides, so the control-rail
 * "Cinematic mode" item still reads as active.
 *
 * @param zarrConfig - Viewer config from zarr root attributes
 * @param overrides - Overrides built so far; mutated in place
 */
function expandCinematicPreset(
  zarrConfig: ZarrViewerConfig,
  overrides: Partial<RenderingSettings>
): void {
  if (zarrConfig.cinematic_mode !== true) return;

  const preset = buildCinematicValues();
  const authorSetFraming = CINEMATIC_FOV_PAIR.some((key) => key in overrides);
  for (const key of CINEMATIC_SNAPSHOT_KEYS) {
    if (key in overrides) continue; // author-set — leave it alone
    if (authorSetFraming && CINEMATIC_FOV_PAIR.includes(key)) continue;
    (overrides as Record<string, unknown>)[key] = preset[key];
  }
}

/**
 * Convert RenderingSettings (camelCase) to zarr viewer_config format (snake_case).
 * Only includes non-undefined fields from the input settings.
 *
 * @param settings - Full or partial RenderingSettings object
 * @returns Partial ZarrViewerConfig with snake_case keys
 */
export function renderingSettingsToZarr(
  settings: Partial<RenderingSettings>
): Partial<ZarrViewerConfig> {
  const result: Record<string, unknown> = {};

  // Iterate the INPUT keys (not REVERSE_SETTINGS_MAP) so that future-added
  // RenderingSettings fields missing from the bridge map are still dropped
  // (preserving backwards-compat with older zarr files) but surface a single
  // warning per unknown key, instead of being silently invisible.
  for (const [camelKey, value] of Object.entries(settings)) {
    if (value === undefined) continue;
    const snakeKey = REVERSE_SETTINGS_MAP[camelKey];
    if (snakeKey) {
      result[snakeKey] = value;
    } else if (!_warnedUnknownRenderingKeys.has(camelKey)) {
      _warnedUnknownRenderingKeys.add(camelKey);
      log.warning(
        Modules.RENDERING_CONTROLS,
        `renderingSettingsToZarr: dropping unknown RenderingSettings key "${camelKey}" — add it to RENDERING_SETTINGS_MAP to persist.`
      );
    }
  }

  // camera-related fields (fov, fovPreset, near, far) go under camera.*
  // These are handled separately by the state capture function

  return result as Partial<ZarrViewerConfig>;
}

/**
 * Camera overrides extracted from zarr viewer_config.
 * These are NOT part of RenderingSettings — they're spatial state
 * applied directly to the camera on every scene load.
 */
export interface CameraOverrides {
  position?: { x: number; y: number; z: number };
  target?: { x: number; y: number; z: number };
  up?: { x: number; y: number; z: number };
  /** Named node whose bounding box center becomes the camera target */
  targetNode?: string;
}

/**
 * Extract camera configuration from zarr viewer_config.
 *
 * @param zarrConfig - Viewer config from zarr root attributes
 * @returns Camera overrides (position, target, up, targetNode)
 */
export function extractCameraOverrides(zarrConfig: ZarrViewerConfig): CameraOverrides {
  const overrides: CameraOverrides = {};
  if (zarrConfig.camera?.position) {
    const [x, y, z] = zarrConfig.camera.position;
    overrides.position = { x, y, z };
  }
  if (zarrConfig.camera?.target) {
    const [x, y, z] = zarrConfig.camera.target;
    overrides.target = { x, y, z };
  }
  if (zarrConfig.camera?.up) {
    const [x, y, z] = zarrConfig.camera.up;
    overrides.up = { x, y, z };
  }
  if (zarrConfig.camera?.target_node) {
    overrides.targetNode = zarrConfig.camera.target_node;
  }
  return overrides;
}

/**
 * Extract background color from zarr viewer_config.
 *
 * @param zarrConfig - Viewer config from zarr root attributes
 * @returns Hex color string or undefined
 */
export function extractBackgroundColor(zarrConfig: ZarrViewerConfig): string | undefined {
  return zarrConfig.background_color;
}
