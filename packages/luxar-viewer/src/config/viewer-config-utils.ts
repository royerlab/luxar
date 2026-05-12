/**
 * Utilities for converting between zarr viewer_config (snake_case) and
 * TypeScript RenderingSettings (camelCase).
 *
 * The zarr format uses snake_case to match the Python/zarr convention.
 * The viewer uses camelCase for RenderingSettings.
 */

import type { ZarrViewerConfig } from '../types/zarr';
import type { RenderingSettings } from './types';

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
 * Returns only the fields that are set (partial object).
 *
 * @param zarrConfig - Viewer config from zarr root attributes
 * @returns Partial RenderingSettings with only the fields set in zarr
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

  // camera.fov maps to RenderingSettings.fov
  if (zarrConfig.camera?.fov !== undefined) {
    overrides.fov = zarrConfig.camera.fov;
  }

  // camera.fov_preset maps to RenderingSettings.fovPreset
  if (zarrConfig.camera?.fov_preset !== undefined) {
    overrides.fovPreset = zarrConfig.camera.fov_preset as RenderingSettings['fovPreset'];
  }

  // camera.near/far map to RenderingSettings.near/far
  if (zarrConfig.camera?.near !== undefined) {
    overrides.near = zarrConfig.camera.near;
  }
  if (zarrConfig.camera?.far !== undefined) {
    overrides.far = zarrConfig.camera.far;
  }

  return overrides;
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

  for (const [camelKey, value] of Object.entries(settings)) {
    if (value === undefined) continue;
    const snakeKey = REVERSE_SETTINGS_MAP[camelKey];
    if (snakeKey) {
      result[snakeKey] = value;
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
