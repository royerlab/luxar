/**
 * Pure utility functions for rendering controls settings management
 *
 * This module contains side-effect-free functions extracted from rendering-controls.ts
 * to improve testability and maintainability. These functions handle settings
 * validation, serialization, and merging without external dependencies.
 */

import type { RenderingSettings as ConfigRenderingSettings } from '../../config/types';

/**
 * Re-export the configuration RenderingSettings to maintain consistency
 */
export type RenderingSettings = ConfigRenderingSettings;

import { config } from '../../config';
import { isAutoRotateAxis } from '../../controls/types';
import { TONE_MAPPING_NAMES } from '../../rendering/post-processing/tone-mapping';

/**
 * Get default rendering settings from main config
 */
export function getDefaultRenderingSettings(): RenderingSettings {
  return config.renderingControls.defaults;
}

/** Valid tone mapping options — the shared list, so validation cannot drift. */
const VALID_TONE_MAPPINGS: readonly string[] = TONE_MAPPING_NAMES;

/** Valid control types */
const VALID_CONTROL_TYPES = ['orbit', 'fly', 'ortho'];

/**
 * Validates rendering settings and applies defaults for missing/invalid values.
 * Guards against corrupted localStorage by range-checking numeric values
 * and verifying enum membership for string values.
 *
 * @param settings - Partial settings object (e.g. from localStorage)
 * @returns Complete validated settings with invalid values replaced by defaults
 */
export function validateRenderingSettings(settings: Partial<RenderingSettings>): RenderingSettings {
  const defaults = getDefaultRenderingSettings();
  const merged = { ...defaults, ...settings };

  // Numeric range validation — replace out-of-range with default
  const clampOrDefault = (value: unknown, fallback: number, min: number, max: number): number => {
    if (typeof value !== 'number' || !isFinite(value)) return fallback;
    if (value < min || value > max) return fallback;
    return value;
  };

  const positiveOrDefault = (value: unknown, fallback: number): number => {
    if (typeof value !== 'number' || !isFinite(value) || value <= 0) return fallback;
    return value;
  };

  // Boolean validation — replace non-boolean (strings, numbers, NaN
  // injection from corrupted localStorage / zarr config) with default.
  const booleanOrDefault = (value: unknown, fallback: boolean): boolean => {
    return typeof value === 'boolean' ? value : fallback;
  };

  // Integer-clamp validation — like clampOrDefault but rounds to int.
  // Useful for sample counts and other integer-only fields.
  const integerClampOrDefault = (
    value: unknown,
    fallback: number,
    min: number,
    max: number
  ): number => {
    const clamped = clampOrDefault(value, fallback, min, max);
    return Math.round(clamped);
  };

  merged.fov = clampOrDefault(merged.fov, defaults.fov, config.camera.fovMin, config.camera.fovMax);
  merged.near = positiveOrDefault(merged.near, defaults.near);
  merged.far = positiveOrDefault(merged.far, defaults.far);
  if (merged.far <= merged.near) merged.far = defaults.far;

  // Bloom
  merged.bloomEnabled = booleanOrDefault(merged.bloomEnabled, defaults.bloomEnabled);
  merged.bloomThreshold = clampOrDefault(merged.bloomThreshold, defaults.bloomThreshold, 0, 1);
  merged.bloomStrength = clampOrDefault(merged.bloomStrength, defaults.bloomStrength, 0, 10);
  merged.bloomRadius = clampOrDefault(merged.bloomRadius, defaults.bloomRadius, 0, 10);
  merged.bloomLevels = integerClampOrDefault(merged.bloomLevels, defaults.bloomLevels, 1, 12);

  // Exposure / tone-mapping
  merged.exposure = clampOrDefault(merged.exposure, defaults.exposure, -10, 10);
  merged.globalOffset = clampOrDefault(merged.globalOffset, defaults.globalOffset, -1, 1);
  merged.globalGamma = clampOrDefault(merged.globalGamma, defaults.globalGamma, 0.1, 10);

  // Anti-aliasing (FXAA inline; SSAA + MSAA at framebuffer level)
  merged.fxaaEnabled = booleanOrDefault(merged.fxaaEnabled, defaults.fxaaEnabled);
  merged.msaaEnabled = booleanOrDefault(merged.msaaEnabled, defaults.msaaEnabled);
  merged.msaaSamples = clampMSAASamples(merged.msaaSamples);
  merged.ssaaEnabled = booleanOrDefault(merged.ssaaEnabled, defaults.ssaaEnabled);
  merged.ssaaMultiplier = clampOrDefault(merged.ssaaMultiplier, defaults.ssaaMultiplier, 1, 8);

  // Vignette
  merged.vignetteEnabled = booleanOrDefault(merged.vignetteEnabled, defaults.vignetteEnabled);
  merged.vignetteDarkness = clampOrDefault(
    merged.vignetteDarkness,
    defaults.vignetteDarkness,
    0,
    1
  );
  merged.vignetteOffset = clampOrDefault(merged.vignetteOffset, defaults.vignetteOffset, 0, 1);

  // Detector noise
  merged.detectorNoiseEnabled = booleanOrDefault(
    merged.detectorNoiseEnabled,
    defaults.detectorNoiseEnabled
  );
  merged.detectorNoiseReadoutSigma = clampOrDefault(
    merged.detectorNoiseReadoutSigma,
    defaults.detectorNoiseReadoutSigma,
    0,
    1
  );
  merged.detectorNoisePhotonGain = clampOrDefault(
    merged.detectorNoisePhotonGain,
    defaults.detectorNoisePhotonGain,
    0,
    100
  );
  merged.detectorNoiseFpnSigma = clampOrDefault(
    merged.detectorNoiseFpnSigma,
    defaults.detectorNoiseFpnSigma,
    0,
    1
  );

  // Chromatic lens distortion
  merged.chromaticLensDistortionEnabled = booleanOrDefault(
    merged.chromaticLensDistortionEnabled,
    defaults.chromaticLensDistortionEnabled
  );
  merged.chromaticLensDistortionX = clampOrDefault(
    merged.chromaticLensDistortionX,
    defaults.chromaticLensDistortionX,
    -1,
    1
  );
  merged.chromaticLensDistortionY = clampOrDefault(
    merged.chromaticLensDistortionY,
    defaults.chromaticLensDistortionY,
    -1,
    1
  );
  merged.chromaticLensDispersion = clampOrDefault(
    merged.chromaticLensDispersion,
    defaults.chromaticLensDispersion,
    0,
    1
  );
  merged.chromaticLensPrincipalPointX = clampOrDefault(
    merged.chromaticLensPrincipalPointX,
    defaults.chromaticLensPrincipalPointX,
    0,
    1
  );
  merged.chromaticLensPrincipalPointY = clampOrDefault(
    merged.chromaticLensPrincipalPointY,
    defaults.chromaticLensPrincipalPointY,
    0,
    1
  );
  merged.chromaticLensFocalLengthX = positiveOrDefault(
    merged.chromaticLensFocalLengthX,
    defaults.chromaticLensFocalLengthX
  );
  merged.chromaticLensFocalLengthY = positiveOrDefault(
    merged.chromaticLensFocalLengthY,
    defaults.chromaticLensFocalLengthY
  );
  merged.chromaticLensSkew = clampOrDefault(
    merged.chromaticLensSkew,
    defaults.chromaticLensSkew,
    -1,
    1
  );

  // Navigation
  merged.autoRotate = booleanOrDefault(merged.autoRotate, defaults.autoRotate);
  merged.autoRotateSpeed = clampOrDefault(
    merged.autoRotateSpeed,
    defaults.autoRotateSpeed,
    -100,
    100
  );
  merged.naturalDrag = booleanOrDefault(merged.naturalDrag, defaults.naturalDrag);

  // Orbit feel (optional in the type — only validate when present).
  if (merged.orbitZoomSpeed !== undefined) {
    const zs = config.controls.orbit.zoom.speed;
    merged.orbitZoomSpeed = clampOrDefault(merged.orbitZoomSpeed, zs.default, zs.min, zs.max);
  }
  if (merged.orbitDampingFactor !== undefined) {
    const df = config.controls.orbit.damping.factor;
    merged.orbitDampingFactor = clampOrDefault(
      merged.orbitDampingFactor,
      df.default,
      df.min,
      df.max
    );
  }

  // Fly controls (all optional in the type — only validate when present).
  if (merged.flyMovementSpeed !== undefined) {
    merged.flyMovementSpeed = positiveOrDefault(
      merged.flyMovementSpeed,
      defaults.flyMovementSpeed ?? 1
    );
  }
  if (merged.flyRotationSpeed !== undefined) {
    merged.flyRotationSpeed = positiveOrDefault(
      merged.flyRotationSpeed,
      defaults.flyRotationSpeed ?? 1
    );
  }
  if (merged.flyLookSpeed !== undefined) {
    const look = config.controls.fly.look.mouseSpeed;
    merged.flyLookSpeed = clampOrDefault(merged.flyLookSpeed, look.default, look.min, look.max);
  }
  if (merged.flyInertialMode !== undefined) {
    merged.flyInertialMode = booleanOrDefault(
      merged.flyInertialMode,
      defaults.flyInertialMode ?? false
    );
  }
  if (merged.flyDamping !== undefined) {
    merged.flyDamping = clampOrDefault(merged.flyDamping, defaults.flyDamping ?? 0.1, 0, 1);
  }
  if (merged.flyRotationDamping !== undefined) {
    merged.flyRotationDamping = clampOrDefault(
      merged.flyRotationDamping,
      defaults.flyRotationDamping ?? 0.1,
      0,
      1
    );
  }

  // Toggles
  merged.dynamicClippingEnabled = booleanOrDefault(
    merged.dynamicClippingEnabled,
    defaults.dynamicClippingEnabled
  );
  merged.adaptiveDPREnabled = booleanOrDefault(
    merged.adaptiveDPREnabled,
    defaults.adaptiveDPREnabled
  );
  merged.allowHighDPR = booleanOrDefault(merged.allowHighDPR, defaults.allowHighDPR);
  merged.cinematicMode = booleanOrDefault(merged.cinematicMode, defaults.cinematicMode);
  merged.autoDolly = booleanOrDefault(merged.autoDolly, defaults.autoDolly);
  // Bound to the slider's own range: a hand-edited amplitude of 900% would
  // fling the camera through the subject on every cycle, and a non-positive
  // period would divide by zero in the phase advance.
  merged.autoDollyAmplitudePercent = clampOrDefault(
    merged.autoDollyAmplitudePercent,
    defaults.autoDollyAmplitudePercent,
    config.controls.orbit.autoDolly.amplitudePercent.min,
    config.controls.orbit.autoDolly.amplitudePercent.max
  );
  merged.autoDollyPeriod = positiveOrDefault(merged.autoDollyPeriod, defaults.autoDollyPeriod);

  // Enum validation — replace invalid strings with default
  if (!VALID_TONE_MAPPINGS.includes(merged.toneMapping)) {
    merged.toneMapping = defaults.toneMapping;
  }
  if (!VALID_CONTROL_TYPES.includes(merged.controlType)) {
    merged.controlType = defaults.controlType;
  }
  // The controls module's own guard, so a token the viewer cannot actually
  // rotate about can never survive validation.
  if (!isAutoRotateAxis(merged.autoRotateAxis)) {
    merged.autoRotateAxis = defaults.autoRotateAxis;
  }

  return merged;
}

/**
 * Merges settings with defaults, preserving valid values
 *
 * @param settings - Partial settings to merge
 * @param defaults - Default settings
 * @returns Merged settings
 */
export function mergeSettings(
  settings: Partial<RenderingSettings>,
  defaults: RenderingSettings = getDefaultRenderingSettings()
): RenderingSettings {
  return validateRenderingSettings({ ...defaults, ...settings });
}

/**
 * Serializes settings for storage.
 *
 * Accepts a PARTIAL snapshot — symmetric with {@link deserializeSettings},
 * which returns one. Callers legitimately omit keys they do not own:
 * `saveSettingsToStorage` drops `near` / `far` while dynamic clipping owns
 * them, so a transient camera readout is never persisted as user intent.
 *
 * @param settings - Settings to serialize
 * @returns JSON string
 */
export function serializeSettings(settings: Partial<RenderingSettings>): string {
  return JSON.stringify(settings);
}

/**
 * Deserializes settings from storage
 *
 * @param data - JSON string
 * @returns Parsed settings or null if invalid
 */
export function deserializeSettings(data: string): Partial<RenderingSettings> | null {
  try {
    const parsed = JSON.parse(data);
    if (typeof parsed === 'object' && parsed !== null) {
      return parsed as Partial<RenderingSettings>;
    }
  } catch {
    // Invalid JSON
  }
  return null;
}

/**
 * Checks if a color string is valid
 *
 * @param color - Color string to validate
 * @returns True if valid hex color or CSS color
 */
export function isValidColor(color: string): boolean {
  // Check hex color (both 3 and 6 character formats)
  if (/^#[0-9A-F]{6}$/i.test(color) || /^#[0-9A-F]{3}$/i.test(color)) {
    return true;
  }

  // Check RGB/RGBA
  if (/^rgba?\(/.test(color)) {
    return true;
  }

  // Check named colors (simplified list)
  const namedColors = ['black', 'white', 'red', 'green', 'blue', 'gray', 'grey'];
  return namedColors.includes(color.toLowerCase());
}

/**
 * Clamps MSAA samples to valid power-of-2 values.
 *
 * Non-numeric / NaN / Infinity inputs default to 0 (MSAA disabled).
 * Without the type/finite guard, NaN and string injection from
 * corrupted localStorage / zarr config would fall through every
 * `<=` comparison and silently return 16.
 *
 * @param samples - Requested sample count
 * @returns Valid sample count (0, 2, 4, 8, 16)
 */
export function clampMSAASamples(samples: unknown): number {
  if (typeof samples !== 'number' || !isFinite(samples)) return 0;
  if (samples <= 0) return 0;
  if (samples <= 2) return 2;
  if (samples <= 4) return 4;
  if (samples <= 8) return 8;
  return 16;
}

/**
 * Determines if settings have changed
 *
 * @param current - Current settings
 * @param previous - Previous settings
 * @returns True if settings differ
 */
export function settingsChanged(current: RenderingSettings, previous: RenderingSettings): boolean {
  // Simple deep equality check for settings
  return JSON.stringify(current) !== JSON.stringify(previous);
}

/**
 * Gets settings that require pipeline rebuild
 *
 * @param current - Current settings
 * @param previous - Previous settings
 * @returns List of changed settings that require rebuild
 */
export function getSettingsRequiringRebuild(
  current: RenderingSettings,
  previous: RenderingSettings
): string[] {
  const rebuildRequired: string[] = [];

  // Anti-aliasing changes require rebuild
  if (
    current.msaaEnabled !== previous.msaaEnabled ||
    current.msaaSamples !== previous.msaaSamples
  ) {
    rebuildRequired.push('msaa');
  }
  if (
    current.ssaaEnabled !== previous.ssaaEnabled ||
    current.ssaaMultiplier !== previous.ssaaMultiplier
  ) {
    rebuildRequired.push('ssaa');
  }
  if (current.fxaaEnabled !== previous.fxaaEnabled) {
    rebuildRequired.push('fxaa');
  }

  // Tone mapping type change requires rebuild
  if (current.toneMapping !== previous.toneMapping) {
    rebuildRequired.push('toneMapping');
  }

  return rebuildRequired;
}

/**
 * Calculates performance impact score for settings
 *
 * @param settings - Rendering settings
 * @returns Performance score (0-100, higher is more demanding)
 */
export function calculatePerformanceImpact(settings: RenderingSettings): number {
  let score = 0;

  // Anti-aliasing impact
  if (settings.fxaaEnabled) score += 5;
  if (settings.msaaEnabled) score += 15 + settings.msaaSamples * 2;
  if (settings.ssaaEnabled) score += 20 + settings.ssaaMultiplier * 10;

  // Post-processing impact
  if (settings.bloomEnabled) {
    score += 5 + settings.bloomStrength * 5;
  }

  // Tone mapping impact
  if (settings.toneMapping !== 'None') {
    score += 5;
  }

  // Chromatic lens distortion impact (3 texture samples in mega-shader)
  if (settings.chromaticLensDistortionEnabled) score += 4;

  // Auto-rotate impact (continuous rendering)
  if (settings.autoRotate) score += 5;

  return Math.min(100, score);
}
