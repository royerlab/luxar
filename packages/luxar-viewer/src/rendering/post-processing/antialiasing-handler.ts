/**
 * Antialiasing helpers for the post-processing pipeline.
 *
 * Pure helpers: SMAA-preset name → enum, SSAA multiplier clamp, MSAA
 * sample-count validation against the GPU's `MAX_SAMPLES`. The actual
 * `recreateComposer()` and the `gl.getParameter` capability checks
 * still live in the manager — extracted only the parts that are pure
 * arithmetic and string-table lookups.
 *
 * @module rendering/post-processing/antialiasing-handler
 */

import { SMAAPreset } from 'postprocessing';

/** SMAA preset names exposed to the UI / public API. */
export type SMAAPresetName = 'LOW' | 'MEDIUM' | 'HIGH' | 'ULTRA';

const SMAA_PRESET_TABLE: Record<SMAAPresetName, SMAAPreset> = {
  LOW: SMAAPreset.LOW,
  MEDIUM: SMAAPreset.MEDIUM,
  HIGH: SMAAPreset.HIGH,
  ULTRA: SMAAPreset.ULTRA,
};

/**
 * Resolve a textual SMAA preset to its pmndrs enum value. Falls back
 * to `HIGH` for unknown names — matches the manager's previous default.
 */
export function mapSMAAPreset(name: SMAAPresetName | undefined): SMAAPreset {
  if (!name) return SMAAPreset.HIGH;
  return SMAA_PRESET_TABLE[name] ?? SMAAPreset.HIGH;
}

/** Allowed MSAA sample counts (0 = disabled). */
export const VALID_MSAA_SAMPLES: ReadonlyArray<number> = [0, 2, 4, 8, 16];

/**
 * Result of validating a requested MSAA sample count against the GPU's
 * advertised `MAX_SAMPLES`. The caller logs the optional `warning`.
 */
export interface MSAASampleValidation {
  /** The sample count actually applied (always one of {@link VALID_MSAA_SAMPLES}). */
  samples: number;
  /** Human-readable explanation when the request was modified. */
  warning?: string;
}

/**
 * Validate a requested MSAA sample count. Two failure modes:
 *   1. Not in {@link VALID_MSAA_SAMPLES} — fall back to 4.
 *   2. Above the GPU's `maxSamples` — clamp to `maxSamples`. Note that
 *      `MAX_SAMPLES` can advertise non-power-of-two values (e.g. 6) and
 *      this helper does *not* round down to the nearest entry of
 *      {@link VALID_MSAA_SAMPLES}; preserving the original behavior.
 *
 * Pure: takes the requested count and the GPU max, returns the
 * applied count plus an optional warning string.
 */
export function validateMSAASamples(requested: number, maxSamples: number): MSAASampleValidation {
  let samples = requested;
  let warning: string | undefined;

  if (!VALID_MSAA_SAMPLES.includes(samples)) {
    warning = `Invalid MSAA samples: ${samples}. Using 4.`;
    samples = 4;
  }

  if (samples > maxSamples) {
    const clamped = Math.min(samples, maxSamples);
    warning = `Requested ${samples} MSAA samples but GPU only supports ${maxSamples}. Using ${clamped}.`;
    samples = clamped;
  }

  return warning ? { samples, warning } : { samples };
}

/**
 * Clamp an SSAA multiplier to the supported [1.0, 4.0] range.
 *
 * The manager allows decimal multipliers (1.5, 2.0, 3.0, 4.0). Values
 * below 1 are not meaningful (would *down*-sample) and above 4 are
 * impractical (16x pixel budget).
 */
export function clampSSAAMultiplier(value: number): number {
  return Math.max(1.0, Math.min(4.0, value));
}

/** Result of {@link checkMSAACapability}. */
export interface MSAACapability {
  /** GPU advertises enough sample support to actually use MSAA. */
  supported: boolean;
  /** GPU's max sample count from `gl.MAX_SAMPLES`. */
  maxSamples: number;
  /** True when float-color-buffer MSAA is OK; false → HDR may not blend properly. */
  floatBuffersOK: boolean;
}

/**
 * Probe a WebGL2 context for MSAA capability. The call sites need both
 * `MAX_SAMPLES` and the `EXT_color_buffer_float` extension; this
 * helper bundles them into a single struct so the manager only does
 * the imperative `recreate composer` step.
 */
export function checkMSAACapability(gl: WebGL2RenderingContext): MSAACapability {
  const maxSamples = (gl.getParameter(gl.MAX_SAMPLES) as number) ?? 0;
  const floatBuffersOK = gl.getExtension('EXT_color_buffer_float') !== null;
  return {
    supported: maxSamples >= 2,
    maxSamples,
    floatBuffersOK,
  };
}
