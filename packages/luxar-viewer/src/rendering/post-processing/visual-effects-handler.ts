/**
 * Visual-effects helpers for the post-processing pipeline.
 *
 * Pure helpers used by DOF, detector-noise, vignette, and chromatic-lens
 * distortion paths in the manager: default resolution, the DPR-scaled
 * noise math, and the partial Vector2 update routines used when only one
 * component of a 2-vector parameter changes.
 *
 * The actual `new <Effect>(...)` construction stays in the manager.
 *
 * @module rendering/post-processing/visual-effects-handler
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// DOF
// ---------------------------------------------------------------------------

/** Default DOF focus distance in world units. */
export const DOF_DEFAULT_FOCUS = 10.0;
/** Default DOF strength (mapped to bokeh scale via × 4). */
export const DOF_DEFAULT_STRENGTH = 0.5;
/** Bokeh-scale multiplier — strength is exposed in [0, 1]; bokeh in [0, 4]. */
export const DOF_BOKEH_SCALE_MULTIPLIER = 4.0;

/**
 * Compute the focal length used by `DepthOfFieldEffect`, derived from
 * the camera FOV and normalized to a 50mm equivalent. Pure.
 */
export function computeDOFFocalLength(cameraFov: number): number {
  return 0.035 * (50.0 / cameraFov);
}

/** Convert UI strength [0,1] to pmndrs `bokehScale` [0,4]. */
export function strengthToBokehScale(strength: number): number {
  return strength * DOF_BOKEH_SCALE_MULTIPLIER;
}

// ---------------------------------------------------------------------------
// Detector noise
// ---------------------------------------------------------------------------

/** Default Gaussian readout-noise sigma. */
export const NOISE_DEFAULT_READOUT_SIGMA = 0.01;
/** Default photon gain (controls shot-noise visibility). */
export const NOISE_DEFAULT_PHOTON_GAIN = 0.01;
/** Default fixed-pattern-noise sigma (per-pixel offset). */
export const NOISE_DEFAULT_FPN_SIGMA = 0.005;

/** Triple of base noise parameters (user-configured, pre-DPR-scaling). */
export interface NoiseSettings {
  readoutSigma: number;
  photonGain: number;
  fpnSigma: number;
}

/**
 * Resolve a partial noise spec against the documented defaults.
 */
export function resolveNoiseDefaults(partial: Partial<NoiseSettings> = {}): NoiseSettings {
  return {
    readoutSigma: partial.readoutSigma ?? NOISE_DEFAULT_READOUT_SIGMA,
    photonGain: partial.photonGain ?? NOISE_DEFAULT_PHOTON_GAIN,
    fpnSigma: partial.fpnSigma ?? NOISE_DEFAULT_FPN_SIGMA,
  };
}

/**
 * Apply a DPR scale to base noise settings.
 *
 * - Gaussian noise (readout, FPN): σ scales linearly with DPR.
 * - Shot noise (photonGain): output σ ∝ √photonGain, so to scale σ by
 *   DPR we scale photonGain by DPR².
 *
 * Pure.
 */
export function scaleNoiseSettings(base: NoiseSettings, dprScale: number): NoiseSettings {
  return {
    readoutSigma: base.readoutSigma * dprScale,
    fpnSigma: base.fpnSigma * dprScale,
    photonGain: base.photonGain * dprScale * dprScale,
  };
}

/** Clamp the DPR scale used for noise scaling to its supported [0.25, 1.0] range. */
export function clampDPRScale(dpr: number): number {
  return Math.max(0.25, Math.min(1.0, dpr));
}

/**
 * `setDPRScale` on the manager only acts when the change is significant
 * (≥ 0.01 difference). Centralized here so tests can pin it.
 */
export function dprScaleChanged(prev: number, next: number, epsilon = 0.01): boolean {
  return Math.abs(next - prev) >= epsilon;
}

// ---------------------------------------------------------------------------
// Vignette
// ---------------------------------------------------------------------------

/** Default vignette darkness (0 = no darkening; 1 = full black at edges). */
export const VIGNETTE_DEFAULT_DARKNESS = 0.5;
/** Default vignette offset (radius from center where darkening starts). */
export const VIGNETTE_DEFAULT_OFFSET = 0.5;

export interface VignetteSettings {
  darkness: number;
  offset: number;
}

export function resolveVignetteDefaults(partial: Partial<VignetteSettings> = {}): VignetteSettings {
  return {
    darkness: partial.darkness ?? VIGNETTE_DEFAULT_DARKNESS,
    offset: partial.offset ?? VIGNETTE_DEFAULT_OFFSET,
  };
}

// ---------------------------------------------------------------------------
// Chromatic lens distortion
// ---------------------------------------------------------------------------

export interface ChromaticDistortionSettings {
  distortionX: number;
  distortionY: number;
  dispersion: number;
  principalPointX: number;
  principalPointY: number;
  focalLengthX: number;
  focalLengthY: number;
  skew: number;
}

export function resolveChromaticDistortionDefaults(
  partial: Partial<ChromaticDistortionSettings> = {}
): ChromaticDistortionSettings {
  return {
    distortionX: partial.distortionX ?? 0,
    distortionY: partial.distortionY ?? 0,
    dispersion: partial.dispersion ?? 0,
    principalPointX: partial.principalPointX ?? 0,
    principalPointY: partial.principalPointY ?? 0,
    focalLengthX: partial.focalLengthX ?? 1,
    focalLengthY: partial.focalLengthY ?? 1,
    skew: partial.skew ?? 0,
  };
}

/**
 * Build a fresh THREE.Vector2 from a current value plus optional X/Y
 * overrides. Used by the chromatic-lens-distortion update path where
 * only one of X/Y may be set.
 *
 * Returns a *new* Vector2 — pmndrs setters tend to assume identity
 * change to trigger a uniform upload.
 */
export function mergeVector2(
  current: { x: number; y: number },
  overrideX: number | undefined,
  overrideY: number | undefined
): THREE.Vector2 {
  return new THREE.Vector2(overrideX ?? current.x, overrideY ?? current.y);
}
