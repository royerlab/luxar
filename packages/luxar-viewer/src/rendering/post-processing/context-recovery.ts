/**
 * Context-recovery helpers for the post-processing pipeline.
 *
 * Pure capture/apply functions for every effect's user-visible state.
 * These are used by both:
 *
 *  - `recreateComposer()` (called whenever AA settings flip) to round-
 *    trip current settings through a composer rebuild without losing
 *    user choices.
 *  - `rebuildAfterContextRestore()` (called after a WebGL context loss)
 *    to restore settings on the freshly created default effects.
 *
 * Originally those two paths each held a near-identical save/restore
 * block. Centralizing them here removes the duplication and makes the
 * shape of the durable state explicit.
 *
 * @module rendering/post-processing/context-recovery
 */

import type { Vector2 } from 'three';
import { ToneMappingMode } from 'postprocessing';

// ---------------------------------------------------------------------------
// State shapes
// ---------------------------------------------------------------------------

export interface BloomState {
  intensity: number;
  luminanceThreshold?: number;
  radius?: number;
}

export interface ToneMappingState {
  mode: ToneMappingMode;
  whitePoint?: number;
  exposure?: number;
  globalOffset?: number;
  globalGamma?: number;
}

export interface DOFState {
  bokehScale?: number;
  focusDistance?: number;
}

export interface VignetteState {
  darkness: number;
  offset: number;
}

export interface ChromaticLensDistortionState {
  distortion: Vector2;
  principalPoint: Vector2;
  focalLength: Vector2;
  skew: number;
  dispersion: number;
}

export interface DetectorNoiseState {
  readoutSigma: number;
  photonGain: number;
  fpnSigma: number;
}

/**
 * Snapshot of every user-visible setting of a `PostProcessingManager`.
 * Captured before a context-restore rebuild and re-applied to the
 * freshly recreated effects so the rebuild is transparent to the user.
 */
export interface PostProcessingDurableState {
  bloom: BloomState | null;
  toneMapping: ToneMappingState | null;
  dof: DOFState | null;
  vignette: VignetteState | null;
  chromaticLensDistortion: ChromaticLensDistortionState | null;
  detectorNoise: DetectorNoiseState | null;
  aoEnabled: boolean;
}

// ---------------------------------------------------------------------------
// Structural slices
// ---------------------------------------------------------------------------
// We intentionally use Pick-style structural types so tests can drive
// these helpers with plain object stubs. The manager passes its concrete
// effect instances at runtime — TypeScript's structural matching makes
// the wider types compatible.

export interface BloomTarget {
  intensity: number;
  mipmapBlurPass?: { radius: number };
  luminanceMaterial?: { threshold: number };
}

export interface ToneMappingTarget {
  mode: ToneMappingMode;
  whitePoint: number;
  exposure: number;
  globalOffset: number;
  globalGamma: number;
}

export interface DOFTarget {
  bokehScale: number;
  circleOfConfusionMaterial?: {
    uniforms?: { focusDistance?: { value: number } };
  };
}

export interface VignetteTarget {
  darkness: number;
  offset: number;
}

export interface ChromaticLensDistortionTarget {
  distortion: Vector2;
  principalPoint: Vector2;
  focalLength: Vector2;
  skew: number;
  dispersion: number;
}

export interface DetectorNoiseTarget {
  readoutSigma: number;
  photonGain: number;
  fpnSigma: number;
}

// ---------------------------------------------------------------------------
// Capture functions
// ---------------------------------------------------------------------------

export function captureBloomState(effect: BloomTarget | null | undefined): BloomState | null {
  if (!effect) return null;
  return {
    intensity: effect.intensity,
    luminanceThreshold: effect.luminanceMaterial?.threshold,
    radius: effect.mipmapBlurPass?.radius,
  };
}

export function captureToneMappingState(
  effect: ToneMappingTarget | null | undefined
): ToneMappingState | null {
  if (!effect) return null;
  return {
    mode: effect.mode,
    whitePoint: effect.whitePoint,
    exposure: effect.exposure,
    globalOffset: effect.globalOffset,
    globalGamma: effect.globalGamma,
  };
}

export function captureDOFState(effect: DOFTarget | null | undefined): DOFState | null {
  if (!effect) return null;
  return {
    bokehScale: effect.bokehScale,
    focusDistance: effect.circleOfConfusionMaterial?.uniforms?.focusDistance?.value,
  };
}

export function captureVignetteState(
  effect: VignetteTarget | null | undefined
): VignetteState | null {
  if (!effect) return null;
  return { darkness: effect.darkness, offset: effect.offset };
}

export function captureChromaticLensDistortionState(
  effect: ChromaticLensDistortionTarget | null | undefined
): ChromaticLensDistortionState | null {
  if (!effect) return null;
  return {
    distortion: effect.distortion.clone(),
    principalPoint: effect.principalPoint.clone(),
    focalLength: effect.focalLength.clone(),
    skew: effect.skew,
    dispersion: effect.dispersion,
  };
}

export function captureDetectorNoiseState(
  effect: DetectorNoiseTarget | null | undefined
): DetectorNoiseState | null {
  if (!effect) return null;
  return {
    readoutSigma: effect.readoutSigma,
    photonGain: effect.photonGain,
    fpnSigma: effect.fpnSigma,
  };
}

// ---------------------------------------------------------------------------
// Apply functions
// ---------------------------------------------------------------------------

export function applyBloomState(
  effect: BloomTarget | null | undefined,
  state: BloomState | null | undefined
): void {
  if (!effect || !state) return;
  effect.intensity = state.intensity;
  if (state.luminanceThreshold !== undefined && effect.luminanceMaterial) {
    effect.luminanceMaterial.threshold = state.luminanceThreshold;
  }
  if (state.radius !== undefined && effect.mipmapBlurPass) {
    effect.mipmapBlurPass.radius = state.radius;
  }
}

export function applyToneMappingState(
  effect: ToneMappingTarget | null | undefined,
  state: ToneMappingState | null | undefined
): void {
  if (!effect || !state) return;
  effect.mode = state.mode;
  if (state.whitePoint !== undefined) effect.whitePoint = state.whitePoint;
  if (state.exposure !== undefined) effect.exposure = state.exposure;
  if (state.globalOffset !== undefined) effect.globalOffset = state.globalOffset;
  if (state.globalGamma !== undefined) effect.globalGamma = state.globalGamma;
}

export function applyDOFFocusDistance(
  effect: DOFTarget | null | undefined,
  state: DOFState | null | undefined
): void {
  if (!effect || !state) return;
  if (
    state.focusDistance !== undefined &&
    effect.circleOfConfusionMaterial?.uniforms?.focusDistance
  ) {
    effect.circleOfConfusionMaterial.uniforms.focusDistance.value = state.focusDistance;
  }
}

export function applyVignetteState(
  effect: VignetteTarget | null | undefined,
  state: VignetteState | null | undefined
): void {
  if (!effect || !state) return;
  effect.darkness = state.darkness;
  effect.offset = state.offset;
}

export function applyChromaticLensDistortionState(
  effect: ChromaticLensDistortionTarget | null | undefined,
  state: ChromaticLensDistortionState | null | undefined
): void {
  if (!effect || !state) return;
  effect.distortion = state.distortion.clone();
  effect.principalPoint = state.principalPoint.clone();
  effect.focalLength = state.focalLength.clone();
  effect.skew = state.skew;
  effect.dispersion = state.dispersion;
}

export function applyDetectorNoiseState(
  effect: DetectorNoiseTarget | null | undefined,
  state: DetectorNoiseState | null | undefined
): void {
  if (!effect || !state) return;
  effect.readoutSigma = state.readoutSigma;
  effect.photonGain = state.photonGain;
  effect.fpnSigma = state.fpnSigma;
}
