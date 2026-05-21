/**
 * Setter logic for the PostProcessingManager's user-toggle surface.
 *
 * Each helper owns the validation, logging, and side-effect routing for
 * one user-facing setting. The orchestrator class still owns the state
 * fields, which the helpers read via the supplied snapshot.
 *
 * @module rendering/post-processing/post-processing-manager/settings
 */

import * as THREE from 'three';
import { config } from '../../../config';
import { log, Modules } from '../../../utils/log';
import { clamp } from '../../../utils/clamp';
import type { LuxarMegaShaderMaterial } from '../../material-manager';
import type { BloomChain } from '../bloom/chain';

/** Validated MSAA sample counts. */
export const VALID_MSAA_SAMPLES = [0, 2, 4, 8, 16] as const;

// =====================================================================
// Bloom
// =====================================================================

export interface BloomLiveState {
  bloomChain: BloomChain | null;
  bloomIntensity: number;
  bloomRadius: number;
  bloomThreshold: number;
  bloomLevels: number;
  readonly megaShader: LuxarMegaShaderMaterial;
}

/** Push (strength, radius, threshold) into the live mega-shader / chain. */
export function updateBloomSettings(
  state: BloomLiveState,
  strength?: number,
  radius?: number,
  threshold?: number
): void {
  if (strength !== undefined) {
    state.bloomIntensity = strength;
    state.megaShader.setBloom(strength, state.bloomChain?.outputTexture ?? null);
  }
  if (radius !== undefined) {
    state.bloomRadius = radius;
    state.bloomChain?.setRadius(radius);
  }
  if (threshold !== undefined) {
    state.bloomThreshold = threshold;
    state.bloomChain?.setThreshold(threshold);
  }
  log.update(
    Modules.POST_PROCESSING,
    `Bloom updated: strength=${state.bloomIntensity.toFixed(2)}, ` +
      `radius=${state.bloomRadius.toFixed(2)}, threshold=${state.bloomThreshold.toFixed(2)}`
  );
}

/**
 * Clamp `levels` to [1, 12] and propagate to the chain. Returns the
 * clamped value so the orchestrator can update its field; returns
 * `null` when the request is a no-op.
 */
export function clampBloomLevels(
  current: number,
  requested: number,
  chain: BloomChain | null,
  physSize: { width: number; height: number },
  bloomIntensity: number,
  megaShader: LuxarMegaShaderMaterial
): number | null {
  const next = clamp(Math.round(requested), 1, 12);
  if (current === next) return null;
  if (chain) {
    // Pass the CURRENT physical canvas size so the rebuilt pyramid
    // matches the present render target dimensions. Without this
    // explicit argument, setLevels would derive the size from the
    // stale mip[0] — wrong if the canvas resized since the last
    // setSize() but before setLevels().
    chain.setLevels(next, physSize);
    // Re-bind in case texture identity changed after reallocation.
    megaShader.setBloom(bloomIntensity, chain.outputTexture);
  }
  log.info(Modules.POST_PROCESSING, `Bloom mipmap levels set to ${next}`);
  return next;
}

// =====================================================================
// MSAA + SSAA
// =====================================================================

/** Validate + clamp an MSAA sample count against device capabilities. */
export function validateMSAASamples(samples: number, maxSamples: number): number {
  if (!(VALID_MSAA_SAMPLES as readonly number[]).includes(samples)) {
    log.warning(Modules.POST_PROCESSING, `Invalid MSAA samples: ${samples}. Using 4.`);
    samples = 4;
  }
  if (samples > maxSamples) {
    log.warning(
      Modules.POST_PROCESSING,
      `Requested ${samples} MSAA samples but GPU max is ${maxSamples}. Clamping.`
    );
    samples = maxSamples;
  }
  return samples;
}

// =====================================================================
// Vignette
// =====================================================================

/** Enable + configure or disable the vignette pass. */
export function setVignetteEnabled(
  megaShader: LuxarMegaShaderMaterial,
  enabled: boolean,
  darkness?: number,
  offset?: number
): void {
  if (enabled) {
    const d = darkness ?? config.renderingControls.defaults.vignetteDarkness;
    const o = offset ?? config.renderingControls.defaults.vignetteOffset;
    megaShader.setVignette(d, o);
    megaShader.toggleVignette(true);
    log.success(Modules.POST_PROCESSING, `Vignette enabled: darkness=${d}, offset=${o}`);
  } else {
    megaShader.toggleVignette(false);
    log.info(Modules.POST_PROCESSING, 'Vignette disabled');
  }
}

// =====================================================================
// Chromatic lens distortion
// =====================================================================

export interface LensDistortionParams {
  distortionX?: number;
  distortionY?: number;
  dispersion?: number;
  principalPointX?: number;
  principalPointY?: number;
  focalLengthX?: number;
  focalLengthY?: number;
  skew?: number;
}

/** Enable lens distortion with full default-or-supplied parameters. */
export function setChromaticLensDistortionEnabled(
  megaShader: LuxarMegaShaderMaterial,
  enabled: boolean,
  params: LensDistortionParams = {}
): void {
  if (enabled) {
    const d = config.renderingControls.defaults;
    megaShader.setLensDistortion({
      distortion: new THREE.Vector2(
        params.distortionX ?? d.chromaticLensDistortionX,
        params.distortionY ?? d.chromaticLensDistortionY
      ),
      dispersion: params.dispersion ?? d.chromaticLensDispersion,
      principalPoint: new THREE.Vector2(
        params.principalPointX ?? d.chromaticLensPrincipalPointX,
        params.principalPointY ?? d.chromaticLensPrincipalPointY
      ),
      focalLength: new THREE.Vector2(
        params.focalLengthX ?? d.chromaticLensFocalLengthX,
        params.focalLengthY ?? d.chromaticLensFocalLengthY
      ),
      skew: params.skew ?? d.chromaticLensSkew,
    });
    megaShader.toggleLensDistortion(true);
    log.info(Modules.POST_PROCESSING, 'Chromatic lens distortion enabled');
  } else {
    megaShader.toggleLensDistortion(false);
    log.info(Modules.POST_PROCESSING, 'Chromatic lens distortion disabled');
  }
}

/** Partial-update path that preserves current uniforms for unspecified fields. */
export function updateChromaticLensDistortion(
  megaShader: LuxarMegaShaderMaterial,
  params: LensDistortionParams
): void {
  const distortion =
    params.distortionX !== undefined || params.distortionY !== undefined
      ? new THREE.Vector2(
          params.distortionX ?? (megaShader.uniforms.uDistortion.value as THREE.Vector2).x,
          params.distortionY ?? (megaShader.uniforms.uDistortion.value as THREE.Vector2).y
        )
      : undefined;
  const principalPoint =
    params.principalPointX !== undefined || params.principalPointY !== undefined
      ? new THREE.Vector2(
          params.principalPointX ?? (megaShader.uniforms.uPrincipalPoint.value as THREE.Vector2).x,
          params.principalPointY ?? (megaShader.uniforms.uPrincipalPoint.value as THREE.Vector2).y
        )
      : undefined;
  const focalLength =
    params.focalLengthX !== undefined || params.focalLengthY !== undefined
      ? new THREE.Vector2(
          params.focalLengthX ?? (megaShader.uniforms.uFocalLength.value as THREE.Vector2).x,
          params.focalLengthY ?? (megaShader.uniforms.uFocalLength.value as THREE.Vector2).y
        )
      : undefined;
  megaShader.setLensDistortion({
    distortion,
    principalPoint,
    focalLength,
    skew: params.skew,
    dispersion: params.dispersion,
  });
}

/**
 * Picking-system hook: the same UV transform the mega-shader uses
 * for chromatic lens distortion, exposed so input coordinates can be
 * corrected before lookup. Returns `null` when distortion is off.
 *
 * Vector2 values are cloned so a caller mutating the returned
 * object can't accidentally pollute the shader's live uniforms.
 */
export function getLensDistortionParams(megaShader: LuxarMegaShaderMaterial): {
  distortion: THREE.Vector2;
  principalPoint: THREE.Vector2;
  focalLength: THREE.Vector2;
  skew: number;
} | null {
  if (!megaShader.isLensDistortionEnabled()) return null;
  const u = megaShader.uniforms;
  return {
    distortion: (u.uDistortion.value as THREE.Vector2).clone(),
    principalPoint: (u.uPrincipalPoint.value as THREE.Vector2).clone(),
    focalLength: (u.uFocalLength.value as THREE.Vector2).clone(),
    skew: u.uSkew.value as number,
  };
}
