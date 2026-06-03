/**
 * Mega-shader material: single-pass post-processing.
 *
 * Wraps {@link MEGA_FRAGMENT_SHADER} as a `THREE.ShaderMaterial` that
 * the host renders against a fullscreen triangle. One pass fuses
 * chromatic lens distortion, detector noise, EOG, tone mapping,
 * vignette, and sRGB encoding.
 *
 * The host renders bloom into a separate texture before invoking this
 * material (see {@link BloomChain}) and runs FXAA on the output if
 * enabled (see {@link FxaaPass}).
 *
 * Toggling an effect on/off triggers a shader recompile via the
 * `#ifdef USE_*` gates. Value tweaks just touch uniforms.
 *
 * @module rendering/post-processing/mega/material
 */

import * as THREE from 'three';
import { MEGA_VERTEX_SHADER, MEGA_FRAGMENT_SHADER } from './shader.glsl';

/**
 * Map THREE.ToneMapping → the `LUXAR_TONE_MAPPING_MODE` shader-define
 * value. These are Luxar-internal compressed IDs (1..6), NOT THREE's
 * enum values (which split 0,1,2,3,4,5=Custom,6=AgX,7=Neutral).
 *
 * `NoToneMapping` deliberately routes to mode 1 (Linear) so its
 * shader behavior is clamp/saturate to [0,1].
 */
function toneMappingModeDefine(mode: THREE.ToneMapping): string {
  switch (mode) {
    case THREE.NoToneMapping:
      return '1'; // alias to Linear — clamps to [0,1]
    case THREE.LinearToneMapping:
      return '1';
    case THREE.ReinhardToneMapping:
      return '2';
    case THREE.CineonToneMapping:
      return '3';
    case THREE.ACESFilmicToneMapping:
      return '4';
    case THREE.AgXToneMapping:
      return '5';
    case THREE.NeutralToneMapping:
      return '6';
    default:
      return '6'; // Neutral — preserves hue fidelity for scientific data
  }
}

/** Constructor config for {@link MegaShaderMaterial}. All fields optional. */
export interface MegaShaderConfig {
  // EOG
  exposure?: number;
  globalOffset?: number;
  globalGamma?: number;

  // Tone mapping
  toneMapping?: THREE.ToneMapping;

  // Lens distortion
  lensDistortion?: {
    enabled?: boolean;
    distortion?: THREE.Vector2;
    principalPoint?: THREE.Vector2;
    focalLength?: THREE.Vector2;
    skew?: number;
    dispersion?: number;
  };

  // Detector noise
  detectorNoise?: {
    enabled?: boolean;
    readoutSigma?: number;
    photonGain?: number;
    fpnSigma?: number;
  };

  // Vignette
  vignette?: {
    enabled?: boolean;
    darkness?: number;
    offset?: number;
  };

  // Bloom (host renders the texture; material just mixes it in)
  bloom?: {
    enabled?: boolean;
    intensity?: number;
  };
}

/**
 * Single-pass post-processing material. See module header for the
 * shader operation order and the bloom/FXAA pre/post-pass contract.
 */
export class MegaShaderMaterial extends THREE.ShaderMaterial {
  constructor(cfg: MegaShaderConfig = {}) {
    super({
      vertexShader: MEGA_VERTEX_SHADER,
      fragmentShader: MEGA_FRAGMENT_SHADER,
      glslVersion: THREE.GLSL3,
      depthTest: false,
      depthWrite: false,
      // CRITICAL: We `#include <tonemapping_pars_fragment>` explicitly
      // in our shader. With toneMapped=true (the default), THREE would
      // also auto-inject that chunk into the program prefix —
      // duplicating the `toneMappingExposure` uniform declaration and
      // failing shader compilation with "redefinition" errors.
      toneMapped: false,

      uniforms: {
        // Inputs
        uHdrScene: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },

        // EOG
        uExposure: { value: cfg.exposure ?? 0.0 },
        uGlobalOffset: { value: cfg.globalOffset ?? 0.0 },
        uGlobalGamma: { value: Math.max(0.001, cfg.globalGamma ?? 1.0) },

        // THREE's tone-mapping chunk reads this; we pin it to 1.0
        // because uExposure already pre-multiplies.
        toneMappingExposure: { value: 1.0 },

        // Lens distortion (always present; gated by define)
        uDistortion: { value: cfg.lensDistortion?.distortion ?? new THREE.Vector2(0, 0) },
        uPrincipalPoint: {
          value: cfg.lensDistortion?.principalPoint ?? new THREE.Vector2(0, 0),
        },
        uFocalLength: {
          value: cfg.lensDistortion?.focalLength ?? new THREE.Vector2(1, 1),
        },
        uSkew: { value: cfg.lensDistortion?.skew ?? 0 },
        uDispersion: { value: cfg.lensDistortion?.dispersion ?? 0 },

        // Detector noise
        uTime: { value: 0 },
        uReadoutSigma: { value: cfg.detectorNoise?.readoutSigma ?? 0.01 },
        uPhotonGain: { value: cfg.detectorNoise?.photonGain ?? 0.01 },
        uFpnSigma: { value: cfg.detectorNoise?.fpnSigma ?? 0.005 },

        // Vignette
        uVignetteDarkness: { value: cfg.vignette?.darkness ?? 0.5 },
        uVignetteOffset: { value: cfg.vignette?.offset ?? 0.5 },

        // Bloom
        uBloomTexture: { value: null as THREE.Texture | null },
        uBloomIntensity: { value: cfg.bloom?.intensity ?? 0.25 },
      },

      defines: {
        LUXAR_TONE_MAPPING_MODE: toneMappingModeDefine(cfg.toneMapping ?? THREE.ACESFilmicToneMapping),
      },
    });

    // Apply initial toggle state via the setters so defines stay
    // consistent with the constructor flags.
    if (cfg.lensDistortion?.enabled) this.toggleLensDistortion(true);
    if (cfg.detectorNoise?.enabled) this.toggleDetectorNoise(true);
    if (cfg.vignette?.enabled) this.toggleVignette(true);
    if (cfg.bloom?.enabled) this.toggleBloom(true);
  }

  // ----------------------------------------------------------------
  // Per-frame inputs
  // ----------------------------------------------------------------

  setHdrSceneTexture(texture: THREE.Texture | null): void {
    this.uniforms.uHdrScene.value = texture;
  }

  setResolution(width: number, height: number): void {
    this.uniforms.uResolution.value.set(width, height);
  }

  /** Advance the detector-noise time clock by `dt` seconds. */
  advanceTime(dt: number): void {
    this.uniforms.uTime.value += dt;
  }

  // ----------------------------------------------------------------
  // EOG
  // ----------------------------------------------------------------

  setExposure(value: number): void {
    this.uniforms.uExposure.value = value;
  }

  setGlobalOffset(value: number): void {
    this.uniforms.uGlobalOffset.value = value;
  }

  setGlobalGamma(value: number): void {
    this.uniforms.uGlobalGamma.value = Math.max(0.001, value);
  }

  // ----------------------------------------------------------------
  // Tone mapping
  // ----------------------------------------------------------------

  setToneMapping(mode: THREE.ToneMapping): void {
    const next = toneMappingModeDefine(mode);
    if (this.defines.LUXAR_TONE_MAPPING_MODE === next) return;
    this.defines.LUXAR_TONE_MAPPING_MODE = next;
    this.needsUpdate = true;
  }

  /**
   * Read back the current tone-mapping mode as a THREE constant.
   * Note: `THREE.NoToneMapping` is aliased to `LinearToneMapping` on
   * set (to clamp to [0,1] matching old behavior), so this getter
   * returns `LinearToneMapping` for either input.
   */
  getToneMapping(): THREE.ToneMapping {
    switch (this.defines.LUXAR_TONE_MAPPING_MODE) {
      case '1':
        return THREE.LinearToneMapping;
      case '2':
        return THREE.ReinhardToneMapping;
      case '3':
        return THREE.CineonToneMapping;
      case '4':
        return THREE.ACESFilmicToneMapping;
      case '5':
        return THREE.AgXToneMapping;
      case '6':
      default:
        return THREE.NeutralToneMapping;
    }
  }

  // ----------------------------------------------------------------
  // Lens distortion
  // ----------------------------------------------------------------

  toggleLensDistortion(enabled: boolean): void {
    this.setDefineFlag('USE_LENS_DISTORTION', enabled);
  }

  isLensDistortionEnabled(): boolean {
    return 'USE_LENS_DISTORTION' in this.defines;
  }

  setLensDistortion(p: {
    distortion?: THREE.Vector2;
    principalPoint?: THREE.Vector2;
    focalLength?: THREE.Vector2;
    skew?: number;
    dispersion?: number;
  }): void {
    if (p.distortion) this.uniforms.uDistortion.value.copy(p.distortion);
    if (p.principalPoint) this.uniforms.uPrincipalPoint.value.copy(p.principalPoint);
    if (p.focalLength) this.uniforms.uFocalLength.value.copy(p.focalLength);
    if (p.skew !== undefined) this.uniforms.uSkew.value = p.skew;
    if (p.dispersion !== undefined) {
      // Clamp matches the original ChromaticLensDistortionEffect.
      this.uniforms.uDispersion.value = Math.max(0, Math.min(1, p.dispersion));
    }
  }

  // ----------------------------------------------------------------
  // Detector noise
  // ----------------------------------------------------------------

  toggleDetectorNoise(enabled: boolean): void {
    this.setDefineFlag('USE_DETECTOR_NOISE', enabled);
  }

  isDetectorNoiseEnabled(): boolean {
    return 'USE_DETECTOR_NOISE' in this.defines;
  }

  setDetectorNoise(p: { readoutSigma?: number; photonGain?: number; fpnSigma?: number }): void {
    if (p.readoutSigma !== undefined) {
      this.uniforms.uReadoutSigma.value = Math.max(0, p.readoutSigma);
    }
    if (p.photonGain !== undefined) {
      this.uniforms.uPhotonGain.value = Math.max(0.0001, p.photonGain);
    }
    if (p.fpnSigma !== undefined) {
      this.uniforms.uFpnSigma.value = Math.max(0, p.fpnSigma);
    }
  }

  // ----------------------------------------------------------------
  // Vignette
  // ----------------------------------------------------------------

  toggleVignette(enabled: boolean): void {
    this.setDefineFlag('USE_VIGNETTE', enabled);
  }

  isVignetteEnabled(): boolean {
    return 'USE_VIGNETTE' in this.defines;
  }

  setVignette(darkness: number, offset: number): void {
    this.uniforms.uVignetteDarkness.value = darkness;
    this.uniforms.uVignetteOffset.value = offset;
  }

  // ----------------------------------------------------------------
  // Bloom
  // ----------------------------------------------------------------

  toggleBloom(enabled: boolean): void {
    this.setDefineFlag('USE_BLOOM', enabled);
  }

  isBloomEnabled(): boolean {
    return 'USE_BLOOM' in this.defines;
  }

  /** Set the bloom intensity. Disables the bloom branch if texture is null. */
  setBloom(intensity: number, texture: THREE.Texture | null): void {
    this.uniforms.uBloomIntensity.value = intensity;
    this.uniforms.uBloomTexture.value = texture;
  }

  // ----------------------------------------------------------------
  // EXR / HDR capture modes
  // ----------------------------------------------------------------

  /**
   * Bypass everything past the initial sample + bloom mix — output
   * pre-EOG linear HDR. Use only during `captureHDRPixels`
   * `'hdr-effects-pre-tone'` mode; callers MUST also disable
   * detector noise / vignette / lens distortion via the regular
   * `toggle*` methods so they don't pollute the linear HDR output.
   */
  toggleRawHdrCapture(enabled: boolean): void {
    this.setDefineFlag('LUXAR_CAPTURE_RAW_HDR', enabled);
  }

  /**
   * Skip ONLY the final sRGB encoding step; the rest of the pipeline
   * (EOG, tone mapping, vignette) runs normally. Output is linear LDR
   * (post-tone-mapped, pre-sRGB). Used by `captureHDRPixels`
   * `'visible-ldr'` mode.
   */
  toggleLinearLdrCapture(enabled: boolean): void {
    this.setDefineFlag('LUXAR_CAPTURE_LINEAR_LDR', enabled);
  }

  // ----------------------------------------------------------------
  // Internal helpers
  // ----------------------------------------------------------------

  /**
   * Add or remove a boolean define. Triggers a recompile only when
   * the flag actually changed, so repeated toggles to the same value
   * are free.
   */
  private setDefineFlag(name: string, on: boolean): void {
    const present = name in this.defines;
    if (present === on) return;
    if (on) {
      this.defines[name] = '';
    } else {
      delete this.defines[name];
    }
    this.needsUpdate = true;
  }
}
