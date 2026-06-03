/**
 * TSL / NodeMaterial counterpart to `MegaShaderMaterial`.
 *
 * Mirrors the GLSL `MegaShaderMaterial` wrapper one-for-one — same
 * constructor shape (`MegaShaderConfig`), same setter / toggle / getter
 * surface (`setExposure`, `toggleBloom`, `getToneMapping`, …).
 *
 * Mechanics:
 *
 *   - Holds a `uniforms` IUniform table identical to the GLSL wrapper's
 *     shape; the `megaWebGPUFactory` binds each primitive TSL uniform
 *     node to its IUniform via `.onUpdate(() => iuniform.value,
 *     'render')`. Setter calls mutate `this.uniforms.X.value`, the
 *     change propagates to the GPU on the next render.
 *
 *   - Feature toggles (`toggleBloom`, `toggleLensDistortion`,
 *     `toggleDetectorNoise`, `toggleVignette`) and tone-mapping mode
 *     selection drive the factory's JS-side `if` branches via the
 *     `MegaTSLConfig`; flipping any of them changes the graph shape,
 *     so the wrapper rebuilds the graph and sets `needsUpdate = true`
 *     (the same trigger point the GLSL wrapper uses for its
 *     `#ifdef`-driven recompiles).
 *
 *   - `toggleRawHdrCapture` / `toggleLinearLdrCapture` are EXR/HDR
 *     export shortcuts. Each one sets a config flag and rebuilds the
 *     TSL graph; the factory then routes around tone mapping / sRGB
 *     encoding to mirror the GLSL `LUXAR_CAPTURE_RAW_HDR` /
 *     `LUXAR_CAPTURE_LINEAR_LDR` defines.
 *
 * @module rendering/post-processing/mega/material-tsl
 */

import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { megaWebGPUFactory, type LuxarToneMappingMode } from './shader.tsl';
import type { MegaShaderConfig } from './material';

/**
 * Map THREE.ToneMapping → the compact Luxar mode index used by the
 * TSL factory (1=Linear, 2=Reinhard, 3=Cineon, 4=ACES, 5=AgX,
 * 6=Neutral). `NoToneMapping` aliases to Linear so the shader still
 * clamps to [0, 1] (matches the GLSL wrapper's behaviour).
 */
function toneMappingToMode(mode: THREE.ToneMapping): LuxarToneMappingMode {
  switch (mode) {
    case THREE.NoToneMapping:
    case THREE.LinearToneMapping:
      return 1;
    case THREE.ReinhardToneMapping:
      return 2;
    case THREE.CineonToneMapping:
      return 3;
    case THREE.ACESFilmicToneMapping:
      return 4;
    case THREE.AgXToneMapping:
      return 5;
    case THREE.NeutralToneMapping:
      return 6;
    default:
      return 6;
  }
}

function modeToToneMapping(mode: LuxarToneMappingMode): THREE.ToneMapping {
  switch (mode) {
    case 1:
      return THREE.LinearToneMapping;
    case 2:
      return THREE.ReinhardToneMapping;
    case 3:
      return THREE.CineonToneMapping;
    case 4:
      return THREE.ACESFilmicToneMapping;
    case 5:
      return THREE.AgXToneMapping;
    case 6:
    default:
      return THREE.NeutralToneMapping;
  }
}

/**
 * Mega-shader material rendered via TSL / NodeMaterial. The host
 * uses this exactly like `MegaShaderMaterial`:
 *
 *     const mega = materialManager.createMegaShaderMaterial(cfg);
 *     mega.setHdrSceneTexture(hdrTex);
 *     mega.setResolution(w, h);
 *     mega.advanceTime(dt);
 *     mega.setExposure(2.0);
 *     mega.toggleBloom(true);
 *     mega.setBloom(0.25, bloomTex);
 */
export class MegaShaderTSLMaterial extends NodeMaterial {
  /** Public uniforms table, same shape as `MegaShaderMaterial.uniforms`. */
  uniforms: Record<string, THREE.IUniform>;

  private useBloom = false;
  private useLensDistortion = false;
  private useDetectorNoise = false;
  private useVignette = false;
  private toneMappingMode: LuxarToneMappingMode = 6;
  private captureRawHDR = false;
  private captureLinearLDR = false;

  constructor(cfg: MegaShaderConfig = {}) {
    super();

    this.uniforms = {
      uHdrScene: { value: null as THREE.Texture | null },
      uResolution: { value: new THREE.Vector2(1, 1) },

      // EOG
      uExposure: { value: cfg.exposure ?? 0.0 },
      uGlobalOffset: { value: cfg.globalOffset ?? 0.0 },
      uGlobalGamma: { value: Math.max(0.001, cfg.globalGamma ?? 1.0) },

      // Tone-mapping carries a host-side `toneMappingExposure` for
      // the GLSL `<tonemapping_pars_fragment>` chunk; the TSL path
      // doesn't read it but the field is part of the uniforms shape
      // for cross-backend mirror-friendliness.
      toneMappingExposure: { value: 1.0 },

      // Lens distortion (always present; gated by useLensDistortion).
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
    };

    this.toneMappingMode = toneMappingToMode(cfg.toneMapping ?? THREE.ACESFilmicToneMapping);

    // Initial toggle state from the config — same behaviour as the
    // GLSL wrapper's constructor.
    this.useBloom = !!cfg.bloom?.enabled;
    this.useLensDistortion = !!cfg.lensDistortion?.enabled;
    this.useDetectorNoise = !!cfg.detectorNoise?.enabled;
    this.useVignette = !!cfg.vignette?.enabled;

    this.rebuildGraph();
  }

  /**
   * Re-run the TSL factory with the current feature flags and
   * tone-mapping mode. Called from the constructor and whenever a
   * toggle changes; mirrors the `needsUpdate = true` recompile the
   * GLSL wrapper triggers via define flips.
   */
  private rebuildGraph(): void {
    megaWebGPUFactory(
      this.uniforms,
      {
        useBloom: this.useBloom,
        useLensDistortion: this.useLensDistortion,
        useDetectorNoise: this.useDetectorNoise,
        useVignette: this.useVignette,
        toneMappingMode: this.toneMappingMode,
        captureRawHDR: this.captureRawHDR,
        captureLinearLDR: this.captureLinearLDR,
      },
      this
    );
    this.needsUpdate = true;
  }

  // ----------------------------------------------------------------
  // Per-frame inputs
  // ----------------------------------------------------------------

  setHdrSceneTexture(texture: THREE.Texture | null): void {
    // The TextureNode captures the Texture at factory-call time, so a
    // texture swap requires a graph rebuild (same constraint
    // PointTSLMaterial's colormap-swap path documents).
    const old = this.uniforms.uHdrScene.value as THREE.Texture | null;
    if (old === texture) return;
    this.uniforms.uHdrScene.value = texture;
    this.rebuildGraph();
  }

  setResolution(width: number, height: number): void {
    (this.uniforms.uResolution.value as THREE.Vector2).set(width, height);
  }

  advanceTime(dt: number): void {
    this.uniforms.uTime.value = (this.uniforms.uTime.value as number) + dt;
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
    const next = toneMappingToMode(mode);
    if (next === this.toneMappingMode) return;
    this.toneMappingMode = next;
    this.rebuildGraph();
  }

  getToneMapping(): THREE.ToneMapping {
    return modeToToneMapping(this.toneMappingMode);
  }

  // ----------------------------------------------------------------
  // Lens distortion
  // ----------------------------------------------------------------

  toggleLensDistortion(enabled: boolean): void {
    if (this.useLensDistortion === enabled) return;
    this.useLensDistortion = enabled;
    this.rebuildGraph();
  }

  isLensDistortionEnabled(): boolean {
    return this.useLensDistortion;
  }

  setLensDistortion(p: {
    distortion?: THREE.Vector2;
    principalPoint?: THREE.Vector2;
    focalLength?: THREE.Vector2;
    skew?: number;
    dispersion?: number;
  }): void {
    if (p.distortion) (this.uniforms.uDistortion.value as THREE.Vector2).copy(p.distortion);
    if (p.principalPoint) {
      (this.uniforms.uPrincipalPoint.value as THREE.Vector2).copy(p.principalPoint);
    }
    if (p.focalLength) (this.uniforms.uFocalLength.value as THREE.Vector2).copy(p.focalLength);
    if (p.skew !== undefined) this.uniforms.uSkew.value = p.skew;
    if (p.dispersion !== undefined) {
      this.uniforms.uDispersion.value = Math.max(0, Math.min(1, p.dispersion));
    }
  }

  // ----------------------------------------------------------------
  // Detector noise
  // ----------------------------------------------------------------

  toggleDetectorNoise(enabled: boolean): void {
    if (this.useDetectorNoise === enabled) return;
    this.useDetectorNoise = enabled;
    this.rebuildGraph();
  }

  isDetectorNoiseEnabled(): boolean {
    return this.useDetectorNoise;
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
    if (this.useVignette === enabled) return;
    this.useVignette = enabled;
    this.rebuildGraph();
  }

  isVignetteEnabled(): boolean {
    return this.useVignette;
  }

  setVignette(darkness: number, offset: number): void {
    this.uniforms.uVignetteDarkness.value = darkness;
    this.uniforms.uVignetteOffset.value = offset;
  }

  // ----------------------------------------------------------------
  // Bloom
  // ----------------------------------------------------------------

  toggleBloom(enabled: boolean): void {
    if (this.useBloom === enabled) return;
    this.useBloom = enabled;
    this.rebuildGraph();
  }

  isBloomEnabled(): boolean {
    return this.useBloom;
  }

  setBloom(intensity: number, texture: THREE.Texture | null): void {
    this.uniforms.uBloomIntensity.value = intensity;
    const oldTexture = this.uniforms.uBloomTexture.value as THREE.Texture | null;
    if (oldTexture !== texture) {
      this.uniforms.uBloomTexture.value = texture;
      // Texture swap requires a graph rebuild for the TSL TextureNode
      // binding to pick up the new texture.
      if (this.useBloom) this.rebuildGraph();
    }
  }

  // ----------------------------------------------------------------
  // EXR / HDR capture modes
  // ----------------------------------------------------------------

  /**
   * Enable/disable the RAW-HDR capture short-circuit. When on, the
   * fragment graph returns the post-bloom linear-HDR sample and skips
   * detector noise, EOG, tone mapping, vignette, and sRGB encoding —
   * matching the GLSL `LUXAR_CAPTURE_RAW_HDR` define exactly.
   *
   * Toggling forces a TSL graph rebuild because the factory branches
   * on this flag (see {@link shader.tsl.ts}). Cost is identical to
   * toggling any other feature flag (bloom, vignette, …).
   */
  toggleRawHdrCapture(enabled: boolean): void {
    if (this.captureRawHDR === enabled) return;
    this.captureRawHDR = enabled;
    this.rebuildGraph();
  }

  /**
   * Enable/disable the linear-LDR capture short-circuit. When on, the
   * pipeline runs every enabled effect, EOG, and tone mapping but
   * skips the final `linearToSRGB` encoding — matches the GLSL
   * `LUXAR_CAPTURE_LINEAR_LDR` define exactly.
   */
  toggleLinearLdrCapture(enabled: boolean): void {
    if (this.captureLinearLDR === enabled) return;
    this.captureLinearLDR = enabled;
    this.rebuildGraph();
  }
}
