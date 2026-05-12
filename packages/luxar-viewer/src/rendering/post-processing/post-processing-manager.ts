/**
 * Post-processing pipeline manager.
 *
 * Three-stage pipeline:
 *
 *   1. Scene render → HDR HalfFloat target (with optional MSAA + SSAA)
 *   2. Bloom pyramid → bloom texture       (see {@link BloomChain})
 *   3. Mega-shader → LDR intermediate      (see {@link MegaShaderMaterial})
 *      ↳ samples HDR + bloom; applies chromatic lens distortion,
 *        detector noise, EOG, tone mapping, vignette in one pass.
 *   4. Optional FXAA → backbuffer          (see {@link FxaaPass})
 *
 * When FXAA is disabled, stage (3) writes directly to the backbuffer.
 *
 * @module rendering/post-processing/post-processing-manager
 */

import * as THREE from 'three';
import { EXRExporter, ZIP_COMPRESSION } from 'three/examples/jsm/exporters/EXRExporter.js';
import { log, Modules } from '../../utils/log';
import { config } from '../../config';
import { MegaShaderMaterial } from './mega-shader-material';
import { BloomChain } from './bloom-chain';
import { FxaaPass } from './fxaa-pass';
import { computeEffectiveRenderSize } from './render-target-sizing';
import {
  halfFloatToFloat32,
  float32ToHalfFloat,
  flipPixelsVerticallyRGBA,
} from './hdr-pixel-utils';
import { formatHDRExrLogLine } from './hdr-capture';
import { clamp } from '../../utils/clamp';

/** Validated MSAA sample counts. */
const VALID_MSAA_SAMPLES = [0, 2, 4, 8, 16] as const;

/**
 * Manages HDR post-processing: scene → bloom → mega-shader → (FXAA) →
 * backbuffer. Owns all transient GPU resources and a small amount of
 * persisted state (DPR noise scaling baseline, quality preset, etc.).
 */
export class PostProcessingManager {
  // ----------------------------------------------------------------
  // Transient GPU resources (recreated on context loss)
  // ----------------------------------------------------------------
  private hdrTarget!: THREE.WebGLRenderTarget;
  private ldrTarget!: THREE.WebGLRenderTarget;
  private bloomChain: BloomChain | null = null;
  private megaShader!: MegaShaderMaterial;
  private megaMesh!: THREE.Mesh;
  private megaScene!: THREE.Scene;
  private megaCamera!: THREE.OrthographicCamera;
  private fxaaPass: FxaaPass | null = null;

  // ----------------------------------------------------------------
  // Persisted state (survives context loss; mirrors UI state)
  // ----------------------------------------------------------------
  private fxaaEnabled = config.renderingControls.defaults.fxaaEnabled;
  private msaaEnabled = config.renderingControls.defaults.msaaEnabled;
  private msaaSamples: number = config.renderingControls.defaults.msaaSamples;
  private ssaaEnabled = config.renderingControls.defaults.ssaaEnabled;
  private ssaaMultiplier: number = config.renderingControls.defaults.ssaaMultiplier;
  private renderSize: { width: number; height: number };

  /** User-configured noise sigmas (before DPR scaling). */
  private baseNoiseSettings = {
    readoutSigma: config.renderingControls.defaults.detectorNoiseReadoutSigma,
    photonGain: config.renderingControls.defaults.detectorNoisePhotonGain,
    fpnSigma: config.renderingControls.defaults.detectorNoiseFpnSigma,
  };
  private currentDPRScale = 1.0;

  private bloomLevels: number = config.renderingControls.defaults.bloomLevels;
  private bloomIntensity: number = config.renderingControls.defaults.bloomStrength;
  private bloomRadius: number = config.renderingControls.defaults.bloomRadius;
  private bloomThreshold: number = config.renderingControls.defaults.bloomThreshold;

  // Deferred-rebuild: kept for API compatibility with cinematic-mode batching;
  // with the mega-shader, rebuilds are cheap so the deferred path is a
  // no-op pass-through.
  private deferRebuildDepth = 0;

  private disposed = false;

  // Wall-clock timestamp of the previous render() call, used to derive
  // the inter-frame delta for detector-noise time advancement. 0 means
  // "no previous frame" — first frame supplies dt=0 (no animation
  // step), matching THREE.Timer's initial behavior.
  private _previousRenderTimestamp = 0;

  /**
   * @param onResize  Optional callback invoked after every
   *   reallocation of the render-target pyramid (resize, SSAA
   *   toggle, MSAA toggle, DPR change). SceneManager wires this to
   *   `updateMaterialsForCurrentCamera()` so point/line/gsplat
   *   shaders pick up the new drawing-buffer size — otherwise their
   *   pre-computed `pointSizeFactor` / `uResolution` uniforms go
   *   stale on AA toggles and the scene looks subtly wrong until the
   *   next window resize.
   */
  constructor(
    private renderer: THREE.WebGLRenderer,
    private scene: THREE.Scene,
    private camera: THREE.Camera,
    size: { width: number; height: number },
    private onResize?: () => void
  ) {
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping; // We tone-map in mega-shader.

    this.renderSize = { ...size };

    this.initializeTransientResources({ applyDefaults: true });

    log.success(
      Modules.POST_PROCESSING,
      `PostProcessingManager initialized - Output: ${size.width}x${size.height}, ` +
        `Render: ${this.computeEffectiveSize().width}x${this.computeEffectiveSize().height}` +
        `${this.ssaaEnabled ? ' (SSAA)' : ''}, ` +
        `MSAA: ${this.msaaEnabled ? this.msaaSamples + 'x' : 'off'}`
    );
  }

  // ================================================================
  // Resource lifecycle
  // ================================================================

  private computeEffectiveSize(): { width: number; height: number } {
    return computeEffectiveRenderSize(this.renderSize, this.ssaaEnabled, this.ssaaMultiplier);
  }

  /**
   * Physical-pixel framebuffer size = effective (logical SSAA) size ×
   * renderer's `getPixelRatio()`. Render targets, the mega-shader and
   * the bloom/FXAA passes are all sized here so they match the
   * renderer's canvas backbuffer AND what materials read from
   * `renderer.getDrawingBufferSize()`.
   *
   * Without this, at DPR > 1 the hdrTarget would be smaller than the
   * canvas (and smaller than what point/line materials expect), and
   * `gl_PointSize` values would overshoot the viewport — visibly
   * brightening the scene through extra additive-blended pixel
   * coverage.
   */
  private getPhysicalSize(): { width: number; height: number } {
    const { width, height } = this.computeEffectiveSize();
    const dpr = this.renderer.getPixelRatio();
    return {
      width: Math.max(1, Math.round(width * dpr)),
      height: Math.max(1, Math.round(height * dpr)),
    };
  }

  private initializeTransientResources(opts: { applyDefaults: boolean } = { applyDefaults: true }): void {
    const { width, height } = this.getPhysicalSize();

    // HDR target: scene renders here (linear, HalfFloat, optional MSAA).
    this.hdrTarget = new THREE.WebGLRenderTarget(width, height, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
      stencilBuffer: false,
      samples: this.msaaEnabled ? this.msaaSamples : 0,
    });
    this.hdrTarget.texture.name = 'PostProcessing.hdrTarget';

    // LDR intermediate: mega-shader output. HalfFloat keeps the EXR
    // export path lossless even though values are tone-mapped to [0,1].
    this.ldrTarget = new THREE.WebGLRenderTarget(width, height, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.ldrTarget.texture.name = 'PostProcessing.ldrTarget';

    // Mega-shader + fullscreen mesh + ortho camera.
    this.megaShader = new MegaShaderMaterial({
      exposure: config.renderingControls.defaults.exposure,
      globalOffset: config.renderingControls.defaults.globalOffset,
      globalGamma: config.renderingControls.defaults.globalGamma,
      toneMapping: this.resolveToneMappingDefault(),
    });
    this.megaShader.setResolution(width, height);

    // Fullscreen triangle (NDC positions {-1,-1}, {3,-1}, {-1,3}).
    // The mega-shader vertex shader reads `position` directly; we use
    // an explicit attribute (not gl_VertexID) so THREE's WebGLRenderer
    // wires the VAO correctly.
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
    );
    this.megaMesh = new THREE.Mesh(geo, this.megaShader);
    this.megaMesh.frustumCulled = false;
    this.megaScene = new THREE.Scene();
    this.megaScene.add(this.megaMesh);
    this.megaCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // Bloom chain (built only when enabled; on by default per config).
    // Skip during context-restore rebuilds — the caller restores the
    // user's bloom-enabled choice from a snapshot, which may have been
    // off even if config defaults say on.
    if (opts.applyDefaults && config.renderingControls.defaults.bloomEnabled) {
      this.allocateBloomChain(width, height);
    }

    // FXAA pass (built only when enabled).
    if (this.fxaaEnabled) {
      this.fxaaPass = new FxaaPass(width, height);
    }

    // Apply config defaults to the mega-shader uniforms / defines.
    // Skipped during context-restore rebuilds — the caller restores user
    // toggle state from a snapshot, which would otherwise be clobbered
    // back to defaults here.
    if (opts.applyDefaults) {
      this.applyConfigDefaults();
    }
  }

  private allocateBloomChain(width: number, height: number): void {
    this.bloomChain = new BloomChain({
      levels: this.bloomLevels,
      threshold: this.bloomThreshold,
      smoothing: 0.01,
      radius: this.bloomRadius,
      width,
      height,
    });
    this.megaShader.toggleBloom(true);
    this.megaShader.setBloom(this.bloomIntensity, this.bloomChain.outputTexture);
  }

  private resolveToneMappingDefault(): THREE.ToneMapping {
    const name = config.renderingControls.defaults.toneMapping;
    switch (name) {
      case 'None':
        return THREE.NoToneMapping;
      case 'Linear':
        return THREE.LinearToneMapping;
      case 'Reinhard':
        return THREE.ReinhardToneMapping;
      case 'Cineon':
        return THREE.CineonToneMapping;
      case 'ACES':
        return THREE.ACESFilmicToneMapping;
      case 'AgX':
        return THREE.AgXToneMapping;
      case 'Neutral':
        return THREE.NeutralToneMapping;
      default:
        return THREE.NeutralToneMapping;
    }
  }

  private applyConfigDefaults(): void {
    const d = config.renderingControls.defaults;

    if (d.detectorNoiseEnabled) {
      this.setDetectorNoiseEnabled(true);
    }
    if (d.vignetteEnabled) {
      this.setVignetteEnabled(true);
    }
    if (d.chromaticLensDistortionEnabled) {
      this.setChromaticLensDistortionEnabled(true);
    }
  }

  private disposeTransientResources(): void {
    this.hdrTarget?.dispose();
    this.ldrTarget?.dispose();
    this.bloomChain?.dispose();
    this.bloomChain = null;
    this.fxaaPass?.dispose();
    this.fxaaPass = null;
    this.megaShader?.dispose();
    this.megaMesh?.geometry?.dispose();
  }

  // ================================================================
  // Deferred rebuild (kept for cinematic-mode batching API)
  // ================================================================

  startDeferRebuild(): void {
    this.deferRebuildDepth++;
  }

  endDeferRebuild(): void {
    if (this.deferRebuildDepth === 0) {
      log.warning(Modules.POST_PROCESSING, 'endDeferRebuild called with depth=0; ignoring');
      return;
    }
    this.deferRebuildDepth--;
  }

  withDeferredRebuild<T>(fn: () => T): T {
    this.startDeferRebuild();
    try {
      return fn();
    } finally {
      this.endDeferRebuild();
    }
  }

  // ================================================================
  // Camera
  // ================================================================

  setCamera(camera: THREE.Camera): void {
    this.camera = camera;
  }

  // ================================================================
  // Bloom
  // ================================================================

  updateBloomSettings(strength?: number, radius?: number, threshold?: number): void {
    if (strength !== undefined) {
      this.bloomIntensity = strength;
      this.megaShader.setBloom(strength, this.bloomChain?.outputTexture ?? null);
    }
    if (radius !== undefined) {
      this.bloomRadius = radius;
      this.bloomChain?.setRadius(radius);
    }
    if (threshold !== undefined) {
      this.bloomThreshold = threshold;
      this.bloomChain?.setThreshold(threshold);
    }
    log.update(
      Modules.POST_PROCESSING,
      `Bloom updated: strength=${this.bloomIntensity.toFixed(2)}, ` +
        `radius=${this.bloomRadius.toFixed(2)}, threshold=${this.bloomThreshold.toFixed(2)}`
    );
  }

  setBloomEnabled(enabled: boolean, strength?: number, radius?: number, threshold?: number): void {
    if (enabled && !this.bloomChain) {
      if (strength !== undefined) this.bloomIntensity = strength;
      if (radius !== undefined) this.bloomRadius = radius;
      if (threshold !== undefined) this.bloomThreshold = threshold;
      const { width, height } = this.getPhysicalSize();
      this.allocateBloomChain(width, height);
      log.success(Modules.POST_PROCESSING, 'Bloom enabled');
    } else if (!enabled && this.bloomChain) {
      // Disable in shader FIRST so any in-flight or pre-cached
      // material program no longer samples uBloomTexture; then drop
      // the texture handle and dispose the chain. Order matters: if
      // we disposed first, a render landing between dispose and the
      // shader toggle would sample a freed texture.
      this.megaShader.toggleBloom(false);
      this.megaShader.setBloom(this.bloomIntensity, null);
      this.bloomChain.dispose();
      this.bloomChain = null;
      log.info(Modules.POST_PROCESSING, 'Bloom disabled');
    } else if (enabled && this.bloomChain) {
      this.updateBloomSettings(strength, radius, threshold);
    }
  }

  isBloomEnabled(): boolean {
    return this.bloomChain !== null;
  }

  setBloomLevels(levels: number): void {
    const next = clamp(Math.round(levels), 1, 12);
    if (this.bloomLevels === next) return;
    this.bloomLevels = next;
    if (this.bloomChain) {
      // Pass the CURRENT physical canvas size so the rebuilt pyramid
      // matches the present render target dimensions. Without this
      // explicit argument, setLevels would derive the size from the
      // stale mip[0] — wrong if the canvas resized since the last
      // setSize() but before setLevels() (e.g. a quality-preset
      // change in a resize-debounce window).
      this.bloomChain.setLevels(next, this.getPhysicalSize());
      // Re-bind in case texture identity changed after reallocation.
      this.megaShader.setBloom(this.bloomIntensity, this.bloomChain.outputTexture);
    }
    log.info(Modules.POST_PROCESSING, `Bloom mipmap levels set to ${next}`);
  }

  // ================================================================
  // Tone mapping + EOG
  // ================================================================

  setToneMapping(mode: THREE.ToneMapping): void {
    this.megaShader.setToneMapping(mode);
    log.update(Modules.POST_PROCESSING, `Tone mapping set to mode ${mode}`);
  }

  updateExposure(value: number): void {
    this.megaShader.setExposure(value);
  }

  getExposure(): number {
    return this.megaShader.uniforms.uExposure.value as number;
  }

  updateGlobalOffset(value: number): void {
    this.megaShader.setGlobalOffset(value);
  }

  updateGlobalGamma(value: number): void {
    this.megaShader.setGlobalGamma(value);
  }

  // ================================================================
  // Anti-aliasing
  // ================================================================

  setFXAAEnabled(enabled: boolean): void {
    if (this.fxaaEnabled === enabled) return;
    this.fxaaEnabled = enabled;
    if (enabled && !this.fxaaPass) {
      const { width, height } = this.getPhysicalSize();
      this.fxaaPass = new FxaaPass(width, height);
    } else if (!enabled && this.fxaaPass) {
      this.fxaaPass.dispose();
      this.fxaaPass = null;
    }
    log.update(Modules.POST_PROCESSING, `FXAA ${enabled ? 'enabled' : 'disabled'}`);
  }

  // ================================================================
  // Detector noise (with DPR scaling)
  // ================================================================

  setDetectorNoiseEnabled(
    enabled: boolean,
    readoutSigma?: number,
    photonGain?: number,
    fpnSigma?: number
  ): void {
    if (enabled) {
      if (readoutSigma !== undefined) this.baseNoiseSettings.readoutSigma = readoutSigma;
      if (photonGain !== undefined) this.baseNoiseSettings.photonGain = photonGain;
      if (fpnSigma !== undefined) this.baseNoiseSettings.fpnSigma = fpnSigma;
      this.megaShader.toggleDetectorNoise(true);
      this.applyScaledNoiseSettings();
      log.info(
        Modules.POST_PROCESSING,
        `Detector noise enabled: readout=${this.baseNoiseSettings.readoutSigma}, ` +
          `gain=${this.baseNoiseSettings.photonGain}, fpn=${this.baseNoiseSettings.fpnSigma}`
      );
    } else {
      this.megaShader.toggleDetectorNoise(false);
      log.info(Modules.POST_PROCESSING, 'Detector noise disabled');
    }
  }

  updateDetectorNoiseSettings(params: {
    readoutSigma?: number;
    photonGain?: number;
    fpnSigma?: number;
  }): void {
    if (params.readoutSigma !== undefined) {
      this.baseNoiseSettings.readoutSigma = params.readoutSigma;
    }
    if (params.photonGain !== undefined) {
      this.baseNoiseSettings.photonGain = params.photonGain;
    }
    if (params.fpnSigma !== undefined) {
      this.baseNoiseSettings.fpnSigma = params.fpnSigma;
    }
    this.applyScaledNoiseSettings();
    log.update(
      Modules.POST_PROCESSING,
      `Detector noise updated (DPR scale: ${this.currentDPRScale.toFixed(2)})`
    );
  }

  /**
   * Re-apply base sigmas through the DPR scaling. When DPR < 1 the
   * sigma scales linearly with DPR (Gaussian) and photonGain scales
   * with DPR² (shot noise).
   */
  private applyScaledNoiseSettings(): void {
    if (!this.megaShader.isDetectorNoiseEnabled()) return;
    const s = this.currentDPRScale;
    this.megaShader.setDetectorNoise({
      readoutSigma: this.baseNoiseSettings.readoutSigma * s,
      photonGain: this.baseNoiseSettings.photonGain * s * s,
      fpnSigma: this.baseNoiseSettings.fpnSigma * s,
    });
  }

  setDPRScale(dpr: number): void {
    const scale = clamp(dpr, 0.1, 4.0);
    if (Math.abs(this.currentDPRScale - scale) < 0.01) return;
    this.currentDPRScale = scale;
    this.applyScaledNoiseSettings();
    if (scale < 1.0) {
      log.info(
        Modules.POST_PROCESSING,
        `Noise scaled for DPR ${dpr.toFixed(2)} (noise × ${scale.toFixed(2)})`
      );
    }
  }

  // ================================================================
  // Vignette
  // ================================================================

  setVignetteEnabled(enabled: boolean, darkness?: number, offset?: number): void {
    if (enabled) {
      const d = darkness ?? config.renderingControls.defaults.vignetteDarkness;
      const o = offset ?? config.renderingControls.defaults.vignetteOffset;
      this.megaShader.setVignette(d, o);
      this.megaShader.toggleVignette(true);
      log.success(Modules.POST_PROCESSING, `Vignette enabled: darkness=${d}, offset=${o}`);
    } else {
      this.megaShader.toggleVignette(false);
      log.info(Modules.POST_PROCESSING, 'Vignette disabled');
    }
  }

  // ================================================================
  // Chromatic lens distortion
  // ================================================================

  setChromaticLensDistortionEnabled(
    enabled: boolean,
    distortionX?: number,
    distortionY?: number,
    dispersion?: number,
    principalPointX?: number,
    principalPointY?: number,
    focalLengthX?: number,
    focalLengthY?: number,
    skew?: number
  ): void {
    if (enabled) {
      const d = config.renderingControls.defaults;
      this.megaShader.setLensDistortion({
        distortion: new THREE.Vector2(
          distortionX ?? d.chromaticLensDistortionX,
          distortionY ?? d.chromaticLensDistortionY
        ),
        dispersion: dispersion ?? d.chromaticLensDispersion,
        principalPoint: new THREE.Vector2(
          principalPointX ?? d.chromaticLensPrincipalPointX,
          principalPointY ?? d.chromaticLensPrincipalPointY
        ),
        focalLength: new THREE.Vector2(
          focalLengthX ?? d.chromaticLensFocalLengthX,
          focalLengthY ?? d.chromaticLensFocalLengthY
        ),
        skew: skew ?? d.chromaticLensSkew,
      });
      this.megaShader.toggleLensDistortion(true);
      log.info(Modules.POST_PROCESSING, 'Chromatic lens distortion enabled');
    } else {
      this.megaShader.toggleLensDistortion(false);
      log.info(Modules.POST_PROCESSING, 'Chromatic lens distortion disabled');
    }
  }

  updateChromaticLensDistortion(params: {
    distortionX?: number;
    distortionY?: number;
    dispersion?: number;
    principalPointX?: number;
    principalPointY?: number;
    focalLengthX?: number;
    focalLengthY?: number;
    skew?: number;
  }): void {
    const distortion =
      params.distortionX !== undefined || params.distortionY !== undefined
        ? new THREE.Vector2(
            params.distortionX ?? this.megaShader.uniforms.uDistortion.value.x,
            params.distortionY ?? this.megaShader.uniforms.uDistortion.value.y
          )
        : undefined;
    const principalPoint =
      params.principalPointX !== undefined || params.principalPointY !== undefined
        ? new THREE.Vector2(
            params.principalPointX ?? this.megaShader.uniforms.uPrincipalPoint.value.x,
            params.principalPointY ?? this.megaShader.uniforms.uPrincipalPoint.value.y
          )
        : undefined;
    const focalLength =
      params.focalLengthX !== undefined || params.focalLengthY !== undefined
        ? new THREE.Vector2(
            params.focalLengthX ?? this.megaShader.uniforms.uFocalLength.value.x,
            params.focalLengthY ?? this.megaShader.uniforms.uFocalLength.value.y
          )
        : undefined;
    this.megaShader.setLensDistortion({
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
  getLensDistortionParams(): {
    distortion: THREE.Vector2;
    principalPoint: THREE.Vector2;
    focalLength: THREE.Vector2;
    skew: number;
  } | null {
    if (!this.megaShader.isLensDistortionEnabled()) return null;
    const u = this.megaShader.uniforms;
    return {
      distortion: (u.uDistortion.value as THREE.Vector2).clone(),
      principalPoint: (u.uPrincipalPoint.value as THREE.Vector2).clone(),
      focalLength: (u.uFocalLength.value as THREE.Vector2).clone(),
      skew: u.uSkew.value as number,
    };
  }

  // ================================================================
  // MSAA + SSAA (canvas-level; reallocate HDR target on change)
  // ================================================================

  setMSAAEnabled(enabled: boolean): void {
    if (this.msaaEnabled === enabled) return;
    if (enabled) {
      const gl = this.renderer.getContext() as WebGL2RenderingContext;
      const maxSamples = (gl.getParameter(gl.MAX_SAMPLES) as number) ?? 0;
      if (maxSamples < 2) {
        log.error(Modules.POST_PROCESSING, `MSAA not supported (MAX_SAMPLES: ${maxSamples})`);
        return;
      }
    }
    this.msaaEnabled = enabled;
    this.reallocateForSize();
    log.update(
      Modules.POST_PROCESSING,
      `MSAA ${enabled ? `enabled (${this.msaaSamples}x)` : 'disabled'}`
    );
  }

  setMSAASamples(samples: number): void {
    if (!(VALID_MSAA_SAMPLES as readonly number[]).includes(samples)) {
      log.warning(Modules.POST_PROCESSING, `Invalid MSAA samples: ${samples}. Using 4.`);
      samples = 4;
    }
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const maxSamples = (gl.getParameter(gl.MAX_SAMPLES) as number) ?? 0;
    if (samples > maxSamples) {
      log.warning(
        Modules.POST_PROCESSING,
        `Requested ${samples} MSAA samples but GPU max is ${maxSamples}. Clamping.`
      );
      samples = maxSamples;
    }
    if (this.msaaSamples === samples) return;
    this.msaaSamples = samples;
    if (this.msaaEnabled) {
      this.reallocateForSize();
      log.info(Modules.POST_PROCESSING, `MSAA samples set to ${samples}`);
    }
  }

  setSSAAEnabled(enabled: boolean): void {
    if (this.ssaaEnabled === enabled) return;
    this.ssaaEnabled = enabled;
    this.reallocateForSize();
    log.update(
      Modules.POST_PROCESSING,
      `SSAA ${enabled ? `enabled (${this.ssaaMultiplier}x)` : 'disabled'}`
    );
  }

  setSSAAMultiplier(multiplier: number): void {
    multiplier = clamp(multiplier, 1.0, 4.0);
    if (this.ssaaMultiplier === multiplier) return;
    this.ssaaMultiplier = multiplier;
    if (this.ssaaEnabled) {
      this.reallocateForSize();
      log.update(Modules.POST_PROCESSING, `SSAA multiplier changed to ${multiplier}x`);
    }
  }

  // ================================================================
  // Animation loop hook
  // ================================================================

  needsContinuousAnimation(): boolean {
    return this.megaShader.isDetectorNoiseEnabled();
  }

  // ================================================================
  // Render
  // ================================================================

  render(): void {
    // Wall-clock delta since previous render() (in seconds). Detector
    // noise uses this to animate its temporal pattern. Must be
    // wall-clock delta, not render duration — render duration is
    // ~4 ms at 60 fps and would slow the noise animation 4x.
    const now = performance.now();
    const dt =
      this._previousRenderTimestamp === 0 ? 0 : (now - this._previousRenderTimestamp) / 1000;
    this._previousRenderTimestamp = now;

    // Advance detector-noise time BEFORE rendering so the first frame
    // with the new dt is what gets sampled.
    if (this.megaShader.isDetectorNoiseEnabled()) {
      this.megaShader.advanceTime(dt);
    }

    this.runPipeline({ applyFxaa: this.fxaaPass !== null, finalTarget: null });
  }

  /**
   * Run the full pipeline: scene → HDR → (bloom) → mega-shader →
   * (FXAA) → finalTarget.
   *
   * - `applyFxaa = true`: mega-shader writes to ldrTarget, FXAA reads
   *   ldrTarget and writes to `finalTarget`.
   * - `applyFxaa = false`: mega-shader writes directly to `finalTarget`.
   * - `finalTarget = null`: write to the canvas backbuffer.
   *
   * The capture paths use `applyFxaa = false` + an explicit target so
   * they can read the post-tone-mapping LDR buffer without an FXAA
   * pass between mega-shader and readback.
   */
  private runPipeline(opts: {
    applyFxaa: boolean;
    finalTarget: THREE.WebGLRenderTarget | null;
  }): void {
    // Defensive save/restore: the typical call sites (`render()` and
    // `captureHDRPixels`) don't care about pre-existing renderer
    // state, but a future caller (picking, offscreen probe) might
    // invoke runPipeline while another target is bound. Mirroring
    // BloomChain.render's pattern keeps the pipeline composable.
    const prevTarget = this.renderer.getRenderTarget();
    const prevAutoClear = this.renderer.autoClear;
    try {
      // (0) Scene → HDR target
      this.renderer.setRenderTarget(this.hdrTarget);
      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);

      // (1) Bloom pyramid
      if (this.bloomChain) {
        this.bloomChain.render(this.renderer, this.hdrTarget.texture);
      }

      // (2) Mega-shader
      this.megaShader.setHdrSceneTexture(this.hdrTarget.texture);

      if (opts.applyFxaa && this.fxaaPass) {
        // Mega → ldrTarget → FXAA → finalTarget
        this.renderer.setRenderTarget(this.ldrTarget);
        this.renderer.render(this.megaScene, this.megaCamera);
        this.renderer.setRenderTarget(opts.finalTarget);
        this.fxaaPass.render(this.renderer, this.ldrTarget.texture);
      } else {
        // Mega → finalTarget directly (no FXAA)
        this.renderer.setRenderTarget(opts.finalTarget);
        this.renderer.render(this.megaScene, this.megaCamera);
      }
    } finally {
      this.renderer.setRenderTarget(prevTarget);
      this.renderer.autoClear = prevAutoClear;
    }
  }

  // ================================================================
  // Resize
  // ================================================================

  resize(width: number, height: number): void {
    this.renderSize = { width, height };
    this.reallocateForSize();
  }

  /**
   * Re-allocate every GPU resource whose size depends on the effective
   * render size. Two unit systems are at play and they must NOT be
   * confused:
   *   - **Logical** (CSS) pixels — what `renderer.setSize(...)` takes.
   *     THREE multiplies internally by `pixelRatio` to derive the
   *     canvas backbuffer.
   *   - **Physical** pixels — `logical × pixelRatio`. This is what
   *     `getDrawingBufferSize()` reports to materials, what the canvas
   *     backbuffer is, and what our render targets MUST match.
   */
  private reallocateForSize(): void {
    const { width: logicalW, height: logicalH } = this.computeEffectiveSize();
    const { width: physW, height: physH } = this.getPhysicalSize();

    // Renderer takes logical size; it multiplies by pixelRatio for the
    // canvas backbuffer. SceneManager owns CSS sizing.
    this.renderer.setSize(logicalW, logicalH, false);
    this.renderer.domElement.style.width = `${this.renderSize.width}px`;
    this.renderer.domElement.style.height = `${this.renderSize.height}px`;

    // All render targets at PHYSICAL size — matches canvas backbuffer
    // and the resolution materials read from `getDrawingBufferSize()`.
    this.hdrTarget.dispose();
    this.hdrTarget = new THREE.WebGLRenderTarget(physW, physH, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
      stencilBuffer: false,
      samples: this.msaaEnabled ? this.msaaSamples : 0,
    });
    this.hdrTarget.texture.name = 'PostProcessing.hdrTarget';

    this.ldrTarget.setSize(physW, physH);
    if (this.bloomChain) {
      this.bloomChain.setSize(physW, physH);
      // setSize may have reallocated the mip targets — rebind the
      // (possibly new) bloom texture identity into the mega-shader.
      this.megaShader.setBloom(this.bloomIntensity, this.bloomChain.outputTexture);
    }
    this.fxaaPass?.setSize(physW, physH);
    this.megaShader.setResolution(physW, physH);

    log.info(
      Modules.POST_PROCESSING,
      `Resized: display ${this.renderSize.width}x${this.renderSize.height}, ` +
        `render ${logicalW}x${logicalH} logical → ${physW}x${physH} physical ` +
        `(DPR=${this.renderer.getPixelRatio().toFixed(2)})`
    );

    // Notify the host (SceneManager) that the canvas backbuffer
    // dimensions changed. Scene materials cache pointSizeFactor /
    // uResolution based on `renderer.getDrawingBufferSize()` and
    // would otherwise stay at the pre-resize values until the next
    // window resize fired.
    this.onResize?.();
  }

  // ================================================================
  // Capture
  // ================================================================

  /**
   * Read raw HDR float pixel data in one of three capture modes:
   *
   *   - `'hdr-effects-pre-tone'` (default): keep HDR-space effects
   *     (bloom) but disable tone mapping, EOG, vignette, detector
   *     noise, chromatic lens distortion. **Linear HDR output** —
   *     the canonical EXR-export mode. Implemented via the
   *     mega-shader's `LUXAR_CAPTURE_RAW_HDR` early-exit (skips EOG,
   *     tone mapping, vignette, sRGB encoding).
   *   - `'visible-ldr'`: full pipeline (EOG + tone mapping + every
   *     enabled effect), but skip the final sRGB encoding so the
   *     captured pixels are **linear LDR** floats.
   *   - `'raw-scene-hdr'`: bypass the mega-shader entirely — render
   *     the scene to hdrTarget and read it (no bloom, no effects).
   */
  captureHDRPixels(
    mode: 'visible-ldr' | 'hdr-effects-pre-tone' | 'raw-scene-hdr' = 'hdr-effects-pre-tone'
  ): { pixels: Float32Array; width: number; height: number } {
    if (mode === 'raw-scene-hdr') {
      // Scene render only — no bloom, no mega-shader.
      // Save/restore the renderer's current target + autoClear so a
      // caller that invokes capture while another target is bound
      // (picking, offscreen probe) doesn't get its state clobbered.
      // `runPipeline` wraps the same way; mirror it here too.
      const prevTarget = this.renderer.getRenderTarget();
      const prevAutoClear = this.renderer.autoClear;
      try {
        this.renderer.setRenderTarget(this.hdrTarget);
        this.renderer.clear();
        this.renderer.render(this.scene, this.camera);
        return this.readTarget(this.hdrTarget);
      } finally {
        this.renderer.setRenderTarget(prevTarget);
        this.renderer.autoClear = prevAutoClear;
      }
    }

    if (mode === 'hdr-effects-pre-tone') {
      // Save state, disable LDR-space effects, set the RAW_HDR
      // capture define so the mega-shader bypasses EOG, tone mapping,
      // vignette, and sRGB encoding. Bloom is intentionally kept in
      // the sample (USE_BLOOM is left as-is) — bloom is an HDR-space
      // effect and belongs in linear-HDR captures.
      const wasNoise = this.megaShader.isDetectorNoiseEnabled();
      const wasVignette = this.megaShader.isVignetteEnabled();
      const wasLens = this.megaShader.isLensDistortionEnabled();

      this.megaShader.toggleDetectorNoise(false);
      this.megaShader.toggleVignette(false);
      this.megaShader.toggleLensDistortion(false);
      this.megaShader.toggleRawHdrCapture(true);

      try {
        this.runPipeline({ applyFxaa: false, finalTarget: this.ldrTarget });
        return this.readTarget(this.ldrTarget);
      } finally {
        this.megaShader.toggleRawHdrCapture(false);
        if (wasNoise) this.megaShader.toggleDetectorNoise(true);
        if (wasVignette) this.megaShader.toggleVignette(true);
        if (wasLens) this.megaShader.toggleLensDistortion(true);
      }
    }

    // 'visible-ldr': full pipeline (every enabled effect, EOG, tone
    // mapping) but skip the final sRGB encoding so the captured
    // pixels are post-tone-mapping LINEAR LDR.
    this.megaShader.toggleLinearLdrCapture(true);
    try {
      this.runPipeline({ applyFxaa: false, finalTarget: this.ldrTarget });
      return this.readTarget(this.ldrTarget);
    } finally {
      this.megaShader.toggleLinearLdrCapture(false);
    }
  }

  private readTarget(target: THREE.WebGLRenderTarget): {
    pixels: Float32Array;
    width: number;
    height: number;
  } {
    const width = target.width;
    const height = target.height;
    const pixelCount = width * height * 4;
    const isHalfFloat = target.texture.type === THREE.HalfFloatType;

    let pixels: Float32Array;
    if (isHalfFloat) {
      const halfData = new Uint16Array(pixelCount);
      this.renderer.readRenderTargetPixels(target, 0, 0, width, height, halfData);
      pixels = halfFloatToFloat32(halfData);
    } else {
      pixels = new Float32Array(pixelCount);
      this.renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
    }
    return { pixels, width, height };
  }

  async captureHDRAsEXR(options?: {
    type?: THREE.TextureDataType;
    mode?: 'visible-ldr' | 'hdr-effects-pre-tone' | 'raw-scene-hdr';
  }): Promise<Uint8Array> {
    const exrType: THREE.TextureDataType = options?.type ?? THREE.HalfFloatType;
    const { pixels, width, height } = this.captureHDRPixels(options?.mode);

    const data: Float32Array | Uint16Array =
      exrType === THREE.HalfFloatType ? float32ToHalfFloat(pixels) : pixels;

    const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, exrType);
    texture.needsUpdate = true;

    try {
      const exporter = new EXRExporter();
      const exrData = await exporter.parse(texture, {
        type: exrType,
        compression: ZIP_COMPRESSION,
      });
      log.info(
        Modules.POST_PROCESSING,
        formatHDRExrLogLine(width, height, exrType === THREE.HalfFloatType, exrData.byteLength)
      );
      return exrData;
    } finally {
      texture.dispose();
    }
  }

  /**
   * Run a full render and read the backbuffer back as an ImageData
   * (used by screenshot/video export paths).
   */
  renderToImageData(): ImageData {
    this.render();
    // `gl.readPixels` reads from whichever framebuffer is currently
    // bound. `runPipeline` is now defensive about restoring its prior
    // render target (see save/restore inside `runPipeline`); to be
    // safe regardless of what the prior target was, bind the canvas
    // backbuffer explicitly before reading.
    this.renderer.setRenderTarget(null);
    const gl = this.renderer.getContext();
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const flipped = flipPixelsVerticallyRGBA(
      pixels,
      width,
      height
    ) as Uint8ClampedArray<ArrayBuffer>;
    return new ImageData(flipped, width, height);
  }

  // ================================================================
  // Context loss / disposal
  // ================================================================

  /**
   * Capture every uniform / define state then rebuild from scratch.
   * Used after a WebGL context-restore event.
   */
  rebuildAfterContextRestore(): void {
    if (this.disposed) return;
    log.info(Modules.POST_PROCESSING, 'Rebuilding post-processing pipeline after context restore');

    // Discard the pre-loss render-time baseline. Otherwise the first
    // post-restore frame would advance detector-noise `uTime` by the
    // (potentially long) elapsed wall-clock duration of the context
    // loss, producing a visible noise jump.
    this._previousRenderTimestamp = 0;

    // Snapshot user-facing state from the mega-shader before disposal.
    const snapshot = {
      bloomIntensity: this.bloomIntensity,
      bloomRadius: this.bloomRadius,
      bloomThreshold: this.bloomThreshold,
      bloomLevels: this.bloomLevels,
      bloomEnabled: this.isBloomEnabled(),
      toneMapping: this.megaShader.getToneMapping(),
      exposure: this.megaShader.uniforms.uExposure.value as number,
      offset: this.megaShader.uniforms.uGlobalOffset.value as number,
      gamma: this.megaShader.uniforms.uGlobalGamma.value as number,
      noiseEnabled: this.megaShader.isDetectorNoiseEnabled(),
      vignetteEnabled: this.megaShader.isVignetteEnabled(),
      vignetteDarkness: this.megaShader.uniforms.uVignetteDarkness.value as number,
      vignetteOffset: this.megaShader.uniforms.uVignetteOffset.value as number,
      lensEnabled: this.megaShader.isLensDistortionEnabled(),
      lensDistortion: (this.megaShader.uniforms.uDistortion.value as THREE.Vector2).clone(),
      lensPrincipalPoint: (this.megaShader.uniforms.uPrincipalPoint.value as THREE.Vector2).clone(),
      lensFocalLength: (this.megaShader.uniforms.uFocalLength.value as THREE.Vector2).clone(),
      lensSkew: this.megaShader.uniforms.uSkew.value as number,
      lensDispersion: this.megaShader.uniforms.uDispersion.value as number,
    };

    this.disposeTransientResources();
    // Pass applyDefaults=false: the snapshot below is authoritative for
    // every user-togglable effect. Letting initialize re-apply config
    // defaults would clobber user-disabled effects (regression vs. the
    // pre-mega-shader pipeline, which built optional effects lazily).
    this.initializeTransientResources({ applyDefaults: false });

    // Re-apply state.
    this.bloomIntensity = snapshot.bloomIntensity;
    this.bloomRadius = snapshot.bloomRadius;
    this.bloomThreshold = snapshot.bloomThreshold;
    this.bloomLevels = snapshot.bloomLevels;
    this.megaShader.setToneMapping(snapshot.toneMapping);
    this.megaShader.setExposure(snapshot.exposure);
    this.megaShader.setGlobalOffset(snapshot.offset);
    this.megaShader.setGlobalGamma(snapshot.gamma);

    if (snapshot.bloomEnabled && !this.bloomChain) {
      const { width, height } = this.getPhysicalSize();
      this.allocateBloomChain(width, height);
    }
    if (snapshot.noiseEnabled) {
      this.setDetectorNoiseEnabled(true);
    }
    if (snapshot.vignetteEnabled) {
      this.megaShader.setVignette(snapshot.vignetteDarkness, snapshot.vignetteOffset);
      this.megaShader.toggleVignette(true);
    }
    if (snapshot.lensEnabled) {
      this.megaShader.setLensDistortion({
        distortion: snapshot.lensDistortion,
        principalPoint: snapshot.lensPrincipalPoint,
        focalLength: snapshot.lensFocalLength,
        skew: snapshot.lensSkew,
        dispersion: snapshot.lensDispersion,
      });
      this.megaShader.toggleLensDistortion(true);
    }

    log.success(Modules.POST_PROCESSING, 'Post-processing pipeline rebuilt after context restore');
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.disposeTransientResources();
    log.info(Modules.POST_PROCESSING, 'PostProcessingManager disposed');
  }
}
