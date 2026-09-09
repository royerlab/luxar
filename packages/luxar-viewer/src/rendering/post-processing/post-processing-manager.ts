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
import { log, Modules } from '../../utils/log';
import { config } from '../../config';
import { type LuxarMegaShaderMaterial } from '../material-manager';
import { BloomChain } from './bloom/chain';
import { FxaaPass } from './fxaa/pass';
import { FullscreenPass } from './fullscreen/pass';
import type { Renderer, RendererCapabilities } from '../renderer-capabilities';
import { clamp } from '../../utils/clamp';
import {
  computeEffectiveSize,
  getPhysicalSize,
  getRenderTargetAllocation,
  createHdrTarget,
  buildTransientResources,
  buildBloomChain,
  disposeTransientResources,
  applyScaledNoiseSettings,
} from './post-processing-manager/resource-lifecycle';
import { runPipeline, type PipelineCtx } from './post-processing-manager/pipeline';
import type { DataRefractionSplit } from './post-processing-manager/refraction-split';
import {
  applyGlassPartition,
  collectRefractingGlass,
  collectUnpartitionedMeshes,
} from '../depth-sort-coordinator';
import {
  captureHDRPixels as captureHDRPixelsImpl,
  captureHDRAsEXR as captureHDRAsEXRImpl,
  renderToImageData as renderToImageDataImpl,
  type CaptureCtx,
  type CaptureMode,
} from './post-processing-manager/capture';
import {
  updateBloomSettings as updateBloomSettingsImpl,
  clampBloomLevels,
  validateMSAASamples,
  setVignetteEnabled as setVignetteEnabledImpl,
  setChromaticLensDistortionEnabled as setChromaticLensDistortionEnabledImpl,
  updateChromaticLensDistortion as updateChromaticLensDistortionImpl,
  getLensDistortionParams as getLensDistortionParamsImpl,
  type LensDistortionParams,
} from './post-processing-manager/settings';

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
  private megaShader!: LuxarMegaShaderMaterial;
  private megaPass!: FullscreenPass;
  private fxaaPass: FxaaPass | null = null;
  /** The scene-pass split for `refract_data` glass, both backends (null once disposed). */
  private refractionSplit: DataRefractionSplit | null = null;

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
    private renderer: Renderer,
    private capabilities: RendererCapabilities,
    private scene: THREE.Scene,
    private camera: THREE.Camera,
    size: { width: number; height: number },
    private onResize?: () => void
  ) {
    // Keep the renderer's output color space at the working space
    // (linear) so it doesn't auto-encode our output. Both the
    // mega-shader (linearToSRGB at the end of its fragment) and the
    // FXAA-out path treat the framebuffer texels as already-encoded
    // sRGB bytes; the canvas display expects exactly that.
    //
    // Why this matters: Three's `WebGPURenderer` runs an unconditional
    // "Output Color Transform" quad-pass whenever
    // `currentColorSpace !== workingColorSpace` (Renderer.js
    // `needsFrameBufferTarget` getter) and applies
    // `workingToColorSpace(outputColorSpace)`. Setting
    // outputColorSpace=sRGB here would double-encode every pixel that
    // already went through `linearToSRGB` in the mega-shader. The
    // `WebGLRenderer` doesn't double-encode only because its
    // chunk-injection path keys on `gl_FragColor` and our GLSL3 `out
    // vec4 fragColor` declaration causes the chunk to no-op; that
    // bypass is not available on the WebGPU path.
    //
    // `material.toneMapped = false` (set on the mega-shader and FXAA)
    // plays no part in that on the node path: the property appears
    // nowhere in three's `three.webgpu` build. Tone mapping there is
    // an output-pass concern keyed on `renderer.toneMapping`, which the
    // line below pins to `NoToneMapping`; the color-space branch is
    // independent of it and would fire on its own. With both lines
    // below in force `needsFrameBufferTarget` is false, so the output
    // pass — `RenderOutputNode` included — is never built at all.
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
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
  // Resource lifecycle (delegates to resource-lifecycle.ts)
  // ================================================================

  private computeEffectiveSize(): { width: number; height: number } {
    return computeEffectiveSize({
      renderer: this.renderer,
      renderSize: this.renderSize,
      ssaaEnabled: this.ssaaEnabled,
      ssaaMultiplier: this.ssaaMultiplier,
      maxPhysicalDimension: this.maxPhysicalDimension,
    });
  }

  private get maxPhysicalDimension(): number {
    return Math.min(this.capabilities.maxTextureSize, this.capabilities.maxRenderbufferSize);
  }

  private getPhysicalSize(): { width: number; height: number } {
    return getPhysicalSize({
      renderer: this.renderer,
      renderSize: this.renderSize,
      ssaaEnabled: this.ssaaEnabled,
      ssaaMultiplier: this.ssaaMultiplier,
      maxPhysicalDimension: this.maxPhysicalDimension,
    });
  }

  private getRenderTargetAllocation() {
    return getRenderTargetAllocation({
      renderer: this.renderer,
      renderSize: this.renderSize,
      ssaaEnabled: this.ssaaEnabled,
      ssaaMultiplier: this.ssaaMultiplier,
      maxPhysicalDimension: this.maxPhysicalDimension,
    });
  }

  private initializeTransientResources(
    opts: { applyDefaults: boolean } = { applyDefaults: true }
  ): void {
    const { width, height } = this.getPhysicalSize();
    const r = buildTransientResources({
      physW: width,
      physH: height,
      msaaEnabled: this.msaaEnabled,
      msaaSamples: this.msaaSamples,
      fxaaEnabled: this.fxaaEnabled,
      capabilities: this.capabilities,
      bloomLevels: this.bloomLevels,
      bloomRadius: this.bloomRadius,
      bloomThreshold: this.bloomThreshold,
      bloomIntensity: this.bloomIntensity,
      allocateBloomFromDefaults: opts.applyDefaults,
      collectRefractingGlass,
      collectUnpartitionedMeshes,
      setGlassPartition: applyGlassPartition,
    });
    this.hdrTarget = r.hdrTarget;
    this.ldrTarget = r.ldrTarget;
    this.megaShader = r.megaShader;
    this.megaPass = r.megaPass;
    this.bloomChain = r.bloomChain;
    this.fxaaPass = r.fxaaPass;
    this.refractionSplit = r.refractionSplit;

    // Apply config defaults to the mega-shader uniforms / defines.
    // Skipped during context-restore rebuilds — the caller restores user
    // toggle state from a snapshot.
    if (opts.applyDefaults) {
      this.applyConfigDefaults();
    }
  }

  private allocateBloomChain(width: number, height: number): void {
    this.bloomChain = buildBloomChain(width, height, {
      levels: this.bloomLevels,
      threshold: this.bloomThreshold,
      radius: this.bloomRadius,
      intensity: this.bloomIntensity,
      caps: this.capabilities,
      megaShader: this.megaShader,
    });
  }

  private applyConfigDefaults(): void {
    const d = config.renderingControls.defaults;
    if (d.detectorNoiseEnabled) this.setDetectorNoiseEnabled(true);
    if (d.vignetteEnabled) this.setVignetteEnabled(true);
    if (d.chromaticLensDistortionEnabled) this.setChromaticLensDistortionEnabled(true);
  }

  private disposeTransientResources(): void {
    disposeTransientResources({
      hdrTarget: this.hdrTarget,
      ldrTarget: this.ldrTarget,
      bloomChain: this.bloomChain,
      fxaaPass: this.fxaaPass,
      megaShader: this.megaShader,
      megaPass: this.megaPass,
      refractionSplit: this.refractionSplit,
    });
    this.bloomChain = null;
    this.fxaaPass = null;
    this.refractionSplit = null;
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
    const state = {
      bloomChain: this.bloomChain,
      bloomIntensity: this.bloomIntensity,
      bloomRadius: this.bloomRadius,
      bloomThreshold: this.bloomThreshold,
      bloomLevels: this.bloomLevels,
      megaShader: this.megaShader,
    };
    updateBloomSettingsImpl(state, strength, radius, threshold);
    this.bloomIntensity = state.bloomIntensity;
    this.bloomRadius = state.bloomRadius;
    this.bloomThreshold = state.bloomThreshold;
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
    const next = clampBloomLevels(
      this.bloomLevels,
      levels,
      this.bloomChain,
      this.getPhysicalSize(),
      this.bloomIntensity,
      this.megaShader
    );
    if (next !== null) this.bloomLevels = next;
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

  /**
   * The display transform the mega-shader applies after the point where
   * an EXR capture is read (`hdr-effects-pre-tone` bypasses all of it):
   * exposure → offset → gamma → tone mapping → sRGB.
   *
   * Exported so the offline EXR path can bundle an ffmpeg recipe that
   * reproduces what the viewer showed, instead of encoding scene-linear
   * floats as if they were already display-referred.
   */
  getGradeSettings(): {
    toneMapping: THREE.ToneMapping;
    exposure: number;
    offset: number;
    gamma: number;
  } {
    return {
      toneMapping: this.megaShader.getToneMapping(),
      exposure: this.megaShader.uniforms.uExposure.value as number,
      offset: this.megaShader.uniforms.uGlobalOffset.value as number,
      gamma: this.megaShader.uniforms.uGlobalGamma.value as number,
    };
  }

  // ================================================================
  // Anti-aliasing
  // ================================================================

  setFXAAEnabled(enabled: boolean): void {
    if (this.fxaaEnabled === enabled) return;
    this.fxaaEnabled = enabled;
    if (enabled && !this.fxaaPass) {
      const { width, height } = this.getPhysicalSize();
      this.fxaaPass = new FxaaPass(width, height, this.capabilities);
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
    applyScaledNoiseSettings({
      megaShader: this.megaShader,
      currentDPRScale: this.currentDPRScale,
      baseNoiseSettings: this.baseNoiseSettings,
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
    setVignetteEnabledImpl(this.megaShader, enabled, darkness, offset);
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
    setChromaticLensDistortionEnabledImpl(this.megaShader, enabled, {
      distortionX,
      distortionY,
      dispersion,
      principalPointX,
      principalPointY,
      focalLengthX,
      focalLengthY,
      skew,
    });
  }

  updateChromaticLensDistortion(params: LensDistortionParams): void {
    updateChromaticLensDistortionImpl(this.megaShader, params);
  }

  getLensDistortionParams(): {
    distortion: THREE.Vector2;
    principalPoint: THREE.Vector2;
    focalLength: THREE.Vector2;
    skew: number;
  } | null {
    return getLensDistortionParamsImpl(this.megaShader);
  }

  // ================================================================
  // MSAA + SSAA (canvas-level; reallocate HDR target on change)
  // ================================================================

  setMSAAEnabled(enabled: boolean): void {
    if (this.msaaEnabled === enabled) return;
    if (enabled) {
      const maxSamples = this.capabilities.maxMSAASamples;
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
    const validated = validateMSAASamples(samples, this.capabilities.maxMSAASamples);
    if (this.msaaSamples === validated) return;
    this.msaaSamples = validated;
    if (this.msaaEnabled) {
      this.reallocateForSize();
      log.info(Modules.POST_PROCESSING, `MSAA samples set to ${validated}`);
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

  /**
   * The SSAA factor sitting between the size passed to {@link resize}
   * and the requested render-target size. It EXCLUDES the pixel ratio.
   * A framebuffer-limit clamp may reduce the achieved physical size,
   * but limited allocations are always even in both axes. This keeps
   * the factor sufficient for the recording session's encoder alignment
   * after it forces the pixel ratio to 1.
   */
  getEffectiveRenderScale(): number {
    return this.ssaaEnabled ? this.ssaaMultiplier : 1;
  }

  /**
   * The display (CSS-pixel) size the pipeline is configured for — i.e.
   * exactly what {@link resize} was last given, and what it takes back.
   *
   * This is NOT `renderer.getSize()`. `reallocateForSize` hands the
   * renderer the SSAA-MULTIPLIED size and stamps the display size on
   * the canvas CSS instead, so under SSAA the renderer reports
   * `display × ssaaMultiplier`. Feeding that back into {@link resize}
   * multiplies by the SSAA factor a second time.
   */
  getDisplaySize(): { width: number; height: number } {
    return { ...this.renderSize };
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

    this.pipeline({ applyFxaa: this.fxaaPass !== null, finalTarget: null });
  }

  private pipelineCtx(): PipelineCtx {
    return {
      renderer: this.renderer,
      scene: this.scene,
      camera: this.camera,
      hdrTarget: this.hdrTarget,
      ldrTarget: this.ldrTarget,
      megaShader: this.megaShader,
      megaPass: this.megaPass,
      bloomChain: this.bloomChain,
      fxaaPass: this.fxaaPass,
      refractionSplit: this.refractionSplit,
    };
  }

  private pipeline(opts: {
    applyFxaa: boolean;
    finalTarget: THREE.WebGLRenderTarget | null;
  }): void {
    runPipeline(this.pipelineCtx(), opts);
  }

  private captureCtx(): CaptureCtx {
    return {
      renderer: this.renderer,
      capabilities: this.capabilities,
      scene: this.scene,
      camera: this.camera,
      hdrTarget: this.hdrTarget,
      ldrTarget: this.ldrTarget,
      megaShader: this.megaShader,
      pipelineCtx: this.pipelineCtx(),
    };
  }

  // ================================================================
  // Resize
  // ================================================================

  resize(width: number, height: number): void {
    this.renderSize = { width, height };
    this.reallocateForSize();
  }

  /**
   * Memo of the last applied allocation. Every window resize reaches
   * `reallocateForSize` through TWO paths — the window `resize` listener
   * (rAF-coalesced via ResizeOrchestrator) AND the canvas-parent
   * ResizeObserver — and the MSAA/SSAA setters call it directly too.
   * Without a guard, each redundant call disposed + recreated the
   * full-screen half-float HDR target. The memo keys on everything that
   * influences the allocation: display (CSS) size, effective logical
   * size (folds in SSAA), physical size (folds in DPR), and the HDR
   * target's MSAA sample count. `null` forces the next call to
   * reallocate (initial state; reset by `rebuildAfterContextRestore`).
   */
  private lastAllocation: {
    displayW: number;
    displayH: number;
    logicalW: number;
    logicalH: number;
    physW: number;
    physH: number;
    msaaSamples: number;
  } | null = null;
  private wasAllocationLimited = false;

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
   *
   * A call whose full allocation key matches the last applied one is a
   * complete no-op (see `lastAllocation`).
   */
  private reallocateForSize(): void {
    const allocation = this.getRenderTargetAllocation();
    const { width: logicalW, height: logicalH } = allocation.logical;
    const { width: physW, height: physH } = allocation.physical;
    const msaaSamples = this.msaaEnabled ? this.msaaSamples : 0;

    const last = this.lastAllocation;
    if (
      last &&
      last.displayW === this.renderSize.width &&
      last.displayH === this.renderSize.height &&
      last.logicalW === logicalW &&
      last.logicalH === logicalH &&
      last.physW === physW &&
      last.physH === physH &&
      last.msaaSamples === msaaSamples
    ) {
      return; // identical allocation — the redundant resize path lands here
    }
    // All render targets at PHYSICAL size — matches canvas backbuffer
    // and the resolution materials read from `getDrawingBufferSize()`.
    // Resize them before the renderer: WebGPURenderer dispatches a
    // synchronous resize event from setSize(), so observers must never
    // see a new backbuffer paired with stale post-processing targets.
    this.hdrTarget.dispose();
    this.hdrTarget = createHdrTarget(physW, physH, this.msaaEnabled ? this.msaaSamples : 0);

    this.ldrTarget.setSize(physW, physH);
    if (this.bloomChain) {
      this.bloomChain.setSize(physW, physH);
      // setSize may have reallocated the mip targets — rebind the
      // (possibly new) bloom texture identity into the mega-shader.
      this.megaShader.setBloom(this.bloomIntensity, this.bloomChain.outputTexture);
    }
    this.fxaaPass?.setSize(physW, physH);
    this.refractionSplit?.setSize(physW, physH);
    this.megaShader.setResolution(physW, physH);

    this.lastAllocation = {
      displayW: this.renderSize.width,
      displayH: this.renderSize.height,
      logicalW,
      logicalH,
      physW,
      physH,
      msaaSamples,
    };

    // Renderer takes logical size; it multiplies by pixelRatio for the
    // canvas backbuffer. SceneManager owns CSS sizing.
    this.renderer.setSize(logicalW, logicalH, false);
    this.renderer.domElement.style.width = `${this.renderSize.width}px`;
    this.renderer.domElement.style.height = `${this.renderSize.height}px`;

    if (allocation.limited && !this.wasAllocationLimited) {
      const requested = this.computeEffectiveSize();
      const dpr = this.renderer.getPixelRatio();
      log.warning(
        Modules.POST_PROCESSING,
        `Render target ${Math.floor(requested.width * dpr)}x${Math.floor(requested.height * dpr)} ` +
          `exceeds the ${this.maxPhysicalDimension}px framebuffer limit; ` +
          `using ${physW}x${physH}`
      );
    } else if (!allocation.limited && this.wasAllocationLimited) {
      log.update(Modules.POST_PROCESSING, 'Render target is back within framebuffer limits');
    }
    this.wasAllocationLimited = allocation.limited;

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
   *
   * Returned rows are in canonical **top-down** order (row 0 = top of
   * the source target) on both backends. Pass `opts.flipY = true` to
   * receive bottom-up rows instead — only the EXR exporter does this,
   * to preserve the orientation external tools expect.
   */
  async captureHDRPixels(
    mode: CaptureMode = 'hdr-effects-pre-tone',
    opts: { flipY?: boolean } = {}
  ): Promise<{ pixels: Float32Array; width: number; height: number }> {
    return captureHDRPixelsImpl(this.captureCtx(), mode, opts);
  }

  async captureHDRAsEXR(options?: {
    type?: THREE.TextureDataType;
    mode?: CaptureMode;
  }): Promise<Uint8Array> {
    return captureHDRAsEXRImpl(this.captureCtx(), options);
  }

  async renderToImageData(): Promise<ImageData> {
    // Mirror render()'s detector-noise time advance — capture is a
    // frame in its own right.
    const now = performance.now();
    const dt =
      this._previousRenderTimestamp === 0 ? 0 : (now - this._previousRenderTimestamp) / 1000;
    this._previousRenderTimestamp = now;
    if (this.megaShader.isDetectorNoiseEnabled()) {
      this.megaShader.advanceTime(dt);
    }
    return renderToImageDataImpl(this.captureCtx(), this.getPhysicalSize(), this.fxaaPass !== null);
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
    // All transient targets were just rebuilt against the fresh context —
    // invalidate the allocation memo so the restore path's follow-up
    // updateRendererSize() → resize() is not skipped as a no-op.
    this.lastAllocation = null;

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
