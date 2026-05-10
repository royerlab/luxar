/**
 * PostProcessing Manager using pmndrs/postprocessing library
 *
 * This module provides a modern, high-performance post-processing pipeline
 * with HDR support, advanced effects, and optimized render passes.
 */

import {
  EffectComposer,
  RenderPass,
  EffectPass,
  BloomEffect,
  DepthOfFieldEffect,
  Effect,
  ToneMappingMode,
  SMAAEffect,
  SMAAPreset,
  FXAAEffect,
  SSAOEffect,
} from 'postprocessing';
import { LuxarToneMappingEffect } from './luxar-tone-mapping-effect';
import { DetectorNoiseEffect, isDetectorNoiseEffect } from './detector-noise-effect';
import { RobustVignetteEffect, isRobustVignetteEffect } from './robust-vignette-effect';
import {
  ChromaticLensDistortionEffect,
  isChromaticLensDistortionEffect,
} from './chromatic-lens-distortion-effect';
import * as THREE from 'three';
import { EXRExporter, ZIP_COMPRESSION } from 'three/examples/jsm/exporters/EXRExporter.js';
import { log, Modules } from '../../utils/log';
import { config } from '../../config';
import {
  BloomEffectTyped,
  DepthOfFieldEffectTyped,
  PerspectiveDepthMapper,
  isBloomEffectTyped,
  isDepthOfFieldEffectTyped,
} from './postprocessing-types';
import { safeDisposeEffect, safeRemoveAndDisposePass } from './effect-disposal';
import { computeEffectiveRenderSize } from './render-target-sizing';
import {
  halfFloatToFloat32,
  float32ToHalfFloat,
  flipPixelsVerticallyRGBA,
} from './hdr-pixel-utils';
import { estimatePostProcMemoryMB, formatHDRExrLogLine, pickResultBuffer } from './hdr-capture';
import {
  type PostProcessingDurableState,
  applyBloomState,
  applyChromaticLensDistortionState,
  applyDOFFocusDistance,
  applyDetectorNoiseState,
  applyToneMappingState,
  applyVignetteState,
  captureBloomState,
  captureChromaticLensDistortionState,
  captureDOFState,
  captureDetectorNoiseState,
  captureToneMappingState,
  captureVignetteState,
} from './context-recovery';
import { buildOrderedEffects, partitionEffectsIntoPasses } from './effect-orchestrator';
import { toneMappingModeName } from './tone-mapping-mode-names';
import {
  applyToneMapping,
  readToneMapping,
  applyExposure,
  applyGlobalOffset,
  applyGlobalGamma,
} from './tone-mapping-handler';
import {
  applyBloomRadius,
  applyBloomSettings,
  buildBloomConstructorOptions,
  clampBloomLevels,
  readBloomSettings,
  resolveBloomSettings,
} from './bloom-handler';
import {
  checkMSAACapability,
  clampSSAAMultiplier,
  mapSMAAPreset,
  validateMSAASamples,
} from './antialiasing-handler';
import {
  DOF_DEFAULT_FOCUS,
  DOF_DEFAULT_STRENGTH,
  clampDPRScale,
  computeDOFFocalLength,
  dprScaleChanged,
  mergeVector2,
  resolveChromaticDistortionDefaults,
  resolveNoiseDefaults,
  resolveVignetteDefaults,
  scaleNoiseSettings,
  strengthToBokehScale,
} from './visual-effects-handler';

/**
 * AO quality preset → SSAOEffect (samples, radius). Pulled out as a
 * module-level helper so both `setAOEnabled(true, q)` and
 * `setAOQuality(q)` share a single source of truth.
 */
function aoQualitySettings(quality: 'low' | 'medium' | 'high' | 'ultra'): {
  samples: number;
  radius: number;
} {
  switch (quality) {
    case 'low':
      return { samples: 4, radius: 0.1 };
    case 'medium':
      return { samples: 8, radius: 0.2 };
    case 'high':
      return { samples: 16, radius: 0.3 };
    case 'ultra':
      return { samples: 32, radius: 0.4 };
  }
}

/**
 * Manages HDR post-processing effects using pmndrs/postprocessing library.
 *
 * Coordinates the complete post-processing pipeline:
 * - HDR rendering with 16-bit float buffers
 * - Bloom effect with mipmap blur
 * - ACES filmic tone mapping
 * - Anti-aliasing (FXAA, SMAA, MSAA, SSAA)
 * - Visual effects (DOF, chromatic lens distortion, vignette, detector noise)
 * - Dynamic effect management with deferred rebuild
 *
 * The manager handles effect lifecycle, settings persistence, and optimized
 * rebuilding to minimize performance impact when changing multiple settings.
 *
 * @example
 * ```typescript
 * const postProcessing = new PostProcessingManager(
 *   renderer,
 *   scene,
 *   camera,
 *   { width: window.innerWidth, height: window.innerHeight }
 * );
 *
 * // Enable bloom
 * postProcessing.updateBloomSettings(1.5, 0.4, 0.85);
 *
 * // Render each frame
 * requestAnimationFrame(() => postProcessing.render());
 * ```
 */
export class PostProcessingManager {
  // Definite-assignment: these are initialized via the constructor's call
  // to `initializeTransientResources()` — TS can't trace through the helper.
  // They are also reassigned by `rebuildAfterContextRestore()`.
  private composer!: EffectComposer;
  private renderPass!: RenderPass;
  private effectPass?: EffectPass;
  private secondaryPass?: EffectPass;

  // Individual effect references for runtime updates with proper typing
  private bloomEffect?: BloomEffectTyped;
  private dofEffect?: DepthOfFieldEffectTyped;
  private toneMappingEffect?: LuxarToneMappingEffect;
  private smaaEffect?: SMAAEffect;
  private fxaaEffect?: FXAAEffect;
  private aoEffect?: SSAOEffect;
  /** tracks current AO quality for in-place updates and debug state. */
  private _aoQuality: 'low' | 'medium' | 'high' | 'ultra' = 'medium';
  private vignetteEffect?: RobustVignetteEffect;
  private detectorNoiseEffect?: DetectorNoiseEffect;
  private chromaticLensDistortionEffect?: ChromaticLensDistortionEffect;

  // State tracking
  private fxaaEnabled: boolean = false;
  private smaaEnabled: boolean = false;
  private msaaEnabled: boolean = false;
  private msaaSamples: number = 0;
  private ssaaEnabled: boolean = false;
  private ssaaMultiplier: number = 1.0;
  private renderSize: { width: number; height: number };

  // Rebuild control - prevents redundant rebuilds during bulk changes
  private deferRebuild: boolean = false;

  // Idempotency guard for dispose(); prevents double-dispose of the composer
  // and the underlying GPU resources when shutdown paths overlap.
  private disposed: boolean = false;

  // DPR-based noise scaling
  // Store base (user-configured) noise values separately from DPR-scaled effective values
  private baseNoiseSettings = {
    readoutSigma: config.renderingControls.defaults.detectorNoiseReadoutSigma,
    photonGain: config.renderingControls.defaults.detectorNoisePhotonGain,
    fpnSigma: config.renderingControls.defaults.detectorNoiseFpnSigma,
  };
  private currentDPRScale: number = 1.0;

  // Performance features
  private bloomLevels: number = config.renderingControls.defaults.bloomLevels;
  private _qualityPreset: 'low' | 'medium' | 'high' | 'ultra' = 'medium';
  private performanceMetrics = {
    lastFrameTime: 0,
    avgFrameTime: 0,
    frameCount: 0,
  };

  /**
   * Create post-processing manager with HDR pipeline.
   *
   * Initializes EffectComposer with 16-bit float buffers, sets up initial
   * effects (bloom, tone mapping, AA), and configures renderer for linear
   * workflow with sRGB output.
   *
   * @param renderer - THREE.js WebGL renderer (will be configured for post-processing)
   * @param scene - THREE.js scene to render
   * @param camera - Camera for rendering and depth-based effects
   * @param size - Initial render size {width, height} in pixels
   */
  constructor(
    private renderer: THREE.WebGLRenderer,
    private scene: THREE.Scene,
    private camera: THREE.Camera,
    size: { width: number; height: number }
  ) {
    // Configure renderer for postprocessing library
    // IMPORTANT: Use SRGBColorSpace for correct color output
    // The pmndrs library handles linear workflow internally and outputs to sRGB
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping; // Let postprocessing handle tone mapping

    // Store render size for SSAA calculations
    this.renderSize = { ...size };

    // Initialize AA settings from config BEFORE using them
    this.ssaaEnabled = config.renderingControls.defaults.ssaaEnabled;
    this.ssaaMultiplier = config.renderingControls.defaults.ssaaMultiplier;
    this.msaaEnabled = config.renderingControls.defaults.msaaEnabled;
    this.msaaSamples = config.renderingControls.defaults.msaaSamples;
    this.fxaaEnabled = config.renderingControls.defaults.fxaaEnabled;
    this.smaaEnabled = config.renderingControls.defaults.smaaEnabled;

    // Build composer + render pass + initial effects, then wire the effect
    // pass. Both the constructor and rebuildAfterContextRestore() share this
    // path so context-restore stays a single source of truth for transient
    // resource construction.
    this.initializeTransientResources();
    this.rebuildEffectPass();

    log.success(
      Modules.POST_PROCESSING,
      `pmndrs/postprocessing initialized - Output: ${size.width}x${size.height}, ` +
        `Render: ${this.computeEffectiveSize().width}x${this.computeEffectiveSize().height}` +
        `${this.ssaaEnabled ? ' (SSAA)' : ''}, ` +
        `MSAA: ${this.msaaEnabled ? this.msaaSamples + 'x' : 'off'}`
    );
  }

  /**
   * Compute the effective render-target size given current SSAA settings.
   * Delegates to the pure helper so the SSAA arithmetic is unit-testable.
   */
  private computeEffectiveSize(): { width: number; height: number } {
    return computeEffectiveRenderSize(this.renderSize, this.ssaaEnabled, this.ssaaMultiplier);
  }

  /**
   * (Re)create all GPU-bound transient resources: composer, render pass,
   * and the initial set of effects. Caller is responsible for invoking
   * `rebuildEffectPass()` afterwards to wire the effect chain.
   *
   * Used by both the constructor and `rebuildAfterContextRestore()`.
   */
  private initializeTransientResources(): void {
    const { width, height } = this.computeEffectiveSize();

    // Create composer with HDR support using 16-bit float buffers
    this.composer = new EffectComposer(this.renderer, {
      frameBufferType: THREE.HalfFloatType,
      multisampling: this.msaaEnabled ? this.msaaSamples : 0,
    });
    this.composer.setSize(width, height);

    // Initialize render pass
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // Create initial effects (bloom, toneMapping, smaa, fxaa); optional
    // effects are added later via setXxxEnabled() toggles.
    this.createInitialEffects();
  }

  /**
   * Creates the initial set of effects with default settings
   */
  private createInitialEffects(): void {
    // Bloom effect with HDR support — using config defaults via the shared
    // bloom-handler helpers so the constructor-options table has a single
    // source of truth.
    const initialBloomSettings = resolveBloomSettings();
    this.bloomEffect = new BloomEffect(
      buildBloomConstructorOptions(initialBloomSettings, this.bloomLevels)
    ) as BloomEffectTyped;
    applyBloomRadius(
      this.bloomEffect as unknown as { mipmapBlurPass?: { radius: number } },
      initialBloomSettings.radius
    );

    log.info(
      Modules.POST_PROCESSING,
      `Bloom initialized with ${config.renderingControls.defaults.bloomLevels} mipmap levels, radius: ${config.renderingControls.defaults.bloomRadius}`
    );

    // Tone mapping for HDR to LDR conversion with Luxar EOG (Exposure-Offset-Gamma)
    this.toneMappingEffect = new LuxarToneMappingEffect({
      mode: ToneMappingMode.ACES_FILMIC,
      resolution: 256,
      whitePoint: 2.0, // Standard white point
      middleGrey: 0.4, // Lower middle grey for brighter output
      minLuminance: 0.001, // Lower min for better dark detail
      averageLuminance: 1.0,
      adaptationRate: 1.0,
      // EOG defaults: neutral (no change)
      exposure: 0.0,
      globalOffset: 0.0,
      globalGamma: 1.0,
    });

    // SMAA for high-quality anti-aliasing (disabled by default)
    this.smaaEffect = new SMAAEffect({
      preset: SMAAPreset.HIGH,
    });

    // FXAA as a faster alternative (disabled by default)
    this.fxaaEffect = new FXAAEffect();

    log.info(
      Modules.POST_PROCESSING,
      'Initial effects created: Bloom, ToneMapping, SMAA (disabled), FXAA (disabled)'
    );
  }

  /**
   * Start deferred rebuild mode (batch multiple effect changes).
   *
   * Prevents automatic effect pass rebuilding until endDeferRebuild().
   * Use when changing multiple settings to avoid redundant rebuilds.
   *
   * @example
   * ```typescript
   * // Change multiple settings efficiently
   * postProcessing.startDeferRebuild();
   * postProcessing.updateBloomSettings(1.5);
   * postProcessing.setVignetteEnabled(true);
   * postProcessing.setChromaticLensDistortionEnabled(true, -0.05, -0.05, 0.03);
   * postProcessing.endDeferRebuild();  // Single rebuild
   * ```
   */
  startDeferRebuild(): void {
    this.deferRebuild = true;
  }

  /**
   * End deferred rebuild mode and trigger single effect pass rebuild.
   *
   * Rebuilds effect pass with all changes applied. Always call after
   * startDeferRebuild() to apply batched changes.
   */
  endDeferRebuild(): void {
    this.deferRebuild = false;
    this.rebuildEffectPass();
  }

  /**
   * Replace the active camera (e.g., when switching between perspective and orthographic).
   * Updates the render pass and rebuilds effect passes that hold camera references.
   */
  setCamera(camera: THREE.Camera): void {
    this.camera = camera;
    if (this.renderPass) {
      // postprocessing's RenderPass.mainCamera is mutable but not typed
      // as such in the public d.ts; structural cast targets just that field.
      (this.renderPass as unknown as { mainCamera: THREE.Camera }).mainCamera = camera;
    }
    // Effect passes capture the camera at construction — rebuild them
    this.rebuildEffectPass();
  }

  /** Dispose every effect object owned by this manager. */
  private disposeAllEffects(context: string): void {
    safeDisposeEffect(this.bloomEffect, `Bloom ${context}`);
    safeDisposeEffect(this.detectorNoiseEffect, `DetectorNoise ${context}`);
    safeDisposeEffect(this.dofEffect, `DOF ${context}`);
    safeDisposeEffect(this.aoEffect, `AmbientOcclusion ${context}`);
    safeDisposeEffect(this.vignetteEffect, `Vignette ${context}`);
    safeDisposeEffect(this.chromaticLensDistortionEffect, `ChromaticLensDistortion ${context}`);
    safeDisposeEffect(this.smaaEffect, `SMAA ${context}`);
    safeDisposeEffect(this.fxaaEffect, `FXAA ${context}`);
    safeDisposeEffect(this.toneMappingEffect, `ToneMapping ${context}`);
  }

  /**
   * Rebuilds the effect pass with currently active effects
   * This is called when effects are added/removed or toggled
   *
   * IMPORTANT: Uses 3-pass architecture to handle effect incompatibilities:
   * Pass 1 (HDR): Compatible effects before tone mapping
   * Pass 2 (HDR): UV transformation effects (LensDistortion) before tone mapping
   * Pass 3 (LDR): Tone mapping + convolution effects + anti-aliasing
   */
  private rebuildEffectPass(): void {
    // Skip rebuild if in deferred mode
    if (this.deferRebuild) {
      return;
    }
    // Remove old effect passes (if present) and dispose their GPU resources.
    safeRemoveAndDisposePass(this.composer, this.effectPass, 'effectPass');
    this.effectPass = undefined;
    safeRemoveAndDisposePass(this.composer, this.secondaryPass, 'secondaryPass');
    this.secondaryPass = undefined;

    // Build the canonical ordered list of active effects from the
    // current effect references + AA flags. The helper is pure; the
    // partitioner below splits the result into Pass A / Pass B.
    const orderedEffects = buildOrderedEffects<Effect>({
      bloom: this.bloomEffect,
      dof: this.dofEffect,
      ao: this.aoEffect,
      chromaticLensDistortion: this.chromaticLensDistortionEffect,
      detectorNoise: this.detectorNoiseEffect,
      toneMapping: this.toneMappingEffect,
      vignette: this.vignetteEffect,
      smaa: this.smaaEffect,
      fxaa: this.fxaaEffect,
      smaaEnabled: this.smaaEnabled,
      fxaaEnabled: this.fxaaEnabled,
    });

    // Vignette logging (preserved from the inline version).
    if (this.vignetteEffect) {
      log.info(
        Modules.POST_PROCESSING,
        `Vignette added to effects (after tone mapping) - darkness=${this.vignetteEffect.darkness}, offset=${this.vignetteEffect.offset}`
      );
    }

    // Partition into Pass A / Pass B based on pmndrs effect-compatibility rules.
    const partition = partitionEffectsIntoPasses(orderedEffects);
    const passAEffects = partition.passA;
    const passBEffects = partition.passB;
    const passANames = partition.passANames;
    const passBNames = partition.passBNames;
    if (partition.splitAt) {
      log.info(
        Modules.POST_PROCESSING,
        `Effect incompatibility detected at ${partition.splitAt}, switching to Pass B`
      );
    }

    // Create Pass A with error recovery (always created if we have effects)
    if (passAEffects.length > 0) {
      try {
        this.effectPass = new EffectPass(this.camera, ...passAEffects);
        this.composer.addPass(this.effectPass);
      } catch (error) {
        log.error(Modules.POST_PROCESSING, `Failed to create Pass A: ${error}`);
        safeDisposeEffect(this.effectPass, 'Pass A construction failure');
        this.effectPass = undefined;
        throw error;
      }
    }

    // Create Pass B with error recovery (only if needed)
    if (passBEffects.length > 0) {
      try {
        this.secondaryPass = new EffectPass(this.camera, ...passBEffects);
        this.composer.addPass(this.secondaryPass);
      } catch (error) {
        log.error(Modules.POST_PROCESSING, `Failed to create Pass B: ${error}`);
        safeDisposeEffect(this.secondaryPass, 'Pass B construction failure');
        this.secondaryPass = undefined;
        // B.4: Pass A was already added to the composer; tear it down too
        // so the partial pipeline doesn't run forever in a half-built
        // state. This is the only place where partial state survives the
        // per-pass try/catch.
        if (this.effectPass) {
          safeRemoveAndDisposePass(this.composer, this.effectPass, 'effectPass (rollback)');
          this.effectPass = undefined;
        }
        throw error;
      }
    }

    const totalEffects = passAEffects.length + passBEffects.length;
    if (passBEffects.length > 0) {
      log.info(
        Modules.POST_PROCESSING,
        `Dynamic passes created - Pass A: [${passANames.join(', ')}], Pass B: [${passBNames.join(', ')}] (${totalEffects} total)`
      );
    } else {
      log.info(
        Modules.POST_PROCESSING,
        `Single pass created: [${passANames.join(', ')}] (${totalEffects} effects)`
      );
    }
  }

  /**
   * Update bloom effect parameters.
   *
   * Controls HDR bloom glow intensity, spread, and brightness threshold.
   * Omitted parameters retain current values.
   *
   * @param strength - Bloom intensity (0-5, default 1.0). Higher = more glow
   * @param radius - Bloom spread radius (0-1, default 0.4). Higher = wider glow
   * @param threshold - Luminance threshold (0-1, default 0.85). Higher = only brightest pixels bloom
   */
  updateBloomSettings(strength?: number, radius?: number, threshold?: number): void {
    applyBloomSettings(this.bloomEffect, { intensity: strength, radius, threshold });
  }

  /**
   * Enable or disable bloom effect.
   *
   * When disabled, the bloom effect is removed from the pipeline to save GPU cycles.
   * When re-enabled, the effect is recreated with current settings.
   *
   * @param enabled - Whether to enable bloom
   * @param strength - Bloom intensity (optional, uses current/default if not provided)
   * @param radius - Bloom spread radius (optional)
   * @param threshold - Luminance threshold (optional)
   */
  setBloomEnabled(enabled: boolean, strength?: number, radius?: number, threshold?: number): void {
    if (enabled && !this.bloomEffect) {
      const settings = resolveBloomSettings({ intensity: strength, radius, threshold });
      this.bloomEffect = new BloomEffect(
        buildBloomConstructorOptions(settings, this.bloomLevels)
      ) as BloomEffectTyped;
      applyBloomRadius(
        this.bloomEffect as unknown as { mipmapBlurPass?: { radius: number } },
        settings.radius
      );

      this.rebuildEffectPass();
      log.success(Modules.POST_PROCESSING, 'Bloom enabled');
    } else if (!enabled && this.bloomEffect) {
      safeDisposeEffect(this.bloomEffect, 'Bloom');
      this.bloomEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Bloom disabled');
    } else if (enabled && this.bloomEffect) {
      this.updateBloomSettings(strength, radius, threshold);
    }
  }

  /**
   * Check if bloom effect is currently enabled.
   */
  isBloomEnabled(): boolean {
    return !!this.bloomEffect;
  }

  /**
   * Set HDR tone mapping algorithm.
   *
   * Converts HDR scene values to displayable LDR range. Different algorithms
   * produce different aesthetic results.
   *
   * @param mode - Tone mapping mode (NoToneMapping, LinearToneMapping, ReinhardToneMapping,
   *               CineonToneMapping, ACESFilmicToneMapping, AgXToneMapping, NeutralToneMapping)
   */
  setToneMapping(mode: THREE.ToneMapping): void {
    applyToneMapping(this.toneMappingEffect, mode);
  }

  /**
   * Get current tone mapping mode.
   *
   * @returns Active tone mapping algorithm
   */
  getToneMapping(): THREE.ToneMapping {
    return readToneMapping(this.toneMappingEffect);
  }

  // ======================================================================
  // Global EOG (Exposure-Offset-Gamma) controls
  // ======================================================================

  /**
   * Update global exposure (log2 stops).
   * 0 = neutral, +1 = 2x brighter, -1 = half.
   */
  updateExposure(value: number): void {
    applyExposure(this.toneMappingEffect, value);
  }

  /**
   * Update global offset (additive brightness shift).
   */
  updateGlobalOffset(value: number): void {
    applyGlobalOffset(this.toneMappingEffect, value);
  }

  /**
   * Update global gamma correction.
   */
  updateGlobalGamma(value: number): void {
    applyGlobalGamma(this.toneMappingEffect, value);
  }

  /**
   * Enable or disable FXAA (Fast Approximate Anti-Aliasing).
   *
   * FXAA is fast but lower quality. Good for performance-constrained scenarios.
   *
   * @param enabled - true to enable FXAA, false to disable
   */
  setFXAAEnabled(enabled: boolean): void {
    this.fxaaEnabled = enabled;
    this.rebuildEffectPass();
    log.update(Modules.POST_PROCESSING, `FXAA ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Gets FXAA enabled state
   */
  isFXAAEnabled(): boolean {
    return this.fxaaEnabled;
  }

  /**
   * Enables or disables SMAA anti-aliasing
   */
  setSMAAEnabled(enabled: boolean): void {
    this.smaaEnabled = enabled;
    this.rebuildEffectPass();
    log.update(Modules.POST_PROCESSING, `SMAA ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Updates SMAA settings (preset quality)
   */
  updateSMAASettings(preset?: 'LOW' | 'MEDIUM' | 'HIGH' | 'ULTRA'): void {
    if (!this.smaaEffect || !preset) return;

    // B.3: dispose AFTER rebuild — same rationale as setAOQuality.
    const oldEffect = this.smaaEffect;
    this.smaaEffect = new SMAAEffect({ preset: mapSMAAPreset(preset) });

    if (this.smaaEnabled) {
      this.rebuildEffectPass();
    }

    safeDisposeEffect(oldEffect, 'SMAA (preset change)');
    log.update(Modules.POST_PROCESSING, `SMAA quality set to: ${preset}`);
  }

  /**
   * Sets depth of field effect with proper perspective depth mapping
   */
  setDOF(enabled: boolean, focus?: number, strength?: number): void {
    if (enabled && !this.dofEffect && this.camera instanceof THREE.PerspectiveCamera) {
      const focusDistance = focus ?? DOF_DEFAULT_FOCUS;
      const normalizedFocus = PerspectiveDepthMapper.worldToNormalizedDepth(
        focusDistance,
        this.camera.near,
        this.camera.far
      );
      const focalLength = computeDOFFocalLength(this.camera.fov);

      this.dofEffect = new DepthOfFieldEffect(this.camera, {
        focusDistance: normalizedFocus,
        focalLength: focalLength,
        bokehScale: strengthToBokehScale(strength ?? DOF_DEFAULT_STRENGTH),
        height: 480,
      }) as DepthOfFieldEffectTyped;

      this.rebuildEffectPass();

      log.info(
        Modules.POST_PROCESSING,
        `DOF enabled: focus=${focusDistance.toFixed(2)} (normalized: ${normalizedFocus.toFixed(3)}), ` +
          `strength=${strength}, focalLength=${focalLength.toFixed(3)}`
      );
    } else if (!enabled && this.dofEffect) {
      safeDisposeEffect(this.dofEffect, 'DOF');
      this.dofEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'DOF disabled');
    }
  }

  /**
   * Updates DOF parameters with proper depth mapping
   */
  updateDOF(params: { focus?: number; strength?: number }): void {
    if (!this.dofEffect) {
      log.warning(Modules.POST_PROCESSING, 'DOF effect not initialized');
      return;
    }

    if (!isDepthOfFieldEffectTyped(this.dofEffect)) {
      log.error(Modules.POST_PROCESSING, 'Invalid DOF effect type');
      return;
    }

    if (this.camera instanceof THREE.PerspectiveCamera) {
      if (
        params.focus !== undefined &&
        this.dofEffect.circleOfConfusionMaterial?.uniforms?.focusDistance
      ) {
        // Use proper perspective depth mapping
        const normalizedFocus = PerspectiveDepthMapper.worldToNormalizedDepth(
          params.focus,
          this.camera.near,
          this.camera.far
        );
        this.dofEffect.circleOfConfusionMaterial.uniforms.focusDistance.value = normalizedFocus;

        log.update(
          Modules.POST_PROCESSING,
          `DOF focus updated: ${params.focus.toFixed(2)} (normalized: ${normalizedFocus.toFixed(3)})`
        );
      }
    }

    if (params.strength !== undefined) {
      this.dofEffect.bokehScale = strengthToBokehScale(params.strength);
      log.update(Modules.POST_PROCESSING, `DOF strength updated: ${params.strength}`);
    }
  }

  /**
   * Sets physics-based detector noise effect
   *
   * This provides realistic camera/detector noise simulation with three components:
   * - Shot noise (Poisson): Signal-dependent noise from photon statistics
   * - Readout noise (Gaussian, temporal): Signal-independent electronic noise, varies per frame
   * - Fixed Pattern Noise (Gaussian, static): Per-pixel offset from detector non-uniformities
   *
   * @param enabled - Whether to enable the effect
   * @param readoutSigma - Temporal readout noise sigma (0-0.1 typical), default 0.01
   * @param photonGain - Photon gain controlling shot noise visibility (0.0001-0.1 typical), default 0.01
   * @param fpnSigma - Fixed pattern noise sigma (0-0.05 typical), default 0.005
   */
  setDetectorNoiseEnabled(
    enabled: boolean,
    readoutSigma?: number,
    photonGain?: number,
    fpnSigma?: number
  ): void {
    if (enabled && !this.detectorNoiseEffect) {
      const settings = resolveNoiseDefaults({ readoutSigma, photonGain, fpnSigma });
      // Persist the user-configured base settings so subsequent setDPRScale()
      // calls can derive the scaled effect values from the right baseline,
      // not from the class-field defaults.
      this.baseNoiseSettings = { ...settings };
      this.detectorNoiseEffect = new DetectorNoiseEffect(settings);
      // Apply DPR scaling immediately if a non-1.0 DPR is already in effect.
      this.applyScaledNoiseSettings();

      this.rebuildEffectPass();
      log.info(
        Modules.POST_PROCESSING,
        `Detector noise enabled: readout=${settings.readoutSigma}, gain=${settings.photonGain}, fpn=${settings.fpnSigma}`
      );
    } else if (!enabled && this.detectorNoiseEffect) {
      safeDisposeEffect(this.detectorNoiseEffect, 'DetectorNoise');
      this.detectorNoiseEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Detector noise disabled');
    }
  }

  /**
   * Updates detector noise parameters
   *
   * @param params - Object containing parameters to update
   * @param params.readoutSigma - Temporal readout noise sigma (Gaussian)
   * @param params.photonGain - Photon gain controlling shot noise visibility
   * @param params.fpnSigma - Fixed pattern noise sigma (static per-pixel)
   */
  updateDetectorNoiseSettings(params: {
    readoutSigma?: number;
    photonGain?: number;
    fpnSigma?: number;
  }): void {
    // Store base (user-configured) values
    if (params.readoutSigma !== undefined) {
      this.baseNoiseSettings.readoutSigma = params.readoutSigma;
    }
    if (params.photonGain !== undefined) {
      this.baseNoiseSettings.photonGain = params.photonGain;
    }
    if (params.fpnSigma !== undefined) {
      this.baseNoiseSettings.fpnSigma = params.fpnSigma;
    }

    // Apply DPR-scaled values to the effect
    this.applyScaledNoiseSettings();

    log.update(
      Modules.POST_PROCESSING,
      `Detector noise updated: ${Object.keys(params)
        .map((k) => `${k}=${params[k as keyof typeof params]}`)
        .join(', ')} (DPR scale: ${this.currentDPRScale.toFixed(2)})`
    );
  }

  /**
   * Apply DPR-scaled noise settings to the effect
   *
   * When DPR < 1, we render at lower resolution and upscale. Each rendered pixel
   * covers 1/DPR² screen pixels. To maintain perceptual consistency, noise sigma
   * should be scaled by DPR (since averaging N noisy samples reduces sigma by √N,
   * and here each sample covers 1/DPR² pixels, so sigma_effective = sigma × DPR).
   *
   * For Gaussian noise (readoutSigma, fpnSigma): σ_effective = σ × DPR
   *
   * For shot noise (photonGain): The output σ ∝ √photonGain, so to scale σ by DPR,
   * we need photonGain_effective = photonGain × DPR²
   */
  private applyScaledNoiseSettings(): void {
    if (!this.detectorNoiseEffect || !isDetectorNoiseEffect(this.detectorNoiseEffect)) {
      return;
    }
    const scaled = scaleNoiseSettings(this.baseNoiseSettings, this.currentDPRScale);
    this.detectorNoiseEffect.readoutSigma = scaled.readoutSigma;
    this.detectorNoiseEffect.fpnSigma = scaled.fpnSigma;
    this.detectorNoiseEffect.photonGain = scaled.photonGain;
  }

  /**
   * Set DPR scale for noise adjustment
   *
   * When rendering at lower DPR, noise appears as larger blocks when upscaled.
   * This method scales noise parameters to maintain perceptual consistency.
   *
   * @param dpr - Current device pixel ratio (1.0 = native, <1.0 = reduced)
   */
  setDPRScale(dpr: number): void {
    const scale = clampDPRScale(dpr);

    if (!dprScaleChanged(this.currentDPRScale, scale)) {
      return; // No significant change
    }

    this.currentDPRScale = scale;
    this.applyScaledNoiseSettings();

    if (scale < 1.0) {
      log.info(
        Modules.POST_PROCESSING,
        `Noise scaled for DPR ${dpr.toFixed(2)} (noise × ${scale.toFixed(2)})`
      );
    }
  }

  /**
   * Sets vignette effect with proper typing and validation
   */
  setVignetteEnabled(enabled: boolean, darkness?: number, offset?: number): void {
    if (enabled && !this.vignetteEffect) {
      const settings = resolveVignetteDefaults({ darkness, offset });
      this.vignetteEffect = new RobustVignetteEffect(settings);
      this.rebuildEffectPass();
      log.success(
        Modules.POST_PROCESSING,
        `Vignette enabled (HDR-safe): darkness=${settings.darkness}, offset=${settings.offset}`
      );
    } else if (!enabled && this.vignetteEffect) {
      safeDisposeEffect(this.vignetteEffect, 'Vignette');
      this.vignetteEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Vignette disabled');
    } else if (enabled && this.vignetteEffect) {
      // Effect already enabled, update parameters
      if (!isRobustVignetteEffect(this.vignetteEffect)) {
        log.error(Modules.POST_PROCESSING, 'Invalid vignette effect type');
        return;
      }
      if (darkness !== undefined) this.vignetteEffect.darkness = darkness;
      if (offset !== undefined) this.vignetteEffect.offset = offset;
      log.update(
        Modules.POST_PROCESSING,
        `Vignette parameters updated: darkness=${darkness}, offset=${offset}`
      );
    }
  }

  /**
   * Sets chromatic lens distortion effect - combines lens distortion with wavelength-dependent chromatic aberration
   *
   * This effect replaces both the separate lens distortion and chromatic aberration effects,
   * providing a more physically accurate simulation where chromatic aberration follows the
   * lens geometry (stronger at edges where distortion is greater).
   *
   * @param enabled - Whether to enable the effect
   * @param distortionX - Radial distortion coefficient X (negative = barrel, positive = pincushion)
   * @param distortionY - Radial distortion coefficient Y
   * @param dispersion - Chromatic dispersion strength (0 = no chromatic, 0.05 = subtle, 0.2 = strong)
   * @param principalPointX - Principal point offset X (optical center shift)
   * @param principalPointY - Principal point offset Y
   * @param focalLengthX - Focal length scale X (< 1 = wide angle, > 1 = telephoto)
   * @param focalLengthY - Focal length scale Y
   * @param skew - Skew factor in radians
   */
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
    if (enabled && !this.chromaticLensDistortionEffect) {
      const cd = resolveChromaticDistortionDefaults({
        distortionX,
        distortionY,
        dispersion,
        principalPointX,
        principalPointY,
        focalLengthX,
        focalLengthY,
        skew,
      });
      this.chromaticLensDistortionEffect = new ChromaticLensDistortionEffect({
        distortion: new THREE.Vector2(cd.distortionX, cd.distortionY),
        dispersion: cd.dispersion,
        principalPoint: new THREE.Vector2(cd.principalPointX, cd.principalPointY),
        focalLength: new THREE.Vector2(cd.focalLengthX, cd.focalLengthY),
        skew: cd.skew,
      });

      this.rebuildEffectPass();
      log.info(
        Modules.POST_PROCESSING,
        `Chromatic lens distortion enabled: distortion=(${distortionX}, ${distortionY}), ` +
          `dispersion=${dispersion}, principalPoint=(${principalPointX}, ${principalPointY}), ` +
          `focalLength=(${focalLengthX}, ${focalLengthY}), skew=${skew}`
      );
    } else if (!enabled && this.chromaticLensDistortionEffect) {
      safeDisposeEffect(this.chromaticLensDistortionEffect, 'ChromaticLensDistortion');
      this.chromaticLensDistortionEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Chromatic lens distortion disabled');
    } else if (enabled && this.chromaticLensDistortionEffect) {
      // Effect already enabled, update parameters if provided
      this.updateChromaticLensDistortion({
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
  }

  /**
   * Updates chromatic lens distortion parameters with validation
   */
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
    if (!this.chromaticLensDistortionEffect) {
      log.warning(Modules.POST_PROCESSING, 'Chromatic lens distortion effect not initialized');
      return;
    }

    if (!isChromaticLensDistortionEffect(this.chromaticLensDistortionEffect)) {
      log.error(Modules.POST_PROCESSING, 'Invalid chromatic lens distortion effect type');
      return;
    }

    if (params.distortionX !== undefined || params.distortionY !== undefined) {
      this.chromaticLensDistortionEffect.distortion = mergeVector2(
        this.chromaticLensDistortionEffect.distortion,
        params.distortionX,
        params.distortionY
      );
    }

    if (params.dispersion !== undefined) {
      this.chromaticLensDistortionEffect.dispersion = params.dispersion;
    }

    if (params.principalPointX !== undefined || params.principalPointY !== undefined) {
      this.chromaticLensDistortionEffect.principalPoint = mergeVector2(
        this.chromaticLensDistortionEffect.principalPoint,
        params.principalPointX,
        params.principalPointY
      );
    }

    if (params.focalLengthX !== undefined || params.focalLengthY !== undefined) {
      this.chromaticLensDistortionEffect.focalLength = mergeVector2(
        this.chromaticLensDistortionEffect.focalLength,
        params.focalLengthX,
        params.focalLengthY
      );
    }

    if (params.skew !== undefined) {
      this.chromaticLensDistortionEffect.skew = params.skew;
    }

    log.update(
      Modules.POST_PROCESSING,
      `Chromatic lens distortion updated: ${Object.keys(params).join(', ')}`
    );
  }

  /**
   * Sets ambient occlusion effect (SSAO)
   */
  setAOEnabled(enabled: boolean, quality?: 'low' | 'medium' | 'high' | 'ultra'): void {
    if (enabled && this.aoEffect) {
      // AO already enabled — apply quality change in place. Without
      // this branch the UI Quality control was a no-op once AO was on
      // (samples/radius would not change until AO was toggled off and
      // back on).
      if (quality !== undefined) {
        this.setAOQuality(quality);
      }
      return;
    }
    if (enabled && !this.aoEffect && this.camera instanceof THREE.PerspectiveCamera) {
      const settings = aoQualitySettings(quality ?? 'medium');

      this.aoEffect = new SSAOEffect(this.camera, undefined, {
        samples: settings.samples,
        radius: settings.radius,
        intensity: 1.0,
        luminanceInfluence: 0.7,
        color: new THREE.Color(0x000000),
      });
      this._aoQuality = quality ?? 'medium';

      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, `Ambient occlusion enabled: quality=${quality}`);
    } else if (!enabled && this.aoEffect) {
      safeDisposeEffect(this.aoEffect, 'AmbientOcclusion');
      this.aoEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Ambient occlusion disabled');
    }
  }

  /**
   * update AO quality in place when AO is already enabled.
   *
   * pmndrs/postprocessing's `SSAOEffect` exposes `samples` and `radius`
   * on its uniforms map, but they're read at construction time. The
   * safest approach is dispose-and-recreate; that's a one-frame cost
   * that's invisible to the user and keeps the rest of the pipeline
   * untouched. No-op when AO is not enabled.
   */
  setAOQuality(quality: 'low' | 'medium' | 'high' | 'ultra'): void {
    if (!this.aoEffect || !(this.camera instanceof THREE.PerspectiveCamera)) return;
    if (this._aoQuality === quality) return;
    // B.3: dispose AFTER rebuild. Disposing first leaves a window where
    // `this.aoEffect` is freed but `EffectPass` still references it; if
    // a render runs (or an exception bubbles before recreate completes)
    // the post-processing pipeline would dereference the disposed
    // effect. Stash the old, build the new + rebuild the pass, then
    // dispose the old.
    const oldEffect = this.aoEffect;
    const settings = aoQualitySettings(quality);
    this.aoEffect = new SSAOEffect(this.camera, undefined, {
      samples: settings.samples,
      radius: settings.radius,
      intensity: 1.0,
      luminanceInfluence: 0.7,
      color: new THREE.Color(0x000000),
    });
    this._aoQuality = quality;
    this.rebuildEffectPass();
    safeDisposeEffect(oldEffect, 'AmbientOcclusion (quality change)');
    log.info(Modules.POST_PROCESSING, `AO quality updated to ${quality}`);
  }

  /** read-only accessor for current AO quality (used by debug state and controls UI). */
  getAOQuality(): 'low' | 'medium' | 'high' | 'ultra' | undefined {
    return this.aoEffect ? this._aoQuality : undefined;
  }

  /**
   * Check if any active effects require continuous rendering.
   *
   * Returns true if detector noise or other time-varying effects are enabled.
   * Used by animation controller to keep loop running.
   *
   * @returns true if continuous animation needed, false otherwise
   */
  needsContinuousAnimation(): boolean {
    // Detector noise always has temporal components that need continuous updates
    return !!this.detectorNoiseEffect;
  }

  /**
   * Gets the status of all effects
   */
  getEffectsStatus(): {
    bloom: boolean;
    detectorNoise: boolean;
    dof: boolean;
    chromaticAberration: boolean;
    fxaa: boolean;
    smaa: boolean;
    msaa: boolean;
    ssaa: boolean;
    toneMapping: string;
    vignette: boolean;
    ao: boolean;
    lensDistortion: boolean;
    chromaticLensDistortion: boolean;
  } {
    return {
      bloom: !!this.bloomEffect,
      detectorNoise: !!this.detectorNoiseEffect,
      dof: !!this.dofEffect,
      chromaticAberration: false, // Covered by ChromaticLensDistortion
      fxaa: this.fxaaEnabled,
      smaa: this.smaaEnabled,
      msaa: this.msaaEnabled,
      ssaa: this.ssaaEnabled,
      // Preserve original semantics: 'Off' iff effect missing; 'Unknown'
      // iff effect exists but mode is somehow undefined; otherwise the
      // canonical display name.
      toneMapping: this.toneMappingEffect
        ? toneMappingModeName(this.toneMappingEffect.mode)
        : 'Off',
      vignette: !!this.vignetteEffect,
      ao: !!this.aoEffect,
      lensDistortion: false, // Covered by ChromaticLensDistortion
      chromaticLensDistortion: !!this.chromaticLensDistortionEffect,
    };
  }

  /**
   * Get the active lens distortion parameters for picking coordinate correction.
   * Returns null if no distortion effect is active.
   *
   * The picking system uses these to apply the same distortion transform to
   * mouse coordinates before looking them up in the (undistorted) pick buffer.
   */
  getLensDistortionParams(): {
    distortion: THREE.Vector2;
    principalPoint: THREE.Vector2;
    focalLength: THREE.Vector2;
    skew: number;
  } | null {
    if (
      !this.chromaticLensDistortionEffect ||
      !isChromaticLensDistortionEffect(this.chromaticLensDistortionEffect)
    ) {
      return null;
    }
    return {
      distortion: this.chromaticLensDistortionEffect.distortion,
      principalPoint: this.chromaticLensDistortionEffect.principalPoint,
      focalLength: this.chromaticLensDistortionEffect.focalLength,
      skew: this.chromaticLensDistortionEffect.skew,
    };
  }

  /**
   * Enable/disable MSAA (Multi-Sample Anti-Aliasing)
   * Requires recreating the composer with different multisampling settings
   */
  setMSAAEnabled(enabled: boolean): void {
    if (this.msaaEnabled === enabled) return;

    if (enabled) {
      const gl = this.renderer.getContext() as WebGL2RenderingContext;
      const cap = checkMSAACapability(gl);
      if (!cap.supported) {
        log.error(
          Modules.POST_PROCESSING,
          `MSAA not supported by GPU (MAX_SAMPLES: ${cap.maxSamples})`
        );
        return;
      }
      if (!cap.floatBuffersOK) {
        log.warning(
          Modules.POST_PROCESSING,
          'Float color buffers not fully supported - MSAA may not work with HDR'
        );
      }
    }

    this.msaaEnabled = enabled;
    this.recreateComposer();
    this.updateRendererSize();

    // Verify MSAA was applied
    if (enabled) {
      log.info(
        Modules.POST_PROCESSING,
        `MSAA enabled with ${this.msaaSamples} samples (actual: ${(this.composer as unknown as { multisampling?: number }).multisampling || 0})`
      );
    } else {
      log.update(Modules.POST_PROCESSING, 'MSAA disabled');
    }
  }

  /**
   * Set MSAA sample count (2, 4, 8, 16)
   */
  setMSAASamples(samples: number): void {
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const maxSamples = (gl.getParameter(gl.MAX_SAMPLES) as number) ?? 0;
    const validation = validateMSAASamples(samples, maxSamples);
    if (validation.warning) {
      log.warning(Modules.POST_PROCESSING, validation.warning);
    }
    samples = validation.samples;

    if (this.msaaSamples === samples) return;

    this.msaaSamples = samples;
    if (this.msaaEnabled) {
      this.recreateComposer();
      this.updateRendererSize();
      log.info(
        Modules.POST_PROCESSING,
        `MSAA samples set to ${samples} (actual: ${(this.composer as unknown as { multisampling?: number }).multisampling || 0})`
      );
    }
  }

  getMSAASamples(): number {
    return this.msaaSamples;
  }

  isMSAAEnabled(): boolean {
    return this.msaaEnabled;
  }

  /**
   * Get actual MSAA samples being used (for debugging)
   * Returns 0 if MSAA is not active
   */
  getActualMSAASamples(): number {
    if (!this.msaaEnabled) return 0;

    // Try to get the actual multisampling value from composer
    const composerMultisampling = (this.composer as unknown as { multisampling?: number }).multisampling;
    if (composerMultisampling !== undefined) {
      return composerMultisampling;
    }

    // Fallback: check if the render target has MSAA
    type ComposerRTAccess = {
      inputBuffer?: { samples?: number };
      renderTarget?: { samples?: number };
    };
    const composerRT = this.composer as unknown as ComposerRTAccess;
    const renderTarget = composerRT.inputBuffer || composerRT.renderTarget;
    if (renderTarget && renderTarget.samples !== undefined) {
      return renderTarget.samples;
    }

    return 0;
  }

  /**
   * Enable/disable SSAA (Super-Sample Anti-Aliasing)
   * Renders at higher resolution then downsamples
   */
  setSSAAEnabled(enabled: boolean): void {
    if (this.ssaaEnabled === enabled) return;

    this.ssaaEnabled = enabled;
    this.recreateComposer();

    // Update both renderer and composer size for SSAA
    this.updateRendererSize();

    log.update(
      Modules.POST_PROCESSING,
      `SSAA ${enabled ? `enabled (${this.ssaaMultiplier}x)` : 'disabled'}`
    );
  }

  /**
   * Set SSAA resolution multiplier (1.5, 2.0, 3.0, 4.0)
   */
  setSSAAMultiplier(multiplier: number): void {
    multiplier = clampSSAAMultiplier(multiplier);

    if (this.ssaaMultiplier === multiplier) return;

    this.ssaaMultiplier = multiplier;
    if (this.ssaaEnabled) {
      // Need to recreate composer with new size, not just update renderer
      this.recreateComposer();
      // Then update renderer size to match
      this.updateRendererSize();
      log.update(Modules.POST_PROCESSING, `SSAA multiplier changed to ${multiplier}x`);
    }
  }

  /**
   * Recreate composer with updated AA settings and proper state preservation
   */
  private recreateComposer(): void {
    // Snapshot every effect's user-visible state so the rebuild is
    // transparent to the user. Same shape used by rebuildAfterContextRestore.
    const state = this.captureDurableState();

    // Detach passes from the composer FIRST, then dispose them.
    // Disposing while still attached can leave the composer holding
    // a half-disposed pass; future non-idempotent pass disposal
    // could throw mid-recreation.
    safeRemoveAndDisposePass(this.composer, this.effectPass, 'effectPass during recreation');
    this.effectPass = undefined;
    safeRemoveAndDisposePass(
      this.composer,
      this.secondaryPass,
      'secondaryPass during recreation'
    );
    this.secondaryPass = undefined;

    this.composer.dispose();

    // Recreate composer with new MSAA setting. Do NOT call composer.setSize()
    // here — all callers (setSSAAEnabled, setSSAAMultiplier, setMSAAEnabled,
    // setMSAASamples) call updateRendererSize() afterward which handles both
    // renderer.setSize() and composer.setSize() in one pass.
    this.composer = new EffectComposer(this.renderer, {
      frameBufferType: THREE.HalfFloatType,
      multisampling: this.msaaEnabled ? this.msaaSamples : 0,
    });

    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // Re-apply user state to the freshly-defaulted effects.
    applyBloomState(this.bloomEffect, state.bloom);
    applyToneMappingState(this.toneMappingEffect, state.toneMapping);
    applyDOFFocusDistance(this.dofEffect, state.dof);
    applyChromaticLensDistortionState(
      this.chromaticLensDistortionEffect,
      state.chromaticLensDistortion
    );
    applyVignetteState(this.vignetteEffect, state.vignette);
    applyDetectorNoiseState(this.detectorNoiseEffect, state.detectorNoise);

    this.rebuildEffectPass();

    log.info(
      Modules.POST_PROCESSING,
      `Composer recreated - MSAA: ${this.msaaEnabled ? this.msaaSamples : 0}x, SSAA: ${this.ssaaEnabled ? this.ssaaMultiplier : 1}x`
    );
  }

  /**
   * Get current quality preset
   */
  getQualityPreset(): 'low' | 'medium' | 'high' | 'ultra' {
    return this._qualityPreset;
  }

  /**
   * Set quality preset for performance optimization
   */
  setQualityPreset(preset: 'low' | 'medium' | 'high' | 'ultra'): void {
    this._qualityPreset = preset;

    // Use deferred rebuild to prevent multiple rebuilds
    this.startDeferRebuild();

    // Apply preset settings
    switch (preset) {
      case 'low':
        this.setBloomLevels(3); // Coarse bloom for performance
        this.setFXAAEnabled(true);
        this.setSMAAEnabled(false);
        this.setMSAAEnabled(false);
        this.setSSAAEnabled(false);
        this.setAOEnabled(false);
        break;

      case 'medium':
        this.setBloomLevels(6); // Balanced bloom quality
        this.setFXAAEnabled(false);
        this.setSMAAEnabled(true);
        this.updateSMAASettings('MEDIUM');
        this.setMSAAEnabled(false);
        this.setSSAAEnabled(false);
        break;

      case 'high':
        this.setBloomLevels(8); // High quality bloom
        this.setFXAAEnabled(false);
        this.setSMAAEnabled(true);
        this.updateSMAASettings('HIGH');
        this.setMSAAEnabled(true);
        this.setMSAASamples(4);
        break;

      case 'ultra':
        this.setBloomLevels(10); // Very smooth bloom
        this.setFXAAEnabled(false);
        this.setSMAAEnabled(true);
        this.updateSMAASettings('ULTRA');
        this.setMSAAEnabled(true);
        this.setMSAASamples(8);
        this.setSSAAEnabled(true);
        this.setSSAAMultiplier(2.0);
        break;
    }

    // End deferred mode and trigger single rebuild
    this.endDeferRebuild();

    log.info(Modules.POST_PROCESSING, `Quality preset set to: ${preset}`);
  }

  /**
   * Set bloom mipmap levels (1 to 12)
   * Lower values create coarser bloom (faster), higher values create smoother bloom
   * @param levels - Number of mipmap levels
   */
  setBloomLevels(levels: number): void {
    levels = clampBloomLevels(levels);

    if (this.bloomLevels === levels) return;

    this.bloomLevels = levels;

    if (!this.bloomEffect) {
      log.warning(Modules.POST_PROCESSING, 'Bloom effect not initialized');
      return;
    }
    if (!isBloomEffectTyped(this.bloomEffect)) {
      log.error(Modules.POST_PROCESSING, 'Invalid bloom effect type');
      return;
    }

    const settings = readBloomSettings(
      this.bloomEffect as unknown as {
        intensity: number;
        mipmapBlurPass?: { radius: number };
        luminanceMaterial?: { threshold: number };
      }
    );

    // B.3: dispose AFTER rebuild — same rationale as setAOQuality.
    const oldEffect = this.bloomEffect;
    this.bloomEffect = new BloomEffect(
      buildBloomConstructorOptions(settings, levels)
    ) as BloomEffectTyped;
    applyBloomRadius(
      this.bloomEffect as unknown as { mipmapBlurPass?: { radius: number } },
      settings.radius
    );

    this.rebuildEffectPass();

    safeDisposeEffect(oldEffect, 'Bloom (levels change)');
    log.info(Modules.POST_PROCESSING, `Bloom mipmap levels set to ${levels}`);
  }

  /**
   * Get current bloom mipmap levels
   */
  getBloomLevels(): number {
    return this.bloomLevels;
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): { avgFrameTime: number; fps: number; memoryUsageMB: number } {
    const fps =
      this.performanceMetrics.avgFrameTime > 0 ? 1000 / this.performanceMetrics.avgFrameTime : 0;

    const memoryUsageMB = estimatePostProcMemoryMB({
      pixelCount: this.renderSize.width * this.renderSize.height,
      ssaaEnabled: this.ssaaEnabled,
      ssaaMultiplier: this.ssaaMultiplier,
      msaaEnabled: this.msaaEnabled,
      msaaSamples: this.msaaSamples,
      hasBloom: !!this.bloomEffect,
      hasAO: !!this.aoEffect,
    });

    return {
      avgFrameTime: this.performanceMetrics.avgFrameTime,
      fps,
      memoryUsageMB,
    };
  }

  /**
   * Determine which ping-pong buffer holds the final render result.
   *
   * The EffectComposer alternates between inputBuffer and outputBuffer after
   * each enabled pass (via swapBuffers). After an even number of swaps the
   * result is in inputBuffer; after an odd number it's in outputBuffer.
   *
   * This is deterministic and avoids the fragile center-pixel probe that
   * fails when the probed pixel is legitimately black or when the probe
   * reads stale intermediate data from the wrong buffer.
   */
  private getResultBuffer(): THREE.WebGLRenderTarget {
    return pickResultBuffer(
      this.composer.passes,
      this.composer.inputBuffer,
      this.composer.outputBuffer
    );
  }

  /**
   * Capture the current scene as raw HDR float pixel data (pre-tone-mapping).
   *
   * Returns the linear float RGBA pixels from the HDR pipeline. This is the
   * building block for both EXR export and HDR video encoding.
   *
   * `mode` controls which effects are disabled during capture:
   *   - `'visible-ldr'`: keep every effect enabled (the user's full
   *     post-processed pipeline). The pixels are still HDR/linear at
   *     framebuffer level — tone mapping has been baked into them.
   *   - `'hdr-effects-pre-tone'` (default):
   *     disable tone mapping, vignette, AA, detector noise, and chromatic
   *     lens distortion. Bloom / DOF / AO are KEPT (they're HDR-space).
   *   - `'raw-scene-hdr'`: disable EVERY post-processing effect including
   *     bloom / DOF / AO. The pixels are pure scene-material output.
   *
   * @param mode - Which effects to disable during capture (default `'hdr-effects-pre-tone'`).
   * @returns Object with Float32Array pixels and dimensions
   */
  captureHDRPixels(
    mode: 'visible-ldr' | 'hdr-effects-pre-tone' | 'raw-scene-hdr' = 'hdr-effects-pre-tone'
  ): { pixels: Float32Array; width: number; height: number } {
    // Save and disable effects per capture mode.
    const effectStates = new Map<object, boolean>();
    const ldrEffects =
      mode === 'visible-ldr'
        ? []
        : ([
            this.toneMappingEffect,
            this.vignetteEffect,
            this.smaaEffect,
            this.fxaaEffect,
            this.detectorNoiseEffect,
            this.chromaticLensDistortionEffect,
          ].filter(Boolean) as object[]);

    // raw-scene-hdr also disables HDR-space effects (bloom/DOF/AO).
    const hdrEffects =
      mode === 'raw-scene-hdr'
        ? ([
            this.bloomEffect,
            this.dofEffect,
            this.aoEffect,
          ].filter(Boolean) as object[])
        : [];

    for (const effect of [...ldrEffects, ...hdrEffects]) {
      const e = effect as { enabled: boolean };
      effectStates.set(effect, e.enabled !== false);
      e.enabled = false;
    }

    const savedAutoRender = this.composer.autoRenderToScreen;
    this.composer.autoRenderToScreen = false;
    const savedPassStates = this.composer.passes.map((p) => p.renderToScreen);
    for (const pass of this.composer.passes) {
      pass.renderToScreen = false;
    }

    let pixels: Float32Array;
    let width: number;
    let height: number;

    try {
      this.composer.render();

      const sourceBuffer = this.getResultBuffer();
      width = sourceBuffer.width;
      height = sourceBuffer.height;
      const pixelCount = width * height * 4;

      // Determine buffer type based on the render target's texture type.
      // WebGL requires matching typed arrays: HalfFloat → Uint16Array, Float → Float32Array
      const isHalfFloat = sourceBuffer.texture.type === THREE.HalfFloatType;

      if (isHalfFloat) {
        // Read as Uint16Array (half-float encoded), then convert to Float32Array
        const halfData = new Uint16Array(pixelCount);
        this.renderer.readRenderTargetPixels(sourceBuffer, 0, 0, width, height, halfData);
        pixels = halfFloatToFloat32(halfData);
      } else {
        // FloatType — read directly as Float32Array
        pixels = new Float32Array(pixelCount);
        this.renderer.readRenderTargetPixels(sourceBuffer, 0, 0, width, height, pixels);
      }
    } finally {
      this.composer.autoRenderToScreen = savedAutoRender;
      this.composer.passes.forEach((p, i) => {
        p.renderToScreen = savedPassStates[i];
      });
      // restore both LDR + HDR effects (raw-scene-hdr disables both).
      for (const effect of [...ldrEffects, ...hdrEffects]) {
        const e = effect as { enabled: boolean };
        e.enabled = effectStates.get(effect) ?? true;
      }
    }

    return { pixels, width, height };
  }

  /**
   * Capture the current scene as HDR EXR binary data.
   *
   * Uses captureHDRPixels() for the render/readback, then encodes as EXR.
   *
   * `options.mode` selects which effects are included in the capture
   * — see `captureHDRPixels` for the per-mode disable lists.
   *
   * @param options - Export options
   * @param options.type - Texture type: THREE.HalfFloatType (default, smaller) or THREE.FloatType (full precision)
   * @param options.mode - HDR capture mode (default `'hdr-effects-pre-tone'`).
   * @returns EXR file as Uint8Array binary data
   */
  async captureHDRAsEXR(options?: {
    type?: THREE.TextureDataType;
    mode?: 'visible-ldr' | 'hdr-effects-pre-tone' | 'raw-scene-hdr';
  }): Promise<Uint8Array> {
    const exrType: THREE.TextureDataType = options?.type ?? THREE.HalfFloatType;
    const { pixels, width, height } = this.captureHDRPixels(options?.mode);

    // Create DataTexture and export as EXR. Half-float encoding gives a
    // ~2× smaller file with negligible quality loss for typical scenes.
    const data: Float32Array | Uint16Array =
      exrType === THREE.HalfFloatType ? float32ToHalfFloat(pixels) : pixels;

    const texture = new THREE.DataTexture(
      data as BufferSource,
      width,
      height,
      THREE.RGBAFormat,
      exrType
    );
    texture.needsUpdate = true;

    // try/finally ensures texture.dispose() runs even if EXRExporter
    // rejects/throws — offline EXR sequence capture tolerates per-frame
    // failures, so without this a repeating encode failure would leak
    // a DataTexture every frame.
    try {
      const exporter = new EXRExporter();
      // Note: EXRExporter.parse() is synchronous in practice but typed as Promise
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
   * Render the full post-processing pipeline and return the result as ImageData.
   *
   * Strategy: render normally to screen (full pipeline with correct sRGB output),
   * then immediately read the WebGL framebuffer via gl.readPixels(). This works
   * because readPixels forces a GPU sync, and we read synchronously before the
   * browser compositor clears the buffer (preserveDrawingBuffer:false only clears
   * AFTER compositing, which happens at the end of the current JS task).
   *
   * Note: When SSAA (super-sample anti-aliasing) is enabled, the returned
   * ImageData is at the upscaled resolution (e.g., 2x display resolution),
   * not the display resolution. Callers that need display-size output should
   * downscale the result accordingly.
   */
  renderToImageData(): ImageData {
    // Render to screen with the full pipeline (tone mapping, sRGB, AA — everything)
    this.composer.render();

    // Read pixels directly from the WebGL default framebuffer.
    // gl.readPixels() forces a GPU flush so the draw is guaranteed complete.
    const gl = this.renderer.getContext();
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Flip vertically — WebGL framebuffer is bottom-up, ImageData is top-down.
    // The helper allocates with `new Uint8ClampedArray(N)` which is always
    // ArrayBuffer-backed in practice; the cast widens TS's defensive
    // ArrayBufferLike to the concrete ArrayBuffer that ImageData wants.
    const flipped = flipPixelsVerticallyRGBA(
      pixels,
      width,
      height
    ) as Uint8ClampedArray<ArrayBuffer>;
    return new ImageData(flipped, width, height);
  }

  /**
   * Renders the scene with post-processing
   */
  render(): void {
    // Track performance
    const startTime = performance.now();

    this.composer.render();

    // Update performance metrics
    const frameTime = performance.now() - startTime;
    this.performanceMetrics.lastFrameTime = frameTime;
    this.performanceMetrics.frameCount++;

    // Calculate moving average
    const alpha = 0.05; // Smoothing factor
    this.performanceMetrics.avgFrameTime =
      this.performanceMetrics.avgFrameTime * (1 - alpha) + frameTime * alpha;
  }

  /**
   * Update renderer size to match SSAA requirements
   * This ensures the renderer framebuffer matches the composer size
   */
  private updateRendererSize(): void {
    const { width: effectiveWidth, height: effectiveHeight } = this.computeEffectiveSize();

    // Update renderer framebuffer size to match composer.
    // The 'false' parameter prevents updating the canvas CSS size.
    this.renderer.setSize(effectiveWidth, effectiveHeight, false);

    // Manually set canvas display size to maintain viewport dimensions
    this.renderer.domElement.style.width = `${this.renderSize.width}px`;
    this.renderer.domElement.style.height = `${this.renderSize.height}px`;

    // Update composer to match
    this.composer.setSize(effectiveWidth, effectiveHeight);

    // Note: Bloom effect size is managed by the composer

    log.info(
      Modules.POST_PROCESSING,
      `Renderer/Composer size updated - Display: ${this.renderSize.width}x${this.renderSize.height}, ` +
        `Render: ${effectiveWidth}x${effectiveHeight}`
    );
  }

  /**
   * Handles window resize
   */
  resize(width: number, height: number): void {
    // Store the base render size
    this.renderSize = { width, height };

    // Update both renderer and composer sizes
    this.updateRendererSize();
  }

  /**
   * Tear down every GPU-bound transient resource (composer, render passes,
   * effect instances). The manager identity is preserved; durable user
   * settings stored on `this.*Enabled` / `this.*Multiplier` / `this.bloomLevels`
   * etc. are intentionally NOT cleared, so the manager remains useful for a
   * subsequent `initializeTransientResources()` call.
   *
   * Used by both `dispose()` and `rebuildAfterContextRestore()`.
   */
  private disposeTransientResources(): void {
    // Detach from composer first, then dispose — same rationale as
    // recreateComposer above.
    safeRemoveAndDisposePass(this.composer, this.effectPass, 'effectPass during cleanup');
    this.effectPass = undefined;
    safeRemoveAndDisposePass(this.composer, this.secondaryPass, 'secondaryPass during cleanup');
    this.secondaryPass = undefined;

    // Dispose individual effect objects. EffectPass disposal alone is not a
    // complete ownership guarantee for all pmndrs/postprocessing effects.
    this.disposeAllEffects('during cleanup');

    // Dispose composer (it also disposes passes added to it; we already did
    // that above for safety).
    this.composer.dispose();

    // Clear individual effect references — symmetric across every effect
    // owned by the manager, so dispose paths are safely idempotent and
    // post-dispose reads are well-defined as `undefined`.
    this.bloomEffect = undefined;
    this.detectorNoiseEffect = undefined;
    this.dofEffect = undefined;
    this.aoEffect = undefined;
    this.vignetteEffect = undefined;
    this.chromaticLensDistortionEffect = undefined;
    this.smaaEffect = undefined;
    this.fxaaEffect = undefined;
    this.toneMappingEffect = undefined;
  }

  /**
   * Capture the user-visible (durable) settings of every effect currently
   * configured on this manager. The returned snapshot is enough to fully
   * restore the post-processing pipeline after a context-loss-driven
   * rebuild — see `applyDurableState()`.
   */
  private captureDurableState(): PostProcessingDurableState {
    const bloomTarget =
      this.bloomEffect && isBloomEffectTyped(this.bloomEffect)
        ? (this.bloomEffect as unknown as Parameters<typeof captureBloomState>[0])
        : null;
    const dofTarget =
      this.dofEffect && isDepthOfFieldEffectTyped(this.dofEffect) ? this.dofEffect : null;
    const vignetteTarget =
      this.vignetteEffect && isRobustVignetteEffect(this.vignetteEffect)
        ? this.vignetteEffect
        : null;
    const chromaticTarget =
      this.chromaticLensDistortionEffect &&
      isChromaticLensDistortionEffect(this.chromaticLensDistortionEffect)
        ? this.chromaticLensDistortionEffect
        : null;
    const detectorTarget =
      this.detectorNoiseEffect && isDetectorNoiseEffect(this.detectorNoiseEffect)
        ? this.detectorNoiseEffect
        : null;

    return {
      bloom: captureBloomState(bloomTarget),
      toneMapping: captureToneMappingState(this.toneMappingEffect),
      dof: captureDOFState(dofTarget),
      vignette: captureVignetteState(vignetteTarget),
      chromaticLensDistortion: captureChromaticLensDistortionState(chromaticTarget),
      detectorNoise: captureDetectorNoiseState(detectorTarget),
      aoEnabled: !!this.aoEffect,
    };
  }

  /**
   * Re-apply captured durable state to freshly recreated effect
   * instances. `initializeTransientResources()` produces default-
   * configured bloom/toneMapping/smaa/fxaa instances; this method restores
   * the user's settings on those, plus toggles optional effects (DOF,
   * vignette, AO, chromatic lens distortion, detector noise) so the
   * pipeline matches the configuration before the rebuild.
   *
   * Caller invokes `rebuildEffectPass()` afterwards.
   */
  private applyDurableState(state: PostProcessingDurableState): void {
    // Bloom: always present after init; only re-apply settings when the
    // captured state includes bloom. If the user had bloom disabled before
    // rebuild, dispose the freshly-created default instance.
    if (state.bloom && this.bloomEffect && isBloomEffectTyped(this.bloomEffect)) {
      applyBloomState(
        this.bloomEffect as unknown as Parameters<typeof applyBloomState>[0],
        state.bloom
      );
    } else if (!state.bloom && this.bloomEffect) {
      safeDisposeEffect(this.bloomEffect, 'Bloom (rebuild: was disabled)');
      this.bloomEffect = undefined;
    }

    // Tone mapping: always present after init; restore user-facing fields.
    applyToneMappingState(this.toneMappingEffect, state.toneMapping);

    // Optional effects: recreate via their toggle methods, then re-apply
    // settings on the new instance. Each toggle calls rebuildEffectPass()
    // internally, but the final caller will rebuild once more so the
    // intermediate calls are absorbed.
    if (state.dof) {
      // setDOF expects (enabled, focus?, strength?). Re-derive the world-
      // space focus from the saved normalized focusDistance via the
      // PerspectiveDepthMapper inverse.
      const focusDistance = state.dof.focusDistance;
      const bokehScale = state.dof.bokehScale;
      const strength = bokehScale !== undefined ? bokehScale / 4.0 : undefined;
      // Note: focusDistance here is normalized [0,1]; setDOF will re-normalize
      // a world-space input. We restore by calling setDOF then writing the
      // normalized uniform back directly to preserve numeric precision.
      this.setDOF(true, undefined, strength);
      if (
        focusDistance !== undefined &&
        this.dofEffect &&
        isDepthOfFieldEffectTyped(this.dofEffect) &&
        this.dofEffect.circleOfConfusionMaterial?.uniforms?.focusDistance
      ) {
        this.dofEffect.circleOfConfusionMaterial.uniforms.focusDistance.value = focusDistance;
      }
    }
    if (state.vignette) {
      this.setVignetteEnabled(true, state.vignette.darkness, state.vignette.offset);
    }
    if (state.aoEnabled) {
      this.setAOEnabled(true, this._qualityPreset);
    }
    if (state.chromaticLensDistortion) {
      this.setChromaticLensDistortionEnabled(true);
      if (
        this.chromaticLensDistortionEffect &&
        isChromaticLensDistortionEffect(this.chromaticLensDistortionEffect)
      ) {
        applyChromaticLensDistortionState(
          this.chromaticLensDistortionEffect,
          state.chromaticLensDistortion
        );
      }
    }
    if (state.detectorNoise) {
      this.setDetectorNoiseEnabled(
        true,
        state.detectorNoise.readoutSigma,
        state.detectorNoise.photonGain,
        state.detectorNoise.fpnSigma
      );
    }
  }

  /**
   * Rebuild GPU-bound resources after a WebGL context-restore event.
   *
   * The manager's identity is preserved across the rebuild so external
   * consumers (PickingSystem, AnimationController, RenderingControls) can
   * keep their cached references — they will transparently see the new
   * composer / effects through the same `PostProcessingManager` reference.
   * All durable user settings (bloom, tone mapping, DOF, vignette, AO,
   * chromatic lens distortion, detector noise) are preserved across the
   * rebuild.
   *
   * Safe to call repeatedly; a no-op if `dispose()` has already run.
   */
  rebuildAfterContextRestore(): void {
    if (this.disposed) return;

    log.info(Modules.POST_PROCESSING, 'Rebuilding post-processing pipeline after context restore');

    const state = this.captureDurableState();
    this.disposeTransientResources();
    this.initializeTransientResources();
    this.applyDurableState(state);
    this.rebuildEffectPass();

    log.success(Modules.POST_PROCESSING, 'Post-processing pipeline rebuilt after context restore');
  }

  /**
   * Disposes all resources. Safe to call multiple times — subsequent calls
   * are no-ops thanks to the `disposed` guard.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.disposeTransientResources();
    log.success(Modules.POST_PROCESSING, 'PostProcessing resources disposed');
  }
}
