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
  ToneMappingEffect,
  ToneMappingMode,
  SMAAEffect,
  SMAAPreset,
  FXAAEffect,
  ChromaticAberrationEffect,
  VignetteEffect,
  SSAOEffect,
  NoiseEffect,
  LensDistortionEffect,
  KernelSize,
  BlendFunction,
} from 'postprocessing';
import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import { config } from '../config';
import {
  BloomEffectTyped,
  ToneMappingEffectTyped,
  DepthOfFieldEffectTyped,
  ChromaticAberrationEffectTyped,
  VignetteEffectTyped,
  LensDistortionEffectTyped,
  PerspectiveDepthMapper,
  isBloomEffectTyped,
  isToneMappingEffectTyped,
  isDepthOfFieldEffectTyped,
  isChromaticAberrationEffectTyped,
  isVignetteEffectTyped,
  isLensDistortionEffectTyped,
} from './postprocessing-types';

/**
 * Manages post-processing effects using the pmndrs/postprocessing library
 * Features HDR rendering, bloom, depth of field, tone mapping, and various effects
 */
export class PostProcessingManager {
  private composer: EffectComposer;
  private renderPass: RenderPass;
  private effectPass?: EffectPass;
  private secondaryPass?: EffectPass;

  // Individual effect references for runtime updates with proper typing
  private bloomEffect?: BloomEffectTyped;
  private dofEffect?: DepthOfFieldEffectTyped;
  private toneMappingEffect!: ToneMappingEffectTyped;
  private smaaEffect?: SMAAEffect;
  private fxaaEffect?: FXAAEffect;
  private aoEffect?: SSAOEffect;
  private vignetteEffect?: VignetteEffectTyped;
  private chromaticEffect?: ChromaticAberrationEffectTyped;
  private noiseEffect?: NoiseEffect;
  private lensDistortionEffect?: LensDistortionEffectTyped;

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

  // Performance features
  private bloomLevels: number = config.renderingControls.defaults.bloomLevels;
  private _qualityPreset: 'low' | 'medium' | 'high' | 'ultra' = 'medium';
  private performanceMetrics = {
    lastFrameTime: 0,
    avgFrameTime: 0,
    frameCount: 0,
  };

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

    // Calculate effective size for SSAA (now works correctly with initialized values)
    const effectiveWidth = this.ssaaEnabled
      ? Math.round(size.width * this.ssaaMultiplier)
      : size.width;
    const effectiveHeight = this.ssaaEnabled
      ? Math.round(size.height * this.ssaaMultiplier)
      : size.height;

    // Create composer with HDR support using 16-bit float buffers
    this.composer = new EffectComposer(this.renderer, {
      frameBufferType: THREE.HalfFloatType,
      multisampling: this.msaaEnabled ? this.msaaSamples : 0, // Native MSAA support
    });

    // Set composer size to effective size (including SSAA if enabled)
    this.composer.setSize(effectiveWidth, effectiveHeight);

    // Initialize render pass
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // Create initial effects and setup passes
    this.createInitialEffects();
    this.rebuildEffectPass();

    log.success(
      Modules.POST_PROCESSING,
      `pmndrs/postprocessing initialized - Output: ${size.width}x${size.height}, ` +
        `Render: ${effectiveWidth}x${effectiveHeight}${this.ssaaEnabled ? ' (SSAA)' : ''}, ` +
        `MSAA: ${this.msaaEnabled ? this.msaaSamples + 'x' : 'off'}`
    );
  }

  /**
   * Creates the initial set of effects with default settings
   */
  private createInitialEffects(): void {
    // Bloom effect with HDR support - use config defaults
    // Using mipmapBlur for better quality and performance
    this.bloomEffect = new BloomEffect({
      intensity: config.renderingControls.defaults.bloomStrength,
      luminanceThreshold: config.renderingControls.defaults.bloomThreshold,
      luminanceSmoothing: 0.01, // Very low to minimize dark halo with additive blending
      mipmapBlur: true, // Use mipmap blur for better quality bloom
      kernelSize: KernelSize.LARGE, // Standard kernel size
      blendFunction: BlendFunction.ADD, // ADD works better with additive point clouds
      levels: config.renderingControls.defaults.bloomLevels, // Number of mipmap levels
    }) as BloomEffectTyped;

    // Set radius on mipmapBlurPass after creation
    const bloom = this.bloomEffect as any;
    if (bloom.mipmapBlurPass) {
      bloom.mipmapBlurPass.radius = config.renderingControls.defaults.bloomRadius;
    }

    log.info(
      Modules.POST_PROCESSING,
      `Bloom initialized with ${config.renderingControls.defaults.bloomLevels} mipmap levels, radius: ${config.renderingControls.defaults.bloomRadius}`
    );

    // Tone mapping for HDR to LDR conversion - using ACES filmic by default
    this.toneMappingEffect = new ToneMappingEffect({
      mode: ToneMappingMode.ACES_FILMIC,
      resolution: 256,
      whitePoint: 2.0, // Standard white point
      middleGrey: 0.4, // Lower middle grey for brighter output
      minLuminance: 0.001, // Lower min for better dark detail
      averageLuminance: 1.0,
      adaptationRate: 1.0,
    }) as ToneMappingEffectTyped;

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
   * Starts deferred rebuild mode - prevents automatic rebuilds until endDeferRebuild()
   */
  startDeferRebuild(): void {
    this.deferRebuild = true;
  }

  /**
   * Ends deferred rebuild mode and triggers a single rebuild
   */
  endDeferRebuild(): void {
    this.deferRebuild = false;
    this.rebuildEffectPass();
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
    // Remove old effect passes if they exist with proper disposal
    if (this.effectPass) {
      this.composer.removePass(this.effectPass);
      try {
        this.effectPass.dispose(); // Explicit disposal to free GPU resources
      } catch (error) {
        log.warning(Modules.POST_PROCESSING, `Error disposing effectPass: ${error}`);
      }
      this.effectPass = undefined;
    }
    if (this.secondaryPass) {
      this.composer.removePass(this.secondaryPass);
      try {
        this.secondaryPass.dispose(); // Explicit disposal to free GPU resources
      } catch (error) {
        log.warning(Modules.POST_PROCESSING, `Error disposing secondaryPass: ${error}`);
      }
      this.secondaryPass = undefined;
    }

    // Define effects in correct visual order
    const orderedEffects: { effect: any; name: string }[] = [];

    // Build ordered list of active effects - CORRECT ORDER per user requirements
    if (this.bloomEffect) orderedEffects.push({ effect: this.bloomEffect, name: 'Bloom' });
    if (this.dofEffect) orderedEffects.push({ effect: this.dofEffect, name: 'DOF' });
    if (this.aoEffect) orderedEffects.push({ effect: this.aoEffect, name: 'AO' });
    if (this.vignetteEffect) orderedEffects.push({ effect: this.vignetteEffect, name: 'Vignette' });
    if (this.chromaticEffect)
      orderedEffects.push({ effect: this.chromaticEffect, name: 'ChromaticAberration' });
    if (this.lensDistortionEffect)
      orderedEffects.push({ effect: this.lensDistortionEffect, name: 'LensDistortion' });
    // Noise comes AFTER lens distortion and chromatic aberration but before tone mapping
    if (this.noiseEffect) orderedEffects.push({ effect: this.noiseEffect, name: 'Noise' });

    // Always add tone mapping and AA at the end
    orderedEffects.push({ effect: this.toneMappingEffect, name: 'ToneMapping' });
    if (this.smaaEnabled && this.smaaEffect)
      orderedEffects.push({ effect: this.smaaEffect, name: 'SMAA' });
    else if (this.fxaaEnabled && this.fxaaEffect)
      orderedEffects.push({ effect: this.fxaaEffect, name: 'FXAA' });

    // Sequential pass assignment: try adding effects to Pass A until incompatibility
    let passAEffects: any[] = [];
    let passBEffects: any[] = [];
    let usingPassB = false;
    const passANames: string[] = [];
    const passBNames: string[] = [];

    // Known incompatibility rules based on pmndrs documentation
    const isUVTransformEffect = (name: string): boolean => {
      return name === 'LensDistortion'; // UV transformation effects
    };

    const isConvolutionEffect = (name: string): boolean => {
      return name === 'ChromaticAberration'; // Convolution effects
    };

    for (const { effect, name } of orderedEffects) {
      if (!usingPassB) {
        // Check if this effect would be incompatible with Pass A
        const hasUVTransform = passANames.some(isUVTransformEffect);
        const hasConvolution = passANames.some(isConvolutionEffect);

        const wouldBeIncompatible =
          (isUVTransformEffect(name) && hasConvolution) ||
          (isConvolutionEffect(name) && hasUVTransform);

        if (wouldBeIncompatible) {
          // Switch to Pass B for this and all remaining effects
          usingPassB = true;
          passBEffects.push(effect);
          passBNames.push(name);

          log.info(
            Modules.POST_PROCESSING,
            `Effect incompatibility detected at ${name}, switching to Pass B`
          );
        } else {
          // Add to Pass A
          passAEffects.push(effect);
          passANames.push(name);
        }
      } else {
        // Add remaining effects to Pass B
        passBEffects.push(effect);
        passBNames.push(name);
      }
    }

    // Create Pass A with error recovery (always created if we have effects)
    if (passAEffects.length > 0) {
      try {
        this.effectPass = new EffectPass(this.camera, ...passAEffects);
        this.composer.addPass(this.effectPass);
      } catch (error) {
        log.error(Modules.POST_PROCESSING, `Failed to create Pass A: ${error}`);
        // Clean up partial state and rethrow
        if (this.effectPass) {
          try {
            this.effectPass.dispose();
          } catch {
            /* Ignore disposal errors during cleanup */
          }
          this.effectPass = undefined;
        }
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
        // Clean up partial state and rethrow
        if (this.secondaryPass) {
          try {
            this.secondaryPass.dispose();
          } catch {
            /* Ignore disposal errors during cleanup */
          }
          this.secondaryPass = undefined;
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
   * Updates bloom effect parameters
   * @param strength - Bloom intensity multiplier (0-2+ range typically)
   * @param radius - Mipmap blur radius (0-1+ range typically)
   * @param threshold - Luminance threshold (0-1 range, higher = less bloom)
   */
  updateBloomSettings(strength?: number, radius?: number, threshold?: number): void {
    if (!this.bloomEffect) {
      log.warning(Modules.POST_PROCESSING, 'Bloom effect not initialized');
      return;
    }

    // Validate bloom effect has expected properties
    if (!isBloomEffectTyped(this.bloomEffect)) {
      log.error(Modules.POST_PROCESSING, 'Invalid bloom effect type');
      return;
    }

    // Access properties correctly according to pmndrs structure
    const bloom = this.bloomEffect as any;

    if (strength !== undefined) {
      bloom.intensity = strength;
    }

    if (radius !== undefined && bloom.mipmapBlurPass) {
      bloom.mipmapBlurPass.radius = radius;
    }

    if (threshold !== undefined && bloom.luminanceMaterial) {
      bloom.luminanceMaterial.threshold = threshold;
    }

    // Get current values for logging
    const currentStrength = bloom.intensity || 0;
    const currentRadius = bloom.mipmapBlurPass?.radius || 0;
    const currentThreshold = bloom.luminanceMaterial?.threshold || 0;

    log.update(
      Modules.POST_PROCESSING,
      `Bloom updated: strength=${currentStrength.toFixed(2)}, ` +
        `radius=${currentRadius.toFixed(2)}, threshold=${currentThreshold.toFixed(2)}`
    );
  }

  /**
   * Sets the tone mapping mode
   */
  setToneMapping(mode: THREE.ToneMapping): void {
    // Map THREE.js tone mapping constants to pmndrs ToneMappingMode
    const modeMap: Record<number, ToneMappingMode> = {
      [THREE.NoToneMapping]: ToneMappingMode.LINEAR,
      [THREE.LinearToneMapping]: ToneMappingMode.LINEAR,
      [THREE.ReinhardToneMapping]: ToneMappingMode.REINHARD,
      [THREE.CineonToneMapping]: ToneMappingMode.OPTIMIZED_CINEON,
      [THREE.ACESFilmicToneMapping]: ToneMappingMode.ACES_FILMIC,
      [THREE.AgXToneMapping]: ToneMappingMode.AGX,
      [THREE.NeutralToneMapping]: ToneMappingMode.NEUTRAL,
    };

    const mappedMode = modeMap[mode] ?? ToneMappingMode.ACES_FILMIC;
    this.toneMappingEffect.mode = mappedMode;

    const modeName = Object.keys(ToneMappingMode).find(
      (key) => ToneMappingMode[key as keyof typeof ToneMappingMode] === mappedMode
    );
    log.update(Modules.POST_PROCESSING, `Tone mapping set to: ${modeName}`);
  }

  /**
   * Gets the current tone mapping mode as THREE constant
   */
  getToneMapping(): THREE.ToneMapping {
    // Map back from pmndrs to THREE constants
    const reverseMap: Record<ToneMappingMode, THREE.ToneMapping> = {
      [ToneMappingMode.LINEAR]: THREE.LinearToneMapping,
      [ToneMappingMode.REINHARD]: THREE.ReinhardToneMapping,
      [ToneMappingMode.OPTIMIZED_CINEON]: THREE.CineonToneMapping,
      [ToneMappingMode.ACES_FILMIC]: THREE.ACESFilmicToneMapping,
      [ToneMappingMode.AGX]: THREE.AgXToneMapping,
      [ToneMappingMode.NEUTRAL]: THREE.NeutralToneMapping,
    };

    return reverseMap[this.toneMappingEffect.mode] ?? THREE.ACESFilmicToneMapping;
  }

  /**
   * Enables or disables FXAA anti-aliasing
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

    const presetMap = {
      LOW: SMAAPreset.LOW,
      MEDIUM: SMAAPreset.MEDIUM,
      HIGH: SMAAPreset.HIGH,
      ULTRA: SMAAPreset.ULTRA,
    };

    // Need to recreate SMAA effect with new preset
    this.smaaEffect = new SMAAEffect({
      preset: presetMap[preset] ?? SMAAPreset.HIGH,
    });

    if (this.smaaEnabled) {
      this.rebuildEffectPass();
    }

    log.update(Modules.POST_PROCESSING, `SMAA quality set to: ${preset}`);
  }

  /**
   * Sets depth of field effect with proper perspective depth mapping
   */
  setDOF(enabled: boolean, focus?: number, strength?: number): void {
    if (enabled && !this.dofEffect && this.camera instanceof THREE.PerspectiveCamera) {
      // Create DOF effect with proper perspective depth mapping
      // IMPORTANT: pmndrs DepthOfFieldEffect expects focus distance as a value between 0 and 1
      // We use inverse depth mapping for better precision distribution
      const focusDistance = focus ?? 10.0;

      // Use proper perspective depth mapping instead of linear interpolation
      const normalizedFocus = PerspectiveDepthMapper.worldToNormalizedDepth(
        focusDistance,
        this.camera.near,
        this.camera.far
      );

      // Calculate focal length based on camera FOV for more realistic bokeh
      const focalLength = 0.035 * (50.0 / this.camera.fov); // Normalize to 50mm equivalent

      this.dofEffect = new DepthOfFieldEffect(this.camera, {
        focusDistance: normalizedFocus,
        focalLength: focalLength,
        bokehScale: (strength ?? 0.5) * 4.0,
        height: 480,
      }) as DepthOfFieldEffectTyped;

      this.rebuildEffectPass();

      log.info(
        Modules.POST_PROCESSING,
        `DOF enabled: focus=${focusDistance.toFixed(2)} (normalized: ${normalizedFocus.toFixed(3)}), ` +
          `strength=${strength}, focalLength=${focalLength.toFixed(3)}`
      );
    } else if (!enabled && this.dofEffect) {
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
      this.dofEffect.bokehScale = params.strength * 4.0;
      log.update(Modules.POST_PROCESSING, `DOF strength updated: ${params.strength}`);
    }
  }

  /**
   * Sets chromatic aberration effect with proper typing
   */
  setChromaticAberration(enabled: boolean, strength?: number): void {
    if (enabled && !this.chromaticEffect) {
      const offset = (strength ?? 0.5) * 0.002;
      this.chromaticEffect = new ChromaticAberrationEffect({
        offset: new THREE.Vector2(offset, offset),
        radialModulation: false,
        modulationOffset: 0,
      }) as ChromaticAberrationEffectTyped;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, `Chromatic aberration enabled: strength=${strength}`);
    } else if (!enabled && this.chromaticEffect) {
      this.chromaticEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Chromatic aberration disabled');
    }
  }

  /**
   * Updates chromatic aberration strength with validation
   */
  updateChromaticAberration(strength: number): void {
    if (!this.chromaticEffect) {
      log.warning(Modules.POST_PROCESSING, 'Chromatic aberration effect not initialized');
      return;
    }

    if (!isChromaticAberrationEffectTyped(this.chromaticEffect)) {
      log.error(Modules.POST_PROCESSING, 'Invalid chromatic aberration effect type');
      return;
    }

    const offset = strength * 0.002;
    this.chromaticEffect.offset = new THREE.Vector2(offset, offset);
    log.update(Modules.POST_PROCESSING, `Chromatic aberration strength: ${strength}`);
  }

  /**
   * Sets noise effect (film grain / static)
   * @param enabled - Whether to enable the effect
   * @param intensity - Opacity/strength of the noise (0-1)
   * @param premultiply - Whether to use premultiplied alpha (film grain style)
   * @param blendMode - Blend mode for the effect
   */
  setNoiseEnabled(
    enabled: boolean,
    intensity?: number,
    premultiply?: boolean,
    blendMode?: 'SCREEN' | 'ADD' | 'MULTIPLY' | 'OVERLAY' | 'SOFT_LIGHT'
  ): void {
    if (enabled && !this.noiseEffect) {
      const blendFunctionMap = {
        SCREEN: BlendFunction.SCREEN,
        ADD: BlendFunction.ADD,
        MULTIPLY: BlendFunction.MULTIPLY,
        OVERLAY: BlendFunction.OVERLAY,
        SOFT_LIGHT: BlendFunction.SOFT_LIGHT,
      };

      this.noiseEffect = new NoiseEffect({
        premultiply: premultiply ?? false,
        blendFunction: blendFunctionMap[blendMode ?? 'SCREEN'],
      });

      // Set initial intensity
      this.noiseEffect.blendMode.setOpacity(intensity ?? 0.05);

      this.rebuildEffectPass();
      log.info(
        Modules.POST_PROCESSING,
        `Noise enabled: intensity=${intensity}, premultiply=${premultiply}, blend=${blendMode}`
      );
    } else if (!enabled && this.noiseEffect) {
      this.noiseEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Noise disabled');
    }
  }

  /**
   * Updates noise effect parameters
   */
  updateNoiseSettings(
    intensity?: number,
    premultiply?: boolean,
    blendMode?: 'SCREEN' | 'ADD' | 'MULTIPLY' | 'OVERLAY' | 'SOFT_LIGHT'
  ): void {
    if (!this.noiseEffect) {
      log.warning(Modules.POST_PROCESSING, 'Noise effect not initialized');
      return;
    }

    if (intensity !== undefined) {
      this.noiseEffect.blendMode.setOpacity(intensity);
    }

    // If premultiply or blend mode changed, we need to recreate the effect
    if (premultiply !== undefined || blendMode !== undefined) {
      const currentIntensity = this.noiseEffect.blendMode.opacity.value;
      const currentPremultiply = (this.noiseEffect as any).premultiply ?? false;

      const blendFunctionMap = {
        SCREEN: BlendFunction.SCREEN,
        ADD: BlendFunction.ADD,
        MULTIPLY: BlendFunction.MULTIPLY,
        OVERLAY: BlendFunction.OVERLAY,
        SOFT_LIGHT: BlendFunction.SOFT_LIGHT,
      };

      // Get current blend function name
      let currentBlendName: 'SCREEN' | 'ADD' | 'MULTIPLY' | 'OVERLAY' | 'SOFT_LIGHT' = 'SCREEN';
      const currentBlendFunction = this.noiseEffect.blendMode.blendFunction;
      for (const [name, func] of Object.entries(blendFunctionMap)) {
        if (func === currentBlendFunction) {
          currentBlendName = name as typeof currentBlendName;
          break;
        }
      }

      // Recreate with new settings
      this.noiseEffect = new NoiseEffect({
        premultiply: premultiply ?? currentPremultiply,
        blendFunction: blendFunctionMap[blendMode ?? currentBlendName],
      });

      this.noiseEffect.blendMode.setOpacity(intensity ?? currentIntensity);
      this.rebuildEffectPass();
    }

    log.update(
      Modules.POST_PROCESSING,
      `Noise updated: intensity=${intensity}, premultiply=${premultiply}, blend=${blendMode}`
    );
  }

  /**
   * Sets vignette effect with proper typing and validation
   */
  setVignetteEnabled(enabled: boolean, darkness?: number, offset?: number): void {
    if (enabled && !this.vignetteEffect) {
      this.vignetteEffect = new VignetteEffect({
        darkness: darkness ?? 0.5,
        offset: offset ?? 0.5,
      }) as VignetteEffectTyped;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, `Vignette enabled: darkness=${darkness}, offset=${offset}`);
    } else if (!enabled && this.vignetteEffect) {
      this.vignetteEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Vignette disabled');
    } else if (this.vignetteEffect) {
      if (!isVignetteEffectTyped(this.vignetteEffect)) {
        log.error(Modules.POST_PROCESSING, 'Invalid vignette effect type');
        return;
      }
      if (darkness !== undefined) this.vignetteEffect.darkness = darkness;
      if (offset !== undefined) this.vignetteEffect.offset = offset;
      log.update(
        Modules.POST_PROCESSING,
        `Vignette updated: darkness=${darkness}, offset=${offset}`
      );
    }
  }

  /**
   * Sets lens distortion effect with proper typing and validation
   */
  setLensDistortionEnabled(
    enabled: boolean,
    distortionX?: number,
    distortionY?: number,
    principalPointX?: number,
    principalPointY?: number,
    focalLengthX?: number,
    focalLengthY?: number,
    skew?: number
  ): void {
    if (enabled && !this.lensDistortionEffect) {
      this.lensDistortionEffect = new LensDistortionEffect({
        distortion: new THREE.Vector2(distortionX ?? 0, distortionY ?? 0),
        principalPoint: new THREE.Vector2(principalPointX ?? 0, principalPointY ?? 0),
        focalLength: new THREE.Vector2(focalLengthX ?? 1, focalLengthY ?? 1),
        skew: skew ?? 0,
      }) as LensDistortionEffectTyped;
      this.rebuildEffectPass();
      log.info(
        Modules.POST_PROCESSING,
        `Lens distortion enabled: distortion=(${distortionX}, ${distortionY}), ` +
          `principalPoint=(${principalPointX}, ${principalPointY}), ` +
          `focalLength=(${focalLengthX}, ${focalLengthY}), skew=${skew}`
      );
    } else if (!enabled && this.lensDistortionEffect) {
      this.lensDistortionEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Lens distortion disabled');
    } else if (enabled && this.lensDistortionEffect) {
      // Effect already enabled, update parameters if provided
      this.updateLensDistortion({
        distortionX,
        distortionY,
        principalPointX,
        principalPointY,
        focalLengthX,
        focalLengthY,
        skew,
      });
    }
  }

  /**
   * Updates lens distortion parameters with validation
   */
  updateLensDistortion(params: {
    distortionX?: number;
    distortionY?: number;
    principalPointX?: number;
    principalPointY?: number;
    focalLengthX?: number;
    focalLengthY?: number;
    skew?: number;
  }): void {
    if (!this.lensDistortionEffect) {
      log.warning(Modules.POST_PROCESSING, 'Lens distortion effect not initialized');
      return;
    }

    if (!isLensDistortionEffectTyped(this.lensDistortionEffect)) {
      log.error(Modules.POST_PROCESSING, 'Invalid lens distortion effect type');
      return;
    }

    if (params.distortionX !== undefined || params.distortionY !== undefined) {
      const currentDistortion = this.lensDistortionEffect.distortion;
      this.lensDistortionEffect.distortion = new THREE.Vector2(
        params.distortionX ?? currentDistortion.x,
        params.distortionY ?? currentDistortion.y
      );
    }

    if (params.principalPointX !== undefined || params.principalPointY !== undefined) {
      const currentPrincipalPoint = this.lensDistortionEffect.principalPoint;
      this.lensDistortionEffect.principalPoint = new THREE.Vector2(
        params.principalPointX ?? currentPrincipalPoint.x,
        params.principalPointY ?? currentPrincipalPoint.y
      );
    }

    if (params.focalLengthX !== undefined || params.focalLengthY !== undefined) {
      const currentFocalLength = this.lensDistortionEffect.focalLength;
      this.lensDistortionEffect.focalLength = new THREE.Vector2(
        params.focalLengthX ?? currentFocalLength.x,
        params.focalLengthY ?? currentFocalLength.y
      );
    }

    if (params.skew !== undefined) {
      this.lensDistortionEffect.skew = params.skew;
    }

    log.update(
      Modules.POST_PROCESSING,
      `Lens distortion updated: ${Object.keys(params).join(', ')}`
    );
  }

  /**
   * Sets ambient occlusion effect (SSAO)
   */
  setAOEnabled(enabled: boolean, quality?: 'low' | 'medium' | 'high' | 'ultra'): void {
    if (enabled && !this.aoEffect && this.camera instanceof THREE.PerspectiveCamera) {
      // Configure quality settings
      const qualityMap = {
        low: {
          samples: 4,
          radius: 0.1,
        },
        medium: {
          samples: 8,
          radius: 0.2,
        },
        high: {
          samples: 16,
          radius: 0.3,
        },
        ultra: {
          samples: 32,
          radius: 0.4,
        },
      };

      const settings = qualityMap[quality ?? 'medium'];

      this.aoEffect = new SSAOEffect(this.camera, undefined, {
        samples: settings.samples,
        radius: settings.radius,
        intensity: 1.0,
        luminanceInfluence: 0.7,
        color: new THREE.Color(0x000000),
      });

      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, `Ambient occlusion enabled: quality=${quality}`);
    } else if (!enabled && this.aoEffect) {
      this.aoEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Ambient occlusion disabled');
    }
  }

  /**
   * Check if any effects require continuous animation
   * @returns True if animation should continue running
   */
  needsContinuousAnimation(): boolean {
    // Noise effect needs continuous updates as it changes every frame
    return !!this.noiseEffect;
  }

  /**
   * Gets the status of all effects
   */
  getEffectsStatus(): {
    bloom: boolean;
    noise: boolean;
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
    } {
    const toneMappingNames: Record<ToneMappingMode, string> = {
      [ToneMappingMode.LINEAR]: 'Linear',
      [ToneMappingMode.REINHARD]: 'Reinhard',
      [ToneMappingMode.OPTIMIZED_CINEON]: 'Cineon',
      [ToneMappingMode.ACES_FILMIC]: 'ACES Filmic',
      [ToneMappingMode.AGX]: 'AgX',
      [ToneMappingMode.NEUTRAL]: 'Neutral',
    };

    return {
      bloom: !!this.bloomEffect,
      noise: !!this.noiseEffect,
      dof: !!this.dofEffect,
      chromaticAberration: !!this.chromaticEffect,
      fxaa: this.fxaaEnabled,
      smaa: this.smaaEnabled,
      msaa: this.msaaEnabled,
      ssaa: this.ssaaEnabled,
      toneMapping: toneMappingNames[this.toneMappingEffect.mode] ?? 'Unknown',
      vignette: !!this.vignetteEffect,
      ao: !!this.aoEffect,
      lensDistortion: !!this.lensDistortionEffect,
    };
  }

  /**
   * Enable/disable MSAA (Multi-Sample Anti-Aliasing)
   * Requires recreating the composer with different multisampling settings
   */
  setMSAAEnabled(enabled: boolean): void {
    if (this.msaaEnabled === enabled) return;

    // Validate MSAA support
    if (enabled) {
      const gl = this.renderer.getContext() as WebGL2RenderingContext;
      const maxSamples = gl.getParameter(gl.MAX_SAMPLES);

      if (maxSamples < 2) {
        log.error(
          Modules.POST_PROCESSING,
          `MSAA not supported by GPU (MAX_SAMPLES: ${maxSamples})`
        );
        return;
      }

      // Check if float buffers support MSAA
      const ext = gl.getExtension('EXT_color_buffer_float');
      if (!ext) {
        log.warning(
          Modules.POST_PROCESSING,
          'Float color buffers not fully supported - MSAA may not work with HDR'
        );
      }

      // Warn about additive blending incompatibility
      log.warning(
        Modules.POST_PROCESSING,
        'MSAA enabled - Note: May cause brightness issues with additive blending. ' +
          'Consider using FXAA or SMAA instead.'
      );
    }

    this.msaaEnabled = enabled;
    this.recreateComposer();

    // Verify MSAA was applied
    if (enabled) {
      log.info(
        Modules.POST_PROCESSING,
        `MSAA enabled with ${this.msaaSamples} samples (actual: ${(this.composer as any).multisampling || 0})`
      );
    } else {
      log.update(Modules.POST_PROCESSING, 'MSAA disabled');
    }
  }

  /**
   * Set MSAA sample count (2, 4, 8, 16)
   */
  setMSAASamples(samples: number): void {
    // Validate samples
    const validSamples = [0, 2, 4, 8, 16];
    if (!validSamples.includes(samples)) {
      log.warning(Modules.POST_PROCESSING, `Invalid MSAA samples: ${samples}. Using 4.`);
      samples = 4;
    }

    // Check GPU maximum supported samples
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const maxSamples = gl.getParameter(gl.MAX_SAMPLES);
    if (samples > maxSamples) {
      log.warning(
        Modules.POST_PROCESSING,
        `Requested ${samples} MSAA samples but GPU only supports ${maxSamples}. Using ${maxSamples}.`
      );
      samples = Math.min(samples, maxSamples);
    }

    if (this.msaaSamples === samples) return;

    this.msaaSamples = samples;
    if (this.msaaEnabled) {
      this.recreateComposer();
      log.info(
        Modules.POST_PROCESSING,
        `MSAA samples set to ${samples} (actual: ${(this.composer as any).multisampling || 0})`
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
    const composerMultisampling = (this.composer as any).multisampling;
    if (composerMultisampling !== undefined) {
      return composerMultisampling;
    }

    // Fallback: check if the render target has MSAA
    const renderTarget = (this.composer as any).inputBuffer || (this.composer as any).renderTarget;
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
      `SSAA ${enabled ? 'enabled' : 'disabled'} (${this.ssaaMultiplier}x)`
    );
  }

  /**
   * Set SSAA resolution multiplier (1.5, 2.0, 3.0, 4.0)
   */
  setSSAAMultiplier(multiplier: number): void {
    // Clamp multiplier to reasonable range
    multiplier = Math.max(1.0, Math.min(4.0, multiplier));

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
    // Save ALL current effects state with proper typing
    const bloom = this.bloomEffect as any;
    const savedEffects = {
      bloom:
        this.bloomEffect && isBloomEffectTyped(this.bloomEffect)
          ? {
            intensity: bloom.intensity,
            luminanceThreshold: bloom.luminanceMaterial?.threshold,
            radius: bloom.mipmapBlurPass?.radius,
          }
          : null,
      toneMapping:
        this.toneMappingEffect && isToneMappingEffectTyped(this.toneMappingEffect)
          ? {
            mode: this.toneMappingEffect.mode,
            whitePoint: this.toneMappingEffect.uniforms?.whitePoint?.value,
          }
          : null,
      dof:
        this.dofEffect && isDepthOfFieldEffectTyped(this.dofEffect)
          ? {
            enabled: true,
            bokehScale: this.dofEffect.bokehScale,
            focusDistance:
                this.dofEffect.circleOfConfusionMaterial?.uniforms?.focusDistance?.value,
          }
          : null,
      chromatic:
        this.chromaticEffect && isChromaticAberrationEffectTyped(this.chromaticEffect)
          ? {
            offset: this.chromaticEffect.offset.clone(),
          }
          : null,
      vignette:
        this.vignetteEffect && isVignetteEffectTyped(this.vignetteEffect)
          ? {
            darkness: this.vignetteEffect.darkness,
            offset: this.vignetteEffect.offset,
          }
          : null,
      lensDistortion:
        this.lensDistortionEffect && isLensDistortionEffectTyped(this.lensDistortionEffect)
          ? {
            distortion: this.lensDistortionEffect.distortion.clone(),
            principalPoint: this.lensDistortionEffect.principalPoint.clone(),
            focalLength: this.lensDistortionEffect.focalLength.clone(),
            skew: this.lensDistortionEffect.skew,
          }
          : null,
      // Save AA states
      ao: this.aoEffect ? { enabled: true } : null,
      smaa: { enabled: this.smaaEnabled },
      fxaa: { enabled: this.fxaaEnabled },
    };

    // Dispose effect passes before disposing composer
    if (this.effectPass) {
      try {
        this.effectPass.dispose();
      } catch (error) {
        log.warning(
          Modules.POST_PROCESSING,
          `Error disposing effectPass during recreation: ${error}`
        );
      }
      this.effectPass = undefined;
    }

    if (this.secondaryPass) {
      try {
        this.secondaryPass.dispose();
      } catch (error) {
        log.warning(
          Modules.POST_PROCESSING,
          `Error disposing secondaryPass during recreation: ${error}`
        );
      }
      this.secondaryPass = undefined;
    }

    // Dispose old composer
    this.composer.dispose();

    // Calculate effective size for SSAA
    const effectiveWidth = this.ssaaEnabled
      ? Math.round(this.renderSize.width * this.ssaaMultiplier)
      : this.renderSize.width;
    const effectiveHeight = this.ssaaEnabled
      ? Math.round(this.renderSize.height * this.ssaaMultiplier)
      : this.renderSize.height;

    // Recreate composer with new settings
    this.composer = new EffectComposer(this.renderer, {
      frameBufferType: THREE.HalfFloatType,
      multisampling: this.msaaEnabled ? this.msaaSamples : 0,
    });

    this.composer.setSize(effectiveWidth, effectiveHeight);

    // Re-add render pass
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // Restore ALL effects settings with proper type checking
    if (savedEffects.bloom && this.bloomEffect && isBloomEffectTyped(this.bloomEffect)) {
      const restoredBloom = this.bloomEffect as any;
      restoredBloom.intensity = savedEffects.bloom.intensity;
      if (savedEffects.bloom.luminanceThreshold && restoredBloom.luminanceMaterial) {
        restoredBloom.luminanceMaterial.threshold = savedEffects.bloom.luminanceThreshold;
      }
      if (savedEffects.bloom.radius && restoredBloom.mipmapBlurPass) {
        restoredBloom.mipmapBlurPass.radius = savedEffects.bloom.radius;
      }
    }

    if (
      savedEffects.toneMapping &&
      this.toneMappingEffect &&
      isToneMappingEffectTyped(this.toneMappingEffect)
    ) {
      this.toneMappingEffect.mode = savedEffects.toneMapping.mode;
      if (savedEffects.toneMapping.whitePoint && this.toneMappingEffect.uniforms?.whitePoint) {
        this.toneMappingEffect.uniforms.whitePoint.value = savedEffects.toneMapping.whitePoint;
      }
    }

    // Restore DOF focus distance if it was saved
    if (savedEffects.dof && this.dofEffect && isDepthOfFieldEffectTyped(this.dofEffect)) {
      if (
        savedEffects.dof.focusDistance &&
        this.dofEffect.circleOfConfusionMaterial?.uniforms?.focusDistance
      ) {
        this.dofEffect.circleOfConfusionMaterial.uniforms.focusDistance.value =
          savedEffects.dof.focusDistance;
      }
    }

    // Restore lens distortion settings if they were saved
    if (
      savedEffects.lensDistortion &&
      this.lensDistortionEffect &&
      isLensDistortionEffectTyped(this.lensDistortionEffect)
    ) {
      this.lensDistortionEffect.distortion = savedEffects.lensDistortion.distortion.clone();
      this.lensDistortionEffect.principalPoint = savedEffects.lensDistortion.principalPoint.clone();
      this.lensDistortionEffect.focalLength = savedEffects.lensDistortion.focalLength.clone();
      this.lensDistortionEffect.skew = savedEffects.lensDistortion.skew;
    }

    // Rebuild effect pass (this will recreate the effect chain)
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
    // Clamp to reasonable range and ensure integer
    levels = Math.round(Math.max(1, Math.min(12, levels)));

    if (this.bloomLevels === levels) return;

    this.bloomLevels = levels;

    if (this.bloomEffect) {
      if (!isBloomEffectTyped(this.bloomEffect)) {
        log.error(Modules.POST_PROCESSING, 'Invalid bloom effect type');
        return;
      }

      const bloom = this.bloomEffect as any;

      // Store current settings
      const settings = {
        intensity: bloom.intensity || config.renderingControls.defaults.bloomStrength,
        luminanceThreshold:
          bloom.luminanceMaterial?.threshold || config.renderingControls.defaults.bloomThreshold,
        radius: bloom.mipmapBlurPass?.radius || config.renderingControls.defaults.bloomRadius,
      };

      // Recreate bloom with new levels setting
      // Levels control the quality/performance of mipmap blur
      this.bloomEffect = new BloomEffect({
        intensity: settings.intensity,
        luminanceThreshold: settings.luminanceThreshold,
        luminanceSmoothing: 0.01,
        mipmapBlur: true,
        kernelSize: KernelSize.LARGE,
        blendFunction: BlendFunction.ADD,
        levels: levels, // Number of mipmap levels
      }) as BloomEffectTyped;

      // Set radius after creation
      const newBloom = this.bloomEffect as any;
      if (newBloom.mipmapBlurPass) {
        newBloom.mipmapBlurPass.radius = settings.radius;
      }

      // Rebuild effect pass
      this.rebuildEffectPass();

      log.info(Modules.POST_PROCESSING, `Bloom mipmap levels set to ${levels}`);
    } else {
      log.warning(Modules.POST_PROCESSING, 'Bloom effect not initialized');
    }
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

    // Estimate memory usage
    const pixelCount = this.renderSize.width * this.renderSize.height;
    const ssaaMultiplier = this.ssaaEnabled ? this.ssaaMultiplier * this.ssaaMultiplier : 1;
    const msaaMultiplier = this.msaaEnabled ? this.msaaSamples : 1;

    // 16-bit float = 2 bytes per channel, RGBA = 8 bytes per pixel
    const bytesPerPixel = 8;
    const totalPixels = pixelCount * ssaaMultiplier * msaaMultiplier;

    // Account for multiple buffers (main, bloom, effects)
    const bufferCount = 3 + (this.bloomEffect ? 2 : 0) + (this.aoEffect ? 1 : 0);
    const memoryUsageMB = (totalPixels * bytesPerPixel * bufferCount) / (1024 * 1024);

    return {
      avgFrameTime: this.performanceMetrics.avgFrameTime,
      fps,
      memoryUsageMB,
    };
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
    const effectiveWidth = this.ssaaEnabled
      ? Math.round(this.renderSize.width * this.ssaaMultiplier)
      : this.renderSize.width;
    const effectiveHeight = this.ssaaEnabled
      ? Math.round(this.renderSize.height * this.ssaaMultiplier)
      : this.renderSize.height;

    // Update renderer framebuffer size to match composer
    // The 'false' parameter prevents updating the canvas CSS size
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
   * Disposes all resources
   */
  dispose(): void {
    // Dispose effect passes first
    if (this.effectPass) {
      try {
        this.effectPass.dispose();
      } catch (error) {
        log.warning(Modules.POST_PROCESSING, `Error disposing effectPass during cleanup: ${error}`);
      }
      this.effectPass = undefined;
    }

    if (this.secondaryPass) {
      try {
        this.secondaryPass.dispose();
      } catch (error) {
        log.warning(
          Modules.POST_PROCESSING,
          `Error disposing secondaryPass during cleanup: ${error}`
        );
      }
      this.secondaryPass = undefined;
    }

    // Dispose composer (this also disposes passes added to it, but we already did it above for safety)
    this.composer.dispose();

    // Clear individual effect references
    this.bloomEffect = undefined;
    this.noiseEffect = undefined;
    this.dofEffect = undefined;
    this.aoEffect = undefined;
    this.vignetteEffect = undefined;
    this.chromaticEffect = undefined;
    this.lensDistortionEffect = undefined;
    this.smaaEffect = undefined;
    this.fxaaEffect = undefined;

    log.success(Modules.POST_PROCESSING, 'PostProcessing resources disposed');
  }
}
