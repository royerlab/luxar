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
  KernelSize,
  BlendFunction,
} from 'postprocessing';
import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import { config } from '../config';

/**
 * Manages post-processing effects using the pmndrs/postprocessing library
 * Features HDR rendering, bloom, depth of field, tone mapping, and various effects
 */
export class PostProcessingManager {
  private composer: EffectComposer;
  private renderPass: RenderPass;
  private effectPass?: EffectPass;

  // Individual effect references for runtime updates
  private bloomEffect?: BloomEffect;
  private dofEffect?: DepthOfFieldEffect;
  private toneMappingEffect!: ToneMappingEffect;
  private smaaEffect?: SMAAEffect;
  private fxaaEffect?: FXAAEffect;
  private aoEffect?: SSAOEffect;
  private vignetteEffect?: VignetteEffect;
  private chromaticEffect?: ChromaticAberrationEffect;

  // State tracking
  private fxaaEnabled: boolean = false;
  private smaaEnabled: boolean = false;

  constructor(
    private renderer: THREE.WebGLRenderer,
    private scene: THREE.Scene,
    private camera: THREE.Camera,
    size: { width: number; height: number }
  ) {
    // Configure renderer for postprocessing library
    // The library expects linear space input and handles sRGB conversion
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping; // Let postprocessing handle tone mapping

    // Create composer with HDR support using 16-bit float buffers
    this.composer = new EffectComposer(this.renderer, {
      frameBufferType: THREE.HalfFloatType,
      multisampling: 0, // We'll handle AA via effects
    });

    this.composer.setSize(size.width, size.height);

    // Initialize render pass
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // Create initial effects and setup passes
    this.createInitialEffects();
    this.rebuildEffectPass();

    log.success(
      Modules.POST_PROCESSING,
      'pmndrs/postprocessing pipeline initialized with HDR support'
    );
  }

  /**
   * Creates the initial set of effects with default settings
   */
  private createInitialEffects(): void {
    // Bloom effect with HDR support - use config defaults
    this.bloomEffect = new BloomEffect({
      intensity: config.renderingControls.defaults.bloomStrength,
      luminanceThreshold: config.renderingControls.defaults.bloomThreshold,
      luminanceSmoothing: 0.01, // Very low to minimize dark halo with additive blending
      radius: config.renderingControls.defaults.bloomRadius,
      mipmapBlur: true,
      kernelSize: KernelSize.LARGE,
      blendFunction: BlendFunction.ADD, // ADD works better with additive point clouds
    });

    // Tone mapping for HDR to LDR conversion - using ACES filmic by default
    this.toneMappingEffect = new ToneMappingEffect({
      mode: ToneMappingMode.ACES_FILMIC,
      resolution: 256,
      whitePoint: config.renderingControls.defaults.exposure * 4.0,
      middleGrey: 0.4, // Lower middle grey for brighter output
      minLuminance: 0.001, // Lower min for better dark detail
      averageLuminance: 1.0,
      adaptationRate: 1.0,
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
   * Rebuilds the effect pass with currently active effects
   * This is called when effects are added/removed or toggled
   */
  private rebuildEffectPass(): void {
    // Remove old effect pass if it exists
    if (this.effectPass) {
      this.composer.removePass(this.effectPass);
      this.effectPass = undefined;
    }

    // Collect all active effects in the correct order
    const effects: any[] = [];

    // Pre-tone mapping effects (work in HDR space)
    if (this.bloomEffect) effects.push(this.bloomEffect);
    if (this.dofEffect) effects.push(this.dofEffect);
    if (this.aoEffect) effects.push(this.aoEffect);

    // Tone mapping (HDR to LDR conversion)
    effects.push(this.toneMappingEffect);

    // Post-tone mapping effects (work in LDR space)
    if (this.vignetteEffect) effects.push(this.vignetteEffect);
    if (this.chromaticEffect) effects.push(this.chromaticEffect);

    // Anti-aliasing (last)
    if (this.smaaEnabled && this.smaaEffect) effects.push(this.smaaEffect);
    else if (this.fxaaEnabled && this.fxaaEffect) effects.push(this.fxaaEffect);

    // Create new effect pass with all active effects
    if (effects.length > 0) {
      this.effectPass = new EffectPass(this.camera, ...effects);
      this.composer.addPass(this.effectPass);
    }

    log.info(Modules.POST_PROCESSING, `Effect pass rebuilt with ${effects.length} active effects`);
  }

  /**
   * Updates bloom effect parameters
   */
  updateBloomSettings(strength?: number, radius?: number, threshold?: number): void {
    if (!this.bloomEffect) return;

    if (strength !== undefined) this.bloomEffect.intensity = strength;
    if (radius !== undefined) (this.bloomEffect as any).radius = radius;
    if (threshold !== undefined) (this.bloomEffect as any).luminanceThreshold = threshold;

    log.update(
      Modules.POST_PROCESSING,
      `Bloom updated: strength=${this.bloomEffect.intensity}, ` +
        `radius=${(this.bloomEffect as any).radius}, threshold=${(this.bloomEffect as any).luminanceThreshold}`
    );
  }

  /**
   * Updates exposure (affects tone mapping white point)
   */
  updateExposure(exposure: number): void {
    // In pmndrs/postprocessing, exposure is controlled via uniforms
    const uniforms = (this.toneMappingEffect as any).uniforms;
    if (uniforms?.whitePoint) {
      uniforms.whitePoint.value = exposure * 4.0;
    }
    log.update(
      Modules.POST_PROCESSING,
      `Exposure updated: ${exposure} (white point: ${exposure * 4.0})`
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
   * Sets depth of field effect
   */
  setDOF(enabled: boolean, focus?: number, strength?: number): void {
    if (enabled && !this.dofEffect && this.camera instanceof THREE.PerspectiveCamera) {
      // Create DOF effect with better default parameters
      // Focus distance is in world units, bokeh scale affects blur strength
      this.dofEffect = new DepthOfFieldEffect(this.camera, {
        focusDistance: focus ?? 10.0,
        focalLength: 0.05,
        bokehScale: (strength ?? 0.5) * 4.0,
        height: 480,
      });
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, `DOF enabled: focus=${focus}, strength=${strength}`);
    } else if (!enabled && this.dofEffect) {
      this.dofEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'DOF disabled');
    }
  }

  /**
   * Updates DOF parameters
   */
  updateDOF(params: { focus?: number; strength?: number }): void {
    if (!this.dofEffect) return;

    const uniforms = (this.dofEffect as any).circleOfConfusionMaterial?.uniforms;
    if (uniforms) {
      if (params.focus !== undefined && uniforms.focusDistance) {
        uniforms.focusDistance.value = params.focus;
      }
    }

    if (params.strength !== undefined) {
      this.dofEffect.bokehScale = params.strength * 4.0;
    }

    log.update(
      Modules.POST_PROCESSING,
      `DOF updated: focus=${params.focus}, strength=${params.strength}`
    );
  }

  /**
   * Sets chromatic aberration effect
   */
  setChromaticAberration(enabled: boolean, strength?: number): void {
    if (enabled && !this.chromaticEffect) {
      const offset = (strength ?? 0.5) * 0.002;
      this.chromaticEffect = new ChromaticAberrationEffect({
        offset: new THREE.Vector2(offset, offset),
        radialModulation: false,
        modulationOffset: 0,
      });
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, `Chromatic aberration enabled: strength=${strength}`);
    } else if (!enabled && this.chromaticEffect) {
      this.chromaticEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Chromatic aberration disabled');
    }
  }

  /**
   * Updates chromatic aberration strength
   */
  updateChromaticAberration(strength: number): void {
    if (!this.chromaticEffect) return;

    const offset = strength * 0.002;
    this.chromaticEffect.offset = new THREE.Vector2(offset, offset);
    log.update(Modules.POST_PROCESSING, `Chromatic aberration strength: ${strength}`);
  }

  /**
   * Sets vignette effect
   */
  setVignetteEnabled(enabled: boolean, darkness?: number, offset?: number): void {
    if (enabled && !this.vignetteEffect) {
      this.vignetteEffect = new VignetteEffect({
        darkness: darkness ?? 0.5,
        offset: offset ?? 0.5,
      });
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, `Vignette enabled: darkness=${darkness}, offset=${offset}`);
    } else if (!enabled && this.vignetteEffect) {
      this.vignetteEffect = undefined;
      this.rebuildEffectPass();
      log.info(Modules.POST_PROCESSING, 'Vignette disabled');
    } else if (this.vignetteEffect) {
      if (darkness !== undefined) this.vignetteEffect.darkness = darkness;
      if (offset !== undefined) this.vignetteEffect.offset = offset;
      log.update(
        Modules.POST_PROCESSING,
        `Vignette updated: darkness=${darkness}, offset=${offset}`
      );
    }
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
   * Gets the status of all effects
   */
  getEffectsStatus(): {
    bloom: boolean;
    dof: boolean;
    chromaticAberration: boolean;
    fxaa: boolean;
    smaa: boolean;
    msaa: boolean;
    ssaa: boolean;
    toneMapping: string;
    vignette: boolean;
    ao: boolean;
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
      dof: !!this.dofEffect,
      chromaticAberration: !!this.chromaticEffect,
      fxaa: this.fxaaEnabled,
      smaa: this.smaaEnabled,
      msaa: false, // Not supported with current setup
      ssaa: false, // Not implemented yet
      toneMapping: toneMappingNames[this.toneMappingEffect.mode] ?? 'Unknown',
      vignette: !!this.vignetteEffect,
      ao: !!this.aoEffect,
    };
  }

  /**
   * Placeholder methods for compatibility (will be removed/updated)
   */
  setMSAAEnabled(_enabled: boolean): void {
    log.warning(Modules.POST_PROCESSING, 'MSAA not supported with pmndrs/postprocessing');
  }

  setMSAASamples(_samples: number): void {
    log.warning(Modules.POST_PROCESSING, 'MSAA not supported with pmndrs/postprocessing');
  }

  getMSAASamples(): number {
    return 0;
  }

  isMSAAEnabled(): boolean {
    return false;
  }

  setSSAAEnabled(_enabled: boolean): void {
    log.warning(Modules.POST_PROCESSING, 'SSAA not yet implemented');
  }

  setSSAAMultiplier(_multiplier: number): void {
    log.warning(Modules.POST_PROCESSING, 'SSAA not yet implemented');
  }

  setExposure(exposure: number): void {
    this.updateExposure(exposure);
  }

  /**
   * Renders the scene with post-processing
   */
  render(): void {
    this.composer.render();
  }

  /**
   * Handles window resize
   */
  resize(width: number, height: number): void {
    this.composer.setSize(width, height);
    log.info(Modules.POST_PROCESSING, `Resized to ${width}x${height}`);
  }

  /**
   * Disposes all resources
   */
  dispose(): void {
    this.composer.dispose();

    // Dispose individual effects if needed
    this.bloomEffect = undefined;
    this.dofEffect = undefined;
    this.aoEffect = undefined;
    this.vignetteEffect = undefined;
    this.chromaticEffect = undefined;
    this.smaaEffect = undefined;
    this.fxaaEffect = undefined;

    log.success(Modules.POST_PROCESSING, 'PostProcessing resources disposed');
  }
}
