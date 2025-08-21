// HDR post-processing pipeline for advanced visual effects
//
// This module implements a sophisticated post-processing chain featuring:
// - HDR render targets with 16-bit precision
// - UnrealBloomPass for realistic bloom effects
// - ACES filmic tone mapping for professional color grading
// - Proper sRGB color space conversion for accurate display

import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/examples/jsm/shaders/FXAAShader.js';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js';
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js';
import { RGBShiftShader } from 'three/examples/jsm/shaders/RGBShiftShader.js';
import { config } from '../config';

/**
 * Configuration for HDR post-processing effects
 *
 * These parameters control the visual quality and performance
 * of the post-processing pipeline, allowing fine-tuning of
 * bloom effects and tone mapping behavior.
 */
export const POST_PROCESSING_CONFIG = config.postProcessing;

/**
 * HDR Post-Processing Pipeline Manager
 *
 * This class manages the complete post-processing chain for HDR rendering:
 * 1. Scene render → HDR render target
 * 2. Bloom pass → blur bright areas in linear space
 * 3. Output pass → ACES tone mapping + sRGB conversion
 */
export class PostProcessingManager {
  private composer!: EffectComposer;
  private renderPass!: RenderPass;
  private bloomPass!: UnrealBloomPass;
  private fxaaPass!: ShaderPass;
  private smaaPass!: SMAAPass;
  private outputPass!: OutputPass;
  private hdrRenderTarget!: THREE.WebGLRenderTarget;
  private fxaaEnabled: boolean = true;
  private smaaEnabled: boolean = false;
  private msaaSamples: number = 4;
  private msaaEnabled: boolean = false;
  private ssaaEnabled: boolean = false;
  private ssaaMultiplier: number = 2.0;
  private width: number;
  private height: number;

  // New post-processing effects
  private bokehPass?: BokehPass;
  private chromaticAberrationPass?: ShaderPass;

  // Effect parameters
  private currentToneMapping: THREE.ToneMapping = THREE.ACESFilmicToneMapping;
  private dofEnabled: boolean = false;
  private dofFocus: number = 10;
  private dofStrength: number = 0.5;
  private chromaticEnabled: boolean = false;
  private chromaticStrength: number = 0.5;

  /**
   * Creates and configures the HDR post-processing pipeline
   *
   * @param renderer - WebGL renderer (must support HDR)
   * @param scene - Three.js scene to render
   * @param camera - Camera for rendering
   * @param width - Render target width
   * @param height - Render target height
   */
  constructor(
    private renderer: THREE.WebGLRenderer,
    private scene: THREE.Scene,
    private camera: THREE.Camera,
    width: number,
    height: number
  ) {
    this.width = width;
    this.height = height;
    this.setupHDRRenderer();
    this.createHDRRenderTarget(width, height);
    this.setupEffectComposer();
    this.setupRenderPasses();
    this.finalizeToneMapping();
  }

  /**
   * Configures renderer for HDR pipeline
   *
   * Sets initial linear color space and no tone mapping so HDR values
   * can flow through to post-processing effects without clamping.
   */
  private setupHDRRenderer(): void {
    // Start in linear space so bloom sees unclamped HDR values
    this.renderer.outputColorSpace = POST_PROCESSING_CONFIG.toneMapping.initial.outputColorSpace;
    this.renderer.toneMapping = POST_PROCESSING_CONFIG.toneMapping.initial.toneMapping;

    console.log('✓ [Luxar] HDR renderer configured for linear pipeline');
  }

  /**
   * Creates HDR render target with 16-bit precision and optional MSAA/SSAA
   *
   * @param width - Target width in pixels
   * @param height - Target height in pixels
   */
  private createHDRRenderTarget(width: number, height: number): void {
    // Apply SSAA multiplier if enabled
    const renderWidth = this.ssaaEnabled ? Math.floor(width * this.ssaaMultiplier) : width;
    const renderHeight = this.ssaaEnabled ? Math.floor(height * this.ssaaMultiplier) : height;

    this.hdrRenderTarget = new THREE.WebGLRenderTarget(renderWidth, renderHeight, {
      // Use 16-bit float for HDR precision without color banding
      type: POST_PROCESSING_CONFIG.hdr.renderTargetType,

      // MSAA samples (WebGL2 only) - only apply if MSAA is enabled
      samples: this.msaaEnabled ? this.msaaSamples : 0,

      // Standard render target settings
      format: THREE.RGBAFormat,
      generateMipmaps: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,

      // Depth buffer settings - crucial for DOF to work properly
      depthBuffer: true,
      stencilBuffer: false,
    });

    const aaInfo = [];
    if (this.ssaaEnabled) aaInfo.push(`${this.ssaaMultiplier}x SSAA`);
    if (this.msaaEnabled) aaInfo.push(`${this.msaaSamples}x MSAA`);
    if (aaInfo.length === 0) aaInfo.push('no AA');

    console.log(
      `HDR render target created: ${renderWidth}x${renderHeight} (16-bit float, ${aaInfo.join(' + ')})`
    );
  }

  /**
   * Initializes the EffectComposer with HDR render target
   */
  private setupEffectComposer(): void {
    this.composer = new EffectComposer(this.renderer, this.hdrRenderTarget);
    console.log('✓ [Luxar] EffectComposer initialized with HDR target');
  }

  /**
   * Sets up the complete post-processing pass chain
   *
   * Chain: Scene Render → Bloom → DOF → Chromatic Aberration → Vignette → Output → SMAA/FXAA
   */
  private setupRenderPasses(): void {
    // 1. Render pass - renders scene to HDR buffer
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // 2. Bloom pass - creates glow effects in linear HDR space
    this.setupBloomPass();
    this.composer.addPass(this.bloomPass);

    // 3. DOF pass - only add if enabled
    if (this.dofEnabled) {
      this.setupDOFPass();
    }

    // 4. Chromatic aberration pass - only add if enabled
    if (this.chromaticEnabled) {
      this.setupChromaticAberrationPass();
    }

    // 6. Output pass - applies tone mapping and color space conversion
    this.outputPass = new OutputPass();
    this.composer.addPass(this.outputPass);

    // 7. SMAA pass - high quality anti-aliasing on final LDR image
    this.setupSMAAPass();
    this.composer.addPass(this.smaaPass);

    // 8. FXAA pass - fast anti-aliasing on final LDR image
    this.setupFXAAPass();
    this.composer.addPass(this.fxaaPass);

    console.log('✓ [Luxar] Post-processing passes configured');
  }

  /**
   * Configures the FXAA anti-aliasing pass
   */
  private setupFXAAPass(): void {
    this.fxaaPass = new ShaderPass(FXAAShader);

    // Get the render target size for resolution
    const width = this.hdrRenderTarget.width;
    const height = this.hdrRenderTarget.height;

    // Set resolution uniform for FXAA
    // FXAA expects the inverse of the resolution (texel size)
    this.fxaaPass.uniforms['resolution'].value.set(1 / width, 1 / height);

    // Initially enabled
    this.fxaaPass.enabled = this.fxaaEnabled;

    console.log(
      `FXAA pass configured: ${width}x${height} ` +
        `(enabled: ${this.fxaaEnabled}) ` +
        `resolution: [${this.fxaaPass.uniforms['resolution'].value.x.toFixed(6)}, ${this.fxaaPass.uniforms['resolution'].value.y.toFixed(6)}]`
    );
  }

  /**
   * Configures the SMAA anti-aliasing pass
   */
  private setupSMAAPass(): void {
    // SMAA Pass constructor takes width and height from the renderer
    this.smaaPass = new (SMAAPass as any)(this.hdrRenderTarget.width, this.hdrRenderTarget.height);

    // Initially disabled (user can enable via UI)
    this.smaaPass.enabled = this.smaaEnabled;

    console.log(
      `SMAA pass configured: ${this.width}x${this.height} ` + `(enabled: ${this.smaaEnabled})`
    );
  }

  /**
   * Configures the UnrealBloomPass for realistic bloom effects
   */
  private setupBloomPass(): void {
    const canvas = this.renderer.domElement;
    const bloomResolution = new THREE.Vector2(
      Math.floor(
        (canvas.clientWidth || window.innerWidth) / POST_PROCESSING_CONFIG.bloom.resolutionScale
      ),
      Math.floor(
        (canvas.clientHeight || window.innerHeight) / POST_PROCESSING_CONFIG.bloom.resolutionScale
      )
    );

    this.bloomPass = new UnrealBloomPass(
      bloomResolution,
      POST_PROCESSING_CONFIG.bloom.strength,
      POST_PROCESSING_CONFIG.bloom.radius,
      POST_PROCESSING_CONFIG.bloom.threshold
    );

    console.log(
      `Bloom pass configured: ${bloomResolution.x}x${bloomResolution.y} ` +
        `(strength: ${POST_PROCESSING_CONFIG.bloom.strength}, ` +
        `radius: ${POST_PROCESSING_CONFIG.bloom.radius})`
    );
  }

  /**
   * Applies final tone mapping settings
   *
   * After setting up the post-processing chain, we switch the renderer
   * to ACES filmic tone mapping and sRGB color space so the OutputPass
   * knows what transformations to apply.
   */
  private finalizeToneMapping(): void {
    this.renderer.toneMapping = POST_PROCESSING_CONFIG.toneMapping.final.toneMapping;
    this.renderer.outputColorSpace = POST_PROCESSING_CONFIG.toneMapping.final.outputColorSpace;

    console.log('✓ [Luxar] ACES filmic tone mapping and sRGB color space enabled');
  }

  /**
   * Renders the scene through the post-processing pipeline
   *
   * Call this instead of renderer.render() to get HDR + bloom effects.
   */
  render(): void {
    // Ensure output size is correct (important for SSAA)
    if (this.ssaaEnabled) {
      // EffectComposer handles downsampling automatically
      this.renderer.setSize(this.width, this.height);
    }
    this.composer.render();
  }

  /**
   * Updates the pipeline when canvas size changes
   *
   * @param width - New width in pixels
   * @param height - New height in pixels
   */
  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;

    // Apply SSAA multiplier to composer size if enabled
    const renderWidth = this.ssaaEnabled ? Math.floor(width * this.ssaaMultiplier) : width;
    const renderHeight = this.ssaaEnabled ? Math.floor(height * this.ssaaMultiplier) : height;

    // Update composer size - this handles all render targets internally
    this.composer.setSize(renderWidth, renderHeight);

    // Update FXAA resolution uniforms with render size
    if (this.fxaaPass) {
      this.fxaaPass.uniforms['resolution'].value.set(1 / renderWidth, 1 / renderHeight);
    }

    // Update SMAA resolution with render size
    if (this.smaaPass) {
      this.smaaPass.setSize(renderWidth, renderHeight);
    }

    console.log(
      `Post-processing resized: output ${width}x${height}, render ${renderWidth}x${renderHeight}`
    );
  }

  /**
   * Disposes of all post-processing resources
   *
   * Call this when shutting down to prevent memory leaks.
   */
  dispose(): void {
    this.hdrRenderTarget?.dispose();
    this.composer?.dispose();

    console.log('✓ [Luxar] Post-processing resources disposed');
  }

  /**
   * Gets the effect composer for advanced configuration
   *
   * @returns The EffectComposer instance
   */
  getComposer(): EffectComposer {
    return this.composer;
  }

  /**
   * Adjusts bloom parameters at runtime
   *
   * @param strength - New bloom strength (0.0 to 3.0)
   * @param radius - New bloom radius (0.0 to 1.0)
   * @param threshold - New bloom threshold (0.0 to 1.0)
   */
  updateBloomSettings(strength?: number, radius?: number, threshold?: number): void {
    if (strength !== undefined) this.bloomPass.strength = strength;
    if (radius !== undefined) this.bloomPass.radius = radius;
    if (threshold !== undefined) this.bloomPass.threshold = threshold;

    console.log(
      `Bloom settings updated: strength=${this.bloomPass.strength}, ` +
        `radius=${this.bloomPass.radius}, threshold=${this.bloomPass.threshold}`
    );
  }

  /**
   * Updates the tone mapping exposure value
   *
   * @param exposure - New exposure value (0.1 to 3.0)
   */
  updateExposure(exposure: number): void {
    this.renderer.toneMappingExposure = exposure;
    console.log(`✓ [Luxar] Tone mapping exposure updated: ${exposure}`);
  }

  /**
   * Enables or disables FXAA anti-aliasing
   *
   * @param enabled - Whether FXAA should be enabled
   */
  setFXAAEnabled(enabled: boolean): void {
    this.fxaaEnabled = enabled;
    if (this.fxaaPass) {
      this.fxaaPass.enabled = enabled;
      console.log(`✓ [Luxar] FXAA ${enabled ? 'enabled' : 'disabled'}`);
    } else {
      console.warn('FXAA pass not initialized!');
    }
  }

  /**
   * Gets the current FXAA enabled state
   *
   * @returns Whether FXAA is enabled
   */
  isFXAAEnabled(): boolean {
    return this.fxaaEnabled;
  }

  /**
   * Updates MSAA sample count by recreating the render target
   *
   * @param samples - Number of MSAA samples (0, 2, 4, 8)
   */
  setMSAASamples(samples: number): void {
    if (this.msaaSamples === samples) return;

    // Validate samples
    const validSamples = [0, 2, 4, 8];
    if (!validSamples.includes(samples)) {
      console.warn(`Invalid MSAA samples: ${samples}. Using 4.`);
      samples = 4;
    }

    this.msaaSamples = samples;
    this.recreateRenderTarget();
  }

  /**
   * Enables or disables MSAA
   *
   * @param enabled - Whether MSAA should be enabled
   */
  setMSAAEnabled(enabled: boolean): void {
    if (this.msaaEnabled === enabled) return;

    this.msaaEnabled = enabled;
    this.recreateRenderTarget();
  }

  /**
   * Recreates the render target preserving all current settings
   */
  private recreateRenderTarget(): void {
    // Save all current settings before recreation
    const currentExposure = this.renderer.toneMappingExposure;
    const currentToneMapping = this.renderer.toneMapping;
    const currentOutputColorSpace = this.renderer.outputColorSpace;
    const currentFXAAEnabled = this.fxaaEnabled;
    const currentSMAAEnabled = this.smaaEnabled;
    const currentBloomStrength = this.bloomPass?.strength;
    const currentBloomRadius = this.bloomPass?.radius;
    const currentBloomThreshold = this.bloomPass?.threshold;

    // Dispose old render target
    this.hdrRenderTarget.dispose();

    // Create new render target with updated samples
    this.createHDRRenderTarget(this.width, this.height);

    // Recreate composer with new render target
    this.composer.dispose();
    this.setupEffectComposer();

    // Re-setup passes
    this.setupRenderPasses();

    // Restore all settings after recreation
    // IMPORTANT: Set tone mapping BEFORE exposure to avoid reset
    this.renderer.toneMapping = currentToneMapping;
    this.renderer.outputColorSpace = currentOutputColorSpace;
    this.renderer.toneMappingExposure = currentExposure;

    // Restore bloom settings
    if (currentBloomStrength !== undefined) this.bloomPass.strength = currentBloomStrength;
    if (currentBloomRadius !== undefined) this.bloomPass.radius = currentBloomRadius;
    if (currentBloomThreshold !== undefined) this.bloomPass.threshold = currentBloomThreshold;

    // Restore AA states
    this.fxaaEnabled = currentFXAAEnabled;
    this.smaaEnabled = currentSMAAEnabled;
    if (this.fxaaPass) this.fxaaPass.enabled = currentFXAAEnabled;
    if (this.smaaPass) this.smaaPass.enabled = currentSMAAEnabled;

    const aaInfo = [];
    if (this.ssaaEnabled) aaInfo.push(`SSAA ${this.ssaaMultiplier}x`);
    if (this.msaaEnabled) aaInfo.push(`MSAA ${this.msaaSamples}x`);

    console.log(`Render target recreated with ${aaInfo.length > 0 ? aaInfo.join(' + ') : 'no AA'}`);
  }

  /**
   * Enables or disables SSAA
   *
   * @param enabled - Whether SSAA should be enabled
   */
  setSSAAEnabled(enabled: boolean): void {
    if (this.ssaaEnabled === enabled) return;

    this.ssaaEnabled = enabled;
    this.recreateRenderTarget();
    this.resize(this.width, this.height);
  }

  /**
   * Sets the SSAA resolution multiplier
   *
   * @param multiplier - Resolution multiplier (1.5, 2.0, etc.)
   */
  setSSAAMultiplier(multiplier: number): void {
    if (this.ssaaMultiplier === multiplier) return;

    // Validate multiplier
    if (multiplier < 1.0 || multiplier > 4.0) {
      console.warn(`Invalid SSAA multiplier: ${multiplier}. Clamping to [1.0, 4.0].`);
      multiplier = THREE.MathUtils.clamp(multiplier, 1.0, 4.0);
    }

    this.ssaaMultiplier = multiplier;
    if (this.ssaaEnabled) {
      this.recreateRenderTarget();
      this.resize(this.width, this.height);
    }
  }

  /**
   * Gets the current MSAA sample count
   *
   * @returns Current MSAA samples
   */
  getMSAASamples(): number {
    return this.msaaSamples;
  }

  /**
   * Gets the current MSAA enabled state
   *
   * @returns Whether MSAA is enabled
   */
  isMSAAEnabled(): boolean {
    return this.msaaEnabled;
  }

  /**
   * Enables or disables SMAA anti-aliasing
   *
   * @param enabled - Whether SMAA should be enabled
   */
  setSMAAEnabled(enabled: boolean): void {
    this.smaaEnabled = enabled;
    if (this.smaaPass) {
      this.smaaPass.enabled = enabled;
      console.log(`✓ [Luxar] SMAA ${enabled ? 'enabled' : 'disabled'}`);
    } else {
      console.warn('SMAA pass not initialized!');
    }
  }

  /**
   * Updates SMAA parameters
   *
   * @param threshold - Edge detection threshold (0.05-0.2)
   * @param searchSteps - Pattern search steps (4-32)
   */
  updateSMAASettings(threshold?: number, searchSteps?: number): void {
    if (!this.smaaPass) return;

    // Cast to any to access internal properties
    const smaa = this.smaaPass as any;

    // Update threshold in edge detection shader
    if (threshold !== undefined && smaa.materialEdges) {
      // Validate threshold
      const validThreshold = THREE.MathUtils.clamp(threshold, 0.05, 0.2);
      if (threshold !== validThreshold) {
        console.warn(`SMAA threshold ${threshold} clamped to [0.05, 0.2]`);
      }
      smaa.materialEdges.defines.SMAA_THRESHOLD = validThreshold.toFixed(3);
      smaa.materialEdges.needsUpdate = true;
      console.log(`✓ [Luxar] SMAA threshold updated to ${validThreshold}`);
    }

    // Update search steps in weights shader
    if (searchSteps !== undefined && smaa.materialWeights) {
      // Validate search steps
      const validSteps = [4, 8, 16, 32];
      if (!validSteps.includes(searchSteps)) {
        searchSteps = 8;
        console.warn('Invalid SMAA search steps. Using 8.');
      }
      smaa.materialWeights.defines.SMAA_MAX_SEARCH_STEPS = searchSteps.toString();
      smaa.materialWeights.needsUpdate = true;
      console.log(`SMAA search steps updated to ${searchSteps}`);
    }
  }

  // ========== New Post-Processing Effects ==========

  /**
   * Sets up the depth of field (DOF) pass using BokehPass
   */
  private setupDOFPass(): void {
    if (this.camera instanceof THREE.PerspectiveCamera) {
      // BokehPass has known tiling/flickering issues with certain parameters.
      // The artifacts appear as a checkerboard pattern that changes per frame.
      // To minimize this:
      // 1. Never use aperture or maxblur values too close to 0
      // 2. Keep the values in a reasonable range
      // 3. Make sure strength > 0.01 before creating the pass (handled in setDOF)

      // Scale parameters for more visible effect
      // Aperture controls the size of the blur kernel
      // Maxblur controls the maximum blur amount in screen space
      const aperture = 0.0025 + this.dofStrength * 0.025; // 0.0025 to 0.0275
      const maxblur = 0.01 + this.dofStrength * 0.02; // 0.01 to 0.03

      // BokehPass focus is in depth buffer space (0-1), not world units
      // Convert world distance to normalized depth
      const normalizedFocus = this.dofFocus / 100.0; // Assuming max distance of 100 units

      this.bokehPass = new BokehPass(this.scene, this.camera, {
        focus: normalizedFocus,
        aperture: aperture,
        maxblur: maxblur,
      });

      // Additional configuration to try to reduce artifacts
      const pass = this.bokehPass as any;

      // Configure bokeh shader parameters if available
      if (pass.materialBokeh) {
        // Lower quality settings can actually reduce artifacts
        pass.materialBokeh.defines.RINGS = 3;
        pass.materialBokeh.defines.SAMPLES = 4;
        pass.materialBokeh.needsUpdate = true;
      }

      // Ensure uniforms are properly initialized
      if (pass.uniforms) {
        pass.uniforms.nearClip = { value: this.camera.near };
        pass.uniforms.farClip = { value: this.camera.far };
      }

      this.bokehPass.enabled = true;
      this.composer.addPass(this.bokehPass);

      console.log(
        `DOF pass created - focus: ${normalizedFocus.toFixed(3)} (distance: ${this.dofFocus}), aperture: ${aperture.toFixed(4)}, maxblur: ${maxblur.toFixed(4)}`
      );
    }
  }

  /**
   * Sets up the chromatic aberration pass
   */
  private setupChromaticAberrationPass(): void {
    this.chromaticAberrationPass = new ShaderPass(RGBShiftShader);
    this.chromaticAberrationPass.uniforms['amount'].value = this.chromaticStrength * 0.005;
    this.chromaticAberrationPass.uniforms['angle'].value = 0.0;
    this.chromaticAberrationPass.enabled = true;
    // Add to composer after previous passes
    this.composer.addPass(this.chromaticAberrationPass);
  }

  /**
   * Sets the tone mapping type
   * @param toneMapping - THREE.ToneMapping constant
   */
  setToneMapping(toneMapping: THREE.ToneMapping): void {
    this.currentToneMapping = toneMapping;
    this.renderer.toneMapping = toneMapping;
    console.log(`Tone mapping changed to ${this.getToneMappingName()}`);
  }

  /**
   * Gets the current tone mapping type
   */
  getToneMapping(): THREE.ToneMapping {
    return this.currentToneMapping;
  }

  /**
   * Gets the name of the current tone mapping
   */
  private getToneMappingName(): string {
    switch (this.currentToneMapping) {
      case THREE.NoToneMapping:
        return 'None';
      case THREE.LinearToneMapping:
        return 'Linear';
      case THREE.ReinhardToneMapping:
        return 'Reinhard';
      case THREE.CineonToneMapping:
        return 'Cineon';
      case THREE.ACESFilmicToneMapping:
        return 'ACES Filmic';
      case THREE.AgXToneMapping:
        return 'AgX';
      case THREE.NeutralToneMapping:
        return 'Neutral';
      default:
        return 'Unknown';
    }
  }

  /**
   * Enables/disables depth of field and sets parameters
   * @param enabled - Whether DOF is enabled
   * @param focus - Focus distance (default 10)
   * @param strength - Blur strength (0-1, default 0.5)
   */
  setDOF(enabled: boolean, focus: number = 10, strength: number = 0.5): void {
    // Store parameters
    this.dofFocus = focus;
    this.dofStrength = strength;

    // Determine if DOF should actually be enabled
    // Only enable if user wants it AND strength is meaningful
    const shouldBeEnabled = enabled && strength > 0.01;

    // Only rebuild if state actually changed
    if (this.dofEnabled !== shouldBeEnabled) {
      this.dofEnabled = shouldBeEnabled;
      this.rebuildPipeline();
    }
  }

  /**
   * Updates DOF parameters
   */
  updateDOF(params: { focus?: number; strength?: number }): void {
    // Update stored parameters
    if (params.focus !== undefined) this.dofFocus = params.focus;
    if (params.strength !== undefined) {
      this.dofStrength = params.strength;

      // Check if we need to rebuild pipeline based on strength threshold
      const shouldBeEnabled = this.dofStrength > 0.01;
      if (shouldBeEnabled !== this.dofEnabled) {
        // State changed - rebuild pipeline
        this.dofEnabled = shouldBeEnabled;
        this.rebuildPipeline();
        return; // Exit early since pipeline was rebuilt
      }
    }

    // If DOF is enabled and pass exists, update uniforms directly for smooth transitions
    if (this.dofEnabled && this.bokehPass) {
      const pass = this.bokehPass as any;
      if (pass.uniforms) {
        if (params.focus !== undefined && pass.uniforms.focus) {
          // Convert world distance to normalized depth
          const normalizedFocus = this.dofFocus / 100.0;
          pass.uniforms.focus.value = normalizedFocus;
        }
        if (params.strength !== undefined) {
          // Use same formula as in setupDOFPass
          const aperture = 0.0025 + this.dofStrength * 0.025;
          const maxblur = 0.01 + this.dofStrength * 0.02;

          if (pass.uniforms.aperture) {
            pass.uniforms.aperture.value = aperture;
          }
          if (pass.uniforms.maxblur) {
            pass.uniforms.maxblur.value = maxblur;
          }
        }
      }
    }
  }

  /**
   * Enables/disables chromatic aberration
   * @param enabled - Whether chromatic aberration is enabled
   * @param strength - Strength of the effect (0-1, default 0.5)
   */
  setChromaticAberration(enabled: boolean, strength: number = 0.5): void {
    // Store parameters
    this.chromaticEnabled = enabled;
    this.chromaticStrength = strength;

    // Rebuild pipeline if state changed
    this.rebuildPipeline();
  }

  /**
   * Updates chromatic aberration strength only
   * @param strength - Strength of the effect (0-1)
   */
  updateChromaticAberration(strength: number): void {
    this.chromaticStrength = strength;

    // If enabled and pass exists, update uniform directly for smooth transitions
    if (this.chromaticEnabled && this.chromaticAberrationPass) {
      this.chromaticAberrationPass.uniforms['amount'].value = strength * 0.005;
    }
  }

  /**
   * Gets info about all available post-processing effects
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
  } {
    return {
      bloom: this.bloomPass?.enabled ?? false,
      dof: this.dofEnabled,
      chromaticAberration: this.chromaticEnabled,
      fxaa: this.fxaaEnabled,
      smaa: this.smaaEnabled,
      msaa: this.msaaEnabled,
      ssaa: this.ssaaEnabled,
      toneMapping: this.getToneMappingName(),
    };
  }

  /**
   * Rebuilds the entire post-processing pipeline
   * This ensures disabled effects don't consume any resources
   */
  private rebuildPipeline(): void {
    // Clear all passes from composer
    this.composer.passes = [];

    // Dispose of effect passes if they exist
    if (this.bokehPass) {
      this.bokehPass = undefined;
    }
    if (this.chromaticAberrationPass) {
      this.chromaticAberrationPass = undefined;
    }

    // Rebuild the pipeline with only enabled effects
    this.setupRenderPasses();

    console.log(`Pipeline rebuilt - DOF: ${this.dofEnabled}, Chromatic: ${this.chromaticEnabled}`);
  }
}
