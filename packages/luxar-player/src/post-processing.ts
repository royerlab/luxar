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
import { config } from './config';

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
  private outputPass!: OutputPass;
  private hdrRenderTarget!: THREE.WebGLRenderTarget;

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

    console.log('✓ HDR renderer configured for linear pipeline');
  }

  /**
   * Creates HDR render target with 16-bit precision
   *
   * @param width - Target width in pixels
   * @param height - Target height in pixels
   */
  private createHDRRenderTarget(width: number, height: number): void {
    this.hdrRenderTarget = new THREE.WebGLRenderTarget(width, height, {
      // Use 16-bit float for HDR precision without color banding
      type: POST_PROCESSING_CONFIG.hdr.renderTargetType,

      // Standard render target settings
      format: THREE.RGBAFormat,
      generateMipmaps: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
    });

    console.log(`✓ HDR render target created: ${width}x${height} (16-bit float)`);
  }

  /**
   * Initializes the EffectComposer with HDR render target
   */
  private setupEffectComposer(): void {
    this.composer = new EffectComposer(this.renderer, this.hdrRenderTarget);
    console.log('✓ EffectComposer initialized with HDR target');
  }

  /**
   * Sets up the complete post-processing pass chain
   *
   * Chain: Scene Render → Bloom → Tone Mapping → Display
   */
  private setupRenderPasses(): void {
    // 1. Render pass - renders scene to HDR buffer
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // 2. Bloom pass - creates glow effects in linear HDR space
    this.setupBloomPass();
    this.composer.addPass(this.bloomPass);

    // 3. Output pass - applies tone mapping and color space conversion
    this.outputPass = new OutputPass();
    this.composer.addPass(this.outputPass);

    console.log('✓ Post-processing passes configured');
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
      `✓ Bloom pass configured: ${bloomResolution.x}x${bloomResolution.y} ` +
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

    console.log('✓ ACES filmic tone mapping and sRGB color space enabled');
  }

  /**
   * Renders the scene through the post-processing pipeline
   *
   * Call this instead of renderer.render() to get HDR + bloom effects.
   */
  render(): void {
    this.composer.render();
  }

  /**
   * Updates the pipeline when canvas size changes
   *
   * @param width - New width in pixels
   * @param height - New height in pixels
   */
  resize(width: number, height: number): void {
    // Update composer size - this handles all render targets internally
    this.composer.setSize(width, height);

    console.log(`✓ Post-processing resized: ${width}x${height}`);
  }

  /**
   * Disposes of all post-processing resources
   *
   * Call this when shutting down to prevent memory leaks.
   */
  dispose(): void {
    this.hdrRenderTarget?.dispose();
    this.composer?.dispose();

    console.log('✓ Post-processing resources disposed');
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
      `✓ Bloom settings updated: strength=${this.bloomPass.strength}, ` +
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
    console.log(`✓ Tone mapping exposure updated: ${exposure}`);
  }
}
