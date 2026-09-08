/**
 * Bloom pyramid: downsample + upsample chain producing a soft-glow
 * texture from an HDR scene.
 *
 * Standard mipmap-blur pattern (Next-Gen Post Processing, GDC 2015):
 *
 *   1. Threshold + 2× downsample → mip[0]
 *   2. For i in 0..levels-2: 2× downsample mip[i] → mip[i+1]
 *   3. For i in levels-2..0:  additively upsample mip[i+1] into mip[i]
 *      using a 4-tap tent filter (cheap, smooth, no banding)
 *
 * Output: `outputTexture` (= mip[0]). The mega-shader samples this and
 * adds it onto the scene color (additive blend).
 *
 * Bloom resolution is half the canvas. Bloom is a soft glow; full-res
 * doesn't visibly improve quality and doubles memory.
 *
 * @module rendering/post-processing/bloom/chain
 */

import * as THREE from 'three';
import { clamp } from '../../../utils/clamp';
import { BLOOM_THRESHOLD_SOURCE, BLOOM_DOWNSAMPLE_SOURCE, BLOOM_UPSAMPLE_SOURCE } from './shaders';
import { buildMaterial } from '../../materials/_shared/material-builder';
import { FullscreenPass } from '../fullscreen/pass';
import type { Renderer, RendererCapabilities } from '../../renderer-capabilities';

export interface BloomChainConfig {
  /** Number of mip levels (1..12). Higher = wider, softer bloom. */
  levels?: number;
  /** Luminance threshold below which input pixels don't contribute. */
  threshold?: number;
  /** Soft-knee width around the threshold for smooth onset. */
  smoothing?: number;
  /** Upsample filter radius in texels; controls bloom spread. */
  radius?: number;
  /** Initial canvas size in pixels (pre-downsample). */
  width: number;
  height: number;
  /** Renderer capabilities; threaded through to the buildMaterial branch. */
  caps: RendererCapabilities;
}

interface MipLevel {
  target: THREE.WebGLRenderTarget;
  width: number;
  height: number;
}

/**
 * Bloom pyramid implementation. The renderer is supplied to render();
 * the chain owns its mip-level render targets and three shader
 * materials.
 */
export class BloomChain {
  private levels: number;
  private threshold: number;
  private readonly smoothing: number;
  private radius: number;

  private mips: MipLevel[] = [];

  private readonly thresholdMat: THREE.Material;
  private readonly downsampleMat: THREE.Material;
  private readonly upsampleMat: THREE.Material;

  // Uniforms held by reference so render() can mutate them under
  // either ShaderMaterial (WebGL2) or NodeMaterial (WebGPU). With
  // NodeMaterial there is no `.uniforms` on the material itself —
  // the TSL `uniform()` nodes hold the IUniform refs we pass in.
  private readonly thresholdUniforms: {
    uInput: THREE.IUniform<THREE.Texture | null>;
    uTexelSize: THREE.IUniform<THREE.Vector2>;
    uThreshold: THREE.IUniform<number>;
    uSmoothing: THREE.IUniform<number>;
  };
  private readonly downsampleUniforms: {
    uInput: THREE.IUniform<THREE.Texture | null>;
    uTexelSize: THREE.IUniform<THREE.Vector2>;
  };
  private readonly upsampleUniforms: {
    uInput: THREE.IUniform<THREE.Texture | null>;
    uTexelSize: THREE.IUniform<THREE.Vector2>;
    uRadius: THREE.IUniform<number>;
  };

  private readonly pass: FullscreenPass;

  constructor(cfg: BloomChainConfig) {
    this.levels = clamp(Math.round(cfg.levels ?? 8), 1, 12);
    this.threshold = cfg.threshold ?? 0.01;
    this.smoothing = cfg.smoothing ?? 0.01;
    this.radius = cfg.radius ?? 1.0;

    this.thresholdUniforms = {
      uInput: { value: null },
      uTexelSize: { value: new THREE.Vector2(1, 1) },
      uThreshold: { value: this.threshold },
      uSmoothing: { value: this.smoothing },
    };
    this.downsampleUniforms = {
      uInput: { value: null },
      uTexelSize: { value: new THREE.Vector2(1, 1) },
    };
    this.upsampleUniforms = {
      uInput: { value: null },
      uTexelSize: { value: new THREE.Vector2(1, 1) },
      uRadius: { value: this.radius },
    };

    this.thresholdMat = buildMaterial(
      BLOOM_THRESHOLD_SOURCE,
      { uniforms: this.thresholdUniforms, depthTest: false, depthWrite: false, toneMapped: false },
      cfg.caps
    );
    this.downsampleMat = buildMaterial(
      BLOOM_DOWNSAMPLE_SOURCE,
      { uniforms: this.downsampleUniforms, depthTest: false, depthWrite: false, toneMapped: false },
      cfg.caps
    );
    this.upsampleMat = buildMaterial(
      BLOOM_UPSAMPLE_SOURCE,
      {
        uniforms: this.upsampleUniforms,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
        // Upsample blends additively onto the previous (larger) mip.
        blending: THREE.AdditiveBlending,
      },
      cfg.caps
    );

    this.pass = new FullscreenPass(this.thresholdMat, cfg.caps);

    this.allocateMips(cfg.width, cfg.height);
  }

  /** The texture that the mega-shader samples for the bloom mix. */
  get outputTexture(): THREE.Texture {
    return this.mips[0].target.texture;
  }

  /** Output (mip[0]) resolution; useful for the mega-shader if it cares. */
  get outputSize(): { width: number; height: number } {
    return { width: this.mips[0].width, height: this.mips[0].height };
  }

  /** Number of mip levels allocated after applying the minimum-size cutoff. */
  get mipCount(): number {
    return this.mips.length;
  }

  /**
   * Change the number of mip levels and rebuild the pyramid.
   *
   * If `canvasSize` is omitted the new pyramid is rebuilt at the SAME
   * canvas size implied by the existing mip[0] (×2 because mip[0] is
   * half-res). Callers that have just resized the canvas MUST pass
   * the new size explicitly to avoid a stale-mip race.
   */
  setLevels(levels: number, canvasSize?: { width: number; height: number }): void {
    const next = clamp(Math.round(levels), 1, 12);
    if (next === this.levels) return;
    const w = canvasSize?.width ?? (this.mips[0]?.width ? this.mips[0].width * 2 : 1);
    const h = canvasSize?.height ?? (this.mips[0]?.height ? this.mips[0].height * 2 : 1);
    this.levels = next;
    this.disposeMips();
    this.allocateMips(w, h);
  }

  setThreshold(t: number): void {
    this.threshold = t;
    this.thresholdUniforms.uThreshold.value = t;
  }

  setRadius(r: number): void {
    this.radius = r;
    this.upsampleUniforms.uRadius.value = r;
  }

  setSize(width: number, height: number): void {
    const newW = Math.max(1, Math.floor(width / 2));
    const newH = Math.max(1, Math.floor(height / 2));
    if (this.mips[0]?.width === newW && this.mips[0]?.height === newH) return;
    this.disposeMips();
    this.allocateMips(width, height);
  }

  /**
   * Render the bloom pyramid from `sceneTexture` (the HDR scene
   * target's texture). After this returns, `outputTexture` holds the
   * bloom result.
   */
  render(renderer: Renderer, sceneTexture: THREE.Texture): void {
    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;

    // Pass 0: threshold + downsample sceneTexture → mip[0]
    this.pass.setMaterial(this.thresholdMat);
    this.thresholdUniforms.uInput.value = sceneTexture;
    this.thresholdUniforms.uTexelSize.value.set(1 / this.mips[0].width, 1 / this.mips[0].height);
    renderer.setRenderTarget(this.mips[0].target);
    this.pass.render(renderer);

    // Downsample chain: mip[i] → mip[i+1]. Iterate over actually-
    // allocated mips, not the user-requested `levels` — allocateMips
    // skips levels whose next mip would fall below MIN_MIP_DIM, so
    // `this.mips.length` may be smaller than `this.levels` at low DPR.
    const actualLevels = this.mips.length;
    this.pass.setMaterial(this.downsampleMat);
    for (let i = 0; i < actualLevels - 1; i++) {
      const src = this.mips[i];
      const dst = this.mips[i + 1];
      this.downsampleUniforms.uInput.value = src.target.texture;
      this.downsampleUniforms.uTexelSize.value.set(1 / dst.width, 1 / dst.height);
      renderer.setRenderTarget(dst.target);
      this.pass.render(renderer);
    }

    // Upsample chain: additively blend mip[i+1] into mip[i].
    // Uses AdditiveBlending on the material so the destination's
    // existing pixels are preserved and the upsampled samples add on.
    this.pass.setMaterial(this.upsampleMat);
    for (let i = actualLevels - 2; i >= 0; i--) {
      const src = this.mips[i + 1];
      const dst = this.mips[i];
      this.upsampleUniforms.uInput.value = src.target.texture;
      this.upsampleUniforms.uTexelSize.value.set(1 / src.width, 1 / src.height);
      renderer.setRenderTarget(dst.target);
      // Skip autoclear so additive blend writes onto existing mip[i]
      renderer.autoClear = false;
      this.pass.render(renderer);
      renderer.autoClear = true;
    }

    // Cast: see post-processing-manager.ts for the union-signature
    // rationale (round-tripping getRenderTarget → setRenderTarget is
    // safe at runtime on either backend).
    renderer.setRenderTarget(prevTarget as THREE.WebGLRenderTarget | null);
    renderer.autoClear = prevAutoClear;
  }

  dispose(): void {
    this.disposeMips();
    this.thresholdMat.dispose();
    this.downsampleMat.dispose();
    this.upsampleMat.dispose();
    this.pass.dispose();
  }

  // ----------------------------------------------------------------
  // Internal
  // ----------------------------------------------------------------

  /**
   * Allocate the mip pyramid for a canvas of the given size.
   *
   * `this.levels` is the user-requested ceiling, but levels are also
   * capped on physical size: a mip whose dimension would drop below
   * `MIN_MIP_DIM` is skipped, because at 1×1 / 2×1 / 3×2 the per-pass
   * overhead (FBO bind, viewport setup, tile setup on TBDR GPUs)
   * dwarfs the actual texture work and adds visible cost without any
   * visual contribution. The first mip is always allocated so the
   * bloom output texture exists even at very low resolutions.
   */
  private allocateMips(canvasWidth: number, canvasHeight: number): void {
    const MIN_MIP_DIM = 4;
    this.mips = [];
    let w = Math.max(1, Math.floor(canvasWidth / 2));
    let h = Math.max(1, Math.floor(canvasHeight / 2));
    for (let i = 0; i < this.levels; i++) {
      // Drop levels that have collapsed below the useful-resolution
      // threshold. Keep at least one mip so `outputTexture` always
      // exists.
      if (this.mips.length > 0 && (w < MIN_MIP_DIM || h < MIN_MIP_DIM)) break;
      const target = new THREE.WebGLRenderTarget(w, h, {
        type: THREE.HalfFloatType,
        format: THREE.RGBAFormat,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        wrapS: THREE.ClampToEdgeWrapping,
        wrapT: THREE.ClampToEdgeWrapping,
        depthBuffer: false,
        stencilBuffer: false,
      });
      target.texture.name = `BloomChain.mip[${i}]`;
      this.mips.push({ target, width: w, height: h });
      w = Math.max(1, Math.floor(w / 2));
      h = Math.max(1, Math.floor(h / 2));
    }
  }

  private disposeMips(): void {
    for (const m of this.mips) m.target.dispose();
    this.mips = [];
  }
}
