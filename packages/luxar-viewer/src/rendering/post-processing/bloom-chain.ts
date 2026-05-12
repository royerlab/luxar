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
 * @module rendering/post-processing/bloom-chain
 */

import * as THREE from 'three';
import { clamp } from '../../utils/clamp';

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
}

/**
 * Shared fullscreen-triangle vertex shader for all three bloom passes
 * (threshold, downsample, upsample). The host supplies a unit triangle
 * in NDC via the geometry's `position` attribute, which THREE's
 * ShaderMaterial auto-declares — do NOT redeclare it here.
 */
const THRESHOLD_VERT = /* glsl */ `
  out vec2 vUv;
  void main() {
    vUv = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/**
 * Threshold + 2× box downsample. Extracts bright pixels above the
 * threshold with a smooth knee to avoid banding at the cutoff.
 *
 * Uses Rec.709 relative luminance for the brightness test. An
 * earlier version used max(r,g,b), which overstated saturated-channel
 * pixels — e.g. pure red would bloom even at low intensity.
 */
const THRESHOLD_FRAG = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uInput;
  uniform vec2 uTexelSize;
  uniform float uThreshold;
  uniform float uSmoothing;

  // Soft-knee: smoothstep around the threshold on relative luma, then
  // multiply by the source color (preserves chroma; only the brightness
  // gate is luma-based).
  vec3 thresholdKnee(vec3 color) {
    float l = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float soft = smoothstep(uThreshold, uThreshold + uSmoothing, l);
    return color * soft;
  }

  void main() {
    // 2x2 box downsample
    vec2 d = uTexelSize * 0.5;
    vec3 s0 = texture(uInput, vUv + d * vec2(-1.0, -1.0)).rgb;
    vec3 s1 = texture(uInput, vUv + d * vec2( 1.0, -1.0)).rgb;
    vec3 s2 = texture(uInput, vUv + d * vec2(-1.0,  1.0)).rgb;
    vec3 s3 = texture(uInput, vUv + d * vec2( 1.0,  1.0)).rgb;
    vec3 avg = (s0 + s1 + s2 + s3) * 0.25;
    fragColor = vec4(thresholdKnee(avg), 1.0);
  }
`;

/** Plain 2× box downsample (no threshold). */
const DOWNSAMPLE_FRAG = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uInput;
  uniform vec2 uTexelSize;

  void main() {
    vec2 d = uTexelSize * 0.5;
    vec3 s0 = texture(uInput, vUv + d * vec2(-1.0, -1.0)).rgb;
    vec3 s1 = texture(uInput, vUv + d * vec2( 1.0, -1.0)).rgb;
    vec3 s2 = texture(uInput, vUv + d * vec2(-1.0,  1.0)).rgb;
    vec3 s3 = texture(uInput, vUv + d * vec2( 1.0,  1.0)).rgb;
    fragColor = vec4((s0 + s1 + s2 + s3) * 0.25, 1.0);
  }
`;

/**
 * 4-tap tent upsample. Samples the smaller mip with a unit-radius
 * tent and additively blends into the larger mip (achieved by
 * blending with `THREE.AdditiveBlending` on the material).
 */
const UPSAMPLE_FRAG = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uInput;
  uniform vec2 uTexelSize;
  uniform float uRadius;

  void main() {
    vec2 r = uTexelSize * uRadius;
    vec3 s0 = texture(uInput, vUv + r * vec2(-1.0,  0.0)).rgb;
    vec3 s1 = texture(uInput, vUv + r * vec2( 1.0,  0.0)).rgb;
    vec3 s2 = texture(uInput, vUv + r * vec2( 0.0, -1.0)).rgb;
    vec3 s3 = texture(uInput, vUv + r * vec2( 0.0,  1.0)).rgb;
    vec3 c  = texture(uInput, vUv).rgb;
    fragColor = vec4(c * 0.5 + (s0 + s1 + s2 + s3) * 0.125, 1.0);
  }
`;

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

  private readonly thresholdMat: THREE.ShaderMaterial;
  private readonly downsampleMat: THREE.ShaderMaterial;
  private readonly upsampleMat: THREE.ShaderMaterial;

  private readonly fullscreenScene: THREE.Scene;
  private readonly fullscreenMesh: THREE.Mesh;
  private readonly camera: THREE.OrthographicCamera;

  constructor(cfg: BloomChainConfig) {
    this.levels = clamp(Math.round(cfg.levels ?? 8), 1, 12);
    this.threshold = cfg.threshold ?? 0.01;
    this.smoothing = cfg.smoothing ?? 0.01;
    this.radius = cfg.radius ?? 1.0;

    // Bloom pyramid uses HalfFloat for HDR preservation.
    const buildMaterial = (frag: string, blending: THREE.Blending = THREE.NoBlending) => {
      return new THREE.ShaderMaterial({
        vertexShader: THRESHOLD_VERT,
        fragmentShader: frag,
        glslVersion: THREE.GLSL3,
        depthTest: false,
        depthWrite: false,
        // Bypass renderer-level tone mapping injection on custom
        // post-processing materials; we manage HDR/LDR explicitly.
        toneMapped: false,
        blending,
        uniforms: {
          uInput: { value: null as THREE.Texture | null },
          uTexelSize: { value: new THREE.Vector2(1, 1) },
          uThreshold: { value: this.threshold },
          uSmoothing: { value: this.smoothing },
          uRadius: { value: this.radius },
        },
      });
    };

    this.thresholdMat = buildMaterial(THRESHOLD_FRAG);
    this.downsampleMat = buildMaterial(DOWNSAMPLE_FRAG);
    // Upsample blends additively onto the previous (larger) mip.
    this.upsampleMat = buildMaterial(UPSAMPLE_FRAG, THREE.AdditiveBlending);

    // Fullscreen triangle (NDC positions {-1,-1}, {3,-1}, {-1,3}).
    // One triangle covers the screen with no clipping waste.
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
    );
    this.fullscreenMesh = new THREE.Mesh(geo, this.thresholdMat);
    this.fullscreenMesh.frustumCulled = false;
    this.fullscreenScene = new THREE.Scene();
    this.fullscreenScene.add(this.fullscreenMesh);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

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
    this.thresholdMat.uniforms.uThreshold.value = t;
  }

  setRadius(r: number): void {
    this.radius = r;
    this.upsampleMat.uniforms.uRadius.value = r;
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
  render(renderer: THREE.WebGLRenderer, sceneTexture: THREE.Texture): void {
    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;

    // Pass 0: threshold + downsample sceneTexture → mip[0]
    this.fullscreenMesh.material = this.thresholdMat;
    this.thresholdMat.uniforms.uInput.value = sceneTexture;
    this.thresholdMat.uniforms.uTexelSize.value.set(
      1 / this.mips[0].width,
      1 / this.mips[0].height
    );
    renderer.setRenderTarget(this.mips[0].target);
    renderer.render(this.fullscreenScene, this.camera);

    // Downsample chain: mip[i] → mip[i+1]
    this.fullscreenMesh.material = this.downsampleMat;
    for (let i = 0; i < this.levels - 1; i++) {
      const src = this.mips[i];
      const dst = this.mips[i + 1];
      this.downsampleMat.uniforms.uInput.value = src.target.texture;
      this.downsampleMat.uniforms.uTexelSize.value.set(1 / dst.width, 1 / dst.height);
      renderer.setRenderTarget(dst.target);
      renderer.render(this.fullscreenScene, this.camera);
    }

    // Upsample chain: additively blend mip[i+1] into mip[i].
    // Uses AdditiveBlending on the material so the destination's
    // existing pixels are preserved and the upsampled samples add on.
    this.fullscreenMesh.material = this.upsampleMat;
    for (let i = this.levels - 2; i >= 0; i--) {
      const src = this.mips[i + 1];
      const dst = this.mips[i];
      this.upsampleMat.uniforms.uInput.value = src.target.texture;
      this.upsampleMat.uniforms.uTexelSize.value.set(1 / src.width, 1 / src.height);
      renderer.setRenderTarget(dst.target);
      // Skip autoclear so additive blend writes onto existing mip[i]
      renderer.autoClear = false;
      renderer.render(this.fullscreenScene, this.camera);
      renderer.autoClear = true;
    }

    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
  }

  dispose(): void {
    this.disposeMips();
    this.thresholdMat.dispose();
    this.downsampleMat.dispose();
    this.upsampleMat.dispose();
    this.fullscreenMesh.geometry.dispose();
  }

  // ----------------------------------------------------------------
  // Internal
  // ----------------------------------------------------------------

  private allocateMips(canvasWidth: number, canvasHeight: number): void {
    this.mips = [];
    let w = Math.max(1, Math.floor(canvasWidth / 2));
    let h = Math.max(1, Math.floor(canvasHeight / 2));
    for (let i = 0; i < this.levels; i++) {
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
