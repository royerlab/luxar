/**
 * GSplat Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for rendering Gaussian splats using instanced quads.
 * Implements projection-aware volumetric rendering with oriented, anisotropic Gaussian density functions.
 *
 * Key features:
 * - Instanced oriented quad geometry (4 vertices per splat)
 * - Full 3D covariance via Cholesky factors
 * - Perspective-correct projection of covariance to 2D
 * - Blending-mode-aware projection (sum for additive/normal, max for max blending)
 * - Shifted Gaussian falloff: scale · max(0, exp(-½ · r²) - C) with C⁰ continuity at truncation
 * - Per-splat attributes (center, cholesky, amplitude, color)
 *
 * GPU Optimizations (GLSL ES 3.0 / WebGL2):
 * - flat interpolation: skips GPU interpolation for per-instance varyings
 * - Reciprocal precomputation: DIV→MUL in vertex and fragment shaders
 * - Early discard at truncation radius before expensive exp()
 * - Higher intensity threshold (1e-4) for fewer blended pixels
 * - mediump precision for color/amplitude to reduce register pressure
 *
 * Mathematical basis:
 * - GSplat density: G(x) = a · scale · max(0, exp(-½ · ‖L⁻¹(x - μ)‖²) - C)
 * - Sum projection: amplitude boost by σ_ray · c_s (ray integration)
 * - Max projection: amplitude = a (peak value, no integration)
 * - 2D covariance: Σ_2D = J · Σ_cam · Jᵀ (perspective Jacobian projection)
 *
 * @module rendering/gsplat-material
 */

import * as THREE from 'three';
import { materialManager } from './material-manager';
import { GSPLAT_VERTEX_SHADER, GSPLAT_FRAGMENT_SHADER } from './shaders/gsplat-shaders';
import type { CameraAwareMaterial } from './camera-aware-material';

/**
 * Configuration for gsplat material creation
 */
export interface GSplatMaterialConfig {
  /** Opacity multiplier (0.0 to 1.0) */
  opacity?: number;
  /** Gamma correction (0.1 to 10.0, default 1.0) */
  gamma?: number;
  /** Intensity (linear color multiplier / gain), default 1.0 */
  intensity?: number;
  /** Offset (additive brightness shift / black level), default 0.0 */
  offset?: number;
  /** Truncation radius in sigmas (default 3.0) */
  truncationRadius?: number;
  /** Blending mode */
  blendingMode?: 'additive' | 'normal' | 'max' | 'opaque' | 'luminous';
  /** Whether material is transparent (default true) */
  transparent?: boolean;
  /** Whether to test against depth buffer (default true; additive sets false) */
  depthTest?: boolean;
  /** Colormap texture for scalar-to-color mapping (256x1 RGB) */
  colormapTexture?: THREE.DataTexture;
  /** Scalar data range [min, max] for normalization before LUT lookup */
  scalarRange?: [number, number];
  /** Max projected splat extent as a fraction of viewport size before fade-out (default 0.33) */
  maxExtentFactor?: number;
}

/**
 * GSplat material uniforms interface
 */
export interface GSplatMaterialUniforms {
  /** Viewport resolution [width, height] */
  uResolution: { value: THREE.Vector2 };
  /** Focal length X in pixels */
  uFx: { value: number };
  /** Focal length Y in pixels */
  uFy: { value: number };
  /** Truncation radius in sigmas */
  uTruncate: { value: number };
  /** Opacity multiplier */
  uOpacity: { value: number };
  /** Projection mode: 0=sum (additive/normal), 1=max (max blending) */
  uProjectionMode: { value: number };
  /** Pre-computed 1/gamma for performance */
  uInvGamma: { value: number };
  /** Near cull distance in world units (scene-scale-aware, perspective only) */
  uNearCull: { value: number };
  /** Max projected splat extent as fraction of viewport before fade-out */
  uMaxExtentFactor: { value: number };
  /** Shifted Gaussian: exp(-0.5 * truncate²) — boundary value */
  uShiftC: { value: number };
  /** Shifted Gaussian: 1/(1 - shiftC) — peak-preserving rescale */
  uInvOneMinusC: { value: number };
  /** Shifted Gaussian: truncate² — replaces hardcoded 9.0 in fragment shader */
  uTruncateSq: { value: number };
  /** Ray integration factor for sum projection (shifted Gaussian integral) */
  uRayIntegralFactor: { value: number };
}

/**
 * GSplat material for volumetric Gaussian splatting.
 *
 * Each splat is rendered as an oriented quad expanded based on the 2D
 * projected covariance eigenvalues. The fragment shader evaluates the
 * Gaussian density using Mahalanobis distance from the projected center.
 */
export class GSplatMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  /**
   * Create a new GSplatMaterial with the specified configuration.
   *
   * @param materialConfig - Material configuration options
   */
  constructor(materialConfig: GSplatMaterialConfig = {}) {
    const blendingMode = materialConfig.blendingMode ?? 'additive';
    const isOpaque = blendingMode === 'opaque';
    const isAdditive = blendingMode === 'additive';
    const gammaValue = Math.max(0.001, materialConfig.gamma ?? 1.0); // Prevent division by zero

    // Determine THREE.js blending mode
    // CRITICAL: For sum projection, we need LINEAR addition of intensities.
    // THREE.AdditiveBlending uses SrcAlpha which SQUARES the intensity - WRONG!
    // We use CustomBlending with OneFactor for correct linear sum projection.
    let blending: THREE.Blending;
    if (isOpaque || blendingMode === 'normal') {
      blending = THREE.NormalBlending;
    } else if (
      blendingMode === 'additive' ||
      blendingMode === 'luminous' ||
      blendingMode === 'max'
    ) {
      // All additive-style modes use CustomBlending for correct linear contribution
      blending = THREE.CustomBlending;
    } else {
      blending = THREE.NormalBlending;
    }

    const truncate = materialConfig.truncationRadius ?? 3.0;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    super({
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uFx: { value: 500 }, // Default focal length in pixels
        uFy: { value: 500 },
        uTruncate: { value: truncate },
        uTruncateSq: { value: truncate * truncate },
        uShiftC: { value: shiftC },
        uInvOneMinusC: { value: invOneMinusC },
        uRayIntegralFactor: {
          value: GSplatMaterial.computeRayIntegralFactor(truncate),
        },
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
        uProjectionMode: { value: blendingMode === 'max' ? 1 : 0 }, // 0=sum, 1=max
        uInvGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },
        uIsOrtho: { value: 0 }, // 0 = perspective, 1 = orthographic
        uNearCull: { value: 0.1 }, // Default; overridden per-scene by updateCameraParams
        uMaxExtentFactor: { value: materialConfig.maxExtentFactor ?? 0.33 },
        // Colormap uniforms (only when USE_COLORMAP define is set)
        ...(materialConfig.colormapTexture
          ? {
              uColormapTex: { value: materialConfig.colormapTexture },
              uScalarMin: { value: materialConfig.scalarRange?.[0] ?? 0.0 },
              uScalarScale: {
                value: materialConfig.scalarRange
                  ? 1.0 /
                    Math.max(1e-10, materialConfig.scalarRange[1] - materialConfig.scalarRange[0])
                  : 1.0,
              },
            }
          : {}),
      },

      vertexShader: GSPLAT_VERTEX_SHADER,
      fragmentShader: GSPLAT_FRAGMENT_SHADER,

      // Preprocessor defines — USE_COLORMAP enables LUT lookup from amplitude
      defines: {
        ...(materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {}),
      },

      // GLSL ES 3.0 for flat interpolation and modern syntax
      glslVersion: THREE.GLSL3,

      transparent: materialConfig.transparent ?? !isOpaque,
      depthWrite:
        isOpaque || (blendingMode === 'normal' && (materialConfig.opacity ?? 1.0) >= 0.99),
      // Additive ignores depth (renders on top), luminous respects depth occlusion
      depthTest: materialConfig.depthTest ?? !isAdditive,
      toneMapped: false, // HDR values pass through to post-processing
      blending: blending,
      side: THREE.DoubleSide, // Splats visible from both sides
    });

    // Configure custom blending for additive-style modes
    // CRITICAL: Use OneFactor to avoid squaring intensity (SrcAlpha would square it)
    if (blendingMode === 'additive' || blendingMode === 'luminous') {
      // Linear additive: final = src + dst (no alpha multiplication)
      this.blendEquation = THREE.AddEquation;
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
      // Prevent alpha accumulation that causes bloom/postprocessing artifacts.
      // With AddEquation, alpha would sum: 1.0 + 1.0 + ... = N per overlapping splat,
      // overflowing HalfFloat16 and causing dark halos via premultipliedAlpha compositing.
      // MaxEquation keeps alpha = max(1.0, existing) = 1.0, preventing accumulation.
      this.blendEquationAlpha = THREE.MaxEquation;
      this.blendSrcAlpha = THREE.OneFactor;
      this.blendDstAlpha = THREE.OneFactor;
    } else if (blendingMode === 'max') {
      this.blendEquation = THREE.MaxEquation; // Max(source, destination)
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
    }

    // Store blendingMode, gamma, and scalarRange in userData for clone()
    this.userData.blendingMode = blendingMode;
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? !isAdditive;
    this.userData.scalarRange = materialConfig.scalarRange;
  }

  /**
   * Update camera parameters for perspective projection.
   *
   * @param fov - Field of view in radians
   * @param resolution - Viewport resolution
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    if (isOrtho) {
      // fov = frustumHeight in world units; direct linear mapping
      const fy = resolution.y / fov;
      this.uniforms.uFx.value = fy;
      this.uniforms.uFy.value = fy;
    } else {
      // Compute focal lengths in pixels from FOV
      // f = height / (2 * tan(fov/2)) for vertical FOV
      const tanHalfFov = Math.tan(fov / 2);
      const fy = resolution.y / (2 * tanHalfFov);
      this.uniforms.uFx.value = fy;
      this.uniforms.uFy.value = fy;
    }

    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
  }

  /**
   * Update opacity.
   */
  updateOpacity(opacity: number): void {
    this.uniforms.uOpacity.value = opacity;
  }

  /**
   * Update truncation radius and recompute shifted Gaussian parameters.
   */
  updateTruncationRadius(radius: number): void {
    this.uniforms.uTruncate.value = radius;
    this.uniforms.uTruncateSq.value = radius * radius;
    const shiftC = Math.exp(-0.5 * radius * radius);
    this.uniforms.uShiftC.value = shiftC;
    this.uniforms.uInvOneMinusC.value = 1.0 / (1.0 - shiftC);
    this.uniforms.uRayIntegralFactor.value = GSplatMaterial.computeRayIntegralFactor(radius);
  }

  /**
   * Update max extent factor (projected splat size limit as fraction of viewport).
   * Lower values = more aggressive culling of close/large splats (better perf).
   * Default 0.33 means fade starts when a splat fills ~1/3 of the viewport.
   */
  updateMaxExtentFactor(factor: number): void {
    this.uniforms.uMaxExtentFactor.value = Math.max(0.01, factor);
  }

  /**
   * Update gamma correction.
   * Only invGamma is used in shader; gamma value stored in userData for clone()
   */
  updateGamma(gamma: number): void {
    const safeGamma = Math.max(0.001, gamma); // Prevent division by zero
    this.userData.gamma = safeGamma;
    this.uniforms.uInvGamma.value = 1.0 / safeGamma;
  }

  /**
   * Update intensity (linear color multiplier)
   */
  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
  }

  /**
   * Update offset (additive brightness shift)
   */
  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
  }

  /**
   * Update the colormap texture and enable/disable colormap mode.
   *
   * @param texture - Colormap LUT texture (256x1 RGB), or null to disable
   */
  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const wasEnabled = 'USE_COLORMAP' in this.defines;
    const nowEnabled = !!texture;

    if (nowEnabled) {
      this.defines.USE_COLORMAP = '';
      if (!this.uniforms.uColormapTex) {
        this.uniforms.uColormapTex = { value: texture };
        this.uniforms.uScalarMin = { value: 0.0 };
        this.uniforms.uScalarScale = { value: 1.0 };
      } else {
        this.uniforms.uColormapTex.value = texture;
      }
    } else {
      delete this.defines.USE_COLORMAP;
    }

    if (wasEnabled !== nowEnabled) {
      this.needsUpdate = true; // Triggers shader recompilation
    }
  }

  /**
   * Set the scalar data range for colormap normalization.
   *
   * @param min - Minimum scalar value (maps to LUT index 0)
   * @param max - Maximum scalar value (maps to LUT index 255)
   */
  updateScalarRange(min: number, max: number): void {
    if (this.uniforms.uScalarMin) {
      this.uniforms.uScalarMin.value = min;
    }
    if (this.uniforms.uScalarScale) {
      this.uniforms.uScalarScale.value = 1.0 / Math.max(1e-10, max - min);
    }
    this.userData.scalarRange = [min, max];
  }

  /**
   * Compute the ray integration factor for the shifted Gaussian.
   *
   * For the unshifted Gaussian, this is sqrt(2π) ≈ 2.507.
   * For the shifted Gaussian: sqrt(2π)·erf(T/√2) - 2·T·exp(-0.5·T²)
   * For T=3: ≈ 2.433
   */
  private static computeRayIntegralFactor(truncate: number): number {
    const SQRT_2PI = Math.sqrt(2 * Math.PI);
    // Abramowitz & Stegun erf approximation (max error 1.5e-7)
    const x = truncate / Math.SQRT2;
    const t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
    const erfVal =
      1.0 -
      t *
        (0.254829592 +
          t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) *
        Math.exp(-x * x);
    const erf = x >= 0 ? erfVal : -erfVal;
    return SQRT_2PI * erf - 2 * truncate * Math.exp(-0.5 * truncate * truncate);
  }

  /**
   * Clone this material.
   */
  clone(): this {
    const cloned = new GSplatMaterial({
      opacity: this.uniforms.uOpacity.value,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      truncationRadius: this.uniforms.uTruncate.value,
      blendingMode: this.userData.blendingMode ?? 'additive',
      transparent: this.transparent,
      depthTest: this.userData.depthTest ?? true,
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
    });

    // Copy blend equation settings for custom blending (additive/luminous/max modes)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
      cloned.blendEquationAlpha = this.blendEquationAlpha;
      cloned.blendSrcAlpha = this.blendSrcAlpha;
      cloned.blendDstAlpha = this.blendDstAlpha;
    }

    cloned.uniforms.uFx.value = this.uniforms.uFx.value;
    cloned.uniforms.uFy.value = this.uniforms.uFy.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    cloned.uniforms.uProjectionMode.value = this.uniforms.uProjectionMode.value;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;

    return cloned as this;
  }

  /**
   * Dispose this material and unregister from MaterialManager.
   */
  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
