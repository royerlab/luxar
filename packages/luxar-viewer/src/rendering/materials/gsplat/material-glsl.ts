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
 * @module rendering/materials/gsplat/material-glsl
 */

import * as THREE from 'three';
import { GSPLAT_VERTEX_SHADER, GSPLAT_FRAGMENT_SHADER } from './shader-glsl';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne } from '../_shared/uniform-helpers';
import { computeFocalLength } from '../_shared/camera-uniforms';
import { computeRayIntegralFactor } from './math';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  getGSplatNormalBlendingState,
  isAdditiveMode,
  isLuminousMode,
  isMaxMode,
  isNormalMode,
  isOpaqueMode,
} from '../../blending-state';

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
 *
 * @internal — reserved extension shape; no current consumer.
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
export class GSplatMaterial
  extends THREE.ShaderMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /**
   * Create a new GSplatMaterial with the specified configuration.
   *
   * @param materialConfig - Material configuration options
   */
  constructor(materialConfig: GSplatMaterialConfig = {}) {
    const blendingMode = materialConfig.blendingMode ?? 'additive';
    const isOpaque = blendingMode === 'opaque';
    const isAdditive = blendingMode === 'additive';
    const gammaValue = clampGamma(materialConfig.gamma);

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
          value: computeRayIntegralFactor(truncate),
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

      // Preprocessor defines — USE_COLORMAP enables LUT lookup from amplitude;
      // LUXAR_GAMMA_ONE skips the per-fragment gamma pow() when gamma == 1.0
      // (toggled by `updateGamma`).
      defines: {
        ...(materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {}),
        ...(isGammaOne(gammaValue) ? { LUXAR_GAMMA_ONE: '' } : {}),
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

    // Apply mode-specific blending state (sets blending, blendEquation,
    // blendSrc/Dst, alpha equation, transparent/depthTest/depthWrite,
    // userData.blendingMode, and uProjectionMode). Same path used by
    // live mode updates from the layers panel — keeps construction and
    // runtime in sync.
    this.applyBlendingMode(blendingMode);

    // Honor explicit overrides from the config after the mode-derived
    // defaults. Production callers don't pass these; preserved for
    // existing consumer parity.
    if (materialConfig.transparent !== undefined) {
      this.transparent = materialConfig.transparent;
    }
    if (materialConfig.depthTest !== undefined) {
      this.depthTest = materialConfig.depthTest;
    }

    // Store gamma + scalarRange in userData for clone(); blendingMode
    // and depthTest are already set by applyBlendingMode.
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

    const fy = computeFocalLength(fov, resolution.y, isOrtho);
    this.uniforms.uFx.value = fy;
    this.uniforms.uFy.value = fy;

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
    this.uniforms.uRayIntegralFactor.value = computeRayIntegralFactor(radius);
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
   *
   * When `gamma` crosses the 1.0 threshold (with epsilon), toggle the
   * `LUXAR_GAMMA_ONE` define so the fragment shader's pow() fast path is
   * recompiled in/out (mirrors the Line material).
   */
  updateGamma(gamma: number): void {
    const safeGamma = clampGamma(gamma);
    this.userData.gamma = safeGamma;
    this.uniforms.uInvGamma.value = 1.0 / safeGamma;

    if (!this.defines) this.defines = {};
    const wantGammaOne = isGammaOne(safeGamma);
    const hadGammaOne = 'LUXAR_GAMMA_ONE' in this.defines;
    if (wantGammaOne && !hadGammaOne) {
      this.defines.LUXAR_GAMMA_ONE = '';
      this.needsUpdate = true;
    } else if (!wantGammaOne && hadGammaOne) {
      delete this.defines.LUXAR_GAMMA_ONE;
      this.needsUpdate = true;
    }
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
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
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
    applyScalarRangeToMaterial(this, min, max);
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

  // Dispose is inherited from THREE.ShaderMaterial. The MaterialManager
  // subscribes to the synchronous `dispose` event THREE fires from
  // super.dispose(), so registry cleanup happens automatically without
  // this file needing to import the manager (which would create a cycle).

  /**
   * Apply a blending mode to this material in-place.
   *
   * GSplat-specific because the constructor uses `CustomBlending +
   * OneFactor` (NOT `THREE.AdditiveBlending` — that uses SrcAlpha which
   * squares intensity) for additive/luminous, and toggles the
   * `uProjectionMode` uniform when switching to/from `max`. Without a
   * type-specific method, the layers panel's generic
   * `mat.blending = state.blending` would either:
   *   - assign `AdditiveBlending` (squaring intensity) when switching to
   *     additive/luminous, or
   *   - leave `uProjectionMode = 0` while the framebuffer blends with
   *     `MaxEquation` — physically wrong max projection.
   *
   * `normal` is gsplat-specific too: the shader emits premultiplied
   * coverage alpha under the `LUXAR_NORMAL_PREMULT` define (toggled
   * here — the only mode-driven define this material has), paired with
   * `getGSplatNormalBlendingState()`'s One / OneMinusSrcAlpha state.
   *
   * Used by both the constructor and runtime mode changes from the
   * layers panel. After this returns, `userData.blendingMode` reflects
   * the live mode so subsequent `clone()` calls preserve it.
   */
  applyBlendingMode(mode: 'additive' | 'normal' | 'max' | 'opaque' | 'luminous'): void {
    const previousMode = this.userData.blendingMode as
      | 'additive'
      | 'normal'
      | 'max'
      | 'opaque'
      | 'luminous'
      | undefined;
    // Predicate-driven mode dispatch.
    const isOpaque = isOpaqueMode(mode);
    const isAdditive = isAdditiveMode(mode);
    const isMax = isMaxMode(mode);

    if (isNormalMode(mode)) {
      // Premultiplied alpha-over — the one mode where the fragment
      // shader emits a real (coverage) alpha. State comes from the
      // shared helper; see its doc for why CustomBlending + symmetric
      // alpha channel + depthWrite:false are each load-bearing.
      const state = getGSplatNormalBlendingState();
      this.blending = state.blending;
      this.blendEquation = state.blendEquation;
      this.blendSrc = state.blendSrc;
      this.blendDst = state.blendDst;
      // Symmetric alpha channel: reset any stranded additive
      // alpha-MaxEquation state (null = "track the RGB equation").
      this.blendEquationAlpha = null;
      this.blendSrcAlpha = null;
      this.blendDstAlpha = null;
      this.transparent = state.transparent;
      this.depthTest = state.depthTest;
      this.depthWrite = state.depthWrite;
      this.defines.LUXAR_NORMAL_PREMULT = '';
      if (this.uniforms.uProjectionMode) {
        this.uniforms.uProjectionMode.value = 0; // sum projection
      }
      this.userData.blendingMode = mode;
      this.userData.depthTest = this.depthTest;
      // Define toggled ⇒ program recompile needed on a real mode change.
      if (previousMode !== mode) {
        this.needsUpdate = true;
      }
      return;
    }

    // Every non-normal mode renders with the alpha=1.0 fragment contract.
    delete this.defines.LUXAR_NORMAL_PREMULT;

    // Pick base blending. Additive-style modes go through CustomBlending
    // so the alpha factors below take effect.
    if (isOpaque) {
      this.blending = THREE.NormalBlending;
    } else {
      this.blending = THREE.CustomBlending;
    }

    this.transparent = !isOpaque;
    this.depthWrite = isOpaque;
    // Additive ignores depth (renders on top); luminous respects it.
    this.depthTest = !isAdditive;

    // Projection mode uniform: 0=sum (additive/luminous/normal/opaque),
    // 1=max. The shader has separate sum vs max branches.
    if (this.uniforms.uProjectionMode) {
      this.uniforms.uProjectionMode.value = isMax ? 1 : 0;
    }

    // Mode-specific blend factors.
    if (isAdditive || isLuminousMode(mode)) {
      // Linear sum projection — OneFactor avoids the SrcAlpha squaring.
      this.blendEquation = THREE.AddEquation;
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
      // MaxEquation on alpha prevents accumulation that would otherwise
      // overflow HalfFloat16 and produce dark halos in post-processing.
      this.blendEquationAlpha = THREE.MaxEquation;
      this.blendSrcAlpha = THREE.OneFactor;
      this.blendDstAlpha = THREE.OneFactor;
    } else if (isMax) {
      this.blendEquation = THREE.MaxEquation;
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
      // Alpha tracks RGB by default in CustomBlending — null means "use
      // the RGB equation". Reset so a previous additive→max switch
      // doesn't strand MaxEquation alpha state.
      this.blendEquationAlpha = null;
      this.blendSrcAlpha = null;
      this.blendDstAlpha = null;
    } else {
      // opaque: NormalBlending is selected above and ignores these.
      // Reset to THREE defaults so a switch back from custom blending
      // starts from a clean slate.
      this.blendEquation = THREE.AddEquation;
      this.blendSrc = THREE.SrcAlphaFactor;
      this.blendDst = THREE.OneMinusSrcAlphaFactor;
      this.blendEquationAlpha = null;
      this.blendSrcAlpha = null;
      this.blendDstAlpha = null;
    }

    this.userData.blendingMode = mode;
    this.userData.depthTest = this.depthTest;
    // Mark needsUpdate when the mode actually changed — blending-state
    // flushes and (normal↔other) LUXAR_NORMAL_PREMULT toggles both need
    // one program-state refresh.
    if (previousMode !== mode) {
      this.needsUpdate = true;
    }
  }

  // -------------------------------------------------------------
  // ColormapAwareMaterial — see colormap-aware-material.ts.
  // -------------------------------------------------------------

  setColormapTexture(texture: THREE.DataTexture | null): void {
    if (texture) {
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
      if (this.uniforms.uColormapTex) this.uniforms.uColormapTex.value = null;
      if (this.uniforms.uScalarMin) this.uniforms.uScalarMin.value = 0.0;
      if (this.uniforms.uScalarScale) this.uniforms.uScalarScale.value = 1.0;
    }
  }

  setScalarRange(min: number, max: number): void {
    if (this.uniforms.uScalarMin) this.uniforms.uScalarMin.value = min;
    if (this.uniforms.uScalarScale) {
      this.uniforms.uScalarScale.value = 1.0 / Math.max(1e-10, max - min);
    }
  }
}
