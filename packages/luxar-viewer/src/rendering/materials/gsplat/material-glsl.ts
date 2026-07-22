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
 * - Blending-mode-aware projection (sum ray-integral for emissive additive/luminous, peak for surface max/normal/opaque)
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
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { BlendingMode } from '../../material-manager';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne } from '../_shared/uniform-helpers';
import { computeFocalLength } from '../_shared/camera-uniforms';
import {
  computeRayIntegralFactor,
  clampTruncationRadius,
  GSPLAT_COV2D_DILATION_DEFAULT,
} from './math';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  getCompleteBlendingState,
  getGSplatNormalBlendingState,
  isNormalMode,
  isVolumetricMode,
  usesPeakProjection,
} from '../../blending-state';

/**
 * Configuration for gsplat material creation
 */
export interface GSplatMaterialConfig {
  /** Opacity multiplier (0.0 to 1.0) */
  opacity?: number;
  /**
   * Absorption coefficient κ (>= 0, default 1.0) — read only by the
   * `volumetric` blending mode's fragment branch (τ = κ·opacity·rayMass);
   * inert in every other mode. κ = 0 renders exactly like `additive`.
   */
  absorption?: number;
  /**
   * True when the dataset's colors carry a per-splat alpha channel (RGBA).
   * Gates the volumetric branch's alpha → optical-depth mapping
   * (w = −ln(1−a)); the linear per-mode alpha factor needs no gate (RGB
   * data carries the identity alpha 1.0 in the splat texture).
   */
  hasElementAlpha?: boolean;
  /** Gamma correction (0.1 to 10.0, default 1.0) */
  gamma?: number;
  /** Intensity (linear color multiplier / gain), default 1.0 */
  intensity?: number;
  /** Offset (additive brightness shift / black level), default 0.0 */
  offset?: number;
  /** Truncation radius in sigmas (default 3.0) */
  truncationRadius?: number;
  /** Blending mode */
  blendingMode?: BlendingMode;
  /**
   * Explicit override, applied AFTER `applyBlendingMode`'s
   * mode-derived value (true for every mode except `opaque`).
   * Production callers never pass it; clone() passes the parent's
   * current (already mode-derived) value, so the round-trip is a
   * no-op — but the constructor DOES consume it, so a caller-supplied
   * value wins over the mode derivation.
   */
  transparent?: boolean;
  /** Whether to test against depth buffer (default true; additive sets false) */
  depthTest?: boolean;
  /** Colormap texture for scalar-to-color mapping (256x1 RGB) */
  colormapTexture?: THREE.DataTexture;
  /** Scalar data range [min, max] for normalization before LUT lookup */
  scalarRange?: [number, number];
  /** Max projected splat extent as a fraction of viewport size before fade-out (default 0.33) */
  maxExtentFactor?: number;
  /**
   * 2D-covariance low-pass dilation in pixels² added to the Σ_2D diagonal
   * (standard 3DGS anti-aliasing; default 0.3). Guarantees every splat covers
   * ≥ ~1px so near-degenerate (edge-on flat) splats — common in imported
   * classical 3DGS fits — render as soft ellipses instead of razor-thin spikes.
   */
  cov2DDilation?: number;
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
  /** Absorption coefficient κ (volumetric mode: τ = κ·opacity·rayMass) */
  uAbsorption: { value: number };
  /** 1 when colors are RGBA (per-splat opacity present), else 0 */
  uHasElementAlpha: { value: number };
  /** Projection mode: 0=sum ray-integral (additive/luminous/volumetric), 1=peak 2D-projected (surface modes max/normal/opaque) */
  uProjectionMode: { value: number };
  /** Pre-computed 1/gamma for performance */
  uInvGamma: { value: number };
  /** Near cull distance in world units (scene-scale-aware, perspective only) */
  uNearCull: { value: number };
  /** Max projected splat extent as fraction of viewport before fade-out */
  uMaxExtentFactor: { value: number };
  /** 2D-covariance low-pass dilation in px² added to the Σ_2D diagonal */
  uCov2DDilation: { value: number };
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
    const gammaValue = clampGamma(materialConfig.gamma);

    const truncate = clampTruncationRadius(materialConfig.truncationRadius ?? 3.0);
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    super({
      uniforms: {
        // Splat data texture (RGBA32F, 4 texels/splat) — bound by the
        // commit's material sync from the acquired pool entry (or the
        // fallback mesh's own texture). The shared placeholder until the
        // first commit (mirrors PointMaterial — the sampler is never
        // unbound).
        uSplatTex: { value: getPlaceholderElementTexture() },
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
        uAbsorption: { value: materialConfig.absorption ?? 1.0 },
        uHasElementAlpha: { value: materialConfig.hasElementAlpha ? 1 : 0 },
        // Placeholder — applyBlendingMode() below is the source of truth. Peak (1)
        // for surface modes (max/normal/opaque), sum (0) for emissive.
        uProjectionMode: {
          value: usesPeakProjection(blendingMode) ? 1 : 0,
        },
        uInvGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },
        uIsOrtho: { value: 0 }, // 0 = perspective, 1 = orthographic
        uNearCull: { value: 0.1 }, // Default; overridden per-scene by updateCameraParams
        uMaxExtentFactor: { value: materialConfig.maxExtentFactor ?? 0.33 },
        uCov2DDilation: {
          value: materialConfig.cov2DDilation ?? GSPLAT_COV2D_DILATION_DEFAULT,
        },
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

      // Blending / transparency / depth state is fully owned by
      // applyBlendingMode (called right below) — the same path live
      // mode updates take. Neutral placeholders here.
      transparent: true,
      depthWrite: false,
      depthTest: materialConfig.depthTest ?? true,
      toneMapped: false, // HDR values pass through to post-processing
      blending: THREE.NormalBlending,
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
    // and userData.depthTest are already set by applyBlendingMode —
    // only an explicit config override re-stamps it here.
    this.userData.gamma = gammaValue;
    if (materialConfig.depthTest !== undefined) {
      this.userData.depthTest = materialConfig.depthTest;
    }
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

  /** Current opacity multiplier (the LOD cross-fade snapshots this as its fade base). */
  getOpacity(): number {
    return this.uniforms.uOpacity.value as number;
  }

  /**
   * Update the absorption coefficient κ (volumetric mode only — a plain
   * uniform write, no recompile; inert in every other mode).
   */
  updateAbsorption(absorption: number): void {
    this.uniforms.uAbsorption.value = absorption;
  }

  /** Current absorption coefficient κ. */
  getAbsorption(): number {
    return this.uniforms.uAbsorption.value as number;
  }

  /**
   * Declare whether the committed dataset's colors carry a per-splat
   * alpha channel (RGBA). Plain uniform write — set by the commit layer
   * once the color layout is known.
   */
  updateHasElementAlpha(hasAlpha: boolean): void {
    this.uniforms.uHasElementAlpha.value = hasAlpha ? 1 : 0;
  }

  /**
   * Update truncation radius and recompute shifted Gaussian parameters.
   */
  updateTruncationRadius(radius: number): void {
    radius = clampTruncationRadius(radius);
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
   * Rebind the splat data texture (pool acquire may hand the node a
   * different geometry+texture pair on growth or best-fit reuse).
   * Plain uniform update — no shader recompilation involved.
   */
  updateSplatTexture(texture: THREE.DataTexture | null): void {
    // `null` falls back to the shared placeholder (never unbind the
    // sampler) — mirrors PointMaterial.updatePointTexture.
    this.uniforms.uSplatTex.value = texture ?? getPlaceholderElementTexture();
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
      // Without this a clone silently reset a tuned κ to the 1.0 default —
      // and the layers panel clones on ANY first panel interaction.
      absorption: this.uniforms.uAbsorption.value,
      hasElementAlpha: this.uniforms.uHasElementAlpha.value === 1,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      truncationRadius: this.uniforms.uTruncate.value,
      blendingMode: this.userData.blendingMode ?? 'additive',
      transparent: this.transparent,
      depthTest: this.userData.depthTest ?? true,
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
      // Without this a clone silently reset a tuned extent factor to the
      // 0.33 constructor default (screen-coverage fade threshold).
      maxExtentFactor: this.uniforms.uMaxExtentFactor.value,
      cov2DDilation: this.uniforms.uCov2DDilation.value,
    });

    // Copy blend equation settings for custom blending (max/normal —
    // additive/luminous use plain AdditiveBlending since the unification)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
      cloned.blendEquationAlpha = this.blendEquationAlpha;
      cloned.blendSrcAlpha = this.blendSrcAlpha;
      cloned.blendDstAlpha = this.blendDstAlpha;
    }

    cloned.uniforms.uSplatTex.value = this.uniforms.uSplatTex.value;
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
   * GSplat-specific because the method sets the `uProjectionMode`
   * uniform — PEAK (1) for the surface modes (`max` / `normal` /
   * `opaque`, see `usesPeakProjection`), SUM ray-integral (0) for
   * emissive (`additive`/`luminous`/`volumetric`) — and owns the
   * `LUXAR_NORMAL_PREMULT` + `LUXAR_VOLUMETRIC` define lifecycles. Without a type-specific method,
   * the layers panel's generic `mat.blending = state.blending` would leave a
   * stale `uProjectionMode` while the framebuffer blend state changed —
   * physically wrong projection.
   *
   * `normal` is gsplat-specific too: the shader emits premultiplied
   * coverage alpha under the `LUXAR_NORMAL_PREMULT` define (toggled
   * here, alongside the `LUXAR_VOLUMETRIC` define for the
   * emission–absorption branch), paired with
   * `getGSplatNormalBlendingState()`'s One / OneMinusSrcAlpha state.
   *
   * Used by both the constructor and runtime mode changes from the
   * layers panel. After this returns, `userData.blendingMode` reflects
   * the live mode so subsequent `clone()` calls preserve it.
   */
  applyBlendingMode(mode: BlendingMode): void {
    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    // NOTE: opacity is INERT for gsplat blend state — the only
    // opacity-sensitive entry in getCompleteBlendingState is the
    // generic 'normal' one, and gsplat normal early-returns to the
    // opacity-independent getGSplatNormalBlendingState above. Passed
    // only to satisfy the shared helper's signature.
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;

    // Defensive: THREE may leave `defines` undefined when none were
    // passed at construction (and mocked-THREE test environments do).
    // Both branches below own the LUXAR_NORMAL_PREMULT /
    // LUXAR_VOLUMETRIC define lifecycles — same guard as the point/line
    // wrappers (three-geometry symmetry).
    if (!this.defines) {
      this.defines = {};
    }

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
      // Every branch owns BOTH mode defines — a volumetric→normal switch
      // must not strand LUXAR_VOLUMETRIC.
      delete this.defines.LUXAR_VOLUMETRIC;
      if (this.uniforms.uProjectionMode) {
        // Peak (2D-projected) projection, NOT the sum ray-integral: alpha-over
        // is the SURFACE compositing model, so a splat's contribution is its
        // projected 2D-Gaussian peak (surface density at the ray hit), not the
        // emissive line-integral through the 3D Gaussian. The ray-integral boost
        // (~2.4×·sigmaRay) would inflate both brightness AND the coverage alpha —
        // saturating classical-3DGS surface splats to fully opaque and producing
        // grazing streaks. Matches standard 3DGS rasterizers. (additive/luminous
        // stay sum; the other surface modes max/opaque are peak too.)
        this.uniforms.uProjectionMode.value = 1; // peak projection
      }
      this.userData.blendingMode = mode;
      this.userData.depthTest = this.depthTest;
      // Define toggled ⇒ program recompile needed on a real mode change.
      if (previousMode !== mode) {
        this.needsUpdate = true;
      }
      return;
    }

    // Every non-normal mode drops the coverage-alpha define; volumetric
    // gets its own fragment branch (emission–absorption: RGB carries the
    // self-screened emission, alpha = 1 − e^(−τ)) — every OTHER mode
    // keeps the alpha=1.0 contract.
    delete this.defines.LUXAR_NORMAL_PREMULT;
    if (isVolumetricMode(mode)) {
      this.defines.LUXAR_VOLUMETRIC = '';
    } else {
      delete this.defines.LUXAR_VOLUMETRIC;
    }

    // Non-normal modes take the SHARED blending state — the same source
    // of truth the TSL wrapper uses, so both backends are identical:
    //   additive/luminous → AdditiveBlending (SrcAlpha + One). With the
    //     shader's alpha = 1.0 contract, SrcAlpha is the identity factor,
    //     so this is exactly the linear One + One sum (TSL relied on
    //     this equivalence all along; parity is pixel-exact).
    //   max → CustomBlending + MaxEquation + One/One.
    //   opaque → NormalBlending + depth write.
    // The historical GLSL-only CustomBlending dance with a SEPARATE
    // alpha-channel MaxEquation guard is gone: its only remaining
    // purpose was keeping accumulated alpha finite for the
    // raw-scene-hdr capture path, which now sanitizes alpha at the
    // readback boundary for every geometry type (see
    // post-processing-manager/capture.ts).
    const state = getCompleteBlendingState(mode, opacity);
    this.blending = state.blending;
    this.blendEquation = state.blendEquation;
    this.blendSrc = state.blendSrc;
    this.blendDst = state.blendDst;
    // Symmetric alpha channel: null = "track the RGB equation". Also
    // clears any stranded per-alpha state from materials created before
    // this unification.
    this.blendEquationAlpha = null;
    this.blendSrcAlpha = null;
    this.blendDstAlpha = null;
    this.transparent = state.transparent;
    this.depthTest = state.depthTest;
    this.depthWrite = state.depthWrite;

    // Projection mode uniform: 0=sum (additive/luminous/volumetric), 1=peak
    // (max/normal/opaque). The shader has separate sum vs peak
    // branches; the surface modes all project the 2D-Gaussian peak
    // (see usesPeakProjection's taxonomy note).
    if (this.uniforms.uProjectionMode) {
      this.uniforms.uProjectionMode.value = usesPeakProjection(mode) ? 1 : 0;
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
