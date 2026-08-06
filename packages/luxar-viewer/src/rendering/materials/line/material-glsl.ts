/**
 * Line Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for rendering thick lines using instanced quads.
 * Uses the semicircle kernel model for mathematically correct joints with additive blending.
 *
 * Key features:
 * - Instanced quad geometry (4 vertices per segment)
 * - World-space line width
 * - Parabolic intensity falloff: (1 - p²)^sharpness
 * - Cap factor for seamless joints (0.5 at endpoints, 1.0 in body)
 * - Per-vertex attributes (color, width, sharpness)
 *
 * @module rendering/materials/line/material-glsl
 */

import * as THREE from 'three';
import { LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER } from './shader-glsl';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne, isNoGOG } from '../_shared/uniform-helpers';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  isVolumetricMode,
  normalModeDepthWrite,
  type CompleteBlendingState,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import { computeScalarRangeUniforms, scalarRangeUniformEntries } from '../_shared/scalar-range';
import { resolveLineJoin, type LineJoinStyle } from '../../../types/line-join';

// `isGammaOne` and `isNoGOG` both live in `../_shared/uniform-helpers`
// (shared across all three geometry types). Re-exported here so
// `material-tsl.ts` and existing importers keep their
// `./material-glsl` import path.
export { isGammaOne, isNoGOG };

/**
 * Configuration for line material creation
 */
export interface LineMaterialConfig {
  /** Opacity multiplier (0.0 to 1.0) */
  opacity?: number;
  /** Gamma correction (0.1 to 10.0, default 1.0) */
  gamma?: number;
  /** Intensity (linear color multiplier / gain), default 1.0 */
  intensity?: number;
  /** Offset (additive brightness shift / black level), default 0.0 */
  offset?: number;
  /**
   * κ — composed node absorption for the volumetric blending mode
   * (identity 1.0). Inert in every other mode; same shape as
   * `PointMaterialConfig.absorption` and `GSplatMaterialConfig.absorption`
   * (three-geometry symmetry).
   */
  absorption?: number;
  /**
   * True when the dataset's colors carry a per-element alpha channel
   * (RGBA). Gates the volumetric branch's alpha → optical-depth mapping
   * (w = −ln(1−a)); the linear per-mode alpha factor needs no gate (RGB
   * data carries the identity alpha 1.0 in the element texture). Same
   * shape as `GSplatMaterialConfig.hasElementAlpha` (three-geometry
   * symmetry). Normally pushed per-commit by the material sync; the
   * config field exists so clone() round-trips it.
   */
  hasElementAlpha?: boolean;
  /** Blending mode */
  blendingMode?: BlendingMode;
  /** Whether material is transparent (default true) */
  transparent?: boolean;
  /** Whether to test against depth buffer (default true; additive sets false) */
  depthTest?: boolean;
  /** Colormap texture for scalar-to-color mapping (256x1 RGB) */
  colormapTexture?: THREE.DataTexture;
  /** Scalar data range [min, max] for normalization before LUT lookup */
  scalarRange?: [number, number];
  /**
   * Join style at degree-2 polyline joints (#790). Omitted ⇒ the session
   * override if one is set, else {@link DEFAULT_LINE_JOIN}. See
   * `types/line-join.ts` for the cost/fidelity ladder.
   */
  join?: LineJoinStyle;
}

/**
 * Line material uniforms interface
 *
 * @internal — reserved extension shape; no current consumer.
 */
export interface LineMaterialUniforms {
  /** Viewport resolution [width, height] */
  uResolution: { value: THREE.Vector2 };
  /** Opacity multiplier */
  uOpacity: { value: number };
  /** Pre-computed 1/gamma for performance */
  uInvGamma: { value: number };
}

/**
 * Line material using instanced quads with semicircle kernel rendering.
 *
 * The semicircle kernel produces parabolic intensity profiles that sum correctly
 * at joints when using additive blending:
 * - Body intensity: (1 - p²)^sharpness where p = perpendicular distance
 * - Endpoint cap factor: 0.5 (half intensity at true endpoints)
 * - Joint rendering: 0.5 + 0.5 = 1.0 (seamless sum)
 */
export class LineMaterial
  extends THREE.ShaderMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /**
   * Create a new LineMaterial with the specified configuration.
   *
   * @param materialConfig - Material configuration options
   */
  constructor(materialConfig: LineMaterialConfig = {}) {
    const blendingMode = materialConfig.blendingMode ?? 'additive';
    const isOpaque = blendingMode === 'opaque';
    const isAdditive = blendingMode === 'additive';
    const gammaValue = clampGamma(materialConfig.gamma);

    // Determine THREE.js blending mode
    // 'additive' and 'luminous' both use AdditiveBlending - only depthTest differs
    const initialMode = blendingMode;
    let blending: THREE.Blending;
    if (isOpaque || initialMode === 'normal') {
      blending = THREE.NormalBlending;
    } else if (initialMode === 'additive' || initialMode === 'luminous') {
      blending = THREE.AdditiveBlending; // Classic additive: SrcAlpha, One
    } else if (initialMode === 'max' || initialMode === 'volumetric') {
      blending = THREE.CustomBlending;
    } else {
      blending = THREE.NormalBlending;
    }

    super({
      uniforms: {
        // Per-node line data texture (6 texels/segment). Starts on the
        // shared placeholder; the commit sync rebinds the geometry's
        // acquired pool entry's texture via `updateLineTexture`.
        uLineTex: { value: getPlaceholderElementTexture() },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uIsOrtho: { value: 0 }, // 0 = perspective, 1 = orthographic
        // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
        // Flipped by the depth-sort coordinator once the inactive buffer
        // holds a whole permutation (runtime uniform: never a define — a
        // flip must not recompile the program).
        uSortedIndexSlot: { value: 0 },
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
        uInvGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },
        // Volumetric (emission–absorption) uniforms — read only under
        // the LUXAR_VOLUMETRIC define; inert in every other mode.
        uAbsorption: { value: materialConfig.absorption ?? 1.0 },
        // 1 when the committed colors carry a real alpha column (RGBA);
        // pushed per-commit (material-sync-helpers.ts), gates only the
        // volumetric w(a) optical-depth map; config seeds it for clone().
        uHasElementAlpha: { value: materialConfig.hasElementAlpha ? 1 : 0 },
        // near-plane safety + max-pixel-width clamp uniforms.
        // 0.1 matches the point/gsplat ctor default (pre-first-broadcast
        // window only; updateCameraParams overwrites with the scene value).
        uNearCull: { value: 0.1 },
        uMaxLinePixelWidth: { value: 540 }, // ≈ resolution.y * 0.5 default; updated in updateCameraParams
        // CPU-precomputed pixel-width scales so the shader avoids
        // per-vertex tan() and one divide. Updated in updateCameraParams.
        uPerspectiveLineScale: { value: 1.0 },
        uOrthoLineScale: { value: 1.0 },
        // Join style at degree-2 polyline joints (#790): 0 none, 1 miter —
        // see types/line-join.ts for the styles and the override precedence.
        // A live uniform rather than a define so the session override costs no
        // recompile, and so an A/B can measure both styles on one identical
        // frame.
        uLineJoin: { value: resolveLineJoin(materialConfig.join) },
        // Colormap uniforms (only when USE_COLORMAP define is set)
        ...(materialConfig.colormapTexture
          ? {
              uColormapTex: { value: materialConfig.colormapTexture },
              // Midpoint identity for degenerate ranges — see scalar-range.ts.
              ...scalarRangeUniformEntries(materialConfig.scalarRange),
            }
          : {}),
      },

      vertexShader: LINE_VERTEX_SHADER,
      fragmentShader: LINE_FRAGMENT_SHADER,

      // Preprocessor defines. Variant `#define`s (e.g.
      // `LUXAR_GAMMA_ONE`) gate fragment-stage fast paths and are
      // toggled by the wrapper's update methods when the underlying
      // value crosses the relevant threshold.
      defines: {
        ...(materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {}),
        ...(isGammaOne(gammaValue) ? { LUXAR_GAMMA_ONE: '' } : {}),
        ...(isNoGOG(materialConfig.intensity ?? 1.0, materialConfig.offset ?? 0.0)
          ? { LUXAR_NO_GOG: '' }
          : {}),
      },

      // GLSL ES 3.0 for consistency with other materials
      glslVersion: THREE.GLSL3,

      transparent: materialConfig.transparent ?? !isOpaque,
      depthWrite:
        isOpaque ||
        (blendingMode === 'normal' && normalModeDepthWrite(materialConfig.opacity ?? 1.0)),
      // Additive ignores depth (renders on top), luminous respects depth occlusion
      depthTest: materialConfig.depthTest ?? !isAdditive,
      toneMapped: false, // HDR values pass through to post-processing
      blending: blending,
      side: THREE.DoubleSide, // Lines visible from both sides
      // Line quads are screen-space billboards, not physically
      // two-sided surfaces. The renderer's two-pass guard
      // (`renderers/WebGLRenderer.js:1340`) trips only when
      // `transparent && side===DoubleSide && forceSinglePass===false`,
      // so forcing single-pass skips a redundant back-face render
      // pass per line layer.
      forceSinglePass: true,
    });

    // Apply mode-specific blending state via the canonical method —
    // same path used by live mode updates from the layers panel.
    this.applyBlendingMode(blendingMode);

    // Honor explicit overrides from config after mode-derived defaults.
    if (materialConfig.transparent !== undefined) {
      this.transparent = materialConfig.transparent;
    }
    if (materialConfig.depthTest !== undefined) {
      this.depthTest = materialConfig.depthTest;
    }

    // gamma + scalarRange in userData for clone(); blendingMode and
    // depthTest are already set by applyBlendingMode.
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? !isAdditive;
    this.userData.scalarRange = materialConfig.scalarRange;
  }

  /**
   * Update camera parameters for world-space line sizing.
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
    // Apply the near-plane safety distance when provided.
    // Accept ANY defined value, including 0 — matching the point/gsplat
    // wrappers (the shader floors at 1e-20). The old `> 0` gate silently
    // KEPT a stale value on zero-diagonal scenes (or, with LRU-cached
    // materials, the previous dataset's nearCull), re-creating the
    // cross-geometry near-fade divergence B9c fixed.
    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
    // clamp screen-space line width to half the viewport height so a
    // near-camera segment can't paint the entire screen.
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
    // Pre-compute pixel-width scales. Only the branch matching uIsOrtho
    // is read in the shader, but writing both keeps the GPU values
    // sane after a mode switch and avoids NaN from tan(frustumHeight/2)
    // when fov stores frustumHeight in ortho mode.
    const safeFov = Math.max(fov, 1e-4);
    if (isOrtho) {
      this.uniforms.uOrthoLineScale.value = (2.0 * resolution.y) / safeFov;
    } else {
      this.uniforms.uPerspectiveLineScale.value =
        resolution.y / Math.max(Math.tan(safeFov * 0.5), 1e-4);
    }
  }

  /**
   * Rebind the line data texture (pool acquire may hand the node a
   * different geometry+texture pair on growth or best-fit reuse).
   * Plain uniform update — no shader recompilation involved. Mirrors
   * `PointMaterial.updatePointTexture`; `null` falls back to the
   * shared placeholder so the sampler is never unbound.
   */
  updateLineTexture(texture: THREE.DataTexture | null): void {
    this.uniforms.uLineTex.value = texture ?? getPlaceholderElementTexture();
  }

  /** The currently bound line data texture. */
  getLineTexture(): THREE.DataTexture | null {
    return (this.uniforms.uLineTex.value as THREE.DataTexture | null) ?? null;
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
   * Update gamma correction.
   * Only invGamma is used in shader; gamma value stored in userData for clone().
   *
   * When `gamma` crosses the 1.0 threshold (with epsilon), toggle the
   * `LUXAR_GAMMA_ONE` define so the fragment shader's pow() path is
   * recompiled in/out. The Three.js shader-program cache rebuilds the
   * program on `needsUpdate = true`.
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
   * Toggle the `LUXAR_NO_GOG` define based on the live uniform values
   * for intensity + offset. Called by both `updateIntensity` and
   * `updateOffset` because the flag depends on both values jointly.
   * Returns true if the define changed (caller may need a rebuild).
   */
  private _refreshNoGOGDefine(): boolean {
    if (!this.defines) this.defines = {};
    const wantNoGOG = isNoGOG(
      this.uniforms.uIntensity.value as number,
      this.uniforms.uOffset.value as number
    );
    const hadNoGOG = 'LUXAR_NO_GOG' in this.defines;
    if (wantNoGOG && !hadNoGOG) {
      this.defines.LUXAR_NO_GOG = '';
      return true;
    }
    if (!wantNoGOG && hadNoGOG) {
      delete this.defines.LUXAR_NO_GOG;
      return true;
    }
    return false;
  }

  /**
   * Update intensity (linear color multiplier)
   */
  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
    if (this._refreshNoGOGDefine()) this.needsUpdate = true;
  }

  /**
   * Update offset (additive brightness shift)
   */
  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
    if (this._refreshNoGOGDefine()) this.needsUpdate = true;
  }

  /**
   * Update the volumetric absorption coefficient κ (composed node
   * attr). Plain uniform write — inert unless the material is in
   * volumetric mode. Mirrors `PointMaterial.updateAbsorption`.
   */
  updateAbsorption(absorption: number): void {
    this.uniforms.uAbsorption.value = absorption;
  }

  /** Current volumetric absorption κ (mirrors GSplatMaterial.getAbsorption). */
  getAbsorption(): number {
    return this.uniforms.uAbsorption.value as number;
  }

  /**
   * Flag whether the committed colors carry a real per-endpoint alpha
   * column (RGBA). Set per-commit by `syncLineMaterialWithGeometry`;
   * gates only the volumetric w(a) optical-depth map. Mirrors
   * `PointMaterial.updateHasElementAlpha`.
   */
  updateHasElementAlpha(hasAlpha: boolean): void {
    this.uniforms.uHasElementAlpha.value = hasAlpha ? 1 : 0;
  }

  /**
   * Update the colormap texture and enable/disable colormap mode.
   */
  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
    if (wasEnabled !== nowEnabled) {
      this.needsUpdate = true;
    }
  }

  /**
   * Set the scalar data range for colormap normalization.
   */
  updateScalarRange(min: number, max: number): void {
    applyScalarRangeToMaterial(this, min, max);
  }

  /**
   * Clone this material.
   */
  clone(): this {
    const cloned = new LineMaterial({
      opacity: this.uniforms.uOpacity.value,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      absorption: this.uniforms.uAbsorption.value,
      hasElementAlpha: (this.uniforms.uHasElementAlpha.value as number) === 1,
      blendingMode: this.userData.blendingMode ?? 'additive',
      transparent: this.transparent,
      depthTest: this.userData.depthTest ?? true,
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
    });

    // Copy blend equation settings for custom blending (max mode)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    // Preserve the line data texture binding (per-node — the clone
    // serves the same node).
    cloned.uniforms.uLineTex.value = this.uniforms.uLineTex.value;
    // Preserve orthographic state, near-plane / max-pixel-width clamp,
    // and the precomputed pixel-width scales.
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    cloned.uniforms.uPerspectiveLineScale.value = this.uniforms.uPerspectiveLineScale.value;
    cloned.uniforms.uLineJoin.value = this.uniforms.uLineJoin.value;
    cloned.uniforms.uOrthoLineScale.value = this.uniforms.uOrthoLineScale.value;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    return cloned as this;
  }

  // Dispose is inherited from THREE.ShaderMaterial. The MaterialManager
  // subscribes to the synchronous `dispose` event THREE fires from
  // super.dispose(), so registry cleanup happens automatically without
  // this file needing to import the manager (which would create a cycle).

  /**
   * Apply a Luxar blending mode to this material in-place.
   *
   * Draws the complete THREE state from the shared
   * `getCompleteBlendingState` — the same source of truth the Point
   * and GSplat wrappers and the TSL twin use — so creation-time wiring,
   * runtime UI transitions, and both backends can never disagree on
   * blend factors, depth state, or transparency.
   *
   * Sets the `LUXAR_MAX_RGB_CONTRIBUTION` shader define for `max` mode
   * so the fragment premultiplies RGB by intensity*opacity (required
   * because MaxEquation+OneFactor doesn't multiply by alpha at
   * composite time). After this returns, `userData.blendingMode`
   * reflects the live mode so subsequent `clone()` calls preserve it.
   */
  applyBlendingMode(mode: BlendingMode): void {
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = getCompleteBlendingState(mode, opacity);

    // Defensive: THREE may leave defines undefined when none were
    // passed at construction.
    if (!this.defines) this.defines = {};

    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const wantsContrib = state.shaderOutputMode === 'rgb-contribution';
    const hasContrib = 'LUXAR_MAX_RGB_CONTRIBUTION' in this.defines;
    // Volumetric = its own output branch (emission–absorption): every
    // non-volumetric transition must clear the define (a
    // volumetric→normal switch must not strand it). Mirrors the
    // point/gsplat wrappers' define lifecycle.
    const wantsVolumetric = isVolumetricMode(mode);
    const hasVolumetric = 'LUXAR_VOLUMETRIC' in this.defines;
    const stateChanged = applyBlendingStateToMaterial(this, state);
    let definesChanged = false;
    if (wantsContrib && !hasContrib) {
      this.defines.LUXAR_MAX_RGB_CONTRIBUTION = '';
      definesChanged = true;
    } else if (!wantsContrib && hasContrib) {
      delete this.defines.LUXAR_MAX_RGB_CONTRIBUTION;
      definesChanged = true;
    }
    if (wantsVolumetric && !hasVolumetric) {
      this.defines.LUXAR_VOLUMETRIC = '';
      definesChanged = true;
    } else if (!wantsVolumetric && hasVolumetric) {
      delete this.defines.LUXAR_VOLUMETRIC;
      definesChanged = true;
    }
    this.userData.blendingMode = mode;
    this.userData.depthTest = state.depthTest;

    if (definesChanged) {
      // Defines changed → shader must recompile.
      this.needsUpdate = true;
    } else if (previousMode !== mode && stateChanged) {
      // Mode changed but no shader recompile required.
      // Mark needsUpdate to refresh blend state on the GPU.
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
    // Midpoint identity for degenerate ranges — see scalar-range.ts.
    const { scalarMin, scalarScale } = computeScalarRangeUniforms(min, max);
    if (this.uniforms.uScalarMin) this.uniforms.uScalarMin.value = scalarMin;
    if (this.uniforms.uScalarScale) {
      this.uniforms.uScalarScale.value = scalarScale;
    }
  }
}
