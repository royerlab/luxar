/**
 * Point Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for physically accurate points rendering.
 * Features world-space sizing, HDR colors, and per-point sharpness control.
 */

import * as THREE from 'three';
import { POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER } from './shader-glsl';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne } from '../_shared/uniform-helpers';
import { computePointSizeFactor, computeMaxPointSize } from '../_shared/camera-uniforms';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  getCompleteBlendingState,
  applyBlendingStateToMaterial,
  effectiveGeometryMode,
  type CompleteBlendingState,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import { computeScalarRangeUniforms, scalarRangeUniformEntries } from '../_shared/scalar-range';

/**
 * Configuration for point material creation
 */
export interface PointMaterialConfig {
  opacity?: number;
  gamma?: number;
  intensity?: number; // Linear color multiplier (gain), default 1.0
  offset?: number; // Additive brightness shift (black level), default 0.0
  /**
   * Luxar blending mode. Same shape as `LineMaterialConfig.blendingMode`
   * and `GSplatMaterialConfig.blendingMode` (three-geometry symmetry —
   * Points/Lines/GSplats expose the same surface). Default `'additive'`.
   * Drives both the THREE blending state and the shader's
   * `LUXAR_MAX_RGB_CONTRIBUTION` define via `applyBlendingMode`.
   */
  blendingMode?: BlendingMode;
  depthTest?: boolean; // Whether to test against depth buffer (default true)
  transparent?: boolean; // Whether material is transparent (default true)
  radiusScale?: number; // Scale factor for radius normalization (e.g., 1/255 for uint8)
  colormapTexture?: THREE.DataTexture; // Colormap LUT texture (256x1 RGB)
  scalarRange?: [number, number]; // Scalar data range [min, max] for normalization
}

/**
 * Points material with physically accurate world-space sizing.
 * Extends THREE.ShaderMaterial to provide specialized point rendering.
 */
export class PointMaterial
  extends THREE.ShaderMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /**
   * Create a new PointMaterial with the specified configuration.
   * Mirrors the constructor pattern in `LineMaterial` and
   * `GSplatMaterial`: derive an initial THREE blending state from
   * the Luxar `blendingMode`, hand it to `super({...})`, then call
   * `applyBlendingMode` to apply the canonical state (defines,
   * depthTest, custom blend factors). One code path drives both
   * creation and runtime mode changes from the layers panel.
   */
  constructor(materialConfig: PointMaterialConfig = {}) {
    const blendingMode: BlendingMode = materialConfig.blendingMode ?? 'additive';
    const isOpaque = blendingMode === 'opaque';
    const isAdditive = blendingMode === 'additive';
    const gammaValue = clampGamma(materialConfig.gamma);
    // Default values for initial computation
    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    // Initial THREE.Blending for `super`. `applyBlendingMode` below
    // overrides this with the canonical mode-derived state — the
    // value here only matters during the brief window between
    // `super({...})` returning and `applyBlendingMode` running.
    // Volumetric maps to its additive (κ=0) fallback for points — the
    // policy lives in effectiveGeometryMode (blending-state.ts).
    const initialMode = effectiveGeometryMode(blendingMode, 'point');
    let initialBlending: THREE.Blending;
    if (isOpaque || initialMode === 'normal') {
      initialBlending = THREE.NormalBlending;
    } else if (initialMode === 'additive' || initialMode === 'luminous') {
      initialBlending = THREE.AdditiveBlending;
    } else {
      initialBlending = THREE.CustomBlending; // max
    }

    super({
      uniforms: {
        // Point data texture (RGBA32F, 3 texels/point) — starts on the
        // shared placeholder; the commit's material sync rebinds the
        // acquired pool entry's texture via `updatePointTexture`.
        uPointTex: { value: getPlaceholderElementTexture() },

        // Color uniforms
        opacity: { value: materialConfig.opacity ?? 1.0 },
        invGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },

        // OPTIMIZED camera uniforms - pre-computed for shader performance
        // pointSizeFactor = 2.0 * resolution.y / tan(fov/2)
        pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
        maxPointSize: { value: defaultResolutionY * 0.5 }, // resolution.y * 0.5

        // Radius scaling for dtype normalization
        radiusScale: { value: materialConfig.radiusScale ?? 1.0 }, // Default 1.0 (no scaling)

        // Projection mode
        uIsOrtho: { value: 0 }, // 0 = perspective, 1 = orthographic
        uNearCull: { value: 0.1 }, // near-fade start (world units; scene-bounds scaled)

        // Physical framebuffer size in pixels (used by the
        // instanced-quad vertex shader to convert pixel offsets to
        // NDC. Defaults match the camera-uniform defaults; overridden
        // by `updateCameraParams`.
        uResolution: { value: new THREE.Vector2(1920, defaultResolutionY) },

        // Colormap uniforms (only active when USE_COLORMAP define is set)
        ...(materialConfig.colormapTexture
          ? {
              uColormapTex: { value: materialConfig.colormapTexture },
              // Midpoint identity for degenerate ranges — see scalar-range.ts.
              ...scalarRangeUniformEntries(materialConfig.scalarRange),
            }
          : {}),
      },

      // Shader source
      vertexShader: POINT_VERTEX_SHADER,
      fragmentShader: POINT_FRAGMENT_SHADER,

      // Preprocessor defines — USE_COLORMAP enables scalar attribute + LUT
      // lookup; LUXAR_GAMMA_ONE skips the per-fragment gamma pow() when
      // gamma == 1.0 (toggled by `updateGamma`).
      defines: {
        ...(materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {}),
        ...(isGammaOne(gammaValue) ? { LUXAR_GAMMA_ONE: '' } : {}),
      },

      // GLSL ES 3.0 for consistency with other materials
      glslVersion: THREE.GLSL3,

      // Material properties.
      // vertexColors=false because point colors come from the point
      // texture (texel1.rgb), not a vertex attribute. Three's
      // auto-injected `color` attribute is for the per-vertex `position`
      // attribute it assumes, which this instanced mesh layout does not use.
      vertexColors: false,
      transparent: materialConfig.transparent ?? !isOpaque,
      depthWrite: false, // applyBlendingMode below overwrites immediately
      depthTest: materialConfig.depthTest ?? !isAdditive,
      toneMapped: false, // HDR values pass through to post-processing
      blending: initialBlending,
    });

    // Apply mode-specific blending state via the canonical method —
    // same path used by live mode updates from the layers panel.
    // Sets `userData.blendingMode`, the `LUXAR_MAX_RGB_CONTRIBUTION`
    // define for max mode, and the THREE custom-blend factors.
    this.applyBlendingMode(blendingMode);

    // Honor explicit overrides from config after mode-derived defaults.
    if (materialConfig.transparent !== undefined) {
      this.transparent = materialConfig.transparent;
    }
    if (materialConfig.depthTest !== undefined) {
      this.depthTest = materialConfig.depthTest;
    }

    // gamma + scalarRange in userData for clone(). blendingMode is
    // already set by applyBlendingMode; depthTest stamped here so
    // the explicit-override path above carries into clone state.
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? !isAdditive;
    this.userData.scalarRange = materialConfig.scalarRange;
  }

  /**
   * Update camera parameters for world-space point sizing
   * Pre-computes pointSizeFactor and maxPointSize for shader performance
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined) this.uniforms.uNearCull.value = nearCull;
    this.uniforms.pointSizeFactor.value = computePointSizeFactor(fov, resolution.y, isOrtho);
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
    // The instanced-quad vertex shader needs the framebuffer size to
    // convert pixel offsets to NDC. This is the physical pixel size
    // (drawing-buffer size), matched to what the SceneManager passes.
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
  }

  /**
   * Update opacity
   */
  updateOpacity(opacity: number): void {
    this.uniforms.opacity.value = opacity;
  }

  /** Current opacity multiplier (the LOD cross-fade snapshots this as its fade base). */
  getOpacity(): number {
    return this.uniforms.opacity.value as number;
  }

  /**
   * Update gamma correction
   * Only invGamma is used in shader; gamma value stored in userData for clone()
   *
   * When `gamma` crosses the 1.0 threshold (with epsilon), toggle the
   * `LUXAR_GAMMA_ONE` define so the fragment shader's pow() fast path is
   * recompiled in/out (mirrors the Line material).
   */
  updateGamma(gamma: number): void {
    const safeGamma = clampGamma(gamma);
    this.userData.gamma = safeGamma; // Store for clone() method
    this.uniforms.invGamma.value = 1.0 / safeGamma;

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
   * Update radius scale for dtype normalization
   * Use 1/255 for uint8 radii, 1.0 for float radii
   */
  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }

  /**
   * Rebind the point data texture (pool acquire may hand the node a
   * different geometry+texture pair on growth or best-fit reuse).
   * Plain uniform update — no shader recompilation involved. Mirrors
   * `GSplatMaterial.updateSplatTexture`; `null` falls back to the
   * shared placeholder so the sampler is never unbound.
   */
  updatePointTexture(texture: THREE.DataTexture | null): void {
    this.uniforms.uPointTex.value = texture ?? getPlaceholderElementTexture();
  }

  /** The currently bound point data texture. */
  getPointTexture(): THREE.DataTexture | null {
    return (this.uniforms.uPointTex.value as THREE.DataTexture | null) ?? null;
  }

  /**
   * Update the colormap texture and enable/disable colormap mode.
   *
   * Under the instanced-quad rendering path, vertexColors is always
   * false — the shader reads the color (USE_COLORMAP off) or the scalar
   * (USE_COLORMAP on) from the point texture's texels. Only the
   * USE_COLORMAP define flips here, triggering a recompile.
   */
  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
    if (wasEnabled !== nowEnabled) {
      this.needsUpdate = true; // Triggers shader recompilation
    }
  }

  /**
   * Apply a Luxar blending mode to this material in-place.
   *
   * Single source of truth for both creation-time wiring (called by
   * `MaterialManager.getPointMaterial`) and runtime UI transitions
   * (called by `LayersPanel.applyBlendingStateToMaterial`). Without
   * this method, the LayersPanel generic path forgot to set
   * `blendSrc`/`blendDst`, so a runtime switch additive→max stranded
   * SrcAlpha factors and produced inconsistent visuals.
   *
   * Sets the `LUXAR_MAX_RGB_CONTRIBUTION` shader define for `max` mode
   * so the fragment shader premultiplies RGB by falloff/opacity. This
   * is required because MaxEquation+OneFactor doesn't multiply by
   * alpha at composite time. Toggling this define triggers a shader
   * recompilation; that's intentional and only happens on actual mode
   * transitions (idempotent — see `userData.blendingMode` early exit).
   */
  applyBlendingMode(mode: BlendingMode): void {
    const opacity = (this.uniforms.opacity?.value as number | undefined) ?? 1.0;
    // Phase-1 volumetric fallback — the policy lives in
    // effectiveGeometryMode (blending-state.ts); userData keeps the
    // REQUESTED mode so stored scenes upgrade automatically.
    const effectiveMode: BlendingMode = effectiveGeometryMode(mode, 'point');
    const state: CompleteBlendingState = getCompleteBlendingState(effectiveMode, opacity);

    // Defensive: THREE may leave `defines` undefined when none were
    // passed at construction. We rely on it as our source of truth for
    // the LUXAR_MAX_RGB_CONTRIBUTION shader define.
    if (!this.defines) {
      this.defines = {};
    }

    // Idempotent fast path: no need to re-apply identical state.
    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const wantsContrib = state.shaderOutputMode === 'rgb-contribution';
    const hasContrib = 'LUXAR_MAX_RGB_CONTRIBUTION' in this.defines;
    const stateChanged = applyBlendingStateToMaterial(this, state);
    let definesChanged = false;
    if (wantsContrib && !hasContrib) {
      this.defines.LUXAR_MAX_RGB_CONTRIBUTION = '';
      definesChanged = true;
    } else if (!wantsContrib && hasContrib) {
      delete this.defines.LUXAR_MAX_RGB_CONTRIBUTION;
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

  /**
   * Set the scalar data range for colormap normalization.
   */
  updateScalarRange(min: number, max: number): void {
    applyScalarRangeToMaterial(this, min, max);
  }

  /**
   * Clone this material with optional config overrides
   * Override base class clone to return PointMaterial type
   */
  clone(): this {
    const cloned = new PointMaterial({
      opacity: this.uniforms.opacity.value,
      gamma: this.userData.gamma ?? 1.0, // gamma stored in userData, not uniforms
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'additive',
      depthTest: this.userData.depthTest ?? true, // depthTest stored in userData
      transparent: this.transparent,
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
    });

    // Copy custom blend factors for max mode (the constructor's
    // applyBlendingMode already set them, but copy verbatim so any
    // post-construction overrides on the source carry through).
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    // Copy current uniform values. The point-texture binding must ride
    // along (mirrors GSplatMaterial.clone copying uSplatTex): the layers
    // panel clones on first interaction, and a clone left on the
    // placeholder would render nothing.
    cloned.uniforms.uPointTex.value = this.uniforms.uPointTex.value;
    cloned.uniforms.pointSizeFactor.value = this.uniforms.pointSizeFactor.value;
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.invGamma.value = this.uniforms.invGamma.value;
    cloned.uniforms.radiusScale.value = this.uniforms.radiusScale.value;
    // Camera-state uniforms must ride along too (mirrors
    // LineMaterial.clone, the reference implementation): a clone taken
    // in ortho mode otherwise renders the perspective branch with stale
    // resolution/nearCull until the next global updateCameraParams
    // broadcast reaches it.
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );

    return cloned as this;
  }

  // -------------------------------------------------------------
  // ColormapAwareMaterial
  //
  // These setters concentrate the colormap-uniform writes inside
  // the material that owns the uniforms. External code reaches
  // colormap state through `material-colormap-helpers.ts`, which
  // delegates here. The WebGPU port rewrites the setter bodies
  // (node-uniform reassignment) without touching call sites.
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

  // Note: `dispose()` is inherited from THREE.ShaderMaterial.
  // MaterialManager subscribes to the synchronous `dispose` event the
  // base class fires, so the material is removed from the registry +
  // cache automatically. No explicit unregister callback needed here,
  // which keeps this file out of the manager's import graph.
}
