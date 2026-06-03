/**
 * TSL / NodeMaterial counterpart to `PointMaterial`.
 *
 * Mirrors the GLSL `PointMaterial` wrapper class one-for-one — same
 * constructor signature (`PointMaterialConfig`), same update methods
 * (`updateOpacity`, `updateGamma`, …), same `applyBlendingMode` +
 * `clone` semantics, same `CameraAwareMaterial` /
 * `ColormapAwareMaterial` interfaces. The host code in
 * `MaterialManager.getPointMaterial` picks one or the other based on
 * `caps.apiSurface`, so callers (NodeFactory, LayersPanel, etc.) never see
 * the divergence.
 *
 * Backend mechanics:
 *
 *   - Holds a `uniforms` table of `THREE.IUniform` records, exactly
 *     like the GLSL wrapper. The `pointWebGPUFactory` binds each TSL
 *     uniform node to its matching IUniform via `.onUpdate(() =>
 *     iuniform.value)`, so every `this.uniforms.X.value = …` write
 *     from an update method propagates to the shader on the next
 *     render frame.
 *   - `defines` carries the same `USE_COLORMAP` shader flag the GLSL
 *     wrapper toggles. Flipping it on the TSL side requires a graph
 *     rebuild (the colormap branch in the factory uses a JS-side
 *     `if`), which is what `updateColormapTexture` triggers via
 *     `rebuildGraph()` whenever the on/off state changes.
 *
 * @module rendering/materials/point/material-tsl
 */

import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { pointWebGPUFactory } from './shader-tsl';
import type { PointMaterialConfig } from './material-glsl';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne } from '../_shared/uniform-helpers';
import { computePointSizeFactor, computeMaxPointSize } from '../_shared/camera-uniforms';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  type CompleteBlendingState,
} from '../../blending-state';
import type { BlendingMode } from '../../material-manager';

/**
 * Points material rendered via TSL / NodeMaterial.
 *
 * Designed to satisfy the same surface as `PointMaterial` so the
 * dispatch in `MaterialManager.getPointMaterial` is a drop-in. The
 * class extends `NodeMaterial` (the WebGPU/WebGL-2-fallback-aware
 * material type) instead of `THREE.ShaderMaterial`.
 */
export class PointTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /** Public uniforms table, same shape as `PointMaterial.uniforms`. */
  uniforms: Record<string, THREE.IUniform>;

  constructor(materialConfig: PointMaterialConfig = {}) {
    super();

    const gammaValue = clampGamma(materialConfig.gamma);
    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    this.uniforms = {
      opacity: { value: materialConfig.opacity ?? 1.0 },
      invGamma: { value: 1.0 / gammaValue },
      uIntensity: { value: materialConfig.intensity ?? 1.0 },
      uOffset: { value: materialConfig.offset ?? 0.0 },

      pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
      maxPointSize: { value: defaultResolutionY * 0.5 },

      radiusScale: { value: materialConfig.radiusScale ?? 1.0 },
      sharpnessScale: { value: materialConfig.sharpnessScale ?? 1.0 },

      uIsOrtho: { value: 0 },
      uResolution: { value: new THREE.Vector2(1920, defaultResolutionY) },

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
    };

    this.defines = materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {};
    // LUXAR_GAMMA_ONE mirrors the GLSL define; it drives the `gammaOne`
    // factory flag in `rebuildGraph` so the gamma pow() is skipped at
    // gamma == 1.0. Toggled by `updateGamma`.
    if (isGammaOne(gammaValue)) this.defines.LUXAR_GAMMA_ONE = '';
    this.toneMapped = false;

    // userData mirrors the GLSL wrapper so `clone()` / `applyBlendingMode`
    // share the same fields.
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? true;
    this.userData.scalarRange = materialConfig.scalarRange;

    // Stamp the requested Luxar blending mode on userData BEFORE
    // `rebuildGraph` so the factory reads the correct mode through
    // `userData.blendingMode` on its first build. Without this the
    // factory defaults to `'additive'` and a follow-up
    // `applyBlendingMode('max')` call would have to rebuild the graph
    // a second time (the max-mode path toggles
    // `LUXAR_MAX_RGB_CONTRIBUTION`). Mirrors the Line/GSplat
    // constructor pattern — three-geometry symmetry.
    this.userData.blendingMode = materialConfig.blendingMode ?? 'additive';

    // For max mode, the shader needs the LUXAR_MAX_RGB_CONTRIBUTION
    // define from the very first compile. Apply it before rebuild
    // so the factory sees the right define set. (Other modes don't
    // touch defines, so they're no-ops here.)
    if (this.userData.blendingMode === 'max') {
      this.defines.LUXAR_MAX_RGB_CONTRIBUTION = '';
    }

    // Build the TSL graph and attach to ourselves. The factory wires
    // every primitive uniform via `.onUpdate(() => iuniform.value)`,
    // so mutating `this.uniforms.X.value` flows through to the GPU.
    this.rebuildGraph();
  }

  /**
   * Re-run the TSL factory and attach the resulting vertexNode /
   * colorNode + blending state to ourselves. Called from the
   * constructor and from `updateColormapTexture` when the
   * colormap state changes (the colormap branch is gated on a
   * JS-side `if`, so the graph itself changes shape; TSL's
   * `texture()` captures the Texture object at factory-call time
   * so a swap also requires a rebuild).
   *
   * Drives `useColormap` from `defines.USE_COLORMAP` — the same
   * source of truth the GLSL wrapper uses. Probing
   * `!!this.uniforms.uColormapTex` would be wrong because
   * `setColormapTexture(null)` keeps the IUniform object around
   * with `.value = null` (mirroring PointMaterial's behaviour); the
   * uniform existence outlives the colormap-enabled state.
   */
  private rebuildGraph(): void {
    pointWebGPUFactory(
      this.uniforms,
      {
        useColormap: !!this.defines && 'USE_COLORMAP' in this.defines,
        gammaOne: !!this.defines && 'LUXAR_GAMMA_ONE' in this.defines,
        blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'additive',
      },
      this
    );
    this.needsUpdate = true;
  }

  /**
   * CameraAwareMaterial. Same body shape as `PointMaterial`: mutate
   * `this.uniforms.X.value`; the TSL uniform nodes track these by
   * reference via the factory's `onUpdate` bindings.
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    _nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    this.uniforms.pointSizeFactor.value = computePointSizeFactor(fov, resolution.y, isOrtho);
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
  }

  updateOpacity(opacity: number): void {
    this.uniforms.opacity.value = opacity;
  }

  updateGamma(gamma: number): void {
    const safeGamma = clampGamma(gamma);
    this.userData.gamma = safeGamma;
    this.uniforms.invGamma.value = 1.0 / safeGamma;

    // Toggle `LUXAR_GAMMA_ONE` define when crossing the threshold and
    // rebuild the TSL graph so the factory picks the new fast-path
    // branch (mirrors the Line material).
    if (!this.defines) this.defines = {};
    const wantGammaOne = isGammaOne(safeGamma);
    const hadGammaOne = 'LUXAR_GAMMA_ONE' in this.defines;
    if (wantGammaOne && !hadGammaOne) {
      this.defines.LUXAR_GAMMA_ONE = '';
      this.rebuildGraph();
    } else if (!wantGammaOne && hadGammaOne) {
      delete this.defines.LUXAR_GAMMA_ONE;
      this.rebuildGraph();
    }
  }

  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
  }

  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
  }

  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }

  updateSharpnessScale(scale: number): void {
    this.uniforms.sharpnessScale.value = scale;
  }

  /**
   * Toggle the colormap branch. Rebuild the graph whenever the
   * colormap state actually changes — either an on/off flip OR a
   * texture-identity swap.
   *
   * TSL's `texture(value, …)` captures the Texture object passed to
   * it at factory-call time; subsequent mutations to
   * `iuniform.value` don't re-route the underlying TextureNode. So a
   * swap (old non-null → new non-null) needs a graph rebuild too,
   * not just the enable/disable transitions. The GLSL wrapper gets
   * away with a single `material.uniforms.uColormapTex.value =
   * texture` write because `ShaderMaterial` reads the IUniform by
   * reference at render time.
   */
  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const oldTexture =
      (this.uniforms.uColormapTex?.value as THREE.Texture | null | undefined) ?? null;
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
    const textureChanged = oldTexture !== texture;
    if (wasEnabled !== nowEnabled || textureChanged) {
      this.rebuildGraph();
    }
  }

  applyBlendingMode(mode: BlendingMode): void {
    const opacity = (this.uniforms.opacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = getCompleteBlendingState(mode, opacity);

    if (!this.defines) {
      this.defines = {};
    }

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

    // The TSL factory reads `blendingMode` to decide the shader-output
    // shape (`useMaxRGBContribution` derives from `mode === 'max'`).
    // Flipping max ↔ non-max changes the graph; rebuild so the
    // colorNode reflects the new branch.
    if (definesChanged) {
      this.rebuildGraph();
    } else if (previousMode !== mode && stateChanged) {
      this.needsUpdate = true;
    }
  }

  updateScalarRange(min: number, max: number): void {
    applyScalarRangeToMaterial(this, min, max);
  }

  clone(): this {
    const cloned = new PointTSLMaterial({
      opacity: this.uniforms.opacity.value,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'additive',
      depthWrite: this.depthWrite,
      depthTest: this.userData.depthTest ?? true,
      transparent: this.transparent,
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
    });

    // Mirrors Line/GSplat clone: the constructor's
    // `applyBlendingMode` (driven by `blendingMode`) sets the TSL
    // graph + framebuffer state correctly. Copy custom blend
    // factors verbatim for max mode so any post-construction
    // overrides on the source carry through.
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    // Copy current uniform values
    cloned.uniforms.pointSizeFactor.value = this.uniforms.pointSizeFactor.value;
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.invGamma.value = this.uniforms.invGamma.value;
    cloned.uniforms.radiusScale.value = this.uniforms.radiusScale.value;
    cloned.uniforms.sharpnessScale.value = this.uniforms.sharpnessScale.value;

    return cloned as this;
  }

  // ColormapAwareMaterial — `material-colormap-helpers.ts` delegates here.
  setColormapTexture(texture: THREE.DataTexture | null): void {
    if (!this.defines) this.defines = {};
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
