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
 * **Uniform plumbing.** This class owns one persistent `UniformNode`
 * per shader input via the `tslNodes` table. The public `uniforms`
 * record exposes each node as an `IUniform`-shaped getter/setter
 * proxy (see `proxyIUniform` in `tsl-helpers.ts`), so mutations to
 * `material.uniforms.X.value` land directly on `node.value` — no
 * per-render `.onUpdate('render')` callback bridge. Matches the
 * pattern already in use by `LineTSLMaterial`, `GSplatTSLMaterial`,
 * and `PointPickingTSLMaterial`.
 *
 * `defines` carries the same `USE_COLORMAP` shader flag the GLSL
 * wrapper toggles. Flipping it on the TSL side requires a graph
 * rebuild (the colormap branch in the factory uses a JS-side `if`),
 * which is what `updateColormapTexture` triggers via `rebuildGraph()`
 * whenever the on/off state changes.
 *
 * @module rendering/materials/point/material-tsl
 */

import * as THREE from 'three';
import { uniform, texture } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { pointWebGPUFactory, type PointTSLNodes } from './shader-tsl';
import type { PointMaterialConfig } from './material-glsl';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne } from '../_shared/uniform-helpers';
import { computePointSizeFactor, computeMaxPointSize } from '../_shared/camera-uniforms';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  isVolumetricMode,
  type CompleteBlendingState,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import { proxyIUniform, type TSLNode } from '../_shared/tsl-helpers';
import { computeScalarRangeUniforms } from '../_shared/scalar-range';

/**
 * Persistent TSL node table owned by the wrapper. Colormap nodes are
 * (re)created lazily inside `rebuildGraph()` when colormap mode
 * toggles — the `PointTSLNodes` factory contract treats them as
 * optional. Keys match the public `uniforms` record (the un-prefixed
 * names mirror the GLSL `PointMaterial`).
 */
interface PointMaterialTSLNodeTable {
  uPointTex: TSLNode;
  pointSizeFactor: TSLNode;
  maxPointSize: TSLNode;
  radiusScale: TSLNode;
  uIsOrtho: TSLNode;
  uNearCull: TSLNode;
  uResolution: TSLNode;
  opacity: TSLNode;
  invGamma: TSLNode;
  uIntensity: TSLNode;
  uOffset: TSLNode;
  uAbsorption: TSLNode;
  uHasElementAlpha: TSLNode;
  uColormapTex?: TSLNode;
  uScalarMin?: TSLNode;
  uScalarScale?: TSLNode;
}

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

  private tslNodes: PointMaterialTSLNodeTable;

  /**
   * Explicit `depthTest` / `transparent` overrides from the constructor
   * config. Every `rebuildGraph()` re-applies the factory tail's
   * MODE-DERIVED blending state (depthTest/transparent included), so an
   * override honored only once in the constructor tail would silently
   * revert on the first later rebuild (e.g. the guaranteed
   * placeholder→real `updatePointTexture` rebuild at first commit). The
   * GLSL twin never rebuilds, so its constructor-tail overrides stick;
   * persisting them here and re-applying at the end of `rebuildGraph`
   * keeps the two backends contract-identical. An explicit later
   * `applyBlendingMode()` call CLEARS both (a runtime mode switch takes
   * full ownership of the blending state — matching the GLSL twin,
   * where `applyBlendingStateToMaterial` overwrites both fields
   * unconditionally).
   */
  private _explicitDepthTest?: boolean;
  private _explicitTransparent?: boolean;

  constructor(materialConfig: PointMaterialConfig = {}) {
    super();

    const gammaValue = clampGamma(materialConfig.gamma);
    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    this.tslNodes = {
      // Point data texture node. Starts on the shared placeholder; the
      // commit's material sync rebinds the acquired pool entry's
      // texture via `updatePointTexture` (node identity change ->
      // graph rebuild, same lifecycle as the colormap texture).
      uPointTex: texture(getPlaceholderElementTexture()),
      opacity: uniform(materialConfig.opacity ?? 1.0),
      invGamma: uniform(1.0 / gammaValue),
      uIntensity: uniform(materialConfig.intensity ?? 1.0),
      uOffset: uniform(materialConfig.offset ?? 0.0),
      // Volumetric (emission–absorption) uniforms — read only when the
      // graph was built in volumetric mode; plain runtime uniforms
      // otherwise (no rebuild on value changes).
      uAbsorption: uniform(materialConfig.absorption ?? 1.0),
      uHasElementAlpha: uniform(0),
      pointSizeFactor: uniform((2.0 * defaultResolutionY) / defaultTanHalfFov),
      maxPointSize: uniform(defaultResolutionY * 0.5),
      radiusScale: uniform(materialConfig.radiusScale ?? 1.0),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.1),
      uResolution: uniform(new THREE.Vector2(1920, defaultResolutionY)),
    };

    // Build the public IUniform-proxy table. Mutations to
    // `material.uniforms.X.value` land directly on the TSL node's
    // value via `proxyIUniform`, so the GPU sees the new value on the
    // next frame without any `.onUpdate('render')` callback.
    this.uniforms = {
      // WARNING: a direct `uniforms.uPointTex.value = tex` write does
      // NOT rebind the sampled texture — TSL `texture()` nodes capture
      // the Texture at build time. `updatePointTexture()` is the only
      // rebind chokepoint (fresh node + graph rebuild).
      uPointTex: proxyIUniform(this.tslNodes.uPointTex),
      opacity: proxyIUniform(this.tslNodes.opacity),
      invGamma: proxyIUniform(this.tslNodes.invGamma),
      uIntensity: proxyIUniform(this.tslNodes.uIntensity),
      uOffset: proxyIUniform(this.tslNodes.uOffset),
      uAbsorption: proxyIUniform(this.tslNodes.uAbsorption),
      uHasElementAlpha: proxyIUniform(this.tslNodes.uHasElementAlpha),
      pointSizeFactor: proxyIUniform(this.tslNodes.pointSizeFactor),
      maxPointSize: proxyIUniform(this.tslNodes.maxPointSize),
      radiusScale: proxyIUniform(this.tslNodes.radiusScale),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
    };

    // Colormap uniforms are added lazily — see `rebuildColormapNodes`.
    if (materialConfig.colormapTexture) {
      this.uniforms.uColormapTex = { value: materialConfig.colormapTexture };
      // Midpoint identity for degenerate ranges — see scalar-range.ts.
      const sr = computeScalarRangeUniforms(
        materialConfig.scalarRange?.[0] ?? 0.0,
        materialConfig.scalarRange?.[1] ?? 1.0
      );
      this.uniforms.uScalarMin = { value: sr.scalarMin };
      this.uniforms.uScalarScale = { value: sr.scalarScale };
    }

    this.defines = materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {};
    // LUXAR_GAMMA_ONE mirrors the GLSL define; it drives the `gammaOne`
    // factory flag in `rebuildGraph` so the gamma pow() is skipped at
    // gamma == 1.0. Toggled by `updateGamma`.
    if (isGammaOne(gammaValue)) this.defines.LUXAR_GAMMA_ONE = '';
    this.toneMapped = false;

    // userData mirrors the GLSL wrapper so `clone()` / `applyBlendingMode`
    // share the same fields. depthTest is stamped after `rebuildGraph`
    // below, from the mode-derived state the factory tail applies.
    this.userData.gamma = gammaValue;
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
    // so the factory sees the right define set. Volumetric mirrors
    // this with LUXAR_VOLUMETRIC (the factory derives the output
    // branch from `blendingMode`; the define is the rebuild-boundary
    // tracker `applyBlendingMode` keys on). (Other modes don't touch
    // defines, so they're no-ops here.)
    if (this.userData.blendingMode === 'max') {
      this.defines.LUXAR_MAX_RGB_CONTRIBUTION = '';
    }
    if (isVolumetricMode(this.userData.blendingMode as BlendingMode)) {
      this.defines.LUXAR_VOLUMETRIC = '';
    }

    // Capture explicit overrides BEFORE the first rebuild —
    // `rebuildGraph`'s tail re-applies them over the factory's
    // mode-derived blending state on EVERY rebuild (see the
    // `_explicitDepthTest` field doc; the GLSL twin applies them once
    // in its constructor tail and never rebuilds).
    this._explicitTransparent = materialConfig.transparent;
    this._explicitDepthTest = materialConfig.depthTest;

    // Build the TSL graph and attach to ourselves. The factory binds
    // directly to the wrapper-owned `tslNodes`, so mutating
    // `this.uniforms.X.value` (via the proxies) flows through to the
    // GPU without per-frame callbacks.
    this.rebuildGraph();

    // Stamp the depthTest the rebuild just settled on — mode-derived
    // from the factory tail, or the explicit override re-applied over
    // it (GLSL twin: applyBlendingMode stamps userData.depthTest) — so
    // clone() round-trips the real state.
    this.userData.depthTest = this.depthTest;
  }

  /**
   * Refresh the colormap TSL nodes so they bind to the current
   * `this.uniforms.uColormap*.value`. `TextureNode` is bound to a
   * specific `Texture` instance at construction; a texture swap
   * requires a fresh node, hence this lives in `rebuildGraph()`.
   * Mirrors `LineTSLMaterial.rebuildColormapNodes`.
   */
  private rebuildColormapNodes(useColormap: boolean): void {
    if (useColormap) {
      const tex =
        (this.uniforms.uColormapTex?.value as THREE.Texture | null | undefined) ??
        new THREE.Texture();
      this.tslNodes.uColormapTex = texture(tex);
      this.tslNodes.uScalarMin = uniform((this.uniforms.uScalarMin?.value as number) ?? 0.0);
      this.tslNodes.uScalarScale = uniform((this.uniforms.uScalarScale?.value as number) ?? 1.0);
      // Re-point the IUniform proxies at the new nodes so updates
      // flow through. (For the texture, we keep the plain IUniform
      // because TextureNode value mutations don't propagate without a
      // rebuild — `setColormapTexture` triggers rebuild explicitly.)
      this.uniforms.uScalarMin = proxyIUniform(this.tslNodes.uScalarMin);
      this.uniforms.uScalarScale = proxyIUniform(this.tslNodes.uScalarScale);
      // Restore uColormapTex.value pointer to the texture we just bound.
      this.uniforms.uColormapTex = { value: tex };
    } else {
      this.tslNodes.uColormapTex = undefined;
      this.tslNodes.uScalarMin = undefined;
      this.tslNodes.uScalarScale = undefined;
    }
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
    const useColormap = !!this.defines && 'USE_COLORMAP' in this.defines;
    this.rebuildColormapNodes(useColormap);
    pointWebGPUFactory(
      this.tslNodes as PointTSLNodes,
      {
        useColormap,
        gammaOne: !!this.defines && 'LUXAR_GAMMA_ONE' in this.defines,
        blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'additive',
      },
      this
    );
    // Re-apply the explicit constructor overrides over the factory
    // tail's mode-derived blending state — on EVERY rebuild, not just
    // the constructor's, so a texture/gamma/colormap rebuild can't
    // silently revert them (see the `_explicitDepthTest` field doc).
    if (this._explicitTransparent !== undefined) {
      this.transparent = this._explicitTransparent;
    }
    if (this._explicitDepthTest !== undefined) {
      this.depthTest = this._explicitDepthTest;
      this.userData.depthTest = this._explicitDepthTest;
    }
    this.needsUpdate = true;
  }

  /**
   * CameraAwareMaterial. Same body shape as `PointMaterial`: mutate
   * `this.uniforms.X.value`; the writes land directly on the
   * wrapper-owned TSL uniform nodes via the `proxyIUniform` bridges.
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined && this.uniforms.uNearCull) {
      this.uniforms.uNearCull.value = nearCull;
    }
    this.uniforms.pointSizeFactor.value = computePointSizeFactor(fov, resolution.y, isOrtho);
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
  }

  updateOpacity(opacity: number): void {
    this.uniforms.opacity.value = opacity;
  }

  /** Current opacity multiplier (the LOD cross-fade snapshots this as its fade base). */
  getOpacity(): number {
    return this.uniforms.opacity.value as number;
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

  /**
   * Update the volumetric absorption coefficient κ (composed node
   * attr). Plain runtime uniform — no graph rebuild. Mirrors
   * `GSplatTSLMaterial.updateAbsorption`.
   */
  updateAbsorption(absorption: number): void {
    this.uniforms.uAbsorption.value = absorption;
  }

  /**
   * Flag whether the committed colors carry a real per-point alpha
   * column (RGBA). Deliberately a uniform, not a define — toggling it
   * never rebuilds the graph. Mirrors
   * `GSplatTSLMaterial.updateHasElementAlpha`.
   */
  updateHasElementAlpha(hasAlpha: boolean): void {
    this.uniforms.uHasElementAlpha.value = hasAlpha ? 1 : 0;
  }

  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }

  /**
   * Rebind the point data texture. TSL `texture()` captures the
   * THREE.Texture at factory time, so an identity change needs a
   * fresh node + graph rebuild (exact mirror of
   * `GSplatTSLMaterial.updateSplatTexture` and the colormap texture
   * lifecycle). No-op when the texture is unchanged — the common
   * per-commit case.
   */
  updatePointTexture(tex: THREE.DataTexture | null): void {
    const current = (this.uniforms.uPointTex?.value as THREE.Texture | null | undefined) ?? null;
    const next = tex ?? getPlaceholderElementTexture();
    if (current === next) return;
    this.tslNodes.uPointTex = texture(next);
    this.uniforms.uPointTex = proxyIUniform(this.tslNodes.uPointTex);
    this.rebuildGraph();
  }

  /** The currently bound point data texture. */
  getPointTexture(): THREE.DataTexture | null {
    return (this.uniforms.uPointTex?.value as THREE.DataTexture | null | undefined) ?? null;
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
    // A runtime mode switch takes FULL ownership of the blending state:
    // clear the constructor's explicit depthTest/transparent overrides
    // so the mode-derived state below (and every later rebuild) wins.
    // Matches the GLSL twin, where applyBlendingStateToMaterial
    // overwrites both fields unconditionally on every call.
    this._explicitDepthTest = undefined;
    this._explicitTransparent = undefined;

    const opacity = (this.uniforms.opacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = getCompleteBlendingState(mode, opacity);

    if (!this.defines) {
      this.defines = {};
    }

    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const wantsContrib = state.shaderOutputMode === 'rgb-contribution';
    const hasContrib = 'LUXAR_MAX_RGB_CONTRIBUTION' in this.defines;
    // Volumetric is a BUILD-TIME output branch in the factory: any
    // volumetric crossing must rebuild the graph. The define is the
    // tracker (mirrors max's LUXAR_MAX_RGB_CONTRIBUTION), and every
    // non-volumetric transition clears it — a volumetric→normal switch
    // must not strand the branch.
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

    // The TSL factory reads `blendingMode` to decide the shader-output
    // shape (`useMaxRGBContribution` derives from `mode === 'max'`;
    // the volumetric output branch directly from the mode). Flipping
    // max ↔ non-max or crossing volumetric changes the graph; rebuild
    // so the colorNode reflects the new branch. userData.blendingMode
    // is already the new mode, so the rebuild's factory config picks
    // up the right branch.
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
      absorption: this.uniforms.uAbsorption.value,
      blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'additive',
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

    // Carry the point-texture binding across (mirrors
    // GSplatTSLMaterial.clone): the layers panel clones on first
    // interaction, and a clone left on the placeholder would render
    // nothing.
    const pointTex = this.uniforms.uPointTex?.value as THREE.DataTexture | null | undefined;
    if (pointTex) cloned.updatePointTexture(pointTex);

    // Copy current uniform values
    cloned.uniforms.pointSizeFactor.value = this.uniforms.pointSizeFactor.value;
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.invGamma.value = this.uniforms.invGamma.value;
    cloned.uniforms.radiusScale.value = this.uniforms.radiusScale.value;
    // Camera-state uniforms must ride along too (mirrors
    // LineTSLMaterial.clone, the reference implementation): a clone
    // taken in ortho mode otherwise renders the perspective branch with
    // stale resolution/nearCull until the next global
    // updateCameraParams broadcast reaches it. Unlike lines, uIsOrtho
    // is a RUNTIME uniform in the points TSL graph (no ortho graph
    // variant), so a plain value copy suffices — no rebuild needed.
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    // Commit-written data flag: the clone shares the source's point
    // texture, so it must share its RGBA-alpha presence too.
    cloned.uniforms.uHasElementAlpha.value = this.uniforms.uHasElementAlpha.value;

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
    // Midpoint identity for degenerate ranges — see scalar-range.ts.
    const { scalarMin, scalarScale } = computeScalarRangeUniforms(min, max);
    if (this.uniforms.uScalarMin) this.uniforms.uScalarMin.value = scalarMin;
    if (this.uniforms.uScalarScale) {
      this.uniforms.uScalarScale.value = scalarScale;
    }
  }
}
