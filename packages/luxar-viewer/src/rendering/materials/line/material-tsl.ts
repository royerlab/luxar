/**
 * TSL / NodeMaterial counterpart to `LineMaterial`.
 *
 * Mirrors the GLSL `LineMaterial` wrapper one-for-one — same
 * constructor signature (`LineMaterialConfig`), same update methods,
 * same `applyBlendingMode` + `clone` semantics, same
 * `CameraAwareMaterial` / `ColormapAwareMaterial` interfaces. The
 * host dispatch in `MaterialManager.getLineMaterial` picks one or
 * the other based on `caps.apiSurface`, so call sites never see the
 * divergence.
 *
 * **Uniform plumbing.** This class owns one persistent `UniformNode`
 * per shader input via the `tslNodes` table. The public `uniforms`
 * record exposes each node as an `IUniform`-shaped getter/setter
 * proxy (see `proxyIUniform` in `tsl-helpers.ts`), so mutations to
 * `material.uniforms.uX.value` land directly on `node.value` — no
 * per-render `.onUpdate('render')` callback bridge. Matches the
 * pattern already in use by `LinePickingTSLMaterial`,
 * `GSplatTSLMaterial`, and `PointPickingTSLMaterial`.
 *
 * @module rendering/materials/line/material-tsl
 */

import * as THREE from 'three';
import { uniform, texture } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { lineWebGPUFactory, type LineTSLNodes } from './shader-tsl';
import { isGammaOne, isNoGOG, type LineMaterialConfig } from './material-glsl';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma } from '../_shared/uniform-helpers';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  effectiveGeometryMode,
  type CompleteBlendingState,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import { proxyIUniform, type TSLNode } from '../_shared/tsl-helpers';

/**
 * Persistent TSL node table owned by the wrapper. Colormap nodes are
 * (re)created lazily inside `rebuildGraph()` when colormap mode
 * toggles — the `LineTSLNodes` factory contract treats them as
 * optional.
 */
interface LineMaterialTSLNodeTable {
  uLineTex: TSLNode;
  uResolution: TSLNode;
  uIsOrtho: TSLNode;
  uNearCull: TSLNode;
  uMaxLinePixelWidth: TSLNode;
  uPerspectiveLineScale: TSLNode;
  uOrthoLineScale: TSLNode;
  uOpacity: TSLNode;
  uInvGamma: TSLNode;
  uIntensity: TSLNode;
  uOffset: TSLNode;
  uColormapTex?: TSLNode;
  uScalarMin?: TSLNode;
  uScalarScale?: TSLNode;
}

export class LineTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /** Public uniforms table, same shape as `LineMaterial.uniforms`. */
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: LineMaterialTSLNodeTable;

  /**
   * Explicit `depthTest` / `transparent` overrides from the constructor
   * config. Every `rebuildGraph()` re-applies the factory tail's
   * MODE-DERIVED blending state (depthTest/transparent included), so an
   * override honored only once in the constructor tail would silently
   * revert on the first later rebuild (e.g. the guaranteed
   * placeholder→real `updateLineTexture` rebuild at first commit). The
   * GLSL twin never rebuilds, so its constructor-tail overrides stick;
   * persisting them here and re-applying at the end of `rebuildGraph`
   * keeps the two backends contract-identical. An explicit later
   * `applyBlendingMode()` call CLEARS both (a runtime mode switch takes
   * full ownership of the blending state — matching the GLSL twin,
   * where `applyBlendingStateToMaterial` overwrites both fields
   * unconditionally). Mirrors `PointTSLMaterial`.
   */
  private _explicitDepthTest?: boolean;
  private _explicitTransparent?: boolean;

  constructor(materialConfig: LineMaterialConfig = {}) {
    super();

    const gammaValue = clampGamma(materialConfig.gamma);

    this.tslNodes = {
      // Line data texture node. Starts on the shared placeholder; the
      // commit's material sync rebinds the acquired pool entry's
      // texture via `updateLineTexture` (node identity change ->
      // graph rebuild, same lifecycle as the colormap texture).
      uLineTex: texture(getPlaceholderElementTexture()),
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.05),
      uMaxLinePixelWidth: uniform(540),
      uPerspectiveLineScale: uniform(1.0),
      uOrthoLineScale: uniform(1.0),
      uOpacity: uniform(materialConfig.opacity ?? 1.0),
      uInvGamma: uniform(1.0 / gammaValue),
      uIntensity: uniform(materialConfig.intensity ?? 1.0),
      uOffset: uniform(materialConfig.offset ?? 0.0),
    };

    // Build the public IUniform-proxy table. Mutations to
    // `material.uniforms.X.value` land directly on the TSL node's
    // value via `proxyIUniform`, so the GPU sees the new value on the
    // next frame without any `.onUpdate('render')` callback.
    this.uniforms = {
      // WARNING: a direct `uniforms.uLineTex.value = tex` write does
      // NOT rebind the sampled texture — TSL `texture()` nodes capture
      // the Texture at build time. `updateLineTexture()` is the only
      // rebind chokepoint (fresh node + graph rebuild).
      uLineTex: proxyIUniform(this.tslNodes.uLineTex),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxLinePixelWidth: proxyIUniform(this.tslNodes.uMaxLinePixelWidth),
      uPerspectiveLineScale: proxyIUniform(this.tslNodes.uPerspectiveLineScale),
      uOrthoLineScale: proxyIUniform(this.tslNodes.uOrthoLineScale),
      uOpacity: proxyIUniform(this.tslNodes.uOpacity),
      uInvGamma: proxyIUniform(this.tslNodes.uInvGamma),
      uIntensity: proxyIUniform(this.tslNodes.uIntensity),
      uOffset: proxyIUniform(this.tslNodes.uOffset),
    };

    // Colormap uniforms are added lazily — see `rebuildColormapNodes`.
    if (materialConfig.colormapTexture) {
      this.uniforms.uColormapTex = { value: materialConfig.colormapTexture };
      this.uniforms.uScalarMin = { value: materialConfig.scalarRange?.[0] ?? 0.0 };
      this.uniforms.uScalarScale = {
        value: materialConfig.scalarRange
          ? 1.0 / Math.max(1e-10, materialConfig.scalarRange[1] - materialConfig.scalarRange[0])
          : 1.0,
      };
    }

    // Variant `defines` — same shape as the GLSL wrapper. Reading
    // these in `rebuildGraph()` selects fragment-stage fast paths in
    // the TSL factory (gamma==1 here; more flags arrive in subsequent
    // commits).
    this.defines = {
      ...(materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {}),
      ...(isGammaOne(gammaValue) ? { LUXAR_GAMMA_ONE: '' } : {}),
      ...(isNoGOG(materialConfig.intensity ?? 1.0, materialConfig.offset ?? 0.0)
        ? { LUXAR_NO_GOG: '' }
        : {}),
    };
    this.toneMapped = false;
    this.side = THREE.DoubleSide;
    // Line quads are screen-space billboards, not physically two-sided
    // surfaces. The renderer's two-pass guard
    // (`renderers/common/Renderer.js:3452`) trips only when
    // `transparent && side===DoubleSide && forceSinglePass===false`,
    // so forcing single-pass skips a redundant back-face render pass
    // per line layer.
    this.forceSinglePass = true;

    // depthTest is stamped after `rebuildGraph` below, from the
    // mode-derived state the factory tail applies.
    this.userData.gamma = gammaValue;
    this.userData.scalarRange = materialConfig.scalarRange;

    // Stamp the requested mode on userData BEFORE rebuildGraph so the
    // factory reads the correct value through
    // `userData.blendingMode` — otherwise the factory defaults to
    // 'additive', wires `premultiplyRGB=false`, and `max` mode
    // rendering is wrong. Mirrors the GLSL `LineMaterial`
    // constructor body where `this.applyBlendingMode(blendingMode)`
    // runs after `super()`.
    this.userData.blendingMode = materialConfig.blendingMode ?? 'additive';

    // Capture explicit overrides BEFORE the first rebuild —
    // `rebuildGraph`'s tail re-applies them over the factory's
    // mode-derived blending state on EVERY rebuild (see the
    // `_explicitDepthTest` field doc; the GLSL twin applies them once
    // in its constructor tail and never rebuilds).
    this._explicitTransparent = materialConfig.transparent;
    this._explicitDepthTest = materialConfig.depthTest;

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
   * Re-run the TSL factory and attach the resulting nodes + blending
   * state to ourselves. `useColormap` reads `defines.USE_COLORMAP`,
   * not the IUniform presence (mirrors GSplatTSLMaterial /
   * PointTSLMaterial).
   */
  private rebuildGraph(): void {
    const useColormap = !!this.defines && 'USE_COLORMAP' in this.defines;
    const gammaOne = !!this.defines && 'LUXAR_GAMMA_ONE' in this.defines;
    const noGOG = !!this.defines && 'LUXAR_NO_GOG' in this.defines;
    // Camera mode lives on the uniform itself, not in defines: the
    // factory reads `tslNodes.uIsOrtho.value` at build time so a fresh
    // rebuild after `updateCameraParams` flips the flag picks up the
    // change. (Defines are also TextureNode-trigger; uniform numeric
    // value is the simpler source of truth here.)
    const isOrtho = (this.tslNodes.uIsOrtho.value as number) === 1;
    this.rebuildColormapNodes(useColormap);
    lineWebGPUFactory(
      this.tslNodes as LineTSLNodes,
      {
        useColormap,
        gammaOne,
        noGOG,
        isOrtho,
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
   * Rebind the line data texture. TSL `texture()` captures the
   * THREE.Texture at factory time, so an identity change needs a
   * fresh node + graph rebuild (exact mirror of
   * `PointTSLMaterial.updatePointTexture` and the colormap texture
   * lifecycle). No-op when the texture is unchanged — the common
   * per-commit case.
   */
  updateLineTexture(tex: THREE.DataTexture | null): void {
    const current = (this.uniforms.uLineTex?.value as THREE.Texture | null | undefined) ?? null;
    const next = tex ?? getPlaceholderElementTexture();
    if (current === next) return;
    this.tslNodes.uLineTex = texture(next);
    this.uniforms.uLineTex = proxyIUniform(this.tslNodes.uLineTex);
    this.rebuildGraph();
  }

  /** The currently bound line data texture. */
  getLineTexture(): THREE.DataTexture | null {
    return (this.uniforms.uLineTex?.value as THREE.DataTexture | null | undefined) ?? null;
  }

  /** Same toggle helper as `LineMaterial._refreshNoGOGDefine` — see there. */
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

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    const prevIsOrtho = (this.uniforms.uIsOrtho.value as number) === 1;
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    // Accept ANY defined value, including 0 — matching the point/gsplat
    // wrappers (the shader floors at 1e-20). The old `> 0` gate silently
    // KEPT a stale value on zero-diagonal scenes (or, with LRU-cached
    // materials, the previous dataset's nearCull), re-creating the
    // cross-geometry near-fade divergence B9c fixed.
    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
    // Precomputed pixel-width scales — see LineMaterial.updateCameraParams.
    const safeFov = Math.max(fov, 1e-4);
    if (isOrtho) {
      this.uniforms.uOrthoLineScale.value = (2.0 * resolution.y) / safeFov;
    } else {
      this.uniforms.uPerspectiveLineScale.value =
        resolution.y / Math.max(Math.tan(safeFov * 0.5), 1e-4);
    }
    // Each projection mode is a separate TSL graph variant. Rebuild
    // when the mode flips so the unused branch is dropped from the
    // generated WGSL/GLSL.
    if (isOrtho !== prevIsOrtho) {
      this.rebuildGraph();
    }
  }

  updateOpacity(opacity: number): void {
    this.uniforms.uOpacity.value = opacity;
  }

  /** Current opacity multiplier (the LOD cross-fade snapshots this as its fade base). */
  getOpacity(): number {
    return this.uniforms.uOpacity.value as number;
  }

  updateGamma(gamma: number): void {
    const safeGamma = clampGamma(gamma);
    this.userData.gamma = safeGamma;
    this.uniforms.uInvGamma.value = 1.0 / safeGamma;

    // Toggle `LUXAR_GAMMA_ONE` define when crossing the threshold and
    // rebuild the TSL graph so the factory picks the new fast-path
    // branch.
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
    if (this._refreshNoGOGDefine()) this.rebuildGraph();
  }

  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
    if (this._refreshNoGOGDefine()) this.rebuildGraph();
  }

  updateColormapTexture(tex: THREE.DataTexture | null): void {
    const oldTexture =
      (this.uniforms.uColormapTex?.value as THREE.Texture | null | undefined) ?? null;
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, tex);
    const textureChanged = oldTexture !== tex;
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

    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
    // Phase-1 volumetric fallback — the policy lives in
    // effectiveGeometryMode (blending-state.ts); userData keeps the
    // REQUESTED mode so stored scenes upgrade automatically.
    const effectiveMode: BlendingMode = effectiveGeometryMode(mode, 'line');
    const state: CompleteBlendingState = getCompleteBlendingState(effectiveMode, opacity);

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
    const cloned = new LineTSLMaterial({
      opacity: this.uniforms.uOpacity.value,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      blendingMode:
        (this.userData.blendingMode as LineMaterialConfig['blendingMode']) ?? 'additive',
      depthTest: this.userData.depthTest ?? true,
      transparent: this.transparent,
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
    });

    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    // `uIsOrtho` is a graph-specialized config — the constructor's
    // `rebuildGraph` ran against the default value 0 (perspective).
    // Copy uniforms, then re-run the rebuild against the now-correct
    // value so the right pixel-width branch is emitted.
    const sourceIsOrtho = (this.uniforms.uIsOrtho.value as number) === 1;
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    cloned.uniforms.uPerspectiveLineScale.value = this.uniforms.uPerspectiveLineScale.value;
    cloned.uniforms.uOrthoLineScale.value = this.uniforms.uOrthoLineScale.value;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;
    if (sourceIsOrtho) {
      cloned.rebuildGraph();
    }
    // Rebind the line data texture LAST (its own rebuild picks up the
    // ortho flag copied above). No-op when still on the placeholder.
    const lineTex = this.uniforms.uLineTex?.value as THREE.DataTexture | null | undefined;
    if (lineTex) cloned.updateLineTexture(lineTex);

    return cloned as this;
  }

  setColormapTexture(tex: THREE.DataTexture | null): void {
    if (!this.defines) this.defines = {};
    if (tex) {
      this.defines.USE_COLORMAP = '';
      if (!this.uniforms.uColormapTex) {
        this.uniforms.uColormapTex = { value: tex };
        this.uniforms.uScalarMin = { value: 0.0 };
        this.uniforms.uScalarScale = { value: 1.0 };
      } else {
        this.uniforms.uColormapTex.value = tex;
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
