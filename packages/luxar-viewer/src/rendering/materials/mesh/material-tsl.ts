/**
 * TSL / NodeMaterial counterpart to `MeshMaterial`.
 *
 * Mirrors the GLSL wrapper one-for-one — same `MeshMaterialConfig`, same update
 * methods, same `applyBlendingMode` + `clone` semantics, same
 * `ColormapAwareMaterial` surface — so `MaterialManager.getMeshMaterial` is a
 * drop-in swap and no caller sees the divergence.
 *
 * **Uniform plumbing.** This class owns one persistent `UniformNode` per shader
 * input in `tslNodes`, and the public `uniforms` record exposes each as an
 * `IUniform`-shaped getter/setter proxy, so `material.uniforms.X.value = …` lands
 * straight on `node.value` with no per-render callback bridge. Identical to the
 * three sibling TSL wrappers.
 *
 * **What needs a graph rebuild.** The factory branches on JS-side conditionals for
 * the colormap path, the gamma/GOG fast paths, the flat-normal variant and the
 * emission shape, so each of those changes the *shape* of the graph and rebuilds.
 * Everything else — opacity, intensity, offset, ambient, shade exponent, alpha
 * cutoff, scalar range — is a plain runtime uniform and never rebuilds.
 *
 * @module rendering/materials/mesh/material-tsl
 */

import * as THREE from 'three';
import { uniform, texture } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { meshWebGPUFactory, type MeshTSLNodes } from './shader-tsl';
import {
  MESH_DEFAULTS,
  clampAppearanceFraction,
  clampShadeExponent,
  resolveMeshBlendingMode,
  resolveMeshOutput,
  syncMeshEmissionDefines,
} from './appearance';
import type { MeshMaterialConfig } from './material-glsl';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne, isNoGOG } from '../_shared/uniform-helpers';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  type CompleteBlendingState,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import { proxyIUniform, type TSLNode } from '../_shared/tsl-helpers';
import { computeScalarRangeUniforms } from '../_shared/scalar-range';

/**
 * Persistent TSL node table owned by the wrapper. Colormap nodes are (re)created
 * lazily inside `rebuildGraph()` when colormap mode toggles — the `MeshTSLNodes`
 * contract treats them as optional. Keys match the public `uniforms` record.
 */
interface MeshMaterialTSLNodeTable {
  uOpacity: TSLNode;
  uInvGamma: TSLNode;
  uIntensity: TSLNode;
  uOffset: TSLNode;
  uAmbient: TSLNode;
  uShadeExponent: TSLNode;
  uAlphaCutoff: TSLNode;
  uColormapTex?: TSLNode;
  uScalarMin?: TSLNode;
  uScalarScale?: TSLNode;
}

/** Mesh surface material rendered via TSL / NodeMaterial. */
export class MeshTSLMaterial extends NodeMaterial implements ColormapAwareMaterial {
  /** Public uniforms table, same shape as `MeshMaterial.uniforms`. */
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: MeshMaterialTSLNodeTable;

  /**
   * Explicit `depthTest` / `transparent` overrides from the constructor config.
   * Every `rebuildGraph()` re-applies the factory tail's MODE-DERIVED blending
   * state, so an override honored only in the constructor would silently revert on
   * the first later rebuild. The GLSL twin never rebuilds, so its constructor-tail
   * overrides stick; persisting them here keeps the two contract-identical. An
   * explicit `applyBlendingMode()` clears both — a runtime mode switch takes full
   * ownership of the blending state, matching the GLSL twin.
   */
  private _explicitDepthTest?: boolean;
  private _explicitTransparent?: boolean;

  constructor(materialConfig: MeshMaterialConfig = {}) {
    super();

    const gammaValue = clampGamma(materialConfig.gamma);

    this.tslNodes = {
      uOpacity: uniform(materialConfig.opacity ?? 1.0),
      uInvGamma: uniform(1.0 / gammaValue),
      uIntensity: uniform(materialConfig.intensity ?? 1.0),
      uOffset: uniform(materialConfig.offset ?? 0.0),
      uAmbient: uniform(clampAppearanceFraction(materialConfig.ambient, MESH_DEFAULTS.ambient)),
      uShadeExponent: uniform(clampShadeExponent(materialConfig.shadeExponent)),
      uAlphaCutoff: uniform(
        clampAppearanceFraction(materialConfig.alphaCutoff, MESH_DEFAULTS.alphaCutoff)
      ),
    };

    this.uniforms = {
      uOpacity: proxyIUniform(this.tslNodes.uOpacity),
      uInvGamma: proxyIUniform(this.tslNodes.uInvGamma),
      uIntensity: proxyIUniform(this.tslNodes.uIntensity),
      uOffset: proxyIUniform(this.tslNodes.uOffset),
      uAmbient: proxyIUniform(this.tslNodes.uAmbient),
      uShadeExponent: proxyIUniform(this.tslNodes.uShadeExponent),
      uAlphaCutoff: proxyIUniform(this.tslNodes.uAlphaCutoff),
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

    // The defines record is the shared source of truth for which graph variant is
    // built — the same flags the GLSL twin hands to the preprocessor drive the
    // factory's JS-side conditionals here, so the two backends can never build
    // different variants from the same config.
    this.defines = materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {};
    if (isGammaOne(gammaValue)) this.defines.LUXAR_GAMMA_ONE = '';
    if (isNoGOG(materialConfig.intensity ?? 1.0, materialConfig.offset ?? 0.0)) {
      this.defines.LUXAR_NO_GOG = '';
    }
    if (materialConfig.flatNormal) this.defines.LUXAR_MESH_FLAT_NORMAL = '';
    this.toneMapped = false;

    this.userData.gamma = gammaValue;
    this.userData.scalarRange = materialConfig.scalarRange;
    this.userData.flatNormal = materialConfig.flatNormal === true;

    // Stamp the resolved mode BEFORE the first rebuild so the factory reads the
    // right branch on its first build (mirrors the sibling TSL constructors).
    const resolved = resolveMeshBlendingMode(materialConfig.blendingMode ?? 'opaque');
    this.userData.blendingMode = resolved;
    syncMeshEmissionDefines(this.defines, resolveMeshOutput(resolved));

    // Capture explicit overrides BEFORE the first rebuild — see the field doc.
    this._explicitTransparent = materialConfig.transparent;
    this._explicitDepthTest = materialConfig.depthTest;

    // `side` is epoch state re-applied per commit by `applyMeshSide`; start on
    // FrontSide like the GLSL twin so the two agree before the first commit.
    this.side = THREE.FrontSide;

    this.rebuildGraph();

    this.userData.depthTest = this.depthTest;
  }

  /**
   * Refresh the colormap TSL nodes so they bind the current
   * `this.uniforms.uColormap*.value`. `TextureNode` binds a specific `Texture` at
   * construction, so a swap needs a fresh node — hence this lives in
   * `rebuildGraph()`. Mirrors `PointTSLMaterial.rebuildColormapNodes`.
   */
  private rebuildColormapNodes(useColormap: boolean): void {
    if (useColormap) {
      const tex =
        (this.uniforms.uColormapTex?.value as THREE.Texture | null | undefined) ??
        new THREE.Texture();
      this.tslNodes.uColormapTex = texture(tex);
      this.tslNodes.uScalarMin = uniform((this.uniforms.uScalarMin?.value as number) ?? 0.0);
      this.tslNodes.uScalarScale = uniform((this.uniforms.uScalarScale?.value as number) ?? 1.0);
      this.uniforms.uScalarMin = proxyIUniform(this.tslNodes.uScalarMin);
      this.uniforms.uScalarScale = proxyIUniform(this.tslNodes.uScalarScale);
      this.uniforms.uColormapTex = { value: tex };
    } else {
      this.tslNodes.uColormapTex = undefined;
      this.tslNodes.uScalarMin = undefined;
      this.tslNodes.uScalarScale = undefined;
    }
  }

  /**
   * Re-run the TSL factory and attach the resulting `vertexNode` / `colorNode` +
   * blending state to ourselves. Every variant flag is read back out of `defines`,
   * the same source of truth the GLSL twin's preprocessor consumes.
   */
  private rebuildGraph(): void {
    const has = (flag: string): boolean => !!this.defines && flag in this.defines;
    const useColormap = has('USE_COLORMAP');
    this.rebuildColormapNodes(useColormap);
    meshWebGPUFactory(
      this.tslNodes as MeshTSLNodes,
      {
        useColormap,
        gammaOne: has('LUXAR_GAMMA_ONE'),
        noGOG: has('LUXAR_NO_GOG'),
        flatNormal: has('LUXAR_MESH_FLAT_NORMAL'),
        blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'opaque',
      },
      this
    );
    // Re-apply the explicit constructor overrides over the factory tail's
    // mode-derived state on EVERY rebuild — see the `_explicitDepthTest` field doc.
    if (this._explicitTransparent !== undefined) this.transparent = this._explicitTransparent;
    if (this._explicitDepthTest !== undefined) {
      this.depthTest = this._explicitDepthTest;
      this.userData.depthTest = this._explicitDepthTest;
    }
    this.needsUpdate = true;
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

  /** Mirrors `MeshMaterial._refreshNoGOGDefine`. */
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

  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
    if (this._refreshNoGOGDefine()) this.rebuildGraph();
  }

  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
    if (this._refreshNoGOGDefine()) this.rebuildGraph();
  }

  updateAmbient(ambient: number): void {
    this.uniforms.uAmbient.value = clampAppearanceFraction(ambient, MESH_DEFAULTS.ambient);
  }

  updateShadeExponent(exponent: number): void {
    // Clamped like `updateGamma` — `pow(0, y)` is undefined for y <= 0, and a
    // face-away fragment has a wrap base of exactly 0 (see `clampShadeExponent`).
    this.uniforms.uShadeExponent.value = clampShadeExponent(exponent);
  }

  updateAlphaCutoff(cutoff: number): void {
    this.uniforms.uAlphaCutoff.value = clampAppearanceFraction(cutoff, MESH_DEFAULTS.alphaCutoff);
  }

  /**
   * Switch between the stored-normal and derivative-normal graph variants — the
   * twin of `MeshMaterial.updateFlatNormal`. Idempotent and guarded: a flip rebuilds
   * the graph, which must not happen on a slice move that changed nothing.
   */
  updateFlatNormal(flatNormal: boolean): void {
    if (!this.defines) this.defines = {};
    const had = 'LUXAR_MESH_FLAT_NORMAL' in this.defines;
    if (flatNormal === had) return;
    if (flatNormal) this.defines.LUXAR_MESH_FLAT_NORMAL = '';
    else delete this.defines.LUXAR_MESH_FLAT_NORMAL;
    this.userData.flatNormal = flatNormal;
    this.rebuildGraph();
  }

  /**
   * Toggle the colormap branch. Rebuilds on an on/off flip OR a texture-identity
   * swap: TSL's `texture()` captures the Texture at factory-call time, so a swap
   * needs a fresh node too. (The GLSL twin gets away with a plain uniform write
   * because `ShaderMaterial` reads the IUniform by reference at render time.)
   */
  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const oldTexture =
      (this.uniforms.uColormapTex?.value as THREE.Texture | null | undefined) ?? null;
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
    if (wasEnabled !== nowEnabled || oldTexture !== texture) this.rebuildGraph();
  }

  applyBlendingMode(mode: BlendingMode): void {
    // A runtime mode switch takes FULL ownership of the blending state — matching
    // the GLSL twin, where applyBlendingStateToMaterial overwrites unconditionally.
    this._explicitDepthTest = undefined;
    this._explicitTransparent = undefined;

    const resolved = resolveMeshBlendingMode(mode);
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = getCompleteBlendingState(resolved, opacity);

    if (!this.defines) this.defines = {};

    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const stateChanged = applyBlendingStateToMaterial(this, state);
    // One shared helper maintains the at-most-one-emission-define invariant for both
    // backends — see `syncMeshEmissionDefines`.
    const definesChanged = syncMeshEmissionDefines(this.defines, state.shaderOutputMode);
    this.userData.blendingMode = resolved;
    this.userData.depthTest = state.depthTest;

    // The emission branch is baked into the graph, so any change to it rebuilds.
    if (definesChanged) {
      this.rebuildGraph();
    } else if (previousMode !== resolved && stateChanged) {
      this.needsUpdate = true;
    }
  }

  updateScalarRange(min: number, max: number): void {
    applyScalarRangeToMaterial(this, min, max);
  }

  clone(): this {
    const cloned = new MeshTSLMaterial({
      opacity: this.uniforms.uOpacity.value,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      ambient: this.uniforms.uAmbient.value,
      shadeExponent: this.uniforms.uShadeExponent.value,
      alphaCutoff: this.uniforms.uAlphaCutoff.value,
      blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'opaque',
      flatNormal: this.userData.flatNormal === true,
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
    // `side` is epoch state, not config — carry the live value (see the GLSL twin).
    cloned.side = this.side;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;
    return cloned as this;
  }

  // ---- ColormapAwareMaterial ----

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
    if (this.uniforms.uScalarScale) this.uniforms.uScalarScale.value = scalarScale;
  }
}
