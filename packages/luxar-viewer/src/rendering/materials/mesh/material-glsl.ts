/**
 * Mesh material (GLSL3 / `THREE.ShaderMaterial`).
 *
 * The fourth member of the visual-material family, and the first one that shades.
 * Structurally simpler than its three siblings — no camera uniforms, no element
 * texture, no sorted-index slot — because a mesh needs none of the instanced-quad
 * machinery: its size *is* its geometry, so there is nothing to project a sprite
 * extent for and no per-element storage to index into.
 *
 * That absence is why this class deliberately does **not** implement
 * `CameraAwareMaterial`. The interface exists so `SceneManager` can broadcast
 * fov/resolution/ortho to materials that compute screen-space sizes; a mesh has no
 * screen-space size, so an empty `updateCameraParams` would be a lie that also
 * costs a per-frame call per node.
 *
 * It does implement `ColormapAwareMaterial`, which mesh needs in full: `scalars` +
 * `colormap` is a first-class authoring path for a surface (curvature, thickness,
 * expression level painted on an isosurface).
 *
 * @module rendering/materials/mesh/material-glsl
 */

import * as THREE from 'three';
import { MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER } from './shader-glsl';
import { MESH_DEFAULTS, resolveMeshBlendingMode, resolveMeshOutput } from './appearance';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne, isNoGOG } from '../_shared/uniform-helpers';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  getCompleteBlendingState,
  applyBlendingStateToMaterial,
  type CompleteBlendingState,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import { computeScalarRangeUniforms, scalarRangeUniformEntries } from '../_shared/scalar-range';

/**
 * Configuration for mesh material creation.
 *
 * The compositing half (`opacity` / `gamma` / `intensity` / `offset` /
 * `blendingMode` / `depthTest` / `transparent` / `colormapTexture` / `scalarRange`)
 * mirrors `PointMaterialConfig` field for field, so the layers panel and the
 * material-sync path treat all four geometry types identically.
 *
 * What differs is named rather than silently omitted:
 * - no `absorption` / `hasElementAlpha` — both exist only to serve the volumetric
 *   mode, which a zero-thickness surface cannot express (§6.3);
 * - no `radiusScale` / `truncationRadius` — those normalize a per-element extent,
 *   and a triangle's extent is its own vertices;
 * - `flatNormal`, `ambient`, `shadeExponent` and `alphaCutoff` are new, because
 *   mesh is the first shaded type and the first with a cutout.
 */
export interface MeshMaterialConfig {
  opacity?: number;
  gamma?: number;
  /** Linear colour multiplier (gain), default 1.0. */
  intensity?: number;
  /** Additive brightness shift (black level), default 0.0. */
  offset?: number;
  /**
   * Luxar blending mode. Default `'opaque'` — the MESH default, unlike the three
   * siblings' `'additive'`. `opaque` is the only mode unconditionally correct
   * without per-triangle depth sorting, and it is what a surface should look like
   * (§6.3). `'volumetric'` is mapped to `'opaque'`; see
   * {@link resolveMeshBlendingMode}.
   */
  blendingMode?: BlendingMode;
  /**
   * Shade from screen-space derivatives instead of the stored `normal` attribute.
   * A compile-time variant, not a runtime branch — see `MeshTSLConfig.flatNormal`
   * for why, and `createMeshNode` for who decides it.
   */
  flatNormal?: boolean;
  /** Headlight shade floor; default {@link MESH_DEFAULTS}.ambient. */
  ambient?: number;
  /** Headlight wrap exponent; default {@link MESH_DEFAULTS}.shadeExponent. */
  shadeExponent?: number;
  /** `opaque`-mode cutout threshold; default {@link MESH_DEFAULTS}.alphaCutoff. */
  alphaCutoff?: number;
  depthTest?: boolean;
  transparent?: boolean;
  /** Colormap LUT texture (256x1 RGB). */
  colormapTexture?: THREE.DataTexture;
  /** Scalar data range `[min, max]` for normalization. */
  scalarRange?: [number, number];
}

/**
 * Build the full define set for a config.
 *
 * Includes the two MODE-derived flags even though `applyBlendingMode` maintains them
 * afterwards, which is deliberate: seeding them here makes the constructor's
 * `applyBlendingMode` call a no-op on the define set, so a freshly built material
 * never takes a pointless first-frame recompile. (The point/line/gsplat
 * constructors omit their max define and eat that recompile; the mesh emission is a
 * three-way choice, so the same omission would recompile every mesh ever built.)
 */
function meshDefines(
  config: MeshMaterialConfig,
  gammaValue: number
): Record<string, string | number | boolean> {
  const output = resolveMeshOutput(config.blendingMode ?? 'opaque');
  return {
    ...(config.colormapTexture ? { USE_COLORMAP: '' } : {}),
    ...(isGammaOne(gammaValue) ? { LUXAR_GAMMA_ONE: '' } : {}),
    ...(isNoGOG(config.intensity ?? 1.0, config.offset ?? 0.0) ? { LUXAR_NO_GOG: '' } : {}),
    ...(config.flatNormal ? { LUXAR_MESH_FLAT_NORMAL: '' } : {}),
    ...(output === 'opaque' ? { LUXAR_MESH_ALPHA_CUTOUT: '' } : {}),
    ...(output === 'rgb-contribution' ? { LUXAR_MAX_RGB_CONTRIBUTION: '' } : {}),
  };
}

/** Mesh surface material — shaded, indexed triangles. */
export class MeshMaterial extends THREE.ShaderMaterial implements ColormapAwareMaterial {
  constructor(materialConfig: MeshMaterialConfig = {}) {
    // Resolve up front: a `volumetric` mode inherited from an ancestor group must
    // not reach either the defines or the framebuffer state (§6.3).
    const blendingMode: BlendingMode = resolveMeshBlendingMode(
      materialConfig.blendingMode ?? 'opaque'
    );
    const gammaValue = clampGamma(materialConfig.gamma);
    const initialState = getCompleteBlendingState(blendingMode, materialConfig.opacity ?? 1.0);

    super({
      uniforms: {
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
        uInvGamma: { value: 1.0 / gammaValue },
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },
        uAmbient: { value: materialConfig.ambient ?? MESH_DEFAULTS.ambient },
        uShadeExponent: {
          value: materialConfig.shadeExponent ?? MESH_DEFAULTS.shadeExponent,
        },
        uAlphaCutoff: { value: materialConfig.alphaCutoff ?? MESH_DEFAULTS.alphaCutoff },
        ...(materialConfig.colormapTexture
          ? {
              uColormapTex: { value: materialConfig.colormapTexture },
              // Midpoint identity for degenerate ranges — see scalar-range.ts.
              ...scalarRangeUniformEntries(materialConfig.scalarRange),
            }
          : {}),
      },

      vertexShader: MESH_VERTEX_SHADER,
      fragmentShader: MESH_FRAGMENT_SHADER,

      defines: meshDefines({ ...materialConfig, blendingMode }, gammaValue) as Record<
        string,
        unknown
      >,

      glslVersion: THREE.GLSL3,

      // `vertexColors` stays FALSE even though the mesh very much has per-vertex
      // colours: the flag exists to make three inject its own `color` attribute
      // declaration plus the `color_pars_*` chunks into a BUILT-IN material's
      // shader. This shader declares `in vec4 color;` itself and reads it directly,
      // so the injection would collide with our own declaration.
      vertexColors: false,
      // `side` is owned by the epoch, not the config: `applyMeshSide` re-applies it
      // on every commit from the projection's decision (an undecidable display frame
      // forces DoubleSide regardless of the authored `double_sided`). Start on the
      // authored-agnostic FrontSide and let the first commit correct it.
      side: THREE.FrontSide,
      toneMapped: false, // HDR values pass through to post-processing
      blending: initialState.blending,
      depthTest: initialState.depthTest,
      depthWrite: initialState.depthWrite,
      transparent: initialState.transparent,
    });

    // Canonical path for both creation and later UI transitions — sets the custom
    // blend factors, the mode-derived defines, and `userData.blendingMode`.
    this.applyBlendingMode(blendingMode);

    if (materialConfig.transparent !== undefined) this.transparent = materialConfig.transparent;
    if (materialConfig.depthTest !== undefined) this.depthTest = materialConfig.depthTest;

    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? initialState.depthTest;
    this.userData.scalarRange = materialConfig.scalarRange;
    // The variant selection rides on userData so `clone()` round-trips it — the
    // layers panel clones on first interaction, and a clone that lost `flatNormal`
    // would silently switch a flat-shaded mesh to smooth (or read an unbound
    // `normal` attribute as (0,0,0) and shade the whole surface at `uAmbient`).
    this.userData.flatNormal = materialConfig.flatNormal === true;
  }

  updateOpacity(opacity: number): void {
    this.uniforms.uOpacity.value = opacity;
  }

  /** Current opacity multiplier (the LOD cross-fade snapshots this as its fade base). */
  getOpacity(): number {
    return this.uniforms.uOpacity.value as number;
  }

  /**
   * Update gamma. Toggles `LUXAR_GAMMA_ONE` across the 1.0 threshold so the
   * fragment `pow()` fast path recompiles in/out — mirrors the sibling materials.
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

  /** Mirrors `PointMaterial._refreshNoGOGDefine` — the flag depends on both values. */
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
    if (this._refreshNoGOGDefine()) this.needsUpdate = true;
  }

  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
    if (this._refreshNoGOGDefine()) this.needsUpdate = true;
  }

  /** Headlight shade floor (1.0 = flat/emissive). Plain uniform write. */
  updateAmbient(ambient: number): void {
    this.uniforms.uAmbient.value = ambient;
  }

  /** Headlight wrap exponent. Plain uniform write. */
  updateShadeExponent(exponent: number): void {
    this.uniforms.uShadeExponent.value = exponent;
  }

  /**
   * `opaque`-mode cutout threshold. Plain uniform write — inert in every other
   * mode, where the coverage is applied by the framebuffer instead of compared.
   */
  updateAlphaCutoff(cutoff: number): void {
    this.uniforms.uAlphaCutoff.value = cutoff;
  }

  /**
   * Switch between the stored-normal and derivative-normal shader variants.
   *
   * Called per commit by `applyMeshShading`, because the `normal_dims ==
   * displayDims` half of the §3.4 rule is view-dependent. Idempotent and guarded:
   * toggling the define recompiles the program, which must not happen on a slice
   * move that changed nothing.
   */
  updateFlatNormal(flatNormal: boolean): void {
    if (!this.defines) this.defines = {};
    const had = 'LUXAR_MESH_FLAT_NORMAL' in this.defines;
    if (flatNormal === had) return;
    if (flatNormal) this.defines.LUXAR_MESH_FLAT_NORMAL = '';
    else delete this.defines.LUXAR_MESH_FLAT_NORMAL;
    this.userData.flatNormal = flatNormal;
    this.needsUpdate = true;
  }

  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
    if (wasEnabled !== nowEnabled) this.needsUpdate = true;
  }

  /**
   * Apply a Luxar blending mode in place.
   *
   * Single source of truth for creation-time wiring and runtime layers-panel
   * transitions alike, exactly as in the sibling materials. Two mesh-specific
   * points:
   *
   * - `volumetric` is mapped to `opaque` here as well as at load. The panel can
   *   offer it (the dropdown is shared), and a mesh must not be left in a mode
   *   whose fragment branch it does not implement.
   * - the emission shape is a THREE-way choice, so BOTH mode defines are
   *   maintained: `LUXAR_MESH_ALPHA_CUTOUT` for `opaque` and
   *   `LUXAR_MAX_RGB_CONTRIBUTION` for `max`. Each must be cleared when leaving its
   *   mode — an `opaque → additive` switch that stranded the cutout would keep
   *   discarding fragments in a mode that has no cutout.
   */
  applyBlendingMode(mode: BlendingMode): void {
    const resolved = resolveMeshBlendingMode(mode);
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = getCompleteBlendingState(resolved, opacity);

    if (!this.defines) this.defines = {};

    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const stateChanged = applyBlendingStateToMaterial(this, state);
    let definesChanged = false;
    for (const [flag, wanted] of [
      ['LUXAR_MESH_ALPHA_CUTOUT', state.shaderOutputMode === 'opaque'],
      ['LUXAR_MAX_RGB_CONTRIBUTION', state.shaderOutputMode === 'rgb-contribution'],
    ] as const) {
      const had = flag in this.defines;
      if (wanted && !had) {
        this.defines[flag] = '';
        definesChanged = true;
      } else if (!wanted && had) {
        delete this.defines[flag];
        definesChanged = true;
      }
    }
    this.userData.blendingMode = resolved;
    this.userData.depthTest = state.depthTest;

    if (definesChanged) {
      this.needsUpdate = true;
    } else if (previousMode !== resolved && stateChanged) {
      this.needsUpdate = true;
    }
  }

  updateScalarRange(min: number, max: number): void {
    applyScalarRangeToMaterial(this, min, max);
  }

  clone(): this {
    const cloned = new MeshMaterial({
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
    // `side` is epoch state, not config (see the constructor note), so carry the
    // live value rather than re-deriving it: a clone taken while an undecidable
    // display frame forced DoubleSide must keep drawing both faces until the next
    // commit re-applies it.
    cloned.side = this.side;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;
    return cloned as this;
  }

  // ---- ColormapAwareMaterial ----

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
    if (this.uniforms.uScalarScale) this.uniforms.uScalarScale.value = scalarScale;
  }

  // `dispose()` is inherited — MaterialManager subscribes to the base class's
  // synchronous `dispose` event, so no explicit unregister hook is needed here.
}
