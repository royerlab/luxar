/**
 * Mesh material (GLSL3 / `THREE.ShaderMaterial`).
 *
 * The fourth member of the visual-material family, and the first one that shades.
 * Structurally simpler than its three siblings — no element texture, no
 * sorted-index slot — because a mesh needs none of the instanced-quad machinery:
 * its size *is* its geometry, so there is nothing to project a sprite extent for
 * and no per-element storage to index into.
 *
 * It *is* a `CameraAwareMaterial`, but for only half of the contract's reason. The
 * fov/resolution half is genuinely inapplicable (there is no screen-space size to
 * recompute), so both are ignored; what mesh does need is the projection mode and
 * the near-cull distance, because the shared `perspectiveNearFade` applies to a
 * surface exactly as it does to a sprite — a triangle that clipped hard against the
 * near plane while every other type faded would be the only popping geometry in the
 * scene (#1431).
 *
 * It does implement `ColormapAwareMaterial`, which mesh needs in full: `scalars` +
 * `colormap` is a first-class authoring path for a surface (curvature, thickness,
 * expression level painted on an isosurface).
 *
 * @module rendering/materials/mesh/material-glsl
 */

import * as THREE from 'three';
import { MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER } from './shader-glsl';
import {
  MESH_DEFAULTS,
  clampAppearanceFraction,
  clampShadeExponent,
  clampShininess,
  resolveMeshBlendingMode,
  resolveMeshColorSource,
  resolveMeshOutput,
  syncMeshColorSourceDefines,
  syncMeshEmissionDefines,
  syncMeshShadingDefines,
  type MeshShadingMode,
} from './appearance';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import { getGlassDepthTexture } from '../_shared/glass-partition';
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
 * - `flatNormal`, the four lighting controls and `alphaCutoff` are new, because
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
   * How normals are obtained: stored, derivative, or none at all.
   *
   * A compile-time variant, not a runtime branch — see `MeshTSLConfig.shading` for
   * why, and `createMeshNode` for who decides it.
   *
   * Replaced the earlier `flatNormal` boolean when `'none'` arrived. Two booleans
   * would admit a meaningless `flatNormal && noShading` state and double the
   * variant count for something with one behaviour; an enum makes the invalid
   * combination not exist.
   */
  shading?: MeshShadingMode;
  /**
   * Base-colour texture, sampled per fragment through the `uv` attribute.
   *
   * The third and most specific base-colour source: it WINS over
   * `colormapTexture`, because a colormap can be switched on at runtime by the
   * layers panel for any node with scalars while a texture only exists if the
   * store carried an image and its UVs. Exactly one source is ever active — see
   * {@link syncMeshColorSourceDefines}.
   */
  baseColorTexture?: THREE.Texture;
  /**
   * Whether {@link baseColorTexture} is single-channel, so the shader must
   * replicate red to RGB.
   *
   * A 1-channel texture uploads as `RedFormat` and samples as `(r, 0, 0, 1)`, so
   * without this a greyscale basemap renders pure red. Carried as a flag rather
   * than read off the texture's `format` so the material never has to reach into
   * the upload's choices.
   */
  baseColorTextureLuminance?: boolean;
  /** Wrapped-diffuse shade floor; default {@link MESH_DEFAULTS}.ambient. */
  ambient?: number;
  /** Wrapped-diffuse exponent; default {@link MESH_DEFAULTS}.shadeExponent. */
  shadeExponent?: number;
  /** Additive specular strength; default {@link MESH_DEFAULTS}.specular. */
  specular?: number;
  /** Specular exponent; default {@link MESH_DEFAULTS}.shininess. */
  shininess?: number;
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
  const defines: Record<string, string | number | boolean> = {
    ...(isGammaOne(gammaValue) ? { LUXAR_GAMMA_ONE: '' } : {}),
    ...(isNoGOG(config.intensity ?? 1.0, config.offset ?? 0.0) ? { LUXAR_NO_GOG: '' } : {}),
    ...(config.baseColorTexture && config.baseColorTextureLuminance
      ? { LUXAR_MESH_TEX_LUMINANCE: '' }
      : {}),
  };
  // All three seeded through the same helpers the runtime paths use, so the
  // constructor and a later mutation cannot disagree about which flag a state
  // implies. USE_COLORMAP is set HERE rather than inline above for exactly that
  // reason: it is one arm of the at-most-one-colour-source invariant.
  syncMeshColorSourceDefines(defines, resolveMeshColorSource(config));
  syncMeshShadingDefines(defines, config.shading ?? 'smooth');
  syncMeshEmissionDefines(defines, output);
  return defines;
}

/** Mesh surface material — shaded, indexed triangles. */
export class MeshMaterial
  extends THREE.ShaderMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
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
        uAmbient: {
          value: clampAppearanceFraction(materialConfig.ambient, MESH_DEFAULTS.ambient),
        },
        uShadeExponent: { value: clampShadeExponent(materialConfig.shadeExponent) },
        uSpecular: {
          value: clampAppearanceFraction(materialConfig.specular, MESH_DEFAULTS.specular),
        },
        uShininess: { value: clampShininess(materialConfig.shininess) },
        uAlphaCutoff: {
          value: clampAppearanceFraction(materialConfig.alphaCutoff, MESH_DEFAULTS.alphaCutoff),
        },
        uNearCull: { value: 0.1 }, // Default; overridden per-scene by updateCameraParams
        // Refraction split (glass-partition.ts): mode 0 outside the split; the ONE
        // shared depth texture the split renders the refracting glass into.
        uGlassPartition: { value: 0 },
        uGlassDepth: { value: getGlassDepthTexture() },
        ...(materialConfig.baseColorTexture
          ? { uBaseColorTex: { value: materialConfig.baseColorTexture } }
          : {}),
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
    this.userData.shading = materialConfig.shading ?? 'smooth';
    // Carried too, because `clone()` rebuilds from the config and the texture is
    // not recoverable from a uniform alone: `uBaseColorTex` exists only when a
    // texture was supplied, and the luminance flag is not derivable from it
    // without reaching into the upload's format choice.
    this.userData.baseColorTextureLuminance = materialConfig.baseColorTextureLuminance === true;
  }

  /**
   * Update the camera-dependent uniforms.
   *
   * `_resolution` and `_isOrtho` are accepted and IGNORED: the resolution exists so
   * a material can size a screen-space sprite, and a mesh's size is its own
   * geometry; the near fade's ortho test reads three's `isOrthographic`, i.e. the
   * camera this draw uses. Only `nearCull` is consumed. Named with a leading
   * underscore so the asymmetry is visible at the signature rather than buried in
   * the body.
   */
  updateCameraParams(
    _resolution: THREE.Vector2,
    _isOrtho: boolean = false,
    nearCull?: number,
    _pixelRatio?: number
  ): void {
    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
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

  /** Wrapped-diffuse shade floor (`1.0` = flat diffuse). Plain uniform write. */
  updateAmbient(ambient: number): void {
    this.uniforms.uAmbient.value = clampAppearanceFraction(ambient, MESH_DEFAULTS.ambient);
  }

  /** Wrapped-diffuse exponent. Plain uniform write. */
  updateShadeExponent(exponent: number): void {
    // Clamped like `updateGamma` — `pow(0, y)` is undefined for y <= 0, and a
    // face-away fragment has a wrap base of exactly 0 (see `clampShadeExponent`).
    this.uniforms.uShadeExponent.value = clampShadeExponent(exponent);
  }

  updateSpecular(specular: number): void {
    this.uniforms.uSpecular.value = clampAppearanceFraction(specular, MESH_DEFAULTS.specular);
  }

  updateShininess(shininess: number): void {
    this.uniforms.uShininess.value = clampShininess(shininess);
  }

  /**
   * `opaque`-mode cutout threshold. Plain uniform write — inert in every other
   * mode, where the coverage is applied by the framebuffer instead of compared.
   */
  updateAlphaCutoff(cutoff: number): void {
    this.uniforms.uAlphaCutoff.value = clampAppearanceFraction(cutoff, MESH_DEFAULTS.alphaCutoff);
  }

  /**
   * Switch between the stored-normal, derivative-normal and unlit variants.
   *
   * Called per commit by `applyMeshShading`, because the `normal_dims ==
   * displayDims` half of the §3.4 rule is view-dependent. Idempotent and guarded:
   * toggling a define recompiles the program, which must not happen on a slice
   * move that changed nothing.
   *
   * `'none'` is NOT view-dependent — it is authored — but it flows through the
   * same setter so there is one place that owns the shading define set. A separate
   * `setUnlit` would let a per-commit `updateShading('flat')` silently clear it.
   */
  updateShading(mode: MeshShadingMode): void {
    if (!this.defines) this.defines = {};
    if (syncMeshShadingDefines(this.defines, mode)) {
      this.userData.shading = mode;
      this.needsUpdate = true;
    } else {
      this.userData.shading = mode;
    }
  }

  /**
   * Swap the base-colour texture, maintaining the one-colour-source invariant.
   *
   * A null texture falls back to whatever the other sources provide, which is why
   * this routes through the shared resolver rather than just deleting its own
   * define: dropping a texture from a node that also has a colormap LUT must
   * re-enable `USE_COLORMAP`, not leave the mesh with no colour source at all.
   */
  updateBaseColorTexture(texture: THREE.Texture | null, luminance = false): void {
    if (!this.defines) this.defines = {};
    if (texture) {
      if (this.uniforms.uBaseColorTex) this.uniforms.uBaseColorTex.value = texture;
      else this.uniforms.uBaseColorTex = { value: texture };
    } else if (this.uniforms.uBaseColorTex) {
      this.uniforms.uBaseColorTex.value = null;
    }
    const wantLuminance = !!texture && luminance;
    const hadLuminance = 'LUXAR_MESH_TEX_LUMINANCE' in this.defines;
    if (wantLuminance && !hadLuminance) this.defines.LUXAR_MESH_TEX_LUMINANCE = '';
    else if (!wantLuminance && hadLuminance) delete this.defines.LUXAR_MESH_TEX_LUMINANCE;
    this.userData.baseColorTextureLuminance = wantLuminance;

    const changed = syncMeshColorSourceDefines(
      this.defines,
      resolveMeshColorSource({
        baseColorTexture: texture ?? undefined,
        colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      })
    );
    if (changed || wantLuminance !== hadLuminance) this.needsUpdate = true;
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
    // One shared helper maintains the at-most-one-emission-define invariant for both
    // backends — see `syncMeshEmissionDefines`.
    const definesChanged = syncMeshEmissionDefines(this.defines, state.shaderOutputMode);
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
      specular: this.uniforms.uSpecular.value,
      shininess: this.uniforms.uShininess.value,
      alphaCutoff: this.uniforms.uAlphaCutoff.value,
      blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'opaque',
      shading: (this.userData.shading as MeshShadingMode | undefined) ?? 'smooth',
      baseColorTexture: (this.uniforms.uBaseColorTex?.value as THREE.Texture | null) ?? undefined,
      baseColorTextureLuminance: this.userData.baseColorTextureLuminance === true,
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
    // Camera state, carried live rather than left at the constructor default: a
    // clone that reverted to 0.1 would fade against the WRONG near plane until the
    // next broadcast reached it.
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    return cloned as this;
  }

  // ---- ColormapAwareMaterial ----

  setColormapTexture(texture: THREE.DataTexture | null): void {
    if (texture) {
      if (!this.uniforms.uColormapTex) {
        this.uniforms.uColormapTex = { value: texture };
        this.uniforms.uScalarMin = { value: 0.0 };
        this.uniforms.uScalarScale = { value: 1.0 };
      } else {
        this.uniforms.uColormapTex.value = texture;
      }
    } else {
      if (this.uniforms.uColormapTex) this.uniforms.uColormapTex.value = null;
      if (this.uniforms.uScalarMin) this.uniforms.uScalarMin.value = 0.0;
      if (this.uniforms.uScalarScale) this.uniforms.uScalarScale.value = 1.0;
    }
    syncMeshColorSourceDefines(
      this.defines,
      resolveMeshColorSource({
        baseColorTexture: this.uniforms.uBaseColorTex?.value ?? undefined,
        colormapTexture: texture ?? undefined,
      })
    );
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
