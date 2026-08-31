/**
 * TSL / NodeMaterial counterpart to `GSplatMaterial`.
 *
 * Mirrors the GLSL `GSplatMaterial` wrapper one-for-one — same
 * constructor signature (`GSplatMaterialConfig`), same update
 * methods (opacity / gamma / intensity / offset / truncationRadius
 * / maxExtentFactor / colormapTexture / scalarRange), same
 * `applyBlendingMode` + `clone` semantics, same
 * `CameraAwareMaterial` / `ColormapAwareMaterial` interfaces.
 *
 * GSplat-specific behaviours:
 *
 *   - `applyBlendingMode` selects the projection model: PEAK
 *     (2D-projected) for surface modes (max/normal/opaque —
 *     `usesPeakProjection`), SUM (ray-integral) for emissive
 *     (additive/luminous/volumetric). In TSL
 *     the graph JS-branches on the mode at build time; the
 *     `uProjectionMode` uniform is decorative (clone/telemetry
 *     parity). The GLSL twin drives the same split via the uniform.
 *   - Blending state is fully unified with the GLSL wrapper: the
 *     shared `getCompleteBlendingState` for additive / luminous /
 *     max / opaque (AdditiveBlending's SrcAlpha factor is identity
 *     under the shader's alpha = 1.0 contract) and
 *     `getGSplatNormalBlendingState` for `normal` (premultiplied
 *     coverage alpha). No separate alpha-channel blend state
 *     anywhere — it trips gl.getError() under the WebGPU→WebGL2
 *     bridge, and its historical overflow-guard purpose is now
 *     handled by the raw-scene-hdr capture sanitization.
 *
 * Mechanics identical to `PointTSLMaterial` for the
 * uniforms-by-reference binding pattern.
 *
 * @module rendering/materials/gsplat/material-tsl
 */

import * as THREE from 'three';
import { texture, uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { gsplatWebGPUFactory, type GSplatTSLNodes } from './shader-tsl';
import type { GSplatMaterialConfig } from './material-glsl';
import type { CameraAwareMaterial } from '../_shared/camera-aware-material';
import type { ColormapAwareMaterial } from '../_shared/colormap-aware-material';
import { clampGamma, isGammaOne, isNoGOG } from '../_shared/uniform-helpers';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
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
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  getGSplatNormalBlendingState,
  isNormalMode,
  isVolumetricMode,
  usesPeakProjection,
  type CompleteBlendingState,
} from '../../blending-state';
import { proxyIUniform, type TSLNode } from '../_shared/tsl-helpers';
import type { BlendingMode } from '../../../types/blending';
import { computeScalarRangeUniforms } from '../_shared/scalar-range';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../config/constants';

export class GSplatTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /**
   * Public uniforms table, same shape as `GSplatMaterial.uniforms`.
   * Each entry is a getter/setter that forwards to the underlying
   * `UniformNode.value` (see module preamble + `proxyIUniform`).
   */
  uniforms: Record<string, THREE.IUniform>;

  /**
   * Persistent TSL leaf nodes — single source of truth for the
   * shader's primitive/vec inputs. Kept across `rebuildGraph()` so
   * mutations stay live after a defines change.
   */
  private tslNodes: {
    uSplatTex: TSLNode;
    uResolution: TSLNode;
    uPixelRatio: TSLNode;
    uFx: TSLNode;
    uFy: TSLNode;
    uTruncate: TSLNode;
    uTruncateSq: TSLNode;
    uShiftC: TSLNode;
    uInvOneMinusC: TSLNode;
    uRayIntegralFactor: TSLNode;
    uOpacity: TSLNode;
    uAbsorption: TSLNode;
    uHasElementAlpha: TSLNode;
    uProjectionMode: TSLNode;
    uInvGamma: TSLNode;
    uIntensity: TSLNode;
    uOffset: TSLNode;
    uIsOrtho: TSLNode;
    uSortedIndexSlot: TSLNode;
    uNearCull: TSLNode;
    uMaxExtentFactor: TSLNode;
    uCov2DDilation: TSLNode;
    uLabelColorMode: TSLNode;
    uLabelFilterIndex: TSLNode;
    uColormapTex?: TSLNode;
    uScalarMin?: TSLNode;
    uScalarScale?: TSLNode;
  };

  /**
   * Explicit `depthTest` / `transparent` overrides from the constructor
   * config. Every `rebuildGraph()` re-applies the factory tail's
   * MODE-DERIVED blending state (depthTest/transparent included), so an
   * override honored only once in the constructor tail would silently
   * revert on the first later rebuild (e.g. the guaranteed
   * placeholder→real `updateSplatTexture` rebuild at first commit). The
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

  constructor(materialConfig: GSplatMaterialConfig = {}) {
    super();

    const gammaValue = clampGamma(materialConfig.gamma);
    const truncate = clampTruncationRadius(
      materialConfig.truncationRadius ?? GSPLAT_DEFAULT_TRUNCATION_RADIUS
    );
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    this.tslNodes = {
      // Splat data texture node. Starts on the shared placeholder; the
      // commit's material sync rebinds the acquired pool entry's
      // texture via `updateSplatTexture` (node identity change ->
      // graph rebuild, same lifecycle as the colormap texture).
      uSplatTex: texture(getPlaceholderElementTexture()),
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uPixelRatio: uniform(1),
      uFx: uniform(500),
      uFy: uniform(500),
      uTruncate: uniform(truncate),
      uTruncateSq: uniform(truncate * truncate),
      uShiftC: uniform(shiftC),
      uInvOneMinusC: uniform(invOneMinusC),
      uRayIntegralFactor: uniform(computeRayIntegralFactor(truncate)),
      uOpacity: uniform(materialConfig.opacity ?? 1.0),
      uAbsorption: uniform(materialConfig.absorption ?? 1.0),
      uHasElementAlpha: uniform(materialConfig.hasElementAlpha ? 1 : 0),
      // Decorative in TSL (the graph JS-branches on blendingMode); kept for
      // clone/telemetry parity. Peak (1) for surface modes (max/normal/opaque).
      uProjectionMode: uniform(
        usesPeakProjection(materialConfig.blendingMode ?? 'additive') ? 1 : 0
      ),
      uInvGamma: uniform(1.0 / gammaValue),
      uIntensity: uniform(materialConfig.intensity ?? 1.0),
      uOffset: uniform(materialConfig.offset ?? 0.0),
      uIsOrtho: uniform(0),
      uSortedIndexSlot: uniform(0),
      uNearCull: uniform(0.1),
      uMaxExtentFactor: uniform(materialConfig.maxExtentFactor ?? 0.33),
      uCov2DDilation: uniform(materialConfig.cov2DDilation ?? GSPLAT_COV2D_DILATION_DEFAULT),
      uLabelColorMode: uniform(0),
      uLabelFilterIndex: uniform(0),
    };
    if (materialConfig.colormapTexture) {
      this.tslNodes.uColormapTex = texture(materialConfig.colormapTexture);
      // Midpoint identity for degenerate ranges — see scalar-range.ts.
      const sr = computeScalarRangeUniforms(
        materialConfig.scalarRange?.[0] ?? 0.0,
        materialConfig.scalarRange?.[1] ?? 1.0
      );
      this.tslNodes.uScalarMin = uniform(sr.scalarMin);
      this.tslNodes.uScalarScale = uniform(sr.scalarScale);
    }

    this.uniforms = this.buildUniformProxies();

    this.defines = materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {};
    // LUXAR_GAMMA_ONE mirrors the GLSL define; it drives the `gammaOne`
    // factory flag in `rebuildGraph` so the gamma pow() is skipped at
    // gamma == 1.0. Toggled by `updateGamma`.
    if (isGammaOne(gammaValue)) this.defines.LUXAR_GAMMA_ONE = '';
    // LUXAR_NO_GOG likewise drives the `noGOG` factory flag so the GOG
    // mul/add/clamp chain is skipped at intensity == 1 && offset == 0.
    // Toggled by `updateIntensity` / `updateOffset` (mirrors the Line
    // material).
    if (isNoGOG(materialConfig.intensity ?? 1.0, materialConfig.offset ?? 0.0)) {
      this.defines.LUXAR_NO_GOG = '';
    }
    // LUXAR_VOLUMETRIC is an INERT introspection tracker here: the TSL
    // factory derives the volumetric output branch from the MODE (the
    // rebuild predicates in applyBlendingMode own the boundary), but
    // point/line TSL and the gsplat GLSL twin all expose the mode via
    // this define — stamping it keeps cross-backend/cross-geometry
    // introspection (tests, debug tooling) uniform.
    if (isVolumetricMode(materialConfig.blendingMode ?? 'additive')) {
      this.defines.LUXAR_VOLUMETRIC = '';
    }
    this.toneMapped = false;
    this.side = THREE.DoubleSide;
    // Single-pass billboards — see the GLSL twin's forceSinglePass note.
    this.forceSinglePass = true;

    // depthTest is stamped after `rebuildGraph` below, from the
    // mode-derived state the factory tail applies.
    this.userData.gamma = gammaValue;
    this.userData.scalarRange = materialConfig.scalarRange;

    // Stamp the requested mode on userData BEFORE rebuildGraph so
    // the factory + the post-factory `applyBlendingMode` override
    // both pick it up. Without this, `userData.blendingMode` is
    // undefined and the factory defaults to 'additive'. Mirrors
    // the GLSL `GSplatMaterial` constructor body where
    // `this.applyBlendingMode(blendingMode)` runs after `super()`.
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
   * Construct the IUniform-shaped getter/setter proxies over the
   * current `tslNodes` set. Called from the constructor and from any
   * setter that adds/removes a colormap node.
   */
  private buildUniformProxies(): Record<string, THREE.IUniform> {
    const u: Record<string, THREE.IUniform> = {
      // WARNING: a direct `uniforms.uSplatTex.value = tex` write does
      // NOT rebind the sampled texture — TSL `texture()` nodes capture
      // the Texture at build time. `updateSplatTexture()` is the only
      // rebind chokepoint (fresh node + graph rebuild).
      uSplatTex: proxyIUniform(this.tslNodes.uSplatTex),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uPixelRatio: proxyIUniform(this.tslNodes.uPixelRatio),
      uFx: proxyIUniform(this.tslNodes.uFx),
      uFy: proxyIUniform(this.tslNodes.uFy),
      uTruncate: proxyIUniform(this.tslNodes.uTruncate),
      uTruncateSq: proxyIUniform(this.tslNodes.uTruncateSq),
      uShiftC: proxyIUniform(this.tslNodes.uShiftC),
      uInvOneMinusC: proxyIUniform(this.tslNodes.uInvOneMinusC),
      uRayIntegralFactor: proxyIUniform(this.tslNodes.uRayIntegralFactor),
      uOpacity: proxyIUniform(this.tslNodes.uOpacity),
      uAbsorption: proxyIUniform(this.tslNodes.uAbsorption),
      uHasElementAlpha: proxyIUniform(this.tslNodes.uHasElementAlpha),
      uProjectionMode: proxyIUniform(this.tslNodes.uProjectionMode),
      uInvGamma: proxyIUniform(this.tslNodes.uInvGamma),
      uIntensity: proxyIUniform(this.tslNodes.uIntensity),
      uOffset: proxyIUniform(this.tslNodes.uOffset),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uSortedIndexSlot: proxyIUniform(this.tslNodes.uSortedIndexSlot),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxExtentFactor: proxyIUniform(this.tslNodes.uMaxExtentFactor),
      uCov2DDilation: proxyIUniform(this.tslNodes.uCov2DDilation),
      uLabelColorMode: proxyIUniform(this.tslNodes.uLabelColorMode),
      uLabelFilterIndex: proxyIUniform(this.tslNodes.uLabelFilterIndex),
    };
    if (this.tslNodes.uColormapTex) {
      u.uColormapTex = proxyIUniform(this.tslNodes.uColormapTex);
    }
    if (this.tslNodes.uScalarMin) {
      u.uScalarMin = proxyIUniform(this.tslNodes.uScalarMin);
    }
    if (this.tslNodes.uScalarScale) {
      u.uScalarScale = proxyIUniform(this.tslNodes.uScalarScale);
    }
    return u;
  }

  /**
   * Re-run the TSL factory and attach nodes. Drives `useColormap`
   * from `defines.USE_COLORMAP` (mirrors GLSL behaviour; uniform
   * presence is not the source of truth — see PointTSLMaterial for
   * the in-depth note).
   *
   * Blending state is applied by the factory tail: `normal` gets the
   * gsplat-specific `getGSplatNormalBlendingState()` (the fragment
   * emits a real premultiplied coverage alpha in that mode — see
   * shader-tsl.ts), every other mode gets the shared
   * `getCompleteBlendingState(mode)` — now the SAME sources of truth
   * the GLSL wrapper uses, so the two backends are state-identical.
   * `AdditiveBlending`'s SrcAlpha factor is the identity under the
   * shader's alpha = 1.0 contract. No separate alpha-channel blend
   * state anywhere (it trips `gl.getError()` under WebGPURenderer's
   * WebGL2 bridge; its historical HalfFloat-overflow-guard purpose is
   * handled by the raw-scene-hdr capture sanitization).
   *
   * (`max` mode goes through the same factory path and gets
   * `CustomBlending + MaxEquation + OneFactor` straight from
   * `getCompleteBlendingState` — that's what gsplat actually
   * needs for max projection.)
   */
  private rebuildGraph(): void {
    gsplatWebGPUFactory(
      this.tslNodes as GSplatTSLNodes,
      {
        useColormap: !!this.defines && 'USE_COLORMAP' in this.defines,
        gammaOne: !!this.defines && 'LUXAR_GAMMA_ONE' in this.defines,
        noGOG: !!this.defines && 'LUXAR_NO_GOG' in this.defines,
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

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number,
    pixelRatio: number = 1
  ): void {
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    this.uniforms.uPixelRatio.value = pixelRatio;
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    const fy = computeFocalLength(fov, resolution.y, isOrtho);
    this.uniforms.uFx.value = fy;
    this.uniforms.uFy.value = fy;

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
   * Update the absorption coefficient κ (volumetric mode only — a plain
   * uniform write, no rebuild; inert in every other mode).
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
   * alpha channel (RGBA). Plain uniform write — no graph rebuild (the
   * volumetric branch gates the alpha → optical-depth mapping with a
   * runtime mix on this uniform).
   */
  updateHasElementAlpha(hasAlpha: boolean): void {
    this.uniforms.uHasElementAlpha.value = hasAlpha ? 1 : 0;
  }

  updateTruncationRadius(radius: number): void {
    radius = clampTruncationRadius(radius);
    this.uniforms.uTruncate.value = radius;
    this.uniforms.uTruncateSq.value = radius * radius;
    const shiftC = Math.exp(-0.5 * radius * radius);
    this.uniforms.uShiftC.value = shiftC;
    this.uniforms.uInvOneMinusC.value = 1.0 / (1.0 - shiftC);
    this.uniforms.uRayIntegralFactor.value = computeRayIntegralFactor(radius);
  }

  updateMaxExtentFactor(factor: number): void {
    this.uniforms.uMaxExtentFactor.value = Math.max(0.01, factor);
  }

  updateGamma(gamma: number): void {
    const safeGamma = clampGamma(gamma);
    this.userData.gamma = safeGamma;
    this.uniforms.uInvGamma.value = 1.0 / safeGamma;

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

  /** Same toggle helper as `GSplatMaterial._refreshNoGOGDefine` — see there. */
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

  /**
   * Rebind the splat data texture. TSL `texture()` captures the
   * THREE.Texture at factory time, so an identity change needs a
   * fresh node + graph rebuild (exact mirror of the colormap
   * texture lifecycle). No-op when the texture is unchanged — the
   * common per-commit case.
   */
  updateSplatTexture(tex: THREE.DataTexture | null): void {
    const current = (this.uniforms.uSplatTex?.value as THREE.Texture | null | undefined) ?? null;
    const next = tex ?? getPlaceholderElementTexture();
    if (current === next) return;
    this.tslNodes.uSplatTex = texture(next);
    this.uniforms = this.buildUniformProxies();
    this.rebuildGraph();
  }

  /** The currently bound splat data texture (mirrors getPointTexture/getLineTexture). */
  getSplatTexture(): THREE.DataTexture | null {
    return (this.uniforms.uSplatTex?.value as THREE.DataTexture | null | undefined) ?? null;
  }

  updateLabelStyle(colorByLabel: boolean, filterIndex: number): void {
    this.uniforms.uLabelColorMode.value = colorByLabel ? 1 : 0;
    this.uniforms.uLabelFilterIndex.value = Math.max(0, Math.floor(filterIndex));
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

  updateScalarRange(min: number, max: number): void {
    applyScalarRangeToMaterial(this, min, max);
  }

  /**
   * Apply a Luxar blending mode at runtime. Same source of truth as
   * the factory tail in `rebuildGraph` — `normal` via
   * `getGSplatNormalBlendingState()` (premultiplied coverage-alpha
   * output; see shader-tsl.ts), everything else via the shared
   * `getCompleteBlendingState` — so subsequent rebuilds (e.g. colormap
   * toggles) don't strand the material in a divergent state.
   *
   * State-identical with the GLSL wrapper (both draw from the same
   * two helpers). `max` gets `CustomBlending + MaxEquation + OneFactor`
   * from `getCompleteBlendingState`; the gsplat-normal state is
   * bridge-safe (symmetric alpha channel; separate alpha-equation
   * state would trip `gl.getError()` under the WebGPU→WebGL2 bridge).
   *
   * Also sets `uProjectionMode` — peak (1) for surface modes
   * (max/normal/opaque, `usesPeakProjection`), sum (0) for emissive
   * (additive/luminous/volumetric). In TSL the uniform is decorative (the graph
   * JS-branches on the mode); the crossing is what triggers
   * `rebuildGraph()` below.
   */
  applyBlendingMode(mode: BlendingMode): void {
    // A runtime mode switch takes FULL ownership of the blending state:
    // clear the constructor's explicit depthTest/transparent overrides
    // so the mode-derived state below (and every later rebuild) wins.
    // Matches the GLSL twin, where applyBlendingStateToMaterial
    // overwrites both fields unconditionally on every call.
    this._explicitDepthTest = undefined;
    this._explicitTransparent = undefined;

    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = isNormalMode(mode)
      ? getGSplatNormalBlendingState()
      : getCompleteBlendingState(mode, opacity);
    const stateChanged = applyBlendingStateToMaterial(this, state);

    if (this.uniforms.uProjectionMode) {
      // Decorative (graph JS-branches on the mode); kept for clone/telemetry
      // parity. Peak (1) for surface modes (max/normal/opaque).
      this.uniforms.uProjectionMode.value = usesPeakProjection(mode) ? 1 : 0;
    }

    this.userData.blendingMode = mode;
    this.userData.depthTest = state.depthTest;

    // Keep the INERT LUXAR_VOLUMETRIC introspection tracker in sync
    // (see the constructor note — the rebuild predicates below own the
    // actual graph boundary; this define changes nothing structurally).
    if (!this.defines) this.defines = {};
    if (isVolumetricMode(mode)) {
      this.defines.LUXAR_VOLUMETRIC = '';
    } else {
      delete this.defines.LUXAR_VOLUMETRIC;
    }

    // Two boundaries force a graph rebuild (the factory JS-conditions
    // fragments/vertex blocks on the mode):
    //   - projection sum↔peak: the cofactor / ray-integration block is
    //     emitted only in SUM mode (additive/luminous/volumetric); the
    //     SURFACE modes (max/normal/opaque) use peak. Compared through
    //     `usesPeakProjection` — the SAME predicate the factory's
    //     `surfaceMode` derivation uses — so any emissive↔surface
    //     switch (e.g. additive→opaque) rebuilds instead of keeping a
    //     stale graph.
    //   - output branch: the fragment output is a build-time JS branch
    //     with THREE shapes — normal's coverage alpha, volumetric's
    //     emission–absorption (GLSL twin: LUXAR_VOLUMETRIC), and the
    //     alpha=1.0 contract. Crossing EITHER the isNormalMode or the
    //     isVolumetricMode boundary changes the graph. Notably
    //     additive↔volumetric crosses NEITHER usesPeakProjection nor
    //     isNormalMode — without the isVolumetricMode term the switch
    //     would keep a stale alpha=1 graph (fail-first-tested).
    // Same pattern bloom / vignette toggles use.
    const projectionChanged =
      previousMode === undefined || usesPeakProjection(previousMode) !== usesPeakProjection(mode);
    const outputBranchChanged =
      previousMode === undefined ||
      isNormalMode(previousMode) !== isNormalMode(mode) ||
      isVolumetricMode(previousMode) !== isVolumetricMode(mode);
    if (projectionChanged || outputBranchChanged) {
      this.rebuildGraph();
    } else if (previousMode !== mode && stateChanged) {
      this.needsUpdate = true;
    }
  }

  clone(): this {
    const cloned = new GSplatTSLMaterial({
      opacity: this.uniforms.uOpacity.value,
      // Without this a clone silently reset a tuned κ to the 1.0 default —
      // and the layers panel clones on ANY first panel interaction.
      absorption: this.uniforms.uAbsorption.value,
      hasElementAlpha: (this.uniforms.uHasElementAlpha.value as number) === 1,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      truncationRadius: this.uniforms.uTruncate.value,
      blendingMode: this.userData.blendingMode ?? 'additive',
      transparent: this.transparent,
      depthTest: this.userData.depthTest ?? true,
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
      // Mirror of the GLSL clone fix: preserve a tuned extent factor.
      maxExtentFactor: this.uniforms.uMaxExtentFactor.value,
      cov2DDilation: this.uniforms.uCov2DDilation.value,
    });

    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
      cloned.blendEquationAlpha = this.blendEquationAlpha;
      cloned.blendSrcAlpha = this.blendSrcAlpha;
      cloned.blendDstAlpha = this.blendDstAlpha;
    }

    const splatTex = this.uniforms.uSplatTex?.value as THREE.DataTexture | null | undefined;
    if (splatTex) cloned.updateSplatTexture(splatTex);
    cloned.uniforms.uFx.value = this.uniforms.uFx.value;
    cloned.uniforms.uFy.value = this.uniforms.uFy.value;
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    cloned.uniforms.uPixelRatio.value = this.uniforms.uPixelRatio.value;
    // Camera-state uniforms ride along with the derived focal scales
    // (mirrors LineTSLMaterial.clone / the points clone fix): uIsOrtho
    // is a runtime uniform in the gsplat TSL graph, so a plain value
    // copy suffices — no rebuild needed.
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uProjectionMode.value = this.uniforms.uProjectionMode.value;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;
    cloned.updateLabelStyle(
      this.uniforms.uLabelColorMode.value === 1,
      this.uniforms.uLabelFilterIndex.value
    );
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;

    return cloned as this;
  }

  setColormapTexture(tex: THREE.DataTexture | null): void {
    if (!this.defines) this.defines = {};
    if (tex) {
      this.defines.USE_COLORMAP = '';
      // TSL `texture()` snapshots at factory time → the node identity
      // must change when the host swaps textures. Build a fresh
      // texture node and (if missing) the matching scalar uniforms.
      this.tslNodes.uColormapTex = texture(tex);
      if (!this.tslNodes.uScalarMin) {
        this.tslNodes.uScalarMin = uniform(0.0);
      }
      if (!this.tslNodes.uScalarScale) {
        this.tslNodes.uScalarScale = uniform(1.0);
      }
      this.uniforms = this.buildUniformProxies();
    } else {
      delete this.defines.USE_COLORMAP;
      this.tslNodes.uColormapTex = undefined;
      this.tslNodes.uScalarMin = undefined;
      this.tslNodes.uScalarScale = undefined;
      this.uniforms = this.buildUniformProxies();
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
