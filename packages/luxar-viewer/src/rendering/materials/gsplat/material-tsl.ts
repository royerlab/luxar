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
 *   - `applyBlendingMode` toggles `uProjectionMode` (0 = sum, 1 =
 *     max) in addition to the THREE blending state. The shader has
 *     separate sum vs. max branches; the uniform drives the
 *     selection at runtime.
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
import { clampGamma, isGammaOne } from '../_shared/uniform-helpers';
import { getPlaceholderSplatTexture } from '../../splat-texture-layout';
import { computeFocalLength } from '../_shared/camera-uniforms';
import { computeRayIntegralFactor, clampTruncationRadius } from './math';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  getGSplatNormalBlendingState,
  isMaxMode,
  isNormalMode,
  type CompleteBlendingState,
} from '../../blending-state';
import { proxyIUniform, type TSLNode } from '../_shared/tsl-helpers';
import type { BlendingMode } from '../../material-manager';

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
    uFx: TSLNode;
    uFy: TSLNode;
    uTruncate: TSLNode;
    uTruncateSq: TSLNode;
    uShiftC: TSLNode;
    uInvOneMinusC: TSLNode;
    uRayIntegralFactor: TSLNode;
    uOpacity: TSLNode;
    uProjectionMode: TSLNode;
    uInvGamma: TSLNode;
    uIntensity: TSLNode;
    uOffset: TSLNode;
    uIsOrtho: TSLNode;
    uNearCull: TSLNode;
    uMaxExtentFactor: TSLNode;
    uColormapTex?: TSLNode;
    uScalarMin?: TSLNode;
    uScalarScale?: TSLNode;
  };

  constructor(materialConfig: GSplatMaterialConfig = {}) {
    super();

    const gammaValue = clampGamma(materialConfig.gamma);
    const truncate = clampTruncationRadius(materialConfig.truncationRadius ?? 3.0);
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    this.tslNodes = {
      // Splat data texture node. Starts on the shared placeholder; the
      // commit's material sync rebinds the acquired pool entry's
      // texture via `updateSplatTexture` (node identity change ->
      // graph rebuild, same lifecycle as the colormap texture).
      uSplatTex: texture(getPlaceholderSplatTexture()),
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uFx: uniform(500),
      uFy: uniform(500),
      uTruncate: uniform(truncate),
      uTruncateSq: uniform(truncate * truncate),
      uShiftC: uniform(shiftC),
      uInvOneMinusC: uniform(invOneMinusC),
      uRayIntegralFactor: uniform(computeRayIntegralFactor(truncate)),
      uOpacity: uniform(materialConfig.opacity ?? 1.0),
      uProjectionMode: uniform(materialConfig.blendingMode === 'max' ? 1 : 0),
      uInvGamma: uniform(1.0 / gammaValue),
      uIntensity: uniform(materialConfig.intensity ?? 1.0),
      uOffset: uniform(materialConfig.offset ?? 0.0),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.1),
      uMaxExtentFactor: uniform(materialConfig.maxExtentFactor ?? 0.33),
    };
    if (materialConfig.colormapTexture) {
      this.tslNodes.uColormapTex = texture(materialConfig.colormapTexture);
      this.tslNodes.uScalarMin = uniform(materialConfig.scalarRange?.[0] ?? 0.0);
      this.tslNodes.uScalarScale = uniform(
        materialConfig.scalarRange
          ? 1.0 / Math.max(1e-10, materialConfig.scalarRange[1] - materialConfig.scalarRange[0])
          : 1.0
      );
    }

    this.uniforms = this.buildUniformProxies();

    this.defines = materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {};
    // LUXAR_GAMMA_ONE mirrors the GLSL define; it drives the `gammaOne`
    // factory flag in `rebuildGraph` so the gamma pow() is skipped at
    // gamma == 1.0. Toggled by `updateGamma`.
    if (isGammaOne(gammaValue)) this.defines.LUXAR_GAMMA_ONE = '';
    this.toneMapped = false;
    this.side = THREE.DoubleSide;

    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? true;
    this.userData.scalarRange = materialConfig.scalarRange;

    // Stamp the requested mode on userData BEFORE rebuildGraph so
    // the factory + the post-factory `applyBlendingMode` override
    // both pick it up. Without this, `userData.blendingMode` is
    // undefined and the factory defaults to 'additive'. Mirrors
    // the GLSL `GSplatMaterial` constructor body where
    // `this.applyBlendingMode(blendingMode)` runs after `super()`.
    this.userData.blendingMode = materialConfig.blendingMode ?? 'additive';

    this.rebuildGraph();
  }

  /**
   * Construct the IUniform-shaped getter/setter proxies over the
   * current `tslNodes` set. Called from the constructor and from any
   * setter that adds/removes a colormap node.
   */
  private buildUniformProxies(): Record<string, THREE.IUniform> {
    const u: Record<string, THREE.IUniform> = {
      uSplatTex: proxyIUniform(this.tslNodes.uSplatTex),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uFx: proxyIUniform(this.tslNodes.uFx),
      uFy: proxyIUniform(this.tslNodes.uFy),
      uTruncate: proxyIUniform(this.tslNodes.uTruncate),
      uTruncateSq: proxyIUniform(this.tslNodes.uTruncateSq),
      uShiftC: proxyIUniform(this.tslNodes.uShiftC),
      uInvOneMinusC: proxyIUniform(this.tslNodes.uInvOneMinusC),
      uRayIntegralFactor: proxyIUniform(this.tslNodes.uRayIntegralFactor),
      uOpacity: proxyIUniform(this.tslNodes.uOpacity),
      uProjectionMode: proxyIUniform(this.tslNodes.uProjectionMode),
      uInvGamma: proxyIUniform(this.tslNodes.uInvGamma),
      uIntensity: proxyIUniform(this.tslNodes.uIntensity),
      uOffset: proxyIUniform(this.tslNodes.uOffset),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxExtentFactor: proxyIUniform(this.tslNodes.uMaxExtentFactor),
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
        blendingMode: (this.userData.blendingMode as BlendingMode | undefined) ?? 'additive',
      },
      this
    );
    this.needsUpdate = true;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
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

  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
  }

  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
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
    const next = tex ?? getPlaceholderSplatTexture();
    if (current === next) return;
    this.tslNodes.uSplatTex = texture(next);
    this.uniforms = this.buildUniformProxies();
    this.rebuildGraph();
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
   * Also toggles `uProjectionMode` (0 = sum, 1 = max) so the shader
   * picks the right projection branch.
   */
  applyBlendingMode(mode: BlendingMode): void {
    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = isNormalMode(mode)
      ? getGSplatNormalBlendingState()
      : getCompleteBlendingState(mode, opacity);
    const stateChanged = applyBlendingStateToMaterial(this, state);

    if (this.uniforms.uProjectionMode) {
      this.uniforms.uProjectionMode.value = isMaxMode(mode) ? 1 : 0;
    }

    this.userData.blendingMode = mode;
    this.userData.depthTest = state.depthTest;

    // Two boundaries force a graph rebuild (the factory JS-conditions
    // fragments/vertex blocks on the mode):
    //   - sum↔max: the cofactor / ray-integration block is emitted
    //     only in sum mode (cheaper than computing it in max mode);
    //   - normal↔other: the fragment output flips between coverage
    //     alpha and the alpha=1.0 contract (GLSL twin: the
    //     LUXAR_NORMAL_PREMULT define toggle).
    // Same pattern bloom / vignette toggles use.
    const projectionChanged =
      previousMode === undefined || isMaxMode(previousMode) !== isMaxMode(mode);
    const premultChanged =
      previousMode === undefined || isNormalMode(previousMode) !== isNormalMode(mode);
    if (projectionChanged || premultChanged) {
      this.rebuildGraph();
    } else if (previousMode !== mode && stateChanged) {
      this.needsUpdate = true;
    }
  }

  clone(): this {
    const cloned = new GSplatTSLMaterial({
      opacity: this.uniforms.uOpacity.value,
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
    cloned.uniforms.uProjectionMode.value = this.uniforms.uProjectionMode.value;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;

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
    if (this.uniforms.uScalarMin) this.uniforms.uScalarMin.value = min;
    if (this.uniforms.uScalarScale) {
      this.uniforms.uScalarScale.value = 1.0 / Math.max(1e-10, max - min);
    }
  }
}
