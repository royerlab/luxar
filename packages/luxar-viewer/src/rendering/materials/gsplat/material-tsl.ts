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
 *   - The blending state for additive / luminous / max uses
 *     `CustomBlending` with `OneFactor` factors (not THREE's
 *     `AdditiveBlending`, which would square intensity via
 *     SrcAlpha). The shared `blending-state.ts` helper provides
 *     the right state for `max`; additive / luminous go through
 *     a GSplat-specific path below to keep the alpha-MaxEquation
 *     hack that prevents HalfFloat16 overflow.
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
import { computeFocalLength } from '../_shared/camera-uniforms';
import { computeRayIntegralFactor } from './math';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from '../../material-colormap-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  isMaxMode,
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
    const truncate = materialConfig.truncationRadius ?? 3.0;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    this.tslNodes = {
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
   * Blending state is applied by the factory tail via
   * `getCompleteBlendingState(mode)`. For the TSL path that produces
   * the right result without a GSplat-specific override: the gsplat
   * shader always outputs `alpha = 1.0`, so the difference between
   * `AdditiveBlending` (`SrcAlpha + One`) and the GLSL-side
   * `CustomBlending + OneFactor` collapses to identity — both reduce
   * to `srcColor + dstColor`. The GLSL wrapper's `CustomBlending`
   * dance is vestigial; reproducing it under WebGPURenderer's WebGL2
   * backend also triggers a `gl.getError()` flag (separate
   * `blendEquationAlpha` state propagation is not perfectly tracked
   * across the WebGPU↔WebGL2 bridge), so the cleanest path is to
   * let `getCompleteBlendingState` drive the state.
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

  updateTruncationRadius(radius: number): void {
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
   * Apply a Luxar blending mode at runtime. Drives the THREE blending
   * state through the shared `getCompleteBlendingState` helper — same
   * source of truth used by the factory tail in `rebuildGraph`, so
   * subsequent rebuilds (e.g. colormap toggles) don't strand the
   * material in a divergent state.
   *
   * Diverges from the GLSL wrapper (which uses `CustomBlending +
   * OneFactor` for additive / luminous): the gsplat shader emits
   * `alpha = 1.0` so `AdditiveBlending` (`SrcAlpha + One`) produces
   * identical pixels, and `CustomBlending` here would also trip a
   * `gl.getError()` flag under WebGPURenderer's WebGL2 backend (the
   * separate alpha-equation state propagation isn't tracked through
   * the bridge). `max` mode still gets `CustomBlending + MaxEquation
   * + OneFactor` straight from `getCompleteBlendingState`.
   *
   * Also toggles `uProjectionMode` (0 = sum, 1 = max) so the shader
   * picks the right projection branch.
   */
  applyBlendingMode(mode: BlendingMode): void {
    const previousMode = this.userData.blendingMode as BlendingMode | undefined;
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
    const state: CompleteBlendingState = getCompleteBlendingState(mode, opacity);
    const stateChanged = applyBlendingStateToMaterial(this, state);

    if (this.uniforms.uProjectionMode) {
      this.uniforms.uProjectionMode.value = isMaxMode(mode) ? 1 : 0;
    }

    this.userData.blendingMode = mode;
    this.userData.depthTest = state.depthTest;

    // Cross the sum↔max boundary? The TSL factory JS-conditionally
    // emits the cofactor / ray-integration block only in sum mode, so
    // crossing the boundary requires a graph rebuild (same pattern
    // bloom / vignette toggles use). Cheaper than computing the
    // cofactors unconditionally on every vertex in max mode.
    const projectionChanged =
      previousMode === undefined || isMaxMode(previousMode) !== isMaxMode(mode);
    if (projectionChanged) {
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
    });

    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
      cloned.blendEquationAlpha = this.blendEquationAlpha;
      cloned.blendSrcAlpha = this.blendSrcAlpha;
      cloned.blendDstAlpha = this.blendDstAlpha;
    }

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
