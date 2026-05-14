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
 * @module rendering/gsplat-material-tsl
 */

import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { gsplatWebGPUFactory } from './gsplat.tsl';
import type { GSplatMaterialConfig } from './gsplat-material';
import type { CameraAwareMaterial } from './camera-aware-material';
import type { ColormapAwareMaterial } from './colormap-aware-material';
import { computeFocalLength } from './camera-uniforms';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from './material-colormap-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  isMaxMode,
  type CompleteBlendingState,
} from './blending-state';
import type { BlendingMode } from './material-manager';

function computeRayIntegralFactor(truncate: number): number {
  // Abramowitz & Stegun erf approximation (max error 1.5e-7).
  const SQRT_2PI = Math.sqrt(2 * Math.PI);
  const x = truncate / Math.SQRT2;
  const t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
  const erfVal =
    1.0 -
    t *
      (0.254829592 +
        t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) *
      Math.exp(-x * x);
  const erf = x >= 0 ? erfVal : -erfVal;
  return SQRT_2PI * erf - 2 * truncate * Math.exp(-0.5 * truncate * truncate);
}

export class GSplatTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /** Public uniforms table, same shape as `GSplatMaterial.uniforms`. */
  uniforms: Record<string, THREE.IUniform>;

  constructor(materialConfig: GSplatMaterialConfig = {}) {
    super();

    const gammaValue = Math.max(0.001, materialConfig.gamma ?? 1.0);
    const truncate = materialConfig.truncationRadius ?? 3.0;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    this.uniforms = {
      uResolution: { value: new THREE.Vector2(1, 1) },
      uFx: { value: 500 },
      uFy: { value: 500 },
      uTruncate: { value: truncate },
      uTruncateSq: { value: truncate * truncate },
      uShiftC: { value: shiftC },
      uInvOneMinusC: { value: invOneMinusC },
      uRayIntegralFactor: { value: computeRayIntegralFactor(truncate) },
      uOpacity: { value: materialConfig.opacity ?? 1.0 },
      uProjectionMode: { value: materialConfig.blendingMode === 'max' ? 1 : 0 },
      uInvGamma: { value: 1.0 / gammaValue },
      uIntensity: { value: materialConfig.intensity ?? 1.0 },
      uOffset: { value: materialConfig.offset ?? 0.0 },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.1 },
      uMaxExtentFactor: { value: materialConfig.maxExtentFactor ?? 0.33 },
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
      this.uniforms,
      {
        useColormap: !!this.defines && 'USE_COLORMAP' in this.defines,
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
    const safeGamma = Math.max(0.001, gamma);
    this.userData.gamma = safeGamma;
    this.uniforms.uInvGamma.value = 1.0 / safeGamma;
  }

  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
  }

  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
  }

  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const oldTexture = (this.uniforms.uColormapTex?.value as THREE.Texture | null | undefined) ??
      null;
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
    const textureChanged = oldTexture !== texture;
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

    if (previousMode !== mode && stateChanged) {
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
