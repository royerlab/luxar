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
 * Mechanics identical to `PointTSLMaterial` — see that module for
 * the in-depth explanation of `uniforms` ↔ TSL-uniform-node binding
 * via `.onUpdate` (set up inside `lineWebGPUFactory`).
 *
 * @module rendering/line-material-tsl
 */

import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { lineWebGPUFactory } from './line.tsl';
import type { LineMaterialConfig } from './line-material';
import type { CameraAwareMaterial } from './camera-aware-material';
import type { ColormapAwareMaterial } from './colormap-aware-material';
import { clampGamma } from './material-uniform-helpers';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from './material-colormap-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  type CompleteBlendingState,
} from './blending-state';
import type { BlendingMode } from './material-manager';

export class LineTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, ColormapAwareMaterial
{
  /** Public uniforms table, same shape as `LineMaterial.uniforms`. */
  uniforms: Record<string, THREE.IUniform>;

  constructor(materialConfig: LineMaterialConfig = {}) {
    super();

    const gammaValue = clampGamma(materialConfig.gamma);

    this.uniforms = {
      uFOV: { value: (60 * Math.PI) / 180 },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uIsOrtho: { value: 0 },
      uOpacity: { value: materialConfig.opacity ?? 1.0 },
      uInvGamma: { value: 1.0 / gammaValue },
      uIntensity: { value: materialConfig.intensity ?? 1.0 },
      uOffset: { value: materialConfig.offset ?? 0.0 },
      uNearCull: { value: 0.05 },
      uMaxLinePixelWidth: { value: 540 },
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

    // Stamp the requested mode on userData BEFORE rebuildGraph so the
    // factory reads the correct value through
    // `userData.blendingMode` — otherwise the factory defaults to
    // 'additive', wires `premultiplyRGB=false`, and `max` mode
    // rendering is wrong. Mirrors the GLSL `LineMaterial`
    // constructor body where `this.applyBlendingMode(blendingMode)`
    // runs after `super()`.
    this.userData.blendingMode = materialConfig.blendingMode ?? 'additive';

    this.rebuildGraph();
  }

  /**
   * Re-run the TSL factory and attach the resulting nodes + blending
   * state to ourselves. `useColormap` reads `defines.USE_COLORMAP`,
   * not the IUniform presence (see PointTSLMaterial for rationale).
   */
  private rebuildGraph(): void {
    lineWebGPUFactory(
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
    this.uniforms.uFOV.value = fov;
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined && nearCull > 0) {
      this.uniforms.uNearCull.value = nearCull;
    }
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
  }

  updateOpacity(opacity: number): void {
    this.uniforms.uOpacity.value = opacity;
  }

  updateGamma(gamma: number): void {
    const safeGamma = clampGamma(gamma);
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

  applyBlendingMode(mode: BlendingMode): void {
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;
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
      blendingMode: (this.userData.blendingMode as LineMaterialConfig['blendingMode']) ?? 'additive',
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

    cloned.uniforms.uFOV.value = this.uniforms.uFOV.value;
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
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
