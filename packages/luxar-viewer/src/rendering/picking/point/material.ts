/**
 * Point Picking Material for GPU object picking.
 *
 * Renders points to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (aSortedIndex — the storage slot) low
 *   16 bits, B = brightness, A = the same elementId's high 16 bits
 *
 * Shader source-of-truth lives in `./shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import {
  computePointSizeFactor,
  computeMaxPointSize,
} from '../../materials/_shared/camera-uniforms';
import { POINT_PICK_SOURCE } from './shaders';
import { requireWebGLSources } from '../../materials/_shared/shader-source';
import {
  getElementTextureWidth,
  applyElementTextureWidthDefine,
  POINT_TEXTURE_LAYOUT,
} from '../../element-texture-layout';

// Module-load assertion: the GLSL wrapper requires the GLSL source.
// Captured once so the constructor can splice the strings into super().
const POINT_PICK_GLSL = requireWebGLSources(POINT_PICK_SOURCE);

export interface PointPickingMaterialConfig {
  nodeId: number;
  radiusScale?: number;
}

export class PointPickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: PointPickingMaterialConfig) {
    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    super({
      uniforms: {
        // Point data texture — rebound by the commit's material sync
        // (shared with the visual material's pool-owned storage).
        uPointTex: { value: null },
        pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
        maxPointSize: { value: defaultResolutionY * 0.5 },
        radiusScale: { value: config.radiusScale ?? 1.0 },
        uIsOrtho: { value: 0 },
        // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
        // Flipped by the depth-sort coordinator once the inactive buffer
        // holds a whole permutation (runtime uniform: never a define — a
        // flip must not recompile the program).
        uSortedIndexSlot: { value: 0 },
        uDensityDrop: { value: 0 },
        uNearCull: { value: 0.1 },
        uPixelRatio: { value: 1 },
        uNodeId: { value: config.nodeId },
        // Resolution needed for instanced-quad expansion (matches
        // PointMaterial). Defaults overwritten by updateCameraParams.
        uResolution: { value: new THREE.Vector2(1920, defaultResolutionY) },
      },
      vertexShader: POINT_PICK_GLSL.vertex,
      fragmentShader: POINT_PICK_GLSL.fragment,
      glslVersion: THREE.GLSL3,
      // Element-texture width, baked as a compile-time constant so the
      // per-vertex %/int-div addressing strength-reduces (see
      // element-texture-layout.ts). Pre-stamped with the session width;
      // the texture-update method re-stamps from the actually bound
      // texture (a no-op recompile-wise in the common path).
      defines: {
        [POINT_TEXTURE_LAYOUT.widthDefine]: String(getElementTextureWidth(POINT_TEXTURE_LAYOUT)),
      },
      // Picking settings: opaque, depth test, no blending
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
    });
  }

  /**
   * Clone this picking material. The inherited `Material.clone()` calls
   * the constructor with no config (throws on `config.nodeId`), so —
   * mirroring `GSplatPickingMaterial.clone()` — construct with the same
   * config and copy the runtime-tuned uniform values (camera params)
   * across explicitly.
   */
  clone(): this {
    const cloned = new PointPickingMaterial({
      nodeId: this.uniforms.uNodeId.value,
      radiusScale: this.uniforms.radiusScale.value,
    });
    // Via the rebind chokepoint so the clone's width define is stamped
    // from the texture it actually binds (not the constructor's
    // session-width pre-stamp).
    cloned.updatePointTexture(this.uniforms.uPointTex.value as THREE.DataTexture | null);
    cloned.uniforms.pointSizeFactor.value = this.uniforms.pointSizeFactor.value;
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uPixelRatio.value = this.uniforms.uPixelRatio.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    cloned.uniforms.uDensityDrop.value = this.uniforms.uDensityDrop.value;
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number,
    pixelRatio: number = 1
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined) this.uniforms.uNearCull.value = nearCull;
    this.uniforms.pointSizeFactor.value = computePointSizeFactor(fov, resolution.y, isOrtho);
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
    this.uniforms.uPixelRatio.value = pixelRatio;
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
  }

  /**
   * keep pick footprint in lock-step with the visible footprint by
   * mirroring radius scale updates. Called from the commit helpers when
   * geometry dtype scaling changes (e.g. placeholder → normalized Uint8
   * commit).
   */
  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }

  /**
   * Rebind the point data texture (plain uniform update). Mirrors
   * `GSplatPickingMaterial.updateSplatTexture`.
   */
  updatePointTexture(texture: THREE.DataTexture | null): void {
    this.uniforms.uPointTex.value = texture;
    // Re-stamp the width define from the texture actually bound
    // (bind-time authority — see applyElementTextureWidthDefine).
    applyElementTextureWidthDefine(this, POINT_TEXTURE_LAYOUT, texture);
  }
}
