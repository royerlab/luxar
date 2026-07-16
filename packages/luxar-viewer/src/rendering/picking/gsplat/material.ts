/**
 * GSplat Picking Material for GPU object picking.
 *
 * Renders Gaussian splats to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Shader source-of-truth lives in `./shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { computeFocalLength } from '../../materials/_shared/camera-uniforms';
import { GSPLAT_PICK_SOURCE } from './shaders';
import { requireWebGLSources } from '../../materials/_shared/shader-source';
import { GSPLAT_COV2D_DILATION_DEFAULT } from '../../materials/gsplat/math';

// Module-load assertion: the GLSL wrapper requires the GLSL source.
const GSPLAT_PICK_GLSL = requireWebGLSources(GSPLAT_PICK_SOURCE);

export interface GSplatPickingMaterialConfig {
  nodeId: number;
}

export class GSplatPickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: GSplatPickingMaterialConfig) {
    // Tighter truncation: 1.5σ instead of 3.0σ
    const truncate = 1.5;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    // Picking always uses max-projection mode; the sum-projection
    // ray-integral factor and the uProjectionMode selector aren't
    // referenced in the picking shader body, so we don't bind them
    // here either (kept the GLSL and TSL paths symmetric — see the
    // matching omission in gsplat-picking-material-tsl.ts).

    super({
      uniforms: {
        // Splat data texture — rebound by the commit's material sync
        // (shared with the visual material's pool-owned storage).
        uSplatTex: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uFx: { value: 500 },
        uFy: { value: 500 },
        uTruncate: { value: truncate },
        uTruncateSq: { value: truncate * truncate },
        uShiftC: { value: shiftC },
        uInvOneMinusC: { value: invOneMinusC },
        uIsOrtho: { value: 0 },
        uNearCull: { value: 0.1 },
        uMaxExtentFactor: { value: 0.33 },
        uCov2DDilation: { value: GSPLAT_COV2D_DILATION_DEFAULT },
        uNodeId: { value: config.nodeId },
      },
      vertexShader: GSPLAT_PICK_GLSL.vertex,
      fragmentShader: GSPLAT_PICK_GLSL.fragment,
      glslVersion: THREE.GLSL3,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
      side: THREE.DoubleSide,
    });
  }

  /** Rebind the splat data texture (plain uniform update). */
  updateSplatTexture(texture: THREE.DataTexture | null): void {
    this.uniforms.uSplatTex.value = texture;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    const fy = computeFocalLength(fov, resolution.y, isOrtho);
    this.uniforms.uFx.value = fy;
    this.uniforms.uFy.value = fy;

    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
  }
}
