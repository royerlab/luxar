/**
 * GSplat Picking Material for GPU object picking.
 *
 * Renders Gaussian splats to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Shader source-of-truth lives in `./picking-shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { computeFocalLength } from '../camera-uniforms';
import { materialManager } from '../material-manager';
import { GSPLAT_PICK_SOURCE } from './picking-shaders';
import { requireWebGLSources } from '../shaders/shader-source';

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

    // Compute ray integral factor for 1.5σ truncation
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
    const rayIntegralFactor = SQRT_2PI * erf - 2 * truncate * Math.exp(-0.5 * truncate * truncate);

    super({
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uFx: { value: 500 },
        uFy: { value: 500 },
        uTruncate: { value: truncate },
        uTruncateSq: { value: truncate * truncate },
        uShiftC: { value: shiftC },
        uInvOneMinusC: { value: invOneMinusC },
        uRayIntegralFactor: { value: rayIntegralFactor },
        uProjectionMode: { value: 1 }, // Max projection for picking
        uIsOrtho: { value: 0 },
        uNearCull: { value: 0.1 },
        uMaxExtentFactor: { value: 0.33 },
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

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
