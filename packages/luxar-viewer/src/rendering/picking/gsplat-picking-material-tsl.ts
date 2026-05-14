/**
 * TSL / NodeMaterial counterpart to `GSplatPickingMaterial`.
 *
 * Mirrors the GLSL `GSplatPickingMaterial` one-for-one — same
 * constructor signature, same `updateCameraParams` / `dispose`
 * surface, same `CameraAwareMaterial` contract. Always uses the
 * 1.5σ truncation + max-projection mode the GLSL path defaults to.
 *
 * @module rendering/picking/gsplat-picking-material-tsl
 */

import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { gsplatPickWebGPUFactory } from './gsplat-pick.tsl';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { computeFocalLength } from '../camera-uniforms';
import { materialManager } from '../material-manager';
import type { GSplatPickingMaterialConfig } from './gsplat-picking-material';

export class GSplatPickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  constructor(config: GSplatPickingMaterialConfig) {
    super();

    // Tighter truncation: 1.5σ (matches GLSL picking).
    const truncate = 1.5;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    // Abramowitz & Stegun erf approximation (max error 1.5e-7) for
    // ray-integral factor at 1.5σ.
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

    this.uniforms = {
      uResolution: { value: new THREE.Vector2(1, 1) },
      uFx: { value: 500 },
      uFy: { value: 500 },
      uTruncate: { value: truncate },
      uTruncateSq: { value: truncate * truncate },
      uShiftC: { value: shiftC },
      uInvOneMinusC: { value: invOneMinusC },
      uRayIntegralFactor: { value: rayIntegralFactor },
      uProjectionMode: { value: 1 }, // Max projection for picking.
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.1 },
      uMaxExtentFactor: { value: 0.33 },
      uNodeId: { value: config.nodeId },
    };

    this.toneMapped = false;
    this.side = THREE.DoubleSide;

    gsplatPickWebGPUFactory(this.uniforms, this);
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

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
