/**
 * TSL / NodeMaterial counterpart to `LinePickingMaterial`.
 *
 * Mirrors the GLSL `LinePickingMaterial` one-for-one — same
 * constructor signature, same `updateCameraParams` / `dispose`
 * surface, same `CameraAwareMaterial` contract.
 *
 * @module rendering/picking/line-picking-material-tsl
 */

import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { linePickWebGPUFactory } from './line-pick.tsl';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { materialManager } from '../material-manager';
import type { LinePickingMaterialConfig } from './line-picking-material';

export class LinePickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  constructor(config: LinePickingMaterialConfig) {
    super();

    this.uniforms = {
      uFOV: { value: (60 * Math.PI) / 180 },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.05 },
      uMaxLinePixelWidth: { value: 540 },
      uNodeId: { value: config.nodeId },
    };

    this.toneMapped = false;
    this.side = THREE.DoubleSide;

    linePickWebGPUFactory(this.uniforms, this);
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

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
