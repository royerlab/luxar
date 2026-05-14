/**
 * TSL / NodeMaterial counterpart to `PointPickingMaterial`.
 *
 * Mirrors the GLSL `PointPickingMaterial` one-for-one — same
 * constructor signature, same `updateCameraParams` /
 * `updateRadiusScale` / `updateSharpnessScale` / `dispose` surface,
 * same `CameraAwareMaterial` contract. Per-mesh lifetime (the
 * NodeFactory creates one per points node and registers it with
 * `materialManager` for camera updates).
 *
 * @module rendering/picking/point-picking-material-tsl
 */

import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { pointPickWebGPUFactory } from './point-pick.tsl';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { computePointSizeFactor, computeMaxPointSize } from '../camera-uniforms';
import { materialManager } from '../material-manager';
import type { PointPickingMaterialConfig } from './point-picking-material';

export class PointPickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  constructor(config: PointPickingMaterialConfig) {
    super();

    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    this.uniforms = {
      pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
      maxPointSize: { value: defaultResolutionY * 0.5 },
      radiusScale: { value: config.radiusScale ?? 1.0 },
      sharpnessScale: { value: config.sharpnessScale ?? 1.0 },
      uIsOrtho: { value: 0 },
      uNodeId: { value: config.nodeId },
      uResolution: { value: new THREE.Vector2(1920, defaultResolutionY) },
    };

    this.toneMapped = false;

    pointPickWebGPUFactory(this.uniforms, this);
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    _nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    this.uniforms.pointSizeFactor.value = computePointSizeFactor(fov, resolution.y, isOrtho);
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
  }

  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }

  updateSharpnessScale(scale: number): void {
    this.uniforms.sharpnessScale.value = scale;
  }

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
