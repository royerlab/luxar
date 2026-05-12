/**
 * Point Picking Material for GPU object picking.
 *
 * Renders points to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_VertexID), B = brightness, A = 1.0
 *
 * Shader source-of-truth lives in `./picking-shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { computePointSizeFactor, computeMaxPointSize } from '../camera-uniforms';
import { materialManager } from '../material-manager';
import { POINT_PICK_SOURCE } from './picking-shaders';

export interface PointPickingMaterialConfig {
  nodeId: number;
  radiusScale?: number;
  sharpnessScale?: number;
}

export class PointPickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: PointPickingMaterialConfig) {
    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    super({
      uniforms: {
        pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
        maxPointSize: { value: defaultResolutionY * 0.5 },
        radiusScale: { value: config.radiusScale ?? 1.0 },
        sharpnessScale: { value: config.sharpnessScale ?? 1.0 },
        uIsOrtho: { value: 0 },
        uNodeId: { value: config.nodeId },
      },
      vertexShader: POINT_PICK_SOURCE.webgl.vertex,
      fragmentShader: POINT_PICK_SOURCE.webgl.fragment,
      glslVersion: THREE.GLSL3,
      // Picking settings: opaque, depth test, no blending
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
    });
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
  }

  /**
   * keep pick footprint in lock-step with the visible footprint by
   * mirroring radius/sharpness scale updates. Called from the commit
   * helpers when geometry dtype scaling changes (e.g. placeholder →
   * normalized Uint8 commit).
   */
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
