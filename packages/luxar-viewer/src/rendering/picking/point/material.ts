/**
 * Point Picking Material for GPU object picking.
 *
 * Renders points to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_VertexID), B = brightness, A = 1.0
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
        pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
        maxPointSize: { value: defaultResolutionY * 0.5 },
        radiusScale: { value: config.radiusScale ?? 1.0 },
        uIsOrtho: { value: 0 },
        uNearCull: { value: 0.1 },
        uNodeId: { value: config.nodeId },
        // Resolution needed for instanced-quad expansion (matches
        // PointMaterial). Defaults overwritten by updateCameraParams.
        uResolution: { value: new THREE.Vector2(1920, defaultResolutionY) },
      },
      vertexShader: POINT_PICK_GLSL.vertex,
      fragmentShader: POINT_PICK_GLSL.fragment,
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
    nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined) this.uniforms.uNearCull.value = nearCull;
    this.uniforms.pointSizeFactor.value = computePointSizeFactor(fov, resolution.y, isOrtho);
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
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
}
