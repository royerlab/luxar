/**
 * Line Picking Material for GPU object picking.
 *
 * Renders line segments to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Shader source-of-truth lives in `./picking-shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { materialManager } from '../material-manager';
import { LINE_PICK_SOURCE } from './picking-shaders';
import { requireWebGLSources } from '../shaders/shader-source';

// Module-load assertion: the GLSL wrapper requires the GLSL source.
const LINE_PICK_GLSL = requireWebGLSources(LINE_PICK_SOURCE);

export interface LinePickingMaterialConfig {
  nodeId: number;
}

export class LinePickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: LinePickingMaterialConfig) {
    super({
      uniforms: {
        uFOV: { value: (60 * Math.PI) / 180 },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uIsOrtho: { value: 0 },
        uNearCull: { value: 0.05 },
        uMaxLinePixelWidth: { value: 540 },
        uNodeId: { value: config.nodeId },
      },
      vertexShader: LINE_PICK_GLSL.vertex,
      fragmentShader: LINE_PICK_GLSL.fragment,
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
    this.uniforms.uFOV.value = fov;
    this.uniforms.uResolution.value.copy(resolution);
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
