/**
 * Line Picking Material for GPU object picking.
 *
 * Renders line segments to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Shader source-of-truth lives in `./shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { LINE_PICK_SOURCE } from './shaders';
import { requireWebGLSources } from '../../materials/_shared/shader-source';

// Module-load assertion: the GLSL wrapper requires the GLSL source.
const LINE_PICK_GLSL = requireWebGLSources(LINE_PICK_SOURCE);

export interface LinePickingMaterialConfig {
  nodeId: number;
}

export class LinePickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: LinePickingMaterialConfig) {
    super({
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uIsOrtho: { value: 0 },
        uNearCull: { value: 0.05 },
        uMaxLinePixelWidth: { value: 540 },
        uNodeId: { value: config.nodeId },
        // CPU-precomputed pixel-width scales — see LineMaterial.
        uPerspectiveLineScale: { value: 1.0 },
        uOrthoLineScale: { value: 1.0 },
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
      // Picking is opaque so the transparent-and-DoubleSide two-pass
      // guard never trips, but setting `forceSinglePass` explicitly
      // matches the visual material and documents intent.
      forceSinglePass: true,
    });
  }

  /**
   * Clone this picking material. The inherited `Material.clone()` calls
   * the constructor with no config (throws on `config.nodeId`), so —
   * mirroring `GSplatPickingMaterial.clone()` — construct with the same
   * nodeId and copy the runtime-tuned uniform values (camera params,
   * pixel-width scales) across explicitly.
   */
  clone(): this {
    const cloned = new LinePickingMaterial({ nodeId: this.uniforms.uNodeId.value });
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    cloned.uniforms.uPerspectiveLineScale.value = this.uniforms.uPerspectiveLineScale.value;
    cloned.uniforms.uOrthoLineScale.value = this.uniforms.uOrthoLineScale.value;
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    // Accept ANY defined value, including 0 — matching the point/gsplat
    // wrappers (the shader floors at 1e-4). The old `> 0` gate silently
    // KEPT a stale value on zero-diagonal scenes (or, with LRU-cached
    // materials, the previous dataset's nearCull), re-creating the
    // cross-geometry near-fade divergence B9c fixed.
    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
    // Precomputed pixel-width scales — see LineMaterial.updateCameraParams.
    const safeFov = Math.max(fov, 1e-4);
    if (isOrtho) {
      this.uniforms.uOrthoLineScale.value = (2.0 * resolution.y) / safeFov;
    } else {
      this.uniforms.uPerspectiveLineScale.value =
        resolution.y / Math.max(Math.tan(safeFov * 0.5), 1e-4);
    }
  }
}
