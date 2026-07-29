/**
 * Line Picking Material for GPU object picking.
 *
 * Renders line segments to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (aSortedIndex — the storage slot),
 *   B = brightness, A = 1.0
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
        // Line data texture — rebound by the commit's material sync
        // (shared with the visual material's pool-owned storage).
        uLineTex: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uIsOrtho: { value: 0 },
        // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
        // Flipped by the depth-sort coordinator once the inactive buffer
        // holds a whole permutation (runtime uniform: never a define — a
        // flip must not recompile the program).
        uSortedIndexSlot: { value: 0 },
        // 0.1 matches the visual line material ctor default (pre-first-
        // broadcast only; updateCameraParams overwrites with the scene value).
        uNearCull: { value: 0.1 },
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
    cloned.uniforms.uLineTex.value = this.uniforms.uLineTex.value;
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
    // wrappers (the shader floors at 1e-20). The old `> 0` gate silently
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

  /**
   * Rebind the line data texture (plain uniform update). Mirrors
   * `PointPickingMaterial.updatePointTexture`.
   */
  updateLineTexture(texture: THREE.DataTexture | null): void {
    this.uniforms.uLineTex.value = texture;
  }
}
