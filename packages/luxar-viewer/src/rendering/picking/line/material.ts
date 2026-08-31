/**
 * Line Picking Material for GPU object picking.
 *
 * Renders line segments to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (aSortedIndex — the storage slot) low
 *   16 bits, B = brightness, A = the same elementId's high 16 bits
 *
 * Shader source-of-truth lives in `./shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { LINE_PICK_SOURCE } from './shaders';
import { CAPSULE_LINE_PICK_SOURCE } from './shaders-capsule';
import { requireWebGLSources } from '../../materials/_shared/shader-source';
import {
  getElementTextureWidth,
  applyElementTextureWidthDefine,
  LINE_TEXTURE_LAYOUT,
} from '../../element-texture-layout';
import { resolveLineJoin, type LineJoinStyle } from '../../../types/line-join';
import { resolveLinePrimitive, type LinePrimitive } from '../../../types/line-primitive';

// Module-load assertion: the GLSL wrapper requires the GLSL sources.
const LINE_PICK_GLSL = requireWebGLSources(LINE_PICK_SOURCE);
const LINE_PICK_CAPSULE_GLSL = requireWebGLSources(CAPSULE_LINE_PICK_SOURCE);

export interface LinePickingMaterialConfig {
  nodeId: number;
  /**
   * Join style at degree-2 polyline joints (#790). MUST be resolved from the
   * same authored attribute the visual material gets: the pick pass builds the
   * same screen-space quad, so a divergence here makes a mitred corner
   * unpickable. Omitted ⇒ the session default, exactly like the visual
   * material's own omitted-config path.
   */
  join?: LineJoinStyle;
  /**
   * Line rendering primitive (#1352). Production passes the visual
   * material's per-node resolution (`createLinesNode` shares one resolved
   * value; the node-factory retro pass recovers it via
   * `linePrimitiveFromVisual`), so the pick footprint always rasterizes
   * the same stencil the eye sees. Omitted ⇒ the session-wide resolution
   * (override > forced policy > default). Explicit values also serve
   * harnesses (the parity page never runs bootstrap). BUILD-time,
   * exactly like the visual material.
   */
  primitive?: LinePrimitive;
}

export class LinePickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: LinePickingMaterialConfig) {
    // The primitive picks the shader-source pair (visual-material parity).
    const primitive = resolveLinePrimitive(config.primitive);
    const glsl = primitive === 'capsule' ? LINE_PICK_CAPSULE_GLSL : LINE_PICK_GLSL;
    super({
      uniforms: {
        // Line data texture — rebound by the commit's material sync
        // (shared with the visual material's pool-owned storage).
        uLineTex: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uPixelRatio: { value: 1 },
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
        // Join style (#790): 0 none, 1 miter — see types/line-join.ts for the
        // override precedence. Resolved the same way LineMaterial resolves it.
        uLineJoin: { value: resolveLineJoin(config.join) },
      },
      vertexShader: glsl.vertex,
      fragmentShader: glsl.fragment,
      glslVersion: THREE.GLSL3,
      // Element-texture width, baked as a compile-time constant so the
      // per-vertex %/int-div addressing strength-reduces (see
      // element-texture-layout.ts). Pre-stamped with the session width;
      // the texture-update method re-stamps from the actually bound
      // texture (a no-op recompile-wise in the common path).
      defines: {
        [LINE_TEXTURE_LAYOUT.widthDefine]: String(getElementTextureWidth(LINE_TEXTURE_LAYOUT)),
      },
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
    // Which primitive this program was built for — clone() re-passes it
    // (visual-material parity; already resolved, so the clone cannot drift
    // if the session override were somehow re-installed).
    this.userData.linePrimitive = primitive;
  }

  /**
   * Clone this picking material. The inherited `Material.clone()` calls
   * the constructor with no config (throws on `config.nodeId`), so —
   * mirroring `GSplatPickingMaterial.clone()` — construct with the same
   * nodeId and copy the runtime-tuned uniform values (camera params,
   * pixel-width scales) across explicitly.
   */
  clone(): this {
    const cloned = new LinePickingMaterial({
      nodeId: this.uniforms.uNodeId.value,
      primitive: this.userData.linePrimitive as LinePrimitive | undefined,
    });
    // Via the rebind chokepoint so the clone's width define is stamped
    // from the texture it actually binds (not the constructor's
    // session-width pre-stamp).
    cloned.updateLineTexture(this.uniforms.uLineTex.value as THREE.DataTexture | null);
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uPixelRatio.value = this.uniforms.uPixelRatio.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    cloned.uniforms.uPerspectiveLineScale.value = this.uniforms.uPerspectiveLineScale.value;
    cloned.uniforms.uOrthoLineScale.value = this.uniforms.uOrthoLineScale.value;
    cloned.uniforms.uLineJoin.value = this.uniforms.uLineJoin.value;
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number,
    pixelRatio: number = 1
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
    this.uniforms.uPixelRatio.value = pixelRatio;
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
    // Re-stamp the width define from the texture actually bound
    // (bind-time authority — see applyElementTextureWidthDefine).
    applyElementTextureWidthDefine(this, LINE_TEXTURE_LAYOUT, texture);
  }
}
