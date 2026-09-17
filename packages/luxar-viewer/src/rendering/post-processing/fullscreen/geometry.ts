/**
 * Fullscreen-triangle geometry shared by every post-processing pass.
 *
 * Single-triangle covering the entire viewport with NDC positions
 * `{ (-1,-1), (3,-1), (-1,3) }`. The accompanying `uv` attribute makes
 * a passthrough sample `texture(src, uv)` return the pixel of `src`
 * that lies under the fragment — independently of which backend is
 * rendering. This abstracts away the WebGL2/WebGPU framebuffer-Y
 * convention difference: WebGL2 FBOs store row 0 at the bottom of the
 * viewport, real WebGPU framebuffers store row 0 at the top. Both
 * backends are handled by swapping the V coordinate in this geometry,
 * so every downstream pass (mega, bloom, FXAA) and both shader paths
 * (GLSL3 vUv = uv;, TSL `uv()`) consume the same contract.
 *
 * Without the `uv` attribute, TSL's `uv()` would read a missing input
 * and fall back to `(0, 0)`. Originally the GLSL3 vertex shaders
 * recomputed `vUv = position.xy * 0.5 + 0.5` themselves; that worked
 * under WebGL2 only — under real WebGPU the resulting sample reads the
 * wrong row of the source target. The GLSL3 shaders now also read this
 * attribute (via `vUv = uv;`) so the correction lives in one place.
 *
 * @module rendering/post-processing/fullscreen/geometry
 */

import * as THREE from 'three';

import type { RendererCapabilities } from '../../renderer-capabilities';

/**
 * Build the shared fullscreen-triangle geometry. Called once per
 * post-processing pass (FxaaPass, BloomChain, the mega-shader's
 * megaMesh, the FullscreenPass class). Caller owns disposal.
 *
 * The UV attribute is chosen to match `caps.framebufferYDown`:
 *
 * - `framebufferYDown = true` — any `WebGPURenderer` (real WebGPU OR
 *   WebGPURenderer running on its WebGL2 compat backend, including
 *   `?webgpuForceWebgl`). Three.js's WebGPURenderer normalises Y
 *   internally so it presents a top-down framebuffer on both
 *   backends. Vertex (-1,-1) sits at screen-bottom-left and needs
 *   `uv.y = 1` to sample the bottom row of the source target.
 *   Per-corner UVs `{(0,1), (2,1), (0,-1)}` (V-inverted relative to
 *   the WebGL case).
 *
 * - `framebufferYDown = false` — the `THREE.WebGLRenderer` GLSL path
 *   only. Row 0 of the source target is at the bottom of
 *   the viewport. Vertex (-1,-1) needs `uv.y = 0` to sample the
 *   bottom row. Per-corner UVs `{(0,0), (2,0), (0,2)}`.
 *
 * Either way, the per-fragment interpolated `vUv` resolves to the
 * canvas-relative UV (a fragment at the screen centre lands at uv
 * `(0.5, 0.5)`).
 */
export function createFullscreenTriangleGeometry(caps: RendererCapabilities): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
  );
  const uvData = caps.framebufferYDown
    ? new Float32Array([0, 1, 2, 1, 0, -1])
    : new Float32Array([0, 0, 2, 0, 0, 2]);
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvData, 2));
  return geo;
}
