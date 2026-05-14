/**
 * Fullscreen-triangle geometry shared by every post-processing pass.
 *
 * Single-triangle covering the entire viewport with NDC positions
 * `{ (-1,-1), (3,-1), (-1,3) }`. The accompanying `uv` attribute is
 * `{ (0,0), (2,0), (0,2) }` so a fragment at the centre of the
 * screen lands at `uv = (0.5, 0.5)` — the standard
 * `vUv = position.xy * 0.5 + 0.5` mapping used by the GLSL3
 * post-processing shaders.
 *
 * Without the `uv` attribute, TSL's `uv()` reads a missing input on
 * the WebGPU/NodeMaterial path and falls back to `(0, 0)`, producing
 * a single-pixel constant output instead of a textured pass.
 * Originally the production fullscreen meshes shipped only the
 * `position` attribute because the GLSL3 vertex shaders computed
 * `vUv` from `position.xy * 0.5 + 0.5` themselves. The TSL factories
 * read the geometry's `uv` attribute directly (a single
 * `uv()` call is cheaper than a `positionLocal.xy.add(1.0).mul(0.5)`
 * chain) so the contract is now "geometry must provide uv".
 *
 * @module rendering/post-processing/fullscreen-geometry
 */

import * as THREE from 'three';

/**
 * Build the shared fullscreen-triangle geometry. Called once per
 * post-processing pass (FxaaPass, BloomChain, the mega-shader's
 * megaMesh). Caller owns disposal.
 */
export function createFullscreenTriangleGeometry(): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
  );
  // uv = (position.xy + 1) / 2 — chosen so a fragment at screen
  // centre samples uv (0.5, 0.5) and the corners map to (0,0)
  // through (1,1). This matches the GLSL3 vertex shader's
  // `vUv = position.xy * 0.5 + 0.5`.
  geo.setAttribute(
    'uv',
    new THREE.Float32BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2)
  );
  return geo;
}
