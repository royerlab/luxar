/**
 * Point Geometry Creation for Luxar
 *
 * Builds the instanced-quad base geometry used by `PointMaterial` and
 * `PointPickingMaterial`. Each point is rendered as an axis-aligned quad
 * with 4 corner vertices; the vertex shader expands the quad into a
 * screen-space sprite of the per-instance size (replacing `gl_PointSize`,
 * which r184's `GLSLNodeBuilder` hardcodes to `1.0` for `THREE.Points`).
 *
 * Mirrors the `line-geometry.ts` / `gsplat-geometry.ts` quad-geometry idiom
 * so the three geometry types share one vertex-shader sprite-expansion
 * model. Per-instance point attributes are packed by the gpu-buffer-pool
 * points adapter (`rendering/gpu-buffer-pool/points-adapter.ts`), not here.
 *
 * @module rendering/point-geometry
 */

import * as THREE from 'three';

/**
 * Build the base quad geometry for instanced point rendering.
 *
 * Four corner vertices in unit quad space (`(-1, -1)` ... `(1, 1)`)
 * plus a 2-triangle index list. The vertex shader expands these into
 * a screen-space sprite of the per-instance size; the sprite UV
 * (replacing the old `gl_PointCoord`) is `(aQuadCorner + 1.0) * 0.5`
 * which lands in `[0, 1]²`.
 *
 * Range and attribute name match `createLineQuadGeometry` and
 * `createGSplatQuadGeometry` exactly so the three geometry types
 * share one vertex-shader idiom for sprite expansion.
 */
export function createPointQuadGeometry(): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

  const quadCorners = new Float32Array([
    -1,
    -1, // Bottom-left
    1,
    -1, // Bottom-right
    -1,
    1, // Top-left
    1,
    1, // Top-right
  ]);

  const indices = new Uint16Array([
    0,
    1,
    2, // First triangle
    2,
    1,
    3, // Second triangle
  ]);

  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  return geometry;
}
