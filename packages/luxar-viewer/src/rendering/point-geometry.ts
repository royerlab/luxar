/**
 * Point Geometry Creation for Luxar
 *
 * Builds the instanced-quad geometry used by `PointMaterial` and
 * `PointPickingMaterial`. Each point is rendered as an axis-aligned
 * quad with 4 vertices; per-point data (position, radius, sharpness,
 * colour, optional scalar) lives in `InstancedBufferAttribute`s so a
 * single draw call expands one quad per instance.
 *
 * Mirrors the line-geometry.ts / gsplat-geometry.ts pattern so the
 * three geometry types share one mental model. Migrating away from
 * `THREE.Points` is what unblocks the TSL/WebGPU port (r184's
 * `GLSLNodeBuilder` hardcodes `gl_PointSize = 1.0` for any
 * `THREE.Points` object).
 *
 * @module rendering/point-geometry
 */

import * as THREE from 'three';

/**
 * Per-instance buffer-attribute payloads for the points mesh.
 *
 * Every field has length `pointCount`. Storage typed-array kinds are
 * deliberately permissive (Float32Array, Uint8Array with `normalized:
 * true`, etc.) so dtype-normalised data can stay packed end-to-end —
 * the per-point material consumes raw normalised values plus uniform
 * `radiusScale` / `sharpnessScale` factors.
 */
export interface InstancedPointsMeshConfig {
  /**
   * Per-instance world-space centre positions (pointCount × 3).
   * Named `centers` to match `InstancedGSplatsMeshConfig.centers` —
   * the attribute is exposed to the vertex shader as `aCenter`.
   */
  centers: Float32Array;
  /** Per-instance radii (pointCount). Float32 or normalised Uint8. */
  radii: THREE.TypedArray;
  /** Whether `radii` should be `normalized: true` on the buffer attribute. */
  radiiNormalized: boolean;
  /** Per-instance sharpness (pointCount). Float32 or normalised Uint8. */
  sharpness: THREE.TypedArray;
  /** Whether `sharpness` should be `normalized: true` on the buffer attribute. */
  sharpnessNormalized: boolean;
  /** Per-instance RGB colours (pointCount × 3). Float32 or normalised Uint8. */
  colors: THREE.TypedArray;
  /** Whether `colors` should be `normalized: true` on the buffer attribute. */
  colorsNormalized: boolean;
  /** Optional per-instance scalar for colormap mode (pointCount). */
  scalars?: THREE.TypedArray;
  /** Whether `scalars` should be `normalized: true` on the buffer attribute. */
  scalarsNormalized?: boolean;
  /** Total points in the mesh. */
  pointCount: number;
}

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
    -1, -1, // Bottom-left
    1, -1, // Bottom-right
    -1, 1, // Top-left
    1, 1, // Top-right
  ]);

  const indices = new Uint16Array([
    0, 1, 2, // First triangle
    2, 1, 3, // Second triangle
  ]);

  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  return geometry;
}

/**
 * Attach per-instance attributes to a points geometry built around
 * the corner-quad base.
 *
 * Sets `instanceCount` on the geometry to drive Three's instanced-
 * draw path. The caller must set `mesh.frustumCulled = false` because
 * Three's culling tests the base geometry bounds, not the per-instance
 * positions.
 */
export function setupInstancedPointsMesh(
  geometry: THREE.InstancedBufferGeometry,
  config: InstancedPointsMeshConfig
): void {
  geometry.setAttribute(
    'aCenter',
    new THREE.InstancedBufferAttribute(config.centers, 3)
  );
  geometry.setAttribute(
    'aRadius',
    new THREE.InstancedBufferAttribute(config.radii, 1, config.radiiNormalized)
  );
  geometry.setAttribute(
    'aSharpness',
    new THREE.InstancedBufferAttribute(config.sharpness, 1, config.sharpnessNormalized)
  );
  geometry.setAttribute(
    'aColor',
    new THREE.InstancedBufferAttribute(config.colors, 3, config.colorsNormalized)
  );
  if (config.scalars) {
    geometry.setAttribute(
      'aScalar',
      new THREE.InstancedBufferAttribute(
        config.scalars,
        1,
        config.scalarsNormalized ?? false
      )
    );
  }
  // WebGLRenderer only issues an instanced draw for InstancedBufferGeometry
  // and uses this explicit visible-instance count. Without this, r184 falls
  // back to a single non-instanced draw of the base quad.
  geometry.instanceCount = config.pointCount;
  geometry.setDrawRange(0, 6);
}
