/**
 * Point Geometry Creation for Luxar
 *
 * Builds the instanced-quad geometry used by `PointMaterial` and
 * `PointPickingMaterial`. Each point is rendered as an axis-aligned
 * quad with 4 vertices; per-point data (position, radius, sharpness,
 * colour, optional scalar) is packed into a single shared
 * `InstancedInterleavedBuffer` (with `InterleavedBufferAttribute`
 * views per attribute), matching the line / gsplat shape.
 *
 * Interleaving collapses N per-attribute vertex buffers into one
 * vertex-buffer slot under WebGPU. With `maxVertexBuffers=8` (Chrome's
 * compat-mode adapter), this matters for materials with many attrs;
 * for points it's a smaller win (only 4–5 attrs) but the three
 * geometry types stay symmetric, and interleaved storage gives
 * better cache locality on both backends.
 *
 * Mirrors the line-geometry.ts / gsplat-geometry.ts pattern so the
 * three geometry types share one mental model. The mesh-based point
 * layout supports both GLSL and TSL/WebGPU materials; r184's
 * `GLSLNodeBuilder` hardcodes `gl_PointSize = 1.0` for any
 * `THREE.Points` object.
 *
 * @module rendering/point-geometry
 */

import * as THREE from 'three';
import {
  packInterleavedAttributes,
  widenToFloat32,
  type InterleavedAttributeSpec,
} from './interleaved-attributes';

/**
 * Per-instance buffer-attribute payloads for the points mesh.
 *
 * Every field has length `pointCount` (or `pointCount × itemSize`).
 * Source typed-arrays may be `Float32Array`, `Uint8Array`, or
 * `Uint16Array`; the geometry layer widens to `Float32` at pack
 * time (honouring the per-attribute `normalized` flag with the
 * appropriate divisor) so the interleaved storage is uniformly
 * Float32. The per-point material's `radiusScale` uniform continues
 * to scale the shader-visible radii exactly as before — the widening
 * preserves the [0, 1]-or-raw range that uniform expects.
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

/**
 * Pick the float divisor to preserve the shader-visible range when
 * widening a typed array. For `normalized` integer sources, WebGPU /
 * WebGL2 would have divided by the type's max — we replicate that
 * divisor at pack time so the Float32 interleaved buffer feeds the
 * shader the same [0, 1] values.
 */
function normalizationDivisor(source: THREE.TypedArray, normalized: boolean): number | undefined {
  if (!normalized) return undefined;
  if (source instanceof Uint8Array) return 255;
  if (source instanceof Uint16Array) return 65535;
  // Other types (Float32, signed) don't auto-normalise in WebGPU
  // even when the `normalized` flag is set — no divisor needed.
  return undefined;
}

/**
 * Build the per-instance attribute specs in canonical declaration
 * order. Source arrays may not be Float32; widen as needed and apply
 * the appropriate normalization divisor.
 */
function buildPointAttributeSpecs(config: InstancedPointsMeshConfig): InterleavedAttributeSpec[] {
  const specs: InterleavedAttributeSpec[] = [
    { name: 'aCenter', data: config.centers, itemSize: 3, semantic: 'coordinate' },
    {
      name: 'aRadius',
      data: widenToFloat32(
        config.radii,
        normalizationDivisor(config.radii, config.radiiNormalized)
      ),
      itemSize: 1,
      semantic: 'positive_scalar',
    },
    {
      name: 'aSharpness',
      data: widenToFloat32(
        config.sharpness,
        normalizationDivisor(config.sharpness, config.sharpnessNormalized)
      ),
      itemSize: 1,
      semantic: 'bounded_scalar',
    },
    {
      name: 'aColor',
      data: widenToFloat32(
        config.colors,
        normalizationDivisor(config.colors, config.colorsNormalized)
      ),
      itemSize: 3,
      semantic: 'color',
    },
  ];
  if (config.scalars) {
    specs.push({
      name: 'aScalar',
      data: widenToFloat32(
        config.scalars,
        normalizationDivisor(config.scalars, config.scalarsNormalized ?? false)
      ),
      itemSize: 1,
      semantic: 'bounded_scalar',
    });
  }
  return specs;
}

/**
 * Attach per-instance attributes to a points geometry built around
 * the corner-quad base.
 *
 * Packs all per-instance attributes (`aCenter`, `aRadius`,
 * `aSharpness`, `aColor`, optional `aScalar`) into one shared
 * `InstancedInterleavedBuffer`. The shader sees the same attribute
 * names and types via `attribute(...)`; the interleaving is
 * transparent.
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
  const specs = buildPointAttributeSpecs(config);
  const { views } = packInterleavedAttributes(specs, config.pointCount);
  for (const spec of specs) {
    geometry.setAttribute(spec.name, views[spec.name]);
  }
  // WebGLRenderer only issues an instanced draw for InstancedBufferGeometry
  // and uses this explicit visible-instance count. Without this, r184 falls
  // back to a single non-instanced draw of the base quad.
  geometry.instanceCount = config.pointCount;
  geometry.setDrawRange(0, 6);
}
