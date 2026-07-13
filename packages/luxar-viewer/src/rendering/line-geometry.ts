/**
 * Line Geometry Creation for Luxar
 *
 * Creates and updates instanced quad geometry for line rendering.
 * Per-segment attributes are packed into a single
 * `InstancedInterleavedBuffer` (shared with `InterleavedBufferAttribute`
 * views) so the WebGPU backend reports one vertex-buffer slot
 * instead of 12+. This is what lets the line material's pipeline
 * compile under Chrome's compat-mode adapter (`maxVertexBuffers=8`)
 * AND yields better cache locality on every backend (all 12 attrs
 * for one segment live in one contiguous stride).
 *
 * @module rendering/line-geometry
 */

import * as THREE from 'three';
import {
  packInterleavedAttributes,
  writeInterleavedAttribute,
  type InterleavedAttributeSpec,
} from './interleaved-attributes';

/**
 * Create the base quad geometry for line instances.
 *
 * Each line segment is rendered as a quad with 4 vertices:
 * - (-1, -1): Start, bottom edge
 * - ( 1, -1): End, bottom edge
 * - (-1,  1): Start, top edge
 * - ( 1,  1): End, top edge
 *
 * The vertex shader expands these in screen space based on line width.
 *
 * @returns THREE.BufferGeometry for instanced rendering
 */
export function createLineQuadGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Quad corners: x determines position along segment, y determines edge
  const quadCorners = new Float32Array([
    -1,
    -1, // Start, bottom
    1,
    -1, // End, bottom
    -1,
    1, // Start, top
    1,
    1, // End, top
  ]);

  // Triangle indices for the quad
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
 * Configuration for instanced lines mesh
 */
export interface InstancedLinesMeshConfig {
  /** Segment start positions (segmentCount * 3) */
  startPositions: Float32Array;
  /** Segment end positions (segmentCount * 3) */
  endPositions: Float32Array;
  /** Start colors (segmentCount * 3) */
  startColors: Float32Array;
  /** End colors (segmentCount * 3) */
  endColors: Float32Array;
  /** Start widths (segmentCount) */
  startWidths: Float32Array;
  /** End widths (segmentCount) */
  endWidths: Float32Array;
  /** Start sharpness (segmentCount) */
  startSharpness: Float32Array;
  /** End sharpness (segmentCount) */
  endSharpness: Float32Array;
  /** Segment lengths (segmentCount) */
  segmentLengths: Float32Array;
  /** Whether start was clipped (segmentCount) */
  startClipped: Uint8Array;
  /** Whether end was clipped (segmentCount) */
  endClipped: Uint8Array;
  /**
   * per-segment start/end scalar values for colormap lookup.
   * When both fields are present, `createInstancedLinesMesh` binds them
   * as instanced `aStartScalar`/`aEndScalar` attributes. The line
   * shader's `USE_COLORMAP` path requires both attributes; if only one
   * is provided the binding is skipped (fail-closed).
   */
  startScalars?: Float32Array;
  endScalars?: Float32Array;
  /** Number of segments */
  segmentCount: number;
}

/**
 * Build the per-instance attribute specs in canonical declaration
 * order. The line shader reads via `attribute('aStartPos', 'vec3')`
 * etc., so layout order within the buffer doesn't affect the shader,
 * but staying consistent across create + update keeps the stride
 * predictable and makes the in-place update path simple.
 *
 * Uint8 clipped flags are widened to Float32 here (one allocation
 * per update) so the interleaved buffer is uniformly Float32.
 */
function buildLineAttributeSpecs(meshConfig: InstancedLinesMeshConfig): InterleavedAttributeSpec[] {
  const specs: InterleavedAttributeSpec[] = [
    { name: 'aStartPos', data: meshConfig.startPositions, itemSize: 3, semantic: 'coordinate' },
    { name: 'aEndPos', data: meshConfig.endPositions, itemSize: 3, semantic: 'coordinate' },
    { name: 'aStartColor', data: meshConfig.startColors, itemSize: 3, semantic: 'color' },
    { name: 'aEndColor', data: meshConfig.endColors, itemSize: 3, semantic: 'color' },
    { name: 'aStartWidth', data: meshConfig.startWidths, itemSize: 1, semantic: 'positive_scalar' },
    { name: 'aEndWidth', data: meshConfig.endWidths, itemSize: 1, semantic: 'positive_scalar' },
    {
      name: 'aStartSharpness',
      data: meshConfig.startSharpness,
      itemSize: 1,
      semantic: 'bounded_scalar',
    },
    {
      name: 'aEndSharpness',
      data: meshConfig.endSharpness,
      itemSize: 1,
      semantic: 'bounded_scalar',
    },
    {
      name: 'aSegmentLength',
      data: meshConfig.segmentLengths,
      itemSize: 1,
      semantic: 'positive_scalar',
    },
    {
      name: 'aStartClipped',
      data: new Float32Array(meshConfig.startClipped),
      itemSize: 1,
      semantic: 'bounded_scalar',
    },
    {
      name: 'aEndClipped',
      data: new Float32Array(meshConfig.endClipped),
      itemSize: 1,
      semantic: 'bounded_scalar',
    },
  ];
  if (meshConfig.startScalars && meshConfig.endScalars) {
    specs.push({
      name: 'aStartScalar',
      data: meshConfig.startScalars,
      itemSize: 1,
      semantic: 'bounded_scalar',
    });
    specs.push({
      name: 'aEndScalar',
      data: meshConfig.endScalars,
      itemSize: 1,
      semantic: 'bounded_scalar',
    });
  }
  return specs;
}

/**
 * Compute bounding box and sphere from line segment start/end positions.
 * Uses a direct min/max pass without temporary geometry or array allocations.
 *
 * bounds are expanded conservatively by `maxWidth × 0.5` (half-width)
 * to capture the rendered footprint. Without this, frustum culling and
 * camera-framing reject thick lines whose centerline is just outside
 * the view but whose pixels are still on-screen. Width here is treated
 * as a half-width (matches the rendering spec); doubling for full
 * footprint is unnecessary.
 */
function computeLineBounds(
  geometry: THREE.InstancedBufferGeometry,
  meshConfig: InstancedLinesMeshConfig
): void {
  const box = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity)
  );
  const v = new THREE.Vector3();
  let maxWidth = 0;

  for (let i = 0; i < meshConfig.segmentCount; i++) {
    const si = i * 3;
    v.set(
      meshConfig.startPositions[si],
      meshConfig.startPositions[si + 1],
      meshConfig.startPositions[si + 2]
    );
    box.expandByPoint(v);
    v.set(
      meshConfig.endPositions[si],
      meshConfig.endPositions[si + 1],
      meshConfig.endPositions[si + 2]
    );
    box.expandByPoint(v);

    const sw = meshConfig.startWidths[i];
    const ew = meshConfig.endWidths[i];
    if (Number.isFinite(sw) && sw > maxWidth) maxWidth = sw;
    if (Number.isFinite(ew) && ew > maxWidth) maxWidth = ew;
  }

  if (meshConfig.segmentCount > 0 && maxWidth > 0) {
    // Half-width margin (widths in the codebase are half-widths per the
    // rendering spec). Slightly conservative by using the full width
    // (i.e. expand by maxWidth) so picking ray pre-cull doesn't reject
    // thick lines whose centerline is just outside the view.
    box.expandByScalar(maxWidth);
  }

  geometry.boundingBox = box;
  geometry.boundingSphere = new THREE.Sphere();
  box.getBoundingSphere(geometry.boundingSphere);
}

/**
 * Bind a fresh `InstancedInterleavedBuffer` + `InterleavedBufferAttribute`
 * views to a geometry. Used by both the create path and the size-change
 * branch of the update path.
 */
function bindInterleavedAttributes(
  geometry: THREE.InstancedBufferGeometry,
  meshConfig: InstancedLinesMeshConfig
): void {
  const specs = buildLineAttributeSpecs(meshConfig);
  const { views } = packInterleavedAttributes(specs, meshConfig.segmentCount);
  for (const spec of specs) {
    geometry.setAttribute(spec.name, views[spec.name]);
  }
}

/**
 * Create an instanced mesh for lines rendering.
 *
 * Sets up the instanced geometry with all per-segment attributes
 * interleaved into a single `InstancedInterleavedBuffer`. Three.js's
 * WebGPU backend collapses the 11 (or 13 with colormap)
 * `InterleavedBufferAttribute` views into a single vertex-buffer
 * slot — see `interleaved-attributes.ts` for rationale.
 *
 * Note: We use THREE.Mesh instead of THREE.InstancedMesh because:
 * - InstancedMesh adds instanceMatrix (mat4 = 4 attribute locations)
 * - Our shader computes positions from custom attributes, not matrices
 * - This avoids exceeding WebGL's 16 attribute location limit
 * - InstancedBufferGeometry with Mesh still uses instanced drawing
 *
 * @param meshConfig - Configuration with all segment data
 * @param material - LineMaterial to use for rendering
 * @returns THREE.Mesh with InstancedBufferGeometry ready for scene addition
 */
export function createInstancedLinesMesh(
  meshConfig: InstancedLinesMeshConfig,
  material: THREE.Material
): THREE.Mesh {
  const baseGeometry = createLineQuadGeometry();

  // Create instanced buffer geometry
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));

  // Pack all per-instance attributes into one interleaved buffer.
  bindInterleavedAttributes(geometry, meshConfig);

  // Set instance count
  geometry.instanceCount = meshConfig.segmentCount;

  // Compute bounding box from segment positions (direct min/max pass, no temp allocations)
  computeLineBounds(geometry, meshConfig);

  // Create mesh with instanced geometry
  // Using THREE.Mesh instead of THREE.InstancedMesh avoids the instanceMatrix attribute
  // which would push us over WebGL's 16 attribute location limit
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;

  return mesh;
}

/**
 * Update an existing instanced lines mesh with new segment data.
 *
 * Mirrors the pattern in `updateInstancedGSplatsMesh` (gsplat-geometry.ts):
 * - Same count: in-place writes into the shared interleaved buffer
 *   (zero GPU re-allocation; the typed array is reused).
 * - Different count: rebuild the interleaved buffer and rebind every
 *   view; force `_maxInstanceCount` cache invalidation.
 * - Colormap toggle (scalars present vs absent): treated like a
 *   size change because the spec-set changed.
 * - Always: recompute bounding box/sphere from segment positions.
 *
 * @param mesh - Existing mesh to update (must have InstancedBufferGeometry)
 * @param meshConfig - New segment data
 * @returns `true` when the interleaved buffer was REBUILT (size or
 *   spec-set change) — the caller must then evict Three's cached
 *   RenderObject (see `invalidate-render-object.ts`); `false` for the
 *   in-place write.
 */
export function updateInstancedLinesMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedLinesMeshConfig
): boolean {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  const currentCount = geometry.instanceCount;
  const hasScalars = !!(meshConfig.startScalars && meshConfig.endScalars);

  // Detect a spec-set change (colormap toggle): the geometry has
  // 'aStartScalar' iff the prior config supplied scalars.
  const hadScalars = geometry.getAttribute('aStartScalar') !== undefined;

  const rebuilt = meshConfig.segmentCount !== currentCount || hasScalars !== hadScalars;
  if (rebuilt) {
    // Size changed OR spec-set changed (colormap toggle). Rebuild
    // the interleaved buffer from scratch — it gets a fresh stride
    // (when toggling scalars on/off) and a fresh `.array` (when
    // resizing).
    //
    // If the prior buffer had scalar views the new buffer omits,
    // remove them explicitly so they don't dangle on the geometry.
    if (hadScalars && !hasScalars) {
      geometry.deleteAttribute('aStartScalar');
      geometry.deleteAttribute('aEndScalar');
    }
    bindInterleavedAttributes(geometry, meshConfig);
    geometry.instanceCount = meshConfig.segmentCount;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    // Meshes created with 0 instances cache _maxInstanceCount=0
    // and subsequent attribute replacements don't invalidate it,
    // so the renderer keeps drawing 0 instances. Mirrors the same
    // workaround in gsplat-geometry.ts.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;
  } else {
    // Same size + same spec-set: write the new data into the
    // existing interleaved buffer at the correct strided offsets.
    // The buffer object is recovered from any one view (every view
    // points at the same underlying buffer).
    const sampleView = geometry.getAttribute('aStartPos') as THREE.InterleavedBufferAttribute;
    const buffer = sampleView.data as THREE.InstancedInterleavedBuffer;
    const specs = buildLineAttributeSpecs(meshConfig);
    let offset = 0;
    for (const spec of specs) {
      // `spec.data` is typed as `Float32 | Uint16 | Uint8` at the
      // interface level, but the Lines spec builder always emits
      // `Float32Array` today (every semantic resolves to `'float32'`
      // post Float16 revert — see `interleaved-attributes.ts` module
      // header). The cast is safe as long as that contract holds; a
      // future narrowing redesign will widen the update path
      // alongside flipping the semantic defaults.
      writeInterleavedAttribute(
        buffer,
        offset,
        spec.itemSize,
        spec.data as Float32Array,
        meshConfig.segmentCount
      );
      offset += spec.itemSize;
    }
  }

  // Recompute bounding box from segment positions (direct min/max pass, no temp allocations)
  computeLineBounds(geometry, meshConfig);

  return rebuilt;
}
