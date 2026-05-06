/**
 * Line Geometry Creation for Luxar
 *
 * Creates and updates instanced quad geometry for line rendering.
 * Extracted from line-material.ts to separate geometry from material concerns.
 *
 * @module rendering/line-geometry
 */

import * as THREE from 'three';
import type { LineMaterial } from './line-material';

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
  /** Number of segments */
  segmentCount: number;
}

/**
 * Compute bounding box and sphere from line segment start/end positions.
 * Uses a direct min/max pass without temporary geometry or array allocations.
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
  }

  geometry.boundingBox = box;
  geometry.boundingSphere = new THREE.Sphere();
  box.getBoundingSphere(geometry.boundingSphere);
}

/**
 * Create an instanced mesh for lines rendering.
 *
 * Sets up the instanced geometry with all per-segment attributes.
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
  material: LineMaterial
): THREE.Mesh {
  const baseGeometry = createLineQuadGeometry();

  // Create instanced buffer geometry
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));

  // Set instanced attributes
  geometry.setAttribute(
    'aStartPos',
    new THREE.InstancedBufferAttribute(meshConfig.startPositions, 3)
  );
  geometry.setAttribute('aEndPos', new THREE.InstancedBufferAttribute(meshConfig.endPositions, 3));
  geometry.setAttribute(
    'aStartColor',
    new THREE.InstancedBufferAttribute(meshConfig.startColors, 3)
  );
  geometry.setAttribute('aEndColor', new THREE.InstancedBufferAttribute(meshConfig.endColors, 3));
  geometry.setAttribute(
    'aStartWidth',
    new THREE.InstancedBufferAttribute(meshConfig.startWidths, 1)
  );
  geometry.setAttribute('aEndWidth', new THREE.InstancedBufferAttribute(meshConfig.endWidths, 1));
  geometry.setAttribute(
    'aStartSharpness',
    new THREE.InstancedBufferAttribute(meshConfig.startSharpness, 1)
  );
  geometry.setAttribute(
    'aEndSharpness',
    new THREE.InstancedBufferAttribute(meshConfig.endSharpness, 1)
  );
  geometry.setAttribute(
    'aSegmentLength',
    new THREE.InstancedBufferAttribute(meshConfig.segmentLengths, 1)
  );

  // Convert Uint8Array to Float32Array for clipped flags (shader expects float)
  const startClippedFloat = new Float32Array(meshConfig.startClipped);
  const endClippedFloat = new Float32Array(meshConfig.endClipped);

  geometry.setAttribute('aStartClipped', new THREE.InstancedBufferAttribute(startClippedFloat, 1));
  geometry.setAttribute('aEndClipped', new THREE.InstancedBufferAttribute(endClippedFloat, 1));

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
 * - Same count: in-place `.set()` on existing attributes (zero GPU allocation)
 * - Different count: `setAttribute` with new InstancedBufferAttribute + `_maxInstanceCount` fix
 * - Always: recompute bounding box/sphere from segment positions
 *
 * @param mesh - Existing mesh to update (must have InstancedBufferGeometry)
 * @param meshConfig - New segment data
 */
export function updateInstancedLinesMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedLinesMeshConfig
): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  const currentCount = geometry.instanceCount;

  // Attribute layout: [name, source data, components per instance]
  const attrSpecs: Array<[string, Float32Array | Uint8Array, number, boolean]> = [
    ['aStartPos', meshConfig.startPositions, 3, false],
    ['aEndPos', meshConfig.endPositions, 3, false],
    ['aStartColor', meshConfig.startColors, 3, false],
    ['aEndColor', meshConfig.endColors, 3, false],
    ['aStartWidth', meshConfig.startWidths, 1, false],
    ['aEndWidth', meshConfig.endWidths, 1, false],
    ['aStartSharpness', meshConfig.startSharpness, 1, false],
    ['aEndSharpness', meshConfig.endSharpness, 1, false],
    ['aSegmentLength', meshConfig.segmentLengths, 1, false],
    ['aStartClipped', meshConfig.startClipped, 1, true], // Uint8 → Float32
    ['aEndClipped', meshConfig.endClipped, 1, true], // Uint8 → Float32
  ];

  if (meshConfig.segmentCount !== currentCount) {
    // Size changed: recreate attributes
    for (const [name, data, size, needsFloat32Convert] of attrSpecs) {
      const arrayData = needsFloat32Convert ? new Float32Array(data) : (data as Float32Array);
      geometry.setAttribute(name, new THREE.InstancedBufferAttribute(arrayData, size));
    }
    geometry.instanceCount = meshConfig.segmentCount;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    // Same issue as gsplats: meshes created with 0 instances cache _maxInstanceCount=0.
    // (THREE.js r163+ internal property)
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;
  } else {
    // Same size: update in place (zero GPU allocation)
    for (const [name, data, , needsFloat32Convert] of attrSpecs) {
      const attr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute;
      const arrayData = needsFloat32Convert ? new Float32Array(data) : data;
      attr.set(arrayData);
      attr.needsUpdate = true;
    }
  }

  // Recompute bounding box from segment positions (direct min/max pass, no temp allocations)
  computeLineBounds(geometry, meshConfig);
}
