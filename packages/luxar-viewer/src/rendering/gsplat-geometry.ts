/**
 * GSplat Geometry Creation for Luxar
 *
 * Creates and updates instanced quad geometry for Gaussian splat rendering.
 * Extracted from gsplat-material.ts to separate geometry from material concerns.
 *
 * @module rendering/gsplat-geometry
 */

import * as THREE from 'three';
import type { GSplatMaterial } from './gsplat-material';

/**
 * Create the base quad geometry for gsplat instances.
 *
 * Each gsplat is rendered as a quad with 4 vertices:
 * - (-1, -1): Bottom-left
 * - ( 1, -1): Bottom-right
 * - (-1,  1): Top-left
 * - ( 1,  1): Top-right
 *
 * The vertex shader expands these based on the 2D covariance eigenvalues.
 *
 * @returns THREE.BufferGeometry for instanced rendering
 */
export function createGSplatQuadGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Quad corners for oriented quad expansion
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
 * Configuration for instanced gsplats mesh
 */
export interface InstancedGSplatsMeshConfig {
  /** Splat centers (splatCount * 3) */
  centers: Float32Array;
  /** Packed Cholesky01 [L00, L10] (splatCount * 2) */
  cholesky01: Float32Array;
  /** Packed Cholesky23 [L11, L20] (splatCount * 2) */
  cholesky23: Float32Array;
  /** Packed Cholesky45 [L21, L22] (splatCount * 2) */
  cholesky45: Float32Array;
  /** Amplitudes (splatCount) */
  amplitudes: Float32Array;
  /** Colors RGB (splatCount * 3) */
  colors: Float32Array;
  /** Number of splats */
  splatCount: number;
}

/**
 * Pack 3D Cholesky factors from flat array into attribute format.
 *
 * Input: choleskyFactors with 6 elements per splat [L00, L10, L11, L20, L21, L22]
 * Output: Three arrays for shader attributes:
 * - cholesky01: [L00, L10] per splat
 * - cholesky23: [L11, L20] per splat
 * - cholesky45: [L21, L22] per splat
 *
 * @param choleskyFactors - Flat array of packed Cholesky factors (N * 6)
 * @param splatCount - Number of splats
 * @returns Object with three packed arrays for shader attributes
 */
export function packCholeskyForShader(
  choleskyFactors: Float32Array,
  splatCount: number
): { cholesky01: Float32Array; cholesky23: Float32Array; cholesky45: Float32Array } {
  const cholesky01 = new Float32Array(splatCount * 2);
  const cholesky23 = new Float32Array(splatCount * 2);
  const cholesky45 = new Float32Array(splatCount * 2);

  for (let i = 0; i < splatCount; i++) {
    const srcOffset = i * 6;
    const dstOffset = i * 2;

    // L00, L10
    cholesky01[dstOffset] = choleskyFactors[srcOffset];
    cholesky01[dstOffset + 1] = choleskyFactors[srcOffset + 1];

    // L11, L20
    cholesky23[dstOffset] = choleskyFactors[srcOffset + 2];
    cholesky23[dstOffset + 1] = choleskyFactors[srcOffset + 3];

    // L21, L22
    cholesky45[dstOffset] = choleskyFactors[srcOffset + 4];
    cholesky45[dstOffset + 1] = choleskyFactors[srcOffset + 5];
  }

  return { cholesky01, cholesky23, cholesky45 };
}

/**
 * Compute max Cholesky row norm across all splats (for bounding box expansion).
 * The row norms determine the maximum spatial extent of any splat, used to expand
 * the bounding box so frustum culling doesn't clip visible splats at screen edges.
 *
 * Cholesky layout: cholesky01=[L00,L10], cholesky23=[L11,L20], cholesky45=[L21,L22]
 */
function computeMaxCholeskyRowNorm(meshConfig: InstancedGSplatsMeshConfig): number {
  let maxRowNorm = 0;
  for (let i = 0; i < meshConfig.splatCount; i++) {
    const L00 = meshConfig.cholesky01[i * 2];
    const L10 = meshConfig.cholesky01[i * 2 + 1];
    const L11 = meshConfig.cholesky23[i * 2];
    const L20 = meshConfig.cholesky23[i * 2 + 1];
    const L21 = meshConfig.cholesky45[i * 2];
    const L22 = meshConfig.cholesky45[i * 2 + 1];

    const row0 = Math.abs(L00);
    const row1 = Math.sqrt(L10 * L10 + L11 * L11);
    const row2 = Math.sqrt(L20 * L20 + L21 * L21 + L22 * L22);
    maxRowNorm = Math.max(maxRowNorm, row0, row1, row2);
  }
  return maxRowNorm;
}

/**
 * Create an instanced mesh for gsplats rendering.
 *
 * Sets up the instanced geometry with all per-splat attributes.
 *
 * Note: We use THREE.Mesh instead of THREE.InstancedMesh because:
 * - InstancedMesh adds instanceMatrix (mat4 = 4 attribute locations)
 * - Our shader computes positions from custom attributes, not matrices
 * - This avoids exceeding WebGL's 16 attribute location limit
 * - InstancedBufferGeometry with Mesh still uses instanced drawing
 *
 * @param meshConfig - Configuration with all splat data
 * @param material - GSplatMaterial to use for rendering
 * @returns THREE.Mesh with InstancedBufferGeometry ready for scene addition
 */
export function createInstancedGSplatsMesh(
  meshConfig: InstancedGSplatsMeshConfig,
  material: GSplatMaterial
): THREE.Mesh {
  const baseGeometry = createGSplatQuadGeometry();

  // Create instanced buffer geometry
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));

  // Set instanced attributes
  geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(meshConfig.centers, 3));
  geometry.setAttribute(
    'aCholesky01',
    new THREE.InstancedBufferAttribute(meshConfig.cholesky01, 2)
  );
  geometry.setAttribute(
    'aCholesky23',
    new THREE.InstancedBufferAttribute(meshConfig.cholesky23, 2)
  );
  geometry.setAttribute(
    'aCholesky45',
    new THREE.InstancedBufferAttribute(meshConfig.cholesky45, 2)
  );
  geometry.setAttribute('aAmplitude', new THREE.InstancedBufferAttribute(meshConfig.amplitudes, 1));
  geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(meshConfig.colors, 3));

  // Set instance count
  geometry.instanceCount = meshConfig.splatCount;

  // Compute bounding box from centers using direct min/max loop (no temp geometry allocation)
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < meshConfig.splatCount; i++) {
    v.set(meshConfig.centers[i * 3], meshConfig.centers[i * 3 + 1], meshConfig.centers[i * 3 + 2]);
    box.expandByPoint(v);
  }

  // Expand bounding box by max splat extent for correct frustum culling
  const maxRowNorm = computeMaxCholeskyRowNorm(meshConfig);
  const truncationRadius = material.uniforms.uTruncate.value;
  box.expandByScalar(maxRowNorm * truncationRadius);

  geometry.boundingBox = box;
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  geometry.boundingSphere = sphere;

  // Create mesh with instanced geometry
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;

  return mesh;
}

/**
 * Update an existing gsplats mesh with new data.
 *
 * This efficiently updates the instanced attributes without recreating the geometry.
 *
 * @param mesh - Existing gsplats mesh to update
 * @param meshConfig - New splat data
 */
export function updateInstancedGSplatsMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedGSplatsMeshConfig
): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

  // Update or recreate attributes based on size change
  const currentCount = geometry.instanceCount;

  if (meshConfig.splatCount !== currentCount) {
    // Size changed, recreate attributes
    geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(meshConfig.centers, 3));
    geometry.setAttribute(
      'aCholesky01',
      new THREE.InstancedBufferAttribute(meshConfig.cholesky01, 2)
    );
    geometry.setAttribute(
      'aCholesky23',
      new THREE.InstancedBufferAttribute(meshConfig.cholesky23, 2)
    );
    geometry.setAttribute(
      'aCholesky45',
      new THREE.InstancedBufferAttribute(meshConfig.cholesky45, 2)
    );
    geometry.setAttribute(
      'aAmplitude',
      new THREE.InstancedBufferAttribute(meshConfig.amplitudes, 1)
    );
    geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(meshConfig.colors, 3));
    geometry.instanceCount = meshConfig.splatCount;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount from the new attributes.
    // When a mesh is initially created with 0 instances (e.g., a gsplat node not at the
    // current time slice), THREE.js caches _maxInstanceCount=0. Later updates that add
    // instances via setAttribute won't trigger recalculation, so the renderer still draws
    // min(instanceCount, 0) = 0 instances. Deleting the cached value forces recalculation
    // on the next render frame.

    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;
  } else {
    // Same size, update in place
    const centerAttr = geometry.getAttribute('aCenter') as THREE.InstancedBufferAttribute;
    const chol01Attr = geometry.getAttribute('aCholesky01') as THREE.InstancedBufferAttribute;
    const chol23Attr = geometry.getAttribute('aCholesky23') as THREE.InstancedBufferAttribute;
    const chol45Attr = geometry.getAttribute('aCholesky45') as THREE.InstancedBufferAttribute;
    const ampAttr = geometry.getAttribute('aAmplitude') as THREE.InstancedBufferAttribute;
    const colorAttr = geometry.getAttribute('aColor') as THREE.InstancedBufferAttribute;

    centerAttr.set(meshConfig.centers);
    chol01Attr.set(meshConfig.cholesky01);
    chol23Attr.set(meshConfig.cholesky23);
    chol45Attr.set(meshConfig.cholesky45);
    ampAttr.set(meshConfig.amplitudes);
    colorAttr.set(meshConfig.colors);

    centerAttr.needsUpdate = true;
    chol01Attr.needsUpdate = true;
    chol23Attr.needsUpdate = true;
    chol45Attr.needsUpdate = true;
    ampAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
  }

  // Update bounding box from centers (direct loop, no temp geometry allocation)
  const box = new THREE.Box3();
  const _v = new THREE.Vector3();
  for (let i = 0; i < meshConfig.splatCount; i++) {
    _v.set(meshConfig.centers[i * 3], meshConfig.centers[i * 3 + 1], meshConfig.centers[i * 3 + 2]);
    box.expandByPoint(_v);
  }

  // Expand by max splat extent (Cholesky row norm × truncation radius)
  const maxRowNorm = computeMaxCholeskyRowNorm(meshConfig);
  const material = mesh.material as GSplatMaterial;
  const truncationRadius = material.uniforms.uTruncate?.value ?? 3.0;
  box.expandByScalar(maxRowNorm * truncationRadius);

  geometry.boundingBox = box;
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  geometry.boundingSphere = sphere;
}
