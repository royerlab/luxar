/**
 * GSplat Geometry Creation for Luxar
 *
 * Creates and updates instanced quad geometry for Gaussian splat
 * rendering. Per-splat attributes are packed into a single shared
 * `InstancedInterleavedBuffer` (with `InterleavedBufferAttribute`
 * views per attribute) for symmetry with `point-geometry.ts` and
 * `line-geometry.ts`, and for better vertex-cache locality.
 *
 * @module rendering/gsplat-geometry
 */

import * as THREE from 'three';
import {
  packInterleavedAttributes,
  writeInterleavedAttribute,
  type InterleavedAttributeSpec,
} from './interleaved-attributes';

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
 * Build the per-instance attribute specs in canonical declaration
 * order. The shader reads via `attribute('aCenter', 'vec3')` etc.,
 * so the layout order within the buffer is only relevant for the
 * stride, but keeping it consistent makes the in-place update path
 * predictable.
 */
function buildGSplatAttributeSpecs(
  meshConfig: InstancedGSplatsMeshConfig
): InterleavedAttributeSpec[] {
  return [
    { name: 'aCenter', data: meshConfig.centers, itemSize: 3, semantic: 'coordinate' },
    { name: 'aCholesky01', data: meshConfig.cholesky01, itemSize: 2, semantic: 'cholesky' },
    { name: 'aCholesky23', data: meshConfig.cholesky23, itemSize: 2, semantic: 'cholesky' },
    { name: 'aCholesky45', data: meshConfig.cholesky45, itemSize: 2, semantic: 'cholesky' },
    { name: 'aAmplitude', data: meshConfig.amplitudes, itemSize: 1, semantic: 'positive_scalar' },
    { name: 'aColor', data: meshConfig.colors, itemSize: 3, semantic: 'color' },
  ];
}

/**
 * Bind a fresh `InstancedInterleavedBuffer` + per-attribute views to
 * a geometry. Used by both the create path and the size-change branch
 * of the update path.
 */
function bindInterleavedAttributes(
  geometry: THREE.InstancedBufferGeometry,
  meshConfig: InstancedGSplatsMeshConfig
): void {
  const specs = buildGSplatAttributeSpecs(meshConfig);
  const { views } = packInterleavedAttributes(specs, meshConfig.splatCount);
  for (const spec of specs) {
    geometry.setAttribute(spec.name, views[spec.name]);
  }
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
  material: THREE.Material
): THREE.Mesh {
  const baseGeometry = createGSplatQuadGeometry();

  // Create instanced buffer geometry
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));

  // Pack all per-instance attributes into one interleaved buffer.
  bindInterleavedAttributes(geometry, meshConfig);

  // Set instance count
  geometry.instanceCount = meshConfig.splatCount;

  // Compute bounding box from centers using direct min/max loop (no temp geometry allocation)
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < meshConfig.splatCount; i++) {
    v.set(meshConfig.centers[i * 3], meshConfig.centers[i * 3 + 1], meshConfig.centers[i * 3 + 2]);
    box.expandByPoint(v);
  }

  // Expand bounding box by max splat extent for correct frustum culling.
  // `material` is either GSplatMaterial or GSplatTSLMaterial; both expose
  // `uniforms.uTruncate` in identical shape.
  const maxRowNorm = computeMaxCholeskyRowNorm(meshConfig);
  const matWithUniforms = material as THREE.Material & {
    uniforms?: { uTruncate?: { value: number } };
  };
  const truncationRadius = matWithUniforms.uniforms?.uTruncate?.value ?? 3.0;
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
 * @returns `true` when the interleaved buffer was REBUILT (size change) —
 *   the caller must then evict Three's cached RenderObject (see
 *   `invalidate-render-object.ts`); `false` for the in-place write.
 */
export function updateInstancedGSplatsMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedGSplatsMeshConfig
): boolean {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

  // Update or recreate attributes based on size change
  const currentCount = geometry.instanceCount;

  const rebuilt = meshConfig.splatCount !== currentCount;
  let liveGeometry = geometry;
  if (rebuilt) {
    // Size changed: build a FRESH geometry and dispose the old one —
    // never rebind new attributes onto a rendered geometry, which
    // strands the old interleaved GPU buffer in the renderer caches
    // (freed only at GC mercy on classic WebGL; pinned FOREVER by the
    // WebGPU renderer's strong Info.memoryMap). geometry.dispose() on
    // the old object frees its buffers correctly on every backend
    // because its dispose listeners were registered when it rendered.
    // (Same pattern as the points non-pool fallback in
    // commit-points-geometry.ts.)
    const fresh = new THREE.InstancedBufferGeometry();
    fresh.index = geometry.index; // shared static quad index
    fresh.setAttribute('aQuadCorner', geometry.getAttribute('aQuadCorner'));
    bindInterleavedAttributes(fresh, meshConfig);
    fresh.instanceCount = meshConfig.splatCount;
    mesh.geometry = fresh;
    liveGeometry = fresh;
    // dispose() also deletes the shared index/aQuadCorner GPU buffers
    // registered under the old geometry; three re-uploads them for
    // `fresh` on its first render (tiny static buffers — negligible).
    geometry.dispose();
  } else {
    // Same size: write new data into the existing interleaved
    // buffer at the correct strided offsets. The buffer object is
    // recovered from any one view (every view points at the same
    // underlying buffer).
    const sampleView = geometry.getAttribute('aCenter') as THREE.InterleavedBufferAttribute;
    const buffer = sampleView.data as THREE.InstancedInterleavedBuffer;
    const specs = buildGSplatAttributeSpecs(meshConfig);
    let offset = 0;
    for (const spec of specs) {
      // `spec.data` is typed as `Float32 | Uint16 | Uint8` at the
      // interface level, but the GSplats spec builder always emits
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
        meshConfig.splatCount
      );
      offset += spec.itemSize;
    }
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
  // Either GSplatMaterial (ShaderMaterial-backed) or GSplatTSLMaterial
  // (NodeMaterial-backed) — both expose the same `uniforms.uTruncate`.
  const material = mesh.material as THREE.Material & {
    uniforms?: { uTruncate?: { value: number } };
  };
  const truncationRadius = material.uniforms?.uTruncate?.value ?? 3.0;
  box.expandByScalar(maxRowNorm * truncationRadius);

  liveGeometry.boundingBox = box;
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  liveGeometry.boundingSphere = sphere;

  return rebuilt;
}
