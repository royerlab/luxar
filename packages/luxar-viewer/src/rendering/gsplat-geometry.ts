/**
 * GSplat Geometry Creation for Luxar
 *
 * Creates and updates instanced quad geometry for Gaussian splat
 * rendering. Per-splat data lives in an RGBA32F **splat texture**
 * (4 texels/splat — see `./splat-texture-layout` for the layout
 * authority) sampled by the vertex shader via `texelFetch`; the only
 * per-instance attribute is `aSortedIndex` (Uint32), which maps the
 * draw slot to a storage slot so draw order can be permuted without
 * rewriting splat data (depth-sorting plan §4). Phase 1 writes
 * identity ordering; the sort worker (Phase 2+) rewrites it.
 *
 * Texture lifetime = geometry lifetime: `attachSplatStorage` registers
 * a `dispose` listener on the geometry, so every dispose site (pool
 * evictors, `pool.dispose()`, the fallback rebuild swap) frees the
 * texture with the geometry — no site-by-site bookkeeping.
 *
 * Points/Lines keep the shared `InstancedInterleavedBuffer` path
 * (symmetry is restored when/if they migrate — spec §8).
 *
 * @module rendering/gsplat-geometry
 */

import * as THREE from 'three';
import {
  SPLAT_FLOATS_PER_SPLAT,
  getSplatTextureWidth,
  splatTextureHeightForCapacity,
} from './splat-texture-layout';

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
 * The per-splat arrays the texel writer consumes. Structurally
 * satisfied by both `InstancedGSplatsMeshConfig` (non-pool fallback)
 * and the pool adapter's remapped `PackedGSplatsData`.
 */
export interface SplatTexelSource {
  /** Splat centers (count × 3). */
  centers: Float32Array;
  /** Packed Cholesky [L00, L10] (count × 2). */
  cholesky01: Float32Array;
  /** Packed Cholesky [L11, L20] (count × 2). */
  cholesky23: Float32Array;
  /** Packed Cholesky [L21, L22] (count × 2). */
  cholesky45: Float32Array;
  /** Amplitudes (count). */
  amplitudes: Float32Array;
  /** Colors RGB (count × 3). */
  colors: Float32Array;
}

/** `geometry.userData` slot carrying the splat texture. */
interface SplatStorageUserData {
  splatTexture?: THREE.DataTexture;
}

/**
 * Create the splat data texture + `aSortedIndex` attribute pair on a
 * geometry, sized for `capacity` splats. Returns the texture.
 *
 * - The texture is RGBA32F, `NearestFilter`, no mips, `flipY: false`
 *   — pure structured storage, addressed by `texelFetch` in the
 *   vertex shader (colormap-LUT precedent).
 * - The texture rides `geometry.userData.splatTexture` and is
 *   disposed BY the geometry's own `dispose` event, so texture
 *   lifetime is structurally pinned to geometry lifetime at every
 *   dispose site. Growth therefore follows the pool contract for
 *   free: release + reacquire swaps in a fresh geometry+texture pair,
 *   never an in-place reallocation (the `Info.memoryMap` strand
 *   class).
 * - `aSortedIndex` is a `Uint32Array` instanced attribute — the GL
 *   type `UNSIGNED_INT` makes three bind it via `vertexAttribIPointer`
 *   (matching the shader's `in uint`), and the WebGPU path derives its
 *   `uint32` vertex format from the array constructor.
 */
export function attachSplatStorage(
  geometry: THREE.InstancedBufferGeometry,
  capacity: number
): THREE.DataTexture {
  const sortedIndex = new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1);
  sortedIndex.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aSortedIndex', sortedIndex);

  const width = getSplatTextureWidth();
  const height = splatTextureHeightForCapacity(capacity);
  const texture = new THREE.DataTexture(
    new Float32Array(width * height * 4),
    width,
    height,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.needsUpdate = true;

  (geometry.userData as SplatStorageUserData).splatTexture = texture;
  geometry.addEventListener('dispose', () => texture.dispose());
  return texture;
}

/** The splat texture attached by `attachSplatStorage`, if any. */
export function getSplatTexture(geometry: THREE.BufferGeometry): THREE.DataTexture | null {
  return (geometry.userData as SplatStorageUserData).splatTexture ?? null;
}

/**
 * Number of splats the texture's backing store can hold (its
 * row-padded float capacity, NOT the pool bucket capacity).
 */
export function splatTexelCapacity(texture: THREE.DataTexture): number {
  const arr = texture.image.data as Float32Array;
  return Math.floor(arr.length / SPLAT_FLOATS_PER_SPLAT);
}

/**
 * Fused texel writer: one pass over the staged arrays into the
 * texture's backing store, in the 4-texel layout documented in
 * `./splat-texture-layout`. Replaces the six per-attribute strided
 * writes of the interleaved era — fewer passes over the data.
 *
 * Returns the written count, clamped to the texture's capacity
 * (capacity clamping warns once at acquire time; this clamp keeps the
 * write memory-safe if a caller slips past it).
 */
export function writeSplatTexels(
  texture: THREE.DataTexture,
  src: SplatTexelSource,
  count: number
): number {
  const arr = texture.image.data as Float32Array;
  const n = Math.min(count, Math.floor(arr.length / SPLAT_FLOATS_PER_SPLAT));
  const { centers, cholesky01, cholesky23, cholesky45, amplitudes, colors } = src;
  for (let i = 0; i < n; i++) {
    const o = i * SPLAT_FLOATS_PER_SPLAT;
    const c3 = i * 3;
    const c2 = i * 2;
    // texel 0: center.xyz, amplitude
    arr[o] = centers[c3];
    arr[o + 1] = centers[c3 + 1];
    arr[o + 2] = centers[c3 + 2];
    arr[o + 3] = amplitudes[i];
    // texel 1: cholesky01.xy, cholesky23.xy
    arr[o + 4] = cholesky01[c2];
    arr[o + 5] = cholesky01[c2 + 1];
    arr[o + 6] = cholesky23[c2];
    arr[o + 7] = cholesky23[c2 + 1];
    // texel 2: cholesky45.xy, color.rg
    arr[o + 8] = cholesky45[c2];
    arr[o + 9] = cholesky45[c2 + 1];
    arr[o + 10] = colors[c3];
    arr[o + 11] = colors[c3 + 1];
    // texel 3: color.b (rest of the texel stays zero)
    arr[o + 12] = colors[c3 + 2];
  }
  texture.needsUpdate = true;
  return n;
}

/**
 * Fill `aSortedIndex[0..count)` with identity ordering and register a
 * single collapsed prefix update range. Ranges accumulate across
 * commits while a mesh is not drawn and the WebGPU backends replay
 * them verbatim (no flush-time merge — see
 * `interleaved-attributes.ts`), so every write collapses the pending
 * set to one `[0, max-end)` range.
 */
export function writeSortedIndexIdentity(
  geometry: THREE.InstancedBufferGeometry,
  count: number
): void {
  const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, arr.length);
  for (let i = 0; i < n; i++) arr[i] = i;
  let rangeEnd = n;
  for (const range of attr.updateRanges) {
    const end = range.start + range.count;
    if (end > rangeEnd) rangeEnd = end;
  }
  attr.clearUpdateRanges();
  attr.addUpdateRange(0, rangeEnd);
  attr.needsUpdate = true;
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

  // Mesh-owned splat texture + identity ordering (exact-size — the
  // non-pool fallback carries no capacity headroom).
  const texture = attachSplatStorage(geometry, meshConfig.splatCount);
  writeSplatTexels(texture, meshConfig, meshConfig.splatCount);
  writeSortedIndexIdentity(geometry, meshConfig.splatCount);

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

  // Bind the mesh-owned texture on the material right away so a mesh
  // created WITH data renders before any commit (node factory initial
  // data, parity harness, tests). Structural: any material exposing
  // the gsplat wrappers' `updateSplatTexture` surface participates.
  (
    material as THREE.Material & {
      updateSplatTexture?: (t: THREE.DataTexture | null) => void;
    }
  ).updateSplatTexture?.(texture);

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
    // Size changed: build a FRESH geometry+texture pair and dispose the
    // old one — never rebind new storage onto a rendered geometry,
    // which strands the old GPU resources in the renderer caches
    // (freed only at GC mercy on classic WebGL; pinned FOREVER by the
    // WebGPU renderer's strong Info.memoryMap). The old splat texture
    // rides the old geometry's dispose event (attachSplatStorage), so
    // the swap frees both. (Same pattern as the points non-pool
    // fallback in commit-points-geometry.ts.)
    const fresh = new THREE.InstancedBufferGeometry();
    fresh.index = geometry.index; // shared static quad index
    fresh.setAttribute('aQuadCorner', geometry.getAttribute('aQuadCorner'));
    const texture = attachSplatStorage(fresh, meshConfig.splatCount);
    writeSplatTexels(texture, meshConfig, meshConfig.splatCount);
    writeSortedIndexIdentity(fresh, meshConfig.splatCount);
    fresh.instanceCount = meshConfig.splatCount;
    mesh.geometry = fresh;
    liveGeometry = fresh;
    // Rebind the render material to the FRESH texture (same structural
    // hook as createInstancedGSplatsMesh; the pick material rides the
    // commit path's syncGSplatMaterialWithGeometry).
    (
      mesh.material as THREE.Material & {
        updateSplatTexture?: (t: THREE.DataTexture | null) => void;
      }
    ).updateSplatTexture?.(texture);
    // dispose() also deletes the shared index/aQuadCorner GPU buffers
    // registered under the old geometry; three re-uploads them for
    // `fresh` on its first render (tiny static buffers — negligible).
    geometry.dispose();
  } else {
    // Same size: rewrite the existing texture's backing store in one
    // fused pass and refresh the identity ordering.
    const texture = getSplatTexture(geometry);
    if (!texture) {
      throw new Error(
        'updateInstancedGSplatsMesh: geometry has no splat texture — ' +
          'was it created by createInstancedGSplatsMesh/attachSplatStorage?'
      );
    }
    writeSplatTexels(texture, meshConfig, meshConfig.splatCount);
    writeSortedIndexIdentity(geometry, meshConfig.splatCount);
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
