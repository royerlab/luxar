/**
 * GSplat Geometry Creation for Luxar
 *
 * Creates and updates instanced quad geometry for Gaussian splat
 * rendering. Per-splat data lives in an RGBA32F **splat texture**
 * (4 texels/splat — see `./element-texture-layout` for the layout
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
 * The geometry-agnostic storage helpers (texture+`aSortedIndex`
 * attachment, dirty-range registration, ordering writers) live in
 * `./element-storage`, shared with the points migration; this module
 * keeps the gsplat-specific texel layout, quad creation, and the
 * non-pool mesh fallback, plus thin gsplat-bound wrappers
 * (`attachSplatStorage`/`getSplatTexture`) so gsplat call sites stay
 * small.
 *
 * @module rendering/gsplat-geometry
 */

import * as THREE from 'three';
import type { GSplatsProjectionBounds } from '../types/gsplats';
import {
  SPLAT_FLOATS_PER_SPLAT,
  SPLAT_TEXTURE_LAYOUT,
  clampSplatCapacity,
} from './element-texture-layout';
import {
  attachElementStorage,
  getElementTexture,
  registerElementTexelDirtyRange,
  writeSortedIndexIdentity,
  elementTexelCapacity,
} from './element-storage';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../config/constants';

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
  /**
   * Packed 3D Cholesky factors (splatCount * 6), row-major
   * [L00, L10, L11, L20, L21, L22] per splat — the projection output
   * layout, consumed directly (the former split into cholesky01/23/45
   * attribute pairs was an extra O(N) pass the texel writer immediately
   * re-interleaved; splat data lives in the texture, so no shader
   * attribute layout forces the split).
   */
  choleskyFactors: Float32Array;
  /** Amplitudes (splatCount) */
  amplitudes: Float32Array;
  /** Colors RGB (splatCount * 3) or RGBA (splatCount * 4) */
  colors: Float32Array;
  /** Components per color item: 3 (RGB) or 4 (RGBA). Absent means 3. */
  colorComponents?: 3 | 4;
  /** Compact 1-based categorical class index per splat; 0 means absent. */
  labelIndices?: Uint32Array | null;
  /** Number of splats */
  splatCount: number;
  /**
   * Precomputed cull metadata (AABB of `centers` + max Cholesky row
   * norm), stamped by the projection's fused scan. When present the
   * bounds/row-norm scans below are skipped (mirrors the Points
   * `metadata.bounds` pattern). Computed over the FULL projected set:
   * if the written count is capacity-clamped below it, the box is a
   * conservative superset — safe for frustum culling (never clips
   * visible splats), same trade the Points path accepts.
   */
  bounds?: GSplatsProjectionBounds;
}

/**
 * The per-splat arrays the texel writer consumes. Structurally
 * satisfied by both `InstancedGSplatsMeshConfig` (non-pool fallback)
 * and the pool adapter's remapped `PackedGSplatsData`.
 */
export interface SplatTexelSource {
  /** Splat centers (count × 3). */
  centers: Float32Array;
  /** Packed 3D Cholesky factors [L00, L10, L11, L20, L21, L22] (count × 6). */
  choleskyFactors: Float32Array;
  /** Amplitudes (count). */
  amplitudes: Float32Array;
  /** Colors RGB (count × 3) or RGBA (count × 4). */
  colors: Float32Array;
  /**
   * Components per color item: 3 (RGB) or 4 (RGBA). Absent means 3.
   * The alpha channel (per-splat opacity) is packed into texel3.y and
   * defaults to 1.0 (opaque — the per-element-opacity identity).
   */
  colorComponents?: 3 | 4;
  /** Compact 1-based categorical class index per splat; 0 means absent. */
  labelIndices?: Uint32Array | null;
}

/**
 * Gsplat-bound wrapper over `element-storage.ts::attachElementStorage`:
 * create the splat data texture + `aSortedIndex` attribute pair on a
 * geometry, sized for `capacity` splats (4 texels/splat). Returns the
 * texture. See the generic helper for the full lifetime contract.
 */
export function attachSplatStorage(
  geometry: THREE.InstancedBufferGeometry,
  capacity: number
): THREE.DataTexture {
  return attachElementStorage(geometry, capacity, SPLAT_TEXTURE_LAYOUT);
}

/**
 * The splat texture attached by `attachSplatStorage`, if any
 * (gsplat-bound wrapper over `element-storage.ts::getElementTexture`).
 */
export function getSplatTexture(geometry: THREE.BufferGeometry): THREE.DataTexture | null {
  return getElementTexture(geometry);
}

/**
 * Fused texel writer: one pass over the staged arrays into the
 * texture's backing store, in the 4-texel layout documented in
 * `./element-texture-layout`. Replaces the six per-attribute strided
 * writes of the interleaved era — fewer passes over the data.
 *
 * Returns the written count, clamped to the texture's capacity
 * (capacity clamping warns once at acquire time; this clamp keeps the
 * write memory-safe if a caller slips past it).
 *
 * `opts.fromSplat` (depth-sorting Phase 4 Stage 2, the append fast path):
 * skip writing texels `[0, fromSplat)` and register the dirty range for
 * only the `[fromSplat, n)` suffix. The prefix texels are already correct
 * on the texture's CPU mirror from the earlier commit that wrote them
 * (the commit layer only sets `fromSplat > 0` when the projected prefix
 * is provably byte-identical — same view state + prefix-identical input +
 * intact GPU buffer), so re-writing and re-uploading them is redundant.
 * Source arrays are still full-length `count`; only the loop lower bound
 * and the dirty-range start move — the length guard below is unchanged.
 */
export function writeSplatTexels(
  texture: THREE.DataTexture,
  src: SplatTexelSource,
  count: number,
  opts?: { fromSplat?: number }
): number {
  const arr = texture.image.data as Float32Array;
  const n = Math.min(count, elementTexelCapacity(texture, SPLAT_FLOATS_PER_SPLAT));
  const from = Math.max(0, Math.min(opts?.fromSplat ?? 0, n));
  const { centers, choleskyFactors, amplitudes, colors, labelIndices } = src;
  const colorK = src.colorComponents ?? 3;
  // Fail loud on source/count mismatch (the interleaved-era writer
  // threw here too) — a silent short read would write NaN texels that
  // the shaders' NaN guards then drop invisibly.
  if (
    centers.length < n * 3 ||
    choleskyFactors.length < n * 6 ||
    amplitudes.length < n ||
    colors.length < n * colorK ||
    (labelIndices !== undefined && labelIndices !== null && labelIndices.length < n)
  ) {
    throw new Error(
      `writeSplatTexels: source arrays shorter than count=${n} ` +
        `(centers=${centers.length}, choleskyFactors=${choleskyFactors.length}, ` +
        `amplitudes=${amplitudes.length}, colors=${colors.length}, ` +
        `labelIndices=${labelIndices?.length ?? 'none'})`
    );
  }
  for (let i = from; i < n; i++) {
    const o = i * SPLAT_FLOATS_PER_SPLAT;
    const p3 = i * 3;
    const ck = i * colorK;
    const c6 = i * 6;
    // texel 0: center.xyz, amplitude
    arr[o] = centers[p3];
    arr[o + 1] = centers[p3 + 1];
    arr[o + 2] = centers[p3 + 2];
    arr[o + 3] = amplitudes[i];
    // texel 1: [L00, L10], [L11, L20] (straight 6-stride copy — same
    // texel bytes as the retired cholesky01/23/45 split-then-reinterleave)
    arr[o + 4] = choleskyFactors[c6];
    arr[o + 5] = choleskyFactors[c6 + 1];
    arr[o + 6] = choleskyFactors[c6 + 2];
    arr[o + 7] = choleskyFactors[c6 + 3];
    // texel 2: [L21, L22], color.rg
    arr[o + 8] = choleskyFactors[c6 + 4];
    arr[o + 9] = choleskyFactors[c6 + 5];
    arr[o + 10] = colors[ck];
    arr[o + 11] = colors[ck + 1];
    // texel 3: color.b, alpha (per-splat opacity). Alpha is written
    // UNCONDITIONALLY — pool textures are reused, so leaving it
    // unspecified would let a previous tenant's alpha leak through.
    // 1.0 (opaque) is the per-element-opacity identity. z carries the
    // compact categorical index; 0 means no label channel. w stays unused.
    arr[o + 12] = colors[ck + 2];
    arr[o + 13] = colorK === 4 ? colors[ck + 3] : 1.0;
    arr[o + 14] = labelIndices?.[i] ?? 0;
    arr[o + 15] = 0;
  }
  // Ranged upload: only the [from, n) rows just written go to the GPU, not
  // the full capacity-sized image (pool slack rows past n never re-upload;
  // on an append, prefix rows [0, from) stay on the GPU untouched).
  registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, from, n);
  return n;
}

/**
 * Compute max Cholesky row norm across all splats (for bounding box expansion).
 * The row norms determine the maximum spatial extent of any splat, used to expand
 * the bounding box so frustum culling doesn't clip visible splats at screen edges.
 *
 * Cholesky layout: 6-stride [L00, L10, L11, L20, L21, L22] per splat.
 * Row norms: ||row0|| = |L00|, ||row1|| = sqrt(L10² + L11²),
 *            ||row2|| = sqrt(L20² + L21² + L22²).
 *
 * Fallback scan for payloads without precomputed projection bounds —
 * shared with the pool adapter (`gpu-buffer-pool/gsplats-adapter.ts`).
 *
 * `count` is the WRITTEN (capacity-clamped) splat count — never
 * the requested splat count, whose tail past the texture bound was not
 * uploaded and must not influence the cull box.
 */
export function computeMaxCholeskyRowNorm(choleskyFactors: Float32Array, count: number): number {
  let maxRowNorm = 0;
  for (let i = 0; i < count; i++) {
    const c6 = i * 6;
    const L00 = choleskyFactors[c6];
    const L10 = choleskyFactors[c6 + 1];
    const L11 = choleskyFactors[c6 + 2];
    const L20 = choleskyFactors[c6 + 3];
    const L21 = choleskyFactors[c6 + 4];
    const L22 = choleskyFactors[c6 + 5];

    const row0 = Math.abs(L00);
    const row1 = Math.sqrt(L10 * L10 + L11 * L11);
    const row2 = Math.sqrt(L20 * L20 + L21 * L21 + L22 * L22);
    maxRowNorm = Math.max(maxRowNorm, row0, row1, row2);
  }
  return maxRowNorm;
}

/**
 * Compute and assign the geometry's cull bounds (box + sphere): AABB of
 * the splat centers expanded by `maxRowNorm × truncationRadius` (the
 * conservative per-splat spatial extent).
 *
 * Uses the projection's precomputed fused-scan `bounds` when present —
 * skipping two O(N) main-thread scans per commit — and falls back to
 * the direct scans otherwise. Shared by `createInstancedGSplatsMesh`
 * and `updateInstancedGSplatsMesh`; the pool adapter applies the same
 * policy on its own payload type.
 */
function applySplatCullBounds(
  geometry: THREE.BufferGeometry,
  meshConfig: InstancedGSplatsMeshConfig,
  count: number,
  truncationRadius: number
): void {
  const box = new THREE.Box3();
  let maxRowNorm: number;
  if (meshConfig.bounds) {
    const { min, max } = meshConfig.bounds;
    box.min.set(min[0], min[1], min[2]);
    box.max.set(max[0], max[1], max[2]);
    maxRowNorm = meshConfig.bounds.maxRowNorm;
  } else {
    const v = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      v.set(
        meshConfig.centers[i * 3],
        meshConfig.centers[i * 3 + 1],
        meshConfig.centers[i * 3 + 2]
      );
      box.expandByPoint(v);
    }
    maxRowNorm = computeMaxCholeskyRowNorm(meshConfig.choleskyFactors, count);
  }
  box.expandByScalar(maxRowNorm * truncationRadius);

  geometry.boundingBox = box;
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  geometry.boundingSphere = sphere;
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
  // SEMANTIC clamp: every consumer below (storage size, texel/ordering
  // writes, instanceCount, bbox/row-norm loops) uses the same clamped
  // count, so a request above the per-node texture bound stays
  // self-consistent instead of drawing instances without texels.
  const count = clampSplatCapacity(meshConfig.splatCount);
  const baseGeometry = createGSplatQuadGeometry();

  // Create instanced buffer geometry
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));

  // Mesh-owned splat texture + identity ordering (exact-size — the
  // non-pool fallback carries no capacity headroom).
  const texture = attachSplatStorage(geometry, count);
  writeSplatTexels(texture, meshConfig, count);
  stampGSplatPresenceFlags(geometry, meshConfig);
  writeSortedIndexIdentity(geometry, count);

  // Set instance count
  geometry.instanceCount = count;

  // Cull bounds: centers AABB expanded by max splat extent (precomputed
  // projection bounds when present, direct scans otherwise).
  // `material` is either GSplatMaterial or GSplatTSLMaterial; both expose
  // `uniforms.uTruncate` in identical shape.
  const matWithUniforms = material as THREE.Material & {
    uniforms?: { uTruncate?: { value: number } };
  };
  const truncationRadius =
    matWithUniforms.uniforms?.uTruncate?.value ?? GSPLAT_DEFAULT_TRUNCATION_RADIUS;
  applySplatCullBounds(geometry, meshConfig, count, truncationRadius);

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
 * Same splat count: rewrites the existing splat texture's backing
 * store in place. Count change: swaps in a FRESH geometry+texture
 * pair and disposes the old one (never rebinds new storage onto a
 * rendered geometry — see the rebuild branch below).
 *
 * @param mesh - Existing gsplats mesh to update
 * @param meshConfig - New splat data
 * @param options - `preserveOrdering`: keep the existing `aSortedIndex`
 *   permutation on the SAME-SIZE (in-place) branch instead of resetting
 *   it to identity (same-node same-count recommit — the commit path
 *   decides; see commit-gsplats-geometry.ts). Ignored on the rebuild
 *   branch: fresh geometries are zero-filled, not identity, so the
 *   identity write is structurally required there.
 * @returns `true` when the geometry+texture pair was swapped for a
 *   fresh one (splat-count change) — the caller must then evict
 *   Three's cached RenderObject (see `invalidate-render-object.ts`);
 *   `false` for the in-place texture rewrite.
 */
export function updateInstancedGSplatsMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedGSplatsMeshConfig,
  options?: { preserveOrdering?: boolean }
): boolean {
  // SEMANTIC clamp, mirrored from createInstancedGSplatsMesh. Also
  // load-bearing for the rebuild check below: instanceCount holds the
  // CLAMPED count, so comparing against the raw request would make an
  // over-bound node take the rebuild branch on every commit forever.
  const count = clampSplatCapacity(meshConfig.splatCount);
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

  // Update or recreate attributes based on size change
  const currentCount = geometry.instanceCount;

  const rebuilt = count !== currentCount;
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
    const texture = attachSplatStorage(fresh, count);
    writeSplatTexels(texture, meshConfig, count);
    stampGSplatPresenceFlags(fresh, meshConfig);
    writeSortedIndexIdentity(fresh, count);
    fresh.instanceCount = count;
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
    // fused pass and refresh the identity ordering — unless the caller
    // vouched for the existing permutation (preserveOrdering; the
    // attribute content is unchanged, so skipping the write correctly
    // skips its update-range registration too).
    const texture = getSplatTexture(geometry);
    if (!texture) {
      throw new Error(
        'updateInstancedGSplatsMesh: geometry has no splat texture — ' +
          'was it created by createInstancedGSplatsMesh/attachSplatStorage?'
      );
    }
    writeSplatTexels(texture, meshConfig, count);
    stampGSplatPresenceFlags(geometry, meshConfig);
    if (!options?.preserveOrdering) {
      writeSortedIndexIdentity(geometry, count);
    }
  }

  // Cull bounds: centers AABB expanded by max splat extent (precomputed
  // projection bounds when present, direct scans otherwise).
  // Either GSplatMaterial (ShaderMaterial-backed) or GSplatTSLMaterial
  // (NodeMaterial-backed) — both expose the same `uniforms.uTruncate`.
  const material = mesh.material as THREE.Material & {
    uniforms?: { uTruncate?: { value: number } };
  };
  const truncationRadius = material.uniforms?.uTruncate?.value ?? GSPLAT_DEFAULT_TRUNCATION_RADIUS;
  applySplatCullBounds(liveGeometry, meshConfig, count, truncationRadius);

  return rebuilt;
}

/**
 * Stamp RGBA-alpha presence on the geometry's userData — the gsplat
 * member of the per-geometry presence-stamp trio
 * (`stampPointPresenceFlags` / `stampLinePresenceFlags`). The fixed
 * 4-texel layout always carries the texel3.y alpha slot (1.0 identity
 * for RGB data), so presence is not readable off the texture;
 * `syncGSplatMaterialWithGeometry` pushes this stamp into the render
 * material's `uHasElementAlpha` gate (volumetric w(a) map) on every
 * commit. Refreshed on EVERY write (pool geometries are reused across
 * tenants) and called AFTER the texel write, matching the point/line
 * throw-consistency ordering (`writeSplatTexels` throws only at its
 * pre-loop guard, so old texels keep their old stamp).
 */
export function stampGSplatPresenceFlags(
  geometry: THREE.BufferGeometry,
  src: { colorComponents?: 3 | 4 }
): void {
  if (!geometry.userData) geometry.userData = {};
  geometry.userData.hasElementAlpha = (src.colorComponents ?? 3) === 4;
}
