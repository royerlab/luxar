/**
 * Line Geometry Creation for Luxar
 *
 * Builds the instanced-quad base geometry used by `LineMaterial` and
 * `LinePickingMaterial`, plus the texture-backed per-segment storage
 * (depth-sorting Phase 4 lines migration). Each line segment is rendered
 * as a quad expanded in screen space by the vertex shader.
 *
 * Per-segment data lives in an RGBA32F **line texture** (6
 * texels/segment — see `./element-texture-layout` for the layout
 * authority) sampled by the vertex shader via `texelFetch`; the only
 * per-instance attribute is `aSortedIndex` (Uint32), which maps the draw
 * slot to a storage slot so draw order can be permuted without rewriting
 * segment data. The per-texel layout is FIXED regardless of which
 * optional fields the dataset has (pool geometries are reused across
 * nodes, so the layout never varies — this also retires the interleaved
 * era's colormap-toggle spec-set rebuild):
 *
 *   | texel | rgba                                                     |
 *   |-------|----------------------------------------------------------|
 *   | 0     | startPos.xyz, startWidth                                 |
 *   | 1     | endPos.xyz, endWidth                                     |
 *   | 2     | startColor.rgb, startSharpness                           |
 *   | 3     | endColor.rgb, endSharpness                               |
 *   | 4     | segmentLength, startCapSuppression, endCapSuppression, 0               |
 *   | 5     | startScalar (0.0), endScalar (0.0), alphas (1.0, 1.0)    |
 *
 * texel5.xy are the colormap scalars and texel5.zw the per-endpoint
 * opacity alphas (from an RGBA color column — volumetric phase 4); ALL
 * FOUR are written UNCONDITIONALLY — pool textures are reused, so
 * leaving them unspecified would let a previous tenant's values leak
 * through. 0.0 is the no-scalar identity and 1.0 (opaque) the
 * per-element-opacity identity. texel4.w stays unspecified (stale on
 * reused pool textures; never read).
 *
 * Texture lifetime = geometry lifetime: `attachLineStorage` registers a
 * `dispose` listener on the geometry, so every dispose site (pool
 * evictors, `pool.dispose()`, the non-pool rebuild) frees the texture
 * with the geometry — no site-by-site bookkeeping.
 *
 * All source arrays consumed by the texel writer arrive as the worker
 * projection's Float32 output (`ProcessedLinesData` — endpoint
 * interpolation always emits Float32), except the Uint8 clipped flags,
 * which the writer reads element-wise (0/1 values are exact in Float32,
 * so no widening allocation is needed — the interleaved era paid one
 * `new Float32Array(uint8)` per update for the same bits).
 *
 * Mirrors `point-geometry.ts` / `gsplat-geometry.ts` so the geometry
 * types share one storage model; the geometry-agnostic helpers live in
 * `./element-storage`.
 *
 * @module rendering/line-geometry
 */

import * as THREE from 'three';
import {
  clampLineCapacity,
  LINE_FLOATS_PER_SEGMENT,
  LINE_TEXTURE_LAYOUT,
} from './element-texture-layout';
import {
  attachElementStorage,
  getElementTexture,
  registerElementTexelDirtyRange,
  elementTexelCapacity,
  writeSortedIndexIdentity,
} from './element-storage';
import type { LinesProjectionBounds } from '../types/lines';

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
 * Range and attribute name match `createPointQuadGeometry` and
 * `createGSplatQuadGeometry` exactly so the three geometry types share
 * one vertex-shader idiom for sprite expansion.
 */
export function createLineQuadGeometry(): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

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
 * The per-segment arrays the texel writer consumes — the worker
 * projection's Float32 endpoint output (see the module header; the
 * Uint8 clipped flags are read element-wise, no widening needed).
 */
export interface LineTexelSource {
  /** Segment start positions (count × 3). */
  startPositions: Float32Array;
  /** Segment end positions (count × 3). */
  endPositions: Float32Array;
  /** Start colors RGB (count × 3; projection white-fills when absent). */
  startColors: Float32Array;
  /** End colors RGB (count × 3). */
  endColors: Float32Array;
  /** Start widths (count). */
  startWidths: Float32Array;
  /** End widths (count). */
  endWidths: Float32Array;
  /** Start sharpness knobs (count; projection fills 0.5 when absent). */
  startSharpness: Float32Array;
  /** End sharpness knobs (count). */
  endSharpness: Float32Array;
  /** 3D segment lengths (count) — cap-ramp math. */
  segmentLengths: Float32Array;
  /**
   * How much of the shader's endpoint cap dimming to suppress at the start
   * endpoint (count, [0, 1] — 1 = no dimming). See
   * `wasm/rust/src/lines_clipping.rs::compute_cap_suppression`.
   */
  startCapSuppression: Float32Array;
  /** Same for the end endpoint (count, [0, 1] — 1 = no dimming). */
  endCapSuppression: Float32Array;
  /**
   * Colormap scalars per endpoint (count each). Absent ⇒ texel5.xy are
   * written 0.0 (the no-scalar identity — written unconditionally so a
   * reused pool texture never leaks a previous tenant's scalars). The
   * line shader's `USE_COLORMAP` path requires both; presence rides the
   * `userData.hasScalars` stamp the call sites write (fail-closed).
   */
  startScalars?: Float32Array;
  endScalars?: Float32Array;
  /**
   * Per-endpoint opacity alphas (count each) from an RGBA color column
   * (volumetric phase 4), packed into texel5.zw. Absent ⇒ both slots
   * are written 1.0 (the per-element-opacity identity — written
   * unconditionally, pool-reuse safe). Presence rides the
   * `userData.hasElementAlpha` stamp the call sites write, which gates
   * only the volumetric w(a) optical-depth map.
   */
  startAlphas?: Float32Array;
  endAlphas?: Float32Array;
  /**
   * Precomputed cull metadata from the projection's fused scan (AABB
   * over start+end positions + max finite width). When present,
   * `computeLineBounds` skips its O(N) per-segment scan (mirrors the
   * points adapter's `metadata.bounds` fast path); absent ⇒ scan
   * fallback. Computed over the FULL projected set: if the written
   * count is capacity-clamped below it, the box is a conservative
   * superset — safe for frustum culling.
   */
  bounds?: LinesProjectionBounds;
}

/**
 * Configuration for instanced lines mesh — the texel source plus the
 * segment count (matches the shape of `ProcessedLinesData`).
 */
export interface InstancedLinesMeshConfig extends LineTexelSource {
  /** Number of segments */
  segmentCount: number;
}

/**
 * Line-bound wrapper over `element-storage.ts::attachElementStorage`:
 * create the line data texture + `aSortedIndex` attribute pair on a
 * geometry, sized for `capacity` segments (6 texels/segment). Returns
 * the texture. See the generic helper for the full lifetime contract.
 */
export function attachLineStorage(
  geometry: THREE.InstancedBufferGeometry,
  capacity: number
): THREE.DataTexture {
  return attachElementStorage(geometry, capacity, LINE_TEXTURE_LAYOUT);
}

/**
 * The line texture attached by `attachLineStorage`, if any (line-bound
 * wrapper over `element-storage.ts::getElementTexture`).
 */
export function getLineTexture(geometry: THREE.BufferGeometry): THREE.DataTexture | null {
  return getElementTexture(geometry);
}

/**
 * Fused texel writer: one pass over the staged arrays into the
 * texture's backing store, in the 6-texel layout documented in the
 * module header. Replaces the 11–13 per-attribute strided writes of the
 * interleaved era — fewer passes over the data, and the guard below
 * runs before ANY store (the interleaved path needed a separate
 * pre-flight sweep to avoid torn multi-attribute writes).
 *
 * Returns the written count, clamped to the texture's capacity
 * (capacity clamping warns once at acquire time; this clamp keeps the
 * write memory-safe if a caller slips past it).
 *
 * `opts.fromSegment` (depth-sorting Phase 4 Stage 2, the append fast
 * path): skip writing texels `[0, fromSegment)` and register the dirty
 * range for only the `[fromSegment, n)` suffix — see
 * `writePointTexels` for the full prefix-identity contract.
 */
export function writeLineTexels(
  texture: THREE.DataTexture,
  src: LineTexelSource,
  count: number,
  opts?: { fromSegment?: number }
): number {
  const arr = texture.image.data as Float32Array;
  const n = Math.min(count, elementTexelCapacity(texture, LINE_FLOATS_PER_SEGMENT));
  const from = Math.max(0, Math.min(opts?.fromSegment ?? 0, n));
  const {
    startPositions,
    endPositions,
    startColors,
    endColors,
    startWidths,
    endWidths,
    startSharpness,
    endSharpness,
    segmentLengths,
    startCapSuppression,
    endCapSuppression,
    startScalars,
    endScalars,
    startAlphas,
    endAlphas,
  } = src;
  // Fail loud on source/count mismatch BEFORE any store (the
  // interleaved-era pre-flight guard's job) — a silent short read would
  // write NaN texels that the shaders' guards then drop invisibly.
  if (
    startPositions.length < n * 3 ||
    endPositions.length < n * 3 ||
    startColors.length < n * 3 ||
    endColors.length < n * 3 ||
    startWidths.length < n ||
    endWidths.length < n ||
    startSharpness.length < n ||
    endSharpness.length < n ||
    segmentLengths.length < n ||
    startCapSuppression.length < n ||
    endCapSuppression.length < n ||
    (startScalars !== undefined && startScalars.length < n) ||
    (endScalars !== undefined && endScalars.length < n) ||
    (startAlphas !== undefined && startAlphas.length < n) ||
    (endAlphas !== undefined && endAlphas.length < n)
  ) {
    throw new Error(
      `writeLineTexels: source arrays shorter than count=${n} ` +
        `(startPositions=${startPositions.length}, endPositions=${endPositions.length}, ` +
        `startColors=${startColors.length}, endColors=${endColors.length}, ` +
        `startWidths=${startWidths.length}, endWidths=${endWidths.length}, ` +
        `startSharpness=${startSharpness.length}, endSharpness=${endSharpness.length}, ` +
        `segmentLengths=${segmentLengths.length}, startCapSuppression=${startCapSuppression.length}, ` +
        `endCapSuppression=${endCapSuppression.length}, startScalars=${startScalars?.length ?? 'absent'}, ` +
        `endScalars=${endScalars?.length ?? 'absent'}, startAlphas=${startAlphas?.length ?? 'absent'}, ` +
        `endAlphas=${endAlphas?.length ?? 'absent'})`
    );
  }
  const hasScalars = startScalars !== undefined && endScalars !== undefined;
  const hasAlphas = startAlphas !== undefined && endAlphas !== undefined;
  for (let i = from; i < n; i++) {
    const o = i * LINE_FLOATS_PER_SEGMENT;
    const p3 = i * 3;
    // texel 0: startPos.xyz, startWidth
    arr[o] = startPositions[p3];
    arr[o + 1] = startPositions[p3 + 1];
    arr[o + 2] = startPositions[p3 + 2];
    arr[o + 3] = startWidths[i];
    // texel 1: endPos.xyz, endWidth
    arr[o + 4] = endPositions[p3];
    arr[o + 5] = endPositions[p3 + 1];
    arr[o + 6] = endPositions[p3 + 2];
    arr[o + 7] = endWidths[i];
    // texel 2: startColor.rgb, startSharpness
    arr[o + 8] = startColors[p3];
    arr[o + 9] = startColors[p3 + 1];
    arr[o + 10] = startColors[p3 + 2];
    arr[o + 11] = startSharpness[i];
    // texel 3: endColor.rgb, endSharpness
    arr[o + 12] = endColors[p3];
    arr[o + 13] = endColors[p3 + 1];
    arr[o + 14] = endColors[p3 + 2];
    arr[o + 15] = endSharpness[i];
    // texel 4: segmentLength, startCapSuppression, endCapSuppression. The .w slot is
    // zero-filled (cheap, keeps reused pool texels deterministic even
    // though nothing reads it yet).
    arr[o + 16] = segmentLengths[i];
    arr[o + 17] = startCapSuppression[i];
    arr[o + 18] = endCapSuppression[i];
    arr[o + 19] = 0.0;
    // texel 5: startScalar, endScalar, per-endpoint opacity alphas.
    // ALL FOUR written UNCONDITIONALLY — pool textures are reused, so
    // leaving them unspecified would let a previous tenant's values
    // leak through. 0.0 = no-scalar identity, 1.0 (opaque) = the
    // per-element-opacity identity for RGB data.
    arr[o + 20] = hasScalars ? startScalars[i] : 0.0;
    arr[o + 21] = hasScalars ? endScalars[i] : 0.0;
    arr[o + 22] = hasAlphas ? startAlphas[i] : 1.0;
    arr[o + 23] = hasAlphas ? endAlphas[i] : 1.0;
  }
  // Ranged upload: only the [from, n) rows just written go to the GPU, not
  // the full capacity-sized image (pool slack rows past n never re-upload;
  // on an append, prefix rows [0, from) stay on the GPU untouched).
  registerElementTexelDirtyRange(texture, LINE_FLOATS_PER_SEGMENT, from, n);
  return n;
}

/**
 * Compute bounding box and sphere from line segment start/end positions.
 * Uses the projection's precomputed fused-scan `bounds` when present
 * (skipping the O(N) per-segment main-thread scan), else a direct
 * min/max pass without temporary geometry or array allocations.
 *
 * bounds are expanded conservatively by `maxWidth` to capture the
 * rendered footprint. Without this, frustum culling and camera-framing
 * reject thick lines whose centerline is just outside the view but
 * whose pixels are still on-screen. Width here is treated as a
 * half-width (matches the rendering spec); using the full width keeps
 * the picking ray pre-cull conservative. Shared by the non-pool paths
 * here and the pool adapter (`gpu-buffer-pool/lines-adapter.ts`).
 */
export function computeLineBounds(
  geometry: THREE.InstancedBufferGeometry,
  meshConfig: LineTexelSource,
  segmentCount: number
): void {
  const box = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity)
  );
  let maxWidth = 0;

  if (meshConfig.bounds && segmentCount > 0) {
    // Fast path: the projection's fused scan precomputed the AABB and
    // max width — no O(N) main-thread scan per commit. Computed over
    // the FULL projected set (conservative superset if `segmentCount`
    // was capacity-clamped below it — safe for frustum culling).
    const { min, max } = meshConfig.bounds;
    box.min.set(min[0], min[1], min[2]);
    box.max.set(max[0], max[1], max[2]);
    maxWidth = meshConfig.bounds.maxWidth;
  } else {
    const v = new THREE.Vector3();
    for (let i = 0; i < segmentCount; i++) {
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
  }

  if (segmentCount > 0 && maxWidth > 0) {
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
 * Stamp per-segment-data presence flags on the geometry's userData.
 * The fixed 6-texel layout always carries the texel5 slots (identity
 * fills when absent), so "does this node have real colormap scalars /
 * a real alpha column?" is not readable off a geometry attribute —
 * `supportsScalarColormap('lines', …)` reads `hasScalars`, and the
 * commit's material sync pushes `hasElementAlpha` into the material's
 * `uHasElementAlpha` gate (volumetric w(a) map). Refreshed on EVERY
 * write (pool geometries are reused across tenants; a presence flip
 * must not leak the previous tenant's stamp — the texel writer already
 * restores the identity fills). Shared by the non-pool paths and the
 * pool adapter.
 */
export function stampLinePresenceFlags(geometry: THREE.BufferGeometry, src: LineTexelSource): void {
  if (!geometry.userData) geometry.userData = {};
  geometry.userData.hasScalars = src.startScalars !== undefined && src.endScalars !== undefined;
  geometry.userData.hasElementAlpha = src.startAlphas !== undefined && src.endAlphas !== undefined;
}

/**
 * Build a lines InstancedBufferGeometry from a mesh config: quad base +
 * EXACT-SIZE line texture / `aSortedIndex` storage pair, one fused texel
 * write, identity ordering (the non-pool path carries no capacity
 * headroom — mirrors `create-points-node.ts::createPointsGeometry`).
 * Shared by `createInstancedLinesMesh` and the rebuild branch of
 * `updateInstancedLinesMesh`.
 */
function buildLinesGeometry(meshConfig: InstancedLinesMeshConfig): THREE.InstancedBufferGeometry {
  const geometry = createLineQuadGeometry();

  // SEMANTIC clamp: every consumer below (storage size, texel/ordering
  // writes, instanceCount) uses the same clamped count, so a request
  // above the per-node texture bound stays self-consistent instead of
  // drawing instances without texels.
  const segmentCount = clampLineCapacity(meshConfig.segmentCount);

  const texture = attachLineStorage(geometry, segmentCount);
  try {
    writeLineTexels(texture, meshConfig, segmentCount);
    writeSortedIndexIdentity(geometry, segmentCount);
  } catch (err) {
    // The texture was attached above; a guard-throwing write would
    // otherwise leak the fresh geometry+texture pair (nobody owns it
    // yet — callers keep the mesh on its OLD geometry when this throws).
    geometry.dispose();
    throw err;
  }

  geometry.instanceCount = segmentCount;
  geometry.setDrawRange(0, 6);
  computeLineBounds(geometry, meshConfig, segmentCount);
  stampLinePresenceFlags(geometry, meshConfig);
  return geometry;
}

/**
 * Create an instanced mesh for lines rendering.
 *
 * Sets up the instanced geometry with the texture-backed per-segment
 * storage pair (RGBA32F line texture + `aSortedIndex`) attached to a
 * shared unit-quad base.
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
  const geometry = buildLinesGeometry(meshConfig);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;
  return mesh;
}

/**
 * Update an existing instanced lines mesh with new segment data.
 *
 * Mirrors the non-pool points commit path (texture-storage era):
 * - Same count: in-place fused texel write into the existing line
 *   texture (zero GPU re-allocation; ranged upload) + identity
 *   ordering reset.
 * - Different count: build a FRESH geometry (the non-pool storage is
 *   exact-size) and dispose the old one — never rebind new storage
 *   onto a rendered geometry, which strands the old GPU resources in
 *   the renderer caches (freed only at GC mercy on classic WebGL;
 *   pinned FOREVER by the WebGPU renderer's strong Info.memoryMap).
 * - Colormap toggles no longer rebuild anything: the fixed 6-texel
 *   layout always carries the scalar slots.
 * - Always: recompute bounding box/sphere + refresh the scalar
 *   presence stamp.
 *
 * @param mesh - Existing mesh to update (must have InstancedBufferGeometry)
 * @param meshConfig - New segment data
 * @returns `true` when the geometry was REBUILT (size change) — the
 *   caller must then evict Three's cached RenderObject (see
 *   `invalidate-render-object.ts`); `false` for the in-place write.
 */
export function updateInstancedLinesMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedLinesMeshConfig
): boolean {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  const texture = getLineTexture(geometry);
  const segmentCount = clampLineCapacity(meshConfig.segmentCount);

  const rebuilt = texture === null || segmentCount !== geometry.instanceCount;
  if (rebuilt) {
    // Build-then-swap-then-dispose: building first keeps the mesh on its
    // old (valid) geometry if the write throws on malformed data —
    // dispose-first would strand the mesh on a disposed geometry whose
    // line texture is already freed.
    const fresh = buildLinesGeometry(meshConfig);
    mesh.geometry = fresh;
    geometry.dispose();
  } else {
    // Same size: fused texel write into the existing texture at the
    // storage slots, then identity ordering (the non-pool path has no
    // preserveOrdering prior — the coordinator re-sorts on commit).
    writeLineTexels(texture, meshConfig, segmentCount);
    writeSortedIndexIdentity(geometry, segmentCount);
    // Force THREE.js to recalculate _maxInstanceCount.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;
    computeLineBounds(geometry, meshConfig, segmentCount);
    stampLinePresenceFlags(geometry, meshConfig);
  }

  return rebuilt;
}
