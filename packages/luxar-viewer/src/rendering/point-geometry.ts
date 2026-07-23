/**
 * Point Geometry Creation for Luxar
 *
 * Builds the instanced-quad base geometry used by `PointMaterial` and
 * `PointPickingMaterial`, plus the texture-backed per-point storage
 * (depth-sorting Phase 4 points migration). Each point is rendered as an
 * axis-aligned quad with 4 corner vertices; the vertex shader expands the
 * quad into a screen-space sprite of the per-instance size (replacing
 * `gl_PointSize`, which r184's `GLSLNodeBuilder` hardcodes to `1.0` for
 * `THREE.Points`).
 *
 * Per-point data lives in an RGBA32F **point texture** (3 texels/point —
 * see `./element-texture-layout` for the layout authority) sampled by the
 * vertex shader via `texelFetch`; the only per-instance attribute is
 * `aSortedIndex` (Uint32), which maps the draw slot to a storage slot so
 * draw order can be permuted without rewriting point data. The per-texel
 * layout is FIXED regardless of which optional fields the dataset has
 * (pool geometries are reused across nodes, so the layout never varies):
 *
 *   | texel | rgba                                                  |
 *   |-------|-------------------------------------------------------|
 *   | 0     | center.xyz, radius                                    |
 *   | 1     | color.rgb, sharpness                                  |
 *   | 2     | scalar (0.0 when no scalars), alpha (1.0), 0, 0       |
 *
 * texel2.x is the colormap scalar and texel2.y the per-point opacity
 * alpha (reserved for volumetric Phase 3); BOTH are written
 * UNCONDITIONALLY — pool textures are reused, so leaving them
 * unspecified would let a previous tenant's values leak through. 0.0 is
 * the no-scalar identity and 1.0 (opaque) the per-element-opacity
 * identity. texel2.zw stay unspecified (stale on reused pool textures;
 * never read).
 *
 * Texture lifetime = geometry lifetime: `attachPointStorage` registers a
 * `dispose` listener on the geometry, so every dispose site (pool
 * evictors, `pool.dispose()`, the non-pool dispose+recreate commit) frees
 * the texture with the geometry — no site-by-site bookkeeping.
 *
 * All source arrays consumed by the texel writer are ALREADY-WIDENED
 * Float32 — dtype normalization (Uint8/255, Uint16/65535, Float16 copy)
 * happens at the call sites (`gpu-buffer-pool/points-adapter.ts`,
 * `node-factory/create-points-node.ts`) with the exact same
 * `widenToFloat32` calls and fallback fills as the interleaved era, so
 * the floats landing in texels are bit-identical to what the old
 * per-attribute path uploaded.
 *
 * Mirrors `gsplat-geometry.ts` (which pioneered this migration) so the
 * geometry types share one storage model; the geometry-agnostic helpers
 * live in `./element-storage`.
 *
 * @module rendering/point-geometry
 */

import * as THREE from 'three';
import { POINT_FLOATS_PER_POINT, POINT_TEXTURE_LAYOUT } from './element-texture-layout';
import {
  attachElementStorage,
  getElementTexture,
  registerElementTexelDirtyRange,
  elementTexelCapacity,
} from './element-storage';

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
 * Pick a normalization divisor for widening Uint8 / Uint16 source data
 * to Float32 while preserving the GPU-shader-visible [0, 1] range that
 * the interleaved era's per-attribute `normalized: true` flag produced.
 * Shared by the pool adapter and the non-pool factory so both paths
 * widen identically. See `widen-to-float32.ts` for context.
 */
export function pointsNormalizationDivisor(
  source: ArrayLike<number> | undefined,
  normalized: boolean
): number | undefined {
  if (!source || !normalized) return undefined;
  if (source instanceof Uint8Array) return 255;
  if (source instanceof Uint16Array) return 65535;
  return undefined;
}

/**
 * The per-point arrays the texel writer consumes — all ALREADY-WIDENED
 * Float32 (see the module header; dtype normalization and fallback
 * fills happen at the call sites, exactly as the interleaved era did).
 */
export interface PointTexelSource {
  /** Point centers (count × 3). */
  positions: Float32Array;
  /** Colors RGB (count × 3). */
  colors: Float32Array;
  /** Radii (count). */
  radii: Float32Array;
  /** Sharpness knobs (count). */
  sharpness: Float32Array;
  /**
   * Colormap scalars (count). Absent ⇒ texel2.x is written 0.0 (the
   * no-scalar identity — written unconditionally so a reused pool
   * texture never leaks a previous tenant's scalars).
   */
  scalars?: Float32Array;
}

/**
 * Point-bound wrapper over `element-storage.ts::attachElementStorage`:
 * create the point data texture + `aSortedIndex` attribute pair on a
 * geometry, sized for `capacity` points (3 texels/point). Returns the
 * texture. See the generic helper for the full lifetime contract.
 */
export function attachPointStorage(
  geometry: THREE.InstancedBufferGeometry,
  capacity: number
): THREE.DataTexture {
  return attachElementStorage(geometry, capacity, POINT_TEXTURE_LAYOUT);
}

/**
 * The point texture attached by `attachPointStorage`, if any
 * (point-bound wrapper over `element-storage.ts::getElementTexture`).
 */
export function getPointTexture(geometry: THREE.BufferGeometry): THREE.DataTexture | null {
  return getElementTexture(geometry);
}

/**
 * Fused texel writer: one pass over the staged arrays into the
 * texture's backing store, in the 3-texel layout documented in the
 * module header. Replaces the five per-attribute strided writes of the
 * interleaved era — fewer passes over the data.
 *
 * Returns the written count, clamped to the texture's capacity
 * (capacity clamping warns once at acquire time; this clamp keeps the
 * write memory-safe if a caller slips past it).
 *
 * `opts.fromPoint` (depth-sorting Phase 4 Stage 2, the append fast path):
 * skip writing texels `[0, fromPoint)` and register the dirty range for
 * only the `[fromPoint, n)` suffix. The prefix texels are already correct
 * on the texture's CPU mirror from the earlier commit that wrote them
 * (the commit layer only sets `fromPoint > 0` when the prefix is provably
 * byte-identical — prefix lineage + presence matches + intact GPU
 * buffer), so re-writing and re-uploading them is redundant. Source
 * arrays are still full-length `count`; only the loop lower bound and
 * the dirty-range start move — the length guard below is unchanged.
 */
export function writePointTexels(
  texture: THREE.DataTexture,
  src: PointTexelSource,
  count: number,
  opts?: { fromPoint?: number }
): number {
  const arr = texture.image.data as Float32Array;
  const n = Math.min(count, elementTexelCapacity(texture, POINT_FLOATS_PER_POINT));
  const from = Math.max(0, Math.min(opts?.fromPoint ?? 0, n));
  const { positions, colors, radii, sharpness, scalars } = src;
  // Fail loud on source/count mismatch (the interleaved-era writer
  // threw here too) — a silent short read would write NaN texels that
  // the shaders' NaN guards then drop invisibly.
  if (
    positions.length < n * 3 ||
    colors.length < n * 3 ||
    radii.length < n ||
    sharpness.length < n ||
    (scalars !== undefined && scalars.length < n)
  ) {
    throw new Error(
      `writePointTexels: source arrays shorter than count=${n} ` +
        `(positions=${positions.length}, colors=${colors.length}, ` +
        `radii=${radii.length}, sharpness=${sharpness.length}, ` +
        `scalars=${scalars?.length ?? 'absent'})`
    );
  }
  for (let i = from; i < n; i++) {
    const o = i * POINT_FLOATS_PER_POINT;
    const p3 = i * 3;
    // texel 0: center.xyz, radius
    arr[o] = positions[p3];
    arr[o + 1] = positions[p3 + 1];
    arr[o + 2] = positions[p3 + 2];
    arr[o + 3] = radii[i];
    // texel 1: color.rgb, sharpness
    arr[o + 4] = colors[p3];
    arr[o + 5] = colors[p3 + 1];
    arr[o + 6] = colors[p3 + 2];
    arr[o + 7] = sharpness[i];
    // texel 2: scalar, alpha (per-point opacity — volumetric Phase 3).
    // BOTH are written UNCONDITIONALLY — pool textures are reused, so
    // leaving them unspecified would let a previous tenant's values
    // leak through. 0.0 = no-scalar identity, 1.0 (opaque) = the
    // per-element-opacity identity. .zw stay unspecified (stale on
    // reused pool textures; never read).
    arr[o + 8] = scalars !== undefined ? scalars[i] : 0.0;
    arr[o + 9] = 1.0;
  }
  // Ranged upload: only the [from, n) rows just written go to the GPU, not
  // the full capacity-sized image (pool slack rows past n never re-upload;
  // on an append, prefix rows [0, from) stay on the GPU untouched).
  registerElementTexelDirtyRange(texture, POINT_FLOATS_PER_POINT, from, n);
  return n;
}
