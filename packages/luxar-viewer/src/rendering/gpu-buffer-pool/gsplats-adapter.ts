/**
 * GSplats buffer-pool adapter.
 *
 * Owns the gsplats-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Gaussian-splat geometries.
 * Shared state (activeBuffers, stats, commit counter) is read from the
 * GPUBufferPool reference passed at construction.
 *
 * The acquire/release/dispose lifecycle is shared with the points and
 * lines adapters (`texture-backed-adapter.ts`); this file owns the GSplats
 * texel layout hook, the held-draw commit ordering, and the update path.
 */

import * as THREE from 'three';
import {
  attachSplatStorage,
  computeMaxCholeskyRowNorm,
  getSplatTexture,
  stampGSplatPresenceFlags,
  writeSplatTexels,
} from '../gsplat-geometry';
import {
  holdSortedIndexDrawForAppend,
  holdSortedIndexDrawFromSeed,
  repairSortedIndexForCount,
  sortedIndexDrawHoldTarget,
  writeSortedIndexIdentity,
} from '../element-storage';
import { clampSplatCapacity } from '../element-texture-layout';
import type { GSplatsProjectionBounds } from '../../types/gsplats';
import type { FreeBucketMap } from './byte-tracked-maps';
import {
  TextureBackedAdapter,
  prepareInstancedQuadForDraw,
  type InstancedOrderingOptions,
  type PoolAdapterHost,
  type TextureBackedLayout,
} from './texture-backed-adapter';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../config/constants';

/**
 * Packed GSplats data ready for GPU upload (from gsplats/projection.ts).
 *
 * Carries arrays only — the splat count M travels as the positional
 * `count` argument of `updateGeometry`, matching the points/lines
 * adapters (whose payloads' own count fields are pipeline metadata the
 * pool never reads).
 */
export interface PackedGSplatsData {
  centers3D: Float32Array; // M * 3
  amplitudes: Float32Array; // M
  /** M * 6, row-major [L00, L10, L11, L20, L21, L22] per splat */
  choleskyFactors: Float32Array;
  colors: Float32Array; // M * 3 (RGB) or M * 4 (RGBA — alpha = per-splat opacity)
  /** Components per color item: 3 (RGB) or 4 (RGBA). Absent means 3. */
  colorComponents?: 3 | 4;
  labelIndices?: Uint32Array;
  /**
   * Precomputed cull metadata from the projection's fused scan (AABB of
   * `centers3D` + max Cholesky row norm). When present, `updateGeometry`
   * skips its two O(N) main-thread scans (mirrors the points adapter's
   * `data.metadata.bounds` fast path); absent ⇒ scan fallback.
   */
  bounds?: GSplatsProjectionBounds;
}

/** Ordering options for {@link GSplatsBufferAdapter.updateGeometry}. */
export interface GSplatsUpdateOptions extends InstancedOrderingOptions {
  /**
   * A full write that EXTENDS the previous population into a grown geometry:
   * the previous geometry's drawn permutation of `[0, seedOrdering.length)`,
   * which the draw is held on until the grown population's ordering lands.
   */
  seedOrdering?: Uint32Array;
}

/**
 * GSplats: 4 texels/splat, so a per-node bound of width × maxTextureSize / 4
 * (4.19M splats on a 4096-class device).
 */
const GSPLATS_LAYOUT: TextureBackedLayout = {
  type: 'gsplats',
  clampCapacity: clampSplatCapacity,
  attachStorage: attachSplatStorage,
};

/**
 * Write the ordering a gsplat commit draws with, and return the instance
 * count to draw NOW (below `count` only while an append's draw is held).
 *
 * - APPEND (`fromSplat > 0`): the texture upload stays suffix-only and the draw
 *   is HELD at the previous population in its previous order until the grown
 *   population's own ordering lands ({@link holdSortedIndexDrawForAppend}).
 *   The old sorted prefix plus a storage-order suffix would render two
 *   independently ordered populations, and a full storage-order reset draws
 *   the whole node unsorted for the 75-225 ms a large sort takes.
 * - `seedOrdering` (a full write into a GROWN geometry that extends the
 *   previous population): the same hold, seeded with the previous geometry's
 *   drawn permutation ({@link holdSortedIndexDrawFromSeed}).
 * - `preserveOrdering` (chosen by commit-gsplats-geometry.ts) keeps a
 *   same-count prior while its re-sort lands; skipping that write also
 *   registers no new update range.
 * - `repairFromCount` on a count CHANGE rebuilds the existing permutation over
 *   the new population rather than discarding it — an nD re-slice changes the
 *   resident count at almost every step, so this is what a timelapse takes.
 * - A vouched prior on a geometry whose draw is still HELD is only valid over
 *   the DRAWN prefix, so it is repaired from there instead.
 * - Anything else resets to identity, which re-homes the geometry on slot 0.
 *   The commit path must therefore call `depthSort.noteCommit` after this update;
 *   its immediate syncSortedIndexSlot pushes the new slot to visual and pick
 *   materials before either can draw (the per-frame pump re-asserts it).
 */
function writeCommitOrdering(
  geometry: THREE.InstancedBufferGeometry,
  count: number,
  fromSplat: number,
  options: GSplatsUpdateOptions | undefined
): number {
  const drawnBefore = geometry.instanceCount;
  if (fromSplat > 0) return holdSortedIndexDrawForAppend(geometry, drawnBefore, count);
  if (options?.seedOrdering) {
    return holdSortedIndexDrawFromSeed(geometry, options.seedOrdering, count);
  }
  const vouched = options?.preserveOrdering === true || options?.repairFromCount !== undefined;
  if (vouched && sortedIndexDrawHoldTarget(geometry) !== undefined) {
    repairSortedIndexForCount(geometry, drawnBefore, count);
    return count;
  }
  if (options?.preserveOrdering) return count;
  const repairFrom = options?.repairFromCount;
  if (repairFrom !== undefined && repairFrom > 0) {
    // A repair deliberately leaves the slot where it is: its callers only fire
    // when the tenant, geometry and buffers are all unchanged, so the
    // slot/uniform pairing is already established.
    repairSortedIndexForCount(geometry, repairFrom, count);
  } else {
    writeSortedIndexIdentity(geometry, count);
  }
  return count;
}

export class GSplatsBufferAdapter extends TextureBackedAdapter {
  constructor(host: PoolAdapterHost) {
    super(host, GSPLATS_LAYOUT);
  }

  /** @internal — pool-bucketed reusable gsplat geometries. */
  get gsplatBuffers(): FreeBucketMap {
    return this.buffers;
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: PackedGSplatsData,
    count: number,
    truncationRadius: number = GSPLAT_DEFAULT_TRUNCATION_RADIUS,
    options?: GSplatsUpdateOptions
  ): void {
    const texture = getSplatTexture(geometry);
    if (!texture) {
      throw new Error(
        'GSplatsBufferAdapter.updateGeometry: geometry has no splat texture — ' +
          'was it acquired from the pool?'
      );
    }
    // Append fast path (Phase 4 Stage 2): the commit layer sets `fromInstance`
    // to the prefix count already on the GPU when this commit only extends it,
    // so the fused writer + ranged upload touch just the `[fromSplat, count)`
    // suffix (see writeSplatTexels). 0 means a full write.
    const fromSplat = options?.fromInstance ?? 0;
    // One fused pass over the staged arrays into the texel layout
    // (replaces the six per-attribute strided writes), then identity
    // ordering. The writer clamps to the texture capacity; mirror that
    // clamp in instanceCount so a bound-clamped node never draws
    // instances whose texels were not written.
    count = writeSplatTexels(
      texture,
      {
        centers: data.centers3D,
        choleskyFactors: data.choleskyFactors,
        amplitudes: data.amplitudes,
        colors: data.colors,
        colorComponents: data.colorComponents,
        labelIndices: data.labelIndices,
      },
      count,
      { fromSplat }
    );
    // Presence stamp — shared chokepoint with the non-pool writer paths;
    // see stampGSplatPresenceFlags (refresh on every update: pool tenants).
    stampGSplatPresenceFlags(geometry, { colorComponents: data.colorComponents });
    const drawCount = writeCommitOrdering(geometry, count, fromSplat, options);
    prepareInstancedQuadForDraw(geometry, drawCount);

    // CRITICAL: Recompute bounding box from updated center positions
    // and expand by max splat extent for correct frustum culling.
    // For each splat, the per-axis extent is truncationRadius × σ_d, where
    // σ_d = ||L[d,:]|| (the row norm of the Cholesky factor). We use the
    // max row norm across all splats and axes as a conservative expansion.
    //
    // Fast path: the projection's fused scan precomputed both (AABB +
    // max row norm) — no O(N) main-thread scans per commit. Computed
    // over the FULL projected set: if `count` was capacity-clamped
    // below it, the box is a conservative superset (safe for frustum
    // culling — same trade the points adapter's metadata.bounds path
    // accepts). Fallback: direct scans over the written count.
    const box = new THREE.Box3();
    let maxRowNorm: number;
    if (data.bounds) {
      const { min, max } = data.bounds;
      box.min.set(min[0], min[1], min[2]);
      box.max.set(max[0], max[1], max[2]);
      maxRowNorm = data.bounds.maxRowNorm;
    } else {
      const v = new THREE.Vector3();
      for (let i = 0; i < count; i++) {
        v.set(data.centers3D[i * 3], data.centers3D[i * 3 + 1], data.centers3D[i * 3 + 2]);
        box.expandByPoint(v);
      }
      maxRowNorm = computeMaxCholeskyRowNorm(data.choleskyFactors, count);
    }
    const expansion = maxRowNorm * truncationRadius;
    box.expandByScalar(expansion);

    geometry.boundingBox = box;
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    geometry.boundingSphere = sphere;
  }
}
