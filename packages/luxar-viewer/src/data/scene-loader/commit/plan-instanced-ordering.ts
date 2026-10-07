/**
 * How a pool commit writes a Points, Lines or GSplats node's ordering buffer —
 * one decision for the three instanced types, so an append or pool grow draws
 * the same way whatever the geometry (the adapters' half is
 * `writeInstancedCommitOrdering` in `rendering/gpu-buffer-pool`).
 *
 * @module data/scene-loader/commit/plan-instanced-ordering
 */

import type * as THREE from 'three';
import { getActiveSortedIndexAttribute } from '../../../rendering/element-storage';
import type { InstancedOrderingOptions } from '../../../rendering/gpu-buffer-pool';

/** The pre-commit facts {@link planInstancedOrdering} decides from. */
export interface InstancedOrderingInput {
  /** The geometry the pool just acquired for this commit. */
  geometry: THREE.BufferGeometry;
  /** The mesh's geometry before the acquire. */
  prevGeometry: THREE.BufferGeometry;
  /**
   * `prevGeometry.instanceCount` read BEFORE the acquire: a pool grow releases
   * the old geometry without ending its held draw, so this is the count that
   * was actually on screen.
   */
  prevDrawnCount: number;
  hadCommittedData: boolean;
  /** The element count the previous commit recorded. */
  prevCount: number | undefined;
  /** This commit's (capacity-clamped) element count. */
  count: number;
  attributesRebuilt: boolean;
  /** `getPrefixParent` of this commit's data (read before it is consumed). */
  prefixParent: unknown;
  /** `getCommittedData` of the mesh: what its last commit uploaded. */
  committedData: unknown;
  /**
   * The GPU still holds the committed prefix exactly as this commit would
   * write it (`gpuPrefixIntact` plus each type's parity conjuncts), so the
   * texel upload may be suffix-only.
   */
  prefixReusable: boolean;
}

/**
 * Keep the permutation, repair it over a changed count, append a suffix and
 * hold the draw, or seed a grown geometry's held draw.
 *
 * - Same buffers, same count: `preserveOrdering` — a permutation of
 *   `[0, count)` is a strictly-no-worse prior than storage order for the frame
 *   or more until the re-sort `depthSort.noteCommit` dispatches lands.
 * - Same buffers, CHANGED count: `repairFromCount` rebuilds the permutation
 *   over the new population instead of falling back to storage order. It is
 *   the branch a timelapse takes: an nD re-slice changes the resident count at
 *   almost every step, and storage order drew one unsorted frame per timepoint
 *   (#2290). Measured on the `cloud` demo at a frozen camera pose, as the
 *   fraction of sampled element pairs composited in correct back-to-front
 *   order: storage order 0.617, repaired 0.858, a real sort 1.000.
 *   "Same buffers" needs all three: committed data (a node's first commit
 *   after LOD demotion no longer vouches for its retained ordering), no
 *   attribute rebuild, and the same geometry (pool best-fit reuse can hand a
 *   node ANOTHER node's permutation over a different count).
 * - An EXTENSION — a larger count whose prefix lineage is the data committed
 *   here, which proves same generation (a view change resets the lineage), a
 *   genuine extension, and that the GPU still holds that parent — on the same
 *   buffers with a reusable prefix: `fromInstance`. The append fast path
 *   uploads only the suffix and holds the draw on the drawn prefix.
 * - Any other extension (a pool GROW — every ladder rung that doubles crosses
 *   the pool's 1.5x capacity headroom — or a prefix that must be rewritten):
 *   the same lineage proof makes the previous geometry's DRAWN permutation an
 *   exact ordering of a grown geometry's prefix, so it is handed over as
 *   `seedOrdering` and the draw is held on it instead of showing the grown
 *   node in storage order. (On the same geometry there is no seed: the
 *   rewrite repairs from the previous count instead.)
 */
export function planInstancedOrdering(input: InstancedOrderingInput): InstancedOrderingOptions {
  const { geometry, prevGeometry, prevCount, count } = input;
  const sameBuffers = reusesSameBuffers(input);
  const extendsCommitted = extendsCommittedPrefix(input);
  const canAppend = extendsCommitted && sameBuffers && input.prefixReusable;
  return {
    preserveOrdering: sameBuffers && prevCount === count,
    repairFromCount: sameBuffers && prevCount !== count ? prevCount : undefined,
    fromInstance: canAppend ? (prevCount ?? 0) : 0,
    seedOrdering:
      extendsCommitted && !canAppend
        ? drawnOrderingSeed(prevGeometry, geometry, input.prevDrawnCount)
        : undefined,
  };
}

/** True when the buffer still holds this node's own committed ordering. */
function reusesSameBuffers(input: InstancedOrderingInput): boolean {
  return (
    input.hadCommittedData && !input.attributesRebuilt && input.geometry === input.prevGeometry
  );
}

/** True when this commit's data provably extends the population committed here. */
function extendsCommittedPrefix(input: InstancedOrderingInput): boolean {
  return (
    input.hadCommittedData &&
    input.count > (input.prevCount ?? 0) &&
    input.prefixParent !== undefined &&
    input.prefixParent === input.committedData
  );
}

/**
 * The permutation `prevGeometry` was DRAWING before the pool acquire (its
 * active ordering over `[0, drawnCount)`), or `undefined` when it has none
 * worth keeping. A view onto the released geometry's CPU array, consumed
 * synchronously by the update that follows.
 */
function drawnOrderingSeed(
  prevGeometry: THREE.BufferGeometry,
  nextGeometry: THREE.BufferGeometry,
  drawnCount: number
): Uint32Array | undefined {
  // Same geometry = not a grow (the append path, or a full in-place rewrite).
  if (prevGeometry === nextGeometry) return undefined;
  const geometry = prevGeometry as THREE.InstancedBufferGeometry;
  if (!geometry.isInstancedBufferGeometry) return undefined;
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr || !Number.isInteger(drawnCount) || drawnCount <= 0) return undefined;
  if (drawnCount > attr.array.length) return undefined;
  return (attr.array as Uint32Array).subarray(0, drawnCount);
}
