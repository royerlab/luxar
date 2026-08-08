/**
 * Per-triangle depth-sort apply for **mesh** nodes — the indexed sibling of
 * `rendering/element-storage.ts`'s `aSortedIndex` machinery.
 *
 * Everything upstream of the apply is shared and needed no change: the
 * coordinator's registration/generation bookkeeping, the SortWorker, and the
 * sort kernel itself all take projected 3D centers in and hand a back-to-front
 * permutation out. A triangle's center is its vertex centroid
 * ({@link computeFaceCentroids}), which is 3 floats per element exactly like a
 * splat center or a segment midpoint. Only the APPLY differs, and it differs
 * structurally rather than by degree.
 *
 * ## Why mesh cannot use the `aSortedIndex` path
 *
 * The three instanced types draw `elementCount` copies of one quad and read
 * per-element data through an `aSortedIndex` **indirection**: permuting the
 * draw order means rewriting a per-instance attribute, and nothing about the
 * element data moves. A mesh has no such indirection — it is a single indexed
 * `drawElements` call, and the draw order of its triangles IS the order of the
 * index buffer. Permuting the draw order therefore means rewriting the index.
 *
 * ## Why there is no double buffering here
 *
 * The instanced path streams a new ordering into an inactive twin attribute and
 * flips a `uSortedIndexSlot` uniform when it completes, so no frame ever samples
 * a half-applied permutation. That trick needs a uniform that can select between
 * two attributes. `geometry.index` is BOUND state, not sampled state: a shader
 * cannot choose between two index buffers, so the only way to swap would be to
 * reassign `geometry.index` — which is precisely the rebind of a drawn
 * geometry's index that `applyMeshIndices` exists to avoid (three caches the
 * attribute→GPU-buffer mapping in a `WeakMap` and frees it only from
 * `WebGLAttributes.remove()`, and the WebGPU backend caches a `RenderObject`
 * keyed on the geometry's bound attributes).
 *
 * So the ordering is written into the ONE index buffer, and it is written
 * **atomically**: the whole visible prefix in one `set` + one update range, never
 * chunked across frames. That is not a preference, it is a correctness
 * requirement — a partially-permuted index buffer is **not a permutation**. Half
 * old and half new means some triangles are drawn twice and others not at all,
 * which is a visibly wrong picture rather than a merely stale one. The
 * instanced path can chunk exactly because its half-written buffer is never the
 * one being drawn.
 *
 * The cost of atomicity is bounded, and by a quantity the mesh path already
 * pays: `applyMeshIndices` re-uploads the same `visibleFaceCount * 3` prefix on
 * **every slice move**. An applied sort costs one more of those, and sorts are
 * already rate-limited to one in flight per node plus the camera-motion
 * thresholds in `config.depthSort`.
 *
 * @module rendering/depth-sort-coordinator/triangle-ordering
 */

import type * as THREE from 'three';
import type { SortedIndexApplyCallbacks } from '../element-storage';

/**
 * Orderings written into an index buffer but not yet drawn.
 *
 * The write itself is synchronous and complete, but the profiler lifecycle
 * (issue #713) must not report *uploaded* until THREE has actually consumed the
 * update range and issued a draw. That is the same acknowledgement point the
 * instanced path uses, reached through the same `onAfterRender` hook the
 * coordinator installs on every tracked node.
 *
 * A `Map` rather than a `WeakMap` so the teardown sweeps can enumerate it;
 * entries are removed on the draw acknowledgement and on every cancellation
 * path, so nothing lingers.
 */
const awaitingDraw = new Map<THREE.BufferGeometry, SortedIndexApplyCallbacks>();

/**
 * Centroid of every visible triangle, `faceCount * 3` floats, in the same
 * display space as `position`.
 *
 * The centroid is the natural per-triangle "center" for the sort kernel, which
 * orders by view-space z of a point: a triangle's sort key is the mean of its
 * vertices' depths. It is the standard choice, and it is an **approximation** —
 * worth being precise about, because there are two distinct ways it falls short
 * and only one of them is inherent to per-primitive sorting:
 *
 * - **Centroid order can disagree with the true order even for disjoint
 *   triangles.** Two triangles that never intersect can overlap in screen space
 *   with one consistently in front across the shared region, while their
 *   centroids — which may both lie outside that region — rank the other way.
 *   That is a property of reducing a triangle to one depth sample, not of
 *   sorting per primitive: an exact answer needs a per-fragment method (depth
 *   peeling, OIT) or a BSP split of the geometry.
 * - **Interpenetrating triangles have no correct order at all**, since the
 *   front-most one changes across the shared region. No primitive-granularity
 *   sort can fix that, whatever key it uses.
 *
 * Both are the same residual the other three geometry types carry (their quads
 * sort by a single center too), and both are why `opaque` — depth-correct per
 * fragment — remains the mesh default.
 *
 * Allocates fresh on every call, deliberately: the buffer is TRANSFERRED to the
 * SortWorker (detached), so a reused scratch array would be destroyed under its
 * next reader. The coordinator invokes this lazily through a thunk, so a node
 * that is not order-dependent never pays it.
 */
export function computeFaceCentroids(
  position: Float32Array,
  indices: Uint32Array,
  faceCount: number
): Float32Array {
  const centers = new Float32Array(faceCount * 3);
  const oneThird = 1 / 3;
  for (let f = 0; f < faceCount; f++) {
    const i = f * 3;
    const a = indices[i] * 3;
    const b = indices[i + 1] * 3;
    const c = indices[i + 2] * 3;
    centers[i] = (position[a] + position[b] + position[c]) * oneThird;
    centers[i + 1] = (position[a + 1] + position[b + 1] + position[c + 1]) * oneThird;
    centers[i + 2] = (position[a + 2] + position[b + 2] + position[c + 2]) * oneThird;
  }
  return centers;
}

/**
 * Apply a back-to-front face ordering by rewriting `geometry.index`.
 *
 * `source` is the CANONICAL (unpermuted, winding-corrected) triple list the
 * commit produced — permuting from the live index buffer instead would compose
 * this permutation with whatever is already applied and produce garbage on the
 * second sort.
 *
 * Returns the number of index entries written, or `0` when the ordering was
 * rejected. A rejection leaves the buffer untouched and the callbacks
 * uninvoked; the caller retains ownership of its profiler session, matching
 * `element-storage.ts`'s `writeSortedIndexOrdering` contract. (Written as code
 * rather than `{@link}`: the symbol is not imported here — only the callbacks type
 * is — and a `{@link import('…').fn}` form does not resolve, which the TypeDoc
 * warning ratchet catches.)
 *
 * Rejections are all "the ordering describes a different face set than the one
 * on screen" — a stale-but-whole picture is strictly better than a corrupt one:
 * - no index attribute (a placeholder that has never committed),
 * - a `source` or index buffer too short for `faceCount`,
 * - `drawRange` disagreeing with `faceCount * 3`, i.e. the commit that produced
 *   this ordering has been superseded by one with a different visible set.
 */
export function writeSortedTriangleOrdering(
  geometry: THREE.BufferGeometry,
  source: Uint32Array,
  ordering: Uint32Array,
  faceCount: number,
  callbacks?: SortedIndexApplyCallbacks
): number {
  const attr = geometry.index;
  const entries = faceCount * 3;
  if (!attr || faceCount <= 0) return 0;
  if (ordering.length < faceCount) return 0;
  if (source.length < entries) return 0;
  if (attr.array.length < entries) return 0;
  // The drawn extent is the commit's authority on how many triangles this
  // geometry currently shows; an ordering sized for a different one is stale by
  // definition (a slice move that raced the sort round-trip).
  if (geometry.drawRange.count !== entries) return 0;

  const dst = attr.array as Uint16Array | Uint32Array;
  for (let f = 0; f < faceCount; f++) {
    const d = f * 3;
    const s = ordering[f] * 3;
    dst[d] = source[s];
    dst[d + 1] = source[s + 1];
    dst[d + 2] = source[s + 2];
  }

  // ONE range covering the whole prefix — see the module note on atomicity. The
  // pending set is cleared first because the commit that produced this ordering
  // registered the identical span; re-registering it is a no-op in extent but
  // keeps the never-accumulate discipline the rest of the tree follows.
  attr.clearUpdateRanges();
  attr.addUpdateRange(0, entries);
  attr.needsUpdate = true;

  // Supersede any earlier ordering still waiting for its draw: it will never be
  // seen, so its lifecycle closes as abandoned rather than applied (issue #713).
  const superseded = awaitingDraw.get(geometry);
  if (superseded) {
    awaitingDraw.delete(geometry);
    superseded.onAbandoned?.();
  }
  if (callbacks) awaitingDraw.set(geometry, callbacks);
  return entries;
}

/**
 * Acknowledge that THREE rendered `geometry` after an ordering was written into
 * its index buffer — the point at which the update range has been consumed and
 * the permutation is actually on screen.
 *
 * Called from the render mesh's `onAfterRender` hook. The entry is removed
 * before the callback runs so re-entrant disposal cannot double-close it.
 */
export function acknowledgeTriangleOrderingDraw(geometry: THREE.BufferGeometry): void {
  const callbacks = awaitingDraw.get(geometry);
  if (!callbacks) return;
  awaitingDraw.delete(geometry);
  callbacks.onApplied?.();
}

/**
 * Drop a written-but-undrawn ordering's lifecycle (node release, geometry
 * dispose, demotion). The permutation itself stays in the index buffer, which is
 * harmless — it is a complete permutation of a face set the geometry still
 * holds, and the next commit overwrites it wholesale.
 */
export function cancelTriangleOrderingApply(geometry: THREE.BufferGeometry): void {
  const callbacks = awaitingDraw.get(geometry);
  if (!callbacks) return;
  awaitingDraw.delete(geometry);
  callbacks.onAbandoned?.();
}

/**
 * Drop every pending acknowledgement (dataset switch, app dispose). The map is
 * snapshotted and cleared before any hook runs, so a hook cannot observe a
 * half-cleared map or re-enter the sweep.
 */
export function cancelAllTriangleOrderingApplies(): void {
  const pending = [...awaitingDraw.values()];
  awaitingDraw.clear();
  for (const callbacks of pending) callbacks.onAbandoned?.();
}
