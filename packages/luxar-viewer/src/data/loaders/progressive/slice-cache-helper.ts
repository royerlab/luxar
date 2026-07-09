/**
 * Shared SliceCache helpers for the three progressive loaders (GSplats, Points,
 * Lines). Keeps the per-slice key, snapshot cloning, and lookup/store logic
 * identical across geometries so the loaders stay symmetric and can't drift.
 *
 * @module data/loaders/progressive/slice-cache-helper
 */

import { SliceCache } from '../../../cache/slice-cache';

/** The query fields that determine which elements a slice loads. */
export interface SliceViewLike {
  displayDims: readonly number[];
  slicePosition: readonly number[];
  tolerance: readonly number[];
}

/**
 * Build the view signature used as the SliceCache key (namespaced per node by
 * the cache itself). Covers exactly the query determinants — displayDims,
 * slicePosition, tolerance — matching the progressive loaders' `viewStatesEqual`.
 * Nothing projection- or render-dependent enters the key: projection re-runs on
 * every hit, and a dataset content-hash change clears the whole cache.
 */
export function buildSliceViewSig(view: SliceViewLike): string {
  return JSON.stringify([view.displayDims, view.slicePosition, view.tolerance]);
}

/** Non-DataView ArrayBufferView (i.e. a TypedArray) with a `.slice()`. */
function isTypedArray(v: unknown): v is { slice(): unknown; byteLength: number } {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

/**
 * Total retained bytes of a per-LOD snapshot (sum of every own typed-array
 * property's `byteLength`). Measured WITHOUT cloning so callers can check the
 * SliceCache budget before paying for a (potentially large) deep copy.
 */
export function measureLodBytes<T extends object>(lods: readonly T[]): number {
  let bytes = 0;
  for (const lod of lods) {
    for (const key of Object.keys(lod)) {
      const value = (lod as Record<string, unknown>)[key];
      if (isTypedArray(value)) bytes += value.byteLength;
    }
  }
  return bytes;
}

/**
 * Deep-copy a per-LOD decoded-data snapshot for storage in the SliceCache.
 *
 * Cloning is REQUIRED: a loaded sub-LOD's typed arrays are views into the
 * spatial-index loader's REUSED accumulator buffer (see the spatial-index
 * loaders' "subarrays, zero copy" return). A cross-slice cache that aliased them
 * would be corrupted by the next load. Every own typed-array property is copied;
 * scalars / null (e.g. `splatCount`, `ndim`, absent `colors`) pass through.
 *
 * Generic over the geometry payload (LoadedGSplatsData / LoadedPointsData /
 * LoadedLinesData) so all three loaders share one implementation — it copies by
 * property shape, never by field name.
 */
export function cloneLodSnapshot<T extends object>(lods: readonly T[]): T[] {
  return lods.map((lod) => {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(lod)) {
      const value = (lod as Record<string, unknown>)[key];
      out[key] = isTypedArray(value) ? value.slice() : value;
    }
    return out as T;
  });
}

/**
 * Look up a FULL-ladder cached snapshot for `view`, or null on miss / disabled /
 * incomplete. The returned arrays are shared read-only: concat allocates fresh
 * output (nLods > 1) and worker projection structured-clones its inputs, so the
 * cached snapshot is never mutated. Shared by all three progressive loaders.
 */
export function restoreLadder<T>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  nLods: number
): T[] | null {
  if (!sliceCache || nLods <= 0) return null;
  const entry = sliceCache.get(SliceCache.makeKey(path, buildSliceViewSig(view)));
  if (!entry) return null;
  const lods = entry.payload as T[];
  return lods.length === nLods ? lods : null; // only complete ladders
}

/**
 * Store a cloned snapshot of the completed ladder. No-ops unless the ladder is
 * complete and not already cached. Crucially, it MEASURES bytes first and skips
 * the clone entirely when the snapshot can't fit the budget — otherwise a slice
 * larger than the whole cache would be cloned (expensive) only to be silently
 * rejected by the LRU on every visit. Shared by all three progressive loaders.
 */
export function storeLadder<T extends object>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  lods: readonly T[],
  nLods: number
): void {
  if (!sliceCache || lods.length !== nLods) return;
  const key = SliceCache.makeKey(path, buildSliceViewSig(view));
  if (sliceCache.has(key)) return;
  const bytes = measureLodBytes(lods);
  if (!sliceCache.willFit(bytes)) return; // too large — don't clone just to drop it
  sliceCache.set(key, { payload: cloneLodSnapshot(lods), bytes });
}
