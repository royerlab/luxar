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
 * Look up a cached ladder snapshot for `view` — FULL or PREFIX — or null on
 * miss / disabled / empty. A prefix (length < nLods, stored when a playback
 * frame budget capped the streaming loop) restores the loader's progress so
 * loading resumes from `startLevel = prefix length` instead of re-streaming
 * level 0; a full ladder short-circuits the whole load. The returned arrays
 * are shared read-only: concat allocates fresh output and worker projection
 * structured-clones its inputs, so the cached snapshot is never mutated —
 * loaders must still shallow-copy the CONTAINER before pushing further levels
 * into it. Shared by all three progressive loaders.
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
  return lods.length > 0 && lods.length <= nLods ? lods : null;
}

/**
 * Store a cloned snapshot of the ladder — full or prefix — with
 * UPGRADE-IF-LONGER semantics: an existing entry is replaced only when the
 * new snapshot carries MORE levels (never downgraded; the length check uses
 * a non-counting `peek` so bookkeeping never perturbs the hit-rate stat).
 * Callers gate WHEN to store (the loaders store prefixes only while a
 * playback frame budget is active, or any time the ladder is complete —
 * storing every refinement pass would clone O(N²) bytes per slice).
 * MEASURES bytes before cloning and skips oversized snapshots entirely
 * (the LRU would silently reject them after an expensive copy). Shared by
 * all three progressive loaders.
 */
export function storeLadder<T extends object>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  lods: readonly T[]
): void {
  if (!sliceCache || lods.length === 0) return;
  const key = SliceCache.makeKey(path, buildSliceViewSig(view));
  const existing = sliceCache.peek(key);
  if (existing && (existing.payload as unknown[]).length >= lods.length) return;
  const bytes = measureLodBytes(lods);
  if (!sliceCache.willFit(bytes)) return; // too large — don't clone just to drop it
  sliceCache.set(key, { payload: cloneLodSnapshot(lods), bytes });
}
