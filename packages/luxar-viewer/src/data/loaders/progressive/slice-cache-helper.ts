/**
 * Shared SliceCache helpers for the three progressive loaders (GSplats, Points,
 * Lines). Keeps the per-slice key, snapshot cloning, and lookup/store logic
 * identical across geometries so the loaders stay symmetric and can't drift.
 *
 * Also used by the three PLAIN spatial-index loaders (plain-leaf nodes, no
 * additive ladder): a plain leaf caches its decoded slice as a 1-element
 * ladder under the same key contract, so plain and progressive nodes share
 * one cache, one signature, and one clone/budget policy.
 *
 * @module data/loaders/progressive/slice-cache-helper
 */

import { SliceCache } from '../../../cache/slice-cache';
import { log, Modules } from '../../../utils/log';
import type { DimensionMetadata } from '../../../types/dims';

/** The query fields that determine which elements a slice loads. */
export interface SliceViewLike {
  displayDims: readonly number[];
  slicePosition: readonly number[];
  tolerance: readonly number[];
  /**
   * Per-dimension metadata (step, discrete, nd_transform, …). Part of the key
   * because the EXECUTED query derives from it: the points fetch reach is
   * `discreteDimTolerance(dimensions[d])` (0.25 × step) while the keyed
   * `tolerance` holds a step-independent ride-along constant for discrete dims
   * (see `dims-to-view-state.ts`) — so a dims-metadata change can alter the
   * decoded set without touching the other three fields.
   */
  dimensions?: ReadonlyArray<Partial<DimensionMetadata>>;
}

/**
 * Build the view signature used as the SliceCache key (namespaced per node by
 * the cache itself). Covers exactly the query determinants — displayDims,
 * slicePosition, tolerance, dimensions — matching the progressive loaders'
 * `viewStatesEqual` field-for-field (that equality is the loaders' stale-skip
 * contract; a key narrower than it can restore a snapshot the loader itself
 * would have reloaded). Nothing projection- or render-dependent enters the
 * key: projection re-runs on every hit, and a dataset content-hash change
 * clears the whole cache. `dimensions` may be undefined (serializes as null —
 * deterministic).
 */
export function buildSliceViewSig(view: SliceViewLike): string {
  return JSON.stringify([
    view.displayDims,
    view.slicePosition,
    view.tolerance,
    view.dimensions ?? null,
  ]);
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
/**
 * True when the view has at least one non-displayed dimension. With every
 * dimension displayed there is exactly ONE possible slice — it never gets
 * re-queried, so caching it can never hit and would only pin a full deep
 * clone of the ladder (up to the whole SliceCache budget) for nothing.
 * Shared gate for {@link restoreLadder} / {@link storeLadder}.
 */
function hasHiddenDims(view: SliceViewLike): boolean {
  return view.displayDims.length < view.slicePosition.length;
}

export function restoreLadder<T>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  nLods: number
): T[] | null {
  if (!sliceCache || nLods <= 0 || !hasHiddenDims(view)) return null;
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
 *
 * `opts.scan` — the caller is storing inside a sequential scan (dimension
 * playback; the loaders pass their per-pass `frameBudgetMs !== null`).
 * Forwarded to `SliceCache.set` to select scan-resistant (MRU-victim)
 * eviction, which keeps the loop-head prefix resident across cyclic loops.
 *
 * `opts.pin` — the caller is the SlicePrefetcher storing a projected t+1
 * ladder; forwarded so the entry survives eviction until the foreground tick
 * restores it (see `SliceCache.set`).
 *
 * OVERSIZED LADDERS: when the full snapshot exceeds the whole cache budget the
 * cache would reject it wholesale, leaving that slice permanently uncacheable
 * (it re-decodes on every visit). Instead we store the largest COARSE-FIRST
 * prefix that fits — the additive ladder is amplitude-ordered, so the coarse
 * levels are the cheapest to keep and give a usable partial revisit; only the
 * fine tail re-decodes. A one-time warning per key surfaces the shortfall
 * (deduped by the cache's `markOversizedWarned`, so it can't leak across
 * sessions or a dataset switch).
 */
export function storeLadder<T extends object>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  lods: readonly T[],
  opts?: { scan?: boolean; pin?: boolean }
): void {
  if (!sliceCache || lods.length === 0 || !hasHiddenDims(view)) return;
  const key = SliceCache.makeKey(path, buildSliceViewSig(view));
  const existing = sliceCache.peek(key);
  if (existing && (existing.payload as unknown[]).length >= lods.length) return;

  // Trim to the largest coarse-first prefix that fits the budget instead of
  // dropping the whole ladder (which would make heavy single slices forever
  // uncacheable). measureLodBytes is O(levels) so the linear scan is cheap.
  let fit = lods.length;
  let bytes = measureLodBytes(lods);
  while (fit > 0 && !sliceCache.willFit(bytes)) {
    fit--;
    bytes = fit > 0 ? measureLodBytes(lods.slice(0, fit)) : 0;
  }
  if (fit === 0) return; // even the coarsest single level exceeds the budget
  if (fit < lods.length && sliceCache.markOversizedWarned(key)) {
    log.warning(
      Modules.CACHE,
      `SliceCache: ladder for ${path} exceeds the budget; caching ${fit}/${lods.length} coarse levels (fine tail re-decodes). Consider a larger cache budget.`
    );
  }
  // Re-check upgrade-if-longer against the (possibly trimmed) prefix length.
  if (existing && (existing.payload as unknown[]).length >= fit) return;
  const snapshot = fit < lods.length ? lods.slice(0, fit) : lods;
  sliceCache.set(key, { payload: cloneLodSnapshot(snapshot), bytes }, opts);
}
