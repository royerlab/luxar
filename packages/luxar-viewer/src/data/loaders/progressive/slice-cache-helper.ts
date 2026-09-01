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
import { isExtendToAll } from '../../../workers/data-worker/projection/hidden-dims';
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
 * the cache itself). It is a CANONICAL projection of the query determinants,
 * so two viewStates that decode the SAME elements produce the SAME key
 * regardless of which builder created them.
 *
 * Why canonical (not a raw JSON of the four fields): two builders feed the
 * loaders — the navigation/playback builder (`dims-to-view-state.ts`) and the
 * init/reprocess builder (`view-state-manager.ts`) — and they differ in ways
 * that DON'T change which elements load: the discrete-dim ride-along tolerance
 * (`0.5` vs `0`), a displayed dim's `step` (`1` vs `null`), and JSON property
 * order. A raw serialization keys the same timepoint under two strings, so a
 * scrub/playback revisit misses forever (observed: N entries, 0 evictions,
 * ~0 hits). The projection strips exactly that query-irrelevant noise:
 *
 * - Displayed dims: the in-plane query uses a fixed huge tolerance, so neither
 *   position, tolerance, nor metadata changes which elements load — contribute
 *   only a marker so a change to the DISPLAYED SET still rekeys.
 * - Non-displayed discrete non-spatial dims: the executed query derives its
 *   reach from `step` (0.25×step via `tolerance-computer.ts`), NOT from the
 *   ride-along `tolerance` value — key on `step`, drop the ride-along.
 * - Non-displayed continuous/spatial dims: key on position + tolerance +
 *   spatial. Keying on the tolerance here is CONSERVATIVE rather than
 *   demonstrated: since #1183 no query path derives its continuous reach from
 *   the ride-along (gsplats/lines recompute via `tolerance-computer.ts`;
 *   points-with-config and `fallbackQueryTolerance` both use the node's
 *   `maxRadius`), so it is kept only because it is not PROVEN irrelevant for
 *   every geometry/loader — not because it selects the decoded set.
 *
 * This is never NARROWER than the decoded-set determinant (it drops only fields
 * that provably cannot change the loaded elements), so it upholds the loaders'
 * `viewStatesEqual` stale-skip contract — it cannot restore a snapshot the
 * loader would have reloaded. Nothing projection- or render-dependent enters
 * the key: projection re-runs on every hit, and a dataset content-hash change
 * clears the whole cache. `dimensions` may be undefined (each dim then keys on
 * position/tolerance alone — deterministic).
 *
 * `name` is deliberately absent even though `viewStatesEqual` compares it:
 * the one query consumer of dimension names — `extend_to_all` tolerance
 * matching — is applied UPSTREAM by `deriveNodeViewState`, so a name-driven
 * reach change materializes in the per-node `tolerance` this key already
 * covers. (The equality's name comparison is belt-and-braces on top.)
 */
export function buildSliceViewSig(view: SliceViewLike): string {
  const dims = view.dimensions;
  const perDim = view.slicePosition.map((pos, i) => {
    if (view.displayDims.includes(i)) return 0;
    const m = dims?.[i];
    const discrete = m?.discrete === true;
    const spatial = m?.spatial === true;
    // Discrete non-spatial: the ride-along tolerance is query-irrelevant
    // (the real reach is 0.25×step) and builder-inconsistent — key on step.
    // EXCEPT the extend_to_all sentinel: it is the one ride-along value the
    // worker's discrete-dim membership DOES read (isExtendToAll in
    // projection/hidden-dims), so key on MEMBERSHIP ('xa') while still
    // collapsing the churn-prone ordinary values to null. Belt-and-braces:
    // extend_to_all attrs are static today (folded upstream by
    // deriveNodeViewState), but a future dynamic toggle must never restore
    // a snapshot decoded under the other membership.
    const rawTol = view.tolerance[i] ?? 0;
    const tol = discrete && !spatial ? (isExtendToAll(rawTol) ? 'xa' : null) : rawTol;
    return {
      p: pos,
      t: tol,
      di: discrete,
      sp: spatial,
      st: m?.step ?? null,
      cy: m?.cyclic === true,
    };
  });
  return JSON.stringify([[...view.displayDims], perDim]);
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
 * True when the view has at least one non-displayed dimension. With every
 * dimension displayed there is exactly ONE possible slice — it never gets
 * re-queried, so caching it can never hit and would only pin a full deep
 * clone of the ladder (up to the whole SliceCache budget) for nothing.
 * Shared gate for {@link restoreLadder} / {@link storeLadder}.
 */
function hasHiddenDims(view: SliceViewLike): boolean {
  return view.displayDims.length < view.slicePosition.length;
}

/** Look up only the cached ladder payloads, without their logical depth. */
export function restoreLadder<T>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  nLods: number
): T[] | null {
  return restoreLadderSnapshot<T>(sliceCache, path, view, nLods)?.lods ?? null;
}

/**
 * A restored progressive snapshot. `lods` is the retained payload container;
 * `depth` is the logical loaded-level count and may exceed `lods.length` when
 * a loader folds several levels into one cumulative payload.
 */
export interface RestoredLadderSnapshot<T> {
  lods: T[];
  depth: number;
}

/**
 * Look up a cached ladder snapshot for `view` — FULL or PREFIX — or null on
 * miss / disabled / empty. Most loaders use one payload per level, while Lines
 * may fold several loaded levels into one cumulative payload and store the
 * logical depth separately. A prefix restores progress so loading resumes from
 * `startLevel = depth` instead of re-streaming level 0; a full ladder
 * short-circuits the whole load. Shared by all three progressive loaders.
 * The returned elements are the cache entry's own read-only objects; worker
 * projection structured-clones its inputs, so downstream processing does not
 * mutate them. A loader that will push more levels must still shallow-copy the
 * CONTAINER first so it never mutates the cache's payload array.
 */
export function restoreLadderSnapshot<T>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  nLods: number
): RestoredLadderSnapshot<T> | null {
  if (!sliceCache || nLods <= 0 || !hasHiddenDims(view)) return null;
  const entry = sliceCache.get(SliceCache.makeKey(path, buildSliceViewSig(view)));
  if (!entry) return null;
  const lods = entry.payload as T[];
  const depth = entry.ladderDepth ?? lods.length;
  return lods.length > 0 && depth >= lods.length && depth <= nLods ? { lods, depth } : null;
}

/** Remove the cached ladder snapshot for one node/view pair. */
export function deleteLadder(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike
): void {
  if (!sliceCache || !hasHiddenDims(view)) return;
  sliceCache.delete(SliceCache.makeKey(path, buildSliceViewSig(view)));
}

/**
 * Store a cloned snapshot of the ladder — full or prefix — with
 * UPGRADE-IF-LONGER semantics: an existing entry is replaced only when the
 * new snapshot carries MORE logical levels (never downgraded; the depth check uses
 * a non-counting `peek` so bookkeeping never perturbs the hit-rate stat).
 * Callers gate WHEN to store (the loaders store prefixes only while a
 * playback frame budget is active, or any time the ladder is complete —
 * storing every refinement pass would clone O(N²) bytes per slice).
 * MEASURES bytes before cloning and skips oversized snapshots entirely
 * (the LRU would silently reject them after an expensive copy). Shared by
 * all three progressive loaders.
 *
 * `opts.ladderDepth` — the logical prefix depth when payload count differs
 * from level count (the Lines loader folds committed rungs cumulatively).
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
 * fine tail re-decodes. A one-time warning per node surfaces the shortfall
 * (deduped by the cache's `markOversizedWarned`, so it can't leak across
 * sessions or a dataset switch). Folded payloads cannot be trimmed safely
 * because payload count no longer identifies level count; those leave any
 * existing shorter entry intact. If the cumulative payload itself exceeds the
 * budget, the folded slice is not cached and re-decodes on every revisit.
 */
export function storeLadder<T extends object>(
  sliceCache: SliceCache | null,
  path: string,
  view: SliceViewLike,
  lods: readonly T[],
  opts?: { scan?: boolean; pin?: boolean; ladderDepth?: number; totalLODCount?: number }
): void {
  if (!sliceCache || lods.length === 0 || !hasHiddenDims(view)) return;
  const key = SliceCache.makeKey(path, buildSliceViewSig(view));
  const existing = sliceCache.peek(key);
  const requestedDepth = opts?.ladderDepth ?? lods.length;
  if (requestedDepth < lods.length) return;
  const existingDepth =
    existing?.ladderDepth ?? (existing?.payload as unknown[] | undefined)?.length;
  if (existingDepth !== undefined && existingDepth >= requestedDepth) return;
  const isFolded = opts?.ladderDepth !== undefined && opts.ladderDepth !== lods.length;

  // Trim to the largest coarse-first prefix that fits the budget instead of
  // dropping the whole ladder (which would make heavy single slices forever
  // uncacheable). measureLodBytes is O(levels) so the linear scan is cheap.
  let fit = lods.length;
  let bytes = measureLodBytes(lods);
  while (fit > 0 && !sliceCache.willFit(bytes)) {
    fit--;
    bytes = fit > 0 ? measureLodBytes(lods.slice(0, fit)) : 0;
  }
  if (fit === 0) {
    if (sliceCache.markOversizedWarned(path)) {
      log.warning(
        Modules.CACHE,
        `SliceCache: ${isFolded ? 'folded ladder' : 'ladder'} for ${path} exceeds the budget even at one level; this slice will re-decode on every visit. Consider a larger cache budget.`
      );
    }
    return;
  }
  // A logical depth that differs from payload count means at least one payload
  // already folds several levels. Trimming by payload count cannot recover the
  // corresponding ladder depth safely, so retain the existing shorter entry
  // instead of caching a snapshot with a cursor ahead of its data.
  if (isFolded && fit < lods.length) {
    if (sliceCache.markOversizedWarned(path)) {
      log.warning(
        Modules.CACHE,
        `SliceCache: folded ladder for ${path} exceeds the budget; retaining the existing cached prefix because its logical depth cannot be trimmed safely. Consider a larger cache budget.`
      );
    }
    return;
  }
  // Dedupe the warning per NODE (path), not per view: a "ladder exceeds budget"
  // report is about the node vs the budget, not any one slice — so a long
  // playback sweep over many views of the same node warns once, not once/view.
  if (fit < lods.length && sliceCache.markOversizedWarned(path)) {
    log.warning(
      Modules.CACHE,
      `SliceCache: ladder for ${path} exceeds the budget; caching ${fit}/${lods.length} coarse levels (fine tail re-decodes). Consider a larger cache budget.`
    );
  }
  // Re-check upgrade-if-longer against the (possibly trimmed) logical depth.
  const storedDepth = fit < lods.length ? fit : (opts?.ladderDepth ?? fit);
  if (existingDepth !== undefined && existingDepth >= storedDepth) return;
  const snapshot = fit < lods.length ? lods.slice(0, fit) : lods;
  const { ladderDepth: _ladderDepth, totalLODCount, ...cacheOpts } = opts ?? {};
  sliceCache.set(
    key,
    {
      payload: cloneLodSnapshot(snapshot),
      bytes,
      ...(_ladderDepth === undefined && totalLODCount === undefined
        ? {}
        : {
            ladderDepth: storedDepth,
            ...(totalLODCount === undefined ? {} : { totalLadderDepth: totalLODCount }),
          }),
    },
    cacheOpts
  );
}
