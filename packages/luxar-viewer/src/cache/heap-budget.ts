/**
 * Heap-aware, two-sided budgeting for the in-memory cache tiers (L0, L1,
 * S-cache).
 *
 * The three JS-heap cache tiers used to be pinned to fixed config sizes
 * (L0 200 MB + L1 100 MB + S-cache 128 MB = 428 MB). That is simultaneously:
 *   - too SMALL on a large desktop heap — a fits-in-RAM timelapse's decoded
 *     working set (~450 MB for the neuromast scene) exceeds the 128 MB S-cache,
 *     so looped playback evicts and re-decodes every lap; and
 *   - too LARGE on a small mobile heap (`jsHeapSizeLimit` ~500 MB–1 GB), where
 *     428 MB of hard-LRU caches is an OOM hazard before the render working set.
 *
 * `computeCacheBudgets` derives per-tier budgets from the device heap so the
 * caches scale UP on big heaps (hold the whole timelapse → decode-free revisit)
 * and DOWN on small heaps (stay within a heap target → no OOM). It is the first
 * live consumer of `config.dataLoading.memory.targetHeapUsage`, and the first
 * stage of a future shared-pool coordinator.
 *
 * Budgets are CEILINGS for demand-filled byte-LRUs, not reservations: a
 * generous S-cache ceiling costs nothing until a large dataset actually fills
 * it. Where the heap size is unknown (`performance.memory` is Chrome-only), the
 * fixed config sizes are returned verbatim — preserving today's behaviour on
 * Firefox/Safari and in the node/jsdom test environment.
 *
 * @module cache/heap-budget
 */

import { config } from '../config';

const MB = 1024 * 1024;

/**
 * Fraction of the usable heap (`jsHeapSizeLimit × targetHeapUsage`) the three
 * cache tiers may occupy together; the remainder is left for the render /
 * decode / scene-graph / WASM working set and the prefetcher's shadow
 * accumulators (which are uncapped and transient).
 */
const CACHE_SHARE_OF_TARGET = 0.6;

/** Hard ceiling on the S-cache so a huge heap can't pin an absurd budget. */
const SLICE_CAP_BYTES = 1024 * MB;

/** The S-cache floor is heap-relative: this fraction of the cache pool… */
const SLICE_MIN_FRACTION = 0.05;
/** …clamped to at least this absolute minimum so it is never fully starved. */
const SLICE_ABS_MIN_BYTES = 16 * MB;

/** Resolved per-tier heap budgets, in bytes. L2 (OPFS/disk) is not included. */
export interface CacheBudgets {
  /** L0 decompressed-chunk cache budget (bytes). */
  l0Bytes: number;
  /** L1 in-memory compressed-chunk cache budget (bytes). */
  l1Bytes: number;
  /** S-cache decoded-slice budget (bytes). */
  sliceBytes: number;
  /** True when derived from a known device heap; false = fixed-config fallback. */
  heapAware: boolean;
}

/**
 * Read `performance.memory.jsHeapSizeLimit` if the (non-standard, Chrome-only)
 * API is present and sane. Returns `undefined` otherwise (Firefox/Safari, node,
 * or a bogus value) so callers fall back to fixed config sizes.
 */
export function readHeapLimitBytes(): number | undefined {
  const mem = (globalThis.performance as Performance & { memory?: { jsHeapSizeLimit?: number } })
    ?.memory;
  const limit = mem?.jsHeapSizeLimit;
  return typeof limit === 'number' && Number.isFinite(limit) && limit > 0 ? limit : undefined;
}

/**
 * Compute two-sided, heap-aware budgets for the L0 / L1 / S-cache tiers.
 *
 * Allocation (when the heap is known):
 *   pool    = heap × targetHeapUsage × CACHE_SHARE_OF_TARGET
 *   sliceMin= clamp(pool × SLICE_MIN_FRACTION, SLICE_ABS_MIN, sliceConfig)  // heap-relative floor
 *   L0, L1  = their config ceilings, scaled DOWN proportionally only if
 *             (pool − sliceMin) cannot seat both — render-proximal priority:
 *             the S-cache floor is protected first, the chunk caches shrink.
 *   slice   = clamp(pool − L0 − L1, sliceMin, SLICE_CAP)  // residual → scales UP
 *
 * @param heapLimitBytes - Override for the device heap limit (tests). When
 *   omitted, {@link readHeapLimitBytes} is consulted; if that is also unknown,
 *   the fixed config sizes are returned (`heapAware: false`).
 */
export function computeCacheBudgets(heapLimitBytes?: number): CacheBudgets {
  const l0Ceil = config.cache.l0MaxSizeMB * MB;
  const l1Ceil = config.cache.l1MaxSizeMB * MB;
  const sliceConfig = config.cache.sliceCacheMaxSizeMB * MB;

  const heap = heapLimitBytes ?? readHeapLimitBytes();
  if (heap === undefined) {
    // Unknown heap (Firefox/Safari/node): preserve the historical fixed sizes.
    return { l0Bytes: l0Ceil, l1Bytes: l1Ceil, sliceBytes: sliceConfig, heapAware: false };
  }

  const pool = heap * config.dataLoading.memory.targetHeapUsage * CACHE_SHARE_OF_TARGET;

  // Heap-relative S-cache floor, never above the historical 128 MB default.
  const sliceMin = Math.min(sliceConfig, Math.max(SLICE_ABS_MIN_BYTES, pool * SLICE_MIN_FRACTION));

  // Seat L0/L1 at their ceilings, but never crowd out the S-cache floor:
  // when the pool is too tight for both, shrink the chunk caches (they rebuild
  // cheaply from L2/network) before the render-proximal S-cache.
  const l0l1Budget = Math.max(0, pool - sliceMin);
  const l0l1Want = l0Ceil + l1Ceil;
  let l0Bytes = l0Ceil;
  let l1Bytes = l1Ceil;
  if (l0l1Want > 0 && l0l1Budget < l0l1Want) {
    const f = l0l1Budget / l0l1Want;
    l0Bytes = l0Ceil * f;
    l1Bytes = l1Ceil * f;
  }

  // The S-cache takes the residual headroom (scales up), floored at sliceMin
  // and capped so it can't pin an unreasonable amount on a very large heap.
  const sliceBytes = Math.min(SLICE_CAP_BYTES, Math.max(sliceMin, pool - l0Bytes - l1Bytes));

  return {
    l0Bytes: Math.floor(l0Bytes),
    l1Bytes: Math.floor(l1Bytes),
    sliceBytes: Math.floor(sliceBytes),
    heapAware: true,
  };
}
