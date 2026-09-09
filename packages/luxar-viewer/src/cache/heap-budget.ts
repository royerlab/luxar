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
import { getInputProfile, type DeviceClass } from '../utils/input-capabilities';

const MB = 1024 * 1024;

/**
 * Fraction of the usable heap (`jsHeapSizeLimit × targetHeapUsage`) the three
 * cache tiers may occupy together; the remainder is left for the render /
 * decode / scene-graph / WASM working set and the prefetcher's shadow
 * accumulators (which are uncapped and transient).
 */
const CACHE_SHARE_OF_TARGET = 0.6;
const NON_CACHE_SHARE_OF_TARGET = 1 - CACHE_SHARE_OF_TARGET;
/**
 * How the two shares DERIVED HERE divide the non-cache remainder: eager line
 * working set 0.5, L2 write-queue retain 0.25, leaving a quarter of the
 * remainder unclaimed by either. That is a bound on what THESE TWO numbers
 * hand out, not on total commitment — the eager 0.5 is claimed independently,
 * in full, by two consumers (`createLineWorkingSetGate` in
 * `data/scene-loader/nodes/load-children-concurrently.ts`,
 * `RefinementResidencyBudget.forSession` in
 * `data/scene-loader/progressive/residency-budget.ts`). GPU geometry takes the
 * full remainder through `computeNonCacheRemainderBytes`, with its own 2 GB
 * ceiling, because its evictor counts many of the same renderer rows as the
 * refinement refusal gate rather than reserving an additive population. Keep
 * that overlap in mind when retuning: these are independent ceilings on
 * demand-filled structures, not shares whose sum is a memory reservation.
 */
const EAGER_WORKING_SET_SHARE_OF_REMAINDER = 0.5;
const EAGER_WORKING_SET_CAP_BYTES = 512 * MB;
/** Historical line-admission fallback when no session memory signal exists. */
const EAGER_WORKING_SET_FIXED_FALLBACK_BYTES = 256 * MB;

/**
 * The L2 (OPFS) write queue's share of that same remainder (see the split
 * above). Its own constants deliberately — the eager working set and the write
 * queue are unrelated consumers, so one's ceiling must never move because the
 * other's was retuned.
 */
const OPFS_WRITE_QUEUE_SHARE_OF_REMAINDER = 0.25;
/** Ceiling so a very large heap can't let the queue pin an absurd retain. */
const OPFS_WRITE_QUEUE_CAP_BYTES = 512 * MB;
/** No-signal fallback: clears today's largest demo while still being a bound. */
const OPFS_WRITE_QUEUE_FIXED_FALLBACK_BYTES = 256 * MB;

/** Hard ceiling on the S-cache so a huge heap can't pin an absurd budget. */
const SLICE_CAP_BYTES = 2048 * MB;

/** The S-cache floor is heap-relative: this fraction of the cache pool… */
const SLICE_MIN_FRACTION = 0.05;
/** …clamped to at least this absolute minimum so it is never fully starved. */
const SLICE_ABS_MIN_BYTES = 16 * MB;

/**
 * Total cache pool (bytes) per inferred DEVICE CLASS — the fallback used where
 * the heap is unmeasurable (WebKit: WKWebView / Safari have no
 * `performance.memory`) AND no explicit `?cacheBudgetMB=` override was given.
 * Values are demand-filled ceilings, split across tiers by the same rule as the
 * heap path. Desktop reaches ≥2 GB (with `SLICE_CAP_BYTES` raised to match).
 * Exported so the GPU byte budget can size its own mobile candidate from the
 * same table (`rendering/gpu-byte-budget.ts`).
 */
export const DEVICE_CLASS_POOL_BYTES: Readonly<Record<DeviceClass, number>> = {
  mobile: 384 * MB,
  laptop: 1024 * MB,
  desktop: 2048 * MB,
};

function positive(value: number | undefined): value is number {
  return value != null && Number.isFinite(value) && value > 0;
}

/** Convert a positive cache-pool override from MiB to bytes. */
export function cachePoolOverrideBytes(
  cacheBudgetMB: number | null | undefined
): number | undefined {
  return cacheBudgetMB != null && cacheBudgetMB > 0 ? cacheBudgetMB * MB : undefined;
}

/**
 * Device-class cache pool (bytes), or `undefined` outside a browser (node/tests)
 * so callers fall through to the fixed config sizes there rather than a guess.
 * The class comes from the shared input profile (`utils/input-capabilities`),
 * so the cache pool, the GPU budget and the touch UI all agree on what the
 * device is.
 */
export function deviceClassPoolBytes(): number | undefined {
  if (typeof navigator === 'undefined') return undefined;
  return DEVICE_CLASS_POOL_BYTES[getInputProfile().deviceClass];
}

/** Resolved per-tier heap budgets, in bytes. L2 (OPFS/disk) is not included. */
export interface CacheBudgets {
  /** L0 decompressed-chunk cache budget (bytes). */
  l0Bytes: number;
  /** L1 in-memory compressed-chunk cache budget (bytes). */
  l1Bytes: number;
  /** S-cache decoded-slice budget (bytes). */
  sliceBytes: number;
  /** True when derived (heap OR explicit override); false = fixed-config fallback. */
  heapAware: boolean;
  /**
   * Where the pool came from:
   *   `heap`     — measured `performance.memory.jsHeapSizeLimit` (Chrome).
   *   `explicit` — a caller-supplied pool override (`?cacheBudgetMB=` / the
   *                native launcher), used where `performance.memory` is absent
   *                (WKWebView/Safari) so the app still gets a real budget.
   *   `device-class` — no heap and no override: an inferred mobile/laptop/desktop
   *                pool (see {@link deviceClassPoolBytes}).
   *   `fixed`    — none of the above (non-browser / no device signals): the
   *                historical fixed config sizes.
   */
  source: 'heap' | 'explicit' | 'device-class' | 'fixed';
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
 * Heap headroom left OUTSIDE the cache pool — the shared memory model every
 * non-cache consumer sizes itself from (the eager child loader, refinement
 * residency, GPU geometry, the L2 write queue), so a session uses ONE remainder
 * rather than each consumer inventing its own view of the heap.
 *
 * Resolution matches the cache pool: explicit pool override → measured heap →
 * device-class pool. A pool-based input is converted back to the remainder it
 * implies (`pool × NON_CACHE/CACHE`) so a WebKit session — where the heap
 * cannot be measured at all — stays on the same model as a Chrome one.
 *
 * Returns `undefined` when there is no memory signal at all (non-browser /
 * Firefox+Safari without an override), so each consumer applies its own
 * no-signal fallback.
 *
 * @param heapLimitBytes - Override for the device heap limit (tests). When
 *   omitted, {@link readHeapLimitBytes} is consulted; an explicitly passed
 *   non-positive value skips the heap path WITHOUT probing browser state.
 * @param poolOverrideBytes - Explicit total cache pool in bytes. Takes
 *   precedence over the heap-derived remainder.
 * @param fallbackPoolBytes - Device-class total cache pool in bytes, used only
 *   when neither an explicit pool nor a measurable heap is available.
 */
export function computeNonCacheRemainderBytes(
  heapLimitBytes?: number,
  poolOverrideBytes?: number,
  fallbackPoolBytes?: number
): number | undefined {
  const remainderFromCachePool = (poolBytes: number): number =>
    poolBytes * (NON_CACHE_SHARE_OF_TARGET / CACHE_SHARE_OF_TARGET);
  if (positive(poolOverrideBytes)) return remainderFromCachePool(poolOverrideBytes);
  const heap = heapLimitBytes === undefined ? readHeapLimitBytes() : heapLimitBytes;
  if (positive(heap)) {
    return heap * config.dataLoading.memory.targetHeapUsage * NON_CACHE_SHARE_OF_TARGET;
  }
  if (positive(fallbackPoolBytes)) return remainderFromCachePool(fallbackPoolBytes);
  return undefined;
}

/**
 * The eager child loader's share of the heap headroom left outside the cache
 * pool (`computeNonCacheRemainderBytes`): half of it. The L2 write-queue retain takes
 * a quarter (see {@link computeOpfsWriteQueueBudgetBytes}) and the last quarter
 * is claimed by neither, while the absolute cap prevents a large V8 heap limit
 * from recreating an eight-wide allocation spike. Note that this budget is
 * handed to several independent consumers in full — see the ledger on the share
 * constants — so it bounds each of them, not their sum.
 *
 * @param heapLimitBytes - Override for the device heap limit (tests). When
 *   omitted, {@link readHeapLimitBytes} is consulted; invalid explicit values
 *   skip the heap path without probing browser state.
 * @param poolOverrideBytes - Explicit total cache pool in bytes. Takes
 *   precedence over the heap-derived budget.
 * @param fallbackPoolBytes - Device-class total cache pool in bytes, used only
 *   when neither an explicit pool nor a measurable heap is available.
 */
export function computeWorkingSetBudgetBytes(
  heapLimitBytes?: number,
  poolOverrideBytes?: number,
  fallbackPoolBytes?: number
): number {
  const remainder = computeNonCacheRemainderBytes(
    heapLimitBytes,
    poolOverrideBytes,
    fallbackPoolBytes
  );
  if (remainder === undefined) return EAGER_WORKING_SET_FIXED_FALLBACK_BYTES;
  return Math.floor(
    Math.min(remainder * EAGER_WORKING_SET_SHARE_OF_REMAINDER, EAGER_WORKING_SET_CAP_BYTES)
  );
}

/**
 * The L2 (OPFS) write queue's share of the same non-cache heap remainder
 * (`computeNonCacheRemainderBytes`) — the ceiling on bytes pending background writes
 * may retain.
 *
 * NOT `max(l1Size, 64 MB)`, which is what this cap was at first (#2528). A
 * pending write's buffer is the SAME `Uint8Array` the L1 entry holds (see the
 * enqueue site in `multi-level-caching-store.ts`), so while the entry is still
 * L1-resident it costs one extra REFERENCE, not one extra buffer; the only
 * genuinely extra retention is the pending-and-since-evicted set. Charging the
 * queue for bytes L1 already owns and bounds made the cap track the wrong
 * quantity — on a machine whose L1 resolves to ~100 MB it bound at roughly half
 * the 214 MB a single pathology scene streams, which re-introduced the write
 * drops #2528 had removed (0 → 3 966, #2561).
 *
 * What this bounds is nonetheless TOTAL pending bytes, not that evicted-and-
 * pending subset. Accounting only the subset would be tighter (it is what
 * #2561's option 3 proposed) but needs an L1 eviction hook the queue does not
 * have; total-pending is a strict over-estimate of the extra retention, so it
 * is conservative in the safe direction and costs nothing but a lower ceiling.
 *
 * Resolved values (`targetHeapUsage` 0.8, so remainder = heap × 0.8 × 0.4, of
 * which this takes a quarter — i.e. simply heap × 0.08):
 *   - 4 GiB Chrome heap: remainder 1310 MiB → 328 MiB. That clears both the
 *     214 MB the cmu1 pathology scene streams before its cap bound (peak
 *     pending ≈ cumulative streamed here: its 10 961 writes drain 4-wide at
 *     tens of ms each — see the queue module's docstring — which is several
 *     times slower than the ~9.7 s arrival) and its 277 MB whole-store
 *     ceiling. The margin over that ceiling is 1.18×, not comfortable.
 *   - 2 GiB heap: remainder 655 MiB → 164 MiB.
 *   - 8 GiB heap: remainder 2621 MiB → 655 MiB, capped at 512 MiB.
 *   - `?cacheBudgetMB=2048`: remainder 1365 MiB → 341 MiB.
 *   - mobile device-class pool (384 MiB): remainder 256 MiB → 64 MiB.
 *   - no memory signal at all: 256 MiB.
 * So for that scene the cap BINDS below roughly a 2.6 GiB heap (214/0.08 =
 * 2675 MiB) and its whole store no longer fits below ~3.4 GiB (277/0.08 =
 * 3462 MiB). That is the intended heap-relative behaviour, not a regression: a
 * small heap SHOULD drop writes rather than retain a third of a gigabyte it
 * does not have. Hence no absolute floor on the measured path either — a fixed
 * 256 MiB floor on a 1 GiB heap (whose whole non-cache remainder is 328 MiB)
 * would be no bound at all. The no-signal 256 MiB is deliberately more generous
 * than a measured mid-size heap resolves to: it is a no-INFORMATION default,
 * sized to clear today's demos rather than to a heap nobody could observe.
 *
 * The absent floor has one pathological edge, left visible rather than papered
 * over: a tiny explicit pool resolves below a single chunk — `?cacheBudgetMB=1`
 * gives a 0.67 MiB remainder and a ~170 KiB allowance, under the 256 KiB a
 * chunk can reach — and because overflow drops the ARRIVAL rather than evicting
 * pending work, nothing is ever persisted. L2 is then effectively off for the
 * session. That is self-inflicted by the setting, and a floor would have to
 * exceed the whole non-cache remainder to prevent it.
 *
 * @param heapLimitBytes - Override for the device heap limit (tests). When
 *   omitted, {@link readHeapLimitBytes} is consulted; invalid explicit values
 *   skip the heap path without probing browser state.
 * @param poolOverrideBytes - Explicit total cache pool in bytes (`?cacheBudgetMB=`
 *   / the native launcher). Takes precedence over the heap-derived budget.
 * @param fallbackPoolBytes - Device-class total cache pool in bytes, used only
 *   when neither an explicit pool nor a measurable heap is available.
 */
export function computeOpfsWriteQueueBudgetBytes(
  heapLimitBytes?: number,
  poolOverrideBytes?: number,
  fallbackPoolBytes?: number
): number {
  const remainder = computeNonCacheRemainderBytes(
    heapLimitBytes,
    poolOverrideBytes,
    fallbackPoolBytes
  );
  if (remainder === undefined) return OPFS_WRITE_QUEUE_FIXED_FALLBACK_BYTES;
  return Math.floor(
    Math.min(remainder * OPFS_WRITE_QUEUE_SHARE_OF_REMAINDER, OPFS_WRITE_QUEUE_CAP_BYTES)
  );
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
 *   omitted, {@link readHeapLimitBytes} is consulted.
 * @param poolOverrideBytes - Explicit total cache pool (L0+L1+S-cache) in bytes,
 *   from `?cacheBudgetMB=` or the native launcher. Takes precedence over the
 *   heap: this is how the WKWebView app / Safari (no `performance.memory`) still
 *   get a real budget instead of the tiny fixed fallback. Split across tiers by
 *   the same rule as the heap path.
 * @param fallbackPoolBytes - Total pool to use when there is NO explicit override
 *   AND no measurable heap — typically {@link deviceClassPoolBytes}. Lets WebKit
 *   without an override still get a device-class-appropriate budget rather than
 *   the tiny fixed sizes.
 *
 * Resolution order: explicit override → measured heap → device-class fallback →
 * fixed config sizes (`source: 'fixed'`, e.g. non-browser / no device signals).
 *
 * @param enabled - Which tiers are actually active this session. A disabled tier
 *   gets a 0 budget and does NOT reserve any pool: e.g. `?no-slice-cache` frees
 *   the S-cache floor back to the chunk caches, and disabling L0 gives its share
 *   to the S-cache. Defaults to all enabled (identical to the pre-flag behavior).
 */
export interface CacheTiersEnabled {
  l0?: boolean;
  l1?: boolean;
  slice?: boolean;
}

export function computeCacheBudgets(
  heapLimitBytes?: number,
  poolOverrideBytes?: number,
  fallbackPoolBytes?: number,
  enabled: CacheTiersEnabled = {}
): CacheBudgets {
  const l0On = enabled.l0 !== false;
  const l1On = enabled.l1 !== false;
  const sliceOn = enabled.slice !== false;
  const l0Ceil = l0On ? config.cache.l0MaxSizeMB * MB : 0;
  const l1Ceil = l1On ? config.cache.l1MaxSizeMB * MB : 0;
  const sliceConfig = config.cache.sliceCacheMaxSizeMB * MB;

  // Determine the total cache pool: an explicit override wins; else derive it
  // from the measured heap; else a device-class fallback; else fixed config.
  let pool: number;
  let source: 'heap' | 'explicit' | 'device-class';
  if (positive(poolOverrideBytes)) {
    pool = poolOverrideBytes;
    source = 'explicit';
  } else {
    const heap = heapLimitBytes ?? readHeapLimitBytes();
    if (heap !== undefined) {
      pool = heap * config.dataLoading.memory.targetHeapUsage * CACHE_SHARE_OF_TARGET;
      source = 'heap';
    } else if (positive(fallbackPoolBytes)) {
      pool = fallbackPoolBytes;
      source = 'device-class';
    } else {
      return {
        l0Bytes: l0Ceil,
        l1Bytes: l1Ceil,
        sliceBytes: sliceOn ? sliceConfig : 0,
        heapAware: false,
        source: 'fixed',
      };
    }
  }

  // Heap-relative S-cache floor, never above the historical 128 MB default —
  // zero when the S-cache is disabled so it reserves nothing from the pool.
  const sliceMin = sliceOn
    ? Math.min(sliceConfig, Math.max(SLICE_ABS_MIN_BYTES, pool * SLICE_MIN_FRACTION))
    : 0;

  // Seat L0/L1 at their ceilings, but never crowd out the S-cache floor:
  // when the pool is too tight for both, shrink the chunk caches (they rebuild
  // cheaply from L2/network) before the render-proximal S-cache. Disabled tiers
  // have a 0 ceiling, so their share flows to whatever is enabled.
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
  const sliceBytes = sliceOn
    ? Math.min(SLICE_CAP_BYTES, Math.max(sliceMin, pool - l0Bytes - l1Bytes))
    : 0;

  return {
    l0Bytes: Math.floor(l0Bytes),
    l1Bytes: Math.floor(l1Bytes),
    sliceBytes: Math.floor(sliceBytes),
    heapAware: true,
    source,
  };
}
