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
const NON_CACHE_SHARE_OF_TARGET = 1 - CACHE_SHARE_OF_TARGET;
const EAGER_WORKING_SET_SHARE_OF_REMAINDER = 0.5;
const EAGER_WORKING_SET_CAP_BYTES = 512 * MB;
/** Historical line-admission fallback when no session memory signal exists. */
const EAGER_WORKING_SET_FIXED_FALLBACK_BYTES = 256 * MB;

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
 */
const DEVICE_CLASS_POOL_BYTES = {
  mobile: 384 * MB,
  laptop: 1024 * MB,
  desktop: 2048 * MB,
} as const;

/**
 * Logical-core count at/above which a NON-mobile device is treated as a desktop
 * rather than a laptop. This is a deliberately WEAK proxy: laptop vs desktop is
 * not reliably distinguishable in-browser — there is no RAM API in WebKit
 * (`navigator.deviceMemory` is Chromium-only), no battery API in Safari, and
 * UA / core counts overlap (an 8-core MacBook vs an 8-core Mac mini). The
 * industry consensus is that only an "educated guess" is possible. Both the
 * laptop and desktop budgets are safe on the 8 GB+ machines that run the
 * desktop app, so a misclassification is low-consequence.
 */
const DESKTOP_CORE_THRESHOLD = 12;

function positive(value: number | undefined): value is number {
  return value != null && Number.isFinite(value) && value > 0;
}

/** Convert a positive cache-pool override from MiB to bytes. */
export function cachePoolOverrideBytes(
  cacheBudgetMB: number | null | undefined
): number | undefined {
  return cacheBudgetMB != null && cacheBudgetMB > 0 ? cacheBudgetMB * MB : undefined;
}

/** Device signals used by {@link inferDeviceClass} (injectable for tests). */
export interface DeviceSignals {
  userAgent: string;
  maxTouchPoints: number;
  coarsePointer: boolean;
  cores: number;
}

/** Read device signals from the browser (best-effort; safe in non-browser envs). */
function readDeviceSignals(): DeviceSignals {
  const nav = typeof navigator !== 'undefined' ? navigator : undefined;
  const coarsePointer =
    typeof matchMedia === 'function' ? matchMedia('(pointer: coarse)').matches : false;
  return {
    userAgent: nav?.userAgent ?? '',
    maxTouchPoints: nav?.maxTouchPoints ?? 0,
    coarsePointer,
    cores: nav?.hardwareConcurrency ?? 0,
  };
}

/**
 * Infer a coarse device class from browser signals. `mobile` is reliable
 * (mobile UA, or a touch + coarse-pointer device — which also catches iPadOS,
 * whose UA masquerades as macOS). `desktop` vs `laptop` is the weak core-count
 * proxy (see {@link DESKTOP_CORE_THRESHOLD}).
 */
export function inferDeviceClass(
  signals: DeviceSignals = readDeviceSignals()
): 'mobile' | 'laptop' | 'desktop' {
  const mobileUA = /Mobi|Android|iPhone|iPod|iPad/i.test(signals.userAgent);
  if (mobileUA || (signals.maxTouchPoints > 0 && signals.coarsePointer)) return 'mobile';
  return signals.cores >= DESKTOP_CORE_THRESHOLD ? 'desktop' : 'laptop';
}

/**
 * Device-class cache pool (bytes), or `undefined` outside a browser (node/tests)
 * so callers fall through to the fixed config sizes there rather than a guess.
 */
export function deviceClassPoolBytes(): number | undefined {
  if (typeof navigator === 'undefined') return undefined;
  return DEVICE_CLASS_POOL_BYTES[inferDeviceClass()];
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
 * The eager child loader's share of the heap headroom left outside the cache
 * pool. Half of that shared remainder stays available for render, WASM,
 * prefetch, and unrelated scene-graph work, while the absolute cap prevents a
 * large V8 heap limit from recreating an eight-wide allocation spike.
 *
 * Resolution matches the cache pool: explicit pool override → measured heap →
 * device-class pool → fixed fallback. Pool-based inputs are converted back to
 * their corresponding non-cache share so a WebKit session uses one coherent
 * memory model for caches and eager line loading.
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
  const budgetFromCachePool = (poolBytes: number): number =>
    poolBytes *
    (NON_CACHE_SHARE_OF_TARGET / CACHE_SHARE_OF_TARGET) *
    EAGER_WORKING_SET_SHARE_OF_REMAINDER;

  let budgetBytes: number;
  if (positive(poolOverrideBytes)) {
    budgetBytes = budgetFromCachePool(poolOverrideBytes);
  } else {
    const heap = heapLimitBytes === undefined ? readHeapLimitBytes() : heapLimitBytes;
    if (positive(heap)) {
      budgetBytes =
        heap *
        config.dataLoading.memory.targetHeapUsage *
        NON_CACHE_SHARE_OF_TARGET *
        EAGER_WORKING_SET_SHARE_OF_REMAINDER;
    } else if (positive(fallbackPoolBytes)) {
      budgetBytes = budgetFromCachePool(fallbackPoolBytes);
    } else {
      return EAGER_WORKING_SET_FIXED_FALLBACK_BYTES;
    }
  }
  return Math.floor(Math.min(budgetBytes, EAGER_WORKING_SET_CAP_BYTES));
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
