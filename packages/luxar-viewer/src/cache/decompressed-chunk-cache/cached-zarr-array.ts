/**
 * Proxy wrapper for zarr.Array that adds L0 decompressed chunk caching.
 *
 * Uses ES6 Proxy to intercept `getChunk()` calls and check the L0 cache before
 * triggering Blosc decompression. This approach:
 * - Preserves full TypeScript type compatibility
 * - Works with any zarr.Array without modification
 * - Requires no changes to zarrita source code
 *
 * @module cache/decompressed-chunk-cache/cached-zarr-array
 */

import type * as zarr from '../../data/zarr';
import {
  DecompressedChunkCache,
  type DecodedChunkResult,
  type DecompressedChunk,
  type InflightDecode,
} from '../decompressed-chunk-cache';
import type { ResidencyProbe } from '../residency-probe';
import type { CachingStoreGetOptions } from '../multi-level-caching-store';
import { perfCounters, type PerfCounterSlot } from '../../profiling/perf-counters';
import { signalPriority, tagSignalPriority } from '../../utils/fetch-concurrency';
import { signalOrigin } from './decode-origin';
import { WARM_CHUNK_DEFAULT_ORIGIN, WARM_CHUNK_METHOD, type WarmableArray } from './warm-chunk';

export { tagSignalOrigin } from './decode-origin';
export { warmChunk, type WarmChunkOptions, type WarmableArray } from './warm-chunk';

// ---------------------------------------------------------------------------
// Perf counters (always-on; see profiling/perf-counters.ts). Resolved once at
// module scope so the per-getChunk cost is one typed-array store.
// ---------------------------------------------------------------------------
const S_L0_HITS = perfCounters.slot('l0.hits');
const S_L0_MISSES = perfCounters.slot('l0.misses');
const S_L0_COALESCED = perfCounters.slot('l0.coalesced');
const S_L0_CLONE_BYTES = perfCounters.slot('l0.cloneBytes');
const S_DECODE_COUNT = perfCounters.slot('decode.count');
const S_DECODE_BYTES = perfCounters.slot('decode.bytes');
const S_DECODE_DUPLICATES = perfCounters.slot('decode.duplicates');

/** Origin assumed when neither the call's signal nor `getOrigin` names one. */
export const DEFAULT_DECODE_ORIGIN = 'foreground';

/** Origins with a pre-resolved `decode.count.<origin>` slot. */
const KNOWN_ORIGIN_SLOTS = new Map<string, PerfCounterSlot>(
  [DEFAULT_DECODE_ORIGIN, 'shadow', 'lookahead', 'prefetch'].map((origin) => [
    origin,
    perfCounters.slot(`decode.count.${origin}`),
  ])
);

/** A re-decode of the same chunk within this window counts as a duplicate. */
export const DUPLICATE_DECODE_WINDOW_MS = 3000;

/** Bound on the decode-history map (oldest-decoded entries evicted first). */
export const DECODE_HISTORY_MAX_ENTRIES = 65_536;

/**
 * Last decode time per chunk identity (`<L0 cache id>|<L0 key>`). Insertion
 * order == last-decode order (a re-decode is deleted and re-inserted), so the
 * first key is always the oldest and eviction is O(1).
 */
const decodeHistory = new Map<string, number>();

/** Per-L0-cache-instance id: the cache is one per loaded scene (= store). */
const cacheIds = new WeakMap<DecompressedChunkCache, number>();
let nextCacheId = 0;

function cacheIdentity(cache: DecompressedChunkCache): number {
  let id = cacheIds.get(cache);
  if (id === undefined) {
    id = nextCacheId++;
    cacheIds.set(cache, id);
  }
  return id;
}

/** Forget the decode history (tests, and every perf-counter reset below). */
export function resetDecodeHistory(): void {
  decodeHistory.clear();
}
// A perf-counter reset opens a new measurement window: a decode from the
// previous window must not make this window's first decode a "duplicate".
perfCounters.onReset(resetDecodeHistory);

/** Tally one completed miss decode (count, bytes, origin, duplicate check). */
function recordDecode(historyKey: string, origin: string, bytes: number): void {
  perfCounters.add(S_DECODE_COUNT);
  perfCounters.add(S_DECODE_BYTES, bytes);
  const originSlot = KNOWN_ORIGIN_SLOTS.get(origin);
  if (originSlot === undefined) perfCounters.inc(`decode.count.${origin}`);
  else perfCounters.add(originSlot);

  const now = performance.now();
  const last = decodeHistory.get(historyKey);
  if (last !== undefined) {
    if (now - last <= DUPLICATE_DECODE_WINDOW_MS) perfCounters.add(S_DECODE_DUPLICATES);
    decodeHistory.delete(historyKey);
  } else if (decodeHistory.size >= DECODE_HISTORY_MAX_ENTRIES) {
    const oldest = decodeHistory.keys().next();
    if (!oldest.done) decodeHistory.delete(oldest.value);
  }
  decodeHistory.set(historyKey, now);
}

/** Resolve a miss's origin: call signal tag, then the wrap's thunk, then default. */
function resolveOrigin(callOptions: unknown, getOrigin: (() => string) | undefined): string {
  const signal = (callOptions as { signal?: AbortSignal } | undefined)?.signal;
  return signalOrigin(signal) ?? getOrigin?.() ?? DEFAULT_DECODE_ORIGIN;
}

/**
 * Clone an ArrayBufferView by allocating a fresh underlying buffer.
 * Handles both TypedArray (Float32Array, Uint8Array, Uint16Array,
 * BigInt64Array, etc.) and DataView. TypedArrays expose `.slice()`;
 * DataView needs an explicit buffer slice to preserve byte offsets.
 *
 * Exported only for unit testing; not part of the public package
 * surface.
 *
 * @internal
 */
export function cloneArrayBufferView(view: ArrayBufferView): ArrayBufferView {
  if (view instanceof DataView) {
    const buffer = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
    return new DataView(buffer);
  }
  // All TypedArray subtypes implement slice() returning their own subtype.
  return (view as ArrayBufferView & { slice(): ArrayBufferView }).slice();
}

/** Symbol to mark proxied arrays (for detection) */
const CACHE_MARKER = Symbol('luxar.l0cache');

/** Symbol to store original array reference */
const ORIGINAL_ARRAY = Symbol('luxar.originalArray');

/**
 * Optional per-wrap accessors consulted by the L0 proxy. All are thunks that
 * read the owning loader's transient state at `getChunk` time.
 */
export interface CachedArrayHooks {
  /**
   * Accessor for the currently-active residency probe. Called on every
   * `getChunk` to report hit/miss. Returning `null` (the default / when no
   * load is in flight) disables reporting — this is how prefetch traffic is
   * kept from contaminating a demand load's signal.
   */
  getProbe?: () => ResidencyProbe | null;
  /**
   * Accessor for the currently-active per-update `AbortSignal`. Called on
   * every `getChunk`; if it returns an aborted signal the call throws
   * (`AbortError`) BEFORE any L0 lookup, cache fetch, or Blosc decode — so a
   * superseded `updateView` bails on the warm-cache hit path too (zarrita's
   * own `throwIfAborted` only fires between chunks of a multi-chunk
   * selection). When the call itself carries no signal, this one is also the
   * caller's WAITER signal: aborting it rejects only this caller — a decode
   * shared with a still-live caller keeps running (see {@link wrapWithCache}).
   */
  getSignal?: () => AbortSignal | null;
  /**
   * Accessor naming who triggered a miss decode (`decode.count.<origin>` perf
   * counter). Consulted only on a genuine miss, and only when the call's
   * `AbortSignal` carries no {@link tagSignalOrigin} tag; defaults to
   * `'foreground'`. Instrumentation only — never changes what is fetched or
   * decoded.
   */
  getOrigin?: () => string;
}

/**
 * Wrap a zarr.Array with L0 decompressed chunk caching.
 *
 * The returned proxy intercepts `getChunk()` calls:
 * 1. Checks L0 cache for the requested chunk
 * 2. On hit: returns cached decompressed data immediately (~1μs)
 * 3. If the chunk is already being decoded — through ANY proxy over the same
 *    `cache` — joins that decode instead of starting another (the in-flight
 *    map lives on the {@link DecompressedChunkCache}, not on the proxy)
 * 4. On miss: calls original `getChunk()`, caches result, returns data
 *
 * READ-ONLY CONTRACT: every caller — the miss that decoded a chunk, the
 * waiters that joined that decode, and every later hit — receives the SAME
 * buffer L0 holds, so no consumer may mutate a chunk it is handed. zarrita
 * `get()` (`readArray`) satisfies this by construction: it copies each chunk
 * into its own freshly allocated output (`setter.setFromChunk`). A direct
 * `getChunk()` caller must copy before writing. A decoded view that does not
 * span its whole `ArrayBuffer` (e.g. an uncompressed chunk sliced out of a
 * larger shard/archive buffer) is still copied for storage, so L0 never
 * retains more bytes than it accounts for.
 *
 * Abort isolation: the shared decode runs under its own `AbortController`,
 * never under a caller's signal. Each caller races the decode against its own
 * signal (the call's `options.signal`, else `hooks.getSignal()`), and the
 * decode is cancelled only once EVERY waiter has aborted. A decode that is not
 * fully abandoned completes and is cached; so does an abandoned one that the
 * cancellation could no longer stop.
 *
 * All other zarr.Array properties and methods pass through unchanged.
 *
 * @param array - Original zarr.Array to wrap
 * @param cache - L0 decompressed chunk cache instance
 * @param arrayPath - Full path to the array (used for cache key generation)
 * @param hooks - Optional residency-probe / abort-signal / decode-origin
 *   accessors (see {@link CachedArrayHooks}).
 * @returns Proxied zarr.Array with L0 caching enabled
 *
 * @example
 * ```typescript
 * const cache = new DecompressedChunkCache({ maxSize: 200 * 1024 * 1024 });
 *
 * // Wrap array after opening
 * const rawArray = await zarr.open(location.resolve('positions'), { kind: 'array' });
 * const cachedArray = wrapWithCache(rawArray, cache, '/scene/points/positions');
 *
 * // Use normally - getChunk() now checks L0 cache
 * const chunk = await cachedArray.getChunk([0, 1, 2]);
 * // First call: decompresses and caches
 * // Subsequent calls: returns from L0 cache instantly
 * ```
 */
export function wrapWithCache<D extends zarr.DataType>(
  array: zarr.Array<D, zarr.Readable>,
  cache: DecompressedChunkCache,
  arrayPath: string,
  hooks: CachedArrayHooks = {}
): zarr.Array<D, zarr.Readable> {
  // Don't double-wrap
  if (isCachedArray(array)) {
    return array;
  }
  const { getProbe, getSignal, getOrigin } = hooks;
  const ctx: AcquireContext = {
    cache,
    historyPrefix: `${cacheIdentity(cache)}|`,
  };

  return new Proxy(array, {
    get(target, prop, _receiver) {
      // Handle cache marker check
      if (prop === CACHE_MARKER) {
        return true;
      }

      // Handle original array access
      if (prop === ORIGINAL_ARRAY) {
        return target;
      }

      // CRITICAL: Direct access for properties that use private fields internally.
      // zarrita's getters (attrs, shape, dtype, etc.) access private fields like #metadata.
      // Even with Reflect.get(target, prop, target), the getter can fail because the
      // property descriptor is retrieved from the proxy, not the original object.
      // Solution: Access these properties directly on target, bypassing Reflect entirely.
      if (DIRECT_PROPS.has(prop)) {
        return (target as unknown as Record<string, unknown>)[prop as string];
      }

      // Intercept getChunk() to add caching
      if (prop === 'getChunk') {
        return async function (
          ...args: Parameters<typeof target.getChunk>
        ): Promise<ChunkResult<D>> {
          const [chunkCoords, callOptions, opts] = args;
          // Per-update abort chokepoint: bail BEFORE the L0 lookup / fetch /
          // Blosc decode if the owning load was superseded. This covers the
          // warm-cache hit and coalesced paths, which short-circuit before
          // zarrita's between-chunk throwIfAborted would run.
          const activeSignal = getSignal?.() ?? undefined;
          activeSignal?.throwIfAborted();
          const result = await acquireChunk(ctx, {
            key: DecompressedChunkCache.makeKey(arrayPath, chunkCoords),
            // This caller's own abort: the call's signal, else the loader's
            // per-update one. Only THIS waiter bails on it (see raceWaiter).
            signal: callOptions?.signal ?? activeSignal,
            demand: true,
            probe: getProbe?.() ?? null,
            origin: () => resolveOrigin(callOptions, getOrigin),
            run: (signal) => target.getChunk(chunkCoords, { ...callOptions, signal }, opts),
          });
          return result as ChunkResult<D>;
        };
      }

      // Cache warm-up (see warm-chunk.ts): decode + cache through the shared
      // in-flight map, but with the CALLER'S signal only — never `getSignal`
      // (the demand load's) — no residency-probe record, no output assembly,
      // and as prefetch traffic: no L0 hit/miss statistic, and a store read
      // the multi-level store neither counts as demand nor prefetches from.
      if (prop === WARM_CHUNK_METHOD) {
        const warm: WarmableArray['warmChunk'] = async (chunkCoords, options = {}) => {
          const { signal } = options;
          await acquireChunk(ctx, {
            key: DecompressedChunkCache.makeKey(arrayPath, chunkCoords),
            signal,
            demand: false,
            probe: null,
            origin: () => options.origin ?? signalOrigin(signal) ?? WARM_CHUNK_DEFAULT_ORIGIN,
            run: (decodeSignal) => {
              const read: CachingStoreGetOptions = { signal: decodeSignal, suppressPrefetch: true };
              return target.getChunk(chunkCoords, read);
            },
          });
        };
        return warm;
      }

      // Pass through all other property access
      // CRITICAL: Use `target` as receiver, NOT `receiver` (the proxy)!
      // zarrita uses private class fields (e.g., #e, #store). When getters
      // access private fields, `this` must be the original object, not the proxy.
      // Using `receiver` (the proxy) causes: "Cannot read private member #e"
      return Reflect.get(target, prop, target);
    },
  }) as zarr.Array<D, zarr.Readable>;
}

/** zarrita getters that read private fields and must bypass `Reflect.get`. */
const DIRECT_PROPS = new Set<PropertyKey>([
  'attrs',
  'shape',
  'dtype',
  'chunks',
  'order',
  'fill_value',
  'fillValue',
  'dimensionNames',
  'compressor',
  'filters',
  'codec',
  'codecs',
]);

type ChunkResult<D extends zarr.DataType> = {
  data: zarr.TypedArray<D>;
  shape: number[];
  stride: number[];
};

/** Per-wrap state shared by every call through one proxy. */
interface AcquireContext {
  cache: DecompressedChunkCache;
  /** `<L0 cache id>|` — prefix of the decode-history identity. */
  historyPrefix: string;
}

/** One L0 acquisition: a lookup that joins or starts a decode on a miss. */
interface AcquireRequest {
  key: string;
  /** This waiter's own abort signal (never the shared decode's). */
  signal: AbortSignal | undefined;
  /** `false` for a warm-up: the lookup moves no L0 hit/miss statistic. */
  demand: boolean;
  /** Residency probe to record hit/miss into, or `null` for none. */
  probe: ResidencyProbe | null;
  /** Origin of a miss decode — resolved only on a genuine miss. */
  origin: () => string;
  /** Run the underlying fetch + decode under the shared decode's signal. */
  run: (signal: AbortSignal) => Promise<DecodedChunkResult>;
}

/**
 * The L0 core: hit → return the cached chunk; in flight → join the cache-wide
 * decode as another waiter; miss → start a decode every proxy over this cache
 * can join.
 */
function acquireChunk(ctx: AcquireContext, req: AcquireRequest): Promise<DecodedChunkResult> {
  const { cache } = ctx;
  // An already-aborted waiter must not join (its abort listener would never fire).
  if (req.signal?.aborted) return Promise.reject(req.signal.reason as Error);
  const cached = cache.get(req.key, { countStats: req.demand });
  if (cached) {
    // Resident: served from L0 with no fresh fetch/decode.
    perfCounters.add(S_L0_HITS);
    req.probe?.record(true);
    return Promise.resolve({ data: cached.data, shape: cached.shape, stride: cached.stride });
  }

  // Same-chunk coalescing: another caller (through ANY proxy over this cache)
  // is already decoding it. No fresh Blosc work is triggered for this caller,
  // so the coalesced wait counts as a hit for residency.
  const pending = cache.getInflight(req.key);
  if (pending?.abandoned) return awaitAbandoned(ctx, req, pending);
  if (pending) {
    perfCounters.add(S_L0_COALESCED);
    req.probe?.record(true);
    pending.waiters++;
    raiseSharedDecodePriority(pending, req.signal);
    return raceWaiter(pending, req.signal);
  }

  // Genuine miss: a fetch + Blosc decode is about to run.
  perfCounters.add(S_L0_MISSES);
  req.probe?.record(false);
  const entry = startDecode(ctx, req);
  return raceWaiter(entry, req.signal);
}

/**
 * A caller arriving while an ABANDONED decode of its chunk is still settling
 * (every earlier waiter aborted, see `InflightDecode.abandoned`). If the abort
 * landed after the bytes arrived the decode completes and commits, and this
 * caller takes that result — one decode, not two. If it really was cancelled,
 * the entry is deregistered by the time this caller sees the rejection, so it
 * retries as a fresh acquisition. Its own abort still releases it at once.
 */
async function awaitAbandoned(
  ctx: AcquireContext,
  req: AcquireRequest,
  pending: InflightDecode
): Promise<DecodedChunkResult> {
  try {
    const result = await raceWaiter(pending, req.signal);
    perfCounters.add(S_L0_COALESCED);
    req.probe?.record(true);
    return result;
  } catch (error) {
    if (req.signal?.aborted || !isCancellation(error, pending)) throw error;
    return acquireChunk(ctx, req);
  }
}

/**
 * Whether `error` is the abandoned decode's own cancellation: the reason its
 * controller was aborted with (fetch rejects with the signal's reason, which
 * may be any value), or an `AbortError` raised by the read it cancelled.
 */
function isCancellation(error: unknown, pending: InflightDecode): boolean {
  return (
    error === pending.controller.signal.reason ||
    (error as { name?: unknown } | null)?.name === 'AbortError'
  );
}

/**
 * A caller joining an in-flight decode lifts that decode's fetch class to its
 * own (an untagged caller is `demand`), so a frame waiting on a chunk that a
 * refinement read started does not queue behind the refinement class.
 */
function raiseSharedDecodePriority(entry: InflightDecode, signal: AbortSignal | undefined): void {
  const shared = signalPriority(entry.controller.signal);
  if (shared) shared.raise(signalPriority(signal)?.value ?? 'demand');
}

/** Mutable view of an {@link InflightDecode} while it is being built. */
type InflightBuilder = { -readonly [K in keyof InflightDecode]: InflightDecode[K] };

/**
 * Start the shared decode for `req.key` under its own controller and register
 * it on the cache. Only a still-registered entry commits its result, so a
 * decode orphaned by `cache.clear()` is dropped. A fully abandoned one stays
 * registered until it settles, so if its cancellation came too late to stop
 * it (the bytes had already arrived) its result is still cached.
 */
function startDecode(ctx: AcquireContext, req: AcquireRequest): InflightDecode {
  const { cache, historyPrefix } = ctx;
  const origin = req.origin();
  const entry: InflightBuilder = {
    controller: new AbortController(),
    waiters: 1,
    promise: undefined as unknown as Promise<DecodedChunkResult>,
  };
  // The decode fetches under its OWN signal, so carry the starting caller's
  // fetch class onto it (a refinement read stays `refinement` at the store).
  // A fresh cell, not the caller's: a later demand joiner raises THIS decode
  // only, never every read of the caller's run.
  const callerPriority = signalPriority(req.signal);
  if (callerPriority) tagSignalPriority(entry.controller.signal, callerPriority.value);
  entry.promise = (async () => {
    // Cache miss — call original getChunk (triggers Blosc decompression)
    const chunk = await req.run(entry.controller.signal);
    // `?? 0`: an object-dtype chunk is a plain array (no byteLength).
    const decodedBytes = (chunk.data as { byteLength?: number }).byteLength ?? 0;
    recordDecode(historyPrefix + req.key, origin, decodedBytes);
    if (cache.getInflight(req.key) === entry) {
      cache.set(req.key, toCacheEntry(chunk));
    }
    return chunk;
  })();
  // Deregister on settle. `then(f, f)` also marks a rejection as handled, so a
  // fully-abandoned decode's AbortError is never an unhandled rejection.
  const forget = (): void => cache.deleteInflight(req.key, entry);
  entry.promise.then(forget, forget);
  cache.setInflight(req.key, entry);
  return entry;
}

/** True when `view` covers its whole underlying buffer (aliasing retains nothing extra). */
function spansWholeBuffer(view: ArrayBufferView): boolean {
  return view.byteOffset === 0 && view.byteLength === view.buffer.byteLength;
}

/**
 * Build the L0 entry for a freshly decoded chunk: the decoded view itself (the
 * read-only contract, see {@link wrapWithCache}), copied only when it does not
 * span its whole buffer so L0 never retains bytes it does not account for.
 * cloneArrayBufferView branches on TypedArray vs DataView — the union type's
 * public d.ts doesn't expose a common slice() so a single cast hid the DataView
 * gap. zarrita's TypedArray<D> union includes object-dtype as unknown[], which
 * is not an ArrayBufferView; cast through ArrayBufferView since numeric chunks
 * (the only kind we cache) always are.
 */
function toCacheEntry(chunk: DecodedChunkResult): DecompressedChunk {
  const view = chunk.data as ArrayBufferView;
  let data = view;
  if (!spansWholeBuffer(view)) {
    data = cloneArrayBufferView(view);
    perfCounters.add(S_L0_CLONE_BYTES, (data as { byteLength?: number }).byteLength ?? 0);
  }
  return { data, shape: chunk.shape.slice(), stride: chunk.stride.slice() };
}

/**
 * One waiter's view of a shared decode: settles with the decode, or rejects
 * with THIS waiter's abort reason as soon as its own signal fires. A waiter
 * that aborts leaves the decode running for the others; the last one to leave
 * aborts the shared controller and marks the entry abandoned (it deregisters
 * on settle; a later caller waits for that outcome and starts a fresh decode
 * only if it was really cancelled). A waiter without a signal never leaves,
 * so its decode is never cancelled.
 */
function raceWaiter(
  entry: InflightDecode,
  signal: AbortSignal | undefined
): Promise<DecodedChunkResult> {
  if (!signal) return entry.promise;
  return new Promise<DecodedChunkResult>((resolve, reject) => {
    const onAbort = (): void => {
      entry.waiters--;
      if (entry.waiters <= 0 && !entry.abandoned) {
        // The last waiter left: cancel the shared decode, but keep it
        // registered until it settles (see `InflightDecode.abandoned`).
        entry.abandoned = true;
        entry.controller.abort(signal.reason);
      }
      reject(signal.reason as Error);
    };
    signal.addEventListener('abort', onAbort, { once: true });
    entry.promise.then(
      (chunk) => {
        signal.removeEventListener('abort', onAbort);
        resolve(chunk);
      },
      (error: unknown) => {
        signal.removeEventListener('abort', onAbort);
        reject(error as Error);
      }
    );
  });
}

/**
 * Check if an array is already wrapped with L0 caching.
 *
 * @param array - Array to check
 * @returns true if array is wrapped with cache proxy
 */
export function isCachedArray(array: unknown): boolean {
  if (array === null || typeof array !== 'object') {
    return false;
  }
  return (array as Record<symbol, unknown>)[CACHE_MARKER] === true;
}

/**
 * Get the underlying unwrapped zarr.Array from a cached proxy.
 *
 * Useful when you need direct access to the original array without caching.
 *
 * @param array - Possibly cached zarr.Array
 * @returns Original unwrapped array, or the input if not cached
 */
export function unwrapCachedArray<D extends zarr.DataType>(
  array: zarr.Array<D, zarr.Readable>
): zarr.Array<D, zarr.Readable> {
  if (!isCachedArray(array)) {
    return array;
  }
  return (array as unknown as Record<symbol, unknown>)[ORIGINAL_ARRAY] as zarr.Array<
    D,
    zarr.Readable
  >;
}
