# Luxar Viewer Cache Package

Three-level caching system with intelligent prefetching for zarr chunks enabling offline viewing, instant reloads, and reduced bandwidth.

## Overview

This package implements a transparent caching and prefetching layer for zarr datasets:

- **L0 (Decompressed)**: 200MB LRU cache for decoded TypedArrays (eliminates Blosc decompression)
- **L1 (Memory)**: 100MB segmented LRU cache with metadata protection
- **L2 (OPFS)**: 2GB persistent storage surviving browser restarts
- **Intelligent Prefetching**: Proactive loading of adjacent chunks to hide network latency
- **Content-hash validation**: Automatic cache invalidation when data changes
- **Zero overhead**: Stores raw compressed chunks (no re-compression)

## Cache Hierarchy

```
Request → L0 (Decompressed) → L1 (Memory) → L2 (OPFS) → Remote HTTP
              ↓                   ↓             ↓            ↓
           ~1μs               ~1μs+2ms       ~1ms+2ms     ~100ms+2ms
       (no decompress)    (decompress)   (decompress)   (decompress)
```

| Level | Storage | Speed         | Size  | Persistence  | Content           |
| ----- | ------- | ------------- | ----- | ------------ | ----------------- |
| L0    | Memory  | ~1μs          | 200MB | Session only | Decompressed data |
| L1    | Memory  | ~1μs + ~2ms\* | 100MB | Session only | Compressed chunks |
| L2    | OPFS    | ~1ms + ~2ms\* | 2GB   | Permanent    | Compressed chunks |
| L3    | Remote  | ~100ms        | ∞     | N/A          | Compressed chunks |

\*~2ms is Blosc decompression time per chunk (skipped on L0 hit)

## Quick Start

The cache is **enabled by default** and works transparently through the Luxar Zarr facade:

```typescript
import { MultiLevelCachingStore } from '../cache/multi-level-caching-store';
import * as zarr from '../data/zarr';

// Create caching store
const store = new MultiLevelCachingStore('https://example.com/dataset.zarr', {
  l1MaxSize: 100 * 1024 * 1024, // 100MB (optional)
  l2MaxSize: 2 * 1024 * 1024 * 1024, // 2GB (optional)
  debug: false, // Set true for verbose logging (optional)
});

await store.init();

// Open through the Luxar Zarr facade - completely transparent!
const zarrStore = await zarr.openStore(store);
const root = await zarr.openGroup(zarrStore);

// First load: HTTP → L2 → L1 → decompress → L0 → render
// Second load: L0 → render (~1μs, no decompression!)
// After reload: L2 → L1 → decompress → L0 → render (~1ms)

// Clean up when done
await store.dispose();
```

## URL Parameters

Override cache behavior via URL parameters:

- `?no-cache` - Disable all caching (L0 + L1 + L2) for this session
- `?cache-debug` - Enable verbose cache logging for all layers
- `?clear-cache` - Clear all caches (L0 + L1 + L2) before loading dataset
- `?no-prefetch` - Disable prefetching (caches still active)
- `?prefetch-debug` - Enable verbose prefetch logging
- `?cache-stats` - Auto-open the data-loading monitor expanded on the Cache tab

Example:

```
http://localhost:5173/?src=http://example.com/data.zarr&cache-debug&prefetch-debug
```

## Cache invariants and lifecycle

These contracts span every tier and are enforced by the unit tests in
`tests/unit/cache/invalidation-chain.test.ts` and friends:

### Lifecycle

- **`MultiLevelCachingStore.dispose()`** is async. It sets a `disposed`
  flag (synchronously short-circuiting `getResult`), aborts the
  store-level `dataAbort` controller (cancelling in-flight prefetch /
  demand fetches), tears down the prefetcher, and awaits L2 dispose.
- **`SceneLoader.dispose()`** is async and awaited by `loadScene` before
  the next caching store is constructed. Sync callers (the
  `beforeunload` path, `SceneLoaderManager.destroyLoader`) keep
  working — the returned promise just unwinds in the background.
- **`SceneLoaderManager.destroyLoaderAsync` / `destroyAllAsync`** are
  awaitable variants for callers that need deterministic teardown
  (dataset switches, tests).

### Invalidation

- `clearAll()` invokes every registered `onInvalidate` callback so L0
  (wired through `cache-setup.ts`) is cleared alongside L1 and L2.
- `clearL1()` and `clearL2()` are single-tier ops — they do NOT fire
  invalidation callbacks. L0 stays populated.
- Content-hash mismatch in `doValidateCache` defensively clears L1
  alongside L2, then fans out via `onInvalidate` to L0.

### OPFS mutation ordering

- **Generation token**: every `clear()` bumps a counter. A `set()` in
  flight when `clear()` runs detects the mismatch on completion and
  skips its index update (best-effort delete the just-written file).
- **Pending writes drain**: `clear()` awaits in-flight `pendingWrites`
  via `Promise.allSettled` before resetting state.
- **Disposed flag**: post-`dispose()` `set/get/touch` are no-ops.
- **In-flight metadata save tracking**: `dispose()` awaits
  `metadataSaveInFlight` before the final flush.

### L0 read-only chunk contract

Cached zarr chunks are returned by reference on the L0 hit path.
Loaders MUST treat chunk data as immutable input and copy into
accumulator/output buffers before mutating. See the dedicated section
under "L0 Decompressed Chunk Cache" below.

## Cache health and validation modes

`MultiLevelCachingStore.getStats()` returns a `health` field with:

- `validationMode: 'content-hash' | 'ttl' | 'none'`
  - `content-hash`: Luxar dataset with `content_hash` attr — strongest
    invalidation guarantee.
  - `ttl`: External dataset; cache is invalidated after
    `cache.externalDatasetTtlMs` elapses.
  - `none`: External dataset, no TTL configured — cache may be stale
    indefinitely until manually cleared. The cache tab surfaces this
    as an `unvalidated-external-dataset` badge.
- `lastValidatedAt: number | null`: wall-clock millis at last
  successful validation.
- `unvalidatedExternalDataset: boolean`: convenience flag.

`OPFSStore.getStats()` exposes health counters:

- `oversizedWriteSkipped`: entry larger than `maxSize` was rejected.
- `quotaWriteSkipped`: browser reported insufficient quota.
- `evictions`: own-LRU evictions to make room for incoming writes.
- `writeFailures`: I/O exceptions during `set()`.
- `corruptedEntries`: get() detected a size mismatch and removed the
  bad entry.
- `metadataParseFailures`: `_cache_meta.json` could not be parsed.
- `orphanedFilesRemoved`: files reclaimed by `cleanupOrphans()` (run
  on metadata parse failure).

## Cache Status Badges

The data monitor's cache tab renders a row of small pill badges
summarising the cache's operational state. Each badge is also exposed
on `CacheMetrics.status: CacheStatusBadge[]` so programmatic consumers
(debug snapshots, E2E tests) can assert on the same set.

| Badge                          | Meaning                                                       | Source                                                            |
| ------------------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------- |
| `cache-enabled`                | Caching is wired and operational.                             | `telemetryState.kind === 'enabled'`                               |
| `no-cache`                     | The `?no-cache` URL flag is set; all tiers disabled.          | `telemetryState.kind === 'disabled-no-cache'`                     |
| `disabled-config`              | App config disabled caching (e.g. `cache.enabled: false`).    | `telemetryState.kind === 'disabled-config'`                       |
| `opfs-unavailable`             | The browser does not expose OPFS; L2 is disabled.             | OPFS provider absent                                              |
| `quota-constrained`            | L2 has skipped at least one write because of browser quota.   | `l2.quotaWriteSkipped > 0`                                        |
| `cache-errors-detected`        | L2 has accumulated I/O / corruption failures.                 | `l2.writeFailures + corruptedEntries + metadataParseFailures > 0` |
| `unvalidated-external-dataset` | External dataset, no TTL configured — entries may stay stale. | `health.unvalidatedExternalDataset === true`                      |
| `provider-missing`             | Telemetry says cache is enabled but no provider is attached.  | classifier/provider contradiction                                 |

Several badges may coexist — for example, a Luxar dataset on a near-full
browser disk can show `cache-enabled` + `quota-constrained` simultaneously.
The cache tab also shows a dedicated **Cache Health** section with the
current validation mode and last-validated timestamp, and an inline
**Errors** card on the L2 section when any of the four OPFS error
counters is non-zero.

## Privacy / local persistence

The L2 OPFS layer **persists dataset bytes locally in the browser** for
this origin. Cleared by:

- `__luxarDebug.cache.clearL2()` from the JS console
- The cache tab's **Clear L2** button (with confirmation)
- The user's browser data-clearing UI (origin-wide)
- `?clear-cache` URL param at the next page load
- A content-hash mismatch (Luxar datasets) or TTL expiry (external
  datasets)

OPFS storage is sandboxed per origin. Cross-origin pages cannot read
this cache. Private/incognito browser windows typically expose a
reduced-quota OPFS that wipes on tab close — the cache degrades to L1

- network with no persistence.

## OPFS availability and quota

OPFS is available in modern Chrome/Edge/Safari/Firefox. Older browsers
fall back to L1-only (no L2 persistence). The cache tab surfaces:

- `cache-enabled`: caching is operational.
- `no-cache`: `?no-cache` URL flag set.
- `disabled-config`: app config disabled the cache.
- `quota-constrained`: at least one write was skipped due to quota.
- `cache-errors-detected`: writeFailures / corruptedEntries /
  metadataParseFailures > 0.
- `unvalidated-external-dataset`: external dataset, no TTL configured.
- `provider-missing`: telemetry says enabled but no provider wired
  (scene transition mid-flight or wiring bug).

## L0 Decompressed Chunk Cache

The L0 cache is the **fastest cache layer**, storing already-decoded TypedArrays (Float32Array, Uint8Array, etc.) to eliminate Blosc decompression overhead.

### Why L0 Matters

Without L0, even an L1 cache hit requires ~2ms for Blosc decompression. For a typical view update accessing 4 attributes (positions, colors, radii, sharpness):

- **Without L0**: ~8ms decompression overhead per view update
- **With L0**: ~0.004ms (essentially zero)

This makes the difference between 120 FPS and 60 FPS during navigation.

### Usage

L0 caching is **enabled by default** and integrated into the scene loading pipeline:

```typescript
import { DecompressedChunkCache } from '../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../cache/decompressed-chunk-cache/cached-zarr-array';

// L0 is automatically enabled when loading scenes via SceneLoader
// Manual usage (advanced):
const l0Cache = new DecompressedChunkCache({
  maxSize: 200 * 1024 * 1024, // 200MB (default)
  debug: false,
});

// Wrap zarr arrays to enable L0 caching
const cachedArray = wrapWithCache(zarrArray, l0Cache, '/points/positions');

// First getChunk(): decompress + cache in L0
const chunk1 = await cachedArray.getChunk([0, 1, 2]);

// Second getChunk(): instant from L0 (~1μs, no decompression)
const chunk2 = await cachedArray.getChunk([0, 1, 2]);
```

### API

**DecompressedChunkCache**

```typescript
const cache = new DecompressedChunkCache({
  maxSize?: number,  // Max size in bytes (default: 200MB)
  debug?: boolean,   // Enable debug logging (default: false)
});

// Get cache statistics
const stats = cache.getStats();
// { size, count, hits, misses, evictions, hitRate }

// Clear the cache
cache.clear();
```

**wrapWithCache**

```typescript
// Wrap a zarr.Array with L0 caching
const cached = wrapWithCache(array, cache, arrayPath);

// Check if array is wrapped
isCachedArray(cached); // true

// Get original unwrapped array
unwrapCachedArray(cached);
```

### Read-only chunk contract

Cached zarr chunks are **read-only** to every loader downstream of `wrapWithCache`.

The L0 cache stores the same `ArrayBufferView` it returns to subsequent
hit-path callers — no defensive clone on hit. Mutating that view in
place would corrupt the cached entry for the next caller. Loaders MUST
treat chunk data as immutable input and copy into accumulator buffers,
output `BufferAttribute` allocations, or fresh `TypedArray`s before
mutating.

In practice, viewer loaders never call `getChunk()` directly; they go
through `zarr.get()` which combines multiple chunks into a fresh result
buffer. Mutation concerns therefore only apply to lower-level code that
might wire `getChunk()` results into a render path. If you find a
loader that does mutate `chunk.data`, either copy first or — if the
mutation is unavoidable — switch the wrapped array to skip caching for
that path.

The miss-path clone in `wrapWithCache` (via `cloneArrayBufferView`)
gives the _first caller_ a private buffer they can technically mutate
without poisoning the cache, but this should not be relied upon: the
contract is "read-only on every path" so future cache changes (e.g.
removing the miss-path clone for a perf win) don't break loaders.

## Intelligent Prefetching

The cache includes an **intelligent prefetching system** that proactively loads adjacent chunks to hide network latency.

### How It Works

When a chunk is accessed from L2 (OPFS) or L3 (HTTP):

1. Parse chunk indices from the key (e.g., `positions/1.2.3` → `[1, 2, 3]`)
2. Calculate adjacent chunks (±1 in each dimension)
3. Queue them for background prefetching
4. Limit to 4 concurrent prefetches (leaves bandwidth for normal requests)

### Example

```
User requests: positions/1.2.3
             ↓
Cache prefetches (in background):
  positions/0.2.3  (dim 0: -1)
  positions/2.2.3  (dim 0: +1)
  positions/1.1.3  (dim 1: -1)
  positions/1.3.3  (dim 1: +1)
  positions/1.2.2  (dim 2: -1)
  positions/1.2.4  (dim 2: +1)
```

### Benefits

- **Reduced Latency**: Adjacent chunks pre-loaded before explicit request
- **Transparent**: No API changes required, works automatically
- **Bandwidth-Friendly**: Limited concurrency prevents congestion
- **Non-Blocking**: Fire-and-forget pattern, never blocks normal requests

### Configuration

Prefetching is **enabled by default** and configurable via ChunkPrefetcher options:

```typescript
import { ChunkPrefetcher } from '../cache/chunk-prefetcher';

const prefetcher = new ChunkPrefetcher(store, {
  maxConcurrent: 4, // Max concurrent prefetch requests
  enabled: true, // Enable/disable prefetching
  debug: false, // Enable debug logging
});

store.setPrefetcher(prefetcher);
```

For full details, see [`docs/guides/specs/CACHE_PREFETCHING_SPEC.md`](../../../../docs/guides/specs/CACHE_PREFETCHING_SPEC.md).

## API Reference

### MultiLevelCachingStore

Main class implementing zarrita's `AsyncReadable` interface.

#### Constructor

```typescript
new MultiLevelCachingStore(baseUrl: string, options?: {
  l1MaxSize?: number;      // L1 size in bytes (default: 100MB)
  l2MaxSize?: number;      // L2 size in bytes (default: 2GB)
  debug?: boolean;         // Enable debug logging (default: false)
  urlParams?: URLSearchParams;  // For testing (optional)
})
```

#### Methods

**`async init(): Promise<void>`**

Initialize OPFS storage and validate cache. Must be called before first use.

**`async get(key: string): Promise<Uint8Array | undefined>`**

Get a zarr chunk with L1 → L2 → HTTP cascade. Implements zarrita's AsyncReadable interface.

**`getStats(): MultiLevelCacheStats`**

Get the aggregated multi-tier statistics snapshot consumed by the
data-loading monitor, debug overlay, and cache E2E suite:

```typescript
{
  l1: {
    metadataSize: number,    // Bytes in metadata segment
    chunksSize: number,      // Bytes in chunks segment
    metadataCount: number,   // Files in metadata segment
    chunksCount: number,     // Files in chunks segment
    hits: number,            // Total cache hits
    misses: number,          // Total cache misses
    evictions: number        // Total evictions
  },
  l2: {
    size: number,            // Total bytes in OPFS
    count: number,           // Total files in OPFS
    reads: number,           // Total reads from L2
    writes: number,          // Total writes to L2
    misses: number           // L2 lookup misses
    // plus health counters: available, oversizedWriteSkipped,
    // quotaWriteSkipped, evictions, writeFailures,
    // corruptedEntries, metadataParseFailures, orphanedFilesRemoved
  },
  network: {
    bytesTransferred: number,   // Total bytes fetched from network (L3 only)
    requestCount: number,       // Total HTTP requests (incl. prefetch)
    bandwidth: number,          // Current bandwidth (bytes/sec, sliding window)
    totalBytesServed: number,   // Cumulative bytes delivered to demand callers
                                // across ALL tiers (L1 + L2 + network). Stays
                                // non-zero on a warm/cache-served reload where
                                // bytesTransferred is 0 — drives the monitor's
                                // "DATA LOADED" card.
    totalRequestsServed: number // Demand reads served across all tiers
  },
  demand: {
    l1Hits: number,           // User-demand L1 hits (excludes prefetch)
    l2Hits: number,           // User-demand L2 hits (excludes prefetch)
    networkRequests: number   // User-demand network requests
  },
  health: {
    validationMode: 'content-hash' | 'ttl' | 'none',
    lastValidatedAt: number | null,
    unvalidatedExternalDataset: boolean,
    opfsAvailable: boolean
  },
  clearOnInitCount: number    // Times ?clear-cache fired on init
}
```

See `types.ts` (`MultiLevelCacheStats`) for the authoritative shape.

**`clearL1(): void`**

Clear L1 memory cache only (L2 persists).

**`async clearL2(): Promise<void>`**

Clear L2 OPFS cache only (L1 untouched).

**`async clearAll(): Promise<void>`**

Clear both L1 and L2 caches.

**`isEnabled(): boolean`**

Check if caching is enabled. Returns `false` when `?no-cache` URL parameter is present.

**`onInvalidate(callback: () => void): void`**

Register a callback invoked when caches are invalidated (e.g., `clearAll()`, content hash mismatch). Used by the L0 DecompressedChunkCache to clear itself when L1/L2 are invalidated.

**`setPrefetcher(prefetcher: ChunkPrefetcher | null): void`**

Attach a prefetcher to enable intelligent adjacent chunk prefetching. Pass `null` to disable.

```typescript
const prefetcher = new ChunkPrefetcher(store, { maxConcurrent: 4 });
store.setPrefetcher(prefetcher); // Enable prefetching
```

**`getPrefetcher(): ChunkPrefetcher | null`**

Get the attached prefetcher instance (if any).

**`async dispose(): Promise<void>`**

Mark the store disposed (synchronously short-circuiting `getResult`),
abort the store-level `dataAbort` controller (cancelling in-flight
prefetch / demand fetches), tear down the prefetcher, cancel any
in-flight or queued validation, await L2 dispose (which flushes pending
metadata writes), and clear L1. Call when navigating away or switching
datasets.

**`async listDatasets(): Promise<Array<{...}>>`**

List all cached datasets in OPFS:

```typescript
[
  {
    url: "https://example.com/dataset.zarr",
    hash: "8649296f56b76790",
    size: 52428800,  // bytes
    count: 1234      // number of cached chunks
  },
  ...
]
```

### ChunkPrefetcher

Intelligent prefetcher for proactive adjacent chunk loading.

#### Constructor

```typescript
new ChunkPrefetcher(store: MultiLevelCachingStore, options?: {
  maxConcurrent?: number;       // Max concurrent prefetches (default: 4)
  enabled?: boolean;            // Enable/disable (default: true)
  debug?: boolean;              // Enable debug logging (default: false)
  urlParams?: URLSearchParams;  // For testing (optional)
})
```

#### Methods

**`onAccess(key: string, priority?: 'high' | 'normal'): void`**

Called by store when a chunk is accessed from L2 or L3. Enqueues
adjacent chunks for prefetching at the given priority (default
`'normal'`). Called automatically - not for direct use.

**`enqueueWithPriority(keys: Iterable<string>, priority?: 'high' | 'normal'): void`**

Explicitly enqueue an iterable of chunk keys for prefetching at a chosen
priority (default `'high'`). Used by higher-level scheduling code to
front-load chunks ahead of demand.

**`registerArrayBounds(arrayPath: string, shape: number[], chunks: number[]): void`**

Register array shape and chunk sizes for bounds checking during prefetch. When registered, adjacent chunk generation skips indices beyond valid bounds, preventing 404s for small arrays.

```typescript
prefetcher.registerArrayBounds('gsplats_t0023/centers', [2096, 4], [1024, 4]);
```

**`getStats(): { queued, inFlight, enabled }`**

Get prefetch queue statistics:

```typescript
{
  queued: number,    // Chunks waiting to be prefetched
  inFlight: number,  // Chunks currently being prefetched
  enabled: boolean   // Whether prefetching is enabled
}
```

**`dispose(): void`**

Dispose the prefetcher, clearing all internal state (seen set, parsed cache, bounds, queue) and stopping processing.

### Modules

**DecompressedChunkCache** - L0 cache for decoded zarr chunks (eliminates Blosc decompression)

**wrapWithCache** - ES6 Proxy wrapper to add L0 caching to zarr.Array

**MultiLevelCachingStore** - Main orchestrator for L1/L2 caching, implements AsyncReadable interface

**ChunkPrefetcher** - Intelligent adjacent chunk prefetcher (enabled by default)

**LRUCache** - Generic LRU cache with O(1) operations

**SegmentedLRUCache** - 20/80 metadata/chunks split with name-based routing

**OPFSStore** - OPFS persistence layer with LRU eviction and shallow bucketing

## L2 Storage Structure

The L2 cache uses **shallow bucketing** to distribute files across 256 directories, avoiding filesystem limits with large datasets.

### Why Bucketing?

With a 2GB cache limit and typical chunk sizes:

| Avg Chunk Size | Max Files |
| -------------- | --------- |
| 32 KB          | ~65,000   |
| 64 KB          | ~32,000   |

Storing 65,000 files in a single directory can cause performance issues. Bucketing distributes them into ~250 files per bucket.

### Structure

```
zarr-cache-{url-hash}/
├── 00/                      # Bucket directories (256 total)
│   ├── cG9pbnRz...          # Base64-encoded zarr keys
│   └── ...
├── 01/
├── ...
├── ff/
└── _cache_meta.json         # Index + LRU metadata
```

### How It Works

1. **Hash the key**: `"points/positions/0.0.0"` → bucket `23`
2. **Base64 encode**: `"points/positions/0.0.0"` → `cG9pbnRzL3Bvc2l0aW9ucy8wLjAuMA`
3. **Store**: `23/cG9pbnRzL3Bvc2l0aW9ucy8wLjAuMA`

### Benefits

- **Max ~250 files per directory** instead of 65,000
- **Only 256 bucket handles** to cache (trivial memory overhead)
- **2 async calls** per file access (bucket + file) vs 1 - negligible overhead
- **Efficient clear()**: Iterates 256 directories, not 65,000 files

## Cache Behavior

### First Visit

```
User loads dataset
└─→ HTTP fetches all chunks (~100ms each)
    └─→ Stores in L2 (OPFS, persists across sessions)
    └─→ Stores in L1 (memory, session only)
    └─→ Blosc decompresses (~2ms)
        └─→ Stores in L0 (decompressed TypedArray)
        └─→ Renders
```

### Same Session (L0 Hit)

```
User navigates/zooms to previously viewed area
└─→ L0 hit (~1μs) - instant, no decompression!
    └─→ Renders
```

### Same Session (L0 Miss, L1 Hit)

```
User navigates to new area
└─→ L0 miss
    └─→ L1 hit (~1μs)
        └─→ Blosc decompresses (~2ms)
            └─→ Stores in L0
            └─→ Renders
```

### Browser Restart

```
User returns later
└─→ L0 miss (cleared on page load)
    └─→ L1 miss (cleared on page load)
        └─→ L2 hit (~1ms) - fast!
            └─→ Promotes to L1
            └─→ Blosc decompresses (~2ms)
                └─→ Stores in L0
                └─→ Renders
```

### Dataset Updated

```
Dataset content changed (new content_hash)
└─→ Cache validation detects mismatch
    └─→ Clears L0, L1, L2 automatically
    └─→ Fetches fresh data
```

## Cache Invalidation Chain

L1/L2 invalidation is the root signal for the whole cache stack. Components that
cache derived data must subscribe with `MultiLevelCachingStore.onInvalidate()`:

```typescript
const store = new MultiLevelCachingStore(datasetUrl);
const decompressedCache = new DecompressedChunkCache();
store.onInvalidate(() => decompressedCache.clear());
```

This is required because L0 stores decoded typed arrays derived from compressed
L1/L2 bytes. When `clearAll()`, `?clear-cache`, or content-hash mismatch clears
L1/L2, every L0 wrapper for that dataset must clear too; otherwise stale decoded
arrays can survive even though the compressed source cache was invalidated. New
cache layers should either register their own invalidation callback or be owned
by an object that does.

## Content Hash System

All Luxar-generated datasets include hierarchical content hashes:

**Python side** (automatic):

```python
# In LuxarZarrCompiler.finalize():
# - Computes xxhash64 for each node
# - Children hashed before parents (post-order)
# - Stored in .zattrs['content_hash']
```

**TypeScript side** (automatic):

```typescript
// On MultiLevelCachingStore.init():
// - Reads root .zattrs['content_hash']
// - Compares with cached hash
// - Clears cache if mismatch
```

**CRITICAL: Cache Validation Bypass**

The validation process MUST bypass the cache when checking for dataset changes:

```typescript
// CORRECT: Bypass cache to get true server state
private async getRemoteContentHash(): Promise<string | null> {
  // Direct HTTP fetch - NO cache lookup
  const response = await fetch(`${this.baseUrl}/.zattrs`);
  // ... extract content_hash from response
}

// WRONG: Would compare cached hash against itself (always matches!)
// const attrs = await this.get('.zattrs');  // DON'T DO THIS
```

**Why Bypass Matters**: If validation used the cache, it would read the cached `.zattrs` with the OLD hash, compare it to itself, and always validate successfully (false positive). This bug was discovered when switching datasets on the same port showed stale cached data.

**External datasets** (non-Luxar):

- If no `content_hash` attribute → validation skipped
- Cache remains functional but won't auto-invalidate

## Performance

### Expected Improvements

| Metric                   | HTTP Only        | With Cache                  |
| ------------------------ | ---------------- | --------------------------- |
| First load               | 100ms/chunk      | 100ms/chunk (same)          |
| Reload (same session)    | 100ms/chunk      | 1μs/chunk (100,000x faster) |
| Reload (new session)     | 100ms/chunk      | 1ms/chunk (100x faster)     |
| Offline viewing          | Impossible       | Fully functional            |
| Bandwidth (10MB dataset) | 10MB every visit | 10MB once                   |

### Memory Budget

**200MB L0** + **100MB L1** compressed ≈ **500MB-1.5GB** effective coverage

**Typical session** (100K points, 4D):

- L0 Cache: ~200MB (decompressed chunks)
- L1 Cache: ~100MB (compressed chunks)
- Three.js: ~50MB (geometries)
- **Total**: ~350MB (well within browser limits)

## Browser Support

**OPFS Available**:

- ✅ Chrome 86+ (October 2020)
- ✅ Firefox 111+ (March 2023)
- ✅ Safari 15.2+ (December 2021)

**Graceful Degradation**:

- Older browsers → L1-only mode (still faster than no cache)
- OPFS quota exceeded → Skip L2, use L1 + HTTP
- Network offline → Use cached data if available

## Debug API

When `?debug` URL parameter is present, cache management available at:

```typescript
await window.__luxarDebug.cache.getStats(); // Get L0 + L1 + L2 statistics
await window.__luxarDebug.cache.listDatasets(); // List all cached datasets
window.__luxarDebug.cache.clearL0(); // Clear L0 decompressed cache only
window.__luxarDebug.cache.clearL1(); // Clear L1 memory cache only
await window.__luxarDebug.cache.clearL2(); // Clear L2 OPFS cache only
await window.__luxarDebug.cache.clearAll(); // Clear L0 + L1 + L2
```

L0 cache statistics are also surfaced by the Data Monitor's Cache tab
(press `M` to cycle the data monitor; `?cache-stats` auto-opens it
expanded on the Cache tab).

Example usage:

```typescript
// Check cache usage (L1/L2)
const stats = await window.__luxarDebug.cache.getStats();
console.log('L1 Memory:', stats.l1.metadataCount, 'metadata,', stats.l1.chunksCount, 'chunks');
console.log('L2 OPFS:', stats.l2.size, 'bytes,', stats.l2.count, 'files');

// List all cached datasets
const datasets = await window.__luxarDebug.cache.listDatasets();
console.log('Cached datasets:', datasets);

// Clear all caches during development
await window.__luxarDebug.cache.clearAll();
```

## File Layout

- `multi-level-caching-store.ts` — L1+L2 facade implementing zarrita's
  `AsyncReadable` with validation, prefetcher hookup, and disposal.
- `decompressed-chunk-cache.ts` — L0 LRU of decoded TypedArrays.
- `chunk-prefetcher.ts` — Background prefetcher for adjacent chunks
  with per-array bounds registration and high/normal priority queues.
- `lru-cache.ts` — Generic LRU with O(1) get/set/delete.
- `types.ts` — Shared cache types (`MultiLevelCacheStats`,
  `CacheStatusBadge`, validation-mode helpers, etc.).
- `decompressed-chunk-cache/cached-zarr-array.ts` — ES6 Proxy that
  wraps a `zarr.Array` with the L0 cache (`wrapWithCache`,
  `isCachedArray`, `unwrapCachedArray`).
- `multi-level-caching-store/` — internals split into
  `segmented-lru-cache.ts` (L1), `opfs-store.ts` + `opfs-store/`
  (L2 persistence, bucketing, metadata), `fetch-retry.ts` (HTTP
  retry/abort helpers, URL hashing), `bandwidth-window.ts` (sliding
  window), and `validation-queue.ts` (per-dataset validation
  serialization).

## Related Packages

- **Scene Loader** (`../data/scene-loader.ts`) - Uses MultiLevelCachingStore transparently
- **Configuration** (`../config/index.ts`) - Cache configuration options
