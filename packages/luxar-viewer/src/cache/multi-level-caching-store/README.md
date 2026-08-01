# Multi-Level Caching Store Internals

Helper modules backing `../multi-level-caching-store.ts` — the L1 (in-memory)
and L2 (OPFS-persistent) tiers of the viewer cache. The parent module is
the public `AsyncReadable` facade; everything here is an internal
implementation split that keeps each concern independently testable.

## Overview

These files implement the four mechanics the L1+L2 facade needs to behave
correctly under realistic browser load:

- **Routing compressed bytes** between a small metadata segment and a
  large chunks segment so frequent `.zarray`/`.zattrs` reads do not get
  evicted by chunk pressure (`segmented-lru-cache.ts`).
- **Persisting chunks to OPFS** with LRU eviction, shallow bucketing, and
  robust recovery from corrupt metadata or hung handles
  (`opfs-store.ts` + the `opfs-store/` subpackage).
- **Fetching from the network** with retry, jittered backoff, abort-merge,
  and stable URL hashing for the per-dataset cache root
  (`fetch-retry.ts`).
- **Measuring bandwidth** with an amortized sliding window
  (`bandwidth-window.ts`).
- **Serializing cache-validation runs** across two stores that point at
  the same dataset URL, and probing the remote `content_hash` with a
  cache-bypassing fetch (`validation-queue.ts`).

Nothing in this folder is exported from the cache package — every helper
is consumed by the parent `MultiLevelCachingStore` (and, transitively, by
the OPFS store's own helpers).

## File Structure

```
multi-level-caching-store/
├── segmented-lru-cache.ts    # L1 routing: metadata segment + chunks segment
├── opfs-store.ts             # L2 OPFS persistence with bucketing and LRU
├── opfs-store/               # Subpackage: OPFS layout, metadata, timeout
├── fetch-retry.ts            # HTTP retry/abort/backoff + URL build + URL hash
├── bandwidth-window.ts       # Sliding-window bytes/sec tracker
└── validation-queue.ts       # Per-dataset validation serializer + remote-hash probe
```

## Components

### `segmented-lru-cache.ts` — L1 router

`SegmentedLRUCache` wraps two `LRUCache<Uint8Array>` instances and routes
incoming keys by filename pattern. Zarr metadata keys (`.zmetadata`,
`.zarray`, `.zattrs`, `zarr.json`) go to a dedicated **metadata
segment** sized at 20% of the total budget with a 10MB floor; everything
else goes to the **chunks segment** sized at the remaining 80%. This
protects small, high-traffic metadata files from being evicted when a
single navigation pulls in many large chunks.

`get(key)` consults exactly one segment (the one the name pattern
selects) so chunk lookups do not record a spurious miss on the metadata
segment. `getStats()` returns the aggregated `CacheStats` shape consumed
by the cache monitor (`metadataSize`, `chunksSize`, `metadataCount`,
`chunksCount`, `hits`, `misses`, `evictions`).

### `opfs-store.ts` — L2 OPFS persistence

`OPFSStore` is the L2 cache: an LRU-evicting persistent store layered
over OPFS, scoped per dataset via a `zarr-cache-<16hex>` root directory.
Files are distributed across 256 hex-named bucket directories (`00`..`ff`)
to avoid filesystem-level fanout limits — see the
[`opfs-store/`](./opfs-store/README.md) subpackage for the
bucket-hashing, filename-encoding, and metadata-lifecycle helpers.

Key behaviours:

- **Generation token** — every `clear()` bumps a counter so a `set()` in
  flight when `clear()` lands detects the mismatch on completion and
  best-effort deletes the just-written file instead of indexing it.
- **Pending-writes drain** — `clear()` awaits in-flight `pendingWrites`
  via `Promise.allSettled` before resetting state.
- **Disposed flag** — post-`dispose()` `set/get/touch` are no-ops; the
  in-flight metadata save is awaited before the final flush.
- **Per-call timeout** — every OPFS call goes through `withTimeout` so a
  hung browser handle cannot stall the cache indefinitely.
- **Missing-file delete reconciliation** — `delete()` treats `NotFoundError` as
  logical success, removes the stale index/size entry, and schedules the repaired
  metadata snapshot for persistence. A phantom LRU head therefore cannot wedge
  max-size or quota eviction or reappear next session. Other I/O failures preserve
  state for a safe retry.
- **Health counters** — `oversizedWriteSkipped`, `quotaWriteSkipped`,
  `evictions`, `writeFailures`, `corruptedEntries`,
  `metadataParseFailures`, `orphanedFilesRemoved` are surfaced via
  `getStats()` and the cache monitor's "Errors" card.
- **External-dataset validation state** — the persisted
  `validationMode` (`content-hash` | `zattrs-hash` | `ttl` | `none`) and
  `lastValidatedAt` ride along in `_cache_meta.json` so a TTL window
  survives page reloads.

### `fetch-retry.ts` — network primitives

- **`mergeAbortSignals(primary, caller?)`** — produces a scoped signal that
  fires when either source aborts. Uses native `AbortSignal.any` when
  available (Node 22+, modern browsers) and falls back to a relay
  controller otherwise. The fallback scope removes both source listeners
  immediately on abort or when its idempotent `dispose()` is called, so a
  long-lived dataset signal does not retain one closure per completed fetch.
- **`buildUrl(baseUrl, key)`** — joins a base URL and a zarr key while
  stripping trailing slashes on the base and leading slashes on the key
  (defends against the triple-slash bug when a base URL ends in `/`
  and a zarr key begins with `/`).
- **`hashUrl(url)`** — SHA-256-truncated dataset identifier of the form
  `zarr-cache-<16 hex>`. Used as the OPFS root directory name and the
  validation-queue key.
- **`fetchWithRetry(url, options?)`** — retry-budget fetch keyed off
  `config.dataLoading.network.retryAttempts` and `.timeoutMs`. 4xx
  responses return immediately (no retry can fix a missing key). 429,
  5xx, network errors, and per-attempt timeouts are retried with
  exponential backoff bounded by ±25% jitter and a 500ms ceiling. The
  total timeout is split across attempts so retries do not multiply
  worst-case load time. A caller-aborted signal exits immediately
  without consuming retry budget. Accepts `timeoutMsOverride` for
  validation probes that want a shorter budget than data fetches. A
  terminal response is returned with an idempotent `dispose()` callback;
  consumers hold that scope through body consumption and release it in
  `finally`, preserving body cancellation without retaining fallback
  listeners for the dataset lifetime. `dispose()` also cancels a body
  the consumer never read (retryable 429/5xx, terminal non-OK, or a
  post-fetch abort), so an ignored response stops streaming instead of
  holding bandwidth and a connection slot until GC.

### `bandwidth-window.ts` — sliding-window throughput

`BandwidthWindow` tracks `(timestamp, bytes)` samples and returns
average bytes/sec over the trailing `windowMs`. The implementation
avoids `Array.shift()` (O(n) per pop, O(n²) under sustained high fetch
rates) by advancing a `start` index past expired entries and only
slicing off the dead prefix when it exceeds half the buffer — keeping
both `record()` and `rate()` amortized O(1). Feeds the
`network.bandwidth` field of `MultiLevelCacheStats`.

### `validation-queue.ts` — cross-instance validation serializer

`ValidationQueue` is a per-`datasetId` (= `hashUrl(baseUrl)`) FIFO that
serializes cache-validation runs across MultiLevelCachingStore
instances pointed at the same URL. Without it, a slower older
validation could overwrite a newer content-hash. Each queue entry
carries an `AbortController` so the owning instance's `dispose()` can
both cancel the in-flight HTTP fetch and remove the queue entry — the
task is skipped entirely if abort fires while still waiting in line,
preventing a closure that captured a now-disposed `this` from running
`setContentHash()` against a disposed L2 store. The queue only
serializes; it does not emit invalidation events (that stays at the
caller).

`getRemoteContentHash(baseUrl, options)` fetches the dataset root
`.zattrs` via `fetchWithRetry` and extracts `content_hash`. It always
goes to the network — never the cache — so validation can detect
server-side dataset changes. Returns `null` for missing/malformed
responses or for external (non-Luxar) datasets without a
`content_hash` attribute. Accepts `timeoutMsOverride` so the cache
validator can use the dedicated `validationTimeoutMs` budget rather
than the full data-fetch timeout.

## Subpackages

- [`opfs-store/`](./opfs-store/README.md) — bucket hashing, filename
  encoding, `_cache_meta.json` lifecycle, and the `withTimeout`
  wrapper. Internals of `opfs-store.ts`.

## Invariants

- **`fetchWithRetry` never throws.** Network errors and exhausted retry
  budgets log a warning and return `undefined`; callers treat that as
  "miss" and fall back to the next tier (or surface a load error).
- **Metadata segment never starves.** `SegmentedLRUCache` enforces a
  10MB floor on the metadata segment regardless of `totalSize` — small
  L1 budgets only shrink the chunks segment.
- **URL hash is part of the on-disk format.** `hashUrl` is a SHA-256
  prefix; changing the prefix length or hash function orphans every
  existing OPFS dataset directory.
- **Validation queue is static.** The `Map` lives at module scope so
  two `MultiLevelCachingStore` instances constructed independently for
  the same URL still serialize against each other.

## See Also

- [`../README.md`](../README.md) — full cache package overview (L0/L1/L2
  hierarchy, content-hash validation, prefetching, health badges).
- [`../multi-level-caching-store.ts`](../multi-level-caching-store.ts) —
  the L1+L2 facade that wires every helper here into the
  `AsyncReadable` interface consumed by zarrita.
- [`../lru-cache.ts`](../lru-cache.ts) — generic LRU primitive used by
  `SegmentedLRUCache`.
- [`../types.ts`](../types.ts) — `CacheStats`, `MultiLevelCacheStats`,
  `OPFSMetadata`, `OPFS_ENCODING_VERSION`, `CacheValidationMode`.
- [`opfs-store/README.md`](./opfs-store/README.md) — OPFS layout and
  metadata helpers.
