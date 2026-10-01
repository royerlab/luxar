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
├── opfs-read-gate.ts         # Page-wide FIFO read cap + live backpressure stats
├── opfs-store/               # Subpackage: OPFS layout, metadata, timeout
├── fetch-retry.ts            # HTTP retry/abort/backoff + URL build + URL hash
├── bandwidth-window.ts       # Sliding-window bytes/sec tracker
└── validation-queue.ts       # Per-dataset validation serializer + remote-hash probe
```

## Components

### `opfs-read-gate.ts` — bounded read fan-out

- **`withOpfsReadGate(run, signal?)`** — page-wide FIFO gate for chunk reads,
  built on the shared `AsyncGate` (`../../utils/async-gate.ts`). Deep
  progressive passes can fan out several hundred L2 hits at once, and reads
  were the last unbounded browser-filesystem path after writes gained their own
  cap. The reported multi-second L2 stall remains unattributed. Queue wait is
  outside each operation's timeout, so healthy backpressure is never
  misclassified as hung I/O. `run` receives `hold(io)`: a lease lasts until
  every held file-I/O promise settles too, so a read whose timeout gave up
  still occupies its slot while the browser finishes it. The optional
  `signal` lets a queued read leave the queue on abort (rejecting with the
  signal's reason).
- **`getOpfsReadGateStats()`** — exposes active and queued reads to the cache
  monitor so a live stall distinguishes backend work from gate backpressure.
- **`resetOpfsReadGate()`** — clears module-global gate state for test isolation;
  releases from the prior epoch cannot corrupt the new counters.

### `segmented-lru-cache.ts` — L1 router

`SegmentedLRUCache` wraps two `LRUCache<Uint8Array>` instances and routes
incoming keys by filename pattern. Zarr metadata keys (`.zmetadata`,
`.zarray`, `.zattrs`, `zarr.json`) go to a dedicated **metadata
segment** sized at 20% of the total budget with a 10MB floor (capped at half
the total on a small budget); everything
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
  `getStats()` and the cache monitor's "Errors" card (plus
  `orphansReindexed`).
- **Index persistence** — the index save is a 1 s debounce with a 2 s
  ceiling and a 150 ms leading edge (the first write after a quiet second is
  indexed within 150 ms), flushed on `pagehide` / hidden (`flushAllMetadata`)
  and on `dispose()`. Neither unload-time write survives a navigation: measured
  on Chromium, a reload stops the save after `getFileHandle(create)`, so entries
  written since the last completed save are lost (the leading edge keeps a
  load's first burst out of that window). The zero-byte file
  this leaves is read as a cold start, not as corruption. An open-time,
  per-session-budgeted orphan reconcile re-indexes
  or deletes chunk files the index never recorded (see `../README.md`,
  "L2 index persistence").
- **Quota estimate cache** — `navigator.storage.estimate()` is re-run at most
  every 30 s / 64 MB written, or when the debited cached headroom cannot cover
  a write.
- **No write-path copy** — `set()` writes the caller's `Uint8Array` view
  directly (only a detached buffer, detected by a length check, is refused).
- **External-dataset validation state** — the persisted
  `validationMode` (`content-hash` | `zattrs-hash` | `archive-etag` | `ttl` |
  `none`) and `lastValidatedAt` ride along in `_cache_meta.json` so a TTL
  window survives page reloads.

### `fetch-retry.ts` — network primitives

- Each attempt's timeout signal is merged with the caller's by the shared
  `combineAbortSignals` (`../../utils/abort-signals.ts`, also used by the
  store for its dispose signal): native `AbortSignal.any` when available,
  else a relay whose idempotent `dispose()` removes both source listeners,
  so a long-lived dataset signal does not retain one closure per completed
  fetch.
- **`buildUrl(baseUrl, key)`** — joins a base URL and a zarr key while
  stripping trailing slashes on the base and leading slashes on the key
  (defends against the triple-slash bug when a base URL ends in `/`
  and a zarr key begins with `/`).
- **`hashUrl(url)`** — SHA-256-truncated dataset identifier of the form
  `zarr-cache-<16 hex>`. Used as the OPFS root directory name and the
  validation-queue key.
- **`fetchWithRetry(url, options, consume)`** — retry-budget fetch keyed off
  `config.dataLoading.network.retryAttempts` and `.timeoutMs`. 4xx
  responses return immediately (no retry can fix a missing key). 429,
  5xx, network errors, and per-attempt timeouts are retried with
  exponential backoff bounded by ±25% jitter and a 500ms ceiling. The
  configured timeout is split across attempts and applied separately to
  time-to-headers and aggregate body-progress stalls. A quiet multiplexed
  stream stays alive while another live lease receives bytes. Its absolute
  deadline starts when body reading begins and is derived from `Content-Length`
  at a 16 KiB/s aggregate floor shared across at most eight active leases, or
  eight stall windows shared across at most four leases when the length is
  unavailable. Metadata probes use a separate lane from data bodies: the caps
  are 24 data + 4 metadata, with the data lane widened to 96 for an origin
  whose resource timing shows it negotiated h2/h3, and an origin seen over a
  plain `http:` URL held to 4 data + 2 metadata of its own (HTTP/1.1's six
  sockets) while other origins keep their width. `priority` (`demand` > `refinement` > `speculative`, or a
  `FetchPriorityCell` a coalescing caller may raise) orders the gate's queue;
  speculative requests never hold more than a quarter of a lane. The store
  passes `speculative` for prefetcher reads and raises a pending read to
  `demand` when a demand caller joins it. A read whose abort signal carries a
  class (`tagSignalPriority` — the refinement loop tags its run's signal
  `refinement`) is fetched at that class unless the call names one; the L0
  chunk proxy copies the class onto its shared decode's signal and lifts it
  when a more urgent caller joins. `cache` forwards a `RequestCache` mode
  (the zip range reader sends `no-store`). A caller-aborted signal exits
  immediately without consuming retry budget. The consumer runs inside its
  fetch-gate lease and may call `readBody()` once; returning without reading
  cancels the body before release.

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
validation could overwrite a newer content-hash. `serialize()` hands
each caller its own `QueueEntry` (via an `onStart` callback fired
synchronously, before the first `await`); each entry carries an
`AbortController`. Cancellation is identity-scoped: the owning
instance's `dispose()` aborts ITS OWN entry — it does NOT remove the
entry from the map. This is deliberate — an older store must never
cancel a newer same-URL store's validation, and map cleanup is left to
`serialize`'s `finally` head-guard, which evicts an entry only when it
is still the current head (so an aborted waiting entry keeps its
successors FIFO-chained until its predecessor settles). The task is
skipped entirely if abort fires while still waiting in line, preventing
a closure that captured a now-disposed `this` from running
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

- **`fetchWithRetry` does not throw transport failures.** Network errors and
  exhausted retry budgets log a warning and return `undefined`; a consumer's
  own error propagates unchanged so status and archive validation failures
  keep their domain-specific type.
- **Metadata segment never starves.** `SegmentedLRUCache` gives the
  metadata segment a 10MB floor, capped at half of `totalSize`: a heap- or
  `?cacheBudgetMB`-scaled L1 below 20MB splits evenly rather than leaving
  the chunks segment at zero (every L1 chunk rejected) and the two segments
  over budget.
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
