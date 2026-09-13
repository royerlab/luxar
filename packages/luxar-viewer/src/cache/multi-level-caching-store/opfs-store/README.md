# OPFS Store Internals

Helper modules backing `../opfs-store.ts` — the L2 persistent layer of the
cache. Splits the bucket directory layout, the `_cache_meta.json`
lifecycle, and the per-call I/O timeout into independently-testable
pieces.

## Overview

The L2 cache stores hundreds to tens of thousands of small files in
OPFS. Putting them in a single directory is slow on every browser, so
keys are hashed into 256 hex-named buckets (`00`..`ff`) and the index
of `{key -> {size, order}}` lives in a JSON file at the root. These
helpers encapsulate that layout and the recovery paths around it
(corrupt metadata, stale handles, hung I/O calls). `OPFSStore` itself
remains the only public surface — nothing here is exported from the
package.

## File Structure

```
opfs-store/
├── buckets.ts        # Key hashing, base64url filename encoding, bucket-handle cache
├── metadata.ts       # _cache_meta.json load/save/orphan-cleanup lifecycle
└── opfs-timeout.ts   # Promise.race-based timeout wrapper for OPFS I/O
```

## Components

### `buckets.ts` — Directory layout

- **`getBucket(key)`** — djb2-like rolling hash masked to 8 bits, returns
  a two-char hex bucket name. Stable across sessions: changing this
  function would orphan every existing OPFS entry.
- **`keyToFileName(key)`** — UTF-8 → base64url. Zarr keys may include
  non-ASCII group/array names; the UTF-8 encode step keeps the output
  filesystem-safe on every browser. Bumping `OPFS_ENCODING_VERSION` (in
  `../../types`) is what invalidates filenames produced by a previous
  encoding scheme — `metadata.ts` treats a version mismatch as a cold
  cache.
- **`OPFSBucketCache`** — caches up to 256 `FileSystemDirectoryHandle`s
  so reads/writes don't re-walk the root every call.
  - `getHandle(root, bucket, create)` — memoised lookup; returns `null`
    rather than throwing when `create=false` and the bucket is missing.
  - `invalidate(bucket)` — drops a single stale handle after OPFS
    raises "could not be found" (recovery path for handles dangling
    after a concurrent `clear()`).
  - `clear()` — drops every cached handle (called after the directory
    tree is wiped).
  - `navigateToFile(root, key, create)` — convenience: hash the key,
    resolve the bucket handle, return the file handle in one call.

### `metadata.ts` — `_cache_meta.json` lifecycle

`OPFSMetadataManager` owns the JSON index file at the root of the
cache directory. The persisted `OPFSMetadata` shape (in `../../types`)
records `{baseUrl, entries, totalSize, orderCounter, contentHash,
encodingVersion, validationMode, lastValidatedAt}`.

- **`load(root)`** — read + parse. Returns:
  - `null` on cold start (file missing).
  - A fresh empty `LoadOutcome` with `needsOrphanCleanup: false` on
    encoding-version mismatch (the on-disk filenames no longer match
    what `keyToFileName` would produce — start over).
  - A fresh empty `LoadOutcome` with `needsOrphanCleanup: true` on
    `JSON.parse` failure. `parseFailures` is incremented and the
    caller (`OPFSStore.init`) is expected to run `cleanupOrphans`.
  - The parsed index otherwise, with defensive sanitisation:
    `totalSize`/`orderCounter` are clamped to non-negative finite
    values, and `totalSize` is recomputed from the live entries when
    the persisted value disagrees with the sum by more than 1 byte.
    Entries are sorted by ascending `order` so `Map` insertion order
    equals LRU order.
- **`scheduleSave(...)`** — debounced save. Replaces any previously-
  scheduled timer (last-writer-wins); on fire, calls `getSnapshot()` to
  capture the latest state and writes it. The in-flight promise is
  tracked so `dispose()` can await it.
- **`hasPendingSave()` / `cancelPendingSave()` / `awaitInFlight()` /
  `save(...)`** — the four primitives `OPFSStore.dispose()` uses to
  flush cleanly: cancel the timer, optionally write a final synchronous
  snapshot, then await whatever the timer had already started.
- **`cleanupOrphans(root, expectedFileNames)`** — iterate every
  hex-named bucket directory and delete files not in the expected set.
  Increments `orphansRemoved`. Only runs after a metadata parse
  failure; bounded by quota otherwise. Skips the `_cache_meta.json`
  file itself and any non-bucket directories.

Counters `parseFailures` and `orphansRemoved` are surfaced through
`OPFSStore.getStats()` (mapped to `metadataParseFailures` and
`orphanedFilesRemoved` in the public cache stats).

### `opfs-timeout.ts` — I/O timeout

- **`withTimeout(promise, ms, label)`** — `Promise.race` against a
  `setTimeout` that rejects with `Error('OPFS timeout: <label>
exceeded <ms>ms')`. The timer is always cleared in `finally`.
  Wraps individual OPFS calls so a hung browser handle cannot stall
  the cache indefinitely — every call in `OPFSStore` that touches the
  filesystem goes through this helper.

### `opfs-read-gate.ts` — bounded read fan-out

- **`withOpfsReadGate(run)`** — page-wide FIFO gate for chunk reads. Deep
  progressive passes can fan out several hundred L2 hits at once; Chromium's
  main-thread OPFS path stalls under that pressure even when smaller batches
  read the same files quickly. The 64-slot cap preserves local parallelism
  without stampeding the browser backend. Queue wait is outside each
  operation's timeout, so healthy backpressure is never misclassified as hung
  I/O.

## Invariants

- **Bucket hash is part of the on-disk format.** Changing `getBucket`
  re-buckets every key; the in-memory index would no longer find any
  existing file. Treat as a format-version bump and increment
  `OPFS_ENCODING_VERSION`.
- **`keyToFileName` is part of the on-disk format.** Same caveat —
  bumping `OPFS_ENCODING_VERSION` is the documented escape hatch and is
  honoured by `metadata.ts::load`.
- **The bucket-handle cache is best-effort.** Any OPFS call that fails
  with "could not be found" must call `invalidate(bucket)` before
  retrying, because the cached handle may now point at a deleted
  directory after a concurrent `clear()`.
- **`load` never throws.** Cold-start, version mismatch, and corrupt
  JSON all return a usable `LoadOutcome` (or `null` only for the
  cold-start case); the cache always boots.
- **`save` never throws.** Failures are logged and swallowed so
  `dispose()` is safe to call from `beforeunload`.

## See Also

- [`../README.md`](../README.md) — segmented LRU, OPFS store, fetch
  retry, validation queue.
- [`../opfs-store.ts`](../opfs-store.ts) — the consumer that wires
  these helpers together into the L2 `OPFSStore`.
- [`../../README.md`](../../README.md) — full cache package overview
  (L0/L1/L2 hierarchy, content-hash validation, health badges).
- [`../../types.ts`](../../types.ts) — `OPFSMetadata`,
  `OPFS_ENCODING_VERSION`, `CacheValidationMode`.
