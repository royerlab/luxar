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
├── buckets.ts           # Key hashing, base64url filename encoding (+ inverse), bucket-handle cache
├── metadata.ts          # _cache_meta.json load/save lifecycle (bounded debounce, flush)
├── orphan-reconcile.ts  # Budgeted crawl for chunk files the index does not list
├── opfs-root.ts         # The viewer's `luxar/` OPFS namespace directory
└── opfs-timeout.ts      # Promise.race-based timeout wrapper for OPFS I/O
```

## Components

### `buckets.ts` — Directory layout

- **`getBucket(key)`** — djb2-like rolling hash masked to 8 bits, returns
  a two-char hex bucket name. Stable across sessions: changing this
  function would orphan every existing OPFS entry.
- **`hashTag(hash)`** — 16 hex chars identifying the content hash a chunk
  file was written under (`''` for none).
- **`keyToFileName(key, tag)`** — `{tag}.{base64url(UTF-8 key)}` (bare
  base64url for an empty tag). Zarr keys may include non-ASCII group/array
  names; the UTF-8 encode step keeps the output filesystem-safe on every
  browser. The tag makes a file of one hash unreadable as another hash's chunk
  (a second tab at an older hash writing into a cleared directory). Bumping
  `OPFS_ENCODING_VERSION` (in `../../types`) is what invalidates filenames
  produced by a previous encoding scheme — `metadata.ts` treats a version
  mismatch as a cold cache.
- **`fileNameToKey(name)`** — the inverse of `keyToFileName` (`{ key, tag }`),
  or `null` for a name that encoding cannot produce (accepted only if it
  round-trips). Lets the orphan reconcile re-index a file whose index entry
  was never saved, under the recovery hash's tag only.
- **`OPFSBucketCache`** — caches up to 256 `FileSystemDirectoryHandle`s
  so reads/writes don't re-walk the root every call.
  - `getHandle(root, bucket, create)` — memoised lookup; returns `null`
    rather than throwing when `create=false` and the bucket is missing.
  - `invalidate(bucket)` — drops a single stale handle after OPFS
    raises "could not be found" (recovery path for handles dangling
    after a concurrent `clear()`).
  - `clear()` — drops every cached handle (called after the directory
    tree is wiped).
  - `navigateToFile(root, key, tag, create)` — convenience: hash the key,
    resolve the bucket handle, return the file handle in one call.

### `metadata.ts` — `_cache_meta.json` lifecycle

`OPFSMetadataManager` owns the JSON index file at the root of the
cache directory. The persisted `OPFSMetadata` shape (in `../../types`)
records `{baseUrl, entries, totalSize, orderCounter, contentHash,
encodingVersion, validationMode, lastValidatedAt}`.

- **`load(root)`** — read + parse. Returns:
  - `null` on cold start (file missing — any `NotFoundError`, whatever its
    message says).
  - A fresh empty `LoadOutcome` with `needsOrphanCleanup: false` on
    encoding-version mismatch (the on-disk filenames no longer match
    what `keyToFileName` would produce — start over).
  - A fresh empty `LoadOutcome` with `needsOrphanCleanup: true` on
    `JSON.parse` failure. `parseFailures` is incremented; `OPFSStore`
    then treats every file on disk as an orphan to delete (its provenance
    is unknown).
  - The parsed index otherwise, with defensive sanitisation:
    `totalSize`/`orderCounter` are clamped to non-negative finite
    values, and `totalSize` is recomputed from the live entries when
    the persisted value disagrees with the sum by more than 1 byte.
    Entries are sorted by ascending `order` so `Map` insertion order
    equals LRU order.
- **`scheduleSave({ root, getSnapshot, delayMs, maxWaitMs?, leadingDelayMs?, onError })`** —
  debounced save, BOUNDED by `maxWaitMs`: each call re-arms the timer
  (last-writer-wins) but never past `maxWaitMs` after the first unsaved
  call, so a continuous write stream still saves (a pure trailing debounce
  never fired during playback). With `leadingDelayMs`, the first call after a
  quiet period (no write started for `delayMs`) is written within that delay,
  and later calls cannot push it back. On fire, calls `getSnapshot()` to capture
  the latest state and writes it. Writes never overlap: a save due while
  one is in flight is coalesced into ONE follow-up write. The in-flight
  promise is tracked so `dispose()` can await it.
- **`flushPending()`** — start a scheduled save now (the `pagehide` /
  hidden path); a no-op when nothing is scheduled.
- **`hasPendingSave()` / `cancelPendingSave()` / `awaitInFlight()` /
  `save(...)`** — the four primitives `OPFSStore.dispose()` uses to
  flush cleanly: cancel the timer, optionally write a final synchronous
  snapshot, then await whatever the timer had already started.
- **`loadIdentity(root)` / `writeIdentity(root, hash)`** — the dataset
  identity file `_cache_identity.json` (`{contentHash, encodingVersion}`): the
  hash the directory's chunk files were written under. Written by
  `OPFSStore.setContentHash` only when the hash changes, and every chunk write
  waits for it, so it survives a reload that the debounced index save does not.
  `loadIdentity` returns `null` for a missing, empty, unparsable, hashless or
  other-encoding file. `OPFSStore` recovers unindexed files only under this hash
  (or, with no identity file, under a parsed index's hash), and uses it as the
  cached hash when no index landed.
- **`cleanupOrphans(root, expectedFileNames)`** — iterate every
  hex-named bucket directory and delete files not in the expected set.
  Increments `orphansRemoved`. Unbounded; `OPFSStore` itself now uses the
  budgeted `orphan-reconcile.ts` crawl instead.

### `orphan-reconcile.ts` — unindexed files

- **`crawlOrphans(root, { expectedFileNames, maxOrphans, maxExamined,
shouldStop, onOrphan })`** — walk the hex buckets, listing each bucket's
  names before acting on them, and hand every name missing from the index
  snapshot to `onOrphan` (with the decoded key and hash tag, or `null`), until
  a budget is spent or `shouldStop()`. `OPFSStore.reconcileOrphans` decides per
  file: re-index (a recovery hash is set, the file carries its tag, key hashes
  to that bucket, non-empty, fits under `maxSize` — re-checked at the merge) or
  delete, re-checking its live index / pending writes before each action and
  registering deletes in the same-key delete barrier (#1073); a recorded key
  that is written meanwhile drops its record. The crawl runs in the background,
  so until it ENDS `OPFSStore.get()` also reads an unindexed key's file
  directly (the path is a function of the key and tag) and indexes a hit: a
  reload's first reads do not wait for the crawl.

Counters `parseFailures` and `orphansRemoved` are surfaced through
`OPFSStore.getStats()` (mapped to `metadataParseFailures` and
`orphanedFilesRemoved` in the public cache stats), next to the store's own
`orphansReindexed`.

### `opfs-timeout.ts` — I/O timeout

- **`withTimeout(promise, ms, label)`** — `Promise.race` against a
  `setTimeout` that rejects with `Error('OPFS timeout: <label>
exceeded <ms>ms')`. The timer is always cleared in `finally`.
  Wraps individual OPFS calls so a hung browser handle cannot stall
  the cache indefinitely — every call in `OPFSStore` that touches the
  filesystem goes through this helper.

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
