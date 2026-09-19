import type { CacheValidationMode } from '../types';
import { log, Modules } from '../../utils/log';
import { getErrorMessage } from '../../utils/format-error';
import { config } from '../../config';
import { OPFSBucketCache, getBucket, keyToFileName } from './opfs-store/buckets';
import { getLuxarOpfsRoot } from './opfs-store/opfs-root';
import { OPFSMetadataManager, type MetadataSnapshot } from './opfs-store/metadata';
import { withTimeout } from './opfs-store/opfs-timeout';
import { getOpfsReadGateStats, withOpfsReadGate } from './opfs-read-gate';

type IterableFileSystemDirectoryHandle = FileSystemDirectoryHandle & {
  keys(): AsyncIterableIterator<string>;
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
};

interface PersistedMetadataFile {
  baseUrl?: string;
  contentHash?: string;
  totalSize?: number;
  entries?: unknown[];
}

function isNotFoundError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false;
  return (error as { name?: unknown }).name === 'NotFoundError';
}

export interface CachedDatasetSummary {
  url: string;
  hash: string;
  size: number;
  count: number;
}

/**
 * OPFS persistence layer (L2 cache) with LRU eviction and shallow bucketing.
 *
 * Uses 256 bucket directories to distribute files and avoid filesystem limits.
 * Files are stored as: `{bucket}/{base64-encoded-key}`
 *
 * Structure (everything under the viewer's `luxar/` OPFS namespace dir, see
 * `opfs-store/opfs-root.ts`):
 * ```
 * luxar/zarr-cache-{hash}/
 * ├── 00/           (~250 files per bucket)
 * │   ├── UG9pbnRz...
 * │   └── ...
 * ├── 01/
 * ├── ...
 * ├── ff/
 * └── _cache_meta.json
 * ```
 *
 * Benefits:
 * - Max ~250-500 files per directory instead of 65,000+
 * - Only 256 bucket handles to cache (trivial memory)
 * - clear() iterates 256 directories, not 65,000 files
 * - Preserves fast flat access (2 async calls vs 1)
 */
export class OPFSStore {
  private opfsRoot: FileSystemDirectoryHandle | null = null;
  private index = new Map<string, { size: number; order: number }>();
  private orderCounter = 0;
  private totalSize = 0;
  private maxSize: number;
  private datasetId: string;
  private baseUrl: string;
  private contentHash: string | null = null;
  // External-dataset validation mode + last-validated timestamp.
  // Persisted to OPFSMetadata so the next session can apply a TTL
  // window across page loads.
  private validationMode: CacheValidationMode = 'none';
  private lastValidatedAt: number | null = null;

  // Read/write tracking for monitoring. `readCount` is the L2 hit
  // counter — only incremented when get() returns a value. `missCount`
  // counts get() calls that returned undefined (file not present /
  // size mismatch / I/O error). Canceled reads are tracked separately
  // because they do not fall through to a network download.
  private readCount = 0;
  private writeCount = 0;
  private missCount = 0;
  private canceledReadCount = 0;

  // Health counters surfaced via getStats().
  // `oversizedWriteSkipped`: doSet() rejected an entry larger than
  //   maxSize so it could not have been written without violating the
  //   cache size invariant.
  // `quotaWriteSkipped`: navigator.storage.estimate() reported
  //   insufficient quota even after own-LRU eviction.
  // `evictions`: number of own-LRU entries evicted to make room for
  //   incoming writes.
  // `writeFailures`: doSet() catch branch — file I/O threw after
  //   retries.
  // `corruptedEntries`: get() detected a size mismatch between index
  //   and on-disk data and removed the bad file.
  // `clobberBarrierWriteSkipped`: set() dropped a write because a prior
  //   same-key delete's non-cancellable removeEntry did not settle within
  //   the deadline — writing would have risked the eventual removeEntry
  //   clobbering the replacement (#1073).
  // Counters: `parseFailures` (loadMetadata could not JSON-parse) and
  // `orphansRemoved` (cleanupOrphans reclaim count) live on the
  // metadata manager and are read by getStats().
  private oversizedWriteSkipped = 0;
  private quotaWriteSkipped = 0;
  private evictions = 0;
  private writeFailures = 0;
  private corruptedEntries = 0;
  private clobberBarrierWriteSkipped = 0;

  // Bucket handle cache (256 possible buckets: 00-ff)
  private buckets = new OPFSBucketCache();

  // Owns _cache_meta.json: load/save + debounced-save timer + orphan reclaim.
  private metadata = new OPFSMetadataManager();
  private static readonly METADATA_SAVE_DELAY = 1000; // 1 second debounce

  // Serialize concurrent writes to the same key to prevent race conditions
  private pendingWrites = new Map<string, Promise<void>>();

  // In-flight REAL (un-timed-out) deletes, keyed by datasetId → key (#1073).
  // `delete()`'s caller await is bounded by opfsOperationTimeoutMs (preserving
  // #991), but removeEntry is NOT cancellable — when the timeout wins, the
  // caller returns while removeEntry keeps running. A same-key replacement
  // write must NOT start (and therefore cannot complete and then be clobbered)
  // while that straggling removeEntry is still in flight, so a set() waits for
  // every outstanding real delete of its key to actually settle before writing.
  // The link stored here is the ACTUAL op settlement (not the caller's early
  // timeout return). Deletes deliberately do NOT wait on each other: they are
  // idempotent (a second removeEntry of an absent file is NotFound), so a fresh
  // delete can retry even if an earlier one hangs — that earlier op's caller
  // already gave up. If a delete genuinely never settles the set() barrier
  // times out and the write is DROPPED (degrade-safe: the key stays a miss)
  // rather than risk a clobber or stall dispose's pending-write drain.
  //
  // STATIC (class-level) so the barrier survives instance teardown: dispose()'s
  // delete-drain is bounded, so a removeEntry that outlives that deadline can
  // still be in flight when a successor same-URL store — which shares the
  // datasetId (SHA-256 of the base URL) and therefore the OPFS directory —
  // takes over. Keying the registry by datasetId lets the successor's set()
  // barrier see the predecessor's straggling deletes and gate on them exactly
  // like its own; an instance-level map would let the old removeEntry clobber
  // the successor's replacement. Entries self-clean on settlement, so the
  // registry only ever holds genuinely-outstanding operations.
  private static pendingDeletesByDataset = new Map<string, Map<string, Set<Promise<void>>>>();

  // Generation token: every clear() bumps this. doSet() captures the
  // generation when it begins and discards its index/metadata mutation
  // if the generation has advanced — preventing a slow write that
  // started before clear() from repopulating the post-clear index.
  // Plain bookkeeping for stale-write detection; not a public API.
  private generation = 0;

  // Lifecycle: set SYNCHRONOUSLY at dispose() entry — before dispose()'s own
  // first await — so every guard observes it the moment dispose() is invoked.
  // Synchronous early-return on get/set/touch so a disposed store cannot
  // mutate state. Distinct
  // from the no-OPFS path (`!this.opfsRoot`) — disposed means the
  // owner explicitly tore the store down, OPFS-unavailable means the
  // browser never gave us a directory.
  private disposed = false;

  // True only once init() has fully opened the directory, probed writability,
  // and loaded on-disk metadata. Gates the dispose() final save so a dispose
  // that raced an incomplete init can't overwrite good `_cache_meta.json` with
  // an empty (not-yet-loaded) snapshot.
  private initialized = false;

  // In-flight init()/clear() promises. dispose() awaits them: the disposed
  // guards stop FURTHER mutations, but an OPFS operation already initiated at
  // an await cannot be cancelled — so dispose() must not resolve while one is
  // still outstanding. A newer same-URL store may take over the shared
  // per-datasetId directory the moment dispose() resolves, and a straggling
  // removeEntry/recreate from this store would clobber it.
  private pendingInit: Promise<void> | null = null;
  private pendingClear: Promise<void> | null = null;

  // The one teardown run, shared by every dispose() caller. A second caller
  // arriving while the first is still draining must receive the SAME
  // completion — resolving early on the disposed flag would tell that caller
  // the directory is safe to hand to a newer same-URL store while this
  // store's init/clear/write operations are still outstanding.
  private pendingDispose: Promise<void> | null = null;

  // Circuit breaker: consecutive OPFS-timeout count and the sticky trip
  // flag. Lifecycle state like `disposed` — deliberately NOT reset by
  // doClear()'s counter block, and doInit() refuses to resurrect a
  // tripped store: the stall is environmental, so re-arming the tier
  // against the same backend would just re-burn the timeouts. The reset
  // boundary is a FRESH store — a page reload, or a dataset switch,
  // each of which re-probes the backend once and pays at most another
  // threshold's worth of timeouts if it is still stalled.
  private consecutiveTimeouts = 0;
  private breakerTripped = false;

  /**
   * Run one OPFS operation under the per-op timeout, feeding the
   * circuit breaker. Counts ONLY the `OPFS timeout` rejection: any
   * other settlement (success or a fast error such as NotFoundError)
   * proves the backend is responsive and resets the count — the
   * breaker targets systemic stalls, not error rate. Applied to the
   * three hot real-I/O sites (get/set/delete) only: the init canary's
   * catch already degrades the store itself, the delete-barrier awaits
   * a delete whose own timeout already counted, and the dispose drain
   * runs on an already-dead store.
   */
  private async timed<T>(promise: Promise<T>, label: string): Promise<T> {
    try {
      const result = await withTimeout(promise, config.cache.opfsOperationTimeoutMs, label);
      this.consecutiveTimeouts = 0;
      return result;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.startsWith('OPFS timeout')) {
        this.consecutiveTimeouts++;
        if (
          !this.breakerTripped &&
          this.consecutiveTimeouts >= config.cache.opfsTimeoutTripThreshold
        ) {
          this.tripBreaker(label);
        }
      } else {
        this.consecutiveTimeouts = 0;
      }
      throw error;
    }
  }

  /**
   * Disable the L2 tier for the rest of this store's lifetime after
   * repeated consecutive timeouts. Nulling `opfsRoot` reuses the
   * init-failure degradation path — every entry point already
   * short-circuits on it, so all later ops are instant no-ops instead
   * of serial full-timeout burns, and `getStats().available` flips the
   * existing `opfs-unavailable` badge. The pending debounced metadata
   * save is cancelled so its timer cannot fire a full-index write
   * against the hung backend and stall dispose().
   */
  private tripBreaker(label: string): void {
    this.breakerTripped = true;
    this.metadata.cancelPendingSave();
    this.opfsRoot = null;
    log.warning(
      Modules.CACHE,
      `OPFSStore circuit breaker tripped after ${this.consecutiveTimeouts} consecutive ` +
        `OPFS timeouts (last: ${label}) — disabling the L2 disk cache for this dataset load. ` +
        'L0/L1 in-memory tiers continue to serve.'
    );
  }

  constructor(datasetId: string, baseUrl: string, maxSize: number) {
    this.datasetId = datasetId;
    this.baseUrl = baseUrl;
    this.maxSize = maxSize;
  }

  /**
   * Enumerate every `zarr-cache-*` dataset present under the viewer's
   * `luxar/` OPFS namespace directory. Reads each dataset's
   * `_cache_meta.json` directly — no OPFSStore instance is created. Used by
   * the debug-cache helpers (`window.__luxarDebug`) and the cache E2E suite
   * to inspect persisted datasets without mounting them.
   *
   * Never creates the namespace directory: on a cold origin (no `luxar/`
   * yet) the lookup's `NotFoundError` resolves to an empty list.
   */
  static async listAll(): Promise<CachedDatasetSummary[]> {
    const datasets: CachedDatasetSummary[] = [];
    try {
      let opfsRoot: FileSystemDirectoryHandle;
      try {
        opfsRoot = await getLuxarOpfsRoot({ create: false });
      } catch (error) {
        if (error instanceof DOMException && error.name === 'NotFoundError') return datasets;
        throw error;
      }
      const iterableRoot = opfsRoot as IterableFileSystemDirectoryHandle;
      for await (const [name, handle] of iterableRoot.entries()) {
        if (!name.startsWith('zarr-cache-') || handle.kind !== 'directory') continue;
        try {
          const directoryHandle = handle as FileSystemDirectoryHandle;
          const metaHandle = await directoryHandle.getFileHandle('_cache_meta.json');
          const file = await metaHandle.getFile();
          const meta = JSON.parse(await file.text()) as PersistedMetadataFile;
          datasets.push({
            url: meta.baseUrl || 'unknown',
            hash: meta.contentHash?.slice(0, 16) || 'none',
            size: meta.totalSize || 0,
            count: meta.entries?.length || 0,
          });
        } catch {
          // Skip corrupted/invalid cache directories.
        }
      }
    } catch {
      // OPFS not available.
    }
    return datasets;
  }

  /**
   * Initialize OPFS directory, prove writability, and load metadata.
   */
  async init(): Promise<void> {
    // Track the in-flight run so dispose() can await it (see pendingInit).
    const run = this.doInit();
    this.pendingInit = run;
    try {
      await run;
    } finally {
      if (this.pendingInit === run) this.pendingInit = null;
    }
  }

  private async doInit(): Promise<void> {
    // A tripped breaker is sticky for the instance lifetime — nothing
    // calls init() twice in production, but this makes stickiness
    // structural rather than incidental.
    if (this.breakerTripped) return;
    try {
      const root = await getLuxarOpfsRoot({ create: true });
      // dispose() may land during any of these awaits. Re-check after each so
      // a disposed store never probe-writes, runs orphan-cleanup deletes, or
      // resurrects a live opfsRoot handle (which would undo dispose()'s null).
      if (this.disposed) return;
      this.opfsRoot = await root.getDirectoryHandle(this.datasetId, { create: true });
      if (this.disposed) {
        this.opfsRoot = null;
        return;
      }
      await this.probeWritability();
      if (this.disposed) {
        this.opfsRoot = null;
        return;
      }
      await this.applyMetadataOnLoad();
      if (this.disposed) {
        this.opfsRoot = null;
        return;
      }
      // Only a fully-initialized store may persist a snapshot on dispose.
      this.initialized = true;
    } catch (error) {
      log.warning(
        Modules.CACHE,
        'OPFSStore unavailable (init / write-probe failed) — running without the L2 disk cache',
        error
      );
      this.opfsRoot = null;
    }
  }

  /**
   * Prove OPFS is actually WRITABLE, not merely mounted. WebKit (Safari and
   * the WKWebView native launcher) implements `navigator.storage
   * .getDirectory()` and directory/file handles but NOT the main-thread
   * `FileSystemFileHandle.createWritable()` — WebKit supports OPFS writes
   * only through worker-side `createSyncAccessHandle`. Without this probe
   * the store mounts "healthy" there and then fails EVERY put: tens of
   * thousands of write errors, a permanently empty L2, and an all-miss read
   * path (observed in the native macOS app's cache monitor). One tiny probe
   * write at init converts that failure mode into the ordinary
   * OPFS-unavailable degradation (L1-only, `opfs-unavailable` badge) with a
   * single clear warning. Timeout-wrapped like every other OPFS operation so
   * a hung handle cannot stall startup. Throws on failure — init()'s catch
   * nulls `opfsRoot`.
   */
  private async probeWritability(): Promise<void> {
    const PROBE_NAME = '.opfs-write-probe';
    // Capture the root once: dispose() nulls this.opfsRoot, so re-reading the
    // field after an await could turn a benign disposed-race into a TypeError.
    const root = this.opfsRoot!;
    await withTimeout(
      (async () => {
        const fileHandle = await root.getFileHandle(PROBE_NAME, { create: true });
        // dispose() may land during any of the probe's own awaits. init()'s
        // between-await guards can't see inside this method, so re-check here
        // before each mutation of the shared directory: never write the probe
        // after dispose, and never remove the fixed-name probe file a newer
        // same-URL store may be probing with concurrently.
        if (this.disposed) return;
        const writable = await fileHandle.createWritable();
        if (this.disposed) {
          try {
            await writable.close();
          } catch {
            // Best-effort handle release; the probe file is transient.
          }
          return;
        }
        await writable.write(new Uint8Array([1]).buffer as ArrayBuffer);
        await writable.close();
        if (this.disposed) return;
        await root.removeEntry(PROBE_NAME);
      })(),
      config.cache.opfsOperationTimeoutMs,
      'write-probe'
    );
  }

  private async applyMetadataOnLoad(): Promise<void> {
    if (!this.opfsRoot) return;
    // Capture the root once — dispose() nulls this.opfsRoot mid-await.
    const root = this.opfsRoot;
    const outcome = await this.metadata.load(root);
    // dispose() may land during load(). Bail before applying the snapshot and
    // ESPECIALLY before orphan cleanup: a disposed store's index is stale, so
    // its cleanup would classify a newer same-URL store's freshly written
    // files as orphans and delete them.
    if (this.disposed) return;
    if (!outcome) {
      // Cold start (file missing). Defaults already match the constructor.
      return;
    }
    this.index = outcome.index;
    this.totalSize = outcome.totalSize;
    this.orderCounter = outcome.orderCounter;
    this.contentHash = outcome.contentHash;
    this.validationMode = outcome.validationMode;
    this.lastValidatedAt = outcome.lastValidatedAt;
    if (outcome.needsOrphanCleanup) {
      const expected = new Set<string>();
      for (const key of this.index.keys()) {
        expected.add(keyToFileName(key));
      }
      // The stop predicate halts the bucket crawl as soon as dispose() lands
      // mid-cleanup (the crawl can span many awaits on a large cache).
      await this.metadata
        .cleanupOrphans(root, expected, () => this.disposed)
        .catch(() => {
          // Best-effort; ignore reclaim failures.
        });
    }
  }

  private metadataSnapshot(): MetadataSnapshot {
    return {
      baseUrl: this.baseUrl,
      entries: Array.from(this.index.entries()),
      totalSize: this.totalSize,
      orderCounter: this.orderCounter,
      contentHash: this.contentHash,
      validationMode: this.validationMode,
      lastValidatedAt: this.lastValidatedAt ?? undefined,
    };
  }

  /**
   * Get a file from OPFS and update LRU order.
   */
  async get(key: string, options?: { signal?: AbortSignal }): Promise<Uint8Array | undefined> {
    if (!this.readableRoot(key, options?.signal)) {
      this.countUnavailableRead(options?.signal);
      return undefined;
    }

    try {
      const data = await withOpfsReadGate(async () => {
        const root = this.readableRoot(key, options?.signal);
        if (!root) return undefined;
        return this.timed(
          (async () => {
            const fileHandle = await this.buckets.navigateToFile(root, key, false);
            const file = await fileHandle.getFile();
            return new Uint8Array(await file.arrayBuffer());
          })(),
          `get(${key})`
        );
      });

      if (!data) {
        this.countUnavailableRead(options?.signal);
        return undefined;
      }

      // Verify size matches metadata
      if (this.hasSizeMismatch(key, data)) {
        log.warning(Modules.CACHE, `OPFSStore size mismatch for ${key}, removing corrupted entry`);
        this.corruptedEntries++;
        await this.delete(key);
        this.missCount++;
        return undefined;
      }

      // Update LRU order
      this.touch(key);
      this.readCount++;

      return data;
    } catch (error) {
      // Timeouts and any other I/O failures degrade to a cache miss;
      // production code never observes a throw here (zarrita's
      // AsyncReadable.get must not throw).
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.startsWith('OPFS timeout')) {
        log.warning(Modules.CACHE, msg);
      }
      this.missCount++;
      return undefined;
    }
  }

  private readableRoot(key: string, signal?: AbortSignal): FileSystemDirectoryHandle | null {
    if (this.disposed || signal?.aborted || !this.index.has(key)) return null;
    return this.opfsRoot;
  }

  private countUnavailableRead(signal?: AbortSignal): void {
    if (signal?.aborted) this.canceledReadCount++;
    else this.missCount++;
  }

  private hasSizeMismatch(key: string, data: Uint8Array): boolean {
    const entry = this.index.get(key);
    return entry !== undefined && entry.size !== data.byteLength;
  }

  /**
   * Write a file to OPFS with LRU eviction.
   */
  async set(key: string, data: Uint8Array): Promise<void> {
    if (this.disposed || !this.opfsRoot) return;

    // Chain this write onto any in-flight write for the same key so the
    // doSet() bodies run strictly in arrival order. Previously we only
    // awaited the single pending write at entry; if ≥2 callers awaited the
    // same promise they would resume together and run doSet() — and thus
    // createWritable()+write()+close() on the SAME OPFS file — concurrently.
    // The File System Access API does not guarantee overlapping writables
    // to one file are safe (they can corrupt the file or throw). Chaining
    // makes a given key's file I/O strictly sequential. (The in-memory
    // index update cannot itself race: it is a synchronous block with no
    // await, so it is atomic per call.) `prev.then(run, run)` runs our
    // write whether the previous one resolved or rejected (doSet swallows
    // its own errors, but stay defensive). The tail-check in `finally`
    // avoids a finishing earlier write deleting a newer writer's entry.
    const prev = this.pendingWrites.get(key);
    // Clobber guard (#1073): snapshot the in-flight REAL deletes of this key
    // SYNCHRONOUSLY, right here at set() entry, in the same block that reads
    // `prev` and installs this write into the chain. Only a delete registered
    // BEFORE this write joined the chain can clobber it (its non-cancellable
    // removeEntry may already be running while its caller returned on a
    // timeout), so run() must wait for exactly those. A delete registered
    // AFTER this point sees THIS write as its `prevWrite` tail and is already
    // sequenced behind it (`real = thisWrite.then(doDelete)`), so it must NOT
    // be in the barrier — waiting on it would deadlock/self-stall (the barrier
    // would burn the full timeout on a `real` chained behind this very write).
    const inflightDeletes = OPFSStore.pendingDeletesByDataset.get(this.datasetId)?.get(key);
    const deleteBarrier = inflightDeletes && inflightDeletes.size > 0 ? [...inflightDeletes] : null;
    const run = async (): Promise<void> => {
      // Before touching the file, wait for the deletes captured above to
      // actually settle. Timeout-bounded so a genuinely hung delete cannot
      // stall the write chain (dispose drains pendingWrites) — it instead
      // DROPS the write (degrade-safe: the key stays a miss until a later
      // write succeeds).
      if (deleteBarrier && deleteBarrier.length > 0) {
        try {
          await withTimeout(
            Promise.allSettled(deleteBarrier),
            config.cache.opfsOperationTimeoutMs,
            `set(${key}) delete-barrier`
          );
        } catch {
          // A same-key delete is hung past the deadline. Drop this write
          // rather than risk it being clobbered by the eventual removeEntry.
          this.clobberBarrierWriteSkipped++;
          log.warning(
            Modules.CACHE,
            `OPFSStore dropped write for ${key}: a prior same-key delete's removeEntry did not settle within the deadline (clobber guard)`
          );
          return;
        }
      }
      await this.doSet(key, data);
    };
    // Chain behind the prior same-key write so their file I/O never overlaps.
    // Its links are internally timeout-bounded (doSet's write + the delete
    // barrier above), so this stays bounded without wrapping the whole chain.
    const writePromise = prev ? prev.then(run, run) : run();
    this.pendingWrites.set(key, writePromise);
    try {
      await writePromise;
    } finally {
      if (this.pendingWrites.get(key) === writePromise) {
        this.pendingWrites.delete(key);
      }
    }
  }

  /**
   * Internal write implementation (called by set() after serialization).
   */
  private async doSet(key: string, data: Uint8Array): Promise<void> {
    if (!this.opfsRoot) return;

    // Capture the generation at entry. If clear() (or dispose()) bumps
    // the generation while the write is pending, the post-write index
    // mutation must be skipped — otherwise a slow set() that began
    // before clear() will repopulate the just-cleared cache.
    const startGeneration = this.generation;

    const size = data.byteLength;

    // Reject oversized entries up front. A single item larger than
    // maxSize would otherwise evict every existing entry and still
    // leave totalSize > maxSize after insertion, breaking the cache
    // size invariant. Mirror of LRUCache.set()'s oversized guard.
    if (size > this.maxSize) {
      this.oversizedWriteSkipped++;
      return;
    }

    // LRU eviction until we have space — O(1) per eviction via Map
    // insertion order. Run BEFORE the quota check so the browser sees
    // the freed space when we ask navigator.storage.estimate().
    // Each iteration must make progress (index shrinks) — bail if delete()
    // fails to remove the head entry (transient I/O error) so we never
    // spin forever re-selecting the same undeletable key, and so evictions
    // counts only entries actually removed.
    //
    // Credit the write-key's existing entry against the budget: the overwrite
    // branch below frees `existingSize` when it re-inserts, so overwriting a
    // large near-capacity key must not needlessly evict OTHER keys (we already
    // skip evicting `key` itself via lruHeadExcluding).
    const existingSize = this.index.get(key)?.size ?? 0;
    while (this.totalSize - existingSize + size > this.maxSize && this.index.size > 0) {
      // Never evict the key we are writing (#1073): doSet(key) runs inside
      // key's own pendingWrites chain, and delete(key) chains behind that same
      // in-flight write — so evicting `key` here would deadlock (the delete
      // waits on the write that is awaiting the delete). It is also pointless:
      // the overwrite branch below already frees this key's old size when it
      // re-inserts the entry.
      const lruKey = this.lruHeadExcluding(key);
      if (lruKey === undefined) break; // only the key being written remains
      const before = this.index.size;
      // Note: delete() already decrements totalSize, don't double-decrement
      await this.delete(lruKey);
      if (this.index.size === before) break; // no progress — avoid spinning on a failing delete
      this.evictions++;
    }

    // Quota-pressure eviction. The own-LRU loop above only fires when
    // `totalSize` approaches `maxSize`, but the browser-granted OPFS quota
    // is frequently smaller than `maxSize` (default 2 GB) — on Firefox,
    // private mode, and small disks. Without this loop, a tight quota
    // would skip the write while the maxSize-based trigger never fires, so
    // the cache freezes holding stale entries and silently drops new ones
    // (TODO #4). checkQuota() reflects bytes actually on disk and delete()
    // frees them, so evicting LRU entries genuinely recovers quota. Evict
    // and re-check until the write fits or there is nothing left to evict.
    // Each iteration must make progress (index shrinks) — bail if delete()
    // fails to remove the head entry (transient I/O error) so we never spin.
    let hasQuota = await this.checkQuota(size);
    while (!hasQuota && this.index.size > 0) {
      // Skip the key being written — same self-deadlock reason as the own-LRU
      // loop above (#1073).
      const lruKey = this.lruHeadExcluding(key);
      if (lruKey === undefined) break; // only the key being written remains
      const before = this.index.size;
      await this.delete(lruKey);
      if (this.index.size === before) break; // no progress — avoid spinning on a failing delete
      this.evictions++;
      hasQuota = await this.checkQuota(size);
    }

    if (!hasQuota) {
      this.quotaWriteSkipped++;
      log.warning(
        Modules.CACHE,
        'OPFSStore insufficient storage quota even after eviction, skipping write'
      );
      return;
    }

    // Write to OPFS. ONLY a stale bucket handle retries — nothing else may.
    // There is no backoff and no space is reclaimed between attempts (the quota
    // eviction loop above already ran and is not re-entered), so retrying an
    // ENOSPC / quota / timeout error buys nothing and costs a second
    // `opfsOperationTimeoutMs` of caller stall plus a duplicate warning line.
    //
    // The entire navigate→createWritable→write→close chain is wrapped in
    // withTimeout so a hung handle cannot stall the cache.
    //
    // A single broken write counts exactly 1 in `writeFailures`; that invariant
    // is now STRUCTURAL (the non-stale path returns) rather than flag-guarded, so
    // don't reintroduce an "already counted" flag.
    //
    // Re-check the root: the eviction/quota awaits above can span a circuit-
    // breaker trip, which nulls `opfsRoot` while this doSet is mid-flight (the
    // entry guard already passed). Bail rather than write through a cached
    // bucket handle — that would burn another full `opfsOperationTimeoutMs`
    // against the backend the breaker just gave up on, which is exactly the
    // serial-stall cost the breaker exists to stop.
    if (!this.opfsRoot) return;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        await this.timed(
          (async () => {
            const fileHandle = await this.buckets.navigateToFile(this.opfsRoot!, key, true);
            const writable = await fileHandle.createWritable();
            // Slice the view's portion, NOT data.buffer directly. If the Uint8Array is
            // a view on a larger ArrayBuffer (e.g., from a sub-slice), data.buffer would
            // write the entire underlying buffer, corrupting the stored data. slice()
            // copies only the relevant bytes. Cast is safe: network data is never SharedArrayBuffer.
            const bytes = data.buffer.slice(
              data.byteOffset,
              data.byteOffset + data.byteLength
            ) as ArrayBuffer;
            await writable.write(bytes);
            await writable.close();
          })(),
          `set(${key})`
        );

        // Stale-write check: if clear()/dispose() ran while we were
        // writing, the post-write index update must be skipped. Best-
        // effort delete the file we just wrote so the directory matches
        // the (now-empty) index.
        if (this.generation !== startGeneration) {
          try {
            const bucket = getBucket(key);
            const bucketHandle = await this.buckets.getHandle(this.opfsRoot!, bucket, false);
            if (bucketHandle) {
              await bucketHandle.removeEntry(keyToFileName(key));
            }
          } catch {
            // Best-effort; orphaned file is harmless and will be reclaimed
            // by the next clear() / orphan cleanup pass.
          }
          return;
        }

        // Update index — delete+re-insert to move to end (MRU position)
        const existingEntry = this.index.get(key);
        if (existingEntry) {
          this.totalSize -= existingEntry.size;
          this.totalSize = Math.max(0, this.totalSize);
          this.index.delete(key);
        }
        this.index.set(key, { size, order: this.orderCounter++ });
        this.totalSize += size;
        this.writeCount++;

        // Compact order counters to prevent overflow after long sessions
        if (this.orderCounter > 1e12) {
          this.compactOrderCounter();
        }

        // Debounced metadata save
        this.scheduleMetadataSave();
        return; // Success — exit retry loop
      } catch (error) {
        const errorMsg = getErrorMessage(error);
        if (attempt === 0 && errorMsg.includes('could not be found')) {
          // Stale bucket handle from a concurrent clear() — invalidate and retry
          const bucket = getBucket(key);
          this.buckets.invalidate(bucket);
          continue;
        }
        this.writeFailures++;
        log.warning(Modules.CACHE, `OPFSStore failed to write ${key}: ${errorMsg}`);
        return;
      }
    }
  }

  /**
   * Delete a file from OPFS.
   *
   * The CALLER await is bounded by opfsOperationTimeoutMs (preserving #991 —
   * delete() runs inside doSet()'s eviction loops and get()'s corrupted-entry
   * path, none of which may stall). The chain LINK a later same-key write waits
   * on, however, is the REAL (un-timed-out) removeEntry settlement (`doDelete`),
   * registered in `pendingDeletesByDataset`: when the caller returns on timeout the
   * non-cancellable removeEntry keeps running, and a replacement write must not
   * start until it has ACTUALLY settled (#1073 clobber invariant). Index/size
   * are reconciled off that real settlement, never off the early timeout return.
   */
  async delete(key: string): Promise<void> {
    // The disposed guard matters for the in-flight-get() race: a get() that
    // entered before dispose() can hit its corrupted-entry path afterwards
    // and must not removeEntry from a directory a newer same-URL store may
    // now own (nor schedule a post-dispose metadata save below).
    if (this.disposed || !this.opfsRoot) return;

    // A key that is neither indexed nor mid-write is a genuine no-op. The
    // pendingWrites check matters: a delete arriving while the key's write is
    // still in flight (index not yet updated — e.g. a first write, or a
    // replacement whose predecessor delete already reconciled) must still
    // chain behind that write and remove it, not silently return and let the
    // write survive the later-arriving delete.
    if (!this.index.has(key) && !this.pendingWrites.has(key)) return;

    // Chain the real removeEntry behind any in-flight WRITE for this key so
    // removeEntry never overlaps a createWritable on the same file. It does NOT
    // chain behind other in-flight deletes: deletes are idempotent, so a fresh
    // delete can retry (its own working handle) even if an earlier one hangs
    // forever — that earlier op's caller already gave up on its timeout.
    const prevWrite = this.pendingWrites.get(key);
    const run = (): Promise<void> => this.doDelete(key);
    const real = prevWrite ? prevWrite.then(run, run) : run();

    // Register the real settlement so a later same-key set() waits for it even
    // after this caller returns on timeout (see pendingDeletesByDataset).
    const datasetDeletes =
      OPFSStore.pendingDeletesByDataset.get(this.datasetId) ??
      new Map<string, Set<Promise<void>>>();
    if (!OPFSStore.pendingDeletesByDataset.has(this.datasetId)) {
      OPFSStore.pendingDeletesByDataset.set(this.datasetId, datasetDeletes);
    }
    const inflight = datasetDeletes.get(key) ?? new Set<Promise<void>>();
    if (!datasetDeletes.has(key)) datasetDeletes.set(key, inflight);
    inflight.add(real);
    void real.finally(() => {
      inflight.delete(real);
      if (inflight.size === 0 && datasetDeletes.get(key) === inflight) {
        datasetDeletes.delete(key);
        if (
          datasetDeletes.size === 0 &&
          OPFSStore.pendingDeletesByDataset.get(this.datasetId) === datasetDeletes
        ) {
          OPFSStore.pendingDeletesByDataset.delete(this.datasetId);
        }
      }
    });

    try {
      await this.timed(real, `delete(${key})`);
    } catch (error) {
      // Timeout: the caller unblocks, but `real` stays outstanding in
      // pendingDeletes and continues to gate any replacement write until the
      // removeEntry settles. Other errors cannot reach here — doDelete never
      // rejects.
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.startsWith('OPFS timeout')) {
        log.warning(Modules.CACHE, msg);
      }
    }
  }

  /**
   * Real (un-timed-out) delete of a single file plus index/size reconcile.
   * Runs to ACTUAL settlement as a link in the per-key serialization chain so a
   * later same-key write can never start (and then be clobbered) while this
   * removeEntry is still in flight (#1073). Never rejects.
   */
  private async doDelete(key: string): Promise<void> {
    if (!this.opfsRoot) return;
    const entry = this.index.get(key);
    // Nothing indexed: either a concurrent retry-delete already reconciled, or
    // the in-flight write this delete chained behind was dropped/failed and
    // never indexed — no file to remove in either case.
    if (!entry) return;

    try {
      // [cache OOS] Order matters: file removal must succeed BEFORE we update
      // `totalSize` and the index. The previous order (totalSize decrement →
      // removeEntry → index.delete) left the accumulator desynchronised when
      // removeEntry threw — totalSize had already been decremented but the file
      // was still on disk and the index still had the entry. The next `set()`
      // then made eviction decisions based on the wrong size. With this
      // ordering, a transient removeEntry failure leaves all three pieces of
      // state unchanged so callers can retry safely.
      const bucket = getBucket(key);
      const bucketHandle = await this.buckets.getHandle(this.opfsRoot!, bucket, false);
      if (bucketHandle) {
        await bucketHandle.removeEntry(keyToFileName(key));
      }
    } catch (error) {
      // NotFound means the desired disk state already holds. Reconcile the
      // stale index entry below; retaining it would pin a phantom at the LRU
      // head and make every eviction loop stop for lack of progress. Any other
      // I/O failure is potentially transient, so preserve index/size atomically
      // and let a later caller retry.
      if (!isNotFoundError(error)) {
        const msg = error instanceof Error ? error.message : String(error);
        log.warning(Modules.CACHE, `OPFSStore failed to delete ${key}: ${msg}`);
        return;
      }
    }

    // Re-read the entry: a concurrent retry-delete may have reconciled while
    // our removeEntry was in flight (deletes do not serialize against each
    // other). Guard against double-subtracting `totalSize`.
    const current = this.index.get(key);
    if (!current) return;
    this.totalSize = Math.max(0, this.totalSize - current.size);
    this.index.delete(key);
    this.scheduleMetadataSave();
  }

  /**
   * First key in LRU order (Map insertion order) that is not `exclude`. Used by
   * doSet()'s eviction loops to skip the key currently being written (#1073) —
   * see the loop comments for the self-deadlock rationale.
   */
  private lruHeadExcluding(exclude: string): string | undefined {
    for (const key of this.index.keys()) {
      if (key !== exclude) return key;
    }
    return undefined;
  }

  /**
   * Clear all OPFS data for this dataset.
   *
   * Uses atomic delete-and-recreate instead of iterating entries, which avoids
   * race conditions when a previous page context still holds open file handles
   * (e.g., quick-succession page refreshes with fire-and-forget L2 writes).
   */
  async clear(): Promise<void> {
    // A disposed store must not recreate/wipe the shared per-datasetId OPFS
    // directory — that could clobber a newer same-URL store.
    if (this.disposed) return;
    // Chain concurrent clears (mirroring set()'s per-key chaining) so
    // pendingClear is always the TAIL of every in-flight clear — dispose()
    // awaits that single promise and is covered no matter how many clears
    // were racing.
    const prev = this.pendingClear;
    const start = (): Promise<void> => this.doClear();
    const run = prev ? prev.then(start, start) : start();
    this.pendingClear = run;
    try {
      await run;
    } finally {
      if (this.pendingClear === run) this.pendingClear = null;
    }
  }

  private async doClear(): Promise<void> {
    // Bump generation FIRST so any in-flight doSet() that completes
    // after this point sees the mismatch and skips its index update.
    this.generation++;

    // Drain pending same-key writes. Each set() pushes a promise into
    // pendingWrites; awaiting them lets in-flight writes finish their
    // file I/O — they will detect the generation mismatch and skip
    // mutating the index.
    if (this.pendingWrites.size > 0) {
      await Promise.allSettled([...this.pendingWrites.values()]);
    }

    // dispose() may have landed during the drain above. Bail BEFORE the
    // in-memory reset and the directory wipe: dispose() awaits this clear,
    // and skipping the reset keeps the final dispose-time metadata snapshot
    // consistent with the (un-wiped) files still on disk.
    if (this.disposed) return;

    // Clear all in-memory state first — ensures no stale handles are used
    // even if the filesystem operations below fail
    this.index = new Map();
    this.buckets.clear();
    this.totalSize = 0;
    this.orderCounter = 0;
    this.contentHash = null;
    this.validationMode = 'none';
    this.lastValidatedAt = null;
    this.readCount = 0;
    this.writeCount = 0;
    this.missCount = 0;
    this.oversizedWriteSkipped = 0;
    this.quotaWriteSkipped = 0;
    this.evictions = 0;
    this.writeFailures = 0;
    this.corruptedEntries = 0;
    this.clobberBarrierWriteSkipped = 0;
    this.metadata.parseFailures = 0;
    this.metadata.orphansRemoved = 0;

    if (this.opfsRoot) {
      try {
        // Atomic approach: remove entire dataset directory and recreate it.
        // This is more robust than iterating entries, which can fail if a
        // previous page context still holds open file handles on bucket dirs.
        const root = await getLuxarOpfsRoot({ create: true });
        // dispose() may have landed while we awaited the namespace dir. Null the
        // root and bail before the destructive removeEntry — a newer same-URL
        // store may be about to take over this directory.
        if (this.disposed) {
          this.opfsRoot = null;
          return;
        }
        await root.removeEntry(this.datasetId, { recursive: true });
        this.opfsRoot = await root.getDirectoryHandle(this.datasetId, { create: true });
      } catch (error) {
        log.warning(
          Modules.CACHE,
          'OPFSStore failed to clear atomically, retrying entry-by-entry',
          error
        );
        // Fallback: try to remove entries individually (best-effort)
        try {
          if (this.opfsRoot) {
            const iterableRoot = this.opfsRoot as IterableFileSystemDirectoryHandle;
            for await (const name of iterableRoot.keys()) {
              // Stop deleting the moment dispose() lands mid-crawl.
              if (this.disposed) break;
              try {
                await this.opfsRoot.removeEntry(name, { recursive: true });
              } catch {
                // Skip locked/in-use entries — they'll be orphaned but harmless
              }
            }
          }
        } catch {
          // Iterator itself failed — directory may be inaccessible
        }
        // Re-obtain a fresh root handle regardless
        try {
          const root = await getLuxarOpfsRoot({ create: true });
          this.opfsRoot = await root.getDirectoryHandle(this.datasetId, { create: true });
        } catch {
          this.opfsRoot = null;
        }
      }
      // A dispose() that landed during the awaits above already nulled
      // opfsRoot; the reassignments here must not resurrect a live handle on
      // a disposed store (it would re-arm scheduleMetadataSave's root gate).
      if (this.disposed) {
        this.opfsRoot = null;
      }
    }
  }

  /**
   * Check if enough storage quota is available.
   */
  async checkQuota(requiredBytes: number): Promise<boolean> {
    try {
      const estimate = await navigator.storage.estimate();
      const available = (estimate.quota || 0) - (estimate.usage || 0);
      return available > requiredBytes * 1.1; // 10% safety margin
    } catch {
      return true; // Assume OK if API unavailable
    }
  }

  /**
   * Get cache statistics.
   */
  getStats(): {
    size: number;
    count: number;
    reads: number;
    writes: number;
    misses: number;
    canceledReads: number;
    activeReads: number;
    queuedReads: number;
    oversizedWriteSkipped: number;
    quotaWriteSkipped: number;
    evictions: number;
    writeFailures: number;
    corruptedEntries: number;
    clobberBarrierWriteSkipped: number;
    metadataParseFailures: number;
    orphanedFilesRemoved: number;
    /**
     * S2: `true` when OPFS was reachable on init and the store is
     * still alive. `false` when init couldn't acquire a directory
     * handle (browser without OPFS support, private mode in some
     * configs), after the circuit breaker trips, or after dispose().
     * Drives the `opfs-unavailable` status badge.
     */
    available: boolean;
    /**
     * `true` once the circuit breaker disabled the tier after
     * consecutive OPFS timeouts. Distinguishes "gave up after repeated
     * stalls" from "never had OPFS" in debug snapshots and tests; the
     * monitor badge keys on `available` alone.
     */
    breakerTripped: boolean;
  } {
    const readGate = getOpfsReadGateStats();
    return {
      size: this.totalSize,
      count: this.index.size,
      reads: this.readCount,
      writes: this.writeCount,
      misses: this.missCount,
      canceledReads: this.canceledReadCount,
      activeReads: readGate.active,
      queuedReads: readGate.queued,
      oversizedWriteSkipped: this.oversizedWriteSkipped,
      quotaWriteSkipped: this.quotaWriteSkipped,
      evictions: this.evictions,
      writeFailures: this.writeFailures,
      corruptedEntries: this.corruptedEntries,
      clobberBarrierWriteSkipped: this.clobberBarrierWriteSkipped,
      metadataParseFailures: this.metadata.parseFailures,
      orphanedFilesRemoved: this.metadata.orphansRemoved,
      available: this.opfsRoot !== null && !this.disposed,
      breakerTripped: this.breakerTripped,
    };
  }

  /**
   * Update LRU order for a key.
   */
  touch(key: string): void {
    if (this.disposed) return;
    const entry = this.index.get(key);
    if (entry) {
      // Delete+re-insert to move to end (MRU position) — O(1) with Map
      this.index.delete(key);
      this.index.set(key, { size: entry.size, order: this.orderCounter++ });

      // Compact order counters to prevent overflow after long sessions
      if (this.orderCounter > 1e12) {
        this.compactOrderCounter();
      }

      // Intentionally do NOT schedule a metadata save here. touch() only
      // reorders the in-memory LRU; persisting that on every read forced a
      // full-index JSON serialize per read-burst. Read-driven order is
      // best-effort — it rides along with the next structural save (set /
      // delete / contentHash / validationMode) and is flushed
      // unconditionally on dispose(). Losing it on a hard crash only yields
      // a slightly stale eviction order next session, which is harmless.
    }
  }

  /**
   * Set content hash for cache invalidation.
   */
  setContentHash(hash: string | null): void {
    if (this.disposed) return;
    this.contentHash = hash;
    this.scheduleMetadataSave();
  }

  /**
   * Get cached content hash.
   */
  getContentHash(): string | null {
    return this.contentHash;
  }

  /**
   * Record the validation mode used for this dataset. Persisted to
   * `_cache_meta.json` so a follow-up session can re-evaluate (e.g.
   * a TTL window).
   *
   * `validated` (default true) means a genuine validation just succeeded,
   * which stamps `lastValidatedAt = now`. The no-token/offline branch passes
   * `validated: false`: it still records the mode, but must NOT slide the TTL
   * clock forward on every revisit — it only establishes the baseline the
   * first time (when `lastValidatedAt` is still unset) so a headerless-server
   * cache can still age out.
   */
  setValidationMode(mode: CacheValidationMode, options?: { validated?: boolean }): void {
    if (this.disposed) return;
    this.validationMode = mode;
    const validated = options?.validated ?? true;
    if (validated || this.lastValidatedAt == null) {
      this.lastValidatedAt = Date.now();
    }
    this.scheduleMetadataSave();
  }

  /**
   * Read the current validation mode and last-validated timestamp.
   * Returned together so callers can apply a TTL check atomically.
   */
  getValidationState(): { mode: CacheValidationMode; lastValidatedAt: number | null } {
    return { mode: this.validationMode, lastValidatedAt: this.lastValidatedAt };
  }

  /**
   * Renumber all order entries to prevent orderCounter overflow.
   * Called when orderCounter exceeds a safe threshold (1e12).
   */
  private compactOrderCounter(): void {
    const entries = [...this.index.entries()].sort((a, b) => a[1].order - b[1].order);
    this.index.clear();
    entries.forEach(([key, entry], i) => {
      entry.order = i;
      this.index.set(key, entry);
    });
    this.orderCounter = entries.length;
  }

  /**
   * Tear down the store. Marks the store disposed, drains pending writes,
   * awaits any in-flight metadata save, then flushes a final snapshot so
   * a read-only session's LRU order still gets persisted.
   *
   * Every caller shares ONE completion (pendingDispose): dispose() resolving
   * is the take-over signal for a newer same-URL store, so a concurrent or
   * repeat caller must wait for the same drain rather than resolve early on
   * the disposed flag.
   *
   * Order matters:
   * 1. Set disposed = true SYNCHRONOUSLY, before the first await. dispose()
   *    itself suspends below (drain + awaitInFlight), and every `disposed`
   *    guard — init()'s re-checks, clear(), the validation setters — must
   *    observe the flag from the moment dispose() is invoked, not only once
   *    the final flush completes; otherwise a mid-init store could keep
   *    probe-writing or orphan-cleaning the shared directory during that
   *    window. The final flush below calls the metadata manager directly, so
   *    it is unaffected by the flag.
   * 2. Bump generation so any in-flight doSet that resolves afterwards
   *    detects the mismatch and skips its index update.
   * 3. Cancel the debounced timer (we flush directly below).
   * 4. Await any in-flight init() or clear(). Their disposed guards stop
   *    FURTHER mutations, but an OPFS operation already initiated at an
   *    await cannot be cancelled — dispose() resolving is the signal that a
   *    newer same-URL store may take over the shared directory, so nothing
   *    this store started may still be outstanding at that point.
   * 5. Drain pendingWrites so file I/O for in-flight set() calls finishes
   *    and the index reflects settled state before we snapshot it.
   * 5b. Drain in-flight real deletes (pendingDeletesByDataset): doDelete
   *    chains behind pendingWrites, so a timed-out delete's non-cancellable
   *    removeEntry + reconcile can still be outstanding — draining lets it
   *    finish before we snapshot (#1073). Bounded by the op timeout; a delete
   *    that outlives the drain stays in the static registry, which is what
   *    actually protects a successor same-URL store's writes.
   * 6. Await any metadata save already mid-write so the final save wins
   *    on disk (last writer), then write the latest snapshot. The flush is
   *    unconditional (not gated on hasPendingSave) because read-driven LRU
   *    order no longer schedules its own save (see touch()); dispose is
   *    where a read-only session's order gets persisted. Deadline-bounded:
   *    a stalled backend must not leave dispose() unresolved.
   */
  async dispose(): Promise<void> {
    if (this.pendingDispose) return this.pendingDispose;
    this.pendingDispose = this.doDispose();
    return this.pendingDispose;
  }

  // doDispose() is invoked synchronously from dispose(), and an async body
  // runs synchronously up to its first await — so `disposed = true` below is
  // still observable the moment dispose() is called.
  private async doDispose(): Promise<void> {
    this.disposed = true;
    this.generation++;

    this.metadata.cancelPendingSave();

    // Neither doInit() nor doClear() can reject (both swallow their own
    // errors), but keep dispose() unable to throw regardless.
    if (this.pendingInit) await this.pendingInit.catch(() => {});
    if (this.pendingClear) await this.pendingClear.catch(() => {});

    if (this.pendingWrites.size > 0) {
      await Promise.allSettled([...this.pendingWrites.values()]);
    }

    // Drain in-flight REAL deletes (#1073). doDelete now chains behind
    // pendingWrites, so a delete whose caller returned on a timeout can still
    // have its non-cancellable removeEntry + index/size reconcile OUTSTANDING
    // here. Draining lets the removeEntry complete and the index settle BEFORE
    // metadataSnapshot(), so the persisted snapshot doesn't list a key whose
    // file is mid-removal. Bounded by the op timeout so a genuinely hung
    // removeEntry cannot stall dispose (we deliberately do NOT early-return
    // inside doDelete on `disposed` — that would leave the file orphaned).
    // NOTE the drain is best-effort, NOT the cross-instance clobber guard: a
    // removeEntry that outlives this deadline stays registered in the STATIC
    // pendingDeletesByDataset (keyed by datasetId), so a successor same-URL
    // store's set() barrier still gates on it — that registry, not this drain,
    // is what stops a straggling delete from clobbering the successor's
    // replacement write. The cost of an out-drained delete is only a stale
    // final snapshot (a phantom index entry the next session self-heals as a
    // miss / NotFound reconcile), never a clobber.
    const datasetDeletes = OPFSStore.pendingDeletesByDataset.get(this.datasetId);
    if (datasetDeletes && datasetDeletes.size > 0) {
      const deletes: Promise<void>[] = [];
      for (const inflight of datasetDeletes.values()) deletes.push(...inflight);
      try {
        await withTimeout(
          Promise.allSettled(deletes),
          config.cache.opfsOperationTimeoutMs,
          'dispose delete-drain'
        );
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        if (msg.startsWith('OPFS timeout')) {
          log.warning(Modules.CACHE, msg);
        }
      }
    }

    // Metadata flush, deadline-bounded like every other OPFS await in this
    // store. Neither call rejects on its own (both swallow + log), so the only
    // rejection here is the timeout — and without it these were the last
    // unbounded OPFS awaits: a save already mid-write when the backend stalled
    // would leave dispose() unresolved FOREVER, hanging the dataset switch that
    // is waiting to take over the directory. Gate the save on `initialized` so
    // a dispose that raced an incomplete init() does not overwrite good on-disk
    // metadata with an empty (not-yet-loaded) snapshot; a normally-initialized
    // store still persists its final LRU-order snapshot as before.
    try {
      await withTimeout(
        (async () => {
          await this.metadata.awaitInFlight();
          if (this.opfsRoot && this.initialized) {
            await this.metadata.save(this.opfsRoot, this.metadataSnapshot());
          }
        })(),
        config.cache.opfsOperationTimeoutMs,
        'dispose metadata flush'
      );
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.startsWith('OPFS timeout')) {
        log.warning(Modules.CACHE, msg);
      }
    }

    // Null the root AFTER the final save so a debounced scheduleMetadataSave()
    // (gated on !this.opfsRoot) becomes a no-op and getStats/clear see the
    // store as unavailable.
    this.opfsRoot = null;
  }

  // ========== Private Methods ==========

  private scheduleMetadataSave(): void {
    // The disposed check closes a re-arm hole: dispose() cancels the pending
    // timer at ENTRY, so a mutation that slips in during dispose()'s awaits
    // (opfsRoot is only nulled at the end) could otherwise arm a fresh timer
    // whose captured root writes `_cache_meta.json` after dispose() resolved.
    if (this.disposed || !this.opfsRoot) return;
    this.metadata.scheduleSave({
      root: this.opfsRoot,
      getSnapshot: () => this.metadataSnapshot(),
      delayMs: OPFSStore.METADATA_SAVE_DELAY,
      onError: (error) => {
        this.writeFailures++;
        log.warning(
          Modules.CACHE,
          `OPFSStore metadata save failed: ${getErrorMessage(error)}`,
          error
        );
      },
    });
  }
}
