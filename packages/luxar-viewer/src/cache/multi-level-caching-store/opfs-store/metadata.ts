import type { OPFSMetadata, CacheValidationMode } from '../../types';
import { OPFS_ENCODING_VERSION } from '../../types';
import { log, Modules } from '../../../utils/log';
import { getErrorMessage } from '../../../utils/format-error';

type IterableFileSystemDirectoryHandle = FileSystemDirectoryHandle & {
  keys(): AsyncIterableIterator<string>;
};

export interface MetadataSnapshot {
  baseUrl: string;
  entries: Array<[string, { size: number; order: number }]>;
  totalSize: number;
  orderCounter: number;
  contentHash: string | null;
  validationMode: CacheValidationMode;
  lastValidatedAt?: number;
}

/**
 * Applied to OPFSStore on `load()` return. `needsOrphanCleanup` is true
 * when the metadata file existed but didn't parse — caller should run
 * `cleanupOrphans` to reclaim files that no longer have index entries.
 */
export interface LoadOutcome {
  index: Map<string, { size: number; order: number }>;
  totalSize: number;
  orderCounter: number;
  contentHash: string | null;
  validationMode: CacheValidationMode;
  lastValidatedAt: number | null;
  needsOrphanCleanup: boolean;
}

const METADATA_FILE = '_cache_meta.json';

async function writeSnapshot(
  root: FileSystemDirectoryHandle,
  snapshot: MetadataSnapshot
): Promise<void> {
  const metaHandle = await root.getFileHandle(METADATA_FILE, { create: true });
  const writable = await metaHandle.createWritable();
  const metadata: OPFSMetadata = {
    baseUrl: snapshot.baseUrl,
    entries: snapshot.entries,
    totalSize: snapshot.totalSize,
    orderCounter: snapshot.orderCounter,
    contentHash: snapshot.contentHash,
    encodingVersion: OPFS_ENCODING_VERSION,
    validationMode: snapshot.validationMode,
    lastValidatedAt: snapshot.lastValidatedAt,
  };
  await writable.write(JSON.stringify(metadata));
  await writable.close();
}

/**
 * Owns the OPFS metadata file (`_cache_meta.json`) lifecycle:
 * load + parse on init, debounced save on mutation, orphan cleanup
 * when the file is corrupt, and the in-flight + pending-timer state
 * that lets `OPFSStore.dispose()` flush cleanly.
 *
 * Counters (`parseFailures`, `orphansRemoved`) are surfaced through
 * `OPFSStore.getStats()`.
 */
export class OPFSMetadataManager {
  parseFailures = 0;
  orphansRemoved = 0;

  private timer: ReturnType<typeof setTimeout> | null = null;
  private inFlight: Promise<void> | null = null;

  /**
   * Read and parse `_cache_meta.json`. Returns null on cold start
   * (file missing). On parse failure, returns a fresh snapshot with
   * `needsOrphanCleanup = true` and increments `parseFailures` —
   * caller should run `cleanupOrphans` to reclaim stray files.
   */
  async load(root: FileSystemDirectoryHandle): Promise<LoadOutcome | null> {
    try {
      const metaHandle = await root.getFileHandle(METADATA_FILE);
      const file = await metaHandle.getFile();
      const meta: OPFSMetadata = JSON.parse(await file.text());

      // Encoding-version mismatch ⇒ stale directory: previous cache
      // entries used a different keyToFileName encoding and won't be
      // findable. Treat as cold cache; the cache is best-effort and
      // rebuilds itself in seconds.
      const persistedVersion = meta.encodingVersion ?? 1;
      if (persistedVersion !== OPFS_ENCODING_VERSION) {
        log.info(
          Modules.CACHE,
          `OPFSStore encoding version ${persistedVersion} != ${OPFS_ENCODING_VERSION}, starting fresh`
        );
        return {
          index: new Map(),
          totalSize: 0,
          orderCounter: 0,
          contentHash: null,
          validationMode: 'none',
          lastValidatedAt: null,
          needsOrphanCleanup: false,
        };
      }

      // Reconstruct Map sorted by ascending order so Map insertion order = LRU order.
      const entries = (meta.entries || []).slice();
      entries.sort((a, b) => a[1].order - b[1].order);
      const index = new Map(entries);

      // R6e: defensive hardening against partial metadata corruption.
      // Values like NaN, Infinity, or negatives are accepted by `|| 0`
      // but surface as nonsensical stats downstream. Clamp to non-
      // negative and recompute from the live entries when the
      // persisted value disagrees by more than a trivial amount —
      // entries[] is the source of truth for what's actually stored.
      const persistedTotal =
        Number.isFinite(meta.totalSize) && meta.totalSize >= 0 ? meta.totalSize : 0;
      const computedTotal = entries.reduce(
        (sum, [, e]) => sum + (Number.isFinite(e.size) && e.size > 0 ? e.size : 0),
        0
      );
      const totalSize =
        Math.abs(persistedTotal - computedTotal) > 1 ? computedTotal : persistedTotal;
      const orderCounter =
        Number.isFinite(meta.orderCounter) && meta.orderCounter >= 0 ? meta.orderCounter : 0;

      return {
        index,
        totalSize,
        orderCounter,
        contentHash: meta.contentHash || null,
        validationMode: meta.validationMode ?? 'none',
        lastValidatedAt: meta.lastValidatedAt ?? null,
        needsOrphanCleanup: false,
      };
    } catch (error) {
      // Two cases reach here:
      //  - getFileHandle threw "not found" → no metadata yet, cold start
      //  - JSON.parse threw → metadata file is corrupt; treat as cold
      //    start, count the failure, and signal that the caller should
      //    run a best-effort orphan cleanup so files left over from
      //    the corrupt run don't take up quota indefinitely.
      const wasParseFailure =
        error instanceof SyntaxError ||
        (error instanceof Error && /JSON|parse|Unexpected/i.test(error.message));
      if (wasParseFailure) {
        this.parseFailures++;
        log.warning(
          Modules.CACHE,
          'OPFSStore metadata corrupt, starting fresh and reclaiming orphans'
        );
        return {
          index: new Map(),
          totalSize: 0,
          orderCounter: 0,
          contentHash: null,
          validationMode: 'none',
          lastValidatedAt: null,
          needsOrphanCleanup: true,
        };
      }
      return null;
    }
  }

  /**
   * Schedule a debounced metadata save. Replaces any previously-
   * scheduled save (last writer wins on the timer). When the timer
   * fires, `getSnapshot` is called to produce the latest state and
   * the file is written; failures invoke `onError`.
   */
  scheduleSave(params: {
    root: FileSystemDirectoryHandle;
    getSnapshot: () => MetadataSnapshot;
    delayMs: number;
    onError: (error: unknown) => void;
  }): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.inFlight = writeSnapshot(params.root, params.getSnapshot())
        .catch((error) => {
          params.onError(error);
        })
        .finally(() => {
          this.inFlight = null;
        });
    }, params.delayMs);
  }

  /** True if a debounced save is queued and has not yet fired. */
  hasPendingSave(): boolean {
    return this.timer !== null;
  }

  /**
   * Cancel the debounced timer without running its action. Caller is
   * responsible for any final synchronous save (typically only useful
   * during `dispose`, paired with a direct `save()` call).
   */
  cancelPendingSave(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  /** Await any save started by the timer; no-op if none. */
  async awaitInFlight(): Promise<void> {
    if (this.inFlight) {
      try {
        await this.inFlight;
      } catch {
        // Already-logged inside scheduleSave's onError handler.
      }
    }
  }

  /**
   * Synchronously write the snapshot to disk. Used by `dispose` for
   * the final flush; swallows + logs errors so dispose never throws.
   */
  async save(root: FileSystemDirectoryHandle, snapshot: MetadataSnapshot): Promise<void> {
    try {
      await writeSnapshot(root, snapshot);
    } catch (error) {
      log.warning(
        Modules.CACHE,
        `OPFSStore failed to save metadata: ${getErrorMessage(error)}`,
        error
      );
    }
  }

  /**
   * Reclaim OPFS files that exist on disk but have no entry in the
   * provided expected set. Called from `OPFSStore.init()` after a
   * metadata-parse failure; safe to skip otherwise (the cache
   * rebuilds itself in seconds and orphans are bounded by quota).
   *
   * Increments `orphansRemoved` per file actually deleted.
   *
   * `shouldStop` is re-polled across the crawl's awaits; when it turns true
   * (the owning store was disposed) the crawl halts before its next delete,
   * so a stale expected-set can never keep reclaiming files a newer same-URL
   * store is concurrently writing.
   */
  async cleanupOrphans(
    root: FileSystemDirectoryHandle,
    expectedFileNames: Set<string>,
    shouldStop?: () => boolean
  ): Promise<void> {
    const iterableRoot = root as IterableFileSystemDirectoryHandle;
    for await (const bucketName of iterableRoot.keys()) {
      if (shouldStop?.()) return;
      // Only iterate hex-buckets (00-ff); skip _cache_meta.json itself.
      if (!/^[0-9a-f]{2}$/.test(bucketName)) continue;
      try {
        const bucketHandle = await root.getDirectoryHandle(bucketName);
        const iterableBucket = bucketHandle as IterableFileSystemDirectoryHandle;
        for await (const fileName of iterableBucket.keys()) {
          if (shouldStop?.()) return;
          if (expectedFileNames.has(fileName)) continue;
          try {
            await bucketHandle.removeEntry(fileName);
            this.orphansRemoved++;
          } catch {
            // Skip file we can't remove; surface in stats but don't bail.
          }
        }
      } catch {
        // Bucket may have disappeared mid-scan; ignore.
      }
    }
  }
}
