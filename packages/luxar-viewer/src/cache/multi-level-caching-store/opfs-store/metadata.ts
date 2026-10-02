import type { OPFSMetadata, CacheValidationMode } from '../../types';
import { OPFS_ENCODING_VERSION } from '../../types';
import { log, Modules } from '../../../utils/log';
import { getErrorMessage } from '../../../utils/format-error';
import { perfCounters } from '../../../profiling/perf-counters';

/** Perf counter: every `scheduleSave` call (debounced or not). */
const S_SAVE_ATTEMPTS = perfCounters.slot('opfs.saveAttempts');
/** Perf counter: every index-file write actually issued (debounced or final). */
const S_INDEX_SAVES = perfCounters.slot('opfs.indexSaves');

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

/**
 * The directory's dataset IDENTITY: the content hash its chunk files were
 * written under. Separate from the index because the index is rewritten on a
 * debounce and a reload can land before any save of it completes, while this
 * file changes only when the hash does and is written BEFORE the first chunk
 * under that hash (`OPFSStore.doSet` waits for it). So every chunk file in the
 * directory has a known provenance even when the index never landed.
 */
export const IDENTITY_FILE = '_cache_identity.json';

interface PersistedIdentity {
  contentHash: string | null;
  encodingVersion: number;
}

/**
 * Read the persisted dataset identity. Returns the content hash, or null when
 * the file is missing, empty (a first write cut off before close), unparsable,
 * hashless, or from another key encoding — every case where the files'
 * provenance is unknown.
 */
export async function loadIdentity(root: FileSystemDirectoryHandle): Promise<string | null> {
  try {
    const file = await (await root.getFileHandle(IDENTITY_FILE)).getFile();
    const text = await file.text();
    if (text.length === 0) return null;
    const identity = JSON.parse(text) as Partial<PersistedIdentity>;
    if (identity.encodingVersion !== OPFS_ENCODING_VERSION) return null;
    return typeof identity.contentHash === 'string' && identity.contentHash.length > 0
      ? identity.contentHash
      : null;
  } catch {
    return null;
  }
}

/** Write the dataset identity (see {@link IDENTITY_FILE}). Throws on failure. */
export async function writeIdentity(
  root: FileSystemDirectoryHandle,
  contentHash: string | null
): Promise<void> {
  const handle = await root.getFileHandle(IDENTITY_FILE, { create: true });
  const writable = await handle.createWritable();
  const identity: PersistedIdentity = { contentHash, encodingVersion: OPFS_ENCODING_VERSION };
  await writable.write(JSON.stringify(identity));
  await writable.close();
}

/** Arguments of {@link OPFSMetadataManager.scheduleSave}. */
export interface ScheduleSaveParams {
  root: FileSystemDirectoryHandle;
  getSnapshot: () => MetadataSnapshot;
  /** Trailing debounce: quiet time after the last call before writing. */
  delayMs: number;
  /**
   * Ceiling on how long a continuous stream of calls may postpone the write,
   * measured from the first call since the last write. Omitted = unbounded
   * (pure trailing debounce).
   */
  maxWaitMs?: number;
  /**
   * Leading edge: the first call after a quiet period (no save started for
   * `delayMs`) writes within this many ms, and later calls in the same window
   * cannot push that write back. Omitted = no leading edge.
   */
  leadingDelayMs?: number;
  onError: (error: unknown) => void;
}

async function writeSnapshot(
  root: FileSystemDirectoryHandle,
  snapshot: MetadataSnapshot
): Promise<void> {
  perfCounters.add(S_INDEX_SAVES);
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
  /** Latest scheduleSave() arguments not yet written (null once written/cancelled). */
  private params: ScheduleSaveParams | null = null;
  /** Epoch ms of the first scheduleSave() since the last write: the maxWait anchor. */
  private firstUnsavedAt: number | null = null;
  /** Epoch ms the last index write started (null = none this session). */
  private lastSaveStartedAt: number | null = null;
  /** Epoch ms the pending leading-edge save is due (null = no leading edge armed). */
  private leadingDeadline: number | null = null;
  /** A save came due while one was in flight; write once more when it settles. */
  private saveQueuedBehindFlight = false;

  /**
   * Read and parse `_cache_meta.json`. Returns null on cold start
   * (file missing, or empty: a first save interrupted before close). On parse failure, returns a fresh snapshot with
   * `needsOrphanCleanup = true` and increments `parseFailures` —
   * caller should run `cleanupOrphans` to reclaim stray files.
   */
  async load(root: FileSystemDirectoryHandle): Promise<LoadOutcome | null> {
    try {
      const metaHandle = await root.getFileHandle(METADATA_FILE);
      const file = await metaHandle.getFile();
      const text = await file.text();
      // Zero bytes means a FIRST save that never got to close(). The file
      // exists from getFileHandle({ create: true }), but its bytes only land
      // at close(), and a page that navigates away mid-save stops in between.
      // On Chromium a reload inside the save debounce leaves exactly this. It
      // means "never saved", which is a cold start, not a corrupt index.
      if (text.length === 0) return null;
      const meta: OPFSMetadata = JSON.parse(text);

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
      // A NotFoundError is the cold start, whatever its message says: the
      // message-sniff below would otherwise read a message that NAMES the file
      // (`..._cache_meta.json`) as "JSON" and count a parse failure.
      const isNotFound =
        typeof error === 'object' &&
        error !== null &&
        (error as { name?: unknown }).name === 'NotFoundError';
      const wasParseFailure =
        !isNotFound &&
        (error instanceof SyntaxError ||
          (error instanceof Error && /JSON|parse|Unexpected/i.test(error.message)));
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
   * Schedule a debounced metadata save.
   *
   * Trailing debounce of `delayMs`, BOUNDED by `maxWaitMs`: a call re-arms the
   * timer (last writer wins), but never past `maxWaitMs` after the first
   * still-unsaved call. Without the ceiling a continuous write stream re-arms
   * the timer forever — measured in real playback: 65k save attempts and zero
   * index writes in 53 s, so an interrupted session left an index of ~40
   * entries for ~5,000 files on disk. When the timer fires, `getSnapshot` is
   * called to produce the latest state and the file is written; failures
   * invoke `onError`.
   *
   * With `leadingDelayMs`, the first call after a quiet period (no write
   * started for `delayMs`) is written within that delay instead. The trailing
   * debounce alone lost a whole session's index to a reload landing inside it,
   * because a save started at unload never completes across a navigation
   * (measured on Chromium). A sustained burst still gets the trailing debounce
   * once the leading write has gone out.
   *
   * Index writes never overlap: a save due while the previous one is still in
   * flight is deferred until it settles, and any number of such saves collapse
   * into ONE follow-up write of the then-latest snapshot (two concurrent
   * `createWritable()` streams on one file are not safe).
   */
  scheduleSave(params: ScheduleSaveParams): void {
    perfCounters.add(S_SAVE_ATTEMPTS);
    const now = Date.now();
    if (this.firstUnsavedAt === null) {
      this.firstUnsavedAt = now;
      this.leadingDeadline = this.leadingDeadlineFor(params, now);
    }
    this.params = params;
    let delay = params.delayMs;
    if (params.maxWaitMs !== undefined && Number.isFinite(params.maxWaitMs)) {
      const ceiling = this.firstUnsavedAt + Math.max(0, params.maxWaitMs);
      delay = Math.max(0, Math.min(delay, ceiling - now));
    }
    if (this.leadingDeadline !== null) {
      delay = Math.max(0, Math.min(delay, this.leadingDeadline - now));
    }
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.fire();
    }, delay);
  }

  /** When a leading-edge save is due for a first call at `now`, or null when none applies. */
  private leadingDeadlineFor(params: ScheduleSaveParams, now: number): number | null {
    const lead = params.leadingDelayMs;
    if (lead === undefined || !Number.isFinite(lead)) return null;
    const quiet = this.lastSaveStartedAt === null || now - this.lastSaveStartedAt >= params.delayMs;
    return quiet ? now + Math.max(0, lead) : null;
  }

  /**
   * Start the scheduled save NOW instead of waiting for its timer. The
   * page-lifecycle hook (`pagehide` / `visibilitychange → hidden`): the page
   * may be frozen or killed right after the event, so the write is started
   * synchronously and not awaited. A no-op when nothing is scheduled.
   */
  flushPending(): void {
    if (!this.timer) return;
    clearTimeout(this.timer);
    this.timer = null;
    this.fire();
  }

  /** Run (or, while a save is in flight, defer) the scheduled index write. */
  private fire(): void {
    const params = this.params;
    if (!params) return;
    if (this.inFlight) {
      // Coalesce behind the running write; `finally` below picks it up.
      this.saveQueuedBehindFlight = true;
      return;
    }
    this.params = null;
    this.firstUnsavedAt = null;
    this.leadingDeadline = null;
    this.lastSaveStartedAt = Date.now();
    this.saveQueuedBehindFlight = false;
    this.inFlight = writeSnapshot(params.root, params.getSnapshot())
      .catch((error) => {
        params.onError(error);
      })
      .finally(() => {
        this.inFlight = null;
        if (this.saveQueuedBehindFlight) {
          this.saveQueuedBehindFlight = false;
          // Keep `params` null-safe: a cancelPendingSave() in the meantime
          // cleared it, and then there is nothing left to write.
          if (this.params) this.fire();
        }
      });
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
    this.params = null;
    this.firstUnsavedAt = null;
    this.leadingDeadline = null;
    this.saveQueuedBehindFlight = false;
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
