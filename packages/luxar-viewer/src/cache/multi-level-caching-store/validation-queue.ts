import { buildUrl, fetchWithRetry } from './fetch-retry';
import { sha256Hex } from './sha256';

/**
 * Cross-instance validation serializer.
 *
 * Two MultiLevelCachingStore instances pointing at the same URL must
 * serialize their cache-validation runs across instances so a slower
 * older validation cannot overwrite a newer content-hash. The queue
 * is keyed by `datasetId` (SHA-256 hash of the dataset URL, see
 * `hashUrl`).
 *
 * Each entry carries an `AbortController` so the owning instance's
 * `dispose()` can both cancel the in-flight HTTP fetch and remove the
 * queue entry — preventing a closure that captured the disposed
 * instance from running `setContentHash()` against a disposed L2 store.
 *
 * Cancellation is *identity-scoped*: a caller holds the {@link QueueEntry}
 * it received from {@link ValidationQueue.serialize} and passes it back to
 * {@link ValidationQueue.cancel}. This matters because a newer store may
 * have replaced the older store's entry as the map head while the older
 * entry (having captured the now-disposed `this`) is still running — the
 * disposing owner must abort ITS OWN entry, never "whatever is currently
 * the head", or it would kill the newer store's validation and let a stale
 * OPFS cache be served.
 *
 * Invalidation event-dispatch sites (e.g. "TTL expired", "content-hash
 * mismatch") stay at the caller — this class only serializes, it does
 * NOT emit events.
 */
export interface QueueEntry {
  promise: Promise<void>;
  abort: AbortController;
}

export class ValidationQueue {
  private static readonly queues = new Map<string, QueueEntry>();

  /**
   * Run `task` after any previously-queued validation for `datasetId`
   * resolves. `task` receives an `AbortSignal` that fires when
   * {@link ValidationQueue.cancel} is called with THIS call's
   * {@link QueueEntry} (typically from the owning instance's `dispose()`).
   *
   * The entry is created synchronously and handed to the optional
   * `onStart` callback BEFORE the first `await`, so the caller can capture
   * its own handle and later cancel exactly that entry — even after a newer
   * same-`datasetId` validation has replaced it as the map head. The
   * returned promise still resolves only when the validation completes, so
   * `await serialize(...)` keeps its "wait until done" contract.
   *
   * If the abort fires while waiting in line, `task` is skipped
   * entirely (the closure may have captured a now-disposed `this`).
   */
  static async serialize(
    datasetId: string,
    task: (signal: AbortSignal) => Promise<void>,
    onStart?: (entry: QueueEntry) => void
  ): Promise<void> {
    const abort = new AbortController();
    const previous = ValidationQueue.queues.get(datasetId);
    const validation = (previous?.promise ?? Promise.resolve())
      .catch(() => undefined)
      .then(() => {
        if (abort.signal.aborted) return;
        return task(abort.signal);
      });

    const entry: QueueEntry = { promise: validation, abort };
    ValidationQueue.queues.set(datasetId, entry);
    onStart?.(entry);
    try {
      await validation;
    } finally {
      // Only delete if this entry is still the head — a newer validation
      // may have replaced it after we started.
      if (ValidationQueue.queues.get(datasetId) === entry) {
        ValidationQueue.queues.delete(datasetId);
      }
    }
  }

  /**
   * Cancel the specific `entry` the caller owns by firing its `AbortSignal`
   * — nothing more. If the entry is still queued, its chained `then`
   * skips the task body via the `abort.signal.aborted` guard; if it is
   * running, the task observes the signal and unwinds.
   *
   * Identity-scoped on purpose: it acts ONLY on the caller's own entry, so
   * an older store's `dispose()` can never cancel a newer same-`datasetId`
   * store's validation.
   *
   * Deliberately does NOT touch the map. Map cleanup is left entirely to
   * `serialize`'s `finally` head-guard (`queues.get(datasetId) === entry`).
   * Deleting here would be wrong for a WAITING head: an aborted waiting
   * entry still settles when its predecessor settles (keeping successors
   * FIFO-chained via its `promise`), and removing it early would let the
   * next `serialize` see `previous === undefined` and run concurrently with
   * the still-live predecessor. Safe to call more than once and on an
   * already-settled entry.
   */
  static cancel(entry: QueueEntry): void {
    entry.abort.abort();
  }
}

/**
 * Remote validation token for a dataset: either the producer-stamped
 * `content_hash` attr, or — when the dataset lacks one — an implicit
 * token derived from the raw root `.zattrs` bytes (`zattrs-hash` mode).
 */
export interface RemoteValidationToken {
  /** Comparison token. Implicit tokens carry a `zattrs:` prefix so they can
   *  never collide with a producer-stamped content hash. */
  hash: string;
  mode: 'content-hash' | 'zattrs-hash';
}

/**
 * Fetch the dataset's validation token directly from the server, bypassing
 * every cache tier. Used by validation to detect server-side dataset
 * changes. Uses the dedicated `validationTimeoutMs` budget so a flaky
 * network does not block scene loading for the full data-fetch timeout.
 *
 * When the root `.zattrs` carries Luxar's `content_hash` attr, that is the
 * token (`content-hash` mode — strongest guarantee). Otherwise the SHA-256
 * of the raw `.zattrs` bytes serves as an implicit token (`zattrs-hash`
 * mode): every Luxar writer re-stamps a per-save `timestamp` attr and most
 * external producers rewrite root metadata on regeneration, so a dataset
 * replaced in place at the same URL still invalidates instead of being
 * served stale from OPFS forever (the pre-fix behaviour with the default
 * `externalDatasetTtlMs: null`).
 *
 * Returns `null` if the `.zattrs` fetch fails or is non-ok (offline /
 * truly headerless store) — callers then fall back to the TTL path.
 */
export async function getRemoteContentHash(
  baseUrl: string,
  options: { signal?: AbortSignal; timeoutMsOverride?: number }
): Promise<RemoteValidationToken | null> {
  try {
    const fetched = await fetchWithRetry(buildUrl(baseUrl, '.zattrs'), {
      timeoutMsOverride: options.timeoutMsOverride,
      signal: options.signal,
    });
    if (!fetched) return null;

    try {
      const { response } = fetched;
      if (!response.ok) return null;

      const data = await response.arrayBuffer();
      const attrs = JSON.parse(new TextDecoder().decode(data));
      const stamped = attrs?.content_hash;
      if (typeof stamped === 'string' && stamped.length > 0) {
        return { hash: stamped, mode: 'content-hash' };
      }

      // Implicit token: hash the exact bytes served. Any rewrite of the root
      // attrs (Luxar writers always bump `timestamp`) changes the token.
      // Digest a Uint8Array view rather than the raw ArrayBuffer: `instanceof
      // ArrayBuffer` checks fail across realms (jsdom/worker), and a view
      // carries explicit byteOffset/byteLength either way.
      const digest = await sha256Hex(new Uint8Array(data));
      return { hash: `zattrs:${digest}`, mode: 'zattrs-hash' };
    } finally {
      fetched.dispose();
    }
  } catch {
    return null;
  }
}
