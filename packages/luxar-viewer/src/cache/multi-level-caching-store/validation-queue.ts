import { buildUrl, fetchWithRetry } from './fetch-retry';

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
 * Invalidation event-dispatch sites (e.g. "TTL expired", "content-hash
 * mismatch") stay at the caller — this class only serializes, it does
 * NOT emit events.
 */
interface QueueEntry {
  promise: Promise<void>;
  abort: AbortController;
}

export class ValidationQueue {
  private static readonly queues = new Map<string, QueueEntry>();

  /**
   * Run `task` after any previously-queued validation for `datasetId`
   * resolves. `task` receives an `AbortSignal` that fires when
   * {@link ValidationQueue.cancel} is called for the same `datasetId`
   * (typically from the owning instance's `dispose()`).
   *
   * If the abort fires while waiting in line, `task` is skipped
   * entirely (the closure may have captured a now-disposed `this`).
   */
  static async serialize(
    datasetId: string,
    task: (signal: AbortSignal) => Promise<void>
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
   * Cancel any in-flight or queued validation for `datasetId`. Safe to
   * call when no entry exists.
   */
  static cancel(datasetId: string): void {
    const queued = ValidationQueue.queues.get(datasetId);
    if (queued) {
      queued.abort.abort();
      ValidationQueue.queues.delete(datasetId);
    }
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
    const response = await fetchWithRetry(buildUrl(baseUrl, '.zattrs'), {
      timeoutMsOverride: options.timeoutMsOverride,
      signal: options.signal,
    });
    if (!response?.ok) return null;

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
    const digestBuffer = await crypto.subtle.digest('SHA-256', new Uint8Array(data));
    const digest = Array.from(new Uint8Array(digestBuffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
    return { hash: `zattrs:${digest}`, mode: 'zattrs-hash' };
  } catch {
    return null;
  }
}
