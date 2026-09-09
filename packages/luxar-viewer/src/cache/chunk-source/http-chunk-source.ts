/**
 * The directory-store byte source: one HTTP request per chunk.
 *
 * This is the behaviour `MultiLevelCachingStore` had inline before the
 * {@link ChunkSource} seam existed, moved verbatim rather than rewritten —
 * same `buildUrl`, same `fetchWithRetry`, same retry budget, same
 * body-cancellation on a response we decline to read. The store's ~100
 * existing tests drive it through `global.fetch` and none of them changed.
 *
 * The one thing that genuinely moved is ownership of the `Response` lifetime.
 * The store used to hold a `FetchResponseScope` and `dispose()` it in a
 * `finally`; now that lives here, because a non-HTTP source has no response to
 * dispose and the seam must not mention one.
 *
 * @module cache/chunk-source/http-chunk-source
 */

import type { ChunkFetchOutcome, ChunkSource } from '../chunk-source';
import { buildUrl, fetchWithRetry } from '../multi-level-caching-store/fetch-retry';
import {
  getRemoteContentHash,
  type RemoteValidationToken,
} from '../multi-level-caching-store/validation-queue';

/** Reads chunks from a directory-backed zarr store over HTTP. */
export class HttpChunkSource implements ChunkSource {
  constructor(private readonly baseUrl: string) {}

  /**
   * The dataset URL verbatim.
   *
   * Deliberately unchanged from what `hashUrl(baseUrl)` hashed before this
   * refactor, so existing OPFS buckets keep resolving — normalizing it here
   * would silently rotate every user's on-disk cache for no gain.
   */
  get identity(): string {
    return this.baseUrl;
  }

  get describe(): string {
    return this.baseUrl;
  }

  async get(key: string, signal?: AbortSignal): Promise<ChunkFetchOutcome> {
    let scope;
    try {
      scope = await fetchWithRetry(buildUrl(this.baseUrl, key), { signal });
    } catch (error) {
      // Belt and braces: `fetchWithRetry` catches inside its own retry loop and
      // returns undefined rather than rejecting, so this is not reachable
      // today. Kept because the seam's contract is that `get` never throws, and
      // that must not depend on a helper's internals.
      return { kind: 'error', cause: error instanceof Error ? error : new Error(String(error)) };
    }

    if (!scope) {
      // `fetchWithRetry` returns undefined when it gave up: either the caller
      // aborted, or the retry budget ran out. The store distinguishes those by
      // inspecting its own signals, so report the retry case and let it
      // re-classify an abort it knows about.
      return signal?.aborted
        ? { kind: 'aborted' }
        : { kind: 'error', cause: new Error(`fetch exhausted retries for ${key}`) };
    }

    try {
      if (scope.response.status === 404) return { kind: 'missing' };
      if (!scope.response.ok) {
        const statusText = scope.response.statusText ? ` ${scope.response.statusText}` : '';
        return {
          kind: 'error',
          cause: new Error(`HTTP ${scope.response.status}${statusText} fetching ${key}`),
        };
      }

      // Check BEFORE touching the body. The store used to do this between the
      // headers arriving and `arrayBuffer()`, and losing it meant an
      // invalidation abort landing in that window surfaced as a NetworkError —
      // which the store logs and turns into `undefined`, i.e. zarrita decodes
      // the chunk as FILL VALUES. Silently wrong geometry is exactly what
      // `abortPendingGets` exists to prevent. The `finally` still cancels the
      // body we are no longer going to read.
      if (signal?.aborted) return { kind: 'aborted' };

      let data: Uint8Array;
      try {
        data = new Uint8Array(await scope.response.arrayBuffer());
      } catch (error) {
        // An abort during the body read rejects here. This must come back as an
        // outcome, not a throw: the seam's contract is that a source never
        // throws, and a throw would escape as NetworkError → fill values.
        if (signal?.aborted) return { kind: 'aborted' };
        return { kind: 'error', cause: error instanceof Error ? error : new Error(String(error)) };
      }

      // Over HTTP the decoded body IS what the meter counted before this
      // refactor; keeping them equal preserves the existing byte totals.
      return { kind: 'ok', data, bytesOverWire: data.byteLength };
    } finally {
      scope.dispose();
    }
  }

  probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null> {
    return getRemoteContentHash(this.baseUrl, options);
  }

  dispose(): void {
    // Nothing held open: each `get` disposes its own response scope.
  }
}
