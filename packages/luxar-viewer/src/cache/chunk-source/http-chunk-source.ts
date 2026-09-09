/**
 * The directory-store byte source: one HTTP request per chunk.
 *
 * `fetchWithRetry` owns the full response lifetime: this source inspects the
 * status and consumes successful bodies inside its shared fetch-gate lease.
 * Returning without reading cancels the body before another request starts.
 *
 * @module cache/chunk-source/http-chunk-source
 */

import type { ChunkFetchOutcome, ChunkSource } from '../chunk-source';
import { buildUrl, fetchWithRetry } from '../multi-level-caching-store/fetch-retry';
import {
  getRemoteContentHash,
  type RemoteValidationToken,
} from '../multi-level-caching-store/validation-queue';
import { log, Modules } from '../../utils/log';

const ZARR_METADATA_KEYS = new Set(['zarr.json', '.zarray', '.zattrs', '.zgroup', '.zmetadata']);

function isZarrMetadataKey(key: string): boolean {
  return ZARR_METADATA_KEYS.has(key.slice(key.lastIndexOf('/') + 1));
}

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
    let exhaustedCause: unknown;
    try {
      const outcome = await fetchWithRetry(
        buildUrl(this.baseUrl, key),
        { signal, onExhausted: (error) => (exhaustedCause = error) },
        async (attempt) => {
          if (attempt.response.status === 404) return { kind: 'missing' } as const;
          if (
            (attempt.response.status === 403 || attempt.response.status === 410) &&
            isZarrMetadataKey(key)
          ) {
            log.warning(
              Modules.CACHE,
              `HTTP ${attempt.response.status} probing optional zarr metadata ${key}; treating as missing`
            );
            return { kind: 'missing' } as const;
          }
          if (!attempt.response.ok) {
            const statusText = attempt.response.statusText ? ` ${attempt.response.statusText}` : '';
            return {
              kind: 'error',
              cause: new Error(`HTTP ${attempt.response.status}${statusText} fetching ${key}`),
            } as const;
          }

          // Check BEFORE touching the body. The store used to do this between the
          // headers arriving and `arrayBuffer()`, and losing it meant an
          // invalidation abort landing in that window surfaced as a NetworkError —
          // which is retryable rather than a deliberate cancellation. Silently
          // trusting bytes invalidated by `abortPendingGets` is exactly what this
          // check prevents. The retry helper still cancels the unread body.
          if (signal?.aborted) return { kind: 'aborted' } as const;
          const data = await attempt.readBody();

          // Over HTTP the decoded body IS what the meter counted before this
          // refactor; keeping them equal preserves the existing byte totals.
          return { kind: 'ok', data, bytesOverWire: data.byteLength } as const;
        }
      );

      // `fetchWithRetry` returns undefined when it gave up: either the caller
      // aborted, or the retry budget ran out. The store distinguishes those by
      // inspecting its own signals, so report the retry case and let it
      // re-classify an abort it knows about.
      return (
        outcome ??
        (signal?.aborted
          ? { kind: 'aborted' }
          : {
              kind: 'error',
              cause: new Error(
                `fetch exhausted retries for ${key}${
                  exhaustedCause instanceof Error ? `: ${exhaustedCause.message}` : ''
                }`,
                { cause: exhaustedCause }
              ),
            })
      );
    } catch (error) {
      // Belt and braces: consumer failures are not expected from this callback
      // today. Keep the catch because the seam's contract is that `get` never
      // throws, and that must not depend on a helper's internals.
      return { kind: 'error', cause: error instanceof Error ? error : new Error(String(error)) };
    }
  }

  probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null> {
    return getRemoteContentHash(this.baseUrl, options);
  }

  dispose(): void {
    // Nothing held open: each `get` settles its response before returning.
  }
}
