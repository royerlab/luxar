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
          if (!attempt.response.ok) return { kind: 'missing' } as const;
          if (signal?.aborted) return { kind: 'aborted' } as const;
          const data = await attempt.readBody();
          return { kind: 'ok', data, bytesOverWire: data.byteLength } as const;
        }
      );
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
