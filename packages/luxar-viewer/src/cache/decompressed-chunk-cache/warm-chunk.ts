/**
 * Cache-warming entry point for the L0 proxy: `warmChunk`.
 *
 * A warm-up only wants the chunk decoded and resident in L0 (and its bytes in
 * L1/L2). Routing it through zarrita `get()` would allocate and fill a full
 * output selection only to throw it away, and — because `get()` calls the
 * proxy's `getChunk` — would read the owning loader's DEMAND abort signal and
 * record into its DEMAND residency probe. `warmChunk` is the proxy method that
 * does neither: it joins/starts the cache-wide decode with the caller's own
 * signal, no probe, and (by default) origin `'prefetch'`.
 *
 * Kept dependency-free (no `DecompressedChunkCache` import, whose static
 * initializer reads config) so data-layer helpers can call it without pulling
 * the cache module into their import graph; the proxy implements the method.
 *
 * @module cache/decompressed-chunk-cache/warm-chunk
 */

/** Options for a cache warm-up of one chunk. */
export interface WarmChunkOptions {
  /** The warm-up's OWN abort signal (never the demand load's). */
  signal?: AbortSignal;
  /**
   * `decode.count.<origin>` attribution for a miss decode. Defaults to the
   * signal's `tagSignalOrigin` tag, else {@link WARM_CHUNK_DEFAULT_ORIGIN}.
   */
  origin?: string;
}

/** Origin a warm-up's decode is counted under when nothing else names one. */
export const WARM_CHUNK_DEFAULT_ORIGIN = 'prefetch';

/** Name of the proxy method (see `cached-zarr-array.ts`). */
export const WARM_CHUNK_METHOD = 'warmChunk';

/** The shape of an array wrapped by the L0 proxy, as far as warming goes. */
export interface WarmableArray {
  warmChunk(chunkCoords: number[], options?: WarmChunkOptions): Promise<void>;
}

/** Minimal array surface the fallback needs (a raw zarrita array has it). */
interface ChunkReadable {
  getChunk(chunkCoords: number[], options?: { signal?: AbortSignal }): Promise<unknown>;
}

/**
 * Warm one chunk: through the L0 proxy's `warmChunk` when `array` is wrapped
 * (decode + cache, no probe record, no output assembly), else a bare
 * `getChunk` that fetches (warming L1/L2) and decodes with no output assembly.
 */
export async function warmChunk(
  array: unknown,
  chunkCoords: number[],
  options: WarmChunkOptions = {}
): Promise<void> {
  const warm = (array as Partial<WarmableArray>)[WARM_CHUNK_METHOD];
  if (typeof warm === 'function') {
    await warm(chunkCoords, options);
    return;
  }
  options.signal?.throwIfAborted();
  await (array as ChunkReadable).getChunk(chunkCoords, { signal: options.signal });
}
