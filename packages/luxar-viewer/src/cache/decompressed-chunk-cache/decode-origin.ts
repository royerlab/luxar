/**
 * Decode-origin tags for the L0 `decode.count.<origin>` perf counters.
 *
 * The `AbortSignal` is the only per-call channel zarrita forwards into
 * `getChunk` (`get()` passes `{ signal }`), so a speculative path that owns its
 * own controller tags that controller's signal once and every miss decode its
 * reads trigger is attributed to it — safely, even while a foreground update
 * runs concurrently on the same loader. Instrumentation only.
 *
 * Kept dependency-free (no `DecompressedChunkCache` import, whose static
 * initializer reads config) so data-layer helpers can tag signals without
 * pulling the cache module into their import graph.
 *
 * @module cache/decompressed-chunk-cache/decode-origin
 */

const signalOrigins = new WeakMap<AbortSignal, string>();

// lifecycle-exempt: attaches an attribution tag to a signal; starts and owns no work
/**
 * Tag `signal` so decodes triggered by reads carrying it are counted under
 * `decode.count.<origin>`. A signal already tagged keeps its first tag unless
 * `overwrite` is set (a lookahead signal handed to the generic prefetch helper
 * stays 'lookahead').
 */
export function tagSignalOrigin(
  signal: AbortSignal | null | undefined,
  origin: string,
  overwrite = false
): void {
  if (!signal) return;
  if (!overwrite && signalOrigins.has(signal)) return;
  signalOrigins.set(signal, origin);
}

// lifecycle-exempt: looks up a signal's tag; starts and owns no work
/** The origin `signal` was tagged with, if any. */
export function signalOrigin(signal: AbortSignal | null | undefined): string | undefined {
  return signal ? signalOrigins.get(signal) : undefined;
}
