/**
 * Cache-residency probe — a per-call sink the L0 cache proxy reports chunk
 * hit/miss outcomes to, so a loader can tell whether a given load was served
 * entirely from cache (resident) or required a fresh fetch/decode (a miss).
 *
 * This replaces the fragile wall-clock heuristic (`elapsed > 15ms ⇒ miss`)
 * that the progressive loaders previously used to decide whether to keep
 * loading the next LOD level in the same frame. Residency is captured at the
 * single chokepoint — `cached-zarr-array.ts`'s `getChunk` interception — so
 * `RangeLoader` and the encoding bodies need no changes.
 *
 * @module cache/residency-probe
 */

/** A sink the cache proxy reports each chunk hit/miss to. */
export interface ResidencyProbe {
  /** Record one chunk access outcome. `hit=true` ⇒ served from cache. */
  record(hit: boolean): void;
}

/**
 * Accumulates chunk hit/miss outcomes for a single load operation.
 *
 * A loader sets a fresh accumulator as the proxy's active probe before a
 * demand load, runs the load, then reads {@link allResident} to decide
 * whether the next LOD level can be loaded in the same frame.
 */
export class ResidencyAccumulator implements ResidencyProbe {
  hits = 0;
  misses = 0;

  record(hit: boolean): void {
    if (hit) this.hits++;
    else this.misses++;
  }

  /**
   * True when no chunk missed the cache. A load that touched **no** chunks
   * (e.g. an empty spatial query, or attributes served by broadcast/array_ref
   * codecs that bypass the proxy) is treated as resident — conservative: it
   * never reports a false miss that would stall progressive refinement.
   */
  get allResident(): boolean {
    return this.misses === 0;
  }

  /** True if at least one chunk access was recorded. */
  get touched(): boolean {
    return this.hits + this.misses > 0;
  }
}
