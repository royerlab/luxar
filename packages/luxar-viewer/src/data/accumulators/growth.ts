/**
 * Capacity growth policy shared by the three data accumulators.
 *
 * @module data/accumulators/growth
 */

/**
 * Capacity to grow to when `needed` elements must fit and `current` is the
 * present capacity. Returns `current` unchanged when it already fits.
 *
 * Amortised 1.5× growth, but never beyond the larger of `needed` and one
 * 1.5× step from `current`. The distinction matters because the accumulators
 * are grown two very different ways:
 *
 * - **Incrementally**, a chunk at a time, where each call needs slightly more
 *   than the last. Multiplying gives the usual amortised-O(1) behaviour and
 *   avoids a reallocation per chunk.
 * - **In one jump**, from the initial capacity straight to a count the loader
 *   already knows up front (`Vertex index range: [0 - 1308986],
 *   span=1308987`). Here the old repeated-multiply loop
 *   (`while (cap < needed) cap = ceil(cap * 1.5)`) landed on a term of the
 *   growth SEQUENCE rather than on the count, overshooting by however far the
 *   count sat past the previous term — up to the full 1.5×, and with no
 *   relation to what the data needs.
 *
 * That overshoot was a real cost, not a rounding detail. On the nine sibling
 * Lines nodes of `cosmicflows_laniakea_full`, growing from the initial 8192:
 *
 * ```
 * vertices    old capacity   overshoot
 * 1,635,679      2,391,485      46.2%
 * 1,308,987      1,594,323      21.8%
 * 1,132,484      1,594,323      40.8%
 * 1,073,811      1,594,323      48.5%
 * …
 * total         14,614,628      27.4%  (≈69 MiB of worker ArrayBuffers)
 * ```
 *
 * Six of the nine landed on the *same* 1,594,323 — the tell that the figure
 * came from the growth sequence and not from the data. Taking the max with
 * `needed` keeps the multiplicative behaviour for the incremental case (where
 * `needed` is only just past `current`) and collapses to exactly `needed` for
 * a single large jump.
 */
export function nextCapacity(current: number, needed: number): number {
  if (needed <= current) return current;
  return Math.max(needed, Math.ceil(current * 1.5));
}
