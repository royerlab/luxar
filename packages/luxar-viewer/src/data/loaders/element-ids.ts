/**
 * Slot → on-disk element index map composition, shared by the geometry
 * producers (Points projection, GSplats projection, Lines projection).
 *
 * A pick shader reports where an element sits in the buffer uploaded to the
 * GPU — its **storage slot**. The per-element label CSR (`label_offsets` /
 * `label_bytes`) is keyed by the element's **on-disk index**. The two spaces
 * diverge whenever the visible buffer is a proper, reordered, or compacted
 * subset of the on-disk one, and there are three causes across the geometries:
 *
 *  1. **Range loading** — the spatial index yields only the visible ranges
 *     (ascending, disjoint, half-open `[start, end)`) and the loader
 *     concatenates just those, so slot `i` is a position in the CONCATENATED
 *     visible set, not an on-disk index.
 *  2. **Visibility compaction** — a per-element cull (zero effective radius for
 *     Points, hidden-dim attenuation for GSplats) drops elements in place,
 *     renumbering everything after the first removal.
 *  3. **Granularity** (Lines only) — the pick shader reports a visible SEGMENT
 *     slot while line labels are per-VERTEX, so no offset correction alone can
 *     bridge the two spaces.
 *
 * The first two reduce to the same composition: `ranges` + the ascending concat
 * indices that survived the cull ⇒ the on-disk index per slot. That composition
 * lives here so the producers cannot drift. Lines uses this helper for just ONE
 * link of its longer chain — loaded-local vertex → on-disk vertex, i.e. `ranges`
 * with no kept list — and walks the segment→vertex hops itself: its
 * slot → segment-row → vertex sequence must NOT be passed as
 * `keptConcatIndices`, because segment rows are spatially permuted and
 * consecutive segments of a polyline share a vertex, so the sequence is neither
 * monotone nor unique and the strict-ascent guard below would (correctly) reject
 * it. See `data/scene-loader/process/data-processor-lines.ts`.
 *
 * @module data/loaders/element-ids
 */

import { log } from '../../utils/log';

/** Half-open on-disk range `[start, end)` — the shape both `PointRange` and `SplatRange` have. */
export interface ElementIdRange {
  start: number;
  end: number;
}

/**
 * Whether slot IS the on-disk index for these inputs: one range anchored at 0
 * and nothing compacted out.
 *
 * {@link buildElementIdMap} returns `undefined` for FOUR different reasons —
 * this identity, plus three fail-closed bail-outs — and the two meanings are
 * opposite: identity says "the slot is already right", a bail-out says "the
 * slot is WRONG and no map could be built". A consumer that composes maps (the
 * Points additive-ladder concat, #1439) must not read a bail-out as identity,
 * so it re-asks the question here. Exported to keep one source of truth: the
 * fast path below is this same predicate.
 */
export function isIdentityElementIdMap(
  ranges: readonly ElementIdRange[],
  keptConcatIndices: ArrayLike<number> | null
): boolean {
  return keptConcatIndices === null && ranges.length === 1 && ranges[0].start === 0;
}

/**
 * Map visible-buffer slots back to the ON-DISK element indices the per-element
 * label CSR (`label_offsets` / `label_bytes`) is keyed by.
 *
 * @param ranges - The visible on-disk ranges, in ascending order, exactly as
 *                 handed to the projection.
 * @param keptConcatIndices - STRICTLY ascending concatenated-set indices that
 *                            survived the visibility compaction, or `null` when
 *                            no compaction ran. Validated: a non-ascending list
 *                            is rejected rather than mapped to garbage.
 * @param count - Final visible element count; the returned map's length.
 * @param logModule - `Modules.*` value the fail-closed warnings are logged
 *                    under (the calling producer's module).
 * @returns A `count`-long slot → on-disk map, or `undefined` when the identity
 *          holds (slot IS the on-disk index) or the inputs are inconsistent —
 *          in both cases callers fall back to the slot.
 */
export function buildElementIdMap(
  ranges: readonly ElementIdRange[],
  keptConcatIndices: ArrayLike<number> | null,
  count: number,
  logModule: string
): Uint32Array | undefined {
  if (count <= 0) return undefined;

  // Identity fast path: one range anchored at 0 and nothing compacted out ⇒
  // slot === on-disk index. This is the common plain-3D case, and it must not
  // pay for a redundant N-element array.
  if (isIdentityElementIdMap(ranges, keptConcatIndices)) {
    return undefined;
  }

  const concatTotal = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
  const expected = keptConcatIndices === null ? concatTotal : keptConcatIndices.length;
  if (expected !== count) {
    log.warning(
      logModule,
      `Element-ID map skipped: ${expected} source indices for ${count} visible elements. ` +
        'Picking labels fall back to the visible-buffer slot.'
    );
    return undefined;
  }

  const out = new Uint32Array(count);

  if (keptConcatIndices === null) {
    let w = 0;
    for (const r of ranges) {
      for (let g = r.start; g < r.end; g++) out[w++] = g;
    }
    return out;
  }

  // Both the kept list and the ranges are ascending, so one forward cursor
  // over the ranges' prefix sums resolves every kept index in O(n) total —
  // no per-element binary search.
  let k = 0;
  let prefix = 0; // concat index at which ranges[k] begins
  let prevKept = -1;
  for (let i = 0; i < keptConcatIndices.length; i++) {
    const c = keptConcatIndices[i];
    // STRICT ascent is what makes the forward cursor sound: it can only move
    // forward, so a repeated or descending index would compute `c - prefix`
    // with the cursor already past `c` and write a negative value, which wraps
    // to ~4e9 in the Uint32Array — silent garbage. (Real shape this catches: a
    // stale prebuilt `public/wasm/` whose glue predates the kernel's
    // source-index out-param drops it, leaving the caller's buffer all zeros —
    // right length, so the count guard passes, and every slot would otherwise
    // resolve to `ranges[0].start`.) Fail closed like the other guards.
    if (c <= prevKept) {
      log.warning(
        logModule,
        `Element-ID map skipped: kept indices are not strictly ascending (${c} after ${prevKept}). ` +
          'Picking labels fall back to the visible-buffer slot.'
      );
      return undefined;
    }
    prevKept = c;
    while (k < ranges.length && c >= prefix + (ranges[k].end - ranges[k].start)) {
      prefix += ranges[k].end - ranges[k].start;
      k++;
    }
    if (k >= ranges.length) {
      // Malformed input (kept index past the end of the ranges). Bail out
      // rather than writing garbage into the map.
      log.warning(
        logModule,
        `Element-ID map skipped: kept index ${c} lies past the end of the visible ranges. ` +
          'Picking labels fall back to the visible-buffer slot.'
      );
      return undefined;
    }
    out[i] = ranges[k].start + (c - prefix);
  }

  return out;
}
