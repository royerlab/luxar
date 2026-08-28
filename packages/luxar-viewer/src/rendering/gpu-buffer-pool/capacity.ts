/**
 * Pool-capacity sizing primitive.
 *
 * Lives in `gpu-buffer-pool/` (a leaf module) so the per-geometry
 * adapters (`points-adapter.ts`, `lines-adapter.ts`,
 * `gsplats-adapter.ts`) can import it without creating a cycle
 * against the parent `gpu-buffer-pool.ts` barrel.
 *
 * @module rendering/gpu-buffer-pool/capacity
 */

/**
 * Minimum buffer capacity even when very few instances are requested.
 * Keeps tiny scenes from producing degenerate single-instance buffers
 * and ensures the first acquire of any pool always produces a real,
 * non-trivial GPU buffer. 256 is plenty — costs ~16 KB per geometry
 * at the largest stride (gsplats) and is dwarfed by every realistic
 * scene's actual instance count.
 */
const DEFAULT_MIN_INSTANCE_CAPACITY = 256;

let currentMinInstanceCapacity = DEFAULT_MIN_INSTANCE_CAPACITY;

/**
 * Test hook: override the per-acquire minimum capacity floor.
 * Pass `null` to reset to the production default. NOT part of the
 * public API — only for use in unit tests that want to exercise
 * `growXxxGeometry` paths at small instance counts.
 *
 * @internal
 */
export function __setMinInstanceCapacityForTesting(value: number | null): void {
  currentMinInstanceCapacity = value === null ? DEFAULT_MIN_INSTANCE_CAPACITY : value;
}

/**
 * Largest ABSOLUTE headroom, in elements, that {@link chooseCapacity} will add
 * on top of the requested count.
 *
 * Headroom exists to absorb the next update's growth without a fresh
 * allocation, and what it has to absorb is a per-slice count WOBBLE — a few
 * thousand elements as a timelapse advances — not a fixed share of however
 * large the node happens to be. Left purely proportional, the 1.5× factor
 * charges the biggest nodes the most for slack they never use, and it is
 * charged against the layout's real per-element cost: lines carry 6 RGBA32F
 * texels = 96 B/segment (`element-texture-layout.ts::LINE_TEXTURE_LAYOUT`)
 * plus the 8 B/segment `aSortedIndex` pair, so 1.5× is 52 B of slack per
 * segment of the node.
 *
 * Measured on `cosmicflows_laniakea_full` (nine sibling Lines nodes,
 * 11,438,031 segments, no LOD ladder so all of it is resident at once), which
 * failed with "Array buffer allocation failed" against a 2000 MiB budget:
 *
 * ```
 * headroom cap    pool bytes    Basin 1 capacity (1,631,600 segments)
 * unbounded 1.5×    1702 MiB    2,447,400
 * 262,144           1369 MiB    1,893,744
 * sized exactly     1135 MiB    1,631,600
 * ```
 *
 * 262,144 gives back 333 MiB of the 567 MiB overshoot while still leaving a
 * multi-million-element node a quarter-million elements of slack. Nodes below
 * 524,288 elements are unaffected and keep the full 1.5× — including every
 * animated demo node, whose per-timepoint swing is in the thousands.
 *
 * A node that does outgrow its slack is not a failure: `acquireGeometry`
 * takes its existing release + reacquire grow path.
 */
const MAX_CAPACITY_HEADROOM = 262_144;

/**
 * Choose the capacity to allocate for a buffer that needs to hold
 * `requested` instances. Grows by 1.5× to leave headroom for the next update
 * without a fresh grow — capped at `MAX_CAPACITY_HEADROOM` elements of slack,
 * so the factor does not scale into hundreds of wasted MiB on a very large
 * node — but never goes below `currentMinInstanceCapacity` (see
 * `DEFAULT_MIN_INSTANCE_CAPACITY`).
 */
export function chooseCapacity(requested: number): number {
  const headroom = Math.min(Math.ceil(requested * 0.5), MAX_CAPACITY_HEADROOM);
  return Math.max(currentMinInstanceCapacity, requested + headroom);
}
