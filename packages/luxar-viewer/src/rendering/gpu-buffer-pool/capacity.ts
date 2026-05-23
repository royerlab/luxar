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
 * Choose the capacity to allocate for a buffer that needs to hold
 * `requested` instances. Grows by 1.5× to leave headroom for the
 * next update without a fresh grow, but never goes below
 * `currentMinInstanceCapacity` (see `DEFAULT_MIN_INSTANCE_CAPACITY`).
 */
export function chooseCapacity(requested: number): number {
  return Math.max(currentMinInstanceCapacity, Math.ceil(requested * 1.5));
}
