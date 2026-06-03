/**
 * Time threshold (ms) for considering a LOD load "fast" (likely a cache hit).
 * If a LOD loads faster than this, the loader continues to the next LOD.
 * If slower, it stops and lets the refinement loop pick up the rest.
 *
 * Shared across all progressive loaders (Points, Lines, GSplats) for
 * cross-type symmetry — a single generic streaming threshold, not a
 * per-geometry tuning knob.
 */
export const CACHE_HIT_THRESHOLD_MS = 15;
