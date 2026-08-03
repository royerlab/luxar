/**
 * Seedable PRNG helpers for unit tests.
 *
 * Audit ref: C3 (global-pattern-sweep, P7) — tests use `Math.random()`
 * for synthetic data with no seed. Even when the assertions are
 * tolerant (e.g., `toBeDefined()` / count checks), unseeded randomness
 * makes failures hard to reproduce. Use `mulberry32(seed)` to get a
 * deterministic `() => number` substitute.
 *
 * `mulberry32` is the simplest decent-quality 32-bit PRNG in widespread
 * use (https://github.com/bryc/code/blob/master/jshash/PRNGs.md). It is
 * NOT cryptographically secure — but for synthetic-data generation in
 * tests it is sufficient and reproducible across browsers + Node.
 *
 * Pure utility module — no DOM, no vitest globals.
 */

/**
 * Return a deterministic `() => number` (in [0, 1)) seeded by `seed`.
 *
 * Calling the returned function advances internal state, so a single
 * PRNG instance produces a deterministic stream of pseudo-random
 * floats. Construct a fresh PRNG per test (in `beforeEach`) to avoid
 * cross-test coupling.
 *
 * @param seed 32-bit unsigned integer; zero produces a valid stream.
 * @returns Function that returns a deterministic float in [0, 1).
 */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
