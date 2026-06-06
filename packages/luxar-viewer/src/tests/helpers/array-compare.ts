/**
 * Shared array-comparison helpers for the WASM test suite.
 *
 * [wasm.md/O2, O13] Previously defined in two places:
 *   - `tests/unit/wasm/typescript-reference.test.ts:1676-1714` (canonical)
 *   - `tests/unit/wasm/wasm-vs-typescript.test.ts:46-66` (variant with
 *      slightly different default epsilon + console.log diagnostic)
 *
 * Consolidated here so the helpers live with the other test-only
 * infrastructure (test-data builders, fixtures, etc.) and the two
 * call sites import from one canonical implementation.
 *
 * Pure utility module — no DOM, no vitest globals.
 */

/**
 * Compare two arrays for exact equality (or within epsilon when set).
 *
 * `epsilon = 0` (default) does true equality on each element via `Math.abs`.
 * Pass a positive epsilon for tolerance-based comparison.
 *
 * NaN/Inf semantics (load-bearing for WASM-vs-TS parity): a naive
 * `Math.abs(a - b) > epsilon` check is NaN-BLIND — when either side is NaN the
 * subtraction is NaN and `NaN > epsilon` is `false`, so a backend that emits NaN
 * while the other emits a finite value would be reported EQUAL, silently hiding a
 * real divergence. We therefore:
 *   - treat NaN-vs-finite (exactly one side NaN) as a MISMATCH,
 *   - treat NaN-vs-NaN as equal,
 *   - let the existing magnitude check handle ±Inf correctly: same-sign Inf
 *     yields `Inf - Inf = NaN` (not `> epsilon`) → equal, while `+Inf` vs `-Inf`
 *     or Inf vs finite yields `Inf > epsilon` → mismatch.
 */
export function arraysEqual(a: ArrayLike<number>, b: ArrayLike<number>, epsilon = 0): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    const bi = b[i];
    const aNaN = Number.isNaN(ai);
    const bNaN = Number.isNaN(bi);
    if (aNaN || bNaN) {
      if (aNaN !== bNaN) return false; // one NaN, one finite/Inf → genuine divergence
      continue; // both NaN → equal
    }
    if (Math.abs(ai - bi) > epsilon) return false;
  }
  return true;
}

/**
 * Compare two float arrays within a tolerance.
 *
 * NOTE [wasm.md C3]: the default `epsilon = 1e-6` is appropriate for typical
 * Float32 single-step operations. For multi-step algorithms that accumulate
 * rounding error per dimension (e.g. `mahalanobis_distance` does ndim
 * forward-substitution steps), call sites should pass a scaled epsilon —
 * e.g. `arraysAlmostEqual(a, b, 1e-5 * Math.sqrt(ndim))` for ndim > 3 —
 * to avoid silently missing WASM-vs-TS divergence at high dimensions.
 */
export function arraysAlmostEqual(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  epsilon = 1e-6
): boolean {
  return arraysEqual(a, b, epsilon);
}

/**
 * Generate random test data for stress testing.
 *
 * Float32 positions in `[-scale/2, +scale/2]` per axis; radii in `[0, 2]`.
 * Unseeded — callers that need determinism should `vi.spyOn(Math, 'random')`.
 */
export function generateRandomPoints(
  numPoints: number,
  ndim: number,
  scale = 10
): { positions: Float32Array; radii: Float32Array } {
  const positions = new Float32Array(numPoints * ndim);
  const radii = new Float32Array(numPoints);

  for (let i = 0; i < numPoints * ndim; i++) {
    positions[i] = (Math.random() - 0.5) * scale;
  }
  for (let i = 0; i < numPoints; i++) {
    radii[i] = Math.random() * 2;
  }

  return { positions, radii };
}
