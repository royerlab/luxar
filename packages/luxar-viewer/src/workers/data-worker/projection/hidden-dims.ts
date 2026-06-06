/**
 * Hidden-dimension classification shared by the GSplats projection
 * dispatcher (and its golden-equivalence tests).
 *
 * nD → 3D projection splits the non-displayed ("hidden") dimensions
 * into three disjoint groups:
 *
 *   - **extend_to_all** — skipped entirely; the splat is visible across
 *     every value of the dimension (tolerance ≥ {@link EXTEND_TO_ALL_THRESHOLD}).
 *   - **discrete** — binary half-step visibility (time, channel, …).
 *   - **continuous** — shifted-Gaussian Mahalanobis attenuation.
 *
 * This split was previously inlined in both the worker dispatcher and
 * the (now-deleted) main-thread copy. Extracting it keeps the dispatcher
 * lean and makes the classification independently unit-testable.
 *
 * @module workers/data-worker/projection/hidden-dims
 */

import { EXTEND_TO_ALL_THRESHOLD } from './constants';

/** Disjoint partition of the hidden (non-displayed) dimensions. */
export interface HiddenDimClassification {
  /** Display dims in the caller's requested order (axis permutation preserved). */
  orderedDisplayDims: number[];
  /** All hidden dims, sorted ascending (the order the WASM kernels expect). */
  sortedHiddenDims: number[];
  /** Hidden dims using continuous Gaussian attenuation, sorted ascending. */
  continuousHiddenDims: number[];
  /** Hidden dims using binary half-step visibility, sorted ascending. */
  discreteHiddenDims: number[];
}

/**
 * Partition `[0, ndim)` into display dims and the three hidden-dim
 * groups, given the pre-resolved discrete and extend_to_all dim sets
 * (the data-processor derives these from the scene's dimension
 * metadata before dispatch).
 *
 * displayDims order is preserved so axis-permuted views (e.g.
 * `[2, 0, 1]`) map output XYZ correctly; hidden dims are sorted
 * ascending for the marginal-Cholesky / Mahalanobis kernels.
 */
export function classifyHiddenDims(
  ndim: number,
  displayDims: readonly number[],
  discreteDims: ReadonlySet<number>,
  extendToAllDims: ReadonlySet<number>
): HiddenDimClassification {
  const orderedDisplayDims = [...displayDims];

  const sortedHiddenDims: number[] = [];
  for (let d = 0; d < ndim; d++) {
    if (!displayDims.includes(d)) sortedHiddenDims.push(d);
  }
  sortedHiddenDims.sort((a, b) => a - b);

  const continuousHiddenDims: number[] = [];
  const discreteHiddenDims: number[] = [];
  for (const dim of sortedHiddenDims) {
    if (extendToAllDims.has(dim)) continue; // always visible — skip
    if (discreteDims.has(dim)) {
      discreteHiddenDims.push(dim);
    } else {
      continuousHiddenDims.push(dim);
    }
  }

  return { orderedDisplayDims, sortedHiddenDims, continuousHiddenDims, discreteHiddenDims };
}

/**
 * True iff a per-dimension tolerance marks an `extend_to_all` dimension.
 * The finite guard rejects NaN/undefined so a malformed tolerance can't
 * accidentally satisfy the `>=` test.
 */
export function isExtendToAll(tolerance: number): boolean {
  return Number.isFinite(tolerance) && tolerance >= EXTEND_TO_ALL_THRESHOLD;
}
