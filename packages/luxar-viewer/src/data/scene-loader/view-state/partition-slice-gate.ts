/**
 * Hidden-dimension slice gate for `kind=partition` parts (B4).
 *
 * A partition part whose stored bounds miss the current slice on a DISCRETE
 * hidden dimension draws nothing: every renderer gates a discrete hidden
 * dimension by MEMBERSHIP on the element's own coordinate, never by an extent.
 *
 * - gsplats: centre within `0.5 × step` (`workers/data-worker/projection/gsplats.ts`);
 * - points: value within an absolute `0.5` (`data/points/effective-radius-calculator.ts`,
 *   non-`spatial` dims), after a chunk fetch whose reach is at most
 *   `0.25 × step` beyond a write-side pad of at most `0.5 × step`;
 * - lines: the half-cell clipping slab (segments are bounded by their vertices,
 *   which the part bounds contain);
 * - mesh: the half-cell barrier membership slab.
 *
 * So a part is provably empty for the view when, on some such dimension, its
 * bounds lie farther than {@link partitionSliceMargin} from the slice position
 * — `max(0.5, 0.75 × step)` with a relative float32 slop, which covers every
 * gate above with room to spare and still excludes the on-grid neighbours one
 * step away. Anything the rule cannot vouch for is kept (never gated):
 * continuous dimensions (a gsplat's continuous reach is its own σ, which the
 * node attrs do not carry), `spatial` dimensions (points radius slicing), an
 * `extend_to_all` dimension, a dimension without metadata, and — decided by the
 * caller — any part under an `nd_transform`, whose bounds live in a different
 * space from the world slice.
 *
 * Pure and allocation-free: the registry calls it per part per frame.
 *
 * @module data/scene-loader/view-state/partition-slice-gate
 */

import type { DimensionMetadata } from '../../../types/dims';

/** The view fields the gate reads (a structural subset of the loader's `ViewState`). */
export interface PartitionSliceView {
  displayDims: readonly number[];
  slicePosition: readonly number[];
  tolerance: readonly number[];
  dimensions?: readonly DimensionMetadata[];
}

/** nD part bounds, as stored in `position_bounds`. */
export interface PartitionSliceBounds {
  min: readonly number[];
  max: readonly number[];
}

/** Fraction of a step beyond which no renderer can draw a part's element. */
const MARGIN_STEPS = 0.75;
/** Absolute floor: the points membership gate is `0.5`, whatever the step. */
const MIN_MARGIN = 0.5;
/** Relative slop for float32-stored coordinates against a float64 slice. */
const RELATIVE_SLOP = 1e-6;
/** The extend-to-all tolerance sentinel (`>= 1e9` ⇒ always visible). */
const EXTEND_SENTINEL = 1e9;

/** Distance from the slice beyond which a part's element cannot be drawn. */
export function partitionSliceMargin(step: number | undefined, slice: number): number {
  const cell = step !== undefined && step > 0 ? step : 1;
  const margin = Math.max(MIN_MARGIN, MARGIN_STEPS * cell);
  return margin + RELATIVE_SLOP * Math.max(1, Math.abs(slice), margin);
}

/** Whether hidden dimension `d` is one the gate may test (see the module doc). */
function isGatedDim(view: PartitionSliceView, d: number, extendDims: readonly string[]): boolean {
  if (view.displayDims.includes(d)) return false;
  const dim = view.dimensions?.[d];
  if (dim?.discrete !== true || dim.spatial === true) return false;
  if (!((view.tolerance[d] ?? 0) < EXTEND_SENTINEL)) return false;
  return !extendDims.includes(dim.name);
}

/**
 * Whether `bounds` can contribute a drawn element for `view` — `false` only when
 * the part is provably empty on some discrete hidden dimension.
 */
export function partBoundsIntersectSlice(
  bounds: PartitionSliceBounds,
  view: PartitionSliceView,
  extendDims: readonly string[] = []
): boolean {
  const ndim = Math.min(bounds.min.length, bounds.max.length, view.slicePosition.length);
  for (let d = 0; d < ndim; d++) {
    if (!isGatedDim(view, d, extendDims)) continue;
    const slice = view.slicePosition[d];
    const margin = partitionSliceMargin(view.dimensions?.[d]?.step, slice);
    if (bounds.max[d] < slice - margin || bounds.min[d] > slice + margin) return false;
  }
  return true;
}
