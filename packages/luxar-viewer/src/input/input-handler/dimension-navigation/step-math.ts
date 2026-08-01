/**
 * Pure step-size + next-position math for nD navigation. Computes how
 * far to move per [/] key press (with Shift/Ctrl fine/coarse modifiers)
 * and where the next position lands, including discrete rounding and
 * wrap-around or clamping at the dimension's bounds.
 *
 * @module input/input-handler/dimension-navigation/step-math
 */

import type { SimpleDims } from '../../../types/dims';
import { clamp } from '../../../utils/clamp';

/** Keyboard navigation configuration */
export interface NavigationConfig {
  stepSizeMultiplier: number;
  fineStepDivisor: number;
  coarseStepMultiplier: number;
  wrapAround: boolean;
}

/** Default navigation configuration */
export const DEFAULT_NAV_CONFIG: NavigationConfig = {
  stepSizeMultiplier: 1.0,
  fineStepDivisor: 10,
  coarseStepMultiplier: 10,
  wrapAround: false,
};

/**
 * Calculate adaptive step size for dimension navigation.
 *
 * Computes the appropriate step size based on dimension metadata, keyboard
 * modifiers, and navigation configuration. Step sizes adapt to:
 * - Discrete dimensions (frames): Step by 1 or more whole units
 * - Continuous dimensions (time): Step by 1% of range by default
 * - Shift modifier: Fine control (10x smaller steps)
 * - Ctrl modifier: Coarse control (10x larger steps)
 *
 * @param dimIndex - Zero-based index of dimension to navigate
 * @param dims - Complete dimension configuration including metadata
 * @param modifiers - Keyboard modifier state for fine/coarse control
 * @param config - Navigation configuration (step multipliers, etc.)
 * @returns Step size for navigation, guaranteed positive and at least one
 *          grid cell (`meta.step`, default 1) for discrete dims
 */
export function calculateStepSize(
  dimIndex: number,
  dims: SimpleDims,
  modifiers: { shift?: boolean; ctrl?: boolean; alt?: boolean } = {},
  config: NavigationConfig = DEFAULT_NAV_CONFIG
): number {
  const meta = dims.metadata?.[dimIndex];

  // Get base step size
  let stepSize: number;
  if (meta?.step) {
    stepSize = meta.step;
  } else if (meta?.range) {
    // Calculate step as percentage of range
    const range = meta.range[1] - meta.range[0];
    stepSize = range * 0.01; // 1% of range
  } else {
    stepSize = 1.0; // Default
  }

  // Apply modifiers
  if (modifiers.shift) {
    stepSize /= config.fineStepDivisor; // Fine control
  } else if (modifiers.ctrl) {
    stepSize *= config.coarseStepMultiplier; // Coarse control
  }

  // Apply global multiplier
  stepSize *= config.stepSizeMultiplier;

  // For discrete dimensions, quantize to the dim's declared grid and
  // never step below one grid cell. Classic frame-index dims (step 1 or
  // unset) keep the historical `max(1, round(stepSize))`; dims with a
  // fractional declared step (e.g. a 4th spatial axis sampled every
  // 0.04 units) step cell-by-cell instead of collapsing to whole units.
  if (meta?.discrete) {
    const gridStep = meta.step && meta.step > 0 ? meta.step : 1;
    stepSize = Math.max(gridStep, Math.round(stepSize / gridStep) * gridStep);
  }

  return stepSize;
}

/**
 * Calculate the next position in a dimension after applying navigation step.
 *
 * Handles dimension navigation with support for:
 * - Discrete (frame-based) and continuous (time-based) dimensions
 * - Boundary clamping or wrap-around behavior
 * - Rounding for discrete dimensions
 * - Min/max range enforcement
 *
 * @param currentPos - Current position value in dimension coordinates
 * @param direction - Navigation direction: 1 for forward (]), -1 for backward ([)
 * @param stepSize - Step size to apply (from calculateStepSize)
 * @param range - Valid [min, max] bounds for this dimension
 * @param discrete - If true, rounds to the declared grid (`snapStep`) and
 *                   treats `range` as an inclusive set of on-grid positions,
 *                   so the cyclic period spans the `k·snapStep` multiples
 *                   inside the range rather than the continuous `max - min`
 * @param wrapAround - If true, wraps at boundaries; if false, clamps to range
 * @param snapStep - Grid the position snaps to for discrete dims (the dim's
 *                   declared step; default 1 = classic integer frame indices).
 *                   Uses `round(pos/snapStep)*snapStep`, the same expression
 *                   as `SceneDimsManager.setDimensionValue`, so the result is
 *                   bit-identical to the manager's own snap.
 * @returns New position after navigation, guaranteed to be within range
 */
export function calculateNextPosition(
  currentPos: number,
  direction: 1 | -1,
  stepSize: number,
  range: [number, number],
  discrete: boolean = false,
  wrapAround: boolean = false,
  snapStep: number = 1
): number {
  let newPos = currentPos + direction * stepSize;

  // Handle discrete dimensions: snap to the declared grid
  if (discrete) {
    const grid = snapStep > 0 ? snapStep : 1;
    newPos = Math.round(newPos / grid) * grid;
  }

  // Handle boundaries
  if (wrapAround) {
    // The cyclic PERIOD is not the same as the range width for a discrete dim.
    // `[min, max]` is INCLUSIVE, so a discrete axis holds
    // `(max - min) / step + 1` distinct positions and its period is
    // `(max - min) + step`. A continuous cyclic axis (an angle in [0, 360])
    // really does have period `max - min`, because there `max ≡ min`.
    //
    // Using the continuous period for a discrete dim skips one position at
    // every wrap: an 8-category dim declared `[0, 7]` (exactly what
    // `Dimension.__post_init__` derives for `categories=[...8...]`) wrapped
    // 0 -> 6 backward and 7 -> 1 forward, silently making the first and last
    // categories unreachable by keyboard.
    if (discrete) {
      const grid = snapStep > 0 ? snapStep : 1;
      // Wrap over the ON-GRID positions — the `k·grid` multiples inside the
      // range. The snap above (and SceneDimsManager's own setDimensionValue /
      // defaultPosition) rounds to the ZERO-anchored grid, so the reachable
      // positions of `[0.15, 0.75]` with step 0.2 are 0.2, 0.4, 0.6 —
      // `range[0]` itself is not one of them (defaultPosition explicitly
      // supports an off-grid min: 1.3 with step 1 starts at 2). Anchoring the
      // period at `range[0]` instead wrapped a forward step off 0.6 to 0.8,
      // which the clamp turned into the off-grid 0.75. The count/wrap
      // arithmetic runs in integer grid indices, which are exact — no float
      // modulo to drift an ulp off the grid. The epsilon absorbs float error
      // in the index division — `0.7 / 0.1` is 6.999999999999999, which would
      // otherwise floor to 6 and make the position at 0.7 unreachable.
      const firstK = Math.ceil(range[0] / grid - 1e-9);
      const lastK = Math.floor(range[1] / grid + 1e-9);
      if (lastK < firstK) {
        // No on-grid position inside the range at all (degenerate, inverted,
        // or narrower-than-one-step range) — nothing to wrap onto.
        return range[0];
      }
      const k = Math.round(newPos / grid); // exact: newPos was snapped above
      if (k < firstK || k > lastK) {
        const positions = lastK - firstK + 1;
        // Positive modulo, so arbitrary overshoot in either direction lands on
        // a valid position rather than only the single-cycle case.
        const wrappedK = firstK + ((((k - firstK) % positions) + positions) % positions);
        newPos = wrappedK * grid;
      }
      // Defence in depth. Every wrapped index is inside [firstK, lastK], so no
      // current caller can trigger this clamp — it exists so the documented
      // range guarantee stays true if the snapping above is ever changed or
      // bypassed.
      return clamp(newPos, range[0], range[1]);
    } else {
      // Continuous path left EXACTLY as it was: `max ≡ min` here, and the
      // one-full-cycle-backward case is specified to return `max`, not `min`.
      const rangeSize = range[1] - range[0];
      if (rangeSize <= 0) {
        return range[0]; // Degenerate or invalid range
      }
      if (newPos < range[0]) {
        newPos = range[1] - ((range[0] - newPos) % rangeSize);
      } else if (newPos > range[1]) {
        newPos = range[0] + ((newPos - range[1]) % rangeSize);
      }
    }
  } else {
    // Clamp to range
    newPos = clamp(newPos, range[0], range[1]);
  }

  return newPos;
}
