/**
 * Pure step-size + next-position math for nD navigation. Computes how
 * far to move per [/] key press or slider wheel notch — the modifier
 * tiers (Shift fine ÷10, Ctrl coarse ×10, Ctrl+Shift extra-fine ÷100)
 * only reach here from the slider wheel: the [/] key bindings match the
 * unmodified key, so Shift+[ arrives as '{' and never fires — and where
 * the next position lands, including discrete rounding and wrap-around
 * or clamping at the dimension's bounds.
 *
 * @module scene/dims/step-math
 */

import type { SimpleDims } from '../../types/dims';
import { clamp } from '../../utils/clamp';
import { applyModifierTier } from '../../utils/cross-layer/modifier-tiers';

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
 * - Ctrl+Shift together: Extra-fine control (100x smaller steps)
 *
 * @param dimIndex - Zero-based index of dimension to navigate
 * @param dims - Complete dimension configuration including metadata
 * @param modifiers - Keyboard modifier state for fine/coarse control
 * @param config - Navigation configuration (step multipliers, etc.)
 * @param overrideStep - User-set per-dimension step (the animation menu's
 *          Step override). A finite positive value REPLACES the base
 *          derivation (authored step / 1% of range); modifiers and the
 *          discrete grid quantization still apply on top.
 * @returns Step size for navigation, guaranteed positive and at least one
 *          grid cell (`meta.step`, default 1) for discrete dims
 */
export function calculateStepSize(
  dimIndex: number,
  dims: SimpleDims,
  modifiers: { shift?: boolean; ctrl?: boolean; alt?: boolean } = {},
  config: NavigationConfig = DEFAULT_NAV_CONFIG,
  overrideStep?: number | null
): number {
  const meta = dims.metadata?.[dimIndex];

  // Get base step size
  let stepSize: number;
  if (overrideStep != null && Number.isFinite(overrideStep) && overrideStep > 0) {
    stepSize = overrideStep;
  } else if (meta?.step) {
    stepSize = meta.step;
  } else if (meta?.range) {
    // Calculate step as percentage of range
    const range = meta.range[1] - meta.range[0];
    stepSize = range * 0.01; // 1% of range
  } else {
    stepSize = 1.0; // Default
  }

  // Apply modifiers. Shift and Ctrl do not cancel out when held together:
  // Ctrl+Shift is one more rung in the fine direction (÷100), not a
  // coarse/fine tug-of-war.
  stepSize = applyModifierTier(stepSize, modifiers, {
    fineDivisor: config.fineStepDivisor,
    coarseMultiplier: config.coarseStepMultiplier,
  });

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
 *                   so the cyclic period spans the `range[0] + k·snapStep`
 *                   positions rather than the continuous `max - min`
 * @param wrapAround - If true, wraps at boundaries; if false, clamps to range
 * @param snapStep - Grid the position snaps to for discrete dims (the dim's
 *                   declared step; default 1 = classic integer frame indices).
 *                   Uses the same range-min anchor as
 *                   `SceneDimsManager.setDimensionValue`.
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
    newPos = range[0] + Math.round((newPos - range[0]) / grid) * grid;
    const lastK = Math.floor((range[1] - range[0]) / grid + 1e-9);
    if (lastK < 0) return range[0];
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
      // Wrap over integer indices on the range-min-anchored grid. The epsilon
      // absorbs division error for fractional steps.
      const lastK = Math.floor((range[1] - range[0]) / grid + 1e-9);
      const k = Math.round((newPos - range[0]) / grid);
      if (k < 0 || k > lastK) {
        const positions = lastK + 1;
        // Positive modulo, so arbitrary overshoot in either direction lands on
        // a valid position rather than only the single-cycle case.
        const wrappedK = ((k % positions) + positions) % positions;
        newPos = range[0] + wrappedK * grid;
      }
      // Keep the recomputed grid value byte-identical to the scene manager's
      // snap. It may sit an ulp past an exactly on-grid fractional max; a raw
      // clamp would replace it with a different float and desync cache keys.
      return newPos;
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
    if (discrete) {
      const grid = snapStep > 0 ? snapStep : 1;
      const lastK = Math.floor((range[1] - range[0]) / grid + 1e-9);
      const k = Math.round((newPos - range[0]) / grid);
      newPos = range[0] + clamp(k, 0, lastK) * grid;
    } else {
      newPos = clamp(newPos, range[0], range[1]);
    }
  }

  return newPos;
}
