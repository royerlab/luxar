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
 * @returns Step size for navigation, guaranteed positive and >= 1 for discrete dims
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

  // For discrete dimensions, ensure step is at least 1
  if (meta?.discrete) {
    stepSize = Math.max(1, Math.round(stepSize));
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
 * @param discrete - If true, rounds to nearest integer (for frame indices)
 * @param wrapAround - If true, wraps at boundaries; if false, clamps to range
 * @returns New position after navigation, guaranteed to be within range
 */
export function calculateNextPosition(
  currentPos: number,
  direction: 1 | -1,
  stepSize: number,
  range: [number, number],
  discrete: boolean = false,
  wrapAround: boolean = false
): number {
  let newPos = currentPos + direction * stepSize;

  // Handle discrete dimensions
  if (discrete) {
    newPos = Math.round(newPos);
  }

  // Handle boundaries
  if (wrapAround) {
    const rangeSize = range[1] - range[0];
    if (rangeSize <= 0) {
      return range[0]; // Degenerate or invalid range
    }
    if (newPos < range[0]) {
      newPos = range[1] - ((range[0] - newPos) % rangeSize);
    } else if (newPos > range[1]) {
      newPos = range[0] + ((newPos - range[1]) % rangeSize);
    }
  } else {
    // Clamp to range
    newPos = clamp(newPos, range[0], range[1]);
  }

  return newPos;
}
