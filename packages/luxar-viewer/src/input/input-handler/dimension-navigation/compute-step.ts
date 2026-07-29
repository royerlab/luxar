/**
 * Pure dimension-navigation math extracted from `input/input-handler.ts`.
 *
 * Two helpers live here:
 *
 *   - `computeDimensionStep` — given the current dim state, ranges, the
 *     selected-dimension index, and a navigation direction, returns
 *     `{targetDim, newValue, changed}` where `changed` is `true` only
 *     when `newValue` differs from the current value by more than a
 *     step-relative epsilon (`snapStep * 1e-9`, the same relative
 *     tolerance `SceneDimsManager` uses), so a dim whose grid is finer
 *     than a micro-unit is still navigable. The caller is
 *     responsible for actually mutating sceneDimsManager — keeping the
 *     math pure makes the wrap-around / clamping / discrete-rounding
 *     branches unit-testable in isolation.
 *
 *   - `resolveSelectedDimension` — translate a 0-based key index into
 *     the new `selectedDimension` value. Returns `null` when the key
 *     points past the available navigable dimensions; in that case the
 *     caller logs the "not available" hint without mutating any state.
 *
 * @module input/handlers/dimension-navigation
 */

import type { SimpleDims } from '../../../types/dims';
import { getNonDisplayedDimensions, mapKeyToDimension } from './selection';
import { calculateStepSize, calculateNextPosition } from './step-math';

/** Result of `computeDimensionStep` — the next position for the targeted dim. */
export interface DimensionStepResult {
  /** Index into `dims.currentStep` that should be updated. */
  targetDim: number;
  /** The new value to assign at `targetDim`. */
  newValue: number;
  /** `true` iff the new value differs from the current value by more than a step-relative epsilon (snapStep * 1e-9). */
  changed: boolean;
}

/**
 * Compute the next dimension-navigation step given the current dim
 * state. Returns `null` when navigation cannot proceed (no dims, no
 * ranges, no navigable dimensions).
 *
 * The `changed` threshold is a step-relative epsilon (`snapStep * 1e-9`,
 * matching `SceneDimsManager`) so a sub-micro-unit grid is still
 * navigable; the intent is unchanged from the inline original — pure
 * arithmetic identity isn't required, only "movement big enough that
 * the spatial-index loader will requery."
 */
export function computeDimensionStep(
  direction: -1 | 1,
  selectedDimension: number,
  dims: SimpleDims | null | undefined,
  dimensionRanges: ReadonlyArray<readonly [number, number]> | null | undefined
): DimensionStepResult | null {
  if (!dims || !dimensionRanges) return null;

  const navigableDims = getNonDisplayedDimensions(dims);
  if (navigableDims.length === 0) return null;

  // Target the currently selected dimension (bounded by available dimensions)
  const dimIndex = Math.min(selectedDimension, navigableDims.length - 1);
  const targetDim = navigableDims[dimIndex];

  const currentValue = dims.currentStep[targetDim];
  const dimMeta = dims.metadata?.[targetDim];
  const [min, max] = dimensionRanges[targetDim];

  const stepSize = calculateStepSize(targetDim, dims);
  const isCyclic = dimMeta?.cyclic || false;
  // Discrete positions snap to the dim's declared grid (fractional steps
  // included), not to integers — matching SceneDimsManager's own snap.
  const snapStep = dimMeta?.step && dimMeta.step > 0 ? dimMeta.step : 1;
  const newValue = calculateNextPosition(
    currentValue,
    direction,
    stepSize,
    [min, max],
    dimMeta?.discrete,
    isCyclic,
    snapStep
  );
  const eps = snapStep * 1e-9;

  return {
    targetDim,
    newValue,
    changed: Math.abs(newValue - currentValue) > eps,
  };
}

/**
 * Resolve the actual dimension index from a `selectedDimension` slot
 * (a 0-based index into the navigable-dimension list).
 *
 * Returns `-1` for any of the "no resolvable dim" cases the caller
 * needs to short-circuit on:
 *   - `selectedDimension < 0` (sentinel for "nothing selected");
 *   - `dims` is null / undefined (no scene loaded);
 *   - `selectedDimension >= navigableDims.length` (slot now points
 *     past the available navigable dims, e.g. after a scene reload
 *     reduced the dim count).
 *
 * Used by the keyboard bindings that operate on the currently
 * selected dimension (K, Home, End, Shift+↑/↓ animation shortcuts)
 * and by InputHandler's `selectDimension` callsite — sharing the
 * helper here keeps the "what does selectedDimension mean" rule in
 * one place.
 */
export function getSelectedDimensionIndex(
  selectedDimension: number,
  dims: SimpleDims | null | undefined
): number {
  if (selectedDimension < 0) return -1;
  if (!dims) return -1;
  const navigableDims = getNonDisplayedDimensions(dims);
  if (selectedDimension >= navigableDims.length) return -1;
  return navigableDims[selectedDimension];
}

/**
 * Translate a 0-based key index (e.g. 0 for the '1' key, 1 for '2',
 * etc.) into the new `selectedDimension` value, or `null` when the
 * key falls past the available navigable dimensions.
 *
 * Returns the key index unchanged on success — the index is already
 * the navigable position. On failure, returns `null` plus the count of
 * navigable dims so the caller can include it in a "not available"
 * message.
 */
export function resolveSelectedDimension(
  keyIndex: number,
  dims: SimpleDims | null | undefined
): { selectedDimension: number } | { selectedDimension: null; navigableCount: number } {
  if (!dims) return { selectedDimension: null, navigableCount: 0 };

  // mapKeyToDimension uses a 1-based key string; we receive a 0-based index.
  const dimIndex = mapKeyToDimension((keyIndex + 1).toString(), dims);

  if (dimIndex >= 0) {
    return { selectedDimension: keyIndex };
  }

  const navigableDims = getNonDisplayedDimensions(dims);
  return { selectedDimension: null, navigableCount: navigableDims.length };
}
