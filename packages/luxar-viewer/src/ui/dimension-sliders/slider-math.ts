/**
 * Pure math helpers used by dimension-sliders and other range-style UI.
 *
 * These functions are intentionally framework-free — no DOM, no THREE, no
 * scene state. They exist so the value/fraction/wrap logic can be unit
 * tested without a browser environment, and so any UI consumer that needs
 * the same math (range slider, dimension dropdown, etc.) shares one
 * implementation rather than reinventing it.
 *
 * @module ui/dimension-sliders/slider-math
 */

import { clamp } from '../slider-kit';

/**
 * Clamp `value` into `[min, max]`. When `isCyclic` is true, an underflow
 * (value below min) wraps to `max` and an overflow wraps to `min`. When it
 * is false, the value is clamped to the nearest endpoint.
 *
 * The wrap is one-step: this function does not handle multi-period
 * overshoot (e.g. value = max + 5 → max + 5 - period). Callers that need
 * full modular wrap should compose this with a normalization pass.
 *
 * @param value - The candidate value (typically `current ± step`).
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @param isCyclic - When true, out-of-range values wrap to the other end.
 * @returns A value in `[min, max]`.
 */
export function clampWithCyclicWrap(
  value: number,
  min: number,
  max: number,
  isCyclic: boolean
): number {
  if (value < min) return isCyclic ? max : min;
  if (value > max) return isCyclic ? min : max;
  return value;
}

/**
 * Clamp an integer `value` into `[lo, hi]`. Used for the 0–1000 internal
 * range of HTML `<input type=range>` when a slider drives a continuous
 * dimension.
 *
 * @param value - Candidate integer value.
 * @param lo - Inclusive lower bound.
 * @param hi - Inclusive upper bound.
 * @returns A value in `[lo, hi]`.
 */
export function clampInteger(value: number, lo: number, hi: number): number {
  return clamp(value, lo, hi);
}
