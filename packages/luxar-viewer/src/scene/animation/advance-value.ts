/**
 * Pure dimension-advance math for playback: one step of the animation
 * (next value + loop/bounce/once boundary handling), extracted from
 * `DimensionAnimationManager` so it can be used BOTH to advance the real
 * playhead (the manager's per-tick update, which then applies the returned
 * direction) and to PEEK the next value without mutating any state (the
 * t+1 slice prefetcher — `DimensionAnimationManager.peekNextValue`).
 *
 * No `state` object goes in and none is mutated: direction changes come back
 * in the result instead (the manager's old `handleBoundary` mutated
 * `state.direction` in place, which made peeking impossible).
 *
 * @module scene/animation/advance-value
 */

import type { LoopMode, AnimationDirection } from '../../types/animation';

/** Inputs for one playback advance step. */
export interface AdvanceArgs {
  /** Current dimension value (the playhead). */
  current: number;
  /** Dimension range minimum. */
  min: number;
  /** Dimension range maximum. */
  max: number;
  /** Step size for discrete dimensions; null = continuous. */
  step: number | null;
  /** Current playback direction. */
  direction: AnimationDirection;
  /** Boundary behavior: once | loop | bounce. */
  loopMode: LoopMode;
  /** Target playback FPS (sizes the continuous increment). */
  targetFPS: number;
  /** Full-range traverse time for continuous dimensions, in ms. */
  continuousTraverseMs: number;
}

/** Result of one playback advance step. */
export interface AdvanceResult {
  /** The next dimension value (boundary-adjusted). */
  value: number;
  /** Direction AFTER the step (flipped by a bounce turnaround). */
  direction: AnimationDirection;
  /** True when loopMode 'once' hit its boundary — playback should stop. */
  shouldStop: boolean;
  /** True when a bounce turnaround flipped the direction this step. */
  directionChanged: boolean;
}

/**
 * Compute the next playback value for a dimension — pure, no state mutation.
 *
 * Discrete dimensions step by `±step`; continuous dimensions advance by
 * `range / traverseTime × frameTime`. Boundaries follow the loop mode:
 * `once` clamps and stops, `loop` wraps to the opposite end, `bounce`
 * clamps and flips the returned direction.
 */
export function advanceDimensionValue(args: AdvanceArgs): AdvanceResult {
  const { current, min, max, step, loopMode, targetFPS, continuousTraverseMs } = args;
  let direction = args.direction;
  const sign = direction === 'forward' ? 1 : -1;

  let value: number;
  if (step !== null) {
    // Discrete dimension: step by integer increments.
    value = current + sign * step;
  } else {
    // Continuous dimension: advance so the full range takes traverseTime.
    const increment = ((max - min) / continuousTraverseMs) * (1000 / targetFPS);
    value = current + sign * increment;
  }

  let shouldStop = false;
  let directionChanged = false;

  if (args.direction === 'forward' && value >= max) {
    switch (loopMode) {
      case 'once':
        value = max;
        shouldStop = true;
        break;
      case 'loop':
        value = min;
        break;
      case 'bounce':
        value = max;
        direction = 'backward';
        directionChanged = true;
        break;
    }
  } else if (args.direction === 'backward' && value <= min) {
    switch (loopMode) {
      case 'once':
        value = min;
        shouldStop = true;
        break;
      case 'loop':
        value = max;
        break;
      case 'bounce':
        value = min;
        direction = 'forward';
        directionChanged = true;
        break;
    }
  }

  return { value, direction, shouldStop, directionChanged };
}
