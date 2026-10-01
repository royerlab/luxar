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
import { snapDiscreteValue } from '../scene-dims-manager';

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
  /**
   * The dimension's own discrete grid step (its authored `step`), or
   * null/undefined for a continuous dimension. When `max` is off that grid,
   * the last grid point below it is the effective end of the range: the
   * dims manager snaps every write onto the grid, so clamping onto `max`
   * would land back on the current value and freeze playback there.
   */
  gridStep?: number | null;
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

/** The reachable end of the range: `max`, or the last grid point below it. */
function effectiveMax(min: number, max: number, gridStep: number | null | undefined): number {
  return gridStep != null && gridStep > 0 ? snapDiscreteValue(max, gridStep, min, max) : max;
}

/**
 * Loop / once handling for a step that reached or passed the end it is moving
 * toward (`end`, with `start` the opposite end). The endpoint is a FRAME to
 * show: a step that arrives at or overshoots it lands ON it, and only a step
 * taken FROM the endpoint wraps (loop) or completes (once).
 */
function wrapOrStop(
  current: number,
  end: number,
  start: number,
  loopMode: 'once' | 'loop',
  eps: number
): { value: number; shouldStop: boolean } {
  const atEnd = Math.abs(current - end) <= eps;
  if (!atEnd) return { value: end, shouldStop: false };
  return loopMode === 'once'
    ? { value: end, shouldStop: true }
    : { value: start, shouldStop: false };
}

/**
 * Bounce handling for a step that reached or passed the end it moves toward:
 * the playhead lands ON ``end`` — unless it is already there (playback started
 * on the endpoint), in which case it steps the same distance back from it at
 * once, rather than holding the endpoint for a second period (#2944 review B).
 */
function bounceValue(
  current: number,
  stepped: number,
  end: number,
  [min, top]: readonly [number, number],
  eps: number
): number {
  if (Math.abs(current - end) > eps) return end;
  const away = end - (stepped - current);
  return Math.min(top, Math.max(min, away));
}

/**
 * Compute the next playback value for a dimension — pure, no state mutation.
 *
 * Discrete dimensions step by `±step` on the range-min-anchored grid;
 * continuous dimensions advance by `range / traverseTime × frameTime`.
 * Boundaries follow the loop mode. For `loop` and `once` the range ends are
 * shown: a step that reaches or overshoots `max` (forward) or `min`
 * (backward) lands on it, and the NEXT step — taken from the endpoint —
 * wraps to the opposite end (`loop`) or stops (`once`). `bounce` clamps and
 * flips the returned direction on arrival, which already shows the endpoint
 * exactly once; a bounce STARTED on the endpoint it moves toward steps away
 * from it on the first tick.
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

  const top = effectiveMax(min, max, args.gridStep);
  const eps = 1e-9 * Math.max(1, Math.abs(max - min));
  let shouldStop = false;
  let directionChanged = false;

  const forwardHit = args.direction === 'forward' && value >= top - eps;
  const backwardHit = args.direction === 'backward' && value <= min + eps;
  if (loopMode === 'bounce' && (forwardHit || backwardHit)) {
    value = bounceValue(current, value, forwardHit ? top : min, [min, top], eps);
    direction = forwardHit ? 'backward' : 'forward';
    directionChanged = true;
  } else if (loopMode !== 'bounce' && forwardHit) {
    ({ value, shouldStop } = wrapOrStop(current, top, min, loopMode, eps));
  } else if (loopMode !== 'bounce' && backwardHit) {
    ({ value, shouldStop } = wrapOrStop(current, min, top, loopMode, eps));
  }

  return { value, direction, shouldStop, directionChanged };
}
