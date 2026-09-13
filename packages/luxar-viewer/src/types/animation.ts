/**
 * Dimension animation type definitions
 *
 * This file defines types for dimension animation state management.
 * Supports FPS-based animation through dimension ranges with various loop modes.
 */

/**
 * Loop behavior at dimension boundaries
 */
export type LoopMode = 'once' | 'loop' | 'bounce';

/**
 * Animation direction
 */
export type AnimationDirection = 'forward' | 'backward';

/**
 * Animation state for a single dimension
 */
export interface DimensionAnimationState {
  /** Whether animation is currently playing */
  isPlaying: boolean;
  /** Target FPS for animation */
  targetFPS: number;
  /** Loop behavior (once, loop, bounce) */
  loopMode: LoopMode;
  /** Current animation direction */
  direction: AnimationDirection;
  /** Last update timestamp (ms since epoch) */
  lastUpdateTime: number;
  /** Frame counter for FPS measurement */
  frameCount: number;
  /** Time of last FPS measurement */
  lastFPSMeasurementTime: number;
  /** Measured actual FPS (may differ from target) */
  actualFPS: number;
  /**
   * Playback "detail": how deep every frame's additive ladder is loaded while
   * this dimension plays (and scrubs). A number pins that many rungs
   * (`Infinity` = the whole ladder); `'auto'` pins each ladder at the first rung
   * whose cumulative energy reaches `config.dimensionAnimation.playback.autoEnergyThreshold`
   * (an unstamped ladder stays time-budgeted); `null` streams whatever is
   * resident within the tick ("Fast"). Threaded to the progressive loaders as
   * `ViewState.ladderDepth`.
   */
  ladderDepth: number | 'auto' | null;
  /**
   * Explicit per-tick step size in dimension units; null = Auto (continuous
   * dims derive the increment from fps + the range-traversal time, discrete
   * dims use their authored step). Also consumed by the [ / ] keyboard
   * navigation. Deliberately NOT consumed by the slider wheel/drag — hand
   * stepping stays on the dimension's own base step.
   */
  stepSize: number | null;
}

/**
 * Animation events emitted by DimensionAnimationManager
 */
export interface DimensionAnimationEvents {
  /** Emitted when animation starts */
  play: { dimIndex: number };
  /** Emitted when animation pauses */
  pause: { dimIndex: number };
  /** Emitted when animation completes (loop mode: once) */
  complete: { dimIndex: number };
  /** Emitted when target FPS changes */
  speedChange: { dimIndex: number; fps: number };
  /** Emitted when loop mode changes */
  loopModeChange: { dimIndex: number; loopMode: LoopMode };
  /** Emitted when the per-dimension step override changes (null = Auto) */
  stepChange: { dimIndex: number; stepSize: number | null };
  /** Emitted when the playback detail (pinned ladder depth) changes (null = Auto) */
  ladderDepthChange: { dimIndex: number; ladderDepth: number | 'auto' | null };
  /** Emitted when direction changes (bounce mode) */
  directionChange: { dimIndex: number; direction: AnimationDirection };
  /**
   * Playback missed the requested cadence or a visible node is still below the
   * committed-energy display threshold. `committedEnergyFraction` is the
   * worst-served non-empty laddered node's committed energy, or `null` when
   * nothing on screen carries energy stamps (#2374).
   */
  fpsWarning: {
    dimIndex: number;
    targetFPS: number;
    actualFPS: number;
    committedEnergyFraction: number | null;
  };
}
