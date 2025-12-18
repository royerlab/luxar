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
  /** Emitted when direction changes (bounce mode) */
  directionChange: { dimIndex: number; direction: AnimationDirection };
  /** Emitted when measured FPS differs significantly from target */
  fpsWarning: { dimIndex: number; targetFPS: number; actualFPS: number };
}
