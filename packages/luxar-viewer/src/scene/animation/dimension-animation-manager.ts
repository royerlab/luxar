/**
 * Dimension Animation Manager
 *
 * Manages FPS-based animation through dimension ranges with various loop modes.
 * Integrates with AnimationController for frame updates and SceneDimsManager for dimension value updates.
 *
 * Features:
 * - FPS-based throttling with presets [1, 2, 5, 10, 15, 30, 60]
 * - Loop modes: once, loop, bounce
 * - Per-dimension animation state
 * - Actual FPS measurement and feedback
 * - Discrete and continuous dimension support
 *
 * @example
 * ```typescript
 * const animManager = new DimensionAnimationManager(sceneDimsManager, animationController);
 *
 * // Start animation at 10 FPS
 * animManager.play(0, { targetFPS: 10, loopMode: 'loop' });
 *
 * // Listen for events
 * animManager.addEventListener('play', (e) => {
 *   console.log(`Dimension ${e.dimIndex} started`);
 * });
 *
 * // Pause
 * animManager.pause(0);
 * ```
 */

import * as THREE from 'three';
import type { AnimationController } from './animation-controller';
import type { SceneDimsManager } from '../scene-dims-manager';
import { config } from '../../config';
import { log, Modules } from '../../utils/log';
import { clamp } from '../../utils/clamp';
import type {
  DimensionAnimationState,
  DimensionAnimationEvents,
  LoopMode,
  AnimationDirection,
} from '../../types/animation';

/**
 * Options for starting dimension animation
 */
export interface PlayOptions {
  targetFPS?: number;
  loopMode?: LoopMode;
  direction?: AnimationDirection;
}

/**
 * Manages animation state for multiple dimensions
 */
export class DimensionAnimationManager extends THREE.EventDispatcher<DimensionAnimationEvents> {
  /** Animation state per dimension (sparse map) */
  private animationStates = new Map<number, DimensionAnimationState>();

  /** Whether we've registered with AnimationController */
  private isRegistered = false;

  /** Cached dimension ranges for performance */
  private dimensionRanges: [number, number][] | null = null;

  /** Per-dimension update tracking — prevents advancing faster than data loads */
  private pendingUpdates = new Set<number>();

  /** Unsubscribe function for sceneDimsManager listener */
  private removeDimsListener: (() => void) | null = null;

  /**
   * True while dispose() runs — suppresses the pause() refine re-trigger
   * (a torn-down scene must not receive a fresh dimension update).
   */
  private _disposing = false;

  /**
   * Create dimension animation manager
   *
   * @param sceneDimsManager - Singleton manager for dimension state
   * @param animationController - Controller for frame updates
   */
  constructor(
    private sceneDimsManager: SceneDimsManager,
    private animationController: AnimationController
  ) {
    super();

    // Cache dimension ranges for performance
    this.updateDimensionCache();

    // Listen for dimension changes to invalidate cache
    const listener = () => {
      this.updateDimensionCache();
    };
    this.sceneDimsManager.addListener(listener);
    this.removeDimsListener = () => {
      this.sceneDimsManager.removeListener(listener);
    };
  }

  /**
   * Update cached dimension ranges
   */
  private updateDimensionCache(): void {
    this.dimensionRanges = this.sceneDimsManager.getDimensionRanges();
  }

  /**
   * Register frame callback with animation controller
   * Called automatically when first animation starts
   */
  private ensureRegistered(): void {
    if (this.isRegistered) return;

    // Register per-frame callback with unique ID (won't overwrite other callbacks like dynamic clipping)
    // continuous: true because dimension animation needs every frame to advance
    this.animationController.addPerFrameCallback(
      'dimension-animation',
      () => {
        this.onFrame();
      },
      { continuous: true }
    );

    this.isRegistered = true;
    log.info(Modules.ANIMATION, 'Registered with AnimationController');
  }

  /**
   * Per-frame callback - update all active animations
   * Called by AnimationController on each animation frame (~60 FPS)
   */
  private onFrame(): void {
    const currentTime = performance.now();

    // Update each active animation
    for (const [dimIndex, state] of this.animationStates) {
      if (state.isPlaying) {
        this.updateDimension(dimIndex, currentTime);
      }
    }
  }

  /**
   * Update a single dimension's animation
   *
   * Includes frame synchronization: if a data update is still in progress,
   * this method skips the frame to prevent advancing animation faster than
   * data loading can keep up. This ensures complete rendering of each frame.
   *
   * @param dimIndex - Dimension index to update
   * @param currentTime - Current timestamp in milliseconds
   */
  private updateDimension(dimIndex: number, currentTime: number): void {
    try {
      // Skip if this dimension's previous update is still loading data
      // Each dimension tracks independently so multi-dimension animation isn't serialized
      if (this.pendingUpdates.has(dimIndex)) {
        return;
      }

      const state = this.animationStates.get(dimIndex);
      if (!state) {
        log.warning(Modules.ANIMATION, `No state for dimension ${dimIndex}`);
        return;
      }

      // Check if enough time has elapsed for target FPS
      const targetFrameTime = 1000 / state.targetFPS;
      const minFrameTime = config.dimensionAnimation.timing.minFrameTimeMs;
      const effectiveFrameTime = Math.max(targetFrameTime, minFrameTime);
      const elapsed = currentTime - state.lastUpdateTime;

      if (elapsed < effectiveFrameTime) {
        // Not enough time elapsed, skip this frame
        return;
      }

      // Update FPS measurement
      state.frameCount++;
      const measurementElapsed = currentTime - state.lastFPSMeasurementTime;
      if (measurementElapsed >= 1000) {
        // Calculate actual FPS over last second
        state.actualFPS = (state.frameCount / measurementElapsed) * 1000;
        state.frameCount = 0;
        state.lastFPSMeasurementTime = currentTime;

        // Check if actual FPS is significantly lower than target
        const threshold = config.dimensionAnimation.ui.feedbackThreshold;
        if (state.actualFPS < state.targetFPS * threshold) {
          this.dispatchEvent({
            type: 'fpsWarning',
            dimIndex,
            targetFPS: state.targetFPS,
            actualFPS: state.actualFPS,
          });

          if (config.dimensionAnimation.ui.showFPSFeedback) {
            log.warning(
              Modules.ANIMATION,
              `Dim ${dimIndex}: Actual FPS (${state.actualFPS.toFixed(1)}) < Target (${state.targetFPS})`
            );
          }
        }
      }

      // Get current dimension value from dims
      const dims = this.sceneDimsManager.getDims();
      if (!dims || dimIndex >= dims.ndim) {
        log.warning(Modules.ANIMATION, `Invalid dimension index ${dimIndex}`);
        this.pause(dimIndex);
        return;
      }

      const currentValue = dims.currentStep[dimIndex];

      // Get dimension range and step from metadata
      if (!this.dimensionRanges || dimIndex >= this.dimensionRanges.length) {
        log.error(Modules.ANIMATION, `No range for dimension ${dimIndex}`);
        this.pause(dimIndex);
        return;
      }

      const [min, max] = this.dimensionRanges[dimIndex];
      const metadata = dims.metadata?.[dimIndex];
      const step = metadata?.discrete ? (metadata.step ?? 1.0) : null;

      // Calculate next value
      let nextValue = this.calculateNextValue(currentValue, min, max, step, state);

      // Handle boundary conditions
      const result = this.handleBoundary(nextValue, min, max, state);
      nextValue = result.value;

      if (result.shouldStop) {
        // Animation complete (loop mode: once)
        this.pause(dimIndex);
        this.dispatchEvent({ type: 'complete', dimIndex });
        log.info(Modules.ANIMATION, `Dimension ${dimIndex} animation complete`);
        return;
      }

      if (result.directionChanged) {
        // Direction changed (bounce mode)
        this.dispatchEvent({
          type: 'directionChange',
          dimIndex,
          direction: state.direction,
        });
      }

      // Mark this dimension's update in progress
      this.pendingUpdates.add(dimIndex);

      // Update dimension value (triggers async data loading via listeners)
      this.sceneDimsManager.setDimensionValue(dimIndex, nextValue);

      // Track completion for next frame synchronization
      // This ensures animation waits for data loading before advancing
      this.sceneDimsManager
        .waitForUpdate()
        .then(() => {
          this.pendingUpdates.delete(dimIndex);
        })
        .catch((error) => {
          log.error(Modules.ANIMATION, 'Dimension update failed', error);
          this.pendingUpdates.delete(dimIndex);
        });

      // Update last update time
      state.lastUpdateTime = currentTime;
    } catch (error) {
      log.error(Modules.ANIMATION, `Failed to update dimension ${dimIndex}`, error);
      this.pause(dimIndex);
    }
  }

  /**
   * Calculate next dimension value based on step and direction
   *
   * @param current - Current dimension value
   * @param min - Minimum dimension value
   * @param max - Maximum dimension value
   * @param step - Step size (null for continuous dimensions)
   * @param state - Animation state
   * @returns Next dimension value
   */
  private calculateNextValue(
    current: number,
    min: number,
    max: number,
    step: number | null,
    state: DimensionAnimationState
  ): number {
    if (step !== null) {
      // Discrete dimension: step by integer increments
      const direction = state.direction === 'forward' ? 1 : -1;
      return current + direction * step;
    } else {
      // Continuous dimension: calculate based on target FPS
      const range = max - min;
      const traverseTime = config.dimensionAnimation.timing.continuousTraverseSeconds * 1000; // ms
      const increment = (range / traverseTime) * (1000 / state.targetFPS);
      const direction = state.direction === 'forward' ? 1 : -1;
      return current + direction * increment;
    }
  }

  /**
   * Handle boundary conditions (min/max) based on loop mode
   *
   * @param value - Proposed next value
   * @param min - Minimum dimension value
   * @param max - Maximum dimension value
   * @param state - Animation state (may be modified for bounce mode)
   * @returns Adjusted value and control flags
   */
  private handleBoundary(
    value: number,
    min: number,
    max: number,
    state: DimensionAnimationState
  ): { value: number; shouldStop: boolean; directionChanged: boolean } {
    let shouldStop = false;
    let directionChanged = false;

    if (state.direction === 'forward' && value >= max) {
      // Hit max boundary
      switch (state.loopMode) {
        case 'once':
          // Stop at max
          value = max;
          shouldStop = true;
          break;
        case 'loop':
          // Wrap to min
          value = min;
          break;
        case 'bounce':
          // Reverse direction
          value = max;
          state.direction = 'backward';
          directionChanged = true;
          break;
      }
    } else if (state.direction === 'backward' && value <= min) {
      // Hit min boundary
      switch (state.loopMode) {
        case 'once':
          // Stop at min
          value = min;
          shouldStop = true;
          break;
        case 'loop':
          // Wrap to max
          value = max;
          break;
        case 'bounce':
          // Reverse direction
          value = min;
          state.direction = 'forward';
          directionChanged = true;
          break;
      }
    }

    return { value, shouldStop, directionChanged };
  }

  /**
   * Start or resume animation for a dimension
   *
   * @param dimIndex - Dimension index to animate
   * @param options - Animation options (FPS, loop mode, direction)
   * @returns True if animation started, false if already playing
   */
  play(dimIndex: number, options?: PlayOptions): boolean {
    try {
      // Validate the dimension index up front. Without this, play() would
      // create phantom animation state for an invalid index: a negative index
      // slips past the per-frame `dimIndex >= ndim` guard entirely (so the
      // animation runs forever advancing nothing — setDimensionValue rejects
      // the out-of-bounds write), and an out-of-range index dispatches a
      // spurious `play` event before the next frame pauses it. Reject negatives
      // always; reject `>= ndim` only when dims are known, preserving the
      // pre-init path (where the per-frame guard self-corrects once dims load).
      const dims = this.sceneDimsManager.getDims();
      if (dimIndex < 0 || (dims !== null && dimIndex >= dims.ndim)) {
        log.warning(Modules.ANIMATION, `play(): ignoring invalid dimension index ${dimIndex}`);
        return false;
      }

      // Get or create state
      let state = this.animationStates.get(dimIndex);

      if (!state) {
        // Create new state with defaults
        const currentTime = performance.now();
        state = {
          isPlaying: false,
          targetFPS: options?.targetFPS ?? config.dimensionAnimation.defaults.targetFPS,
          loopMode: options?.loopMode ?? config.dimensionAnimation.defaults.loop,
          direction: options?.direction ?? config.dimensionAnimation.defaults.direction,
          lastUpdateTime: currentTime,
          frameCount: 0,
          lastFPSMeasurementTime: currentTime,
          actualFPS: 0,
        };
        this.animationStates.set(dimIndex, state);
      } else {
        // Update existing state
        if (options?.targetFPS !== undefined) {
          state.targetFPS = options.targetFPS;
        }
        if (options?.loopMode !== undefined) {
          state.loopMode = options.loopMode;
        }
        if (options?.direction !== undefined) {
          state.direction = options.direction;
        }
      }

      // If already playing, just update options
      if (state.isPlaying) {
        log.info(Modules.ANIMATION, `Dimension ${dimIndex} already playing, updated options`);
        return false;
      }

      // Start animation
      state.isPlaying = true;
      state.lastUpdateTime = performance.now();
      state.frameCount = 0;
      state.lastFPSMeasurementTime = performance.now();

      // Ensure we're registered with animation controller
      this.ensureRegistered();

      // Ensure animation loop is running
      this.animationController.startAnimation();

      // Emit event
      this.dispatchEvent({ type: 'play', dimIndex });
      log.info(
        Modules.ANIMATION,
        `Playing dimension ${dimIndex} at ${state.targetFPS} FPS (${state.loopMode})`
      );

      return true;
    } catch (error) {
      log.error(Modules.ANIMATION, `Failed to start animation for dimension ${dimIndex}`, error);
      return false;
    }
  }

  /**
   * Pause animation for a dimension
   *
   * @param dimIndex - Dimension index to pause
   * @returns True if animation was paused, false if already paused
   */
  pause(dimIndex: number): boolean {
    const state = this.animationStates.get(dimIndex);
    if (!state || !state.isPlaying) {
      return false;
    }

    state.isPlaying = false;
    this.dispatchEvent({ type: 'pause', dimIndex });
    log.info(Modules.ANIMATION, `Paused dimension ${dimIndex}`);

    // Refine-on-pause: while playing, progressive loaders stream only within
    // the per-tick frame budget (a partial LOD ladder). Once the LAST playing
    // dimension pauses (getFrameBudgetMs() just became null — with another
    // dim still playing this would inject a spurious mid-play update),
    // re-trigger ONE update at the current position; the budget-free pass
    // lets the loaders resume from their prefix and refinement completes the
    // ladder. Suppressed during dispose (torn-down scene). setDimensionValue
    // notifies listeners unconditionally, so a same-value write still fires
    // the update.
    if (!this._disposing && this.getFrameBudgetMs() === null) {
      const dims = this.sceneDimsManager.getDims();
      if (dims && dimIndex < dims.ndim) {
        this.sceneDimsManager.setDimensionValue(dimIndex, dims.currentStep[dimIndex]);
      }
    }

    return true;
  }

  /**
   * Per-tick LOD time budget for the progressive loaders, or `null` when no
   * dimension animation is playing (normal full-refinement behavior).
   *
   * While playing, each update pass should finish within the animation frame
   * window so every tick commits a frame (see the pacing gate in
   * `updateDimension`). The budget is a fraction of the frame window of the
   * FASTEST currently-playing dimension (`config.dimensionAnimation.playback`),
   * floored at `minBudgetMs`. Read per-update by the dims listener and
   * threaded to the loaders as a per-pass directive — never persisted.
   */
  getFrameBudgetMs(): number | null {
    let maxFPS: number | null = null;
    for (const state of this.animationStates.values()) {
      if (state.isPlaying) {
        maxFPS = maxFPS === null ? state.targetFPS : Math.max(maxFPS, state.targetFPS);
      }
    }
    if (maxFPS === null) return null;
    const { budgetFraction, minBudgetMs, overheadReserveMs } = config.dimensionAnimation.playback;
    const frameWindow = 1000 / maxFPS;
    // Fractional share of the window, but at slow FPS give the loaders the
    // whole window minus a fixed projection/commit/render reserve — a 1 fps
    // tick should stream ~950ms of levels, not idle 40% of every second.
    return Math.max(frameWindow * budgetFraction, frameWindow - overheadReserveMs, minBudgetMs);
  }

  /**
   * Toggle play/pause for a dimension
   *
   * @param dimIndex - Dimension index to toggle
   * @param options - Animation options (only used if starting)
   * @returns True if now playing, false if now paused
   */
  togglePlay(dimIndex: number, options?: PlayOptions): boolean {
    const state = this.animationStates.get(dimIndex);
    const isPlaying = state?.isPlaying ?? false;

    if (isPlaying) {
      this.pause(dimIndex);
      return false;
    } else {
      this.play(dimIndex, options);
      return true;
    }
  }

  /**
   * Stop animation and remove state for a dimension
   *
   * @param dimIndex - Dimension index to stop
   */
  stop(dimIndex: number): void {
    this.pause(dimIndex);
    this.animationStates.delete(dimIndex);
    log.info(Modules.ANIMATION, `Stopped dimension ${dimIndex}`);
  }

  /**
   * Check if a dimension is currently animating
   *
   * @param dimIndex - Dimension index to check
   * @returns True if animating, false otherwise
   */
  isAnimating(dimIndex: number): boolean {
    return this.animationStates.get(dimIndex)?.isPlaying ?? false;
  }

  /**
   * Get animation state for a dimension
   *
   * @param dimIndex - Dimension index
   * @returns Animation state, or undefined if not animating
   */
  getState(dimIndex: number): DimensionAnimationState | undefined {
    return this.animationStates.get(dimIndex);
  }

  /**
   * Set target FPS for a dimension
   *
   * @param dimIndex - Dimension index
   * @param fps - Target FPS (clamped to valid range)
   */
  setTargetFPS(dimIndex: number, fps: number): void {
    let state = this.animationStates.get(dimIndex);

    // Create state if it doesn't exist (for pre-configuring settings before starting animation)
    if (!state) {
      const currentTime = performance.now();
      state = {
        isPlaying: false,
        targetFPS: config.dimensionAnimation.defaults.targetFPS,
        loopMode: config.dimensionAnimation.defaults.loop,
        direction: config.dimensionAnimation.defaults.direction,
        lastUpdateTime: currentTime,
        frameCount: 0,
        lastFPSMeasurementTime: currentTime,
        actualFPS: 0,
      };
      this.animationStates.set(dimIndex, state);
    }

    // Clamp to valid range
    const { customMin, customMax } = config.dimensionAnimation.presets;
    state.targetFPS = clamp(fps, customMin, customMax);

    this.dispatchEvent({ type: 'speedChange', dimIndex, fps: state.targetFPS });
    log.info(Modules.ANIMATION, `Set dimension ${dimIndex} speed to ${state.targetFPS} FPS`);
  }

  /**
   * Increase speed to next preset or by increment
   *
   * @param dimIndex - Dimension index
   */
  increaseSpeed(dimIndex: number): void {
    const state = this.animationStates.get(dimIndex);
    if (!state) return;

    const presets = config.dimensionAnimation.presets.fps;
    const currentFPS = state.targetFPS;

    // Find next preset
    const nextPreset = presets.find((fps) => fps > currentFPS);
    if (nextPreset) {
      this.setTargetFPS(dimIndex, nextPreset);
    } else {
      // Already at max preset, increase by 10%
      this.setTargetFPS(dimIndex, currentFPS * 1.1);
    }
  }

  /**
   * Decrease speed to previous preset or by decrement
   *
   * @param dimIndex - Dimension index
   */
  decreaseSpeed(dimIndex: number): void {
    const state = this.animationStates.get(dimIndex);
    if (!state) return;

    const presets = config.dimensionAnimation.presets.fps;
    const currentFPS = state.targetFPS;

    // Find previous preset (iterate backwards)
    for (let i = presets.length - 1; i >= 0; i--) {
      if (presets[i] < currentFPS) {
        this.setTargetFPS(dimIndex, presets[i]);
        return;
      }
    }

    // Already at min preset, decrease by 10%
    this.setTargetFPS(dimIndex, currentFPS * 0.9);
  }

  /**
   * Set loop mode for a dimension
   *
   * @param dimIndex - Dimension index
   * @param loopMode - Loop mode to set
   */
  setLoopMode(dimIndex: number, loopMode: LoopMode): void {
    let state = this.animationStates.get(dimIndex);

    // Create state if it doesn't exist (for pre-configuring settings before starting animation)
    if (!state) {
      const currentTime = performance.now();
      state = {
        isPlaying: false,
        targetFPS: config.dimensionAnimation.defaults.targetFPS,
        loopMode: config.dimensionAnimation.defaults.loop,
        direction: config.dimensionAnimation.defaults.direction,
        lastUpdateTime: currentTime,
        frameCount: 0,
        lastFPSMeasurementTime: currentTime,
        actualFPS: 0,
      };
      this.animationStates.set(dimIndex, state);
    }

    state.loopMode = loopMode;
    this.dispatchEvent({ type: 'loopModeChange', dimIndex, loopMode });
    log.info(Modules.ANIMATION, `Set dimension ${dimIndex} loop mode to ${loopMode}`);
  }

  /**
   * Clean up all animations and unregister from animation controller
   */
  dispose(): void {
    // Suppress the pause() refine re-trigger while tearing down.
    this._disposing = true;

    // Pause all animations
    for (const dimIndex of this.animationStates.keys()) {
      this.pause(dimIndex);
    }

    // Clear state
    this.animationStates.clear();
    this.pendingUpdates.clear();

    // Unregister from animation controller (only removes our callback, not others)
    if (this.isRegistered) {
      this.animationController.removePerFrameCallback('dimension-animation');
      this.isRegistered = false;
    }

    // Unsubscribe from sceneDimsManager to prevent memory leak
    if (this.removeDimsListener) {
      this.removeDimsListener();
      this.removeDimsListener = null;
    }

    log.info(Modules.ANIMATION, 'Disposed dimension animation manager');
  }
}
