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
import type { SceneDimsManager } from './scene-dims-manager';
import { config } from '../config';
import { log, Modules } from '../utils/log';
import type {
  DimensionAnimationState,
  DimensionAnimationEvents,
  LoopMode,
  AnimationDirection,
} from '../types/animation';

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
    this.sceneDimsManager.addListener(() => {
      this.updateDimensionCache();
    });
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

    // Register per-frame callback
    this.animationController.setPerFrameCallback(() => {
      this.onFrame();
    });

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
   * @param dimIndex - Dimension index to update
   * @param currentTime - Current timestamp in milliseconds
   */
  private updateDimension(dimIndex: number, currentTime: number): void {
    try {
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

      // Update dimension value
      this.sceneDimsManager.setDimensionValue(dimIndex, nextValue);

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

    return true;
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
    const state = this.animationStates.get(dimIndex);
    if (!state) return;

    // Clamp to valid range
    const { customMin, customMax } = config.dimensionAnimation.presets;
    state.targetFPS = Math.max(customMin, Math.min(customMax, fps));

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
    const state = this.animationStates.get(dimIndex);
    if (!state) return;

    state.loopMode = loopMode;
    this.dispatchEvent({ type: 'loopModeChange', dimIndex, loopMode });
    log.info(Modules.ANIMATION, `Set dimension ${dimIndex} loop mode to ${loopMode}`);
  }

  /**
   * Clean up all animations and unregister from animation controller
   */
  dispose(): void {
    // Pause all animations
    for (const dimIndex of this.animationStates.keys()) {
      this.pause(dimIndex);
    }

    // Clear state
    this.animationStates.clear();

    // Unregister from animation controller
    if (this.isRegistered) {
      this.animationController.setPerFrameCallback(null);
      this.isRegistered = false;
    }

    log.info(Modules.ANIMATION, 'Disposed dimension animation manager');
  }
}
