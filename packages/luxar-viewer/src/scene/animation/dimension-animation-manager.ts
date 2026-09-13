/**
 * Dimension Animation Manager
 *
 * Manages FPS-based animation through dimension ranges with various loop modes.
 * Integrates with AnimationController for frame updates and SceneDimsManager for dimension value updates.
 *
 * Features:
 * - FPS-based throttling with presets [0.5, 1, 2, 5, 10, 15, 30, 60, 120]
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
import { snapDiscreteValue } from '../scene-dims-manager';
import { config } from '../../config';
import { ENERGY_RELEASE_THRESHOLD } from '../lod-display-gate';
import { log, Modules } from '../../utils/log';
import { clamp } from '../../utils/clamp';
import { advanceDimensionValue } from './advance-value';
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
  /**
   * Playback "detail": pin the additive-ladder depth (rungs) for every frame;
   * null = Auto (time-budgeted). See `DimensionAnimationState.ladderDepth`.
   */
  ladderDepth?: number | null;
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
    private animationController: AnimationController,
    /**
     * Optional probe for the worst-served laddered node's committed energy.
     * Supplied by the wiring in `input/.../dimension-navigation/setup.ts`, which
     * has the scene. Absent (or returning `null`) falls back to cadence-only
     * feedback, which is the honest behaviour when nothing on screen is stamped.
     */
    private committedQuality?: () => number | null
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

        // Cadence alone cannot say whether this is a problem. Since #2377 a
        // playback pass streams every cache-resident rung, so the playhead
        // routinely slows to wait for data — that is the pacing gate working,
        // and frames on screen are complete. Before #2377 the opposite held:
        // cadence was MET because a pass committed one rung and stopped, so
        // this feedback stayed silent through the blank-frame defect (#2374).
        // Consult committed quality so the two are distinguishable.
        const threshold = config.dimensionAnimation.ui.feedbackThreshold;
        const energy = this.committedQuality?.() ?? null;
        const cadenceSlipped = state.actualFPS < state.targetFPS * threshold;
        const framesStillFilling = energy !== null && energy < ENERGY_RELEASE_THRESHOLD;
        if (cadenceSlipped || framesStillFilling) {
          const enoughOnScreen = energy !== null && energy >= ENERGY_RELEASE_THRESHOLD;
          this.dispatchEvent({
            type: 'fpsWarning',
            dimIndex,
            targetFPS: state.targetFPS,
            actualFPS: state.actualFPS,
            committedEnergyFraction: energy,
          });

          if (config.dimensionAnimation.ui.showFPSFeedback) {
            const cadence = `Dim ${dimIndex}: ${state.actualFPS.toFixed(1)} of ${state.targetFPS} fps requested`;
            if (enoughOnScreen) {
              // NOT a warning: enough content is visible to read the frame and
              // the playhead is pacing to data as designed.
              log.info(
                Modules.ANIMATION,
                `${cadence} — pacing to data, enough on screen to read ` +
                  `(committed energy ${(energy * 100).toFixed(0)}%)`
              );
            } else {
              log.warning(
                Modules.ANIMATION,
                energy === null
                  ? `${cadence} — committed quality unknown (no energy stamps on this scene)`
                  : `${cadence} — still filling in ` +
                      `(committed energy ${(energy * 100).toFixed(0)}%)`
              );
            }
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
      const step = this.resolveAnimationStep(state, metadata);

      // Next value + boundary handling (pure — see advance-value.ts). The
      // manager APPLIES the returned direction; peekNextValue() does not.
      const result = advanceDimensionValue({
        current: currentValue,
        min,
        max,
        step,
        direction: state.direction,
        loopMode: state.loopMode,
        targetFPS: state.targetFPS,
        continuousTraverseMs: config.dimensionAnimation.timing.continuousTraverseSeconds * 1000,
      });
      const nextValue = result.value;

      if (result.shouldStop) {
        // Animation complete (loop mode: once)
        this.pause(dimIndex);
        this.dispatchEvent({ type: 'complete', dimIndex });
        log.info(Modules.ANIMATION, `Dimension ${dimIndex} animation complete`);
        return;
      }

      if (result.directionChanged) {
        // Direction changed (bounce mode)
        state.direction = result.direction;
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
   * PEEK the value the next playback tick would move to — loop/bounce/
   * backward aware, WITHOUT mutating any animation state. Used by the t+1
   * slice prefetcher to warm the S-cache for the upcoming tick.
   *
   * @returns The predicted next value, or null when the dimension is not
   *   playing, dims/ranges are unavailable, or the next step would STOP
   *   playback (loop mode 'once' at its boundary — nothing to prefetch).
   */
  peekNextValue(dimIndex: number): number | null {
    const state = this.animationStates.get(dimIndex);
    if (!state?.isPlaying) return null;

    const dims = this.sceneDimsManager.getDims();
    if (!dims || dimIndex >= dims.ndim) return null;
    if (!this.dimensionRanges || dimIndex >= this.dimensionRanges.length) return null;

    const [min, max] = this.dimensionRanges[dimIndex];
    const metadata = dims.metadata?.[dimIndex];
    const step = this.resolveAnimationStep(state, metadata);

    const result = advanceDimensionValue({
      current: dims.currentStep[dimIndex],
      min,
      max,
      step,
      direction: state.direction,
      loopMode: state.loopMode,
      targetFPS: state.targetFPS,
      continuousTraverseMs: config.dimensionAnimation.timing.continuousTraverseSeconds * 1000,
    });
    if (result.shouldStop) return null;
    // Predict the EXACT landing: setDimensionValue snaps a discrete dim to
    // its range-min-anchored grid. Mid-range ticks and loop endpoints are
    // already on-grid, but sharing the helper keeps prefetch and playhead
    // byte-identical for fractional steps.
    return metadata?.discrete
      ? snapDiscreteValue(result.value, metadata.step || 1.0, min, max)
      : result.value;
  }

  /** Indices of every dimension currently playing. */
  getPlayingDimIndices(): number[] {
    const playing: number[] = [];
    for (const [dimIndex, state] of this.animationStates) {
      if (state.isPlaying) playing.push(dimIndex);
    }
    return playing;
  }

  /** Whether ANY dimension is currently playing (== getFrameBudgetMs() !== null). */
  isAnyPlaying(): boolean {
    for (const state of this.animationStates.values()) {
      if (state.isPlaying) return true;
    }
    return false;
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
        // Create new state with defaults, then apply the caller's options.
        state = this.createDefaultState();
        if (options?.targetFPS !== undefined) state.targetFPS = options.targetFPS;
        if (options?.loopMode !== undefined) state.loopMode = options.loopMode;
        if (options?.direction !== undefined) state.direction = options.direction;
        if (options?.ladderDepth !== undefined) state.ladderDepth = options.ladderDepth;
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
        if (options?.ladderDepth !== undefined) {
          state.ladderDepth = options.ladderDepth;
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
   * Playback "detail" for the current tick: the pinned additive-ladder depth
   * (rungs) the progressive loaders must load for every frame, or null when no
   * playing dimension pins one (time-budgeted streaming). With several playing
   * dimensions the DEEPEST pin wins — a pinned dim must never be drawn below
   * its setting because another dim plays on Auto. Threaded to the loaders as
   * `ViewState.ladderDepth` by `updateAllNDNodes`, and to the t+1 shadow
   * prefetch, so the next frame's S-cache entry carries the pinned prefix.
   */
  getPlaybackLadderDepth(): number | null {
    let depth: number | null = null;
    for (const state of this.animationStates.values()) {
      if (!state.isPlaying || state.ladderDepth === null) continue;
      depth = depth === null ? state.ladderDepth : Math.max(depth, state.ladderDepth);
    }
    return depth;
  }

  /**
   * Set the playback detail for a dimension: the pinned additive-ladder depth
   * (rungs, `Infinity` = whole ladder) every frame is drawn at while it plays,
   * or null for Auto (time-budgeted streaming). Takes effect on the next tick.
   */
  setLadderDepth(dimIndex: number, ladderDepth: number | null): void {
    let state = this.animationStates.get(dimIndex);
    if (!state) {
      state = this.createDefaultState();
      this.animationStates.set(dimIndex, state);
    }
    const normalized = ladderDepth === null || !(ladderDepth >= 1) ? null : Math.floor(ladderDepth);
    state.ladderDepth = normalized;
    this.dispatchEvent({ type: 'ladderDepthChange', dimIndex, ladderDepth: normalized });
    log.info(
      Modules.ANIMATION,
      `Set dimension ${dimIndex} playback detail to ${normalized === null ? 'auto' : normalized === Infinity ? 'all rungs' : `${normalized} rungs`}`
    );
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
      state = this.createDefaultState();
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
   * Fresh per-dimension state seeded from config.dimensionAnimation.defaults.
   * Single source for play()/setTargetFPS()/setLoopMode()/setStepSize() so a
   * new default (like stepSize) cannot be forgotten at one call site.
   */
  private createDefaultState(): DimensionAnimationState {
    const currentTime = performance.now();
    return {
      isPlaying: false,
      targetFPS: config.dimensionAnimation.defaults.targetFPS,
      loopMode: config.dimensionAnimation.defaults.loop,
      direction: config.dimensionAnimation.defaults.direction,
      stepSize: config.dimensionAnimation.defaults.stepSize,
      ladderDepth: config.dimensionAnimation.defaults.ladderDepth,
      lastUpdateTime: currentTime,
      frameCount: 0,
      lastFPSMeasurementTime: currentTime,
      actualFPS: 0,
    };
  }

  /**
   * The per-tick step handed to advanceDimensionValue: the user's explicit
   * override when set (quantized to the authored grid for discrete dims,
   * one cell minimum — see #1520), else the authored step for discrete
   * dims, else null (continuous fps-derived increment). MUST be used by
   * BOTH updateDimension and peekNextValue — the playhead and the t+1
   * prefetch have to agree.
   */
  private resolveAnimationStep(
    state: DimensionAnimationState,
    metadata: { discrete?: boolean; step?: number } | undefined
  ): number | null {
    const override = state.stepSize;
    if (override == null) {
      return metadata?.discrete ? (metadata.step ?? 1.0) : null;
    }
    if (metadata?.discrete) {
      // Quantize a discrete dim's override to the authored grid, one cell
      // minimum — the same rule as calculateStepSize's discrete branch.
      // Handing a sub-grid override to advanceDimensionValue would let
      // setDimensionValue's snap round every tick straight back to where it
      // started (playback frozen, no warning), and a non-multiple of the
      // grid would land the snapped playhead on a different value than the
      // unsnapped peekNextValue prefetch (#1520).
      const gridStep = metadata.step && metadata.step > 0 ? metadata.step : 1;
      return Math.max(gridStep, Math.round(override / gridStep) * gridStep);
    }
    return override;
  }

  /**
   * Set (or clear, with null) the per-dimension step override. Drives the
   * animation per-tick increment AND the [ / ] keyboard navigation; the
   * slider wheel/drag deliberately stay on the dimension's own base step.
   * Values are validated (finite, > 0) and clamped to the dimension's range
   * width so one step can never overshoot the whole range.
   */
  setStepSize(dimIndex: number, stepSize: number | null): void {
    if (stepSize !== null && (!Number.isFinite(stepSize) || stepSize <= 0)) {
      log.warning(
        Modules.ANIMATION,
        `setStepSize(${dimIndex}): rejecting invalid step ${stepSize} (must be finite and > 0)`
      );
      return;
    }
    let state = this.animationStates.get(dimIndex);
    if (!state) {
      state = this.createDefaultState();
      this.animationStates.set(dimIndex, state);
    }
    let applied = stepSize;
    if (applied !== null && this.dimensionRanges && dimIndex < this.dimensionRanges.length) {
      const [min, max] = this.dimensionRanges[dimIndex];
      const width = max - min;
      if (width > 0 && applied > width) applied = width;
    }
    state.stepSize = applied;
    this.dispatchEvent({ type: 'stepChange', dimIndex, stepSize: applied });
    log.info(
      Modules.ANIMATION,
      `Set dimension ${dimIndex} step to ${applied === null ? 'auto' : applied}`
    );
  }

  /** The per-dimension step override, or null when on Auto / no state yet. */
  getStepSize(dimIndex: number): number | null {
    return this.animationStates.get(dimIndex)?.stepSize ?? null;
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
      state = this.createDefaultState();
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
