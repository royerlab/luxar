/**
 * Adaptive Pixel Ratio Manager
 *
 * Dynamically adjusts the device pixel ratio based on real-time FPS
 * to maintain smooth rendering performance. Uses hysteresis to prevent
 * rapid toggling between quality levels.
 *
 * Algorithm:
 * - Sample FPS using a 1-second sliding window of frame timestamps
 * - Evaluate every 500ms (configurable)
 * - If FPS < minFPS: Scale DPR down by scaleDownFactor immediately
 * - If FPS > maxFPS for hysteresisSeconds: Scale DPR up by scaleUpFactor
 * - DPR is clamped between minDPR and window.devicePixelRatio
 */

import { config } from '../config';
import type { AdaptiveDPRConfig } from '../config/types';
import { log, Modules, LogEmoji } from '../utils/log';
import { clamp } from '../utils/clamp';

/**
 * Interface for the renderer manager that can set pixel ratio
 * This avoids circular dependencies with SceneManager
 */
export interface DPRRenderer {
  setAdaptivePixelRatio(dpr: number): void;
}

/**
 * State returned by getState() for debugging and UI
 */
export interface AdaptiveDPRState {
  enabled: boolean;
  currentDPR: number;
  currentFPS: number;
  isReducedResolution: boolean;
  nativeDPR: number;
}

/**
 * Callback type for DPR change notifications
 */
export type DPRChangeCallback = (dpr: number, isReducedResolution: boolean) => void;

/**
 * Manages adaptive pixel ratio for performance optimization
 */
export class AdaptiveDPRManager {
  private config: AdaptiveDPRConfig;
  private renderer: DPRRenderer | null = null;

  // FPS tracking using circular buffer for O(1) insertion and trimming
  private frameTimestamps: number[] = [];
  private frameStartIndex: number = 0;
  private readonly FPS_SAMPLE_WINDOW_MS = 1000;

  // State
  private currentDPR: number;
  private readonly nativeDPR: number;
  private isEnabled: boolean;
  private lastEvaluationTime: number = 0;
  private highFPSStartTime: number | null = null;
  private isReducedResolution: boolean = false;

  // Callback for UI updates
  private onDPRChange: DPRChangeCallback | null = null;

  /**
   * Create adaptive DPR manager
   *
   * @param customConfig - Optional partial config to override defaults
   */
  constructor(customConfig?: Partial<AdaptiveDPRConfig>) {
    // Merge custom config with defaults
    this.config = {
      ...config.adaptiveDPR,
      ...customConfig,
    };

    this.nativeDPR = window.devicePixelRatio || 1;
    this.currentDPR = this.nativeDPR;
    // Initial enabled state from config. At runtime, this is overridden by
    // renderingControls.defaults.adaptiveDPREnabled (persisted per-scene in localStorage).
    this.isEnabled = this.config.enabled;

    log.info(
      Modules.ADAPTIVE_DPR,
      `Initialized (enabled: ${this.isEnabled}, native DPR: ${this.nativeDPR.toFixed(2)})`
    );
  }

  /**
   * Set the renderer to use for DPR changes
   * Called after SceneManager is initialized to avoid circular dependencies
   */
  setRenderer(renderer: DPRRenderer): void {
    this.renderer = renderer;
  }

  /**
   * Record frame timestamp for FPS calculation
   * Should be called once per frame from the animation loop
   *
   * @param timestamp - Current timestamp from performance.now()
   */
  recordFrame(timestamp: number): void {
    if (!this.isEnabled) return;

    this.frameTimestamps.push(timestamp);

    // Trim timestamps older than sample window using index advancement (O(1) amortized)
    const cutoff = timestamp - this.FPS_SAMPLE_WINDOW_MS;
    while (
      this.frameStartIndex < this.frameTimestamps.length &&
      this.frameTimestamps[this.frameStartIndex] < cutoff
    ) {
      this.frameStartIndex++;
    }

    // Compact array periodically to prevent unbounded growth
    if (this.frameStartIndex > 120) {
      this.frameTimestamps = this.frameTimestamps.slice(this.frameStartIndex);
      this.frameStartIndex = 0;
    }

    // Evaluate DPR at configured interval
    if (timestamp - this.lastEvaluationTime >= this.config.evaluationIntervalMs) {
      this.evaluateAndAdjust(timestamp);
      this.lastEvaluationTime = timestamp;
    }
  }

  /**
   * Get current FPS from frame timestamps
   * Returns 0 if not enough data to calculate
   */
  getCurrentFPS(): number {
    const frameCount = this.frameTimestamps.length - this.frameStartIndex;
    if (frameCount < 2) return 0;

    // FPS = frame count over the sample window
    // We use frame count - 1 because we're measuring intervals between frames
    const timeSpan =
      this.frameTimestamps[this.frameTimestamps.length - 1] -
      this.frameTimestamps[this.frameStartIndex];

    if (timeSpan <= 0) return 0;

    // Convert from frames per millisecond to frames per second
    return ((frameCount - 1) * 1000) / timeSpan;
  }

  /**
   * Evaluate current performance and adjust DPR if needed
   */
  private evaluateAndAdjust(timestamp: number): void {
    const fps = this.getCurrentFPS();

    // Need at least some frames to make a decision
    if (fps === 0) return;

    if (fps < this.config.minFPS) {
      // Performance is poor - scale down immediately
      this.scaleDown(fps);
      this.highFPSStartTime = null; // Reset hysteresis
    } else if (fps > this.config.maxFPS) {
      // Performance is good - track duration for hysteresis
      if (this.highFPSStartTime === null) {
        this.highFPSStartTime = timestamp;
      } else if (timestamp - this.highFPSStartTime >= this.config.hysteresisSeconds * 1000) {
        // Sustained high FPS for long enough - try scaling up
        this.scaleUp(fps);
        this.highFPSStartTime = null; // Reset after scaling
      }
    } else {
      // FPS is in acceptable range - reset hysteresis
      this.highFPSStartTime = null;
    }
  }

  /**
   * Scale DPR down for better performance
   */
  private scaleDown(fps: number): void {
    const newDPR = Math.max(this.config.minDPR, this.currentDPR * this.config.scaleDownFactor);

    // Only apply if there's a meaningful change
    if (Math.abs(newDPR - this.currentDPR) < 0.01) return;

    this.currentDPR = newDPR;
    this.applyDPR();

    // Update reduced resolution mode status
    this.isReducedResolution = newDPR < this.nativeDPR * 0.95;

    log.custom(
      LogEmoji.PERFORMANCE,
      Modules.ADAPTIVE_DPR,
      `Scaled down: DPR ${newDPR.toFixed(2)} (FPS: ${fps.toFixed(1)})`
    );

    // Notify callback
    if (this.onDPRChange) {
      this.onDPRChange(newDPR, this.isReducedResolution);
    }
  }

  /**
   * Scale DPR up for better quality
   */
  private scaleUp(fps: number): void {
    // Don't exceed native DPR
    const newDPR = Math.min(this.nativeDPR, this.currentDPR * this.config.scaleUpFactor);

    // Only apply if there's a meaningful change and we're not at max
    if (Math.abs(newDPR - this.currentDPR) < 0.01) return;
    if (this.currentDPR >= this.nativeDPR - 0.01) return;

    this.currentDPR = newDPR;
    this.applyDPR();

    // Update reduced resolution mode status
    this.isReducedResolution = newDPR < this.nativeDPR * 0.95;

    log.custom(
      LogEmoji.PERFORMANCE,
      Modules.ADAPTIVE_DPR,
      `Scaled up: DPR ${newDPR.toFixed(2)} (FPS: ${fps.toFixed(1)})`
    );

    // Notify callback
    if (this.onDPRChange) {
      this.onDPRChange(newDPR, this.isReducedResolution);
    }
  }

  /**
   * Apply current DPR to renderer
   */
  private applyDPR(): void {
    if (this.renderer) {
      this.renderer.setAdaptivePixelRatio(this.currentDPR);
    }
  }

  /**
   * Enable or disable adaptive DPR
   *
   * @param enabled - Whether to enable adaptive DPR
   */
  setEnabled(enabled: boolean): void {
    if (this.isEnabled === enabled) return;

    this.isEnabled = enabled;

    if (!enabled) {
      // Reset to native DPR when disabled
      this.currentDPR = this.nativeDPR;
      this.applyDPR();
      this.isReducedResolution = false;
      this.highFPSStartTime = null;
      this.frameTimestamps = [];
      this.frameStartIndex = 0;

      if (this.onDPRChange) {
        this.onDPRChange(this.nativeDPR, false);
      }
    }

    log.info(
      Modules.ADAPTIVE_DPR,
      `${enabled ? 'Enabled' : 'Disabled'} (DPR: ${this.currentDPR.toFixed(2)})`
    );
  }

  /**
   * Check if adaptive DPR is enabled
   */
  isActive(): boolean {
    return this.isEnabled;
  }

  /**
   * Register callback for DPR changes
   * Used by UI components to show/hide resolution indicator
   *
   * @param callback - Function to call when DPR changes
   */
  setOnDPRChangeCallback(callback: DPRChangeCallback | null): void {
    this.onDPRChange = callback;
  }

  /**
   * Get current state for debugging and UI
   */
  getState(): AdaptiveDPRState {
    return {
      enabled: this.isEnabled,
      currentDPR: this.currentDPR,
      currentFPS: this.getCurrentFPS(),
      isReducedResolution: this.isReducedResolution,
      nativeDPR: this.nativeDPR,
    };
  }

  /**
   * Get current DPR value
   */
  getCurrentDPR(): number {
    return this.currentDPR;
  }

  /**
   * Get native DPR value
   */
  getNativeDPR(): number {
    return this.nativeDPR;
  }

  /**
   * Set manual DPR value (only works when adaptive is disabled)
   *
   * @param dpr - Device pixel ratio to set (clamped to 0.25 - native DPR)
   */
  setManualDPR(dpr: number): void {
    if (this.isEnabled) {
      log.warning(
        Modules.ADAPTIVE_DPR,
        'Cannot set manual DPR while adaptive resolution is enabled'
      );
      return;
    }

    // Clamp DPR to reasonable range
    const minDPR = 0.25;
    const clampedDPR = clamp(dpr, minDPR, this.nativeDPR);
    this.currentDPR = clampedDPR;
    this.applyDPR();

    log.info(Modules.ADAPTIVE_DPR, `Manual DPR set to ${clampedDPR.toFixed(2)}`);
  }

  /**
   * Check if currently in reduced resolution mode
   */
  getIsReducedResolution(): boolean {
    return this.isReducedResolution;
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.frameTimestamps = [];
    this.frameStartIndex = 0;
    this.onDPRChange = null;
    this.renderer = null;
    log.info(Modules.ADAPTIVE_DPR, 'Disposed');
  }
}
