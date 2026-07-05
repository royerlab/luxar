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
 * - If FPS < minFPS: Scale DPR down by scaleDownFactor (probe-and-verify;
 *   see U-shape section below)
 * - If FPS > maxFPS for hysteresisSeconds: Scale DPR up by scaleUpFactor
 * - DPR is clamped between minDPR and window.devicePixelRatio
 *
 * # U-shape awareness
 *
 * Lowering DPR is NOT monotonically faster. On macOS/Chrome a small
 * backbuffer must be upscaled by the OS compositor to display at CSS
 * pixel size, and that upscale cost grows as the DPR mismatch grows.
 * The result is a U-shape: render cost drops, hits a minimum around
 * the host's "sweet spot" DPR, then rises again as the upscale cost
 * dominates. Many small render passes (bloom mips, MSAA targets) make
 * the rise sharper.
 *
 * Naive adaptive DPR assumes lower = faster and walks straight past
 * the sweet spot into the slow zone. This manager guards against
 * that:
 *
 * - After each scaleDown, enter a probe window. Remember the
 *   pre-change DPR and FPS.
 * - When the probe window expires, compare the post-change FPS to
 *   the baseline.
 * - If FPS did not improve by a meaningful margin (PROBE_IMPROVEMENT
 *   threshold), revert to the previous DPR and record it as a floor.
 *   Further scaleDown is blocked from crossing the floor.
 * - The floor decays after `floorTTL` so the manager re-probes when
 *   scene content changes.
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
  /** Lowest DPR scaleDown will accept right now (tightens via U-shape probes). */
  dprFloor: number;
  /** True while a scaleDown move is being verified for U-shape improvement. */
  probing: boolean;
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

  // U-shape probe state. Set whenever scaleDown moves to a smaller
  // DPR; cleared once the probe verifies the move helped (or reverts
  // it if not).
  private pendingProbe: {
    previousDPR: number;
    previousFPS: number;
    probedDPR: number;
    startTime: number;
  } | null = null;

  // Lowest DPR known to actually improve FPS. scaleDown will not
  // cross this floor. Initialised to config.minDPR and tightened when
  // a probe reveals lower-DPR was unhelpful. `floorSetAt` lets us
  // decay the floor so the manager re-probes after a while (scene
  // content may have changed enough to move the U-shape minimum).
  private dprFloor: number;
  private floorSetAt: number = 0;

  // How long to wait after a scaleDown before evaluating its effect.
  // Needs to be long enough that the renderer/post-processing reallocation
  // costs are out of the FPS window. The FPS window itself is 1000ms,
  // so we wait somewhat longer for a representative sample.
  private readonly PROBE_WINDOW_MS = 1500;

  // Required relative FPS improvement to consider a scaleDown
  // successful (5%). Anything less and we treat the move as
  // ineffective: at best a wash, at worst a step backward.
  private readonly PROBE_IMPROVEMENT = 1.05;

  // How long the U-shape floor stays sticky before we allow another
  // downward probe. Scene content changes (new layers, camera moves,
  // dimension switches) can shift the U-shape minimum.
  private readonly FLOOR_TTL_MS = 30_000;

  // Callback for UI updates
  private onDPRChange: DPRChangeCallback | null = null;

  // True after pinManualDPR(): the DPR is locked for the session
  // (`?dpr=` URL param) and setEnabled() becomes a no-op so persisted
  // per-scene settings can't silently re-enable adaptation mid-run.
  private pinned: boolean = false;

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
    this.dprFloor = this.config.minDPR;
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

    // If a probe is in flight, settle it first. We don't trigger
    // another scaleDown while a probe is pending — we need a clean
    // FPS sample of the just-applied DPR before deciding anything else.
    if (this.pendingProbe) {
      if (timestamp - this.pendingProbe.startTime < this.PROBE_WINDOW_MS) {
        return; // probe still gathering samples
      }
      this.settleProbe(timestamp, fps);
      return;
    }

    // Floor decays so we re-probe after the configured TTL. Scene
    // content can change enough to shift the U-shape minimum.
    if (this.dprFloor > this.config.minDPR && timestamp - this.floorSetAt > this.FLOOR_TTL_MS) {
      log.info(
        Modules.ADAPTIVE_DPR,
        `DPR floor ${this.dprFloor.toFixed(2)} expired — re-enabling scale-down probes`
      );
      this.dprFloor = this.config.minDPR;
    }

    if (fps < this.config.minFPS) {
      // Performance is poor - scale down (and arm a probe so we can
      // verify the move actually helped).
      this.scaleDown(timestamp, fps);
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
   * Look at the FPS measured after a scaleDown probe. If it improved
   * by at least PROBE_IMPROVEMENT, the move was useful — keep it. If
   * not, revert and record the probed DPR as a floor so we don't try
   * to dip below it again until the TTL expires.
   *
   * `timestamp` is the frame timestamp that triggered the settle —
   * we use it (not `performance.now()`) for floorSetAt so the TTL
   * decay is consistent with the rest of the FPS-window timing.
   */
  private settleProbe(timestamp: number, currentFPS: number): void {
    if (!this.pendingProbe) return;
    const probe = this.pendingProbe;
    this.pendingProbe = null;

    const fpsRatio = probe.previousFPS > 0 ? currentFPS / probe.previousFPS : 0;
    if (fpsRatio >= this.PROBE_IMPROVEMENT) {
      // The move helped — keep it and let normal scaling decisions resume.
      log.custom(
        LogEmoji.PERFORMANCE,
        Modules.ADAPTIVE_DPR,
        `Probe accepted at DPR ${probe.probedDPR.toFixed(2)}: ` +
          `${probe.previousFPS.toFixed(1)} → ${currentFPS.toFixed(1)} FPS` +
          ` (×${fpsRatio.toFixed(2)})`
      );
      return;
    }

    // The move did not help. Revert to the previous DPR and set a
    // floor at the probed level so future scaleDown calls skip it.
    log.warning(
      Modules.ADAPTIVE_DPR,
      `Probe rejected at DPR ${probe.probedDPR.toFixed(2)}: ` +
        `FPS ${probe.previousFPS.toFixed(1)} → ${currentFPS.toFixed(1)} ` +
        `(×${fpsRatio.toFixed(2)}, want ≥${this.PROBE_IMPROVEMENT.toFixed(2)}). ` +
        `Reverting to DPR ${probe.previousDPR.toFixed(2)}; ` +
        `floor set, will retry in ${(this.FLOOR_TTL_MS / 1000).toFixed(0)}s.`
    );

    this.currentDPR = probe.previousDPR;
    this.isReducedResolution = this.currentDPR < this.nativeDPR * 0.95;
    this.applyDPR();
    this.dprFloor = probe.probedDPR;
    this.floorSetAt = timestamp;

    if (this.onDPRChange) {
      this.onDPRChange(this.currentDPR, this.isReducedResolution);
    }
  }

  /**
   * Scale DPR down for better performance, arming a probe so we
   * verify the move actually helped (see U-shape comment at the top
   * of this file).
   */
  private scaleDown(timestamp: number, fps: number): void {
    const proposed = this.currentDPR * this.config.scaleDownFactor;
    // Block scaleDown from moving TO OR BELOW the U-shape floor. Using
    // an early-return (rather than `Math.max(dprFloor, proposed)`) is
    // load-bearing: when a probe is rejected, `dprFloor` is tightened
    // to the probed value, and the next tick's `proposed` lands at
    // that exact value. `Math.max` would clamp to the floor and re-fire
    // the same failing probe every ~2s tick — the 30s TTL never gets a
    // chance to expire. Returning early here keeps us at the current
    // DPR until the TTL lifts the floor.
    if (proposed <= this.dprFloor + 0.001) return;
    const newDPR = proposed;

    // Only apply if there's a meaningful change
    if (Math.abs(newDPR - this.currentDPR) < 0.01) return;

    const previousDPR = this.currentDPR;
    this.currentDPR = newDPR;
    this.applyDPR();

    // Update reduced resolution mode status
    this.isReducedResolution = newDPR < this.nativeDPR * 0.95;

    log.custom(
      LogEmoji.PERFORMANCE,
      Modules.ADAPTIVE_DPR,
      `Scaled down: DPR ${previousDPR.toFixed(2)} → ${newDPR.toFixed(2)} ` +
        `(FPS: ${fps.toFixed(1)}, probing for U-shape)`
    );

    // Arm the probe so the next evaluateAndAdjust pass after
    // PROBE_WINDOW_MS judges whether this move helped.
    this.pendingProbe = {
      previousDPR,
      previousFPS: fps,
      probedDPR: newDPR,
      startTime: timestamp,
    };

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
    if (this.pinned) {
      log.info(
        Modules.ADAPTIVE_DPR,
        `Ignoring setEnabled(${enabled}) — DPR is pinned for this session (?dpr= URL param)`
      );
      return;
    }
    if (this.isEnabled === enabled) return;

    this.isEnabled = enabled;

    if (!enabled) {
      // Reset to native DPR when disabled
      this.currentDPR = this.nativeDPR;
      this.applyDPR();
      this.isReducedResolution = false;
      this.highFPSStartTime = null;
      this.pendingProbe = null;
      this.dprFloor = this.config.minDPR;
      this.floorSetAt = 0;
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
      dprFloor: this.dprFloor,
      probing: this.pendingProbe !== null,
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

    // DPR changes force renderer/post-processing target reallocations, so
    // avoid repeating that expensive path for duplicate slider/input events.
    if (Math.abs(clampedDPR - this.currentDPR) < 0.01) return;

    this.currentDPR = clampedDPR;
    this.isReducedResolution = clampedDPR < this.nativeDPR * 0.95;
    this.applyDPR();

    log.info(Modules.ADAPTIVE_DPR, `Manual DPR set to ${clampedDPR.toFixed(2)}`);
  }

  /**
   * Pin a fixed manual DPR for the whole session (`?dpr=` URL param).
   *
   * Disables adaptive mode, applies `dpr` as a manual DPR (clamped to
   * [0.25, native]), and locks the enabled state: subsequent
   * `setEnabled()` calls (persisted per-scene settings, the Performance
   * toggle, scene metadata) are ignored for the rest of the session.
   * Intended for deterministic E2E/visual-regression runs and repros.
   *
   * @param dpr - Device pixel ratio to pin
   */
  pinManualDPR(dpr: number): void {
    this.setEnabled(false);
    this.setManualDPR(dpr);
    this.pinned = true;
    log.info(Modules.ADAPTIVE_DPR, `DPR pinned at ${this.currentDPR.toFixed(2)} for this session`);
  }

  /**
   * Check whether the DPR is pinned for this session (`?dpr=` URL param).
   */
  isPinned(): boolean {
    return this.pinned;
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
