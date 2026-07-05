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
 * - DPR walks multiplicatively below the LIVE window.devicePixelRatio
 *   (re-read on every evaluation/public read — monitor drags and browser
 *   zoom change it at runtime) and stops strictly above
 *   max(minDPR, learned U-shape floor)
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
 * - If FPS did not improve by a meaningful margin (config
 *   probeImprovement), revert to the previous DPR and record it as a
 *   floor. Further scaleDown is blocked from crossing the floor.
 * - The floor decays after config.floorTtlMs so the manager re-probes
 *   when scene content changes.
 *
 * The control loop is decomposed into pure, timestamp-driven modules
 * under ./adaptive-dpr/ (FPS tracker, probe controller, bounds ledger);
 * this class is the orchestrating facade and owns everything
 * environmental (live devicePixelRatio, renderer, config, callbacks).
 */

import { config } from '../config';
import type { AdaptiveDPRConfig } from '../config/types';
import { adaptiveDPRConfig as adaptiveDPRDefaults } from '../config/sections/adaptive-dpr/data';
import { log, Modules, LogEmoji } from '../utils/log';
import { clamp } from '../utils/clamp';
import { FPSTracker } from './adaptive-dpr/fps-tracker';
import { ProbeController } from './adaptive-dpr/probe-controller';
import { BoundsLedger } from './adaptive-dpr/bounds-ledger';

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

  // Sliding-window FPS estimation (see adaptive-dpr/fps-tracker.ts).
  private readonly FPS_SAMPLE_WINDOW_MS = 1000;
  private fpsTracker = new FPSTracker(this.FPS_SAMPLE_WINDOW_MS);

  // State
  private currentDPR: number;
  // Last observed window.devicePixelRatio. The native DPR is NOT a
  // constant: monitor drags and browser zoom change it at runtime, so
  // every consumer reads it live via syncNativeDPR() and this snapshot
  // exists only to detect changes (see syncNativeDPR for the rebase
  // rules applied when it moves).
  private lastSeenNativeDPR: number;
  private isEnabled: boolean;
  private lastEvaluationTime: number = 0;
  private highFPSStartTime: number | null = null;
  private isReducedResolution: boolean = false;

  // U-shape probe lifecycle (see adaptive-dpr/probe-controller.ts) and
  // the learned floor it feeds (see adaptive-dpr/bounds-ledger.ts).
  // Both constructed in the ctor once config is merged.
  private probeController: ProbeController;
  private boundsLedger: BoundsLedger;

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
    // Layered merge: the data.ts literal (imported directly, NOT via the
    // config module) supplies structural per-key defaults, so a partial
    // `config.adaptiveDPR` — e.g. the fixed-shape mock in unit tests —
    // can never leave a knob `undefined` and silently invert a
    // comparison against it.
    this.config = {
      ...adaptiveDPRDefaults,
      ...config.adaptiveDPR,
      ...customConfig,
    };

    this.lastSeenNativeDPR = this.readLiveNativeDPR();
    this.currentDPR = this.lastSeenNativeDPR;
    this.probeController = new ProbeController({
      windowMs: this.config.probeWindowMs,
      improvement: this.config.probeImprovement,
    });
    this.boundsLedger = new BoundsLedger({
      minDPR: this.config.minDPR,
      floorTtlMs: this.config.floorTtlMs,
    });
    // Initial enabled state from config. At runtime, this is overridden by
    // renderingControls.defaults.adaptiveDPREnabled (persisted per-scene in localStorage).
    this.isEnabled = this.config.enabled;

    log.info(
      Modules.ADAPTIVE_DPR,
      `Initialized (enabled: ${this.isEnabled}, native DPR: ${this.lastSeenNativeDPR.toFixed(2)})`
    );
  }

  /** Live `window.devicePixelRatio` with the 0/undefined guard. */
  private readLiveNativeDPR(): number {
    return window.devicePixelRatio || 1;
  }

  /**
   * Detect a native-DPR change (monitor drag, browser zoom) and rebase.
   *
   * The U-shape floor, a pending probe, the scale-up hysteresis timer,
   * and the FPS window are all calibrated against absolute DPR values
   * and frame timings of the OLD display, so a native change clears
   * them wholesale. The operating DPR follows one of two rules:
   *
   * - Tracking native (no reduction/override engaged): follow the new
   *   native silently. The renderer already tracks live DPR by itself
   *   (null override in dpr-policy), so re-applying would only trigger
   *   a redundant render-target reallocation.
   * - Explicitly reduced/manual: clamp to the new native and re-apply,
   *   so a move to a lower-DPI monitor never leaves a supersampling
   *   override behind (the pre-fix failure mode: a 2.0 override kept
   *   rendering 4x the pixels on a 1x monitor).
   *
   * Detection is lazy — evaluation ticks and public reads — which
   * covers browser zoom (resize → interaction → evaluation) and
   * monitor drags on the next activity without a matchMedia listener.
   *
   * @returns the live native DPR
   */
  private syncNativeDPR(): number {
    const live = this.readLiveNativeDPR();
    if (Math.abs(live - this.lastSeenNativeDPR) < 0.01) return live;

    const previousNative = this.lastSeenNativeDPR;
    this.lastSeenNativeDPR = live;

    // Absolute-DPR calibrations from the old display are stale.
    this.boundsLedger.reset();
    this.probeController.void_();
    this.highFPSStartTime = null;
    this.fpsTracker.clear();

    const wasTrackingNative = Math.abs(this.currentDPR - previousNative) < 0.01;
    let applied = false;
    if (wasTrackingNative) {
      this.currentDPR = live;
    } else if (this.currentDPR > live) {
      this.currentDPR = live;
      this.applyDPR();
      applied = true;
    }
    const wasReduced = this.isReducedResolution;
    this.isReducedResolution = this.currentDPR < live * 0.95;

    log.info(
      Modules.ADAPTIVE_DPR,
      `Native DPR changed ${previousNative.toFixed(2)} → ${live.toFixed(2)}; ` +
        `rebased (DPR ${this.currentDPR.toFixed(2)}, floor/probe/FPS state cleared)`
    );

    if (this.onDPRChange && (applied || wasReduced !== this.isReducedResolution)) {
      this.onDPRChange(this.currentDPR, this.isReducedResolution);
    }
    return live;
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

    this.fpsTracker.push(timestamp);

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
    return this.fpsTracker.getFPS();
  }

  /**
   * Evaluate current performance and adjust DPR if needed
   */
  private evaluateAndAdjust(timestamp: number): void {
    // Rebase first if the display changed: a native-DPR change clears
    // the FPS window, so getCurrentFPS() below returns 0 and this tick
    // naturally becomes a no-op while fresh samples accumulate.
    this.syncNativeDPR();

    const fps = this.getCurrentFPS();

    // Need at least some frames to make a decision
    if (fps === 0) return;

    // If a probe is in flight, settle it first. We don't trigger
    // another scaleDown while a probe is pending — we need a clean
    // FPS sample of the just-applied DPR before deciding anything else.
    const verdict = this.probeController.evaluate(timestamp, fps);
    if (verdict) {
      this.applyProbeVerdict(verdict, timestamp, fps);
      return;
    }

    // Floor decays so we re-probe after the configured TTL. Scene
    // content can change enough to shift the U-shape minimum.
    if (this.boundsLedger.decayIfExpired(timestamp)) {
      log.info(
        Modules.ADAPTIVE_DPR,
        'DPR floor expired — re-enabling scale-down probes (floor back at ' +
          `${this.boundsLedger.dprFloor.toFixed(2)})`
      );
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
   * Apply a settled probe verdict from the ProbeController.
   *
   * Accepted: the scale-down helped — keep it, normal decisions resume
   * next tick. Rejected: revert to the pre-probe DPR and tighten the
   * ledger floor to the probed value so scale-down skips it until the
   * TTL expires. `timestamp` is the frame timestamp that triggered the
   * settle — used (not `performance.now()`) for the floor clock so TTL
   * decay stays consistent with FPS-window timing. A 'pending' verdict
   * means the probe window is still open: decide nothing this tick.
   */
  private applyProbeVerdict(
    verdict: NonNullable<ReturnType<ProbeController['evaluate']>>,
    timestamp: number,
    currentFPS: number
  ): void {
    if (verdict.kind === 'pending') return;

    const { probe, fpsRatio } = verdict;
    if (verdict.kind === 'accepted') {
      log.custom(
        LogEmoji.PERFORMANCE,
        Modules.ADAPTIVE_DPR,
        `Probe accepted at DPR ${probe.probedDPR.toFixed(2)}: ` +
          `${probe.previousFPS.toFixed(1)} → ${currentFPS.toFixed(1)} FPS` +
          ` (×${fpsRatio.toFixed(2)})`
      );
      return;
    }

    // Rejected: revert and floor.
    log.warning(
      Modules.ADAPTIVE_DPR,
      `Probe rejected at DPR ${probe.probedDPR.toFixed(2)}: ` +
        `FPS ${probe.previousFPS.toFixed(1)} → ${currentFPS.toFixed(1)} ` +
        `(×${fpsRatio.toFixed(2)}, want ≥${this.config.probeImprovement.toFixed(2)}). ` +
        `Reverting to DPR ${probe.previousDPR.toFixed(2)}; ` +
        `floor set, will retry in ${(this.config.floorTtlMs / 1000).toFixed(0)}s.`
    );

    this.currentDPR = probe.previousDPR;
    this.isReducedResolution = this.currentDPR < this.lastSeenNativeDPR * 0.95;
    this.applyDPR();
    this.boundsLedger.recordRejection(probe.probedDPR, timestamp);

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
    // Block scaleDown from moving TO OR BELOW the U-shape floor. The
    // ledger's to-or-below early-return (rather than clamping to the
    // floor) is load-bearing — see BoundsLedger.blocksScaleDownTo.
    if (this.boundsLedger.blocksScaleDownTo(proposed)) return;
    const newDPR = proposed;

    // Only apply if there's a meaningful change
    if (Math.abs(newDPR - this.currentDPR) < 0.01) return;

    const previousDPR = this.currentDPR;
    this.currentDPR = newDPR;
    this.applyDPR();

    // Update reduced resolution mode status
    this.isReducedResolution = newDPR < this.lastSeenNativeDPR * 0.95;

    log.custom(
      LogEmoji.PERFORMANCE,
      Modules.ADAPTIVE_DPR,
      `Scaled down: DPR ${previousDPR.toFixed(2)} → ${newDPR.toFixed(2)} ` +
        `(FPS: ${fps.toFixed(1)}, probing for U-shape)`
    );

    // Arm the probe so the next evaluateAndAdjust pass after
    // config.probeWindowMs judges whether this move helped.
    this.probeController.arm({
      previousDPR,
      previousFPS: fps,
      probedDPR: newDPR,
      startTime: timestamp,
    });

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
    const newDPR = Math.min(this.lastSeenNativeDPR, this.currentDPR * this.config.scaleUpFactor);

    // Only apply if there's a meaningful change and we're not at max
    if (Math.abs(newDPR - this.currentDPR) < 0.01) return;
    if (this.currentDPR >= this.lastSeenNativeDPR - 0.01) return;

    this.currentDPR = newDPR;
    this.applyDPR();

    // Update reduced resolution mode status
    this.isReducedResolution = newDPR < this.lastSeenNativeDPR * 0.95;

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
      // Reset to the LIVE native DPR when disabled — the display may
      // have changed since construction (monitor drag, browser zoom).
      const nativeDPR = this.syncNativeDPR();
      this.currentDPR = nativeDPR;
      this.applyDPR();
      this.isReducedResolution = false;
      this.highFPSStartTime = null;
      this.probeController.void_();
      this.boundsLedger.reset();
      this.fpsTracker.clear();

      if (this.onDPRChange) {
        this.onDPRChange(nativeDPR, false);
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
    const nativeDPR = this.syncNativeDPR();
    return {
      enabled: this.isEnabled,
      currentDPR: this.currentDPR,
      currentFPS: this.getCurrentFPS(),
      isReducedResolution: this.isReducedResolution,
      nativeDPR,
      dprFloor: this.boundsLedger.dprFloor,
      probing: this.probeController.isPending,
    };
  }

  /**
   * Get current DPR value.
   *
   * Live-consistent: syncs against the current display first, so after
   * a monitor/zoom change the returned value never reports a stale
   * native snapshot (callers like the recording session persist this
   * value and would otherwise install it as a supersampling override).
   */
  getCurrentDPR(): number {
    this.syncNativeDPR();
    return this.currentDPR;
  }

  /**
   * Get the native DPR (live `window.devicePixelRatio`, rebasing
   * internal state if the display changed since the last read).
   */
  getNativeDPR(): number {
    return this.syncNativeDPR();
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

    // Clamp DPR to reasonable range against the LIVE native value.
    const nativeDPR = this.syncNativeDPR();
    const minDPR = 0.25;
    const clampedDPR = clamp(dpr, minDPR, nativeDPR);

    // DPR changes force renderer/post-processing target reallocations, so
    // avoid repeating that expensive path for duplicate slider/input events.
    if (Math.abs(clampedDPR - this.currentDPR) < 0.01) return;

    this.currentDPR = clampedDPR;
    this.isReducedResolution = clampedDPR < nativeDPR * 0.95;
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
    this.fpsTracker.clear();
    this.onDPRChange = null;
    this.renderer = null;
    log.info(Modules.ADAPTIVE_DPR, 'Disposed');
  }
}
