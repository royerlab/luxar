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
 * - Thresholds derive from the display's estimated achievable rAF rate
 *   ("cap"): scale down below scaleDownFpsRatio × cap (probe-and-verify;
 *   see U-shape section below); scale up by scaleUpFactor after FPS has
 *   stayed above scaleUpFpsRatio × cap for hysteresisSeconds (with a
 *   small mid-band grace so isolated dropped-frame samples don't
 *   restart the wait)
 * - DPR walks multiplicatively below the session CEILING — the LIVE
 *   window.devicePixelRatio (re-read on every evaluation/public read:
 *   monitor drags and browser zoom change it at runtime), capped by the
 *   allow-high-DPR setting (see ./pixel-ratio-cap; by default the
 *   ceiling is 1.0 even on a HiDPI display) — and stops strictly above
 *   max(minDPR, learned U-shape floor)
 * - Sustained distress-level FPS (below what any real display throttle
 *   can produce) demotes the operating ceiling to 1.0 in one step —
 *   HiDPI is a luxury a scene at ~10fps has proven it can't afford
 *   (see the estimator's sub-throttle distress verdict). With high DPR
 *   disallowed this demotion machinery is inert by construction, the
 *   ceiling already being 1.0 — the setting is precisely this demotion,
 *   applied up front instead of after the evidence
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
 * under ./adaptive-dpr/ (FPS tracker, stall detector, refresh-rate
 * estimator, hysteresis tracker, probe controller, bounds ledger — see
 * that folder's README); this class is the orchestrating facade and owns
 * everything environmental (live devicePixelRatio, renderer, config,
 * callbacks, pause/idle/resume hooks).
 */

import { config } from '../config';
import type { AdaptiveDPRConfig } from '../config/types';
import {
  getMaxPixelRatioCap,
  getNativePixelRatio,
  isHighDPRAllowed,
  setHighDPRAllowed,
  setMaxPixelRatioCap,
} from './pixel-ratio-cap';
import { adaptiveDPRConfig as adaptiveDPRDefaults } from '../config/sections/adaptive-dpr/data';
import { log, Modules, LogEmoji } from '../utils/log';
import { clamp } from '../utils/clamp';
import { FPSTracker } from './adaptive-dpr/fps-tracker';
import { ProbeController, type ProbeVerdict } from './adaptive-dpr/probe-controller';
import { BoundsLedger } from './adaptive-dpr/bounds-ledger';
import { HysteresisTracker } from './adaptive-dpr/hysteresis-tracker';
import { RefreshRateEstimator } from './adaptive-dpr/refresh-rate-estimator';
import { StallDetector } from './adaptive-dpr/stall-detector';

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
  /** Estimated achievable rAF rate the FPS thresholds derive from. */
  refreshRateCap: number;
  /** Effective scale-up ceiling: the lower of the session ceiling (the
   *  display's DPR, capped unless high DPR is allowed) and a LEARNED 1.0
   *  demotion after repeated punished ascents. */
  dprCeiling: number;
  /** Whether the viewer may currently render above CSS resolution. */
  allowHighDPR: boolean;
}

/**
 * Callback type for DPR change notifications
 */
export type DPRChangeCallback = (dpr: number, isReducedResolution: boolean) => void;

/**
 * Consecutive intervals that must read as the FRAME RATE (rather than
 * dead time) after a gap reset before the FPS window is trusted for
 * LEARNING again.
 *
 * The stall detector compares each interval against the median of its
 * four PRECEDING neighbours, so by the THIRD consecutive dead interval
 * half that memory is dead time and the interval is reclassified as the
 * frame rate. From timestamps alone that is not even wrong — three 5s
 * gaps in a row genuinely is 0.2fps for 15s — but the FPS window it
 * lands in holds nothing except dead time. Left trusted, that window
 * arms a probe, settles it against a baseline measured on the SAME dead
 * time, rejects it for want of an improvement, and pins a U-shape floor
 * on data containing zero render measurement — a floor whose TTL then
 * backs off exponentially on every repeat. That is exactly the harm the
 * gap reset exists to prevent, in an otherwise healthy session.
 *
 * Classifying better is impossible from intervals alone, so the
 * newly-absorbed cadence is instead made UNTRUSTED for a couple of
 * intervals: scale-downs still apply (fewer pixels help a stuttering
 * scene too) but nothing is learned from them. The cost on a genuine
 * slowdown is that the FIRST scale-down goes unratified — it is applied
 * immediately either way — after which the normal probe machinery
 * engages.
 *
 * The protection is bounded, deliberately: a burst of up to
 * CADENCE_TRUST_INTERVALS + 2 dead intervals teaches nothing, and past
 * that the "burst" is a sustained slow regime the manager must be free
 * to learn from. The bound is counted in INTERVALS, and that is the only
 * form the guarantee takes — what it buys in wall clock depends on how
 * long they are (four 5s gaps is 20s, four 400ms hitches is 1.6s) and on
 * how much of its own `probeWindowMs` plus representative span a probe
 * can still gather afterwards. Measured with the production 0.9 step
 * inside a 60fps session, the turnover for consecutive 400ms hitches sits
 * at NINE: eight (3.2s) leave the floor untouched, nine (3.6s) pin a 0.81
 * floor.
 */
const CADENCE_TRUST_INTERVALS = 2;

/**
 * Frames the loop must fit inside one `probeWindowMs` before a content
 * change is allowed to VOID an in-flight probe (see
 * notifyContentChanged). Two is the ProbeController's own settle
 * minimum — below it a replacement probe could not be judged either, so
 * voiding trades a contaminated verdict for no verdict at all.
 */
const MIN_FRAMES_TO_RERUN_PROBE = 2;

/**
 * Why a tick's FPS window is unrepresentative of steady-state render
 * cost, or null when it is trustworthy. Both causes suppress LEARNING
 * identically (see evaluateAndAdjust); the distinction is carried only
 * so the scale-down log can name the real one.
 *
 * Exported deliberately, despite having no caller outside this file:
 * `scaleDown` is private but still documented (`excludePrivate: false`) and
 * names this type in its signature, and a documented member referencing a
 * non-exported symbol trips the TypeDoc warning ratchet (86/86 with the
 * export). Do not un-export it as cleanup.
 */
export type SuppressionCause = 'load' | 'cadence' | null;

/** One settled probe, as recorded for `getDiagnostics()`. */
export interface AdaptiveDPRProbeRecord {
  /** `confounded` = a content change ran through the window (direction not trusted). */
  kind: 'accepted' | 'rejected' | 'inconclusive' | 'confounded';
  previousDPR: number;
  probedDPR: number;
  previousFPS: number;
  /** FPS at the verdict; `null` for inconclusive probes (no clean sample). */
  currentFPS: number | null;
  /** currentFPS / previousFPS; `null` for inconclusive probes. */
  fpsRatio: number | null;
  timestamp: number;
}

/** Bounded probe history kept for diagnostics. */
const PROBE_RECORD_LIMIT = 16;

/**
 * `getState()` plus the controller's decision history — what a perf probe
 * needs to tell "the DPR walked down during loading" from "a probe proved
 * the step helped". Read via `__luxarDebug.getPerf().adaptiveDpr`.
 */
export interface AdaptiveDPRDiagnostics extends AdaptiveDPRState {
  /** Why the last evaluation withheld learning, or `null` when it did not. */
  suppressedBy: SuppressionCause;
  backoffLevel: number;
  scaleDowns: number;
  scaleUps: number;
  /** Most recent last; at most {@link PROBE_RECORD_LIMIT} entries. */
  verdicts: AdaptiveDPRProbeRecord[];
}

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
  // Last observed pixel-ratio cap. Like the native snapshot above this
  // exists only to DETECT a change — the cap has writers outside this
  // class (see rendering/pixel-ratio-cap), so syncNativeDPR watches it
  // rather than assuming every write came through setHighDPRAllowed.
  private lastSeenCap: number;
  private isEnabled: boolean;
  private lastEvaluationTime: number = 0;
  private isReducedResolution: boolean = false;

  // Tells dead time apart from a slow frame rate (see
  // adaptive-dpr/stall-detector.ts): an interval is a STALL only when it
  // is both over `gapResetMs` and a large outlier against the recent
  // cadence. Its cadence memory deliberately survives the gap reset it
  // drives (that is what makes it converge) and is cleared only where
  // the frame STREAM itself breaks — pause, display change, disable,
  // dispose; NOT on a content change (see notifyContentChanged).
  // Constructed in the ctor once config is merged.
  private stallDetector: StallDetector;

  // Consecutive intervals the detector has classified as the frame rate
  // since the last gap reset / session boundary. Below
  // CADENCE_TRUST_INTERVALS the window is untrusted for LEARNING (see
  // that constant and `suppressed` in evaluateAndAdjust). Reset wherever
  // stallDetector.clear() is called, plus by the gap reset itself.
  private nonStallIntervals: number = 0;

  // U-shape probe lifecycle (see adaptive-dpr/probe-controller.ts), the
  // learned floor it feeds (see adaptive-dpr/bounds-ledger.ts), the
  // scale-up streak (see adaptive-dpr/hysteresis-tracker.ts), and the
  // display-cap estimate the thresholds derive from (see
  // adaptive-dpr/refresh-rate-estimator.ts). All constructed in the
  // ctor once config is merged.
  private probeController: ProbeController;
  private boundsLedger: BoundsLedger;
  private hysteresis: HysteresisTracker;
  private refreshRateEstimator: RefreshRateEstimator;

  // Callback for UI updates
  private onDPRChange: DPRChangeCallback | null = null;

  // Injected "data is loading" predicate (mirrors the animation
  // controller's context-lost predicate pattern). While true, FPS
  // samples are treated as jank-polluted: scale-downs apply unprobed
  // and nothing feeds the estimator or settles probes.
  private loadActivityPredicate: (() => boolean) | null = null;

  // Diagnostics only (`getDiagnostics()`): never read by the control logic.
  private lastSuppressedBy: SuppressionCause = null;
  private scaleDownCount = 0;
  private scaleUpCount = 0;
  private readonly probeRecords: AdaptiveDPRProbeRecord[] = [];

  // Coalescing clock for notifyContentChanged() — only advances on the
  // calls that are acted upon.
  private lastContentChangeAt: number | null = null;

  // Idle-restore state: prepareIdleFrame() snaps DPR to the CEILING for
  // the resting frame and remembers where the loop was operating so
  // notifyResumed() can return there in ONE step instead of reactively
  // re-walking the reduction ladder on every interaction burst.
  private restingAtCeiling: boolean = false;
  private lastOperatingDPR: number | null = null;

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
    this.lastSeenCap = getMaxPixelRatioCap();
    // Start at the CEILING, not the native DPR: with high DPR disallowed
    // (the default) the opening frame must already be at CSS resolution
    // rather than paying 4x the fragment cost until the FPS loop
    // reactively walks it back down. The renderer boundary clamps to the
    // same ceiling anyway (dpr-policy.getActivePixelRatio), so this keeps
    // the manager's own bookkeeping honest about what is on screen.
    this.currentDPR = this.ceiling();
    this.probeController = new ProbeController({
      windowMs: this.config.probeWindowMs,
      improvement: this.config.probeImprovement,
      minSamples: this.config.probeMinSamples,
      minSpanMs: this.FPS_SAMPLE_WINDOW_MS * 0.5,
    });
    this.boundsLedger = new BoundsLedger({
      minDPR: this.config.minDPR,
      floorTtlMs: this.config.floorTtlMs,
      backoffMultiplier: this.config.backoffMultiplier,
      backoffMaxTtlMs: this.config.backoffMaxTtlMs,
      ceilingTtlMs: this.config.ceilingTtlMs,
      punishedAscentWindowMs: this.config.punishedAscentWindowMs,
      punishedAscentThreshold: this.config.punishedAscentThreshold,
    });
    this.hysteresis = new HysteresisTracker({
      hysteresisMs: this.config.hysteresisSeconds * 1000,
      graceSamples: this.config.midbandGraceSamples,
    });
    this.refreshRateEstimator = new RefreshRateEstimator(this.config.refreshRateFallback);
    this.stallDetector = new StallDetector(this.config.gapResetMs);
    // Initial enabled state from config. At runtime, this is overridden by
    // renderingControls.defaults.adaptiveDPREnabled (persisted per-scene in localStorage).
    this.isEnabled = this.config.enabled;

    log.info(
      Modules.ADAPTIVE_DPR,
      `Initialized (enabled: ${this.isEnabled}, native DPR: ` +
        `${this.lastSeenNativeDPR.toFixed(2)}, ceiling: ${this.ceiling().toFixed(2)})`
    );
  }

  /** Live `window.devicePixelRatio` with the 0/undefined guard. */
  private readLiveNativeDPR(): number {
    return getNativePixelRatio();
  }

  /**
   * The highest DPR this session may currently render at: the display's
   * own DPR, capped by the pixel-ratio cap (see
   * `rendering/pixel-ratio-cap` — usually 1.0, because high DPR is off by
   * default).
   *
   * Every place that used to treat `lastSeenNativeDPR` as an upper bound
   * reads this instead. The NATIVE snapshot is still what
   * `syncNativeDPR` tracks and rebases on, because a display change is a
   * real event regardless of the cap; the ceiling is what that snapshot
   * is allowed to mean for rendering.
   *
   * Deliberately built from the SNAPSHOT rather than calling
   * `getMaxPixelRatio()` directly, so a tick stays internally consistent
   * with the native value `syncNativeDPR` resolved at its start.
   */
  private ceiling(): number {
    return Math.min(this.lastSeenNativeDPR, getMaxPixelRatioCap());
  }

  /**
   * Detect a native-DPR change (monitor drag, browser zoom) and rebase.
   *
   * The U-shape floor, a pending probe, the scale-up hysteresis timer,
   * and the FPS window are all calibrated against absolute DPR values
   * and frame timings of the OLD display, so a native change clears
   * them wholesale. The operating DPR follows one of two rules:
   *
   * - Tracking the ceiling (no reduction/manual value engaged): follow
   *   the new ceiling.
   * - Explicitly reduced/manual: keep the value, clamped to the new
   *   ceiling, so a move to a lower-DPI monitor never leaves a
   *   supersampling override behind (the pre-fix failure mode: a 2.0
   *   override kept rendering 4x the pixels on a 1x monitor).
   *
   * Either way the new DPR is RE-APPLIED. There used to be a fast path
   * here that skipped the apply while tracking, on the grounds that a
   * null override in dpr-policy already follows the live ceiling for
   * free. It was unsound: the manager cannot see the renderer's
   * override, and "currentDPR equals the ceiling" does not imply the
   * renderer is tracking. The property fuzzer produced the counterexample
   * — a reduction to 1.0 applied while the ceiling was 2.0 (so the
   * override is an explicit 1.0), then the ceiling falls to 1.0, then
   * rises to 3.0. The last step reads as "tracking" because currentDPR
   * happens to equal the OLD ceiling, so nothing was applied, and the
   * manager reported 3.0 while the renderer kept drawing at 1.0 until
   * something else forced a resize. The optimization only ever saved one
   * redundant reallocation on a monitor drag — during which the browser
   * fires a resize anyway.
   *
   * Detection is lazy — evaluation ticks and public reads — which
   * covers browser zoom (resize → interaction → evaluation) and
   * monitor drags on the next activity without a matchMedia listener.
   *
   * The CAP is watched here too, not just the native DPR, because it has
   * writers outside this class (a `?dpr=` pin, a capture lifting it and
   * putting it back). Those all re-apply immediately today, but a cap
   * raised with no follow-up would otherwise leave `currentDPR` reporting
   * the old ceiling while a null override silently rendered at the new
   * one — a divergence the property fuzzer finds in one step.
   *
   * Watching the cap matters for visibility too: a monitor drag or a
   * zoom fires a window resize, which re-sizes the backbuffer on its own,
   * but nothing fires for a cap change. Without this the new ceiling
   * would have no visible effect until the user happened to resize the
   * window.
   *
   * @returns the live native DPR
   */
  private syncNativeDPR(): number {
    const live = this.readLiveNativeDPR();
    const liveCap = getMaxPixelRatioCap();
    const nativeChanged = Math.abs(live - this.lastSeenNativeDPR) >= 0.01;
    const capChanged = liveCap !== this.lastSeenCap;
    if (!nativeChanged && !capChanged) return live;

    const previousNative = this.lastSeenNativeDPR;
    const previousCap = this.lastSeenCap;
    this.lastSeenNativeDPR = live;
    this.lastSeenCap = liveCap;

    // Absolute-DPR calibrations from the old display are stale — and so
    // is its refresh-cap estimate (a new monitor can have a different
    // refresh rate entirely).
    this.boundsLedger.reset();
    this.probeController.void_();
    this.hysteresis.clear();
    this.refreshRateEstimator.clear();
    this.fpsTracker.clear();
    this.stallDetector.clear();
    this.nonStallIntervals = 0;
    this.restingAtCeiling = false;
    this.lastOperatingDPR = null;

    // Both bounds are CEILINGS, not raw natives: under a cap the two
    // differ, and comparing against native would classify a capped
    // session as "explicitly reduced" on every display change and
    // re-apply for nothing.
    const previousCeiling = Math.min(previousNative, previousCap);
    const newCeiling = this.ceiling();

    // Tracking the old ceiling means "follow the new one"; anything else
    // is an explicit reduction/manual value, kept but clamped.
    const wasTrackingCeiling = Math.abs(this.currentDPR - previousCeiling) < 0.01;
    this.currentDPR = wasTrackingCeiling ? newCeiling : Math.min(this.currentDPR, newCeiling);
    this.applyDPR();
    this.isReducedResolution = this.currentDPR < newCeiling * 0.95;

    log.info(
      Modules.ADAPTIVE_DPR,
      `Ceiling changed (native ${previousNative.toFixed(2)} → ${live.toFixed(2)}, ` +
        `cap ${previousCap} → ${liveCap}); rebased (DPR ${this.currentDPR.toFixed(2)}, ` +
        `ceiling ${newCeiling.toFixed(2)}, floor/probe/FPS state cleared)`
    );

    if (this.onDPRChange) {
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

    // Gap detection: a frame arriving long after the previous one means
    // the loop stalled (GC pause, synchronous decode, idle-resume, tab
    // switch the pause hook didn't see). The window's pre-gap frames
    // plus the dead time would read as artificially low FPS and ratchet
    // a spurious, probe-ratified scale-down — so reset the session
    // state and start sampling fresh. An in-flight probe is voided (not
    // judged): the experiment's data is contaminated, learn nothing.
    //
    // "Long after" is an OUTLIER test, not an absolute one (see
    // adaptive-dpr/stall-detector.ts): the interval must clear
    // `gapResetMs` AND be several times the recent inter-frame median.
    // A plain absolute threshold made adaptation structurally inert
    // exactly where shedding pixels matters most — below
    // ~1000/gapResetMs fps (2.9fps at the 350ms default) EVERY interval
    // exceeds it, so the tracker was cleared on every frame AND
    // `lastEvaluationTime` was pushed to `timestamp`, leaving the
    // `>= evaluationIntervalMs` test below comparing 0 against 500;
    // evaluateAndAdjust never ran and a software-rasterized 0.5fps scene
    // sat at native DPR forever. Under the outlier rule a genuine
    // slowdown costs one or two misread intervals while the median
    // follows the new cadence, after which the samples are kept and the
    // normal machinery scales down (the FPS tracker's minimum retention
    // keeps the estimate defined at those rates). Isolated stalls in a
    // healthy session — one, or up to two in a row — stay outliers
    // against the fast median and are still discarded. By the THIRD in a
    // row half the cadence memory is dead time, so it is absorbed as the
    // frame rate: the window is kept and a scale-down may apply, but the
    // newly-absorbed cadence is untrusted for LEARNING for
    // CADENCE_TRUST_INTERVALS intervals (see that constant), so a short
    // stall burst costs a transient reduction and no learned floor.
    //
    // Residual limitation: the test only sees inter-frame intervals, so
    // a dead period ALTERNATING one-for-one with a single fast frame
    // (2000/100/2000/100…, measured) makes the dead intervals half the
    // memory — the median lands between the phases and they are kept as
    // "the frame rate", mixing real dead time into the FPS window. The
    // manager still ADAPTS there (measured at native 2: the reported FPS
    // is only ever 0, 10 or 0.5, all far below the down threshold, and
    // the DPR does come down); what it cannot do is report the ~0.5fps
    // the user actually perceives. Two or more fast frames between dead
    // periods go the other way: the median stays fast, so the dead time
    // is correctly discarded every cycle and the window never
    // accumulates more than those few fast frames — so a 16.7/16.7/400ms
    // cadence (~6.9 perceived fps, measured) evaluates on the fast pair
    // and reads a healthy 60fps.
    const last = this.fpsTracker.lastTimestamp;
    if (last !== null) {
      if (this.stallDetector.isStall(timestamp - last)) {
        this.fpsTracker.clear();
        this.probeController.void_();
        this.hysteresis.clear();
        // Whatever cadence the next intervals establish has not been
        // observed yet — untrust it for learning until
        // CADENCE_TRUST_INTERVALS of it are in.
        this.nonStallIntervals = 0;
        // The estimator's uniform-low plateau clock and recent window
        // span the gap too — dead time must not count toward a
        // "sustained" throttle/distress verdict (learned state survives).
        this.refreshRateEstimator.noteSessionInterrupted();
        // NB: the evaluation clock is deliberately NOT pushed forward
        // here. Pinning it to `timestamp` (the pre-fix line) meant a
        // stall recurring more often than `evaluationIntervalMs` re-pinned
        // the clock before it could mature, so no evaluation with a frame
        // rate to judge ever ran: measured over 43s, a repeating
        // 16.7/16.7/400ms cadence (~6.9 perceived fps) and an
        // 8×16.7+360ms one both produced ZERO. That is the same
        // structural inertness this class was fixed for, on a different
        // pattern. What keeps dead time from counting as progress is the
        // return value below: the freshly cleared window holds a single
        // sample, so the tick that finds it has no frame rate to judge,
        // is not an evaluation, and does not consume the interval budget.
        // Clamping the pin to one interval before `timestamp` was tried
        // and MEASURED INERT (identical evaluations, DPR, floor and
        // applied-ratio counts on six cadences), because the clamped pin
        // leaves the gate satisfied on every subsequent frame anyway.
        //
        // The detector's cadence memory is deliberately not cleared here
        // either — surviving its own reset is what lets it converge.
      } else if (this.nonStallIntervals < CADENCE_TRUST_INTERVALS) {
        // Saturating count: only the distance to the trust threshold
        // matters, so it never grows past it.
        this.nonStallIntervals++;
      }
    }

    this.fpsTracker.push(timestamp);

    // Evaluate DPR at configured interval. A tick that found NO samples
    // (the window was just cleared by the gap reset above) is not an
    // evaluation and must not consume the interval budget — otherwise
    // the frame that arrives with data 16ms later has to wait another
    // full interval, which for a hitch recurring faster than
    // `evaluationIntervalMs` means the only ticks that ever run are the
    // dataless ones (measured: a 16.7/16.7/400ms cadence evaluated on
    // the freshly-cleared window every cycle and never on the two real
    // frames between the gaps).
    if (timestamp - this.lastEvaluationTime >= this.config.evaluationIntervalMs) {
      if (this.evaluateAndAdjust(timestamp)) this.lastEvaluationTime = timestamp;
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
   * Evaluate current performance and adjust DPR if needed.
   *
   * @returns true when the tick had a frame rate to judge. False means
   *   the FPS window held fewer than two samples — a just-cleared window
   *   after a gap reset or a display change — which is NOT an evaluation
   *   and must not consume the interval budget (see recordFrame).
   */
  private evaluateAndAdjust(timestamp: number): boolean {
    // Rebase first if the display changed: a native-DPR change clears
    // the FPS window, so getCurrentFPS() below returns 0 and this tick
    // naturally becomes a no-op while fresh samples accumulate.
    this.syncNativeDPR();

    const fps = this.getCurrentFPS();

    // Need at least some frames to make a decision
    if (fps === 0) return false;

    // Two situations make the window unrepresentative of steady-state
    // render cost, and both get the same treatment: scale-downs still
    // apply (fewer pixels benefit a janky or stuttering scene too) but
    // NOTHING is learned from such samples — no probes armed or settled,
    // no estimator feeding, no punished-ascent or distress bookkeeping.
    //
    // - Data is loading: the samples are decode/upload jank.
    // - The cadence was only just RECLASSIFIED from dead time to frame
    //   rate by the gap detector, so the window can be made entirely of
    //   absorbed dead time (see CADENCE_TRUST_INTERVALS).
    //
    // Which of the two it was is kept (not just the boolean) so the
    // scale-down log names the actual cause: both roads lead to an
    // unprobed reduction, and a line that blames data loading for what
    // was really an untrusted cadence sends a reader debugging this
    // machinery straight to the wrong subsystem.
    const cadenceUntrusted = this.nonStallIntervals < CADENCE_TRUST_INTERVALS;
    const suppressedBy: SuppressionCause =
      (this.loadActivityPredicate?.() ?? false) ? 'load' : cadenceUntrusted ? 'cadence' : null;
    const suppressed = suppressedBy !== null;
    this.lastSuppressedBy = suppressedBy;

    // Feed the refresh-cap estimator — but only clean, full-span
    // windows, so partial post-reset windows and load jank don't
    // pollute the mark.
    if (!suppressed && this.fpsTracker.span() >= this.FPS_SAMPLE_WINDOW_MS * 0.5) {
      this.refreshRateEstimator.addSample(fps, timestamp);
    }

    // Thresholds are RELATIVE to the display's achievable rAF rate:
    // 60Hz → down<45 / up>54; 120Hz → 90/108; a 30Hz-throttled tab →
    // 22.5/27 (so a healthy throttled 30fps neither scale-downs forever
    // nor is barred from ever scaling up, both failure modes of the old
    // fixed 50/58 thresholds).
    const refreshCap = this.refreshRateEstimator.getCap();
    const downThreshold = this.config.scaleDownFpsRatio * refreshCap;
    const upThreshold = this.config.scaleUpFpsRatio * refreshCap;

    // If a probe is in flight, settle it first. We don't trigger
    // another scaleDown while a probe is pending — we need a clean
    // FPS sample of the just-applied DPR before deciding anything else.
    const verdict = this.probeController.evaluate(timestamp, fps, {
      sampleCount: this.fpsTracker.sampleCount(),
      spanMs: this.fpsTracker.span(),
      suppressed,
    });
    if (verdict) {
      this.applyProbeVerdict(verdict, timestamp, fps);
      return true;
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
    if (this.boundsLedger.decayCeilingIfExpired(timestamp)) {
      log.info(
        Modules.ADAPTIVE_DPR,
        'DPR ceiling demotion expired — scale-up may try above 1.0 again'
      );
    }

    // Sustained sub-throttle DISTRESS (FPS below what any real display
    // throttle can produce, for a sustained period) demotes the ceiling
    // to 1.0 in one step: the scene has proven it can't afford the
    // above-1.0 HiDPI luxury, and the multiplicative scale-down walk
    // (≈7 probed steps from 2.0) would leave the user at ~10fps for
    // tens of seconds. The estimator deliberately reports this instead
    // of latching its throttle verdict — the pre-fix failure mode
    // collapsed the cap onto the loaded FPS, read ~10fps as "at the
    // display cap = healthy", and scale-up parked the DPR at native
    // for the content's lifetime.
    if (!suppressed && this.refreshRateEstimator.consumeDistress()) {
      if (this.currentDPR > 1.0 + 0.001) {
        const ttlMs = this.boundsLedger.demoteCeiling(timestamp);
        this.applyCeilingDemotion(
          fps,
          `Sustained distress-level FPS (${fps.toFixed(1)}) — too slow to be a display ` +
            `throttle; demoting the ceiling to 1.0 for ${(ttlMs / 1000).toFixed(0)}s ` +
            '(TTL-decayed; content changes re-check)'
        );
        this.hysteresis.recordLow();
        return true;
      }
      // Already at/below 1.0: nothing to demote — the verdict is
      // consumed (it re-arms) and the normal scale-down walk below
      // keeps working the reduction.
    }

    if (fps < downThreshold) {
      // A slow sample right after a scale-up above 1.0 is a "punished
      // ascent" — enough of those and the ledger demotes the operating
      // ceiling to exactly 1.0 (HiDPI is a luxury this scene has proven
      // it can't sustain). The demotion clamp IS this tick's reduction.
      // Unrepresentative samples never count — neither load jank nor
      // absorbed dead time is the ascent's fault.
      if (!suppressed && this.boundsLedger.recordSlowSample(timestamp)) {
        this.applyCeilingDemotion(fps);
      } else {
        // Performance is poor - scale down (and arm a probe so we can
        // verify the move actually helped — unless the sample is
        // unrepresentative, whether from data loading or an untrusted
        // cadence, in which case the reduction applies unprobed).
        //
        // Known limitation, measured: on a HITCH-HEAVY cadence the walk
        // can still reach minDPR with no probe ever ratifying it, because
        // every hitch is a gap reset and a gap reset VOIDS the in-flight
        // probe. At native 2.0 over 400 frames of 30ms with a 1.2s hitch
        // every 40 (~17 perceived fps, 33fps between hitches — a
        // genuinely below-threshold rate), 6 unprobed and 7 probed steps
        // reach 0.508 and not one probe produced a verdict of any kind.
        // Rate-limiting the reduction was tried: capping it at one
        // UNPROBED step per untrusted episode is inert here (the loop
        // already takes at most one — the same 6), and a latch cleared
        // only by a settled probe cannot clear at all when hitches keep
        // destroying probes, which froze a neighbouring 400ms-hitch
        // cadence (~9 perceived fps) at a single step for 50s — the
        // structural inertness this class was fixed for. The walk is
        // bounded by minDPR and lifts again at the normal hysteresis rate
        // (measured: 16 scale-ups in the 60s after the hitches stop,
        // exactly as many as from a shallower start).
        this.scaleDown(timestamp, fps, suppressedBy);
      }
      this.hysteresis.recordLow();
    } else if (fps > upThreshold) {
      // Performance is good - accumulate the sustained-high streak.
      if (this.hysteresis.recordHigh(timestamp)) {
        this.scaleUp(fps, timestamp);
        this.hysteresis.clear();
      }
    } else {
      // Mid-band: tolerated a configurable number of times in a row
      // (a couple of dropped frames must not restart the whole wait).
      this.hysteresis.recordMidband();
    }
    return true;
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
   *
   * A CONTENT-CONFOUNDED probe (one kept across a content change because
   * the loop was too slow to re-run the experiment) gets neither
   * treatment: see applyConfoundedVerdict.
   */
  private applyProbeVerdict(
    verdict: NonNullable<ReturnType<ProbeController['evaluate']>>,
    timestamp: number,
    currentFPS: number
  ): void {
    if (verdict.kind === 'pending') return;
    this.recordProbeVerdict(verdict, currentFPS, timestamp);

    if (verdict.kind === 'inconclusive') {
      // No clean sample within the extended window — keep the DPR,
      // learn nothing (no revert, no floor, no backoff movement).
      log.info(
        Modules.ADAPTIVE_DPR,
        `Probe at DPR ${verdict.probe.probedDPR.toFixed(2)} voided as inconclusive ` +
          '(no clean FPS sample) — keeping the DPR, learning nothing'
      );
      return;
    }

    const { probe, fpsRatio } = verdict;
    if (probe.contentConfounded) {
      this.applyConfoundedVerdict(verdict);
      return;
    }

    if (verdict.kind === 'accepted') {
      // The regime responds to DPR reduction — stale rejection streaks
      // no longer describe it.
      this.boundsLedger.recordAcceptance();
      log.custom(
        LogEmoji.PERFORMANCE,
        Modules.ADAPTIVE_DPR,
        `Probe accepted at DPR ${probe.probedDPR.toFixed(2)}: ` +
          `${probe.previousFPS.toFixed(1)} → ${currentFPS.toFixed(1)} FPS` +
          ` (×${fpsRatio.toFixed(2)})`
      );
      return;
    }

    // Rejected: revert and floor (repeated identical rejections
    // escalate the retry TTL exponentially — see BoundsLedger).
    this.currentDPR = probe.previousDPR;
    this.isReducedResolution = this.currentDPR < this.ceiling() * 0.95;
    this.applyDPR();
    const ttlMs = this.boundsLedger.recordRejection(probe.probedDPR, timestamp);

    log.warning(
      Modules.ADAPTIVE_DPR,
      `Probe rejected at DPR ${probe.probedDPR.toFixed(2)}: ` +
        `FPS ${probe.previousFPS.toFixed(1)} → ${currentFPS.toFixed(1)} ` +
        `(×${fpsRatio.toFixed(2)}, want ≥${this.config.probeImprovement.toFixed(2)}). ` +
        `Reverting to DPR ${probe.previousDPR.toFixed(2)}; ` +
        `floor set, will retry in ${(ttlMs / 1000).toFixed(0)}s` +
        (this.boundsLedger.backoffLevel > 1
          ? ` (backoff ×${this.boundsLedger.backoffLevel})`
          : '') +
        '.'
    );

    if (this.onDPRChange) {
      this.onDPRChange(this.currentDPR, this.isReducedResolution);
    }
  }

  /**
   * Settle a probe whose measurement window a scene-content change ran
   * through (see notifyContentChanged, which keeps such a probe only
   * when the loop is too slow to run a replacement experiment).
   *
   * Its baseline was measured on the OLD content and its settle sample on
   * the NEW one, so the ratio is confounded in BOTH directions and none
   * of the usual consequences may be applied:
   *
   * - A per-frame LOD swap to a FINER (heavier) level reads as
   *   'rejected'. Reverting would push the DPR back UP — adding pixels to
   *   a scene that just got heavier — and pin a floor for the full
   *   `floorTtlMs`, after which `blocksScaleDownTo` makes scale-down
   *   impossible for 30s. Re-armed on every swap, that recreates the
   *   "sits at native DPR forever" symptom in 30-second windows.
   * - A swap to a COARSER (faster) level reads as 'accepted', crediting
   *   the LOD coarsening to the DPR step and wiping the rejection-backoff
   *   streak that quiets scenes DPR reduction cannot help.
   *
   * So a confounded probe teaches NOTHING — exactly like an inconclusive
   * one: KEEP the reduced DPR (fewer pixels never hurt a stuttering loop,
   * which is the defensible half of the experiment), no revert, no floor,
   * no backoff movement.
   *
   * What this deliberately does NOT do is stop the walk. Under sustained
   * churn no clean experiment exists, so a reduction with no verdict is
   * all the loop can do, and walking down is the right distress response;
   * it is bounded by `minDPR` and lifts again through the normal scale-up
   * hysteresis once content settles. A previous revision held the walk
   * while a churn clock ran, and that mechanism was measured to invert
   * its own goal — its window was the same 5s constant
   * `notifyContentChanged` coalesces on, so churn arriving just slower
   * than the window walked FURTHER down (to minDPR) than not having it,
   * while per-frame churn pinned a 0.33fps scene at 1.62 indefinitely.
   */
  private applyConfoundedVerdict(verdict: Extract<ProbeVerdict, { fpsRatio: number }>): void {
    const { probe, fpsRatio } = verdict;
    log.info(
      Modules.ADAPTIVE_DPR,
      `Probe at DPR ${probe.probedDPR.toFixed(2)} settled across a content change ` +
        `(×${fpsRatio.toFixed(2)} vs the OLD content — confounded): keeping the reduction, ` +
        'learning nothing (no revert, no floor, no backoff movement)'
    );
  }

  /**
   * Scale DPR down for better performance, arming a probe so we
   * verify the move actually helped (see U-shape comment at the top
   * of this file). A scale-down taken on an UNREPRESENTATIVE window
   * applies WITHOUT a probe — the reduction still helps a janky load or
   * a stuttering loop, but such FPS samples must never become floor
   * evidence. `suppressedBy` names which cause it was (see
   * SuppressionCause) purely so the log line is honest about it.
   */
  private scaleDown(timestamp: number, fps: number, suppressedBy: SuppressionCause): void {
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
    this.scaleDownCount += 1;

    // Update reduced resolution mode status
    this.isReducedResolution = newDPR < this.ceiling() * 0.95;

    log.custom(
      LogEmoji.PERFORMANCE,
      Modules.ADAPTIVE_DPR,
      `Scaled down: DPR ${previousDPR.toFixed(2)} → ${newDPR.toFixed(2)} ` +
        `(FPS: ${fps.toFixed(1)}, ${
          suppressedBy === 'load'
            ? 'load-suppressed, unprobed'
            : suppressedBy === 'cadence'
              ? 'untrusted cadence, unprobed'
              : 'probing for U-shape'
        })`
    );

    // Arm the probe so the next evaluateAndAdjust pass after
    // config.probeWindowMs judges whether this move helped.
    if (suppressedBy === null) {
      this.probeController.arm({
        previousDPR,
        previousFPS: fps,
        probedDPR: newDPR,
        startTime: timestamp,
      });
    }

    // Notify callback
    if (this.onDPRChange) {
      this.onDPRChange(newDPR, this.isReducedResolution);
    }
  }

  /**
   * Apply a just-decided ceiling demotion: clamp the operating DPR to
   * 1.0 in one step (this replaces the tick's multiplicative
   * scale-down — the clamp is usually the larger move).
   *
   * @param reason - Optional log line override; default describes the
   *   punished-ascent path.
   */
  private applyCeilingDemotion(fps: number, reason?: string): void {
    log.warning(
      Modules.ADAPTIVE_DPR,
      reason ??
        `Repeated punished ascents above DPR 1.0 (FPS ${fps.toFixed(1)}) — ` +
          'ceiling demoted to 1.0 for this session (TTL-decayed; content changes re-check)'
    );
    if (this.currentDPR <= 1.0 + 0.001) return;

    this.currentDPR = 1.0;
    this.isReducedResolution = this.currentDPR < this.ceiling() * 0.95;
    this.applyDPR();

    if (this.onDPRChange) {
      this.onDPRChange(this.currentDPR, this.isReducedResolution);
    }
  }

  /**
   * Scale DPR up for better quality
   */
  private scaleUp(fps: number, timestamp: number): void {
    // Don't exceed the session ceiling (the display's DPR, capped by the
    // allow-high-DPR setting), nor a LEARNED demoted ceiling (evidence
    // says this scene can't sustain the above-1.0 luxury right now).
    //
    // With high DPR disallowed the two coincide at 1.0 and the learned
    // demotion becomes inert by construction — which is the point: this
    // setting is that same demotion, applied up front instead of after
    // the scene has spent seconds proving it at ~10fps.
    const ceiling = this.ceiling();
    const maxDPR = Math.min(ceiling, this.boundsLedger.dprCeiling ?? ceiling);
    const newDPR = Math.min(maxDPR, this.currentDPR * this.config.scaleUpFactor);

    // Only apply if there's a meaningful change and we're not at max
    if (Math.abs(newDPR - this.currentDPR) < 0.01) return;
    if (this.currentDPR >= maxDPR - 0.01) return;

    // An ascent above 1.0 is on probation: if FPS collapses within the
    // punishment window, it counts toward ceiling demotion.
    this.boundsLedger.recordAscent(newDPR, timestamp);
    this.scaleUpCount += 1;

    this.currentDPR = newDPR;
    this.applyDPR();

    // Update reduced resolution mode status
    this.isReducedResolution = newDPR < this.ceiling() * 0.95;

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
      // Reset to the LIVE ceiling when disabled — sync first because the
      // display may have changed since construction (monitor drag,
      // browser zoom). The ceiling, not the native DPR: turning
      // adaptation off means "stop lowering quality for FPS", not
      // "ignore the allow-high-DPR setting".
      this.syncNativeDPR();
      const ceiling = this.ceiling();
      this.currentDPR = ceiling;
      this.applyDPR();
      this.isReducedResolution = false;
      this.hysteresis.clear();
      this.probeController.void_();
      this.boundsLedger.reset();
      this.refreshRateEstimator.clear();
      this.fpsTracker.clear();
      this.stallDetector.clear();
      this.nonStallIntervals = 0;
      this.restingAtCeiling = false;
      this.lastOperatingDPR = null;

      if (this.onDPRChange) {
        this.onDPRChange(ceiling, false);
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
      refreshRateCap: this.refreshRateEstimator.getCap(),
      dprCeiling: Math.min(this.ceiling(), this.boundsLedger.dprCeiling ?? nativeDPR),
      allowHighDPR: isHighDPRAllowed(),
    };
  }

  /**
   * `getState()` plus the decision history: last suppression cause, ledger
   * backoff level, scale counts and the recent probe verdicts. Diagnostics
   * only — nothing in the controller reads these back.
   */
  getDiagnostics(): AdaptiveDPRDiagnostics {
    return {
      ...this.getState(),
      suppressedBy: this.lastSuppressedBy,
      backoffLevel: this.boundsLedger.backoffLevel,
      scaleDowns: this.scaleDownCount,
      scaleUps: this.scaleUpCount,
      verdicts: this.probeRecords.map((record) => ({ ...record })),
    };
  }

  private recordProbeVerdict(
    verdict: Exclude<ProbeVerdict, { kind: 'pending' }>,
    currentFPS: number,
    timestamp: number
  ): void {
    const { probe } = verdict;
    const settled = verdict.kind !== 'inconclusive';
    const record: AdaptiveDPRProbeRecord = {
      kind: settled && probe.contentConfounded ? 'confounded' : verdict.kind,
      previousDPR: probe.previousDPR,
      probedDPR: probe.probedDPR,
      previousFPS: probe.previousFPS,
      currentFPS: settled ? currentFPS : null,
      fpsRatio: settled ? verdict.fpsRatio : null,
      timestamp,
    };
    if (this.probeRecords.length >= PROBE_RECORD_LIMIT) this.probeRecords.shift();
    this.probeRecords.push(record);
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
   * @param dpr - Device pixel ratio to set (clamped to 0.25 - the session
   *   ceiling, i.e. the display's DPR capped by the allow-high-DPR
   *   setting)
   */
  setManualDPR(dpr: number): void {
    if (this.isEnabled) {
      log.warning(
        Modules.ADAPTIVE_DPR,
        'Cannot set manual DPR while adaptive resolution is enabled'
      );
      return;
    }

    // Clamp against the LIVE ceiling. The manual slider is an
    // interactive control, so the cap binds here too — the setting is a
    // hard ceiling for everything on screen, and the slider's own range
    // is built from the same number (see performance-setup).
    this.syncNativeDPR();
    const ceiling = this.ceiling();
    const minDPR = 0.25;
    const clampedDPR = clamp(dpr, minDPR, ceiling);

    // DPR changes force renderer/post-processing target reallocations, so
    // avoid repeating that expensive path for duplicate slider/input events.
    if (Math.abs(clampedDPR - this.currentDPR) < 0.01) return;

    this.currentDPR = clampedDPR;
    this.isReducedResolution = clampedDPR < ceiling * 0.95;
    this.applyDPR();

    log.info(Modules.ADAPTIVE_DPR, `Manual DPR set to ${clampedDPR.toFixed(2)}`);
  }

  /**
   * Allow or forbid rendering above CSS resolution (DPR 1.0).
   *
   * The runtime face of `renderingControls.defaults.allowHighDPR` /
   * `viewer_config.allow_high_dpr`. Writes the shared pixel-ratio cap
   * (see `rendering/pixel-ratio-cap`) and then re-settles the operating
   * DPR so the click has a visible effect immediately rather than after
   * a hysteresis window:
   *
   * - Turning it ON while ADAPTIVE: jump straight to the new ceiling.
   *   The user asked to see HiDPI; if the scene cannot sustain it the
   *   loop walks back down within a probe window, which is honest
   *   feedback rather than a silent no-op.
   * - Turning it ON while MANUAL: widen the range only, leave the
   *   chosen DPR alone. In manual mode the user's number is
   *   authoritative and must not be moved out from under them.
   * - Turning it OFF: clamp down at once in both modes. A cap that does
   *   not bind immediately is not a cap.
   *
   * No-op while pinned, mirroring `setEnabled` — a `?dpr=` session is
   * deliberately immune to persisted per-scene settings.
   */
  setHighDPRAllowed(allowed: boolean): void {
    if (this.pinned) {
      log.info(
        Modules.ADAPTIVE_DPR,
        `Ignoring setHighDPRAllowed(${allowed}) — DPR is pinned for this session (?dpr= URL param)`
      );
      return;
    }
    if (isHighDPRAllowed() === allowed) return;

    // Remembered across the cap write: syncNativeDPR's rebase settles a
    // session that was tracking the ceiling ONTO the new ceiling, which
    // is right for adaptive mode but would move a manual value the user
    // chose deliberately.
    const beforeDPR = this.currentDPR;

    setHighDPRAllowed(allowed);
    // Does the real work: detects the cap change, clears bounds/probe/FPS
    // state calibrated for the old ceiling, and re-applies.
    this.syncNativeDPR();

    const ceiling = this.ceiling();
    const target = this.isEnabled ? ceiling : Math.min(beforeDPR, ceiling);
    if (Math.abs(target - this.currentDPR) >= 0.01) {
      this.currentDPR = target;
      this.applyDPR();
    }
    this.isReducedResolution = this.currentDPR < ceiling * 0.95;

    log.info(
      Modules.ADAPTIVE_DPR,
      `High DPR ${allowed ? 'allowed' : 'disallowed'} ` +
        `(ceiling ${ceiling.toFixed(2)}, DPR ${this.currentDPR.toFixed(2)})`
    );

    if (this.onDPRChange) {
      this.onDPRChange(this.currentDPR, this.isReducedResolution);
    }
  }

  /** Whether the viewer may currently render above CSS resolution. */
  isHighDPRAllowed(): boolean {
    return isHighDPRAllowed();
  }

  /**
   * Pin a fixed manual DPR for the whole session (`?dpr=` URL param).
   *
   * Disables adaptive mode, applies `dpr` as a manual DPR, and locks the
   * enabled state: subsequent `setEnabled()` / `setHighDPRAllowed()`
   * calls (persisted per-scene settings, the Performance toggles, scene
   * metadata) are ignored for the rest of the session. Intended for
   * deterministic E2E/visual-regression runs and repros.
   *
   * An explicit pin RAISES the pixel-ratio cap to the pinned value if
   * needed, so `?dpr=2` renders at 2 even with high DPR disallowed —
   * otherwise the seam would silently clamp the pin to 1.0 and the
   * parameter would look broken. The pin is still bounded by the
   * display: `?dpr=4` on a 2x screen pins 2.
   *
   * @param dpr - Device pixel ratio to pin
   */
  pinManualDPR(dpr: number): void {
    this.setEnabled(false);
    if (Number.isFinite(dpr) && dpr > getMaxPixelRatioCap()) {
      setMaxPixelRatioCap(dpr);
    }
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
   * Inject a predicate polled at each evaluation to detect active data
   * loading (decode/upload jank). While it returns true, FPS samples
   * are treated as unrepresentative: scale-downs still apply (fewer
   * pixels help a janky load too) but no probes are armed or settled
   * and the refresh-cap estimator is not fed — load jank must never
   * become learned floor/cap evidence. Pass null to disable.
   */
  setLoadActivityPredicate(predicate: (() => boolean) | null): void {
    this.loadActivityPredicate = predicate;
  }

  /**
   * Notify the manager that the animation loop stopped (idle pause,
   * tab hide, dispose). Clears SESSION state only — the FPS window,
   * the scale-up streak, any in-flight probe (voided unjudged: its
   * before/after comparison would otherwise span the pause and compare
   * workloads minutes apart), and the estimator's sample-stream
   * transients (recent window, uniform-low plateau clock, unconsumed
   * distress latch — see noteSessionInterrupted). LEARNED state
   * (floor, backoff streak, refresh-cap mark, throttle verdict)
   * survives: it is expensive evidence about this scene on this
   * display, and wiping it here would replay a full rejected-probe
   * episode on every interaction burst.
   *
   * Idempotent and safe after dispose() (stopAnimation is also called
   * from the controller's dispose path).
   */
  notifyPaused(): void {
    this.fpsTracker.clear();
    this.stallDetector.clear();
    this.nonStallIntervals = 0;
    this.hysteresis.clear();
    this.probeController.void_();
    // The estimator's SESSION transients (recent window, uniform-low
    // plateau clock, an unconsumed distress latch) describe the frame
    // stream the pause just broke — clear them so pause dead time
    // never counts toward a "sustained" verdict and a stale latch
    // can't demote a session that resumes light. Its LEARNED state
    // (high-water mark, throttle verdict, proven rate) survives.
    this.refreshRateEstimator.noteSessionInterrupted();
  }

  /**
   * Restore full quality for the resting frame, just before the loop
   * idle-pauses. The static image the user is about to study should be
   * as sharp as this session allows — reduced DPR only ever traded
   * quality for interaction smoothness, and there is no interaction
   * anymore.
   *
   * "Full quality" is the session CEILING, not the native DPR: with high
   * DPR disallowed, resting at native would both contradict the setting
   * and pay a 4x-pixel render-target reallocation (plus a visible
   * sharpen/soften pop) on every idle/resume cycle.
   *
   * Mutates OPERATING state only (never the floor/backoff — see
   * notifyPaused). Remembers the operating DPR so notifyResumed() can
   * snap straight back.
   *
   * @returns true when the DPR actually changed — the caller must then
   *   render one frame, because the resize clears the canvas.
   */
  prepareIdleFrame(): boolean {
    if (!this.isEnabled) return false;
    this.syncNativeDPR();
    const ceiling = this.ceiling();
    if (this.currentDPR >= ceiling - 0.01) return false;

    this.lastOperatingDPR = this.currentDPR;
    this.currentDPR = ceiling;
    this.restingAtCeiling = true;
    this.isReducedResolution = false;
    this.applyDPR();

    log.info(
      Modules.ADAPTIVE_DPR,
      `Idle: restored DPR to the ceiling ${ceiling.toFixed(2)} for the resting frame ` +
        `(operating DPR ${this.lastOperatingDPR.toFixed(2)} remembered for resume)`
    );

    if (this.onDPRChange) {
      this.onDPRChange(this.currentDPR, this.isReducedResolution);
    }
    return true;
  }

  /**
   * The loop is starting again after an idle rest: snap straight back
   * to the remembered operating DPR in ONE step (clamped to the live
   * ceiling). Without this, every interaction burst after an idle
   * restore would re-discover the reduction reactively — a cascade of
   * scale-downs, probes, and render-target reallocations.
   */
  notifyResumed(): void {
    if (!this.isEnabled || !this.restingAtCeiling) return;
    this.restingAtCeiling = false;

    this.syncNativeDPR();
    const ceiling = this.ceiling();
    const target = Math.min(
      this.lastOperatingDPR ?? ceiling,
      this.boundsLedger.dprCeiling ?? ceiling,
      ceiling
    );
    this.lastOperatingDPR = null;
    if (Math.abs(target - this.currentDPR) < 0.01) return;

    this.currentDPR = target;
    this.isReducedResolution = target < ceiling * 0.95;
    this.applyDPR();

    log.info(
      Modules.ADAPTIVE_DPR,
      `Resume: snapped back to operating DPR ${target.toFixed(2)} in one step`
    );

    if (this.onDPRChange) {
      this.onDPRChange(this.currentDPR, this.isReducedResolution);
    }
  }

  /**
   * Notify the manager that scene content genuinely changed (dataset
   * loaded, layers added/removed, LOD level swapped in). Learned
   * bounds describe the OLD content, so their expiry is pulled forward
   * to at most `contentChangeRecheckMs` from now and the rejection
   * backoff streak resets — a re-probe against the new content is
   * cheap and justified. Calls are coalesced within
   * `contentChangeRecheckMs` so event bursts (per-frame LOD swaps
   * during a zoom) don't spam the ledger.
   *
   * Unlike a pause or a display change this is NOT a frame-stream
   * boundary — see the cadence-memory note in the body.
   *
   * @param timestamp - Caller-supplied clock for tests; defaults to
   *   `performance.now()`, the same clock the frame loop feeds.
   */
  notifyContentChanged(timestamp: number = performance.now()): void {
    if (!this.isEnabled) return;
    // Every notification contaminates an in-flight probe's before/after
    // comparison, whether or not THIS call is the one acted upon — the
    // LOD level swapped either way. So a pending probe is marked ahead of
    // the coalescing early-return, otherwise a probe that happened to be
    // armed and settled BETWEEN two acted-upon calls would be mistaken
    // for a clean experiment and could revert the DPR upward and pin a
    // 30s floor on it.
    this.probeController.markContentConfounded();
    if (
      this.lastContentChangeAt !== null &&
      timestamp - this.lastContentChangeAt < this.config.contentChangeRecheckMs
    ) {
      return;
    }
    this.lastContentChangeAt = timestamp;
    this.boundsLedger.softenForContentChange(timestamp, this.config.contentChangeRecheckMs);
    // The estimator's throttle-downshift permission is likewise scoped
    // to the current content: proof of a high rate earned on the OLD
    // content must not license mis-capping the NEW one as "throttled".
    this.refreshRateEstimator.noteContentChanged();
    // Voiding an in-flight probe is normally right — its baseline was
    // measured on the OLD content, the classic confounded before/after —
    // but only when the loop can actually RUN the replacement
    // experiment. A probe needs frames inside `probeWindowMs` to settle,
    // and content changes recur at most once per
    // `contentChangeRecheckMs`; below roughly two frames per probe
    // window, every armed probe is voided before it can ever be judged.
    // Measured at 0.5fps (native 2) with a content change every frame:
    // every probe was voided and no verdict of any kind was ever
    // produced. A verdict contaminated by a content change beats never
    // having one, so a slow loop keeps its probe — but only as a
    // CONFOUNDED one: its before/after comparison spans two different
    // scenes, so the settle keeps the reduction and learns nothing, never
    // acting on the verdict's direction (see applyConfoundedVerdict).
    // That is what the gate buys, and it is not a stop to the walk — the
    // pixel ratio still descends unratified while content churns. It
    // buys the absence of THRASH: measured over 20 minutes of per-frame
    // churn at 0.33fps (native 2), acting on those verdicts produced 100
    // reductions and 98 reverts back UP (202 renderer pixel-ratio
    // applications, each a render-target reallocation), while discarding
    // them produced 13 monotone reductions and no revert at all. An
    // unknown rate (fewer than two samples, `getFPS()` = 0) keeps the
    // void: nothing says the loop is slow, and that is the historical
    // behaviour.
    const fps = this.fpsTracker.getFPS();
    const framesPerProbeWindow = (fps * this.config.probeWindowMs) / 1000;
    if (fps <= 0 || framesPerProbeWindow >= MIN_FRAMES_TO_RERUN_PROBE) {
      this.probeController.void_();
    }
    // The FPS window still holds old-content frames (a mixed window
    // would instantly re-prove the rate and defeat the scoping above)
    // and the scale-up streak was earned on the old workload, so both go
    // — exactly like notifyPaused; learned bounds were already softened.
    //
    // The stall detector's CADENCE MEMORY deliberately stays, unlike at
    // pause / display change / disable. Those interrupt the frame STREAM
    // and are followed by resume dead time; a content change does not —
    // frames keep arriving at whatever rate they were arriving at. And
    // wiping the cadence there is actively harmful: the detector's
    // cold-memory fallback is the absolute `gapResetMs` floor alone, so
    // for any loop slower than 1000/gapResetMs fps (2.9fps at the 350ms
    // default) the very NEXT interval — a perfectly ordinary frame at
    // the scene's own rate — is misread as dead time. That fires a gap
    // reset which clears the FPS window a second time and voids the
    // in-flight probe the gate above just protected, which is why the
    // gate is inert without this. A stale cadence is self-correcting
    // (the median follows the new content within two or three
    // intervals — the same convergence any genuine slowdown relies on),
    // so the trust counter is not reset here either.
    this.fpsTracker.clear();
    this.hysteresis.clear();
    log.info(
      Modules.ADAPTIVE_DPR,
      'Content changed — floor re-probe allowed within ' +
        `${(this.config.contentChangeRecheckMs / 1000).toFixed(0)}s, backoff streak reset`
    );
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
    this.stallDetector.clear();
    this.nonStallIntervals = 0;
    this.onDPRChange = null;
    this.renderer = null;
    log.info(Modules.ADAPTIVE_DPR, 'Disposed');
  }
}
