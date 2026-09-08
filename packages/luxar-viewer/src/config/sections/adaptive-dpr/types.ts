/**
 * Adaptive pixel ratio configuration for dynamic performance optimization
 *
 * This system dynamically adjusts the device pixel ratio based on real-time FPS
 * to maintain smooth frame rates during heavy rendering. Uses hysteresis to
 * prevent rapid toggling between quality levels, a probe-and-verify step for
 * every scale-down (U-shape awareness), and learned floor/ceiling bounds with
 * exponential backoff so hopeless probes are not repeated forever.
 */
export interface AdaptiveDPRConfig {
  /** Enable adaptive DPR system at construction time (default: true).
   *  Runtime toggle is via renderingControls.defaults.adaptiveDPREnabled. */
  enabled: boolean;
  /** Minimum allowed DPR - lower bound before image becomes too pixelated (default: 0.5) */
  minDPR: number;
  /** Factor to multiply DPR when scaling down (default: 0.9 = 10% reduction) */
  scaleDownFactor: number;
  /** Factor to multiply DPR when scaling up (default: 1.05 = 5% increase) */
  scaleUpFactor: number;
  /** Seconds FPS must stay above the scale-up threshold before scaling up (default: 3) */
  hysteresisSeconds: number;
  /** How often to evaluate FPS and adjust DPR in milliseconds (default: 500) */
  evaluationIntervalMs: number;

  // ── Refresh-rate-relative thresholds ────────────────────────────
  /** Scale down when fps < ratio × estimated refresh cap (default: 0.75) */
  scaleDownFpsRatio: number;
  /** Count toward scale-up when fps > ratio × estimated refresh cap (default: 0.90) */
  scaleUpFpsRatio: number;
  /** Refresh cap assumed before the estimator has warmed up, in Hz
   *  (default: 60). Also drives the indicator's displayed target FPS. */
  refreshRateFallback: number;
  /** Upper bound on the estimated refresh cap, in Hz; `0` = no bound
   *  (default). The mobile runtime sets 60 so a 120 Hz iPad's steady 60 fps
   *  is not read as distress (75% of a learned 120 Hz mark is 90 fps). */
  refreshRateCeiling: number;
  /** Consecutive mid-band samples tolerated before the scale-up streak
   *  resets (default: 1 — a single dropped-frame sample no longer
   *  restarts the whole hysteresis wait). */
  midbandGraceSamples: number;

  // ── U-shape probe ───────────────────────────────────────────────
  /** How long after a scale-down before judging its effect, ms (default: 1500).
   *  Must exceed the 1s FPS sample window so reallocation jank washes out. */
  probeWindowMs: number;
  /** Required relative FPS improvement for a scale-down to be kept
   *  (default: 1.05 = +5%); below this the move is reverted and floored. */
  probeImprovement: number;
  /** Minimum frame samples required to settle a probe (default: 8);
   *  prevents judging a probe on 2 janky post-resume frames. */
  probeMinSamples: number;

  // ── Learned floor (and its backoff) ─────────────────────────────
  /** How long a rejected-probe floor stays sticky before a re-probe is
   *  allowed, ms (default: 30_000). First rung of the backoff ladder. */
  floorTtlMs: number;
  /** TTL multiplier applied per consecutive identical rejection
   *  (default: 2 — 30s → 60s → 2min → ...). */
  backoffMultiplier: number;
  /** Ceiling for the backed-off TTL, ms (default: 300_000 = 5 min). */
  backoffMaxTtlMs: number;

  // ── Evidence-based DPR ceiling ──────────────────────────────────
  /** How long a demoted ceiling (native → 1.0) stays before it lifts,
   *  ms (default: 60_000). Backed off like the floor on re-demotion. */
  ceilingTtlMs: number;
  /** A scale-up above DPR 1.0 followed by an FPS collapse within this
   *  window counts as a "punished ascent", ms (default: 3000). */
  punishedAscentWindowMs: number;
  /** Punished ascents required to demote the ceiling to 1.0 (default: 2). */
  punishedAscentThreshold: number;

  // ── Session hygiene ─────────────────────────────────────────────
  /** Content-change notifications shorten floor/ceiling expiry to at
   *  most this far in the future, and are coalesced within it, ms
   *  (default: 5000). */
  contentChangeRecheckMs: number;
  /** Absolute floor for treating a gap between frames as DEAD TIME, ms
   *  (default: 350). Dead time resets the FPS window and voids any
   *  pending probe — it covers GC/decode stalls and idle-resume gaps
   *  that would otherwise poison samples.
   *
   *  This threshold is necessary but NOT sufficient: the interval must
   *  ALSO be a large outlier (strictly more than 4×) against the median
   *  of the four PRECEDING inter-frame intervals (see
   *  rendering/adaptive-dpr/stall-detector.ts).
   *  A 5s gap in a 60fps stream is a 300× outlier and resets; a 2s
   *  interval in a stream whose recent intervals are all ~2s is simply
   *  the frame rate and is kept, so the manager still scales down below
   *  ~1000/gapResetMs fps instead of going structurally inert there.
   *  A genuine slowdown costs one or two misread intervals while the
   *  median follows the new cadence.
   *
   *  Residual limitations, both from seeing only inter-frame intervals:
   *  a dead period ALTERNATING one-for-one with a SINGLE fast frame
   *  (~2s / ~100ms / ~2s / ~100ms) makes the dead intervals half of the
   *  four-interval memory, so the median lands between the phases and
   *  they are kept as "the frame rate" — the FPS window then mixes real
   *  dead time with render cost. The manager still adapts (measured, the
   *  reported rate is far below the down threshold either way) but it
   *  cannot report the rate the user perceives. Two or more fast frames
   *  between dead periods fail the other way: the median stays fast, the
   *  dead time is correctly discarded every cycle, and the window
   *  accumulates nothing but those few fast frames — so a recurring
   *  hitch pattern like 16.7/16.7/400ms reads a healthy 60fps at ~6.9
   *  perceived fps. */
  gapResetMs: number;
}
