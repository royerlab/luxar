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
  /** A gap between frames larger than this resets the FPS window and
   *  voids any pending probe, ms (default: 350). Covers GC/decode
   *  stalls and idle-resume gaps that would otherwise poison samples.
   *  Trade-off: below ~1000/gapResetMs fps the manager holds state
   *  instead of adapting (adaptivity is moot that slow anyway). */
  gapResetMs: number;
}
