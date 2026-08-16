/**
 * Bounds ledger — the adaptive DPR manager's learned operating bounds.
 *
 * Where `minDPR`/native are STATIC bounds, the ledger holds
 * EVIDENCE-BASED ones learned at runtime:
 *
 * - Floor: the lowest DPR known not to help (set when a U-shape probe
 *   is rejected). Scale-down never moves to-or-below it. Decays after
 *   a TTL — but repeated IDENTICAL rejections escalate the TTL
 *   exponentially (30s → 60s → 2min → capped), so a scene where DPR
 *   reduction never helps stops paying a probe/blur/revert cycle every
 *   30 seconds forever.
 * - Ceiling: on HiDPI displays, DPR above 1.0 is a luxury — 1.0 is
 *   exactly what every standard display renders — with a 4x fill cost
 *   at 2x native. When scale-ups above 1.0 keep getting PUNISHED (FPS
 *   collapses shortly after the ascent), or the estimator reports
 *   sustained sub-throttle DISTRESS (demoteCeiling called directly),
 *   the scene has proven it can't sustain the luxury: the operating
 *   ceiling demotes from native to exactly 1.0, killing the up/down
 *   oscillation instead of slowing it. Session-only; decays after a
 *   TTL (backed off on re-demotion).
 * - Content changes soften the ledger: expiries are pulled forward so
 *   re-probes/re-ascents happen within `recheckMs`, and the escalation
 *   streaks reset — new content deserves fresh evidence.
 *
 * Pure and timestamp-driven; no clocks, no window, no config imports.
 */

export interface BoundsLedgerConfig {
  /** Static lower bound — the floor decays back to this. */
  minDPR: number;
  /** First-rung TTL for a rejected-probe floor, ms. */
  floorTtlMs: number;
  /** TTL multiplier per consecutive identical rejection (>= 1). */
  backoffMultiplier: number;
  /** Ceiling for the backed-off TTL, ms. */
  backoffMaxTtlMs: number;
  /** First-rung TTL for a demoted DPR ceiling, ms. */
  ceilingTtlMs: number;
  /** An ascent punished within this window counts toward demotion, ms. */
  punishedAscentWindowMs: number;
  /** Punished ascents required to demote the ceiling to 1.0 (>= 1). */
  punishedAscentThreshold: number;
}

export class BoundsLedger {
  private floor: number;
  private floorExpiresAt = 0;
  private floorBackoffLevel = 0;
  private lastRejectedProbeDPR: number | null = null;

  private ceilingDemoted = false;
  private ceilingExpiresAt = 0;
  private ceilingBackoffLevel = 0;
  private punishedAscentCount = 0;
  private lastAscent: { dpr: number; timestamp: number } | null = null;

  constructor(private readonly config: BoundsLedgerConfig) {
    this.floor = config.minDPR;
  }

  /** Lowest DPR scale-down will currently accept. */
  get dprFloor(): number {
    return this.floor;
  }

  /** Current escalation level (0 = no active streak); for logs/tests. */
  get backoffLevel(): number {
    return this.floorBackoffLevel;
  }

  /**
   * Whether a proposed scale-down target is blocked by the floor.
   *
   * The to-or-below early-return (rather than clamping to the floor)
   * is load-bearing: after a rejection the next proposed step lands at
   * exactly the floored value, and clamping would re-fire the same
   * failing probe every evaluation tick — the TTL would never get a
   * chance to expire.
   */
  blocksScaleDownTo(proposedDPR: number): boolean {
    return proposedDPR <= this.floor + 0.001;
  }

  /**
   * Record a rejected probe: tighten the floor to the probed DPR.
   *
   * A rejection at (approximately) the same DPR as the previous one is
   * the same experiment failing again — escalate the TTL. A rejection
   * at a different DPR is new evidence and restarts the ladder.
   *
   * @returns the TTL applied, ms (for logging)
   */
  recordRejection(probedDPR: number, timestamp: number): number {
    const identical =
      this.lastRejectedProbeDPR !== null && Math.abs(probedDPR - this.lastRejectedProbeDPR) < 0.01;
    this.floorBackoffLevel = identical ? this.floorBackoffLevel + 1 : 1;
    this.lastRejectedProbeDPR = probedDPR;

    const ttl = Math.min(
      this.config.floorTtlMs * Math.pow(this.config.backoffMultiplier, this.floorBackoffLevel - 1),
      this.config.backoffMaxTtlMs
    );
    this.floor = probedDPR;
    this.floorExpiresAt = timestamp + ttl;
    return ttl;
  }

  /**
   * Record an accepted probe: the current regime responds to DPR
   * reduction after all, so the rejection streak is stale evidence.
   */
  recordAcceptance(): void {
    this.floorBackoffLevel = 0;
    this.lastRejectedProbeDPR = null;
  }

  /**
   * Let an expired floor decay back to minDPR. Returns true when the
   * floor actually lifted (callers log on that edge). The escalation
   * streak SURVIVES expiry — that memory is what makes the next
   * identical rejection back off harder.
   */
  decayIfExpired(timestamp: number): boolean {
    if (this.floor > this.config.minDPR && timestamp > this.floorExpiresAt) {
      this.floor = this.config.minDPR;
      return true;
    }
    return false;
  }

  // ── Ceiling (evidence-based native → 1.0 demotion) ─────────────

  /**
   * The current operating ceiling, or null when not demoted (the
   * caller then uses the live native DPR). By design the only demoted
   * value is exactly 1.0 — a principled Schelling point (CSS-pixel
   * resolution, what every 1x display renders), not a hunted estimate.
   */
  get dprCeiling(): number | null {
    return this.ceilingDemoted ? 1.0 : null;
  }

  /** Punished ascents accumulated toward demotion; for logs/tests. */
  get ascentPunishments(): number {
    return this.punishedAscentCount;
  }

  /**
   * Record a scale-up that moved the DPR above 1.0. If FPS collapses
   * shortly after, recordSlowSample() counts it as a punished ascent.
   *
   * A previous ascent still on record here has, by construction,
   * outlived its punishment window without a slow sample — evidence the
   * scene CAN sustain above-1.0, so the punishment tally is stale and
   * resets. Without this the count is a lifetime tally and two isolated
   * hiccups minutes apart would demote a perfectly HiDPI-capable scene.
   */
  recordAscent(dpr: number, timestamp: number): void {
    if (dpr <= 1.01) return;
    if (
      this.lastAscent &&
      timestamp - this.lastAscent.timestamp > this.config.punishedAscentWindowMs
    ) {
      this.punishedAscentCount = 0;
    }
    this.lastAscent = { dpr, timestamp };
  }

  /**
   * Record a below-down-threshold FPS sample (callers must NOT feed
   * load-suppressed samples — load jank is not the ascent's fault).
   * If it lands within the punishment window of a recorded ascent, the
   * ascent is punished; at the configured threshold the ceiling
   * demotes to 1.0 with a (backed-off) TTL.
   *
   * @returns true when the demotion happened on THIS call — the caller
   *   then clamps its operating DPR to 1.0 in one step.
   */
  recordSlowSample(timestamp: number): boolean {
    if (!this.lastAscent) return false;
    const withinWindow =
      timestamp - this.lastAscent.timestamp <= this.config.punishedAscentWindowMs;
    this.lastAscent = null;
    if (!withinWindow) {
      // The ascent outlived its punishment window before any slow
      // sample arrived — it was SUSTAINED. That is positive evidence
      // for above-1.0 viability; the punishment tally resets.
      this.punishedAscentCount = 0;
      return false;
    }

    this.punishedAscentCount++;
    if (this.punishedAscentCount < this.config.punishedAscentThreshold) return false;

    this.demoteCeiling(timestamp);
    return true;
  }

  /**
   * Demote the ceiling to 1.0 directly on outside evidence (sustained
   * sub-throttle distress — FPS too low to be any real display
   * throttle), bypassing the punished-ascent tally: the scene has
   * already proven it can't afford the above-1.0 luxury without any
   * ascent experiment. Same TTL/backoff ladder as an earned demotion —
   * a scene that keeps re-earning it holds it longer each time.
   *
   * @returns the TTL applied, ms (for logging)
   */
  demoteCeiling(timestamp: number): number {
    this.punishedAscentCount = 0;
    this.lastAscent = null;
    this.ceilingBackoffLevel++;
    const ttl = Math.min(
      this.config.ceilingTtlMs *
        Math.pow(this.config.backoffMultiplier, this.ceilingBackoffLevel - 1),
      this.config.backoffMaxTtlMs
    );
    this.ceilingDemoted = true;
    this.ceilingExpiresAt = timestamp + ttl;
    return ttl;
  }

  /**
   * Lift an expired ceiling demotion. The backoff level SURVIVES —
   * a scene that keeps re-earning the demotion holds it longer each
   * time. Returns true when the ceiling actually lifted.
   */
  decayCeilingIfExpired(timestamp: number): boolean {
    if (this.ceilingDemoted && timestamp > this.ceilingExpiresAt) {
      this.ceilingDemoted = false;
      return true;
    }
    return false;
  }

  /**
   * Scene content changed: pull the floor's and ceiling's expiry
   * forward to at most `recheckMs` from now and reset the escalation
   * streaks — the old evidence described different content.
   */
  softenForContentChange(timestamp: number, recheckMs: number): void {
    if (this.floor > this.config.minDPR) {
      this.floorExpiresAt = Math.min(this.floorExpiresAt, timestamp + recheckMs);
    }
    this.floorBackoffLevel = 0;
    this.lastRejectedProbeDPR = null;

    if (this.ceilingDemoted) {
      this.ceilingExpiresAt = Math.min(this.ceilingExpiresAt, timestamp + recheckMs);
    }
    this.ceilingBackoffLevel = 0;
    this.punishedAscentCount = 0;
    this.lastAscent = null;
  }

  /** Forget everything learned (disable, native-DPR change). */
  reset(): void {
    this.floor = this.config.minDPR;
    this.floorExpiresAt = 0;
    this.floorBackoffLevel = 0;
    this.lastRejectedProbeDPR = null;
    this.ceilingDemoted = false;
    this.ceilingExpiresAt = 0;
    this.ceilingBackoffLevel = 0;
    this.punishedAscentCount = 0;
    this.lastAscent = null;
  }
}
