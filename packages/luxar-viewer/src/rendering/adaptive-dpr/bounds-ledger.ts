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
 * - Content changes soften the ledger: expiry is pulled forward so a
 *   re-probe happens within `recheckMs`, and the escalation streak
 *   resets — new content deserves fresh evidence.
 *
 * The evidence-based ceiling (native → 1.0 demotion) joins the ledger
 * with the ceiling follow-up work.
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
}

export class BoundsLedger {
  private floor: number;
  private floorExpiresAt = 0;
  private floorBackoffLevel = 0;
  private lastRejectedProbeDPR: number | null = null;

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

  /**
   * Scene content changed: pull the floor's expiry forward to at most
   * `recheckMs` from now and reset the escalation streak — the old
   * evidence described different content.
   */
  softenForContentChange(timestamp: number, recheckMs: number): void {
    if (this.floor > this.config.minDPR) {
      this.floorExpiresAt = Math.min(this.floorExpiresAt, timestamp + recheckMs);
    }
    this.floorBackoffLevel = 0;
    this.lastRejectedProbeDPR = null;
  }

  /** Forget everything learned (disable, native-DPR change). */
  reset(): void {
    this.floor = this.config.minDPR;
    this.floorExpiresAt = 0;
    this.floorBackoffLevel = 0;
    this.lastRejectedProbeDPR = null;
  }
}
