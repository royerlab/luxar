/**
 * Bounds ledger — the adaptive DPR manager's learned operating bounds.
 *
 * Where `minDPR`/native are STATIC bounds, the ledger holds
 * EVIDENCE-BASED ones learned at runtime:
 *
 * - Floor: the lowest DPR known not to help (set when a U-shape probe
 *   is rejected). Scale-down never moves to-or-below it. Decays after
 *   a TTL so probes eventually retry when content may have changed.
 *
 * Ceiling (native → 1.0 demotion after punished ascents) joins the
 * ledger with the control-loop follow-up work.
 *
 * Pure and timestamp-driven; no clocks, no window, no config imports.
 */

export interface BoundsLedgerConfig {
  /** Static lower bound — the floor decays back to this. */
  minDPR: number;
  /** How long a rejected-probe floor stays sticky, ms. */
  floorTtlMs: number;
}

export class BoundsLedger {
  private floor: number;
  private floorSetAt = 0;

  constructor(private readonly config: BoundsLedgerConfig) {
    this.floor = config.minDPR;
  }

  /** Lowest DPR scale-down will currently accept. */
  get dprFloor(): number {
    return this.floor;
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

  /** Record a rejected probe: tighten the floor to the probed DPR. */
  recordRejection(probedDPR: number, timestamp: number): void {
    this.floor = probedDPR;
    this.floorSetAt = timestamp;
  }

  /**
   * Let an expired floor decay back to minDPR. Returns true when the
   * floor actually lifted (callers log on that edge).
   */
  decayIfExpired(timestamp: number): boolean {
    if (this.floor > this.config.minDPR && timestamp - this.floorSetAt > this.config.floorTtlMs) {
      this.floor = this.config.minDPR;
      return true;
    }
    return false;
  }

  /** Forget everything learned (disable, native-DPR change). */
  reset(): void {
    this.floor = this.config.minDPR;
    this.floorSetAt = 0;
  }
}
