/**
 * U-shape probe controller — verifies that a DPR scale-down actually
 * improved FPS before letting it stand.
 *
 * Lowering DPR is not monotonically faster (compositor upscale cost
 * grows with the DPR mismatch), and on CPU/draw-call-bound scenes it
 * does nothing at all. Every scale-down therefore arms a probe: after
 * `windowMs` the post-change FPS is compared against the pre-change
 * baseline, and unless it improved by `improvement` the move is judged
 * a rejection (the caller reverts and records a floor).
 *
 * Pure and timestamp-driven; owns only the pending-probe record. The
 * caller applies verdicts (revert, floor bookkeeping, callbacks).
 */

/** Snapshot taken when a scale-down is applied. */
export interface PendingProbe {
  /** DPR before the scale-down (revert target on rejection). */
  previousDPR: number;
  /** FPS baseline measured at decision time. */
  previousFPS: number;
  /** DPR the scale-down moved to. */
  probedDPR: number;
  /** Timestamp the probe was armed. */
  startTime: number;
}

export type ProbeVerdict =
  /** Probe window still open — keep gathering samples, decide nothing. */
  | { kind: 'pending' }
  /** The move helped (>= improvement) — keep the new DPR. */
  | { kind: 'accepted'; probe: PendingProbe; fpsRatio: number }
  /** The move did not help — revert to previousDPR and floor probedDPR. */
  | { kind: 'rejected'; probe: PendingProbe; fpsRatio: number };

export interface ProbeControllerConfig {
  /** How long after arming before the probe may settle, ms. */
  windowMs: number;
  /** Required relative FPS improvement (e.g. 1.05 = +5%) to accept. */
  improvement: number;
}

export class ProbeController {
  private pending: PendingProbe | null = null;

  constructor(private readonly config: ProbeControllerConfig) {}

  /** Whether a probe is currently in flight. */
  get isPending(): boolean {
    return this.pending !== null;
  }

  /** Arm a probe for a just-applied scale-down. */
  arm(probe: PendingProbe): void {
    this.pending = probe;
  }

  /**
   * Drop the pending probe without judging it (pause, native-DPR
   * change, disable). The caller keeps the current DPR as-is so the
   * manager and renderer never diverge; no floor is learned.
   */
  void_(): void {
    this.pending = null;
  }

  /**
   * Evaluate the pending probe at `timestamp` against `currentFPS`.
   * Returns null when no probe is in flight. A non-pending verdict
   * clears the probe; the caller applies its consequences.
   */
  evaluate(timestamp: number, currentFPS: number): ProbeVerdict | null {
    if (!this.pending) return null;

    if (timestamp - this.pending.startTime < this.config.windowMs) {
      return { kind: 'pending' };
    }

    const probe = this.pending;
    this.pending = null;

    const fpsRatio = probe.previousFPS > 0 ? currentFPS / probe.previousFPS : 0;
    if (fpsRatio >= this.config.improvement) {
      return { kind: 'accepted', probe, fpsRatio };
    }
    return { kind: 'rejected', probe, fpsRatio };
  }
}
