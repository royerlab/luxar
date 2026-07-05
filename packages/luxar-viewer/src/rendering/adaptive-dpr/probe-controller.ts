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
  | { kind: 'rejected'; probe: PendingProbe; fpsRatio: number }
  /**
   * No clean sample arrived within the extended window (few frames,
   * short span, or data-loading jank the whole time) — the experiment
   * is inconclusive. The DPR stays as-is and NOTHING is learned: no
   * revert, no floor.
   */
  | { kind: 'inconclusive'; probe: PendingProbe };

/** Sample-quality snapshot the manager passes at each evaluation. */
export interface ProbeSampleQuality {
  /** Frames currently in the FPS window. */
  sampleCount: number;
  /** Time covered by those frames, ms. */
  spanMs: number;
  /** True while data loading makes FPS samples unrepresentative. */
  suppressed: boolean;
}

export interface ProbeControllerConfig {
  /** How long after arming before the probe may settle, ms. */
  windowMs: number;
  /** Required relative FPS improvement (e.g. 1.05 = +5%) to accept. */
  improvement: number;
  /** Minimum frames required in the window to judge the probe. */
  minSamples: number;
  /** Minimum window span required to judge the probe, ms. */
  minSpanMs: number;
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
   *
   * Quality gating: past the probe window, the verdict is only settled
   * from a CLEAN sample — enough frames, enough span, no data-loading
   * suppression. Judging a probe on 2 janky post-stall frames is how
   * wrong floors get learned. While unclean, the probe keeps waiting up
   * to 2× the window; past that it is voided as inconclusive (DPR
   * stays, nothing learned).
   */
  evaluate(
    timestamp: number,
    currentFPS: number,
    quality: ProbeSampleQuality
  ): ProbeVerdict | null {
    if (!this.pending) return null;

    const age = timestamp - this.pending.startTime;
    if (age < this.config.windowMs) {
      return { kind: 'pending' };
    }

    const clean =
      !quality.suppressed &&
      quality.sampleCount >= this.config.minSamples &&
      quality.spanMs >= this.config.minSpanMs;
    if (!clean) {
      if (age < 2 * this.config.windowMs) {
        return { kind: 'pending' }; // keep waiting for a clean sample
      }
      const probe = this.pending;
      this.pending = null;
      return { kind: 'inconclusive', probe };
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
