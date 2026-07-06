/**
 * Scale-up hysteresis tracker — DPR only scales up after FPS has stayed
 * above the up-threshold for a sustained period, so a momentary breeze
 * of fast frames doesn't trigger a quality/perf oscillation.
 *
 * Adds a mid-band GRACE over the naive "any non-high sample resets the
 * timer" rule: on a 60Hz display two dropped frames in one window read
 * as 58.0 FPS (mid-band), and under the naive rule a machine dropping a
 * couple of frames every few seconds could never accumulate the full
 * hysteresis period — quality recovery stalled indefinitely. Up to
 * `graceSamples` consecutive mid-band samples are tolerated inside a
 * streak; only a longer wobble (or any below-down-threshold sample)
 * resets it.
 *
 * Pure and timestamp-driven; no clocks, no window, no config imports.
 */
export class HysteresisTracker {
  private streakStart: number | null = null;
  private midbandStrikes = 0;

  constructor(
    private readonly config: {
      /** Sustained high-FPS time required before firing, ms. */
      hysteresisMs: number;
      /** Consecutive mid-band samples tolerated inside a streak. */
      graceSamples: number;
    }
  ) {}

  /**
   * Record a sample above the up-threshold. Returns true when the
   * streak has lasted the full hysteresis period — the caller then
   * scales up and calls `clear()`.
   */
  recordHigh(timestamp: number): boolean {
    this.midbandStrikes = 0;
    if (this.streakStart === null) {
      this.streakStart = timestamp;
      return false;
    }
    return timestamp - this.streakStart >= this.config.hysteresisMs;
  }

  /**
   * Record an in-between sample (neither high enough to advance nor
   * low enough to scale down). Tolerated `graceSamples` times in a row;
   * beyond that the streak resets.
   */
  recordMidband(): void {
    if (this.streakStart === null) return;
    this.midbandStrikes++;
    if (this.midbandStrikes > this.config.graceSamples) {
      this.clear();
    }
  }

  /** Record a below-down-threshold sample: always resets the streak. */
  recordLow(): void {
    this.clear();
  }

  /** True while a scale-up streak is accumulating. */
  get active(): boolean {
    return this.streakStart !== null;
  }

  /** Reset the streak (after firing, on pause, on display change). */
  clear(): void {
    this.streakStart = null;
    this.midbandStrikes = 0;
  }
}
