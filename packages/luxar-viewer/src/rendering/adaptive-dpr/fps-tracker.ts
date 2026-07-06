/**
 * FPS tracker — sliding-window frame-rate estimation for the adaptive
 * DPR control loop.
 *
 * Pure and timestamp-driven: no `window`, no config imports, no clocks
 * of its own. The caller supplies every timestamp (production feeds
 * `performance.now()` from the animation loop; tests feed literals).
 *
 * Mechanics: timestamps are appended to an array that is trimmed with
 * an advancing start index (O(1) amortized) and compacted periodically
 * to bound memory. FPS is interval-normalized — (samples − 1) intervals
 * over the observed span — so it is not quantized to integer frame
 * counts.
 */
export class FPSTracker {
  private timestamps: number[] = [];
  private startIndex = 0;

  /**
   * @param windowMs - Sliding sample window (frames older than
   *   `newest - windowMs` fall out of the estimate)
   */
  constructor(private readonly windowMs: number) {}

  /** Record a frame timestamp and trim samples older than the window. */
  push(timestamp: number): void {
    this.timestamps.push(timestamp);

    const cutoff = timestamp - this.windowMs;
    while (this.startIndex < this.timestamps.length && this.timestamps[this.startIndex] < cutoff) {
      this.startIndex++;
    }

    // Compact periodically to prevent unbounded growth.
    if (this.startIndex > 120) {
      this.timestamps = this.timestamps.slice(this.startIndex);
      this.startIndex = 0;
    }
  }

  /**
   * Interval-normalized FPS over the current window; 0 when fewer than
   * two samples (or a degenerate non-positive span) — callers treat 0
   * as "not enough data to decide".
   */
  getFPS(): number {
    const count = this.sampleCount();
    if (count < 2) return 0;

    const span = this.span();
    if (span <= 0) return 0;

    return ((count - 1) * 1000) / span;
  }

  /** Number of samples currently inside the window. */
  sampleCount(): number {
    return this.timestamps.length - this.startIndex;
  }

  /** Time covered by the in-window samples, ms (0 with <2 samples). */
  span(): number {
    if (this.sampleCount() < 2) return 0;
    return this.timestamps[this.timestamps.length - 1] - this.timestamps[this.startIndex];
  }

  /** Newest recorded timestamp, or null when the window is empty. */
  get lastTimestamp(): number | null {
    return this.sampleCount() > 0 ? this.timestamps[this.timestamps.length - 1] : null;
  }

  /** Drop all samples (pause, native-DPR change, gap reset). */
  clear(): void {
    this.timestamps = [];
    this.startIndex = 0;
  }
}
