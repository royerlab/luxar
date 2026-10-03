/**
 * Sliding-window bandwidth tracker.
 *
 * Records timestamped byte counts on `record(bytes)` and returns the average
 * bytes/sec over the trailing `windowMs` on `rate()`.
 *
 * Implementation note (R5 — load-bearing):
 * Naive pruning via Array.shift() is O(n) per pop and degrades to O(n²)
 * under sustained high fetch rates. Instead both `record()` and `rate()`
 * advance a `start` index past expired entries (O(1) amortized), and only
 * when the dead prefix exceeds half the buffer does `record()` slice it off
 * in one allocation. Expiring on `record()` too is what bounds the buffer:
 * `rate()` is read only while the monitor is visible.
 */
export class BandwidthWindow {
  private readonly windowMs: number;
  private window: { timestamp: number; bytes: number }[] = [];
  private start = 0;

  constructor(windowMs: number) {
    this.windowMs = windowMs;
  }

  /**
   * Record a network transfer of `bytes` at the current time. Expires aged-out
   * entries, then runs the R5 amortized compaction: if the dead prefix is
   * larger than the live tail, slice it off in one O(n) hit.
   */
  record(bytes: number): void {
    const now = Date.now();
    this.expire(now);
    this.window.push({ timestamp: now, bytes });
    if (this.start > 0 && this.start > this.window.length / 2) {
      this.window = this.window.slice(this.start);
      this.start = 0;
    }
  }

  /**
   * Returns current bytes/sec averaged over the trailing window. Also
   * advances `start` past any entries that have aged out — amortized O(1).
   */
  rate(): number {
    const now = Date.now();
    this.expire(now);

    const liveCount = this.window.length - this.start;
    if (liveCount === 0) return 0;

    let windowBytes = 0;
    for (let i = this.start; i < this.window.length; i++) {
      windowBytes += this.window[i].bytes;
    }
    const windowSpan = Math.max(1, (now - this.window[this.start].timestamp) / 1000);
    return windowBytes / windowSpan;
  }

  /** Advance `start` past every entry older than the window at `now`. */
  private expire(now: number): void {
    const cutoff = now - this.windowMs;
    while (this.start < this.window.length && this.window[this.start].timestamp < cutoff) {
      this.start++;
    }
  }
}
