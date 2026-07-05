/**
 * Refresh-rate estimator — the display's achievable rAF rate, learned
 * at runtime so DPR thresholds can be RELATIVE to it (75%/90% of the
 * cap) instead of hardcoded for 60Hz.
 *
 * Estimate = session high-water mark of window-FPS samples,
 * lower-bounded by a configured fallback until proven throttled:
 *
 * - The mark only grows (a heavy scene's low FPS must never drag the
 *   cap down — that would silently disable scale-down on exactly the
 *   scenes that need it, since `downThr = ratio × cap` would sink
 *   below the loaded FPS).
 * - Until the display proves it can beat the fallback, the cap is
 *   `max(mark, fallback)` so a session that STARTS heavy (mark stuck
 *   at the loaded FPS) still scales down against a sane 60Hz baseline.
 * - Genuine rAF throttling (30Hz low-power, background tabs) has a
 *   distinctive signature a heavy scene lacks the *persistence* of:
 *   every sample uniformly low for a sustained period. When the last
 *   `RECENT_SAMPLES` samples stay below `THROTTLE_FRACTION × cap` with
 *   < `UNIFORMITY_SPREAD` relative spread for `DOWNSHIFT_AFTER_MS`,
 *   the mark reseeds from them and the fallback lower bound is
 *   released — a 120→30Hz throttle re-converges instead of firing
 *   spurious scale-downs against a stale cap forever. A misclassified
 *   heavy scene merely loses scale-down until FPS varies again (the
 *   probe/backoff machinery already contains that regime), and any
 *   sample above 80% of the fallback immediately restores the bound.
 *
 * Pure and timestamp-driven; no clocks, no window, no config imports.
 */

/** Samples the throttle detector inspects for uniformity. */
const RECENT_SAMPLES = 4;
/** Recent samples must all sit below this fraction of the cap. */
const THROTTLE_FRACTION = 0.55;
/** Max relative spread (max−min)/max for "uniform" recent samples. */
const UNIFORMITY_SPREAD = 0.15;
/** How long the uniform-low signature must persist before downshifting. */
const DOWNSHIFT_AFTER_MS = 10_000;
/** A sample above this fraction of the fallback clears throttle mode. */
const UNTHROTTLE_FRACTION = 0.8;

export class RefreshRateEstimator {
  private mark = 0;
  private recent: number[] = [];
  private lowUniformSince: number | null = null;
  private throttled = false;

  constructor(private readonly fallback: number) {}

  /**
   * Feed one window-FPS sample (callers should only feed full-span,
   * unsuppressed windows so partial data doesn't pollute the mark).
   */
  addSample(fps: number, timestamp: number): void {
    if (fps > this.mark) this.mark = fps;
    if (fps >= this.fallback * UNTHROTTLE_FRACTION) this.throttled = false;

    this.recent.push(fps);
    if (this.recent.length > RECENT_SAMPLES) this.recent.shift();

    this.detectThrottle(timestamp);
  }

  /** Current cap estimate for threshold derivation. */
  getCap(): number {
    return this.throttled ? this.mark : Math.max(this.mark, this.fallback);
  }

  /** Forget everything (native-DPR / display change). */
  clear(): void {
    this.mark = 0;
    this.recent = [];
    this.lowUniformSince = null;
    this.throttled = false;
  }

  private detectThrottle(timestamp: number): void {
    if (this.recent.length < RECENT_SAMPLES) {
      this.lowUniformSince = null;
      return;
    }
    const max = Math.max(...this.recent);
    const min = Math.min(...this.recent);
    const uniformLow =
      max < THROTTLE_FRACTION * this.getCap() && (max - min) / Math.max(max, 1e-6) < UNIFORMITY_SPREAD;

    if (!uniformLow) {
      this.lowUniformSince = null;
      return;
    }
    if (this.lowUniformSince === null) {
      this.lowUniformSince = timestamp;
      return;
    }
    if (timestamp - this.lowUniformSince >= DOWNSHIFT_AFTER_MS) {
      this.mark = max;
      this.throttled = true;
      this.lowUniformSince = null;
    }
  }
}
