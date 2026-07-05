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
 * - Genuine rAF throttling (30Hz low-power, background tabs) is only
 *   inferred when the display has PROVEN a higher achievable rate
 *   first (mark >= 80% of the fallback): a 120Hz session throttled to
 *   30 shows every sample uniformly far below its own demonstrated
 *   mark for a sustained period — when the last `RECENT_SAMPLES`
 *   samples stay below `THROTTLE_FRACTION × mark` with
 *   < `UNIFORMITY_SPREAD` relative spread for `DOWNSHIFT_AFTER_MS`,
 *   the mark reseeds from them and the fallback lower bound is
 *   released, so the throttle re-converges instead of firing spurious
 *   scale-downs against a stale cap forever. A HEAVY scene never
 *   proves a higher rate, so it can never be mistaken for a throttle:
 *   without the proven-rate guard, 10s of steady 20fps would collapse
 *   the cap to 20 and the relative thresholds would then read 20fps
 *   as healthy — disabling scale-down and even scaling UP exactly the
 *   scene that needs help. (A tab throttled from the very first frame
 *   also never proves a rate; it keeps the fallback cap and its
 *   futile scale-down probes are contained by the rejection-backoff
 *   machinery.) Any sample above 80% of the fallback immediately
 *   restores the bound.
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
  // True once a sample has proven the display can beat 80% of the
  // fallback SINCE THE LAST CONTENT CHANGE. The throttle downshift is
  // gated on this rather than the lifetime `mark`: proof earned while
  // rendering LIGHT content says nothing about whether today's uniform
  // low FPS is a throttled display or the HEAVY content the user just
  // navigated into — after a content change the display must re-prove
  // itself before a downshift is allowed.
  private provenSinceContentChange = false;

  constructor(private readonly fallback: number) {}

  /**
   * Feed one window-FPS sample (callers should only feed full-span,
   * unsuppressed windows so partial data doesn't pollute the mark).
   */
  addSample(fps: number, timestamp: number): void {
    // Un-throttle when a sample clearly exceeds the throttle plateau:
    // either it approaches the fallback bound, or it rises well above
    // the reseeded mark (mark/THROTTLE_FRACTION) — a genuinely
    // throttled display can produce neither, while a misclassified
    // heavy scene that lightens even a little recovers immediately
    // instead of staying mis-capped until it somehow hits 80% of
    // fallback.
    const unthrottleAt = Math.min(
      this.fallback * UNTHROTTLE_FRACTION,
      this.throttled ? this.mark / THROTTLE_FRACTION : Number.POSITIVE_INFINITY
    );
    if (fps >= unthrottleAt) this.throttled = false;
    if (fps >= this.fallback * UNTHROTTLE_FRACTION) this.provenSinceContentChange = true;
    if (fps > this.mark) this.mark = fps;

    this.recent.push(fps);
    if (this.recent.length > RECENT_SAMPLES) this.recent.shift();

    this.detectThrottle(timestamp);
  }

  /** Current cap estimate for threshold derivation. */
  getCap(): number {
    return this.throttled ? this.mark : Math.max(this.mark, this.fallback);
  }

  /**
   * Scene content genuinely changed (dataset load, layer change, LOD
   * swap): the proven-rate evidence now describes the OLD content, so
   * the throttle downshift is disarmed until the display re-proves its
   * rate against the new content. The monotonic `mark` itself is kept —
   * it still upper-bounds the cap correctly and keeping it avoids
   * threshold flapping on every LOD swap.
   */
  noteContentChanged(): void {
    this.provenSinceContentChange = false;
    this.lowUniformSince = null;
  }

  /** Forget everything (native-DPR / display change). */
  clear(): void {
    this.mark = 0;
    this.recent = [];
    this.lowUniformSince = null;
    this.throttled = false;
    this.provenSinceContentChange = false;
  }

  private detectThrottle(timestamp: number): void {
    if (this.recent.length < RECENT_SAMPLES) {
      this.lowUniformSince = null;
      return;
    }
    // A downshift below the fallback floor is only justified when the
    // display has PROVEN a higher achievable rate — and proven it
    // AGAINST THE CURRENT CONTENT. Without this guard a steady heavy
    // scene (uniformly low FPS, low variance — exactly the signature of
    // a GPU-bound render parked at the DPR floor) would be
    // misclassified as a throttled display, collapsing the cap onto the
    // loaded FPS and inverting the thresholds: scale-down disarmed and
    // scale-up armed on the scene that most needs fewer pixels. The
    // since-content-change scoping closes the light-then-heavy variant
    // (mark pinned at 60 by a light loading screen, then dense data).
    // Residual ambiguity: heaviness arriving with NO content signal at
    // all (e.g. rotating an unchanged scene edge-on) is fundamentally
    // indistinguishable from a throttle by FPS alone; the widened
    // un-throttle line in addSample bounds that mistake's lifetime.
    if (!this.provenSinceContentChange) {
      this.lowUniformSince = null;
      return;
    }
    const max = Math.max(...this.recent);
    const min = Math.min(...this.recent);
    // Compare against the PROVEN mark (not the fallback-floored cap):
    // "uniformly far below what this display demonstrated it can do".
    const uniformLow =
      max < THROTTLE_FRACTION * this.mark &&
      (max - min) / Math.max(max, 1e-6) < UNIFORMITY_SPREAD;

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
