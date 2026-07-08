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
 *   machinery.) Any sample above 80% of the fallback — or 25% above
 *   the reseeded throttle plateau, unreachable under a genuine tight
 *   throttle — immediately restores the bound, and a content change
 *   resets the throttle verdict outright (it was earned against the
 *   old content's frame stream).
 * - A sustained plateau BELOW `MIN_THROTTLE_PLATEAU` is never latched
 *   as a throttle at all — no real display mode runs that slow, so it
 *   must be heavy content — and the throttle latch itself requires a
 *   plateau >= `MIN_REAL_DISPLAY_RATE`, so the band between the two
 *   lines gets neither verdict. The estimator instead raises a one-shot
 *   DISTRESS verdict (consumeDistress) that the manager answers with a
 *   DPR-ceiling demotion to 1.0, while the cap stays fallback-floored
 *   so scale-down remains armed.
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
/**
 * A sample this far above the reseeded throttle plateau also clears
 * throttle mode: a genuine rAF throttle is TIGHT (its uniform-low
 * signature required < UNIFORMITY_SPREAD), so 25% above the plateau is
 * unreachable under the throttle — while a MISclassified heavy scene
 * that lightens recovers as soon as it clears the plateau, instead of
 * staying mis-capped until it somehow hits 80% of the fallback (inert
 * for 30Hz-class plateaus, whose mark/0.55 exceeds that line).
 */
const THROTTLE_EXIT_FACTOR = 1.25;
/**
 * No real display/rAF regime runs below this rate while the user
 * interacts (the slowest genuine modes are 30Hz low-power and the
 * ~23.976Hz film/NTSC desktop modes of 4K TVs over HDMI 1.4; occluded
 * -tab throttles don't interact), so a sustained plateau BELOW it
 * cannot be a display — it is heavy content in distress. The line sits
 * at 22, safely under 23.976, so a healthy vsync-bound 24Hz-TV session
 * is never read as distress. Latching the throttle verdict on such a
 * plateau is the catastrophic misfire: the cap collapses onto the
 * loaded FPS, the relative thresholds then read ~10fps as "at the
 * display cap = healthy", scale-up walks the DPR back to native and
 * parks it there, and the plateau-anchored exit line (×1.25) is
 * unreachable for a GPU-bound scene — the mistake latches for the
 * content's lifetime. Instead the estimator reports DISTRESS (see
 * consumeDistress), which the manager answers with a DPR-ceiling
 * demotion to 1.0.
 */
const MIN_THROTTLE_PLATEAU = 22;
/**
 * The slowest rAF rate any real display mode produces (~23.976Hz
 * film/NTSC, minus measurement jitter). The throttle latch requires the
 * plateau to be AT LEAST this — without a lower bound, uniform ~23fps
 * heavy content on a proven-fast display satisfies the throttle
 * signature and re-opens the catastrophic cap-collapse latch in the
 * [MIN_THROTTLE_PLATEAU, 24) band that the distress verdict was built
 * to close. Between the two lines — [22, 23.5) — neither verdict fires
 * by design: the plateau is too fast to be unambiguous distress and too
 * slow to be any real display, so the normal cap-relative scale-down
 * machinery (cap stays fallback-floored) handles it.
 */
const MIN_REAL_DISPLAY_RATE = 23.5;

export class RefreshRateEstimator {
  private mark = 0;
  private recent: number[] = [];
  private lowUniformSince: number | null = null;
  private throttled = false;
  // The uniform-low level the throttle verdict reseeded from. Kept
  // separate from `mark` (which keeps growing with observed samples)
  // so the un-throttle exit line stays anchored to the actual plateau.
  private throttlePlateau: number | null = null;
  // True once a sample has proven the display can beat 80% of the
  // fallback SINCE THE LAST CONTENT CHANGE. The throttle downshift is
  // gated on this rather than the lifetime `mark`: proof earned while
  // rendering LIGHT content says nothing about whether today's uniform
  // low FPS is a throttled display or the HEAVY content the user just
  // navigated into — after a content change the display must re-prove
  // itself before a downshift is allowed.
  private provenSinceContentChange = false;
  // One-shot sub-throttle distress verdict (sustained plateau below
  // MIN_THROTTLE_PLATEAU); latched here until the manager consumes it.
  private distressSignal = false;

  constructor(private readonly fallback: number) {}

  /**
   * Feed one window-FPS sample (callers should only feed full-span,
   * unsuppressed windows so partial data doesn't pollute the mark).
   */
  addSample(fps: number, timestamp: number): void {
    // Un-throttle when a sample clearly exceeds the throttle plateau:
    // either it approaches the fallback bound, or it rises well above
    // the reseeded plateau — a genuinely throttled display can produce
    // neither, while a misclassified heavy scene that lightens even a
    // little recovers immediately.
    const unthrottleAt = Math.min(
      this.fallback * UNTHROTTLE_FRACTION,
      this.throttled && this.throttlePlateau !== null
        ? this.throttlePlateau * THROTTLE_EXIT_FACTOR
        : Number.POSITIVE_INFINITY
    );
    if (fps >= unthrottleAt) {
      this.throttled = false;
      this.throttlePlateau = null;
    }
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
   * One-shot sub-throttle distress verdict: true when FPS has stayed
   * below MIN_THROTTLE_PLATEAU for DOWNSHIFT_AFTER_MS — a regime no
   * real display throttle can produce, so it is heavy content by
   * construction. Consuming clears the latch; the detector re-arms and
   * re-fires after another sustained period if the distress persists.
   * The intended response is a DPR-ceiling demotion, not a cap change
   * (the cap deliberately stays fallback-floored so scale-down stays
   * armed).
   *
   * RE-VALIDATED AT CONSUMPTION: a verdict can be latched on a tick the
   * manager doesn't consume (probe in flight, load suppression) and go
   * stale if the workload lightens before the next clean tick. The
   * verdict is therefore only honored while the CURRENT sample window
   * still shows distress; otherwise it is dropped (a genuinely
   * distressed scene re-earns it within DOWNSHIFT_AFTER_MS).
   */
  consumeDistress(): boolean {
    const d = this.distressSignal;
    this.distressSignal = false;
    if (!d) return false;
    return this.recent.length >= RECENT_SAMPLES && Math.max(...this.recent) < MIN_THROTTLE_PLATEAU;
  }

  /**
   * The frame loop was interrupted (idle pause, tab hide, a long
   * stall's gap-reset): the sample stream broke, so the SESSION-grade
   * transients are stale — the recent window (pre-gap frames), the
   * uniform-low plateau clock (which would otherwise count unobserved
   * wall-clock dead time toward the "sustained" requirement and let a
   * single janky post-resume window fire an immediate verdict), and an
   * unconsumed distress latch. LEARNED state survives: the high-water
   * mark, a latched throttle verdict, and the proven-rate flag all
   * describe the display/content, not the interrupted sample stream.
   */
  noteSessionInterrupted(): void {
    this.recent = [];
    this.lowUniformSince = null;
    this.distressSignal = false;
  }

  /**
   * Scene content genuinely changed (dataset load, layer change, LOD
   * swap): ALL content-relative throttle state resets — the proof that
   * would arm a new downshift AND an already-latched throttle verdict.
   * The verdict was earned against the OLD content's frame stream;
   * carrying it forward would keep the cap collapsed on the reseeded
   * plateau and invert the thresholds for the NEW content (scale-up
   * armed on a heavy scene) with no reachable exit sample. If the
   * display is STILL genuinely throttled, the new content cannot
   * re-prove ≥80% of fallback, so no new downshift fires and the
   * futile scale-down probes are contained by the rejection backoff —
   * the same accepted corner as a session throttled from its first
   * frame. Only the monotonic `mark` survives: it is used purely as an
   * upper bound via max(mark, fallback), which stays correct and
   * avoids threshold flapping on every LOD swap.
   */
  noteContentChanged(): void {
    this.provenSinceContentChange = false;
    this.lowUniformSince = null;
    this.throttled = false;
    this.throttlePlateau = null;
    // An unconsumed distress verdict was earned against the old
    // content's frame stream — the new content deserves fresh evidence.
    this.distressSignal = false;
  }

  /** Forget everything (native-DPR / display change). */
  clear(): void {
    this.mark = 0;
    this.recent = [];
    this.lowUniformSince = null;
    this.throttled = false;
    this.throttlePlateau = null;
    this.provenSinceContentChange = false;
    this.distressSignal = false;
  }

  private detectThrottle(timestamp: number): void {
    if (this.recent.length < RECENT_SAMPLES) {
      this.lowUniformSince = null;
      return;
    }
    const max = Math.max(...this.recent);
    const min = Math.min(...this.recent);

    // Sub-throttle DISTRESS: a plateau below the slowest real display
    // rate can only be heavy content. No proof or uniformity required —
    // a sustained plateau below the line is trouble regardless of
    // jitter or what the display once demonstrated.
    const distressLow = max < MIN_THROTTLE_PLATEAU;

    // THROTTLE signature: a downshift below the fallback floor is only
    // justified when the plateau is a rate a real display can produce
    // (>= MIN_REAL_DISPLAY_RATE — without this floor, uniform ~23fps
    // heavy content latches the cap-collapse in the [22, 24) band) AND
    // the display has PROVEN a higher achievable rate — proven it
    // AGAINST THE CURRENT CONTENT. Without the proof guard a
    // steady heavy scene (uniformly low FPS, low variance — exactly the
    // signature of a GPU-bound render parked at the DPR floor) would be
    // misclassified as a throttled display, collapsing the cap onto the
    // loaded FPS and inverting the thresholds: scale-down disarmed and
    // scale-up armed on the scene that most needs fewer pixels. The
    // since-content-change scoping closes the light-then-heavy variant
    // (mark pinned at 60 by a light loading screen, then dense data).
    // Residual ambiguity: heaviness arriving with NO content signal at
    // all (e.g. rotating an unchanged scene edge-on) into the ~24-48fps
    // band is fundamentally indistinguishable from a throttle by FPS
    // alone; the widened un-throttle line in addSample and the
    // punished-ascent ceiling machinery bound that mistake's lifetime.
    // Compare against the PROVEN mark (not the fallback-floored cap):
    // "uniformly far below what this display demonstrated it can do".
    const throttleLow =
      max >= MIN_REAL_DISPLAY_RATE &&
      this.provenSinceContentChange &&
      max < THROTTLE_FRACTION * this.mark &&
      (max - min) / Math.max(max, 1e-6) < UNIFORMITY_SPREAD;

    if (!distressLow && !throttleLow) {
      this.lowUniformSince = null;
      return;
    }
    if (this.lowUniformSince === null) {
      this.lowUniformSince = timestamp;
      return;
    }
    if (timestamp - this.lowUniformSince >= DOWNSHIFT_AFTER_MS) {
      if (distressLow) {
        // Never latch the throttle verdict down here (see
        // MIN_THROTTLE_PLATEAU) — signal distress and re-arm, so a
        // scene that STAYS distressed re-fires after another period.
        this.distressSignal = true;
      } else {
        this.mark = max;
        this.throttlePlateau = max;
        this.throttled = true;
      }
      this.lowUniformSince = null;
    }
  }
}
