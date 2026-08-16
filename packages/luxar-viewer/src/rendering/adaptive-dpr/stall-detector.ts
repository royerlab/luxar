/**
 * Stall detector — tells DEAD TIME apart from a genuinely slow frame
 * rate, for the adaptive DPR control loop's frame-gap reset.
 *
 * Pure and value-driven: no `window`, no config imports, no clocks of
 * its own. The caller measures each inter-frame interval and feeds it
 * in (production from the animation loop's timestamps; tests feed
 * literals).
 *
 * A long interval is ambiguous on its own. Two very different things
 * produce one:
 *
 * - A STALL: a GC pause, a synchronous decode, a tab switch the pause
 *   hook never saw, an idle resume. The dead time is not render cost,
 *   so folding it into the FPS window would read as an artificially low
 *   frame rate and ratchet a spurious scale-down. The window must be
 *   thrown away.
 * - The FRAME RATE: a software rasterizer (SwiftShader) on a heavy
 *   scene renders at 0.5–3 fps, where EVERY interval is hundreds of
 *   milliseconds. Throwing the window away on each of those is how
 *   adaptive DPR became structurally inert exactly where shedding
 *   pixels matters most.
 *
 * An absolute threshold cannot separate them, and neither can "only the
 * first of a run counts" — a single sub-threshold interval re-arms that
 * rule, so an alternating cadence (400/300 ms, or a 2 s dead period
 * followed by a 100 ms frame) re-triggers it forever.
 *
 * The discriminator used here is an OUTLIER TEST against the recent
 * cadence: an interval is dead time only when it is both longer than
 * `minGapMs` (an absolute floor — a 60 ms hiccup is never worth
 * discarding a window over) AND much longer than the median of the
 * `windowSize` intervals that PRECEDED it. A 5 s gap in a 60 fps stream
 * is a 300× outlier; a 2 s interval in a stream whose recent intervals
 * are all ~2 s is simply the frame rate.
 *
 * The interval under test is deliberately NOT part of the median it is
 * judged against — it is remembered only afterwards. Mixing it in makes
 * the test partly self-referential, and with a short memory that decides
 * the verdict: after a single warm frame the memory held two samples,
 * the candidate was HALF of their mean, and a 5 s shader-compile stall
 * measured as "the frame rate" (so a startup or idle-resume stall
 * reduced DPR and armed a probe on nothing but dead time).
 *
 * The memory holds an EVEN number of neighbours on purpose. The median
 * of an even sample averages its two middle values, so a cadence that
 * alternates 1:1 between a dead period and a single fast frame
 * (2000/100/2000/100…) lands the threshold BETWEEN the two phases and
 * the slow phase is absorbed as the frame rate after a cycle or two.
 * With an odd memory the fast phase is always the majority, every dead
 * interval stays an outlier forever, and the FPS window never
 * accumulates — the "re-triggers forever" failure again.
 *
 * The cadence memory is deliberately NOT cleared by the gap reset it
 * drives — that is what makes it converge. A session that genuinely
 * slows down spends a couple of intervals being (harmlessly) misread as
 * stalling, the median follows the new cadence, and adaptation resumes.
 * The caller clears it only where the frame STREAM itself breaks and
 * resume dead time is expected next — display change, disable, pause,
 * dispose. A scene-content change is deliberately NOT one of those:
 * frames keep arriving at whatever rate they were arriving at, and a
 * cold memory falls back to the absolute floor alone, which misreads the
 * very next ordinary interval of any loop slower than `1000/minGapMs`
 * fps as dead time.
 */

/** Default outlier ratio an interval must exceed to be dead time. */
const DEFAULT_OUTLIER_FACTOR = 4;
/**
 * Default number of PRECEDING intervals the median is taken over (even
 * on purpose — see the class comment; 4 neighbours plus the interval
 * under test spans the same five-interval history as before).
 */
const DEFAULT_WINDOW_SIZE = 4;
/**
 * Fallback absolute floor used when `minGapMs` is non-finite. Mirrors
 * the `gapResetMs` config default; the class itself imports no config.
 */
const DEFAULT_MIN_GAP_MS = 350;

export class StallDetector {
  /** Intervals PRECEDING the next one to judge, oldest first (≤ windowSize). */
  private readonly recent: number[] = [];
  private readonly minGapMs: number;
  private readonly outlierFactor: number;
  private readonly windowSize: number;

  /**
   * @param minGapMs - Absolute floor: an interval at or under this is
   *   never dead time, however large the outlier ratio
   * @param outlierFactor - How many times the recent median an interval
   *   must exceed to count as dead time (default 4)
   * @param windowSize - How many PRECEDING intervals the median is taken
   *   over (default 4 — short enough to follow a real slowdown within a
   *   few frames, long enough that one hitch cannot move the median, and
   *   EVEN so a 1:1 alternating cadence converges; see the class
   *   comment). CLAMPED to at least 2: a `windowSize` of 0 leaves the
   *   memory permanently empty, so every over-floor interval takes the
   *   cold-memory path and the absolute-threshold-only rule this class
   *   exists to replace is silently back, while 1 leaves a memory that
   *   follows every single hitch. `outlierFactor` is likewise clamped to
   *   at least 1.
   *
   * A NON-FINITE knob is treated as ABSENT and falls back to the default
   * BEFORE the clamp, because `Math.max` propagates NaN rather than
   * rejecting it: `Math.max(2, NaN)` is NaN, and a NaN `windowSize`
   * would leave `recent.length > NaN` permanently false — the ring
   * never trims, so it grows without bound and pays an O(n log n) median
   * on every frame — while a NaN `outlierFactor` makes every comparison
   * false and the detector never fires at all. ±Infinity is just as
   * bad (an infinite ring, or a factor nothing can exceed), so the guard
   * is `Number.isFinite`, not an `isNaN` check. `minGapMs` needs it
   * MOST: `!(intervalMs > NaN)` is true for every interval, so a NaN
   * floor makes `isStall` answer false forever — dead time is never
   * discarded at all and a single tab-resume gap folds into the FPS
   * window of an otherwise healthy 60 fps session.
   */
  constructor(
    minGapMs: number,
    outlierFactor = DEFAULT_OUTLIER_FACTOR,
    windowSize = DEFAULT_WINDOW_SIZE
  ) {
    this.minGapMs = Math.max(0, Number.isFinite(minGapMs) ? minGapMs : DEFAULT_MIN_GAP_MS);
    this.outlierFactor = Math.max(
      1,
      Number.isFinite(outlierFactor) ? outlierFactor : DEFAULT_OUTLIER_FACTOR
    );
    this.windowSize = Math.max(
      2,
      Math.floor(Number.isFinite(windowSize) ? windowSize : DEFAULT_WINDOW_SIZE)
    );
  }

  /**
   * Feed the next inter-frame interval; true when it is DEAD TIME
   * rather than the frame rate.
   *
   * Every interval must be fed, stall or not — the median is the memory
   * that lets the next call tell the two apart.
   */
  isStall(intervalMs: number): boolean {
    const verdict = this.classify(intervalMs);
    this.remember(intervalMs);
    return verdict;
  }

  /**
   * The verdict for `intervalMs`, judged against the PRECEDING cadence
   * only — the interval never contributes to the median it is compared
   * against (see the class comment).
   */
  private classify(intervalMs: number): boolean {
    if (!(intervalMs > this.minGapMs)) return false;

    // Cold memory: there is no cadence to compare against, so fall back
    // to the absolute floor alone — a stall in the first interval after
    // a session boundary (startup, resume) must still be caught.
    if (this.recent.length === 0) return true;

    const cadence = this.median();
    // A non-positive or NaN median describes no cadence either.
    // Duplicate frame timestamps make it exactly 0, and `factor * 0` is
    // 0 — every over-floor interval would be an outlier, silently
    // restoring the absolute-threshold-only rule; a NaN (a NaN interval
    // fed earlier and remembered) makes every comparison false instead,
    // so the detector would go permanently blind. Fall back to the
    // absolute floor, which the interval has already cleared.
    if (!(cadence > 0)) return true;

    return intervalMs > this.outlierFactor * cadence;
  }

  /** Retain the interval as a neighbour for subsequent verdicts. */
  private remember(intervalMs: number): void {
    this.recent.push(intervalMs);
    if (this.recent.length > this.windowSize) this.recent.shift();
  }

  /** Median of the retained intervals (mean of the middle two if even). */
  private median(): number {
    const sorted = [...this.recent].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /**
   * Forget the cadence (display change, disable, pause, dispose). NOT
   * called by the gap reset itself, nor by a content change — see the
   * class comment.
   */
  clear(): void {
    this.recent.length = 0;
  }
}
