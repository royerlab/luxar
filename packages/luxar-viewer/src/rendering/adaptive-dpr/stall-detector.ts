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
 * discarding a window over) AND much longer than the median of the last
 * `windowSize` intervals. A 5 s gap in a 60 fps stream is a 300×
 * outlier; a 2 s interval in a stream whose recent intervals are all
 * ~2 s is simply the frame rate.
 *
 * The cadence memory is deliberately NOT cleared by the gap reset it
 * drives — that is what makes it converge. A session that genuinely
 * slows down spends a couple of intervals being (harmlessly) misread as
 * stalling, the median follows the new cadence, and adaptation resumes.
 * The caller clears it only at genuine session boundaries (display
 * change, disable, pause, content change, dispose), where the previous
 * cadence no longer describes anything.
 */
export class StallDetector {
  /** Most recent inter-frame intervals, oldest first (≤ windowSize). */
  private readonly recent: number[] = [];
  private readonly minGapMs: number;
  private readonly outlierFactor: number;
  private readonly windowSize: number;

  /**
   * @param minGapMs - Absolute floor: an interval at or under this is
   *   never dead time, however large the outlier ratio
   * @param outlierFactor - How many times the recent median an interval
   *   must exceed to count as dead time (default 4)
   * @param windowSize - How many recent intervals the median is taken
   *   over (default 5 — short enough to follow a real slowdown within a
   *   few frames, long enough that one hitch cannot move the median).
   *   CLAMPED to at least 2: a memory of one interval is always its own
   *   median, so every call would take the cold-memory path and the
   *   absolute-threshold-only rule this class exists to replace would be
   *   silently back. `outlierFactor` is likewise clamped to at least 1.
   */
  constructor(minGapMs: number, outlierFactor = 4, windowSize = 5) {
    this.minGapMs = minGapMs;
    this.outlierFactor = Math.max(1, outlierFactor);
    this.windowSize = Math.max(2, Math.floor(windowSize));
  }

  /**
   * Feed the next inter-frame interval; true when it is DEAD TIME
   * rather than the frame rate.
   *
   * Every interval must be fed, stall or not — the median is the memory
   * that lets the next call tell the two apart.
   */
  isStall(intervalMs: number): boolean {
    this.recent.push(intervalMs);
    if (this.recent.length > this.windowSize) this.recent.shift();

    if (!(intervalMs > this.minGapMs)) return false;

    // Cold memory: the only sample is the interval itself, which is its
    // own median, so the outlier test would answer "never a stall". With
    // no cadence to compare against, fall back to the absolute floor
    // alone — a stall in the first interval after a session boundary
    // (startup, resume, content change) must still be caught.
    if (this.recent.length < 2) return true;

    return intervalMs > this.outlierFactor * this.median();
  }

  /** Median of the retained intervals (mean of the middle two if even). */
  private median(): number {
    const sorted = [...this.recent].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /**
   * Forget the cadence (display change, disable, pause, content change,
   * dispose). NOT called by the gap reset itself — see the class
   * comment.
   */
  clear(): void {
    this.recent.length = 0;
  }
}
