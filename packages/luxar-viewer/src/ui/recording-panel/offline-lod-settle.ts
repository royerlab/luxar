/**
 * Per-frame "wait for the LOD selector to settle" drain used by the offline
 * capture loop (#1695).
 *
 * Extracted from `OfflineCaptureStrategy.runOfflineCaptureLoop` (audit A2-01),
 * which had grown to 672 lines at 12 levels of nesting. This is the densest
 * part of it and the part with the most invariants per line, so it is also the
 * part that most benefits from being addressable on its own: the whole
 * tri-state / latch / re-arm / reporting state machine is now reachable without
 * standing up a capture session, a driver and an animation controller.
 *
 * Behaviour is unchanged from the inline version, comments included — the rules
 * encoded here were each paid for by a bug, so they travel with the code.
 */

import { Modules, log } from '../../utils/log';

/**
 * Wall-clock ceiling on the per-frame wait for the LOD selector to settle
 * (`hooks.isLODSettled`). A fine level that is genuinely being refetched over
 * the network can take a while; 2 s is generous enough to cover a normal
 * reload and short enough that a scene which cannot settle does not multiply
 * a 600-frame capture's runtime beyond recovery.
 */
export const LOD_SETTLE_TIMEOUT_MS = 2000;

/**
 * Frame ceiling on the same wait (≈2 s at 60 fps), inclusive of the mandatory
 * selector-catch-up tick. NOT redundant with the millisecond deadline: under a
 * stubbed clock — unit tests, fake timers — `performance.now()` never advances,
 * so the ms deadline alone would spin forever. Whichever bound trips first ends
 * the drain.
 */
export const LOD_SETTLE_MAX_FRAMES = 120;

/**
 * Consecutive per-frame settle timeouts after which the capture PAUSES
 * draining. A scene that can never settle (e.g. resident-byte thrash on a
 * partition that does not fit the budget) would otherwise pay
 * `LOD_SETTLE_TIMEOUT_MS` on EVERY remaining frame, silently multiplying the
 * capture's wall-clock. One successful settle resets the counter.
 *
 * The pause RE-ARMS: a latched frame still spends a free, non-waiting probe of
 * the predicate (no rAF, no poll — so a latched frame costs exactly what it did
 * before the latch fired) and clears the latch the moment that probe reports
 * settled. Without the re-arm the latch was terminal, and on exactly the scenes
 * this drain targets — an over-budget `adaptive` / `overview` partition, or a
 * Capture pressed before the initial load finished — the first three frames
 * would burn their budget, the latch would fire at frame 3, and the remaining
 * hundreds of frames would be exported with the pre-#1695 behaviour even though
 * the scene settles seconds later.
 */
export const MAX_CONSECUTIVE_LOD_TIMEOUTS = 3;

/** What the drain needs from the capture session to know it may keep waiting. */
export interface LodSettleDeps {
  /**
   * The injected quiescence predicate, TRI-STATE — see
   * `OfflineCaptureStrategyHooks.isLODSettled` for what each state means.
   * `null` and absent are identical: this scene has no lod_group to wait for,
   * so the whole drain is skipped, mandatory tick included.
   */
  isLODSettled?: () => boolean | null;
  /** False once the user has pressed Stop. */
  isRecording: () => boolean;
  /** True once the session has been aborted (cancel / dispose / error). */
  isAborted: () => boolean;
  /** One animation frame. Injected so tests can drive the clock. */
  nextFrame: () => Promise<void>;
}

/**
 * The per-frame LOD settle wait, plus the run-scoped bookkeeping its end-of-run
 * report needs.
 *
 * One instance per capture run: the latch, the streak counter and the
 * warn-once flags are all run-scoped.
 */
export class LodSettleDrain {
  private timeouts = 0;
  private consecutiveTimeouts = 0;
  private disabled = false;
  /**
   * Sticky counterpart of `disabled`: true once the latch has fired at any
   * point in the run, and never cleared by a re-arm. The latch itself comes
   * back on, so it cannot answer "were any frames captured without waiting?" at
   * the end of the run — this can. It doubles as the warn-once guard, so a run
   * that latches and re-arms repeatedly logs one line rather than one per latch.
   */
  private everDisabled = false;
  private predicateThrew = false;

  constructor(private readonly deps: LodSettleDeps) {}

  /** Frames that timed out waiting. The exact report's numerator. */
  get settleTimeouts(): number {
    return this.timeouts;
  }

  /**
   * Read the injected quiescence predicate, tri-state and throw-safe.
   *
   * `null` means "do not wait": either this scene has no lod_group to wait
   * for (no hook, no registry, or a registry with none in it), or the
   * predicate threw. The guard mirrors
   * `AnimationController.pacingSuspended()` and exists for the same
   * reason — this is a guard on the INJECTION POINT, not on a known
   * thrower. Unguarded, a throw would be caught by the loop's outer
   * handler, reported as "Recording failed" and discard the whole
   * sequence, which is a catastrophic price for a diagnostic predicate.
   *
   * The degradation is FRAME-SCOPED, not run-scoped: the predicate is
   * re-probed on the next frame, so a transient throw costs that one
   * frame's wait and nothing more. Only the WARNING is once-per-run, so a
   * persistently throwing provider does not emit one line per frame.
   *
   * A frame whose opening probe throws is skipped whole: it counts no
   * settle timeout AND does not reset `consecutiveTimeouts`. So an
   * alternating throw/timeout run latches `disabled` over more
   * than `MAX_CONSECUTIVE_LOD_TIMEOUTS` frames. That is intended — the
   * streak measures consecutive *timeouts*, not consecutive frames — but
   * it is worth stating rather than rediscovering.
   */
  private read(): boolean | null {
    try {
      return this.deps.isLODSettled?.() ?? null;
    } catch (err) {
      if (!this.predicateThrew) {
        this.predicateThrew = true;
        log.warning(
          Modules.RECORDING,
          'LOD settle predicate threw — that frame was captured without waiting for LOD ' +
            `(warned once per run; the predicate is retried on the next frame): ${err}`
        );
      }
      return null;
    }
  }

  /**
   * Wait for this frame's pose to settle, if this scene has anything to wait
   * for and the latch is not currently holding the wait off.
   *
   * MUST be called after the orbit callback has been removed: the camera pose
   * is fixed from that point on, so the extra frames only let pending loads
   * land. Draining before it would keep advancing the turntable while we wait,
   * smearing the sweep.
   */
  async waitForFrame(): Promise<void> {
    if (this.deps.isAborted() || !this.deps.isRecording()) return;

    if (this.disabled) {
      // Latched off after three consecutive timeouts — but the latch
      // re-arms (see MAX_CONSECUTIVE_LOD_TIMEOUTS). This probe is FREE:
      // no rAF, no poll, so a latched frame still costs exactly what it
      // did pre-#1695. Its boolean describes pose N−1 rather than the
      // pose being captured, which is fine for a re-arm signal — this
      // frame is captured undrained either way, and the next frame gets
      // the full, correctly-timed drain.
      if (this.read() === true) {
        this.disabled = false;
        this.consecutiveTimeouts = 0;
      }
      return;
    }

    // Tri-state probe, spent BEFORE the mandatory tick so a scene with
    // nothing to wait for does not pay for it. Only the NULL-ness of
    // this read is used: its boolean answer describes pose N−1 (see
    // the note on `read`) and is deliberately discarded.
    const drainApplies = this.read() !== null;
    if (!drainApplies) return;

    if (await this.pollUntilSettled()) {
      this.consecutiveTimeouts = 0;
      return;
    }
    // A stop/abort broke out of the wait — not a settle timeout, so it must
    // not be counted as one.
    if (this.deps.isAborted() || !this.deps.isRecording()) return;
    this.recordTimeout();
  }

  /**
   * The bounded wait itself: one mandatory selector-catch-up tick, then poll.
   *
   * @returns whether the scene settled. `false` also covers "a Stop or abort
   *          broke us out" — the caller distinguishes the two, because only a
   *          genuine give-up counts as a timeout.
   */
  private async pollUntilSettled(): Promise<boolean> {
    // `performance.now()` rather than `Date.now()`: monotonic, so an
    // NTP/DST step cannot move the deadline backwards mid-drain. It
    // is also the clock the animation controller measures frames
    // with.
    const deadline = performance.now() + LOD_SETTLE_TIMEOUT_MS;
    // The mandatory selector-catch-up tick. Counts against the frame budget
    // like any other drain frame.
    await this.deps.nextFrame();
    let drainFrames = 1;
    // Re-check before polling: a Stop landing during that first frame
    // must break out without being counted as a settle timeout. A
    // `null` here (the predicate threw) reads as settled — "do not
    // wait" — rather than as a timeout to report.
    if (this.deps.isAborted() || !this.deps.isRecording()) return false;
    let settled = this.read() !== false;
    // Bounded by BOTH a wall-clock deadline and a frame count —
    // whichever trips first (see LOD_SETTLE_MAX_FRAMES for why the
    // frame cap is not redundant).
    while (
      !settled &&
      !this.deps.isAborted() &&
      this.deps.isRecording() &&
      drainFrames < LOD_SETTLE_MAX_FRAMES &&
      performance.now() < deadline
    ) {
      await this.deps.nextFrame();
      drainFrames++;
      settled = this.read() !== false;
    }
    return settled;
  }

  /** A genuine timeout: count it, and latch the drain off after a streak. */
  private recordTimeout(): void {
    this.timeouts++;
    this.consecutiveTimeouts++;
    if (this.consecutiveTimeouts < MAX_CONSECUTIVE_LOD_TIMEOUTS) return;

    this.disabled = true;
    // Warned once per run, not once per latch: the latch re-arms,
    // so a scene that keeps stalling would otherwise log a line
    // every three frames.
    if (this.everDisabled) return;
    this.everDisabled = true;
    log.warning(
      Modules.RECORDING,
      `${MAX_CONSECUTIVE_LOD_TIMEOUTS} consecutive LOD settle timeouts — ` +
        'pausing the per-frame LOD wait. It resumes on the first frame whose ' +
        'free probe reports the scene settled; frames captured meanwhile may ' +
        'not show the level the selector settled on.'
    );
  }

  /**
   * Make a degraded sequence visible rather than a mystery: some frames
   * were captured before their LOD levels finished loading, so they may not
   * show the level the selector had settled on.
   *
   * "May not show the settled level" rather than "may be coarse", in both
   * branches, because the predicate is direction-blind: `displayed !==
   * active` also fires while the never-downgrade gate legitimately holds a
   * FINER previously-displayed level over a coarser aspiration that is
   * still streaming. Those frames look better than the selection, not
   * worse, and telling the user they are coarse would be wrong.
   *
   * The two branches report genuinely different things, and conflating
   * them was actively misleading. Once the latch has fired (sticky
   * `everDisabled`, since the latch itself re-arms),
   * `timeouts` stops describing the run: it counts only the
   * frames that WAITED and gave up, while every frame captured while the
   * wait was off was taken with no wait at all — so on a 600-frame capture
   * a handful of counted timeouts can sit in front of hundreds of undrained
   * frames. (It is NOT pinned at MAX_CONSECUTIVE_LOD_TIMEOUTS: that
   * constant bounds the consecutive STREAK, and a run that alternates
   * timeout/settle can reach any total before three land in a row.) So: an
   * exact count only when waiting stayed on for the whole run; otherwise
   * say what actually happened and claim no number. The exact branch's
   * denominator is `attemptedFrames`, not `capturedFrames` — a timeout is
   * counted before the capture attempt, so a frame that timed out and then
   * threw would otherwise be in the numerator but not the denominator.
   *
   * @returns the toast to show, or `null` when the run needs no report.
   */
  report(capturedFrames: number, attemptedFrames: number): string | null {
    if (this.timeouts === 0) return null;

    if (this.everDisabled) {
      log.warning(
        Modules.RECORDING,
        `LOD settle waiting was paused after ${MAX_CONSECUTIVE_LOD_TIMEOUTS} consecutive ` +
          `timeouts (${LOD_SETTLE_TIMEOUT_MS} ms / ${LOD_SETTLE_MAX_FRAMES} frame limit ` +
          'each), and the frames captured while it was off were taken without waiting. An ' +
          `unknown number of this run's ${capturedFrames} frame(s) may therefore not show ` +
          'the level the selector had settled on.'
      );
      // Says what happened (waiting was paused), NOT "LOD never settled" —
      // a 600-frame run that settled cleanly for 550 frames and then hit a
      // network stall lands here too.
      return 'Paused waiting for LOD — some frames may not show the settled level';
    }

    log.warning(
      Modules.RECORDING,
      `${this.timeouts} of ${attemptedFrames} frame(s) were captured before their ` +
        `LOD levels settled (${LOD_SETTLE_TIMEOUT_MS} ms / ${LOD_SETTLE_MAX_FRAMES} frame ` +
        'limit). Those frames may not show the level the selector had settled on.'
    );
    return `${this.timeouts} frame(s) captured before LOD settled`;
  }
}
