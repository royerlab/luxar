// Frame scheduling for the viewer's render loop.
//
// The driver decides WHEN a frame runs: it owns `requestAnimationFrame`, the
// running flag, and frame pacing. WHAT a frame does is the caller's `onFrame`
// (AnimationController.tick). Keeping the two apart is what lets a different
// driver run the same frame, e.g. a WebXR session, whose frames come from
// `XRSession.requestAnimationFrame` rather than the window's.

import { config } from '../../config';

/**
 * Fraction of a slow frame's own cost inserted as a cooldown before the
 * next frame.
 *
 * Deliberately a FRACTION of the cost rather than a constant, over the band
 * where the fraction is what decides the gap: from
 * `config.animation.pacing.slowFrameMs` (250 ms) up to 1 s, above which
 * `config.animation.pacing.maxCooldownMs` (250 ms) clamps it flat. Across
 * that band the gap tracks the cost — at 0.25 it is a quarter of the frame,
 * so about a fifth of wall-clock (`0.25 / 1.25`) is yielded — whereas a flat
 * gap cannot: a fixed 100 ms over-yields at the bottom of the band and
 * under-yields at the top (after a 1 s frame it is 9 % of wall-clock against
 * the fraction's 20 %).
 *
 * Above the clamp the shipped behaviour is flat too, so the fraction wins
 * nothing there: at 5 s the clamped 250 ms is 4.8 % of wall-clock, the same
 * order as the fixed-100 ms strawman's 2 %. That is the clamp doing its job —
 * it bounds the added latency of an on-demand repaint (see `maxCooldownMs`)
 * — not a failure of the constant. The measured wedge (~1042 ms frames) sits
 * essentially at the clamp already.
 *
 * Any of those shares is only real because the cooldown is armed from the first
 * event-loop turn AFTER the frame's work rather than at the frame's start
 * (see `scheduleNextFrame()`): a timer armed at the start would already be
 * overdue by the time the thread frees, and would insert nothing.
 */
const PACING_COOLDOWN_FRACTION = 0.25;

/**
 * Consecutive frames whose own cost must exceed
 * `config.animation.pacing.slowFrameMs` before a cooldown is inserted.
 *
 * Two rather than one, because the two failure modes are asymmetric:
 * - The wedge pacing exists for is SUSTAINED — every frame costs ~1 s and
 *   never recovers. Requiring a second consecutive slow frame therefore only
 *   delays the first cooldown; it never withholds it.
 * - A single outlier is precisely what must NOT be paced: a GC pause, a
 *   shader compile, one synchronous chunk decode, or any foreign
 *   main-thread task charged to the loop because the measurement is a frame
 *   PERIOD. Pacing one of those inserted a cooldown that adaptive DPR then
 *   read as a 3.75 fps frame rate off a freshly cleared two-sample window
 *   (an unprobed 10 % DPR scale-down), and turned a sub-`gapResetMs`
 *   interval into an over-`gapResetMs` one, inventing a stall gap-reset
 *   that had not happened. With the streak, adaptive DPR never sees a
 *   cooldown that was not preceded by genuinely sustained slowness.
 *
 * An ALTERNATING slow/fast cadence is deliberately not paced either: the
 * fast frames are proof the main thread is already getting slots, which is
 * the only thing a cooldown buys.
 *
 * Two MEASURED slow frames means three frames in one uninterrupted run: the
 * first frame of a run has no predecessor and so measures nothing, the second
 * sets the streak to 1, and the third reaches 2 and arms the first cooldown.
 * And since `start()` (reached from `startAnimation()`) clears the streak on the stopped→running edge,
 * pacing is unreachable from a cold start whenever a frame costs more than
 * `config.animation.idleTimeoutMs / 2` — the idle timer fires before a third
 * frame exists and stops the loop.
 *
 * Neither fact costs anything here. The reported wedge holds
 * `animating=true` continuously — every landing depth-sort reply calls
 * `requestRender()`, which pushes the idle timer out again — so the streak
 * accumulates and pacing engages on the third frame. And in the cold-start
 * case the loop reaching its idle pause IS the outcome pacing exists to
 * enable: the main thread is free either way, so there is nothing to fix.
 */
const PACING_SLOW_FRAME_STREAK = 2;

/**
 * requestAnimationFrame driver with frame pacing.
 *
 * Frame pacing (#1724):
 * Because each frame re-arms `requestAnimationFrame` immediately, a scene
 * whose frames cost ~1 s puts the main thread at a 100 % duty cycle of
 * long tasks, and NOTHING else ever gets a slot — not worker message
 * delivery, not a CDP `Runtime.callFunctionOn`. That is not a rendering
 * inconvenience but a livelock: the depth-sort worker's replies (each
 * 0.1 ms of actual work) were dispatched at ~0.5/s, every landed reply
 * staged an ordering apply that called `requestRender()`, and the loop
 * could therefore never idle — the rendering starved the very hand-off
 * that would have let it stop. Measured on
 * `performance_benchmark_example.luxar.zarr`: ~120 of 200 dispatches
 * still outstanding after 70 s, `page.evaluate` timing out at 15 s
 * throughout.
 *
 * So when a frame's own cost exceeds `config.animation.pacing.slowFrameMs`
 * for `PACING_SLOW_FRAME_STREAK` consecutive frames — sustained slowness,
 * not an isolated hiccup — the next frame is scheduled after a bounded
 * cooldown (`setTimeout` → `setTimeout` → `requestAnimationFrame`) instead
 * of back-to-back. BOTH halves of that matter:
 * - no animation-frame request is outstanding while the frame is drawn, so
 *   the compositor stops driving main frames back-to-back on its own;
 * - and a genuine cooldown follows it. The genuineness is why the cooldown
 *   is armed from a zero-delay hop rather than directly: the expensive part
 *   of a slow frame runs after the rAF callback returns but inside the same
 *   main-thread task, so a timer armed at the frame's START is always
 *   already overdue when the thread frees and inserts no gap at all. The
 *   hop runs at the first event-loop turn after that work; only then is the
 *   real cooldown armed.
 *
 * Frames are DELAYED, never skipped: each one that runs still emits exactly
 * one `frame-start` / `frame-end` pair, and records itself with
 * `adaptiveDPRManager.recordFrame()` whenever that frame does GPU work of its
 * own — the call is gated on the context-lost and render-skip predicates, as
 * it was before pacing existed (see the comment at the call site). Both are
 * on the real clock: the achieved frame rate really is lower and neither the
 * FPS readout nor the DPR control loop may be told otherwise.
 */
export class RafDriver {
  /** Whether the loop is currently running */
  private running = false;

  /** RequestAnimationFrame ID for cancellation */
  private animationId: number = 0;

  /**
   * Timeout ID for the pending frame-pacing chain (null = none armed). Holds
   * the zero-delay hop first, then the cooldown that hop arms.
   */
  private pacingTimeout: ReturnType<typeof setTimeout> | null = null;

  /** Start timestamp of the previous frame (null = no frame measured yet) */
  private lastFrameStartTime: number | null = null;

  /** Cooldown we inserted BEFORE the current frame, in ms */
  private appliedPacingDelayMs = 0;

  /** Previous frame's own cost in ms (frame period minus our own cooldown) */
  private lastFrameCostMs = 0;

  /**
   * How many consecutive measured frames have cost more than
   * `config.animation.pacing.slowFrameMs`. Pacing engages only at
   * `PACING_SLOW_FRAME_STREAK`; any fast frame resets it to 0.
   */
  private consecutiveSlowFrames = 0;

  /**
   * Predicate that returns true while frame pacing must stay off because
   * some other owner depends on the loop's exact frame cadence. See
   * {@link AnimationController.setPacingSuspendPredicate}.
   */
  private isPacingSuspended: (() => boolean) | null = null;

  /**
   * @param onFrame - The frame's work, run once per scheduled frame AFTER the
   *   next frame has been armed (so the loop survives an exception it throws)
   */
  constructor(private readonly onFrame: () => void) {}

  /** Whether the loop is running. */
  get isRunning(): boolean {
    return this.running;
  }

  /** See {@link AnimationController.setPacingSuspendPredicate}. */
  setPacingSuspendPredicate(predicate: (() => boolean) | null): void {
    this.isPacingSuspended = predicate;
  }

  /**
   * Start the loop: arm its first frame for the next animation frame. No-op
   * when already running.
   *
   * Deliberately does NOT run a frame synchronously. A wake comes from an
   * input or state-change handler, and the next paint is already at the next
   * animation frame; a frame run here as well rendered every wake twice
   * before that paint. A caller that needs the frame NOW uses
   * AnimationController.renderOnce().
   *
   * @returns true when this call started the loop (the stopped→running edge)
   */
  start(): boolean {
    if (this.running) return false;
    this.running = true;
    // Forget the frame-cost measurement across the rest. The clock kept
    // running while the loop was stopped, so the gap between the last
    // frame before the pause and the first frame after it is idle time —
    // a two-second rest or a tab-hide would otherwise read as a
    // two-second frame and pace the first frame back for nothing.
    this.lastFrameStartTime = null;
    this.appliedPacingDelayMs = 0;
    this.lastFrameCostMs = 0;
    this.consecutiveSlowFrames = 0;
    // Arm the first frame - subsequent frames are scheduled by frame()
    this.animationId = requestAnimationFrame(this.frame);
    return true;
  }

  /** Stop the loop: cancel the pending frame and any pacing cooldown. */
  stop(): void {
    this.running = false;

    // Cancel any pending requestAnimationFrame call
    // This ensures no more frames are scheduled by the browser
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }

    // Cancel a pending frame-pacing cooldown. Its callback also re-checks
    // `running`, but leaving the timer armed would keep a stopped
    // driver holding a live timer (and, under fake timers, fire a
    // needless wake).
    if (this.pacingTimeout !== null) {
      clearTimeout(this.pacingTimeout);
      this.pacingTimeout = null;
    }
  }

  /**
   * One scheduled frame: measure the previous frame's cost, arm the next
   * frame (immediately, or after a pacing cooldown), then run `onFrame`.
   */
  private frame = (): void => {
    // Early exit if the loop is stopped - prevents unnecessary GPU work
    if (!this.running) return;

    // Measure the PREVIOUS frame's own cost for the pacing decision.
    //
    // The measurement is the frame PERIOD minus whatever cooldown we
    // ourselves inserted before this frame — not `end - start` of the JS
    // body. In the wedge this fixes, the ~1 s is non-JS main-thread /
    // compositor time: the body's own span reads ~2 ms, so a body-span
    // measurement would never fire. Subtracting our own gap is what stops
    // the measurement chasing its own tail — otherwise the period is
    // cost + cooldown, which stays above the threshold forever and
    // pacing would latch on for the rest of the session.
    //
    // Only a STREAK of slow frames paces (see PACING_SLOW_FRAME_STREAK): the
    // streak is advanced here, where the cost is measured, and read by
    // nextFramePacingDelayMs(). It stays untouched when there was nothing to
    // measure (the first frame of a run), so the rest before a resume cannot
    // become a streak's first link; start() clears it outright.
    const frameStart = performance.now();
    if (this.lastFrameStartTime !== null) {
      this.lastFrameCostMs = Math.max(
        0,
        frameStart - this.lastFrameStartTime - this.appliedPacingDelayMs
      );
      this.consecutiveSlowFrames =
        this.lastFrameCostMs > config.animation.pacing.slowFrameMs
          ? this.consecutiveSlowFrames + 1
          : 0;
    }
    this.lastFrameStartTime = frameStart;

    // Schedule the next frame - requestAnimationFrame syncs with display
    // refresh (smooth 60fps on most displays, 120fps on high-refresh
    // monitors), with a pacing cooldown first once consecutive frames have
    // been pathologically slow. Scheduled BEFORE the work below on purpose:
    // the loop then survives an exception thrown by controls, a per-frame
    // callback, or render.
    this.scheduleNextFrame();

    this.onFrame();
  };

  /**
   * Schedule the next loop iteration, inserting a pacing cooldown once
   * `PACING_SLOW_FRAME_STREAK` consecutive frames have been pathologically
   * slow.
   *
   * With no cooldown this is byte-for-byte the historical behaviour — a
   * bare `requestAnimationFrame(this.frame)`. With one, the rAF is armed
   * from a chain of two `setTimeout`s so the main thread has an actual gap
   * in which the browser can deliver a worker message, a CDP evaluate, or a
   * network callback. Frames are only ever DELAYED here, never dropped.
   */
  private scheduleNextFrame(): void {
    const cooldownMs = this.nextFramePacingDelayMs();

    if (cooldownMs <= 0) {
      this.appliedPacingDelayMs = 0;
      this.animationId = requestAnimationFrame(this.frame);
      return;
    }

    this.appliedPacingDelayMs = cooldownMs;
    // A timer armed HERE would be measured from the frame's start, and the
    // expensive part of a slow frame is browser rendering work that runs
    // after this callback returns but inside the same main-thread task — so
    // it would already be overdue when the thread frees and would insert no
    // gap at all. The zero-delay hop therefore runs at the first event-loop
    // turn AFTER that work, and only then is the real cooldown armed, so the
    // gap is genuine.
    //
    // The cost measurement subtracts the NOMINAL `cooldownMs`, but the real
    // gap is the hop's own latency plus `max(cooldownMs, when the main thread
    // next frees)` plus the post-cooldown rAF alignment (up to one vsync,
    // likewise not subtracted). So while pacing is engaged the next frame's
    // cost is OVER-estimated, and the over-estimate is NOT bounded by a few
    // milliseconds: a cooldown that comes due while a long foreign task is
    // running inflates it by all of that foreign work. (The hop itself is
    // armed from a rAF callback, so timer nesting restarts every frame and the
    // browser's 4 ms nesting clamp is never reached — it contributes nothing.)
    //
    // Unbounded above, but always in the safe direction: a slow frame is never
    // under-measured into the fast path. It also cannot make pacing ENGAGE
    // spuriously — the two frames that build the streak are by definition
    // unpaced, so neither carries a hop or a cooldown and neither is biased.
    // Its steady-state effect is that a session sharing the main thread with
    // sustained foreign work STAYS paced, which is the intended behaviour:
    // yielding to that work is the whole point.
    this.pacingTimeout = setTimeout(() => {
      // The loop may have been stopped (idle pause, tab hide, dispose)
      // while the cooldown was in flight — stopAnimation() clears these
      // timeouts, but a fire that already landed must not resurrect it.
      if (!this.running) {
        this.pacingTimeout = null;
        return;
      }
      // Re-read the suspend predicate. It was last read at the top of the
      // frame this cooldown was armed for, and a capture can have started
      // since — honouring the stale answer would freeze up to `maxCooldownMs`
      // of duplicate frame into the head of a real-time recording of an
      // already-slow scene. Drop the cooldown instead, and clear the applied
      // delay so the next cost measurement subtracts nothing.
      if (this.pacingSuspended()) {
        this.pacingTimeout = null;
        this.appliedPacingDelayMs = 0;
        this.animationId = requestAnimationFrame(this.frame);
        return;
      }
      this.pacingTimeout = setTimeout(() => {
        this.pacingTimeout = null;
        if (!this.running) return;
        this.animationId = requestAnimationFrame(this.frame);
      }, cooldownMs);
    }, 0);
  }

  /**
   * Whether pacing is currently suspended, with a throwing predicate read as
   * "not suspended".
   *
   * The try/catch is load-bearing because `scheduleNextFrame()` is the loop's
   * ONLY re-arm point: a throw that escaped it would leave nothing armed while
   * `running` stayed true, so `startAnimation()` early-returns forever and
   * no `requestRender()` can recover — an unrecoverable freeze. Today's wiring
   * cannot reach that: `RecordingPanel.isCurrentlyRecording()` is a plain flag
   * read, and the session it reads survives the panel's own `dispose()`. This
   * is therefore a guard on the INJECTION POINT rather than on a known
   * thrower — whatever gets wired here next inherits it, and the trade is a
   * paced frame during a capture against the viewer freezing for the rest of
   * the session.
   *
   * Shared by the two places the answer is needed — when the cooldown is
   * armed, and again in the hop callback before the cooldown is committed —
   * so both read it under the same guarantee.
   */
  private pacingSuspended(): boolean {
    try {
      return this.isPacingSuspended?.() === true;
    } catch {
      // Intentionally ignored — see above.
      return false;
    }
  }

  /**
   * Cooldown (ms) to insert before the next frame; 0 means "re-arm
   * requestAnimationFrame immediately", the untouched fast path.
   *
   * @returns Milliseconds to wait before the next `requestAnimationFrame`
   */
  private nextFramePacingDelayMs(): number {
    const pacing = config.animation.pacing;
    if (!pacing.enabled) return 0;

    // Recording owns the frame cadence for the whole capture. Read again in
    // the hop callback, since a capture can start mid-cooldown — see
    // `scheduleNextFrame()`.
    if (this.pacingSuspended()) return 0;

    // Every healthy frame rate lands here, and so does an ISOLATED slow
    // frame: the streak (advanced in frame(), where the cost is measured)
    // is the whole of the threshold test, deliberately not repeated here.
    // `lastFrameCostMs` is a frame PERIOD, so a foreign main-thread task of
    // that size is charged to the loop — one of them cannot pace anything,
    // but a sustained run of them can, which is what we want (see the
    // `slowFrameMs` comment in config/sections/animation/data.ts).
    if (this.consecutiveSlowFrames < PACING_SLOW_FRAME_STREAK) return 0;

    // A bounded fraction of the cost (see PACING_COOLDOWN_FRACTION). A
    // cooldown that rounds to 0 falls through the `cooldownMs <= 0` fast
    // path above, which is right: at a 0.25 fraction and a cost past the
    // threshold that only happens if `maxCooldownMs` is itself 0, i.e.
    // pacing has been configured off.
    return Math.min(
      pacing.maxCooldownMs,
      Math.round(this.lastFrameCostMs * PACING_COOLDOWN_FRACTION)
    );
  }
}
