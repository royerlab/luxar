import type { AnimationConfig } from './types';

/**
 * Animation loop settings
 */
export const animationConfig: AnimationConfig = {
  idleTimeoutMs: 2000, // Time in milliseconds before pausing animation when idle - saves power
  pacing: {
    enabled: true, // Frame pacing on by default; false restores the back-to-back rAF loop
    // 250 ms is 4 fps — a scene already far past any interactive
    // threshold, so every healthy frame rate stays on the untouched fast
    // path.
    //
    // The trigger is the frame PERIOD, so a FOREIGN main-thread task of
    // this size lands on the loop's account even when the loop did not
    // cause it. Scene loading runs frames in that neighbourhood — the
    // repo's own `tests/e2e/render-ticks.ts` records ~230 ms periods, just
    // under this threshold — so a heavier load crosses it and does pace.
    // That is deliberate on both counts: the period is what the user
    // actually experiences between frames, and the wedge this exists for
    // spends its second outside JS, where a body-span measurement reads
    // ~2 ms and would never fire.
    //
    // An ISOLATED hiccup past this value still keeps the fast path, but
    // that is the loop's STREAK rule doing the work, not this threshold:
    // the controller paces only after two consecutive frames past it (see
    // PACING_SLOW_FRAME_STREAK in
    // `scene/animation/animation-controller.ts`). SUSTAINED foreign work —
    // a heavy scene load, a burst of chunk decodes — therefore does pace,
    // which is the behaviour you want there: yielding to those decode tasks
    // is the point.
    slowFrameMs: 250,
    // Ceiling on the gap we insert. The gap is dead time for anything the
    // main thread would otherwise do, including a `requestRender()` that
    // arrives just after a cooldown starts — so it also bounds the
    // worst-case added latency of an on-demand repaint at a quarter of a
    // second, independent of how slow the frame that armed it was.
    maxCooldownMs: 250,
  },
};
