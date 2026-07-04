/**
 * Frame-boundary scheduling that keeps working in hidden tabs.
 *
 * The update pipeline yields to the render loop between serialized view
 * updates (`queue-next.ts`) and on refinement cancellation hand-off
 * (`SceneLoader.scheduleGSplatsRefinement`'s `onCancel`) so at least one
 * frame paints before the next update. Plain `requestAnimationFrame` is the
 * right primitive while the tab is visible — but browsers suspend rAF when
 * `document.hidden`, so a pending view-state queued in a background tab
 * previously stalled until the tab was foregrounded (e.g. a programmatic
 * scrub driven while the user watches another window never progressed).
 *
 * Policy:
 *   - rAF available and the document visible → schedule via rAF, plus a
 *     shadow `setTimeout` fallback so a tab hidden AFTER scheduling still
 *     progresses (first source to fire wins; the other is a guarded no-op).
 *   - rAF available but the document already hidden → `setTimeout(cb, 0)`
 *     (there is no frame to wait for).
 *   - rAF missing (Vitest / Worker) → invoke synchronously, preserving the
 *     long-standing deterministic test-environment behaviour.
 *
 * @module utils/schedule-frame
 */

/**
 * Shadow-fallback delay for the visible-tab path. Long enough that it never
 * races a healthy rAF (a frame arrives in ≤ ~17 ms at 60 fps), short enough
 * that a tab hidden right after scheduling resumes promptly.
 */
export const FRAME_FALLBACK_MS = 200;

/**
 * Run `cb` exactly once at the next frame boundary — or via a timer when the
 * tab is hidden, or synchronously when `requestAnimationFrame` is missing.
 */
export function scheduleFrame(cb: () => void): void {
  if (typeof requestAnimationFrame === 'undefined') {
    // Non-browser context (Vitest, Worker): synchronous, deterministic.
    cb();
    return;
  }

  const hidden = typeof document !== 'undefined' && document.hidden;
  if (hidden) {
    // rAF is suspended in hidden tabs — don't wait for a frame that will
    // never come.
    setTimeout(cb, 0);
    return;
  }

  let fired = false;
  const runOnce = () => {
    if (fired) return;
    fired = true;
    clearTimeout(timeoutId);
    cb();
  };
  // Shadow fallback: covers the tab being hidden AFTER scheduling (the rAF
  // then never fires until foregrounded).
  const timeoutId = setTimeout(runOnce, FRAME_FALLBACK_MS);
  requestAnimationFrame(runOnce);
}
