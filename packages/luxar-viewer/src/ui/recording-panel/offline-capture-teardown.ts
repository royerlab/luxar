/**
 * The offline capture's teardown, in the one order that is safe.
 *
 * Extracted from `OfflineCaptureStrategy.runOfflineCaptureLoop`'s `finally`
 * (audit A2-01). Almost every line here is ordered with respect to another one,
 * and the reasons are not local — a flag cleared one statement too early leaves
 * the viewer blank or the render target stuck at capture resolution until a
 * page reload. Naming the sequence makes those constraints reviewable in one
 * screen instead of at the bottom of a 672-line method.
 *
 * Runs from a `finally`, so it must be idempotent and must not throw: every
 * removal/restore handles the "wasn't set" case, and the driver abort is
 * wrapped.
 */

import { Modules, log } from '../../utils/log';
import type { AnimationController } from '../../scene/animation/animation-controller';
import type { CaptureContext, OfflineCaptureDriver } from './drivers/offline-capture-driver';
import type { RecordingSession } from './session';

export interface CaptureTeardown {
  session: RecordingSession;
  sessionAbort: AbortController;
  driver: OfflineCaptureDriver;
  ctx: CaptureContext;
  /** Whether `driver.setup` completed — an unset driver must not be aborted. */
  setupCompleted: boolean;
  /** Whether `driver.finalize` completed — a finalized driver must not be aborted. */
  finalizeSucceeded: boolean;
  animationController: AnimationController;
  /** The per-frame callbacks to unregister (capture + keep-alive). */
  callbackIds: readonly string[];
  cleanupOverlay: () => void;
  /** Drop this run's controller if it is still the owning one. */
  releaseSession: (abort: AbortController) => void;
}

/** Tear the capture down. Safe to call from a `finally` on any exit path. */
export async function runCaptureTeardown(t: CaptureTeardown): Promise<void> {
  const { session, sessionAbort, animationController } = t;

  // Clear the render-skip flag FIRST: it globally suppresses the
  // loop's render, and a driver abort that never settles would
  // otherwise leave the viewer frozen with no recovery but a reload.
  // Only this flag — the mutual-exclusion flags below must survive
  // the abort await.
  session.isLoopRenderSuppressed = false;

  if (t.setupCompleted && !t.finalizeSucceeded) {
    try {
      const reason = sessionAbort.signal.aborted
        ? sessionAbort.signal.reason === 'user-cancel'
          ? 'user-cancel'
          : 'disposed'
        : 'error';
      await t.driver.abort?.(t.ctx, reason as 'disposed' | 'user-cancel' | 'error');
    } catch (abortErr) {
      log.warning(Modules.RECORDING, `Driver abort during cleanup failed: ${abortErr}`);
    }
  }

  for (const id of t.callbackIds) {
    animationController.removePerFrameCallback(id);
  }
  session.hideRecordingIndicator();
  session.isRecording = false;
  // NOT cleared before the abort await above: `ScreenshotStrategy`
  // gates mutual exclusion on this flag, so a screenshot started
  // mid-teardown would overwrite and then null the single
  // `savedRecordingState` slot, leaving `restoreRecordingState()`
  // below a no-op — the viewer stuck at capture resolution with
  // resize locked until a page reload.
  session.isOfflineCaptureActive = false;
  session.isEXRSequenceRecording = false;
  t.cleanupOverlay();
  session.restoreAutoRotate();
  session.restoreAutoDolly();
  session.restoreRecordingState();
  // Guarantee exactly one repaint after teardown. restoreRecordingState
  // resizes the render target back, which clears the canvas, and the
  // keep-alive callback is already gone by now — so a loop that is
  // still stopped (or that the idle timer stops in the gap right after
  // the resize) leaves the viewer blank until the next mouse move.
  // The render-skip predicate reads `isLoopRenderSuppressed`, cleared
  // at the top of this function, so this frame is a real render.
  //
  // Never on a disposed session: dispose() tears the AnimationController
  // down BEFORE the RecordingPanel (see `runDisposePipeline`), and
  // RecordingPanel.dispose() only ABORTS an in-flight capture — a loop
  // parked on an await resumes here a tick later. Waking it then would
  // restart the rAF loop against a disposed PostProcessingManager.
  if (!session.isDisposed()) {
    animationController.startAnimation();
  }
  t.releaseSession(sessionAbort);
}
