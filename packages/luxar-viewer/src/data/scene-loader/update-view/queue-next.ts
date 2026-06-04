/**
 * `updateView` finally-phase: figure out what runs next.
 *
 * Three outcomes:
 *   1. A pending view-state was queued during the in-flight update —
 *      yield to the render loop via `requestAnimationFrame`, then
 *      release the lock and re-enter `updateView(pendingState)`.
 *      Falls back to immediate sync re-entry in non-browser contexts
 *      (Vitest, Worker).
 *   2. No pending state, but at least one progressive GSplats loader
 *      still has higher LODs to fetch — kick the refinement loop and
 *      keep the lock held (refinement releases it on completion or
 *      when superseded by a new pending state).
 *   3. No pending, no refinement — release the lock.
 *
 * The lock-keep semantics in branches 1 and 2 are load-bearing: slider
 * events fired during the rAF yield must queue (not race) and pending
 * view-state arrivals during refinement must naturally cancel the
 * refinement loop.
 */

import { log, Modules } from '../../../utils/log';
import type { ViewState } from '../../data-loader-types';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { ViewStateQueue } from '../view-state/view-state-queue';

export interface QueueNextCtx {
  viewStateQueue: ViewStateQueue;
  gsplatLoaders: Map<string, GSplatsDataLoader>;
  /** Re-enter the orchestrator's `updateView` with the next pending state. */
  updateView(state: Partial<ViewState>): Promise<void>;
  /** Set the orchestrator's `_updateInProgress` flag. */
  setUpdateInProgress(value: boolean): void;
  /** Kick the GSplats LOD refinement loop. */
  scheduleGSplatsRefinement(): Promise<void>;
}

/**
 * Decide and dispatch the next action after an updateView cycle completes.
 * No return value — everything happens via the ctx callbacks.
 */
export function queueNext(ctx: QueueNextCtx): void {
  const pendingState = ctx.viewStateQueue.takePending();
  if (pendingState !== null) {
    // Yield to render loop: ensure at least one frame is painted before next update
    // This prevents the "updates faster than renders" problem that causes black screen
    if (typeof requestAnimationFrame !== 'undefined') {
      requestAnimationFrame(() => {
        // Release the lock right before starting the next update
        // Any slider events during the yield were queued (because lock was held)
        ctx.setUpdateInProgress(false);
        ctx.updateView(pendingState);
      });
    } else {
      // Fallback for non-browser environments (e.g., tests)
      ctx.setUpdateInProgress(false);
      ctx.updateView(pendingState);
    }
    return;
  }

  // No pending update — check if progressive GSplats loaders need refinement
  const needsRefinement = [...ctx.gsplatLoaders.values()].some((l) => l.hasMoreLODs === true);

  if (needsRefinement) {
    // Keep _updateInProgress = true during refinement so slider/animation
    // events queue as _pendingViewState (which naturally cancels refinement)
    log.info(
      Modules.SCENE_LOADER,
      'Scheduling GSplats LOD refinement (hasMoreLODs=true after update)'
    );
    // Fire-and-forget: catch so an error escaping the refinement loop is
    // logged rather than surfacing as an unhandled promise rejection.
    ctx.scheduleGSplatsRefinement().catch((error) => {
      log.error(
        Modules.SCENE_LOADER,
        `GSplats refinement scheduling failed: ${(error as Error).message}`
      );
    });
  } else {
    // No pending update, no refinement needed - release the lock now
    ctx.setUpdateInProgress(false);
  }
}
