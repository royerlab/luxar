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
import type { DataLoader, ViewState } from '../../data-loader-types';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { LinesDataLoader } from '../../../types/lines';
import type { ViewStateQueue } from '../view-state/view-state-queue';

export interface QueueNextCtx {
  viewStateQueue: ViewStateQueue;
  /**
   * All three per-type loader maps. Refinement must be gated on EVERY
   * geometry type — progressive Points and Lines loaders carry additive
   * ladders exactly like GSplats (ladders are on by default for all
   * recipes), and the post-load kick in `lifecycle/load-scene.ts` already
   * checks all three. Gating on gsplats alone left points/lines-only
   * scenes stuck at partial LODs after every view change.
   */
  pointsLoaders: Map<string, DataLoader>;
  linesLoaders: Map<string, LinesDataLoader>;
  gsplatLoaders: Map<string, GSplatsDataLoader>;
  /** Re-enter the orchestrator's `updateView` with the next pending state. */
  updateView(state: Partial<ViewState>): Promise<void>;
  /** Set the orchestrator's `_updateInProgress` flag. */
  setUpdateInProgress(value: boolean): void;
  /**
   * Kick the progressive LOD refinement orchestrator (all three geometry
   * types in sequence — gsplats, then points, then lines).
   */
  scheduleGSplatsRefinement(): Promise<void>;
}

/** Structural `hasMoreLODs` probe — only progressive loaders expose it. */
function hasMore(loader: unknown): boolean {
  return (loader as { hasMoreLODs?: boolean }).hasMoreLODs === true;
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

  // No pending update — check if ANY progressive loader (points, lines, or
  // gsplats) still has additive LODs to stream. Mirrors the post-load kick
  // in lifecycle/load-scene.ts, which checks all three symmetrically.
  const needsRefinement =
    [...ctx.gsplatLoaders.values()].some(hasMore) ||
    [...ctx.pointsLoaders.values()].some(hasMore) ||
    [...ctx.linesLoaders.values()].some(hasMore);

  if (needsRefinement) {
    // Keep _updateInProgress = true during refinement so slider/animation
    // events queue as _pendingViewState (which naturally cancels refinement)
    log.info(
      Modules.SCENE_LOADER,
      'Scheduling progressive LOD refinement (hasMoreLODs=true after update)'
    );
    // Fire-and-forget: catch so an error escaping the refinement loop is
    // logged rather than surfacing as an unhandled promise rejection.
    ctx.scheduleGSplatsRefinement().catch((error) => {
      log.error(
        Modules.SCENE_LOADER,
        `Progressive refinement scheduling failed: ${(error as Error).message}`
      );
      // Belt-and-braces lock recovery: each refinement loop releases the
      // lock in its own finally, so a rejection reaching here means the
      // orchestrator glue died OUTSIDE those finallys — without this, the
      // lock stays held forever and every future updateView queues into a
      // pending slot that nothing ever drains (total viewer freeze).
      // Releasing twice is harmless (idempotent boolean), and draining is a
      // no-op when nothing queued during the failed run.
      ctx.setUpdateInProgress(false);
      ctx.viewStateQueue.drain((state) => ctx.updateView(state));
    });
  } else {
    // No pending update, no refinement needed - release the lock now
    ctx.setUpdateInProgress(false);
  }
}
