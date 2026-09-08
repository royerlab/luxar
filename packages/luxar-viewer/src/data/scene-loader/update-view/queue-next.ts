/**
 * `updateView` finally-phase: figure out what runs next.
 *
 * Three outcomes:
 *   1. A pending view-state was queued during the in-flight update —
 *      yield to the render loop via `scheduleFrame` (rAF while visible,
 *      timer in hidden tabs, sync in non-browser contexts), then
 *      release the lock and re-enter `updateView(pendingState)`.
 *   2. No pending state, but at least one progressive loader of any kind
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
import { getErrorMessage } from '../../../utils/format-error';
import { scheduleFrame } from '../../../utils/schedule-frame';
import type { DataLoader, ViewState } from '../../data-loader-types';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { LinesDataLoader } from '../../../types/lines';
import type { MeshDataLoader } from '../../../types/mesh';
import type { ViewStateQueue } from '../view-state/view-state-queue';
import { noteRefinementComplete } from '../../../profiling/load-timeline';

export interface QueueNextCtx {
  viewStateQueue: ViewStateQueue;
  /**
   * All four per-type loader maps. Refinement must be gated on EVERY
   * geometry type — progressive Points and Lines loaders carry additive
   * ladders exactly like GSplats (ladders are on by default for all
   * recipes), Mesh carries a reveal ladder of the same shape, and the
   * post-load kick in `lifecycle/load-scene.ts` checks the same four.
   * Gating on gsplats alone left points/lines-only scenes stuck at partial
   * LODs after every view change.
   */
  pointsLoaders: Map<string, DataLoader>;
  linesLoaders: Map<string, LinesDataLoader>;
  gsplatLoaders: Map<string, GSplatsDataLoader>;
  meshLoaders: Map<string, MeshDataLoader>;
  /** Re-enter the orchestrator's `updateView` with the next pending state. */
  updateView(state: Partial<ViewState>): Promise<void>;
  /** Set the orchestrator's `_updateInProgress` flag. */
  setUpdateInProgress(value: boolean): void;
  /**
   * Kick the progressive LOD refinement orchestrator (all four geometry
   * types in sequence — gsplats, points, lines, then mesh).
   */
  scheduleGSplatsRefinement(): Promise<void>;
  /**
   * Settle the orchestrator's queued-update waiters (callers parked in
   * `updateView`'s supersede branch). Called whenever a pass completes with
   * NO pending state left — i.e. the requested-or-newer view-state has
   * landed its main-pass commit — which is exactly what the
   * dimension-animation pacing gate awaits. When a pending state exists,
   * waiters are carried to the re-entered pass instead.
   */
  resolvePassWaiters(): void;
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
    // Yield to render loop: ensure at least one frame is painted before next
    // update (prevents the "updates faster than renders" black-screen
    // problem). scheduleFrame degrades to a timer in hidden tabs — plain rAF
    // is suspended there and would stall the queued view-state until the tab
    // is foregrounded — and runs synchronously in non-browser contexts.
    scheduleFrame(() => {
      // Release the lock right before starting the next update
      // Any slider events during the yield were queued (because lock was held)
      ctx.setUpdateInProgress(false);
      ctx.updateView(pendingState).catch((error: unknown) => {
        log.error(
          Modules.SCENE_LOADER,
          `Queued updateView re-entry failed: ${getErrorMessage(error)}`,
          error
        );
      });
    });
    return;
  }

  // No pending state left: the pass that just finished IS the latest-wins
  // winner, so any callers parked in the queued branch have had their
  // requested-or-newer state committed — settle them now (synchronous from
  // takePending(), so no competing updateView can interleave). Resolving at
  // refinement ENTRY (not completion) is deliberate: the pacing gate needs
  // first-commit latency, not full-ladder latency.
  ctx.resolvePassWaiters();

  // No pending update — check if ANY progressive loader (points, lines,
  // gsplats or mesh) still has additive LODs to stream. Mirrors the post-load
  // kick in lifecycle/load-scene.ts, which checks the same four symmetrically.
  const needsRefinement =
    [...ctx.gsplatLoaders.values()].some(hasMore) ||
    [...ctx.pointsLoaders.values()].some(hasMore) ||
    [...ctx.linesLoaders.values()].some(hasMore) ||
    [...ctx.meshLoaders.values()].some(hasMore);

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
        `Progressive refinement scheduling failed: ${getErrorMessage(error)}`
      );
      // Belt-and-braces lock recovery: each refinement loop releases the
      // lock in its own finally, so a rejection reaching here means the
      // orchestrator glue died OUTSIDE those finallys — without this, the
      // lock stays held forever and every future updateView queues into a
      // pending slot that nothing ever drains (total viewer freeze).
      // Releasing twice is harmless (idempotent boolean), and draining is a
      // no-op when nothing queued during the failed run.
      ctx.setUpdateInProgress(false);
      // When nothing was queued during the failed run, no re-entry will
      // resolve parked waiters — settle them here. When a state WAS drained,
      // the re-entered pass carries them to its own commit instead; resolving
      // them here too would release the pacing gate before the view they
      // asked for landed.
      if (!ctx.viewStateQueue.drain((state) => ctx.updateView(state))) {
        ctx.resolvePassWaiters();
      }
    });
  } else {
    // No pending update, no refinement needed - release the lock now. The
    // load timeline's "refinement complete" milestone is reached trivially.
    noteRefinementComplete();
    ctx.setUpdateInProgress(false);
  }
}
