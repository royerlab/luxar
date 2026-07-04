/**
 * Progressive GSplats LOD refinement loop.
 *
 * Runs after the main updateView() commits LOD 0, loading additional
 * LODs one pass at a time with a requestAnimationFrame yield between
 * each pass — each LOD level is painted as a separate frame, giving
 * visible progressive refinement.
 *
 * The refinement loop does NOT go through SceneLoader's updateView()
 * entry point (which has the serialization lock). It directly calls
 * loader.updateView() + process + commit for GSplats loaders only.
 *
 * Cancellation: if the view-state queue has pending state (the user
 * navigated), the loop aborts and re-triggers updateView via the
 * normal serialization path. The serialization lock is held by the
 * caller; on cancellation, the loop calls the supplied retriggerUpdate
 * callback which is responsible for releasing the lock and re-entering
 * updateView at the next rAF (or synchronously in test environments).
 *
 * Timing is load-bearing: every iteration starts with a rAF yield so
 * each LOD pass paints separately. Test environments without rAF fall
 * through synchronously.
 *
 * @module data/gsplats/lod-refinement
 */

import * as THREE from 'three';
import type { GSplatsDataLoader, GSplatsMetadata, GSplatsViewState } from '../../types/gsplats';
import type { LoadedGSplatsData } from '../../types/gsplats';
import { log, Modules } from '../../utils/log';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type { StagedGSplatsCommit } from '../scene-loader/process/data-processor-gsplats';
import {
  MAX_CONSECUTIVE_REFINEMENT_FAILURES,
  RefinementFailureTracker,
  runProgressiveRefinement,
} from '../scene-loader/progressive/refinement';

/**
 * Bundle of host references the refinement loop needs. Kept narrow so
 * the loop can be tested with a stub.
 */
export interface GSplatsRefinementCtx {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  gsplatLoaders: Map<string, GSplatsDataLoader>;
  deriveNodeViewState(
    path: string,
    attrs: GSplatsMetadata | undefined,
    opts: { applyPartialExtendTolerance: boolean }
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };
  processGSplats(
    path: string,
    data: LoadedGSplatsData,
    viewState: GSplatsViewState,
    session?: UpdateSession
  ): Promise<StagedGSplatsCommit | null>;
  commitGSplats(staged: StagedGSplatsCommit, session?: UpdateSession): void;
  /**
   * Profiler for background-pass accounting. Each per-loader refinement
   * step opens a 'LOD Refinement' pass root (its own persistent tree,
   * separate from 'Total Update') with a per-node child session threaded
   * through load → process → commit.
   */
  profiler?: UpdateProfiler | null;
  /** Aggregate visible counts from all line + gsplat meshes into the monitor. */
  updateVisibleCountsInMonitor(): void;
  /**
   * Release the SceneLoader's `_updateInProgress` lock. Called when the
   * loop completes normally (no more LODs). On cancellation the loop
   * hands off the lock to retriggerUpdate's rAF callback instead.
   */
  releaseLock(): void;
  /**
   * Re-enter SceneLoader.updateView() with the pending state. Called
   * once on cancellation; this callback is responsible for releasing
   * the lock AND triggering the new update (typically at the next rAF
   * boundary so the cancelled paint resolves first).
   */
  retriggerUpdate(pendingState: Partial<ViewState>): void;
  /** Liveness check; false once the owning SceneLoader was disposed. */
  isActive?(): boolean;
}

/**
 * Run the GSplats LOD refinement loop. Thin wrapper over the generic
 * :func:`runProgressiveRefinement` helper with gsplats-specific
 * derive / process / commit closures.
 */
export async function runGSplatsRefinement(ctx: GSplatsRefinementCtx): Promise<void> {
  // Per-run failure backoff: a loader that fails MAX_CONSECUTIVE times is
  // excluded for the rest of this run (and from anyHasMoreLODs, so the loop
  // can terminate) instead of retrying at frame rate forever.
  const failures = new RefinementFailureTracker();
  await runProgressiveRefinement({
    loaders: ctx.gsplatLoaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      if (loader.hasMoreLODs !== true) return;
      if (failures.isExhausted(path)) return;
      try {
        const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: true,
        });
        if (refined.skip) return;
        const gsplatsViewState: GSplatsViewState = refined.viewState;

        // Account this step to the 'LOD Refinement' tree (opened only when
        // real work happens, so empty sweeps don't record noise passes).
        const pass = ctx.profiler?.beginPass();
        const session = pass?.begin(`GSplats (${path})`);
        try {
          const data = await loader.updateView(gsplatsViewState, session);
          if (data) {
            const staged = await ctx.processGSplats(path, data, gsplatsViewState, session);
            if (staged) ctx.commitGSplats(staged, session);
          }
        } finally {
          session?.end();
          pass?.end();
        }
        failures.recordSuccess(path);
      } catch (error) {
        if (failures.recordFailure(path)) {
          log.error(
            Modules.SCENE_LOADER,
            `GSplats refinement failed for ${path}: ${(error as Error).message} — ` +
              `giving up after ${MAX_CONSECUTIVE_REFINEMENT_FAILURES} consecutive failures ` +
              '(will retry on the next view change)'
          );
        } else {
          log.error(
            Modules.SCENE_LOADER,
            `GSplats refinement failed for ${path}: ${(error as Error).message}`
          );
        }
      }
    },
    anyHasMoreLODs: () =>
      [...ctx.gsplatLoaders.entries()].some(
        ([path, l]) => !failures.isExhausted(path) && l.hasMoreLODs === true
      ),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
    onError: (error) =>
      log.error(Modules.SCENE_LOADER, `GSplats refinement loop error: ${(error as Error).message}`),
  });
}
