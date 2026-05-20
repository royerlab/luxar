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
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state-queue';
import type { StagedGSplatsCommit } from '../scene-loader/process/data-processor-gsplats';

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
    viewState: GSplatsViewState
  ): Promise<StagedGSplatsCommit | null>;
  commitGSplats(staged: StagedGSplatsCommit): void;
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
}

/**
 * Run the GSplats LOD refinement loop. Returns when no progressive
 * loader has additional LODs (normal completion) or when a pending
 * view-state was observed and the loop handed off to retriggerUpdate
 * (cancellation).
 */
export async function runGSplatsRefinement(ctx: GSplatsRefinementCtx): Promise<void> {
  let lockHandedOff = false;
  try {
    while (true) {
      // Yield to let browser paint the current LOD level.
      await new Promise<void>((resolve) => {
        if (typeof requestAnimationFrame !== 'undefined') {
          requestAnimationFrame(() => resolve());
        } else {
          resolve(); // Test environment: proceed immediately.
        }
      });

      // Check cancellation: did the user navigate?
      const pendingState = ctx.viewStateQueue.takePending();
      if (pendingState !== null) {
        // The retriggerUpdate callback owns the lock from here.
        lockHandedOff = true;
        ctx.retriggerUpdate(pendingState);
        return;
      }

      // Load next LOD level for each progressive loader.
      for (const [path, loader] of ctx.gsplatLoaders) {
        if (loader.hasMoreLODs !== true) continue;

        try {
          // GSplats refinement runs the same query-state derivation
          // as main update / retry / initial-load.
          const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
          const nodeAttrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
          const refinedDerived = ctx.deriveNodeViewState(path, nodeAttrs, {
            applyPartialExtendTolerance: true,
          });
          if (refinedDerived.skip) {
            // Full extend_to_all coverage: nothing to refine for this
            // loader. Skip ahead to the next.
            continue;
          }
          const gsplatsViewState: GSplatsViewState = refinedDerived.viewState;

          const data = await loader.updateView(gsplatsViewState);
          if (data) {
            const staged = await ctx.processGSplats(path, data, gsplatsViewState);
            if (staged) ctx.commitGSplats(staged);
          }
        } catch (error) {
          log.error(
            Modules.SCENE_LOADER,
            `GSplats refinement failed for ${path}: ${(error as Error).message}`
          );
        }
      }

      // Update monitor after refinement commit.
      ctx.updateVisibleCountsInMonitor();

      // Check if any progressive loaders still have more LODs after this pass.
      const anyMore = [...ctx.gsplatLoaders.values()].some((l) => l.hasMoreLODs === true);
      if (!anyMore) break; // All LODs loaded.
    }
  } finally {
    if (!lockHandedOff) {
      ctx.releaseLock();
    }
  }
}
