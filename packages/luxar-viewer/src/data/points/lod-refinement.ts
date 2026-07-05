/**
 * Progressive Points LOD refinement — thin wrapper over the generic
 * helper at `data/scene-loader/progressive/refinement.ts`.
 *
 * Mirrors `data/gsplats/lod-refinement.ts` shape so the three leaf
 * types stay symmetric.
 *
 * @module data/points/lod-refinement
 */

import * as THREE from 'three';
import type {
  LoadedPointsData,
  PointsDataLoader,
  PointsMetadata,
  PointsViewState,
} from '../../types/points';
import { log, Modules } from '../../utils/log';
import { isAbortError } from '../loaders/abort-error';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import {
  MAX_CONSECUTIVE_REFINEMENT_FAILURES,
  RefinementFailureTracker,
  runProgressiveRefinement,
} from '../scene-loader/progressive/refinement';

export interface PointsRefinementCtx {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  pointsLoaders: Map<string, PointsDataLoader>;
  deriveNodeViewState(
    path: string,
    attrs: PointsMetadata | undefined,
    opts: { applyPartialExtendTolerance: boolean }
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };
  /**
   * Points commits directly (no async-project step like lines/gsplats);
   * the helper passes the freshly loaded data straight to
   * `updatePointsGeometry`.
   */
  updatePointsGeometry(path: string, data: LoadedPointsData, session?: UpdateSession): void;
  /**
   * Profiler for background-pass accounting. Each per-loader refinement
   * step opens a 'LOD Refinement' pass root (its own persistent tree,
   * separate from 'Total Update') with a per-node child session threaded
   * through load → commit.
   */
  profiler?: UpdateProfiler | null;
  updateVisibleCountsInMonitor(): void;
  releaseLock(): void;
  retriggerUpdate(pendingState: Partial<ViewState>): void;
  /** Liveness check; false once the owning SceneLoader was disposed. */
  isActive?(): boolean;
  /**
   * Per-refinement-run abort signal. The orchestrator assigns the run's
   * controller to the SceneLoader's `_updateAbortController`, so a
   * superseding `updateView` (or dispose) aborts in-flight refinement
   * chunk reads MID-PASS instead of waiting out the whole pass. An
   * `AbortError` in the per-loader catch is cancellation, not failure.
   */
  signal?: AbortSignal;
}

export async function runPointsRefinement(ctx: PointsRefinementCtx): Promise<void> {
  // Per-run failure backoff: a loader that fails MAX_CONSECUTIVE times is
  // excluded for the rest of this run (and from anyHasMoreLODs, so the loop
  // can terminate) instead of retrying at frame rate forever.
  const failures = new RefinementFailureTracker();
  await runProgressiveRefinement({
    loaders: ctx.pointsLoaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      // Only progressive loaders expose `hasMoreLODs`; single-shot
      // PointsSpatialIndexLoader doesn't have it, so skip on absence.
      const progressiveLoader = loader as PointsDataLoader & {
        hasMoreLODs?: boolean;
      };
      if (progressiveLoader.hasMoreLODs !== true) return;
      if (failures.isExhausted(path)) return;
      try {
        const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as PointsMetadata | undefined;
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: true,
        });
        if (refined.skip) return;
        const pointsVS: PointsViewState = refined.viewState;

        // Account this step to the 'LOD Refinement' tree (opened only when
        // real work happens, so empty sweeps don't record noise passes).
        const pass = ctx.profiler?.beginPass();
        const session = pass?.begin(`Points (${path})`);
        try {
          const data = await loader.updateView(pointsVS, session, ctx.signal);
          if (data) {
            ctx.updatePointsGeometry(path, data, session);
          }
        } finally {
          session?.end();
          pass?.end();
        }
        failures.recordSuccess(path);
      } catch (error) {
        // Superseded, not failed: a newer view-state (or dispose) aborted the
        // in-flight read on purpose. Don't count it toward the failure backoff
        // or log an error — the loop's next-pass pending check hands off.
        if (isAbortError(error)) return;
        if (failures.recordFailure(path)) {
          log.error(
            Modules.SCENE_LOADER,
            `Points refinement failed for ${path}: ${(error as Error).message} — ` +
              `giving up after ${MAX_CONSECUTIVE_REFINEMENT_FAILURES} consecutive failures ` +
              '(will retry on the next view change)'
          );
        } else {
          log.error(
            Modules.SCENE_LOADER,
            `Points refinement failed for ${path}: ${(error as Error).message}`
          );
        }
      }
    },
    anyHasMoreLODs: () =>
      [...ctx.pointsLoaders.entries()].some(([path, l]) => {
        const pl = l as PointsDataLoader & { hasMoreLODs?: boolean };
        return !failures.isExhausted(path) && pl.hasMoreLODs === true;
      }),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
    onError: (error) =>
      log.error(Modules.SCENE_LOADER, `Points refinement loop error: ${(error as Error).message}`),
  });
}
