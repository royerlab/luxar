/**
 * Progressive Points LOD refinement — thin wrapper over the generic
 * helper at `data/scene-loader/progressive/refinement.ts`.
 *
 * Mirrors `data/gsplats/lod-refinement.ts` shape so the four geometry
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
import { notifier } from '../../utils/cross-layer/notifier';
import { isAbortError } from '../loaders/abort-error';
import { releaseLineageIfUncommitted } from '../../types/prefix-lineage';
import { tryRollbackToPassStart } from '../loaders/progressive/pass-rollback';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type {
  LadderResidency,
  RefinementResidencyBudget,
} from '../scene-loader/progressive/residency-budget';
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
  ): { skip: false; viewState: ViewState };
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
  /**
   * Shared residency ceiling for sweep-registered progressive leaves. Absent =
   * unbounded (today's behaviour).
   */
  residencyBudget?: RefinementResidencyBudget;
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
        // Optional for the same reason as `hasMoreLODs`: only a progressive
        // loader has a ladder to unwind. Every `PointsProgressiveLoader` has it.
        rollbackToPassStart?: () => number;
        ladderResidency?: () => LadderResidency;
      };
      if (progressiveLoader.hasMoreLODs !== true) return false;
      if (failures.isExhausted(path)) return false;
      // Shared sweep residency ceiling. Declining here is not enough on its own —
      // `anyHasMoreLODs` and `getLoaderProgress` below must also exclude
      // declined paths, or the loop re-offers this loader every frame forever
      // while holding the update lock (the trap `isExhausted` already avoids).
      const residency = progressiveLoader.ladderResidency?.();
      if (residency && ctx.residencyBudget?.admit(path, residency).admitted === false) {
        return false;
      }
      try {
        const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as PointsMetadata | undefined;
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: true,
        });
        // A fully-extended node is a normal node with a slice-invariant query
        // (deriveNodeViewState), so it refines through this path like any other;
        // the `hasMoreLODs` gate above already stops a converged one.
        const pointsVS: PointsViewState = refined.viewState;

        // Account this step to the 'LOD Refinement' tree (opened only when
        // real work happens, so empty sweeps don't record noise passes).
        const pass = ctx.profiler?.beginPass();
        const session = pass?.begin(`Points (${path})`);
        try {
          const data = await loader.updateView(pointsVS, session, ctx.signal);
          let committed = false;
          try {
            // Superseded/disposed while we were loading: an abort that raced the
            // load's resolution (served from cache / abort after the fetch) is
            // not a throw. Skip the commit so no stale-slice geometry reaches the
            // GPU — mirrors runAtomicCommit's signal.aborted guard on the main
            // path (atomic-commit.ts).
            if (data && ctx.signal?.aborted !== true) {
              ctx.updatePointsGeometry(path, data, session);
              committed = true;
            }
          } finally {
            // A skipped commit never reaches the commit layer's own clear, and
            // the pinned parent is the previous cumulative — worth up to a whole
            // redundant copy on a deep ladder. See prefix-lineage.ts.
            releaseLineageIfUncommitted(data, committed);
          }
        } finally {
          session?.end();
          pass?.end();
        }
        failures.recordSuccess(path);
        return true;
      } catch (error) {
        // Superseded, not failed: a newer view-state (or dispose) aborted the
        // in-flight read on purpose. Don't count it toward the failure backoff
        // or log an error — the loop's next-pass pending check hands off.
        if (isAbortError(error)) return false;
        // Unwind the levels this pass appended before the throw, so the retry
        // re-attempts the SAME prefix rather than resuming from the advanced
        // cursor with a larger allocation. See
        // `../loaders/progressive/pass-rollback`.
        const unwound = tryRollbackToPassStart(progressiveLoader);
        if (failures.recordFailure(path)) {
          log.error(
            Modules.SCENE_LOADER,
            `Points refinement failed for ${path}: ${(error as Error).message} — ` +
              `giving up after ${MAX_CONSECUTIVE_REFINEMENT_FAILURES} consecutive failures ` +
              `(will retry on the next view change; unwound ${unwound} level(s))`
          );
          // The node silently freezes at its last valid coarse prefix — a
          // console-only error leaves the user staring at a permanently
          // coarse node with no explanation. Same channel as leaf-load
          // failures (load-leaf-error-dispatch).
          notifier.toast(`Refinement failed for ${path} — showing reduced detail`, 5000);
        } else {
          log.error(
            Modules.SCENE_LOADER,
            `Points refinement failed for ${path}: ${(error as Error).message} ` +
              `(unwound ${unwound} level(s))`
          );
        }
        return false;
      }
    },
    getLoaderProgress: (path, loader) => {
      const progressiveLoader = loader as PointsDataLoader & {
        hasMoreLODs?: boolean;
        loadedLODCount: number;
        totalLODCount: number;
      };
      if (
        failures.isExhausted(path) ||
        ctx.residencyBudget?.isDeclined(path) === true ||
        progressiveLoader.hasMoreLODs !== true
      ) {
        return null;
      }
      return {
        loaded: progressiveLoader.loadedLODCount,
        total: progressiveLoader.totalLODCount,
      };
    },
    // A budget-declined loader still reports `hasMoreLODs`, so it MUST be
    // excluded here as well or the loop never terminates.
    anyHasMoreLODs: () =>
      [...ctx.pointsLoaders.entries()].some(([path, l]) => {
        const pl = l as PointsDataLoader & { hasMoreLODs?: boolean };
        return (
          !failures.isExhausted(path) &&
          ctx.residencyBudget?.isDeclined(path) !== true &&
          pl.hasMoreLODs === true
        );
      }),
    onNoProgress: (stalled) => {
      for (const { path, loaded, total } of stalled) {
        log.warning(
          Modules.SCENE_LOADER,
          `Points refinement stopped for ${path}: no progress at LOD ${loaded}/${total}`
        );
      }
    },
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
    onError: (error) =>
      log.error(Modules.SCENE_LOADER, `Points refinement loop error: ${(error as Error).message}`),
  });
}
