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
import { releaseLineageIfUncommitted } from '../../types/prefix-lineage';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type { RefinementResidencyBudget } from '../scene-loader/progressive/residency-budget';
import { runProgressiveRefinement } from '../scene-loader/progressive/refinement';
import { PARTIAL_EXTEND_TOLERANCE } from '../scene-loader/partial-extend-tolerance';
import {
  admitRefinementCandidate,
  failureTrackerFor,
  handleRefinementError,
  makeRefinementProgressCallbacks,
  recordRefinementResidency,
  type RefinableLoader,
} from '../scene-loader/progressive/refinement-wrapper';
import { isObjectLoadEligible } from '../scene-loader/loaders/run-loader-updates';

/** Geometry name in this wrapper's log lines and toasts. */
const LABEL = 'Points';

export interface PointsRefinementCtx {
  /** Scene objects already resolved by the phase eligibility sweep. */
  objects: ReadonlyMap<string, THREE.Object3D | undefined>;
  viewStateQueue: ViewStateQueue;
  loaders: Map<string, PointsDataLoader>;
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
  const objects = ctx.objects;
  const isPathVisible = (path: string): boolean => isObjectLoadEligible(objects.get(path));
  await runProgressiveRefinement({
    loaders: ctx.loaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      // Every progressive field is optional here because `loaders` is
      // typed as plain `PointsDataLoader`: single-shot PointsSpatialIndexLoader
      // has no ladder. `admitRefinementCandidate` gates on `hasMoreLODs`.
      const progressiveLoader = loader as PointsDataLoader & RefinableLoader;
      const admission = admitRefinementCandidate(
        path,
        progressiveLoader,
        ctx.residencyBudget,
        isPathVisible
      );
      if (!admission.admitted) return false;
      try {
        const mesh = objects.get(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as PointsMetadata | undefined;
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: PARTIAL_EXTEND_TOLERANCE.points,
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
          const data = await progressiveLoader.updateView(
            pointsVS,
            session,
            ctx.signal,
            admission.allowanceBytes
          );
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
        failureTrackerFor(progressiveLoader).recordSuccess(path);
        return true;
      } catch (error) {
        return handleRefinementError({ label: LABEL }, path, error, progressiveLoader);
      } finally {
        recordRefinementResidency(path, progressiveLoader, ctx.residencyBudget);
      }
    },
    ...makeRefinementProgressCallbacks(
      LABEL,
      ctx.loaders as Map<string, PointsDataLoader & RefinableLoader>,
      ctx.residencyBudget,
      isPathVisible
    ),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
  });
}
