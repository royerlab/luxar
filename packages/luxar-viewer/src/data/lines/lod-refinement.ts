/**
 * Progressive Lines LOD refinement — thin wrapper over the generic
 * helper at `data/scene-loader/progressive/refinement.ts`.
 *
 * Mirrors `data/gsplats/lod-refinement.ts` / `data/points/lod-refinement.ts`.
 *
 * @module data/lines/lod-refinement
 */

import * as THREE from 'three';
import type {
  LinesDataLoader,
  LinesMetadata,
  LinesViewState,
  LoadedLinesData,
} from '../../types/lines';
import { releaseLineageIfUncommitted } from '../../types/prefix-lineage';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type { StagedLinesCommit } from '../scene-loader/process/data-processor-lines';
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
const LABEL = 'Lines';

export interface LinesRefinementCtx {
  rootGroup: THREE.Group | null;
  /** Scene objects already resolved by the phase eligibility sweep. */
  objects?: ReadonlyMap<string, THREE.Object3D | undefined>;
  viewStateQueue: ViewStateQueue;
  loaders: Map<string, LinesDataLoader>;
  deriveNodeViewState(
    path: string,
    attrs: LinesMetadata | undefined,
    opts: { applyPartialExtendTolerance: boolean }
  ): { skip: false; viewState: ViewState };
  processLines(
    path: string,
    data: LoadedLinesData,
    viewState: LinesViewState,
    session?: UpdateSession
  ): Promise<StagedLinesCommit | null>;
  commitLines(staged: StagedLinesCommit, session?: UpdateSession): void;
  /**
   * Profiler for background-pass accounting. Each per-loader refinement
   * step opens a 'LOD Refinement' pass root (its own persistent tree,
   * separate from 'Total Update') with a per-node child session threaded
   * through load → process → commit.
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

export async function runLinesRefinement(ctx: LinesRefinementCtx): Promise<void> {
  const objects =
    ctx.objects ??
    new Map([...ctx.loaders.keys()].map((path) => [path, ctx.rootGroup?.getObjectByName(path)]));
  const isPathVisible = (path: string): boolean => isObjectLoadEligible(objects.get(path));
  await runProgressiveRefinement({
    loaders: ctx.loaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      // Every progressive field is optional here because `linesLoaders` is
      // typed as plain `LinesDataLoader`: a non-progressive loader has no
      // ladder. `admitRefinementCandidate` gates on `hasMoreLODs`.
      const progressiveLoader = loader as LinesDataLoader & RefinableLoader;
      const admission = admitRefinementCandidate(
        path,
        progressiveLoader,
        ctx.residencyBudget,
        isPathVisible
      );
      if (!admission.admitted) return false;
      try {
        const mesh = objects.get(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as LinesMetadata | undefined;
        // Lines segment bounds already encode the extent, matching the
        // load-lines-node.ts convention.
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: PARTIAL_EXTEND_TOLERANCE.lines,
        });
        // A fully-extended node is a normal node with a slice-invariant query
        // (deriveNodeViewState), so it refines through this path like any other;
        // the `hasMoreLODs` gate above already stops a converged one.
        const linesVS: LinesViewState = refined.viewState;

        // Account this step to the 'LOD Refinement' tree (opened only when
        // real work happens, so empty sweeps don't record noise passes).
        const pass = ctx.profiler?.beginPass();
        const session = pass?.begin(`Lines (${path})`);
        try {
          const data = await progressiveLoader.updateView(
            linesVS,
            session,
            ctx.signal,
            admission.allowanceBytes
          );
          if (data) {
            let committed = false;
            try {
              const staged = await ctx.processLines(path, data, linesVS, session);
              // Superseded/disposed while we were loading + processing: an abort
              // landing during the async process round-trip is not a throw (so
              // the AbortError catch below misses it). Skip the commit so no
              // stale-slice geometry reaches the GPU — mirrors runAtomicCommit's
              // signal.aborted guard on the main path (atomic-commit.ts).
              if (staged && ctx.signal?.aborted !== true) {
                ctx.commitLines(staged, session);
                committed = true;
              }
            } finally {
              releaseLineageIfUncommitted(data, committed);
            }
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
      ctx.loaders as Map<string, LinesDataLoader & RefinableLoader>,
      ctx.residencyBudget,
      isPathVisible
    ),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
  });
}
