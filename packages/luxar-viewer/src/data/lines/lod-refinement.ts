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
import {
  RefinementFailureTracker,
  runProgressiveRefinement,
} from '../scene-loader/progressive/refinement';
import {
  admitRefinementCandidate,
  handleRefinementError,
  makeRefinementProgressCallbacks,
  recordRefinementResidency,
  type RefinableLoader,
} from '../scene-loader/progressive/refinement-wrapper';

/** Geometry name in this wrapper's log lines and toasts. */
const LABEL = 'Lines';

export interface LinesRefinementCtx {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  linesLoaders: Map<string, LinesDataLoader>;
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
  // Per-run failure backoff: a loader that fails MAX_CONSECUTIVE times is
  // excluded for the rest of this run (and from anyHasMoreLODs, so the loop
  // can terminate) instead of retrying at frame rate forever.
  const failures = new RefinementFailureTracker();
  await runProgressiveRefinement({
    loaders: ctx.linesLoaders,
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
        failures,
        ctx.residencyBudget
      );
      if (!admission.admitted) return false;
      try {
        const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as LinesMetadata | undefined;
        // Lines: applyPartialExtendTolerance=false (segment bounds already
        // encode the extent — matches the load-lines-node.ts convention).
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: false,
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
            admission?.allowanceBytes ?? undefined
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
        failures.recordSuccess(path);
        return true;
      } catch (error) {
        return handleRefinementError(LABEL, path, error, progressiveLoader, failures);
      } finally {
        recordRefinementResidency(path, progressiveLoader, ctx.residencyBudget);
      }
    },
    ...makeRefinementProgressCallbacks(
      LABEL,
      ctx.linesLoaders as Map<string, LinesDataLoader & RefinableLoader>,
      failures,
      ctx.residencyBudget
    ),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
  });
}
