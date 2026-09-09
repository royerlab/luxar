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
import { releaseLineageIfUncommitted } from '../../types/prefix-lineage';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type { RefinementResidencyBudget } from '../scene-loader/progressive/residency-budget';
import type { StagedGSplatsCommit } from '../scene-loader/process/data-processor-gsplats';
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
const LABEL = 'GSplats';

/**
 * Bundle of host references the refinement loop needs. Kept narrow so
 * the loop can be tested with a stub.
 */
export interface GSplatsRefinementCtx {
  rootGroup: THREE.Group | null;
  /** Scene objects already resolved by the phase eligibility sweep. */
  objects?: ReadonlyMap<string, THREE.Object3D | undefined>;
  viewStateQueue: ViewStateQueue;
  loaders: Map<string, GSplatsDataLoader>;
  deriveNodeViewState(
    path: string,
    attrs: GSplatsMetadata | undefined,
    opts: { applyPartialExtendTolerance: boolean }
  ): { skip: false; viewState: ViewState };
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

/**
 * Run the GSplats LOD refinement loop. Thin wrapper over the generic
 * :func:`runProgressiveRefinement` helper with gsplats-specific
 * derive / process / commit closures.
 */
export async function runGSplatsRefinement(ctx: GSplatsRefinementCtx): Promise<void> {
  const objects =
    ctx.objects ??
    new Map([...ctx.loaders.keys()].map((path) => [path, ctx.rootGroup?.getObjectByName(path)]));
  const isPathVisible = (path: string): boolean => isObjectLoadEligible(objects.get(path));
  await runProgressiveRefinement({
    loaders: ctx.loaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      // Every progressive field is optional here because `gsplatLoaders` is
      // typed as plain `GSplatsDataLoader`: a non-progressive loader has no
      // ladder. `admitRefinementCandidate` gates on `hasMoreLODs`.
      const progressiveLoader = loader as GSplatsDataLoader & RefinableLoader;
      const admission = admitRefinementCandidate(
        path,
        progressiveLoader,
        ctx.residencyBudget,
        isPathVisible
      );
      if (!admission.admitted) return false;
      try {
        const mesh = objects.get(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: PARTIAL_EXTEND_TOLERANCE.gsplats,
        });
        // A fully-extended node is a normal node with a slice-invariant query
        // (deriveNodeViewState), so it refines through this path like any other;
        // the `hasMoreLODs` gate above already stops a converged one.
        const gsplatsViewState: GSplatsViewState = refined.viewState;

        // Account this step to the 'LOD Refinement' tree (opened only when
        // real work happens, so empty sweeps don't record noise passes).
        const pass = ctx.profiler?.beginPass();
        const session = pass?.begin(`GSplats (${path})`);
        try {
          const data = await progressiveLoader.updateView(
            gsplatsViewState,
            session,
            ctx.signal,
            admission.allowanceBytes
          );
          if (data) {
            let committed = false;
            try {
              const staged = await ctx.processGSplats(path, data, gsplatsViewState, session);
              // Superseded/disposed while we were loading + processing: an abort
              // landing during the async process round-trip is not a throw (so
              // the AbortError catch below misses it). Skip the commit so no
              // stale-slice geometry reaches the GPU — mirrors runAtomicCommit's
              // signal.aborted guard on the main path (atomic-commit.ts).
              if (staged && ctx.signal?.aborted !== true) {
                ctx.commitGSplats(staged, session);
                committed = true;
              }
            } finally {
              // A skipped commit never reaches the commit layer's own clear, and
              // the pinned parent is the previous cumulative — worth up to a
              // whole redundant copy on a deep ladder. See prefix-lineage.ts.
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
      ctx.loaders as Map<string, GSplatsDataLoader & RefinableLoader>,
      ctx.residencyBudget,
      isPathVisible
    ),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
  });
}
