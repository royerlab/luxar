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
  // Per-run failure backoff: a loader that fails MAX_CONSECUTIVE times is
  // excluded for the rest of this run (and from anyHasMoreLODs, so the loop
  // can terminate) instead of retrying at frame rate forever.
  const failures = new RefinementFailureTracker();
  await runProgressiveRefinement({
    loaders: ctx.gsplatLoaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      if (loader.hasMoreLODs !== true) return false;
      if (failures.isExhausted(path)) return false;
      // Optional for the same reason as `hasMoreLODs` on the interface: only a
      // progressive loader has a ladder to unwind. Every
      // `GSplatsProgressiveLoader` implements it.
      const progressiveLoader = loader as GSplatsDataLoader & {
        rollbackToPassStart?: () => number;
        ladderResidency?: () => LadderResidency;
        updateView: (
          viewState: GSplatsViewState,
          session?: UpdateSession,
          signal?: AbortSignal,
          residencyAllowanceBytes?: number
        ) => Promise<LoadedGSplatsData>;
      };
      // Shared sweep residency ceiling. Declining here is not enough on its own —
      // `anyHasMoreLODs` and `getLoaderProgress` below must also exclude
      // declined paths, or the loop re-offers this loader every frame forever
      // while holding the update lock (the trap `isExhausted` already avoids).
      const residency = progressiveLoader.ladderResidency?.();
      const admission = residency ? ctx.residencyBudget?.admit(path, residency) : undefined;
      if (admission?.admitted === false) {
        return false;
      }
      try {
        const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: true,
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
            admission?.allowanceBytes ?? undefined
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
            `GSplats refinement failed for ${path}: ${(error as Error).message} — ` +
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
            `GSplats refinement failed for ${path}: ${(error as Error).message} ` +
              `(unwound ${unwound} level(s))`
          );
        }
        return false;
      } finally {
        const measured = progressiveLoader.ladderResidency?.();
        if (measured) ctx.residencyBudget?.record(path, measured);
      }
    },
    getLoaderProgress: (path, loader) => {
      const progressiveLoader = loader as GSplatsDataLoader & {
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
      [...ctx.gsplatLoaders.entries()].some(
        ([path, l]) =>
          !failures.isExhausted(path) &&
          ctx.residencyBudget?.isDeclined(path) !== true &&
          l.hasMoreLODs === true
      ),
    onNoProgress: (stalled) => {
      for (const { path, loaded, total } of stalled) {
        log.warning(
          Modules.SCENE_LOADER,
          `GSplats refinement stopped for ${path}: no progress at LOD ${loaded}/${total}`
        );
      }
    },
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
    onError: (error) =>
      log.error(Modules.SCENE_LOADER, `GSplats refinement loop error: ${(error as Error).message}`),
  });
}
