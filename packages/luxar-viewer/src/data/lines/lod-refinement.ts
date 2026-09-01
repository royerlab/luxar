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
import { log, Modules } from '../../utils/log';
import { notifier } from '../../utils/cross-layer/notifier';
import { isAbortError } from '../loaders/abort-error';
import { tryRollbackToPassStart } from '../loaders/progressive/pass-rollback';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type { StagedLinesCommit } from '../scene-loader/process/data-processor-lines';
import {
  MAX_CONSECUTIVE_REFINEMENT_FAILURES,
  RefinementFailureTracker,
  runProgressiveRefinement,
} from '../scene-loader/progressive/refinement';

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
      const progressiveLoader = loader as LinesDataLoader & {
        hasMoreLODs?: boolean;
        // Optional because `linesLoaders` is typed as plain `LinesDataLoader`:
        // a non-progressive loader has no ladder to unwind, and the
        // `hasMoreLODs` gate above has already returned for it. Every real
        // `LinesProgressiveLoader` implements it.
        rollbackToPassStart?: () => number;
      };
      if (progressiveLoader.hasMoreLODs !== true) return false;
      if (failures.isExhausted(path)) return false;
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
          const data = await loader.updateView(linesVS, session, ctx.signal);
          if (data) {
            const staged = await ctx.processLines(path, data, linesVS, session);
            // Superseded/disposed while we were loading + processing: an abort
            // landing during the async process round-trip is not a throw (so
            // the AbortError catch below misses it). Skip the commit so no
            // stale-slice geometry reaches the GPU — mirrors runAtomicCommit's
            // signal.aborted guard on the main path (atomic-commit.ts).
            if (staged && ctx.signal?.aborted !== true) ctx.commitLines(staged, session);
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
        // Unwind the levels this pass appended before the throw. `updateView`
        // advances the ladder cursor as each level arrives, so without this the
        // retry would resume from the ADVANCED cursor and attempt a strictly
        // larger allocation than the one that just failed — and on reaching
        // the last rung would flip `hasMoreLODs` false, silently stranding the
        // node at its last committed prefix with the failure cap never reached.
        // See `../loaders/progressive/pass-rollback`.
        const unwound = tryRollbackToPassStart(progressiveLoader);
        if (failures.recordFailure(path)) {
          log.error(
            Modules.SCENE_LOADER,
            `Lines refinement failed for ${path}: ${(error as Error).message} — ` +
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
            `Lines refinement failed for ${path}: ${(error as Error).message} ` +
              `(unwound ${unwound} level(s))`
          );
        }
        return false;
      }
    },
    getLoaderProgress: (path, loader) => {
      const progressiveLoader = loader as LinesDataLoader & {
        hasMoreLODs?: boolean;
        loadedLODCount: number;
        totalLODCount: number;
      };
      if (failures.isExhausted(path) || progressiveLoader.hasMoreLODs !== true) return null;
      return {
        loaded: progressiveLoader.loadedLODCount,
        total: progressiveLoader.totalLODCount,
      };
    },
    anyHasMoreLODs: () =>
      [...ctx.linesLoaders.entries()].some(([path, l]) => {
        const ll = l as LinesDataLoader & { hasMoreLODs?: boolean };
        return !failures.isExhausted(path) && ll.hasMoreLODs === true;
      }),
    onNoProgress: (stalled) => {
      for (const { path, loaded, total } of stalled) {
        log.warning(
          Modules.SCENE_LOADER,
          `Lines refinement stopped for ${path}: no progress at LOD ${loaded}/${total}`
        );
      }
    },
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
    onError: (error) =>
      log.error(Modules.SCENE_LOADER, `Lines refinement loop error: ${(error as Error).message}`),
  });
}
