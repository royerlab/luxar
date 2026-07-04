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
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };
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
      };
      if (progressiveLoader.hasMoreLODs !== true) return;
      if (failures.isExhausted(path)) return;
      try {
        const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as LinesMetadata | undefined;
        // Lines: applyPartialExtendTolerance=false (segment bounds already
        // encode the extent — matches the load-lines-node.ts convention).
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: false,
        });
        if (refined.skip) return;
        const linesVS: LinesViewState = refined.viewState;

        // Account this step to the 'LOD Refinement' tree (opened only when
        // real work happens, so empty sweeps don't record noise passes).
        const pass = ctx.profiler?.beginPass();
        const session = pass?.begin(`Lines (${path})`);
        try {
          const data = await loader.updateView(linesVS, session);
          if (data) {
            const staged = await ctx.processLines(path, data, linesVS, session);
            if (staged) ctx.commitLines(staged, session);
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
            `Lines refinement failed for ${path}: ${(error as Error).message} — ` +
              `giving up after ${MAX_CONSECUTIVE_REFINEMENT_FAILURES} consecutive failures ` +
              '(will retry on the next view change)'
          );
        } else {
          log.error(
            Modules.SCENE_LOADER,
            `Lines refinement failed for ${path}: ${(error as Error).message}`
          );
        }
      }
    },
    anyHasMoreLODs: () =>
      [...ctx.linesLoaders.entries()].some(([path, l]) => {
        const ll = l as LinesDataLoader & { hasMoreLODs?: boolean };
        return !failures.isExhausted(path) && ll.hasMoreLODs === true;
      }),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
    onError: (error) =>
      log.error(Modules.SCENE_LOADER, `Lines refinement loop error: ${(error as Error).message}`),
  });
}
