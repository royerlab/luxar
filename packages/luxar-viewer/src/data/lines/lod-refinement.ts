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
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type { StagedLinesCommit } from '../scene-loader/process/data-processor-lines';
import { runProgressiveRefinement } from '../scene-loader/progressive/refinement';

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
    viewState: LinesViewState
  ): Promise<StagedLinesCommit | null>;
  commitLines(staged: StagedLinesCommit): void;
  updateVisibleCountsInMonitor(): void;
  releaseLock(): void;
  retriggerUpdate(pendingState: Partial<ViewState>): void;
  /** Liveness check; false once the owning SceneLoader was disposed. */
  isActive?(): boolean;
}

export async function runLinesRefinement(ctx: LinesRefinementCtx): Promise<void> {
  await runProgressiveRefinement({
    loaders: ctx.linesLoaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      const progressiveLoader = loader as LinesDataLoader & {
        hasMoreLODs?: boolean;
      };
      if (progressiveLoader.hasMoreLODs !== true) return;
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

        const data = await loader.updateView(linesVS);
        if (data) {
          const staged = await ctx.processLines(path, data, linesVS);
          if (staged) ctx.commitLines(staged);
        }
      } catch (error) {
        log.error(
          Modules.SCENE_LOADER,
          `Lines refinement failed for ${path}: ${(error as Error).message}`
        );
      }
    },
    anyHasMoreLODs: () =>
      [...ctx.linesLoaders.values()].some((l) => {
        const ll = l as LinesDataLoader & { hasMoreLODs?: boolean };
        return ll.hasMoreLODs === true;
      }),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
  });
}
