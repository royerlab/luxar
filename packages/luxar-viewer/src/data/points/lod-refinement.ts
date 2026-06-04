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
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import { runProgressiveRefinement } from '../scene-loader/progressive/refinement';

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
  updatePointsGeometry(path: string, data: LoadedPointsData): void;
  updateVisibleCountsInMonitor(): void;
  releaseLock(): void;
  retriggerUpdate(pendingState: Partial<ViewState>): void;
  /** Liveness check; false once the owning SceneLoader was disposed. */
  isActive?(): boolean;
}

export async function runPointsRefinement(ctx: PointsRefinementCtx): Promise<void> {
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
      try {
        const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = mesh?.userData?.attrs as PointsMetadata | undefined;
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: true,
        });
        if (refined.skip) return;
        const pointsVS: PointsViewState = refined.viewState;

        const data = await loader.updateView(pointsVS);
        if (data) {
          ctx.updatePointsGeometry(path, data);
        }
      } catch (error) {
        log.error(
          Modules.SCENE_LOADER,
          `Points refinement failed for ${path}: ${(error as Error).message}`
        );
      }
    },
    anyHasMoreLODs: () =>
      [...ctx.pointsLoaders.values()].some((l) => {
        const pl = l as PointsDataLoader & { hasMoreLODs?: boolean };
        return pl.hasMoreLODs === true;
      }),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
    onError: (error) =>
      log.error(Modules.SCENE_LOADER, `Points refinement loop error: ${(error as Error).message}`),
  });
}
