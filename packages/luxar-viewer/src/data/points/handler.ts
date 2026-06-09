/**
 * Points geometry-type handler — per-type wiring for the scene-loader's
 * load + stage phase.
 *
 * Handles the Points branch of `SceneLoader.updateView` as a focused
 * function. The Lines and GSplats handlers mirror this shape so all
 * first-class geometry kinds stay symmetrical.
 *
 * @module data/points/handler
 */

import * as THREE from 'three';
import type { DataLoader, GeometryKind, LoadedPointsData, ViewState } from '../data-loader-types';
import { log, Modules } from '../../utils/log';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';

export const kind: GeometryKind = 'points';
export const label = 'Points' as const;

/** Staged points data ready for GPU commit. Mirror of the inline shape in scene-loader.ts. */
export interface StagedPointsCommit {
  path: string;
  data: LoadedPointsData;
}

/** Bundle of host references the handler needs to do its work. */
export interface PointsHandlerCtx {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  /** Called on a successful load to clear the path's failure record. */
  clearFailure(path: string): void;
  currentVersion: number;
  /** Derives the per-node view-state (skip / partial-extend tolerance / nd_transform inverse). */
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: { applyPartialExtendTolerance: boolean; extendedToleranceCache?: Map<string, number[]> }
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };
  extendedToleranceCache: Map<string, number[]>;
  /** Per-update abort signal forwarded to `loader.updateView` (see DataLoader). */
  signal?: AbortSignal;
}

/**
 * Async load + stage step for one Points node. Mirrors the prior
 * inline points-branch of `updateView` line-for-line (skip path,
 * loader.updateView call, failure-record clear, first-update log,
 * metadata write, predictive-prefetch dispatch).
 *
 * Returns null on skip or no-data, otherwise returns staged commit data.
 */
export async function loadAndStage(
  path: string,
  loader: DataLoader,
  session: UpdateSession,
  ctx: PointsHandlerCtx
): Promise<StagedPointsCommit | null> {
  const obj = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
  const attrs = obj?.userData?.attrs as { extend_to_all?: string[] } | undefined;
  const derived = ctx.deriveNodeViewState(path, attrs, {
    applyPartialExtendTolerance: true,
    extendedToleranceCache: ctx.extendedToleranceCache,
  });
  if (derived.skip) {
    log.info(
      Modules.SCENE_LOADER,
      `Skipping update for ${path} - all non-displayed dims are extended`
    );
    session.markSkipped(derived.skip);
    // S6: drop the path from prev state so the next non-skip update
    // re-baselines rather than extrapolating from a stale snapshot.
    ctx.viewStateQueue.forgetPath(path);
    return null;
  }
  const data = await loader.updateView(derived.viewState, session, ctx.signal);
  ctx.clearFailure(path);
  if (!data) return null;
  if (ctx.currentVersion <= 1) {
    log.info(
      Modules.SCENE_LOADER,
      `[GEOM] v${ctx.currentVersion} points ${path}: ${data.pointCount} visible`
    );
  }
  session.setMetadata({ points: data.metadata.loadedPoints });
  // S6: per-loader predictive prefetch using the derived view-state.
  ctx.viewStateQueue.dispatchPrefetch(path, derived.viewState, loader);
  return { path, data };
}
