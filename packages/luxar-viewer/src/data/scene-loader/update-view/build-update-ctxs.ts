/**
 * Per-type handler-context construction for `updateView`.
 *
 * The four update branches (Points / Lines / GSplats / Mesh) feed into
 * `data/{points,lines,gsplats,mesh}/handler.ts::loadAndStage`. Each handler
 * needs a small per-type ctx that shares most fields (rootGroup,
 * viewStateQueue, clearFailure, currentVersion, deriveNodeViewState)
 * plus a few type-specific bits (updateVersion for the version-gated
 * info log, the extendedToleranceCache shared between Points + GSplats + Mesh).
 *
 * Centralising the construction here keeps `updateView` focused on
 * orchestration and avoids three near-identical literal blocks in the
 * same method.
 */

import type * as THREE from 'three';
import type { ViewStateQueue } from '../view-state/view-state-queue';
import type { DeriveOpts, DerivedNodeViewState } from '../view-state/derive-node-view-state';
import type { PointsHandlerCtx } from '../../points/handler';
import type { LinesHandlerCtx } from '../../lines/handler';
import type { GSplatsHandlerCtx } from '../../gsplats/handler';
import type { MeshHandlerCtx } from '../../mesh/handler';

/**
 * Common inputs every per-type handler ctx shares.
 */
export interface UpdateCtxsInput {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  clearFailure(path: string): void;
  currentVersion: number;
  updateVersion: number;
  extendedToleranceCache: Map<string, number[]>;
  /** Per-update abort signal; forwarded into each per-type handler ctx. */
  signal?: AbortSignal;
  /**
   * Per-tick LOD time budget during dimension-animation playback (see
   * `ViewState.frameBudgetMs`). A per-pass directive: forwarded into each
   * handler ctx and injected into the DERIVED per-node view state there —
   * never persisted on the scene loader.
   */
  frameBudgetMs?: number;
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: DeriveOpts
  ): DerivedNodeViewState;
}

/**
 * Build the four per-type handler ctx objects from a shared input.
 * Returns them as `{ pointsCtx, linesCtx, gsplatsCtx, meshCtx }`.
 */
export function buildUpdateCtxs(input: UpdateCtxsInput): {
  pointsCtx: PointsHandlerCtx;
  linesCtx: LinesHandlerCtx;
  gsplatsCtx: GSplatsHandlerCtx;
  meshCtx: MeshHandlerCtx;
} {
  const pointsCtx: PointsHandlerCtx = {
    rootGroup: input.rootGroup,
    viewStateQueue: input.viewStateQueue,
    clearFailure: input.clearFailure,
    currentVersion: input.currentVersion,
    extendedToleranceCache: input.extendedToleranceCache,
    signal: input.signal,
    frameBudgetMs: input.frameBudgetMs,
    deriveNodeViewState: input.deriveNodeViewState,
  };
  const linesCtx: LinesHandlerCtx = {
    rootGroup: input.rootGroup,
    viewStateQueue: input.viewStateQueue,
    clearFailure: input.clearFailure,
    currentVersion: input.currentVersion,
    updateVersion: input.updateVersion,
    signal: input.signal,
    frameBudgetMs: input.frameBudgetMs,
    deriveNodeViewState: input.deriveNodeViewState,
  };
  const gsplatsCtx: GSplatsHandlerCtx = {
    rootGroup: input.rootGroup,
    viewStateQueue: input.viewStateQueue,
    clearFailure: input.clearFailure,
    currentVersion: input.currentVersion,
    updateVersion: input.updateVersion,
    extendedToleranceCache: input.extendedToleranceCache,
    signal: input.signal,
    frameBudgetMs: input.frameBudgetMs,
    deriveNodeViewState: input.deriveNodeViewState,
  };
  const meshCtx: MeshHandlerCtx = {
    rootGroup: input.rootGroup,
    clearFailure: input.clearFailure,
    currentVersion: input.currentVersion,
    extendedToleranceCache: input.extendedToleranceCache,
    signal: input.signal,
    frameBudgetMs: input.frameBudgetMs,
    deriveNodeViewState: input.deriveNodeViewState,
  };
  return { pointsCtx, linesCtx, gsplatsCtx, meshCtx };
}
