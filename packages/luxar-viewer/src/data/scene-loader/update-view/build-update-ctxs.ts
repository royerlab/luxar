/**
 * Per-type handler-context construction for `updateView`.
 *
 * The three update branches (Points / Lines / GSplats) feed into
 * `data/{points,lines,gsplats}/handler.ts::loadAndStage`. Each handler
 * needs a small per-type ctx that shares most fields (rootGroup,
 * viewStateQueue, clearFailure, currentVersion, deriveNodeViewState)
 * plus a few type-specific bits (updateVersion for the version-gated
 * info log, the extendedToleranceCache shared between Points + GSplats).
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
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: DeriveOpts
  ): DerivedNodeViewState;
}

/**
 * Build the three per-type handler ctx objects from a shared input.
 * Returns them as `{ pointsCtx, linesCtx, gsplatsCtx }`.
 */
export function buildUpdateCtxs(input: UpdateCtxsInput): {
  pointsCtx: PointsHandlerCtx;
  linesCtx: LinesHandlerCtx;
  gsplatsCtx: GSplatsHandlerCtx;
} {
  const pointsCtx: PointsHandlerCtx = {
    rootGroup: input.rootGroup,
    viewStateQueue: input.viewStateQueue,
    clearFailure: input.clearFailure,
    currentVersion: input.currentVersion,
    extendedToleranceCache: input.extendedToleranceCache,
    signal: input.signal,
    deriveNodeViewState: input.deriveNodeViewState,
  };
  const linesCtx: LinesHandlerCtx = {
    rootGroup: input.rootGroup,
    viewStateQueue: input.viewStateQueue,
    clearFailure: input.clearFailure,
    currentVersion: input.currentVersion,
    updateVersion: input.updateVersion,
    signal: input.signal,
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
    deriveNodeViewState: input.deriveNodeViewState,
  };
  return { pointsCtx, linesCtx, gsplatsCtx };
}
