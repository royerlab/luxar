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
import { PARTIAL_EXTEND_TOLERANCE } from '../scene-loader/partial-extend-tolerance';

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
  ): { skip: false; viewState: ViewState };
  extendedToleranceCache: Map<string, number[]>;
  /** Per-update abort signal forwarded to `loader.updateView` (see DataLoader). */
  signal?: AbortSignal;
  /**
   * Per-tick LOD time budget during dimension-animation playback (see
   * `ViewState.frameBudgetMs`). Injected into the DERIVED per-node view
   * state below — a per-pass directive, so refinement/retry passes (which
   * derive independently) stay budget-free.
   */
  frameBudgetMs?: number;
  /**
   * Pinned playback ladder depth (see `ViewState.ladderDepth`). Same per-pass
   * contract as `frameBudgetMs`: injected into the DERIVED view state only.
   */
  ladderDepth?: number;
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
    applyPartialExtendTolerance: PARTIAL_EXTEND_TOLERANCE.points,
    extendedToleranceCache: ctx.extendedToleranceCache,
  });
  /**
   * Mark this path healthy. Called at every terminal success, NOT right after
   * the fetch: the failure record's scope is the whole `loadAndStage` step (see
   * `run-loader-updates`' catch), so clearing after the fetch alone meant a
   * post-fetch failure re-recorded with `retryCount` 0 — pinning the log at
   * "(attempt 1)" forever — and left `hasFailures()` briefly reporting clean.
   */
  const markPathHealthy = (): void => ctx.clearFailure(path);

  // A fully-extended (`extend_to_all` covers all non-displayed dims) node is
  // derived as a normal node with a slice-INVARIANT query, so it flows through
  // the standard load path below: the first sweep fetches it and every later
  // sweep hits the loader's same-view memoized no-op. No skip shortcut.
  // Playback frame budget rides the derived per-node view state (per-pass
  // directive; absent outside animation playback — see ctx.frameBudgetMs).
  const pointsViewState: ViewState =
    ctx.frameBudgetMs !== undefined || ctx.ladderDepth !== undefined
      ? { ...derived.viewState, frameBudgetMs: ctx.frameBudgetMs, ladderDepth: ctx.ladderDepth }
      : derived.viewState;
  const data = await loader.updateView(pointsViewState, session, ctx.signal);
  if (!data) {
    markPathHealthy();
    return null;
  }
  if (ctx.currentVersion <= 1) {
    log.info(
      Modules.SCENE_LOADER,
      `[GEOM] v${ctx.currentVersion} points ${path}: ${data.pointCount} visible`
    );
  }
  markPathHealthy();
  session.setMetadata({ points: data.metadata.loadedPoints });
  // S6: per-loader predictive prefetch using the derived view-state — but
  // not for a SUPERSEDED update: extrapolating from an abandoned state warms
  // the wrong chunks and pollutes the per-path prefetch baseline.
  if (!ctx.signal?.aborted) {
    ctx.viewStateQueue.dispatchPrefetch(path, pointsViewState, loader);
  }
  // DELIBERATE asymmetry vs the Lines/GSplats handlers: no handler-level
  // `isAlreadyCommitted` fast path here. Those handlers must short-circuit
  // BEFORE their `process*` step to skip the worker projection on a
  // same-reference revisit; Points has no process step (projection is folded
  // into the loader and already skipped by the S-cache hit), so its
  // reference-identity no-op lives at the earliest place it can save work —
  // the commit (`commit-points-geometry.ts::isAlreadyCommitted`).
  return { path, data };
}
