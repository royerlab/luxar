/**
 * GSplats geometry-type handler — per-type wiring for the scene-loader's
 * load + stage phase. Mirrors `data/points/handler.ts` and
 * `data/lines/handler.ts` so all first-class geometry kinds share the
 * same loader/update shape.
 *
 * @module data/gsplats/handler
 */

import * as THREE from 'three';
import type { GeometryKind, ViewState } from '../data-loader-types';
import type {
  GSplatsDataLoader,
  GSplatsMetadata,
  GSplatsViewState,
  LoadedGSplatsData,
} from '../../types/gsplats';
import { log, Modules } from '../../utils/log';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import {
  processGSplatsData,
  type StagedGSplatsCommit,
} from '../scene-loader/process/data-processor-gsplats';
import { isAlreadyCommitted } from '../scene-loader/commit/noop-commit';

export const kind: GeometryKind = 'gsplats';
export const label = 'GSplats' as const;

export interface GSplatsHandlerCtx {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  clearFailure(path: string): void;
  currentVersion: number;
  updateVersion: number;
  extendedToleranceCache: Map<string, number[]>;
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: { applyPartialExtendTolerance: boolean; extendedToleranceCache?: Map<string, number[]> }
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };
  /** Per-update abort signal forwarded to `loader.updateView` (see DataLoader). */
  signal?: AbortSignal;
  /**
   * Per-tick LOD time budget during dimension-animation playback (see
   * `ViewState.frameBudgetMs`). Injected into the DERIVED per-node view
   * state below — a per-pass directive, so refinement/retry passes (which
   * derive independently) stay budget-free.
   */
  frameBudgetMs?: number;
}

/**
 * Async load + project + stage step for one GSplats node. Mirrors the
 * prior inline gsplats-branch of `updateView`.
 *
 * Like Points (and unlike Lines) gsplats uses
 * applyPartialExtendTolerance: true — verbatim from the original
 * behaviour.
 */
export async function loadAndStage(
  path: string,
  loader: GSplatsDataLoader,
  session: UpdateSession,
  ctx: GSplatsHandlerCtx
): Promise<StagedGSplatsCommit | null> {
  const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
  const attrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
  const derived = ctx.deriveNodeViewState(path, attrs, {
    applyPartialExtendTolerance: true,
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

  if (derived.skip) {
    log.info(
      Modules.SCENE_LOADER,
      `Skipping gsplats update for ${path} - all non-displayed dims are extended`
    );
    // Deliberately NOT marking healthy here: a skipped node loaded nothing, so a
    // previously-recorded failure is still unresolved. Clearing it would drop
    // the node from the retry set and hide a real breakage.
    session.markSkipped(derived.skip);
    // S6: see Points handler — drop prev to avoid stale extrap.
    ctx.viewStateQueue.forgetPath(path);
    return null;
  }
  // Playback frame budget rides the derived per-node view state (per-pass
  // directive; absent outside animation playback — see ctx.frameBudgetMs).
  const gsplatsViewState: GSplatsViewState =
    ctx.frameBudgetMs !== undefined
      ? { ...derived.viewState, frameBudgetMs: ctx.frameBudgetMs }
      : derived.viewState;
  const data: LoadedGSplatsData | null = await loader.updateView(
    gsplatsViewState,
    session,
    ctx.signal
  );
  if (!data) {
    markPathHealthy();
    return null;
  }
  // No-op fast path: the loader returned the SAME data reference it did
  // last commit (memoized progressive concat, unchanged view state) — the
  // GPU already holds exactly this data. Skip the expensive nD→3D
  // projection and stage a stamp-only commit (see noop-commit.ts).
  if (isAlreadyCommitted(mesh, data)) {
    markPathHealthy();
    session.setMetadata({ splats: data.splatCount, info: 'unchanged' });
    if (!ctx.signal?.aborted) {
      ctx.viewStateQueue.dispatchPrefetch(path, gsplatsViewState, loader);
    }
    return { path, noop: true, sourceData: data };
  }
  const staged = await processGSplatsData(
    path,
    data,
    gsplatsViewState,
    ctx.rootGroup,
    ctx.updateVersion,
    session
  );
  markPathHealthy();
  session.setMetadata({ splats: data.splatCount });
  // S6: per-loader predictive prefetch using the derived view-state — but
  // not for a SUPERSEDED update: extrapolating from an abandoned state warms
  // the wrong chunks and pollutes the per-path prefetch baseline.
  if (!ctx.signal?.aborted) {
    ctx.viewStateQueue.dispatchPrefetch(path, gsplatsViewState, loader);
  }
  return staged;
}
