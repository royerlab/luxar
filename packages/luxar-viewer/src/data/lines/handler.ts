/**
 * Lines geometry-type handler — per-type wiring for the scene-loader's
 * load + stage phase. Mirrors `data/points/handler.ts` and
 * `data/gsplats/handler.ts` so all first-class geometry kinds share the
 * same loader/update shape.
 *
 * @module data/lines/handler
 */

import * as THREE from 'three';
import type { GeometryKind, ViewState } from '../data-loader-types';
import type { LinesDataLoader, LinesViewState, LoadedLinesData } from '../../types/lines';
import { log, Modules } from '../../utils/log';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import {
  processLinesData,
  type StagedLinesCommit,
} from '../scene-loader/process/data-processor-lines';
import { isAlreadyCommitted } from '../scene-loader/commit/noop-commit';

export const kind: GeometryKind = 'lines';
export const label = 'Lines' as const;

export interface LinesHandlerCtx {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  clearFailure(path: string): void;
  currentVersion: number;
  /** Forwarded to the data processor so its first-update logs are version-gated. */
  updateVersion: number;
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: { applyPartialExtendTolerance: boolean; extendedToleranceCache?: Map<string, number[]> }
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };
  /** Per-update abort signal forwarded to `loader.updateView` (see DataLoader). */
  signal?: AbortSignal;
}

/**
 * Async load + project + stage step for one Lines node. Mirrors the
 * prior inline lines-branch of `updateView`.
 *
 * Note: lines uses applyPartialExtendTolerance: false (unlike Points
 * and GSplats) — verbatim from the original behaviour. The shared
 * extendedToleranceCache parameter is therefore unused here; lines
 * read raw tolerance from the global view-state.
 */
export async function loadAndStage(
  path: string,
  loader: LinesDataLoader,
  session: UpdateSession,
  ctx: LinesHandlerCtx
): Promise<StagedLinesCommit | null> {
  const mesh = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
  const attrs = mesh?.userData?.attrs as { extend_to_all?: string[] } | undefined;
  const derived = ctx.deriveNodeViewState(path, attrs, {
    applyPartialExtendTolerance: false,
  });
  if (derived.skip) {
    log.info(
      Modules.SCENE_LOADER,
      `Skipping update for ${path} - all non-displayed dims are extended`
    );
    session.markSkipped(derived.skip);
    // S6: see Points handler — drop prev to avoid stale extrap.
    ctx.viewStateQueue.forgetPath(path);
    return null;
  }
  const linesViewState: LinesViewState = derived.viewState;
  const data: LoadedLinesData | null = await loader.updateView(linesViewState, session, ctx.signal);
  ctx.clearFailure(path);
  if (!data) return null;
  // No-op fast path: the loader returned the SAME data reference it did
  // last commit (memoized progressive concat, unchanged view state) — the
  // GPU already holds exactly this data. Skip the expensive projection and
  // stage a stamp-only commit (see noop-commit.ts).
  if (isAlreadyCommitted(mesh?.userData, data)) {
    session.setMetadata({
      segments: data.segments ? data.segments.length / 2 : 0,
      info: 'unchanged',
    });
    if (!ctx.signal?.aborted) {
      ctx.viewStateQueue.dispatchPrefetch(path, linesViewState, loader);
    }
    return { path, noop: true, sourceData: data };
  }
  if (ctx.currentVersion <= 1) {
    log.info(
      Modules.SCENE_LOADER,
      `[GEOM] v${ctx.currentVersion} lines ${path}: ${data.segmentCount} loaded`
    );
  }
  const staged = await processLinesData(
    path,
    data,
    linesViewState,
    ctx.rootGroup,
    ctx.updateVersion,
    session
  );
  session.setMetadata({ segments: data.segments ? data.segments.length / 2 : 0 });
  // S6: per-loader predictive prefetch using the derived view-state — but
  // not for a SUPERSEDED update: extrapolating from an abandoned state warms
  // the wrong chunks and pollutes the per-path prefetch baseline.
  if (!ctx.signal?.aborted) {
    ctx.viewStateQueue.dispatchPrefetch(path, linesViewState, loader);
  }
  return staged;
}
