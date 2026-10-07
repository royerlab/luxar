/**
 * Lines geometry-type handler — per-type wiring for the scene-loader's
 * load + stage phase. Mirrors `data/points/handler.ts` and
 * `data/gsplats/handler.ts` so all first-class geometry kinds share the
 * same loader/update shape.
 *
 * @module data/lines/handler
 */

import { findObjectByName } from '../../utils/scene-graph-index';
import { withPassDirectives, type PassDirectives } from '../loaders/pass-directives';
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
import { PARTIAL_EXTEND_TOLERANCE } from '../scene-loader/partial-extend-tolerance';

export const kind: GeometryKind = 'lines';
export const label = 'Lines' as const;

export interface LinesHandlerCtx extends PassDirectives {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  clearFailure(path: string): void;
  currentVersion: number;
  /**
   * Passes run so far — gates the first-update `[GEOM]` logs here and in the
   * data processor (the view version does not move on a same-view pass).
   */
  updateVersion: number;
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: { applyPartialExtendTolerance: boolean; extendedToleranceCache?: Map<string, number[]> }
  ): { skip: false; viewState: ViewState };
  /** Per-update abort signal forwarded to `loader.updateView` (see DataLoader). */
  signal?: AbortSignal;
}

/**
 * Async load + project + stage step for one Lines node. Mirrors the
 * prior inline lines-branch of `updateView`.
 *
 * Unlike Points and GSplats, line bounds already encode the non-displayed
 * extent. The shared extendedToleranceCache parameter is therefore unused
 * here; lines read raw tolerance from the global view-state.
 */
export async function loadAndStage(
  path: string,
  loader: LinesDataLoader,
  session: UpdateSession,
  ctx: LinesHandlerCtx
): Promise<StagedLinesCommit | null> {
  const mesh = findObjectByName(ctx.rootGroup, path) as THREE.Mesh | undefined;
  const attrs = mesh?.userData?.attrs as { extend_to_all?: string[] } | undefined;
  const derived = ctx.deriveNodeViewState(path, attrs, {
    applyPartialExtendTolerance: PARTIAL_EXTEND_TOLERANCE.lines,
  });
  /**
   * Mark this path healthy. Called at every terminal success, NOT right after
   * the fetch: the failure record's scope is the whole `loadAndStage` step (see
   * `run-loader-updates`' catch), so clearing after the fetch alone meant a
   * post-fetch failure re-recorded with `retryCount` 0 — pinning the log at
   * "(attempt 1)" forever — and left `hasFailures()` briefly reporting clean.
   */
  const markPathHealthy = (): void => ctx.clearFailure(path);

  // A fully-extended node is derived as a normal node with a slice-INVARIANT
  // query (see deriveNodeViewState — the full-extend tolerance is computed even
  // though lines opt out of the PARTIAL override), so it flows through the
  // standard load path below. No skip shortcut.
  // Playback directives ride the derived per-node view state (per-pass;
  // absent outside animation playback — see `PassDirectives`).
  const linesViewState: LinesViewState = withPassDirectives(derived.viewState, ctx);
  const data: LoadedLinesData | null = await loader.updateView(linesViewState, session, ctx.signal);
  if (!data) {
    markPathHealthy();
    return null;
  }
  // No-op fast path: the loader returned the SAME data reference it did
  // last commit (memoized progressive concat, unchanged view state) — the
  // GPU already holds exactly this data. Skip the expensive projection and
  // stage a stamp-only commit (see noop-commit.ts).
  if (isAlreadyCommitted(mesh, data)) {
    markPathHealthy();
    session.setMetadata({
      segments: data.segments ? data.segments.length / 2 : 0,
      info: 'unchanged',
    });
    if (!ctx.signal?.aborted) {
      ctx.viewStateQueue.dispatchPrefetch(path, linesViewState, loader);
    }
    return { path, noop: true, sourceData: data };
  }
  if (ctx.updateVersion <= 1) {
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
    session,
    ctx.signal
  );
  markPathHealthy();
  session.setMetadata({ segments: data.segments ? data.segments.length / 2 : 0 });
  // S6: per-loader predictive prefetch using the derived view-state — but
  // not for a SUPERSEDED update: extrapolating from an abandoned state warms
  // the wrong chunks and pollutes the per-path prefetch baseline.
  if (!ctx.signal?.aborted) {
    ctx.viewStateQueue.dispatchPrefetch(path, linesViewState, loader);
  }
  return staged;
}
