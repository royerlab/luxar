/**
 * Mesh geometry-type handler — per-type wiring for the scene-loader's
 * load + stage phase. Mirrors `data/points/handler.ts`,
 * `data/lines/handler.ts` and `data/gsplats/handler.ts` so all
 * first-class geometry kinds share the same loader/update shape.
 *
 * @module data/mesh/handler
 */

import * as THREE from 'three';
import type { GeometryKind, ViewState } from '../data-loader-types';
import type { MeshDataLoader, MeshMetadata, MeshViewState } from '../../types/mesh';
import { log, Modules } from '../../utils/log';
import type { UpdateSession } from '../../profiling/update-profiler';
import {
  processMeshData,
  type StagedMeshCommit,
} from '../scene-loader/process/data-processor-mesh';
import { PARTIAL_EXTEND_TOLERANCE } from '../scene-loader/partial-extend-tolerance';

export const kind: GeometryKind = 'mesh';
export const label = 'Mesh' as const;

export interface MeshHandlerCtx {
  rootGroup: THREE.Group | null;
  clearFailure(path: string): void;
  currentVersion: number;
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
 * Async load + project + stage step for one Mesh node.
 *
 * Like Points and GSplats (and unlike Lines), mesh widens tolerance across
 * dimensions the node only partially extends through; see
 * `PARTIAL_EXTEND_TOLERANCE`.
 *
 * DELIBERATE asymmetry vs the Lines/GSplats handlers: there is NO
 * handler-level `isAlreadyCommitted` no-op fast path here. The mesh loader
 * is whole-node resident and returns the SAME `LoadedMeshData` reference on
 * every `updateView`; what varies with the slice is only the nD-slab cull,
 * which lives downstream in `processMeshData` → `projectMeshTo3D`. A
 * reference-identity short-circuit would therefore skip re-projection on
 * every slice move — exactly the freeze this sweep exists to prevent. Every
 * sweep must re-project.
 *
 * Note the hazard is now LIVE rather than merely pointless: `commitMeshGeometry`
 * DOES stamp `committedData` (the depth-sort coordinator reads its presence as
 * "this index buffer still holds the commit whose ordering is resolving"), and
 * the stamped reference is the same one every sweep passes — so adding the fast
 * path here would match on the second sweep and freeze the mesh at its first
 * slice. The stamp is not an identity key for mesh; do not use it as one.
 */
export async function loadAndStage(
  path: string,
  loader: MeshDataLoader,
  session: UpdateSession,
  ctx: MeshHandlerCtx
): Promise<StagedMeshCommit | null> {
  const obj = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
  const meshAttrs = obj?.userData?.attrs as MeshMetadata | undefined;
  const derived = ctx.deriveNodeViewState(
    path,
    { extend_to_all: meshAttrs?.extend_to_all },
    {
      applyPartialExtendTolerance: PARTIAL_EXTEND_TOLERANCE.mesh,
      extendedToleranceCache: ctx.extendedToleranceCache,
    }
  );
  /**
   * Mark this path healthy. Called at every terminal success, NOT right after
   * the fetch: the failure record's scope is the whole `loadAndStage` step (see
   * `run-loader-updates`' catch), so clearing after the fetch alone meant a
   * post-fetch failure re-recorded with `retryCount` 0 — pinning the log at
   * "(attempt 1)" forever — and left `hasFailures()` briefly reporting clean.
   */
  const markPathHealthy = (): void => ctx.clearFailure(path);

  // Playback frame budget rides the derived per-node view state (per-pass
  // directive; absent outside animation playback — see ctx.frameBudgetMs).
  const meshViewState: MeshViewState =
    ctx.frameBudgetMs !== undefined || ctx.ladderDepth !== undefined
      ? { ...derived.viewState, frameBudgetMs: ctx.frameBudgetMs, ladderDepth: ctx.ladderDepth }
      : derived.viewState;
  const data = await loader.updateView(meshViewState, session, ctx.signal);
  if (!data) {
    // The mesh loader is whole-node resident and always returns data in
    // practice; the guard is kept symmetric with the sibling handlers.
    markPathHealthy();
    return null;
  }
  if (ctx.currentVersion <= 1) {
    log.info(
      Modules.SCENE_LOADER,
      `[GEOM] v${ctx.currentVersion} mesh ${path}: ${data.faceCount} faces`
    );
  }
  const staged = await processMeshData(path, data, meshViewState, {
    normal_dims: meshAttrs?.normal_dims,
    double_sided: meshAttrs?.double_sided ?? true,
    // Required, not a ride-along: `processMeshData` RECOMPUTES the membership
    // tolerance from dimension metadata (discarding the derived view state's
    // extended tolerance), then re-applies extend_to_all from these attrs. An
    // extended mesh would otherwise commit on first load but cull on the first
    // slice move, when this sweep re-projects it.
    extend_to_all: meshAttrs?.extend_to_all,
    slab_tolerance: meshAttrs?.slab_tolerance,
  });
  markPathHealthy();
  // The typed per-type counter, not a free-text `info` string: it gets the same
  // formatted tag as the points/lines/gsplats rows AND it SUMS when several mesh
  // layers merge into one aggregated row (a string is last-write, so the row used
  // to report whichever mesh finished last). Loaded faces, matching what the
  // siblings report (loaded points / segments / splats) — on a reveal ladder that
  // is the revealed prefix's total.
  session.setMetadata({ triangles: data.faceCount });
  // No predictive prefetch: a mesh is whole-node resident, so there is no
  // chunk subset to warm — the whole payload is already in hand.
  return staged;
}
