/**
 * Progressive Mesh LOD refinement — thin wrapper over the generic helper at
 * `data/scene-loader/progressive/refinement.ts`.
 *
 * The fourth member of the family (`data/points/`, `data/lines/`,
 * `data/gsplats/lod-refinement.ts`), and the loop that actually makes a reveal
 * ladder reveal: the initial load commits level 0 and this drains the rest a
 * level per pass, yielding to the render loop between them.
 *
 * Shaped after the LINES wrapper rather than the Points one, because mesh
 * likewise has an async project step between load and commit
 * (`processMeshData`). Its one mesh-specific wrinkle is that the projection
 * needs the node's `normal_dims` / `double_sided` / `extend_to_all` to decide
 * winding and the slab, and the refinement path — like the retry path — can only
 * reach them through the committed object's `userData`.
 *
 * @module data/mesh/lod-refinement
 */

import * as THREE from 'three';
import type { LoadedMeshData, MeshDataLoader, MeshMetadata, MeshViewState } from '../../types/mesh';
import type { StagedMeshCommit } from '../scene-loader/process/data-processor-mesh';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import type { ViewState } from '../data-loader-types';
import type { ViewStateQueue } from '../scene-loader/view-state/view-state-queue';
import type { RefinementResidencyBudget } from '../scene-loader/progressive/residency-budget';
import { runProgressiveRefinement } from '../scene-loader/progressive/refinement';
import { PARTIAL_EXTEND_TOLERANCE } from '../scene-loader/partial-extend-tolerance';
import {
  admitRefinementCandidate,
  failureTrackerFor,
  handleRefinementError,
  makeRefinementProgressCallbacks,
  recordRefinementResidency,
  type RefinableLoader,
} from '../scene-loader/progressive/refinement-wrapper';
import { isPartitionPathVisible } from '../scene-loader/loaders/run-loader-updates';

/** Geometry name in this wrapper's log lines and toasts. */
const LABEL = 'Mesh';

export interface MeshRefinementCtx {
  rootGroup: THREE.Group | null;
  viewStateQueue: ViewStateQueue;
  meshLoaders: Map<string, MeshDataLoader>;
  deriveNodeViewState(
    path: string,
    attrs: MeshMetadata | undefined,
    opts: { applyPartialExtendTolerance: boolean }
  ): { skip: false; viewState: ViewState };
  processMesh(
    path: string,
    data: LoadedMeshData,
    viewState: MeshViewState,
    attrs: Pick<MeshMetadata, 'normal_dims' | 'double_sided' | 'extend_to_all' | 'slab_tolerance'>
  ): Promise<StagedMeshCommit>;
  commitMesh(staged: StagedMeshCommit, session?: UpdateSession): void;
  /**
   * Profiler for background-pass accounting. Each per-loader refinement step
   * opens a 'LOD Refinement' pass root (its own persistent tree, separate from
   * 'Total Update') with a per-node child session threaded through load →
   * process → commit.
   */
  profiler?: UpdateProfiler | null;
  updateVisibleCountsInMonitor(): void;
  releaseLock(): void;
  retriggerUpdate(pendingState: Partial<ViewState>): void;
  /** Liveness check; false once the owning SceneLoader was disposed. */
  isActive?(): boolean;
  /**
   * Per-refinement-run abort signal. The orchestrator assigns the run's
   * controller to the SceneLoader's `_updateAbortController`, so a superseding
   * `updateView` (or dispose) aborts in-flight refinement reads MID-PASS instead
   * of waiting out the whole pass. An `AbortError` in the per-loader catch is
   * cancellation, not failure.
   */
  signal?: AbortSignal;
  /**
   * Shared residency ceiling for sweep-registered progressive leaves. Absent =
   * unbounded (today's behaviour).
   */
  residencyBudget?: RefinementResidencyBudget;
}

export async function runMeshRefinement(ctx: MeshRefinementCtx): Promise<void> {
  const isPathVisible = (path: string): boolean => isPartitionPathVisible(ctx.rootGroup, path);
  await runProgressiveRefinement({
    loaders: ctx.meshLoaders,
    viewStateQueue: ctx.viewStateQueue,
    isActive: ctx.isActive,
    processLoader: async (path, loader) => {
      // Every progressive field is optional here because `meshLoaders` is
      // typed as plain `MeshDataLoader`: a non-progressive loader has no
      // ladder. `admitRefinementCandidate` gates on `hasMoreLODs`.
      const progressiveLoader = loader as MeshDataLoader & RefinableLoader;
      const admission = admitRefinementCandidate(
        path,
        progressiveLoader,
        ctx.residencyBudget,
        isPathVisible
      );
      if (!admission.admitted) return false;
      try {
        const object = ctx.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
        const nodeAttrs = object?.userData?.attrs as MeshMetadata | undefined;
        // Mesh takes the partial-extend tolerance (only Lines opts out — its
        // segment bounds already encode the non-displayed extent). Matches
        // `load-mesh-node.ts` and the descriptor's `retryCommit`.
        const refined = ctx.deriveNodeViewState(path, nodeAttrs, {
          applyPartialExtendTolerance: PARTIAL_EXTEND_TOLERANCE.mesh,
        });
        const meshVS: MeshViewState = refined.viewState;

        // Account this step to the 'LOD Refinement' tree (opened only when real
        // work happens, so empty sweeps don't record noise passes).
        const pass = ctx.profiler?.beginPass();
        const session = pass?.begin(`Mesh (${path})`);
        try {
          const data = await progressiveLoader.updateView(
            meshVS,
            session,
            ctx.signal,
            admission.allowanceBytes
          );
          if (data) {
            const staged = await ctx.processMesh(path, data, meshVS, {
              normal_dims: nodeAttrs?.normal_dims,
              double_sided: nodeAttrs?.double_sided ?? true,
              extend_to_all: nodeAttrs?.extend_to_all,
              slab_tolerance: nodeAttrs?.slab_tolerance,
            });
            // Superseded/disposed while we were loading + projecting: an abort
            // landing during the async round-trip is not a throw (so the
            // AbortError catch below misses it). Skip the commit so no
            // stale-slice geometry reaches the GPU — mirrors runAtomicCommit's
            // signal.aborted guard on the main path (atomic-commit.ts).
            if (ctx.signal?.aborted !== true) ctx.commitMesh(staged, session);
          }
        } finally {
          session?.end();
          pass?.end();
        }
        failureTrackerFor(progressiveLoader).recordSuccess(path);
        return true;
      } catch (error) {
        return handleRefinementError(
          { label: LABEL, degradedState: 'showing a partial surface' },
          path,
          error,
          progressiveLoader
        );
      } finally {
        recordRefinementResidency(path, progressiveLoader, ctx.residencyBudget);
      }
    },
    ...makeRefinementProgressCallbacks(
      LABEL,
      ctx.meshLoaders as Map<string, MeshDataLoader & RefinableLoader>,
      ctx.residencyBudget,
      isPathVisible
    ),
    updateVisibleCountsInMonitor: () => ctx.updateVisibleCountsInMonitor(),
    releaseLock: () => ctx.releaseLock(),
    retriggerUpdate: (pending) => ctx.retriggerUpdate(pending),
  });
}
