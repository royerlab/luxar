/**
 * GSplats geometry-commit concern, mirroring `commit-points-geometry.ts`
 * and `commit-lines-geometry.ts`.
 *
 * Keeps the synchronous GSplats GPU commit in a focused module. The
 * async projection step stays in `data-processor-gsplats.ts` because it
 * has no Points counterpart.
 *
 * `StagedGSplatsCommit` remains in `data-processor-gsplats.ts` and is
 * consumed here as a type import.
 *
 * @module data/scene-loader/commit/commit-gsplats-geometry
 */

import * as THREE from 'three';
import { updateInstancedGSplatsMesh } from '../../../rendering/gsplat-geometry';
import { noteGSplatsCommit } from '../../../rendering/depth-sort-coordinator';
import { clampSplatCapacity } from '../../../rendering/splat-texture-layout';
import { syncGSplatMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { isGSplatsUserData } from '../../../types/gsplats';
import { log, Modules } from '../../../utils/log';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import { invalidateRenderObjectFor } from './invalidate-render-object';
import { stampLadderComplete, stampLoadedViewVersion } from './stamp-view-version';
import {
  getCommittedData,
  hasCommittedData,
  setCommittedData,
} from '../../../types/committed-data';
import { getPrefixParent, setPrefixParent } from '../../../types/prefix-lineage';
import type { StagedGSplatsCommit } from '../process/data-processor-gsplats';

const DEFAULT_TRUNCATE = 3.0;

/**
 * Read the `uTruncate` uniform from the mesh material, falling back to
 * the default. Used at commit time for frustum-culling sizing —
 * duplicated from `data-processor-gsplats.ts` (where the worker
 * projection also needs it) so this file stays self-contained.
 */
function readTruncate(mesh: THREE.Mesh): number {
  return (
    (mesh.material as { uniforms?: { uTruncate?: { value: number } } })?.uniforms?.uTruncate
      ?.value ?? DEFAULT_TRUNCATE
  );
}

/**
 * Synchronous GPU commit step: write the staged buffers into the
 * mesh's geometry, either via the GPU buffer pool (if enabled) or via
 * `updateInstancedGSplatsMesh`. Updates `visibleSplatCount` on the
 * mesh's user-data and logs an info line on a zero-splat frame.
 *
 * Must run synchronously inside the atomic commit stage.
 */
export function commitGSplatsGeometry(
  staged: StagedGSplatsCommit,
  rootGroup: THREE.Group | null,
  gpuBufferPool: GPUBufferPool | null,
  session: UpdateSession | undefined,
  loadedViewVersion: number
): void {
  if (!rootGroup) return;

  const mesh = rootGroup.getObjectByName(staged.path) as THREE.Mesh;
  if (!mesh || !isGSplatsUserData(mesh.userData)) return;

  if (staged.noop) {
    // Stamp-only commit: the data reference matches what the GPU already
    // holds (see noop-commit.ts). Refresh the LOD freshness + ladder stamps
    // so the registry keeps treating this node as fresh; touch no geometry.
    stampLoadedViewVersion(mesh.userData, loadedViewVersion);
    stampLadderComplete(mesh.userData);
    return;
  }

  const { processed, cholesky01, cholesky23, cholesky45 } = staged;

  // SEMANTIC clamp at the commit choke point: the GPU writers below clamp
  // the WRITTEN splats to the per-node texture bound (splat-texture-layout),
  // so every count this commit records or hands out — visibleSplatCount,
  // the sort coordinator's `count` — must be the clamped one. Otherwise the
  // SortWorker returns a permutation with slot values ≥ the texture
  // capacity, and those aSortedIndex entries fetch out-of-bounds texels.
  const splatCount = clampSplatCapacity(processed.splatCount);

  // Pre-commit state for the preserve-ordering predicate below. Captured
  // BEFORE the writers run: the pool branch reassigns `mesh.geometry`,
  // and `visibleSplatCount` is overwritten near the end of this function.
  const prevGeometry = mesh.geometry;
  const hadCommittedData = hasCommittedData(mesh);
  const prevCount = mesh.userData.visibleSplatCount;

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      const geometry = gpuBufferPool.acquireGSplatsGeometry(staged.path, splatCount);
      const attributesRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      const truncationRadius = readTruncate(mesh);
      // Keep the previous depth-sort permutation on a same-node same-count
      // in-place recommit (timepoint scrub): a permutation of [0,count) is a
      // strictly-no-worse prior than storage order for the ≥1 frame until the
      // re-sort dispatched by noteGSplatsCommit below lands. Every guard is
      // load-bearing:
      // - !attributesRebuilt / geometry === prevGeometry: pool best-fit reuse
      //   can hand this node a geometry holding ANOTHER node's permutation
      //   over a different prior count — entries could point at texels never
      //   rewritten for this commit.
      // - hadCommittedData: covers this node's own first commit after LOD
      //   demotion (stamp cleared) — the retained geometry's ordering is no
      //   longer vouched for.
      // - prevCount === splatCount: a permutation of [0,prevCount) is not a
      //   permutation of [0,count).
      // No blending-mode gate: under commutative modes / depth-sort-off the
      // ordering is identity anyway (never permuted), so skipping the
      // redundant rewrite is a no-op; under normal mode the sort corrects
      // draw order within ~a frame.
      const preserveOrdering =
        hadCommittedData &&
        !attributesRebuilt &&
        geometry === prevGeometry &&
        prevCount === splatCount;
      // Append fast path (depth-sorting Phase 4 Stage 2): when this commit
      // merely EXTENDS the prefix already on the GPU, write & upload only the
      // new `[prevCount, splatCount)` suffix. Correctness rests on the
      // projected prefix being byte-identical to what the GPU holds, which
      // every conjunct below establishes:
      // - !attributesRebuilt && geometry === prevGeometry: the pool reused
      //   THIS node's buffers in place (a grow/best-fit/fresh acquire sets
      //   attributesRebuilt and may hand back another node's texels). This
      //   also guarantees splatCount ≤ the existing capacity (the pool only
      //   grows via release+reacquire, which rebuilds).
      // - gpuPrefixIntact: a WebGL context restore zeroed the GPU buffers;
      //   the restore hook clears this so the next commit does a full rewrite.
      // - splatCount > prevCount: a genuine append (equal → preserveOrdering
      //   path; shrink/first-commit → full write).
      // - prefix lineage === committedData: the new concat result forward-
      //   chains to the exact object last committed here — proving same
      //   generation (view unchanged: a view change resets the generation, so
      //   the post-reset concat has no parent), a genuine extension, and that
      //   the GPU still holds that parent's projection.
      // - committedTruncate === truncationRadius: `truncate` is a material
      //   uniform outside the loader view state; a change would restyle the
      //   prefix's frustum sizing, so a mismatch forces a full rewrite.
      // Unlike the points/lines gates there is NO optional-field presence
      // conjunct: gsplats' only optional field is `colors`, whose concat
      // white-fills missing parts (never all-or-nothing drops), and whose
      // projection is a pass-through coercion (no interpolation) — a
      // null→colored ladder transition re-fills the prefix with values
      // bit-identical to the colorless white default (fill × (1/fill) is
      // exactly 1.0 in f32; pinned by the coerce-colors "append-gate
      // invariant" test).
      const canAppend =
        hadCommittedData &&
        !attributesRebuilt &&
        geometry === prevGeometry &&
        mesh.userData.gpuPrefixIntact === true &&
        splatCount > (prevCount ?? 0) &&
        getPrefixParent(staged.sourceData) !== undefined &&
        getPrefixParent(staged.sourceData) === getCommittedData(mesh) &&
        mesh.userData.committedTruncate === truncationRadius;
      // Consume-and-clear (see prefix-lineage.ts retention contract): the
      // lineage entry existed solely for the gate check above — clearing it
      // unpins the parent concat's CPU arrays. A retry after a throwing
      // write below reads `undefined` and full-rewrites, the safe direction.
      setPrefixParent(staged.sourceData, null);
      try {
        gpuBufferPool.updateGSplatsGeometry(
          geometry,
          {
            centers3D: processed.centers3D,
            amplitudes: processed.amplitudes,
            cholesky01,
            cholesky23,
            cholesky45,
            colors: processed.colors,
            colorComponents: processed.colorComponents,
          },
          splatCount,
          truncationRadius,
          { preserveOrdering, fromSplat: canAppend ? (prevCount ?? 0) : 0 }
        );
      } finally {
        // Ownership handoff must happen even if the update throws: the
        // acquire may have RELEASED the mesh's current geometry into the
        // free pool (grow path), so bailing out before this assignment
        // would leave the mesh rendering a free-pooled geometry that the
        // evictor can dispose — or another node adopt — mid-render. See
        // commit-points-geometry.ts.
        mesh.geometry = geometry;
        // Rebind uSplatTex (render + pick materials) to the acquired
        // entry's texture — the acquire may have handed the node a
        // different geometry+texture pair. In the finally for the same
        // reason as the handoff: the mesh must never render a geometry
        // whose texture its material doesn't reference.
        syncGSplatMaterialWithGeometry(mesh);
        // Pool rebuilt the geometry's splat storage; evict Three's
        // cached RenderObject so its `vertexBuffers` set is rebuilt
        // against the new buffers next draw.
        if (attributesRebuilt) invalidateRenderObjectFor(mesh);
      }
    } else {
      // Non-pool path: a size change swaps in a fresh geometry+texture
      // pair — evict Three's cached RenderObject exactly like the pool
      // branch above (stale `vertexBuffers` on the WebGPU backend
      // otherwise) and rebind the materials' splat texture.
      //
      // Same preserve-ordering predicate as the pool branch (see the
      // comment there), minus the pool-reuse guards: nothing has swapped
      // `mesh.geometry` yet at this point (a size change swaps it INSIDE
      // updateInstancedGSplatsMesh, whose rebuild branch always writes
      // identity regardless of the flag — fresh geometries are
      // zero-filled), so geometry identity + count are the guards.
      const preserveOrdering =
        hadCommittedData && mesh.geometry === prevGeometry && prevCount === splatCount;
      const rebuilt = updateInstancedGSplatsMesh(
        mesh,
        {
          centers: processed.centers3D,
          cholesky01,
          cholesky23,
          cholesky45,
          amplitudes: processed.amplitudes,
          colors: processed.colors,
          colorComponents: processed.colorComponents,
          splatCount,
        },
        { preserveOrdering }
      );
      syncGSplatMaterialWithGeometry(mesh);
      if (rebuilt) invalidateRenderObjectFor(mesh);
    }

    // Declare the committed color layout to the render material: RGBA colors
    // carry a per-splat opacity alpha, and the volumetric shader branch gates
    // its alpha → optical-depth mapping on this uniform. Plain uniform write
    // (no recompile); the pick material has no such uniform (picking stays
    // brightness-as-depth, deliberately alpha-free in phase 2).
    const renderMat = mesh.material as { updateHasElementAlpha?: (v: boolean) => void };
    renderMat.updateHasElementAlpha?.(processed.colorComponents === 4);

    if (isGSplatsUserData(mesh.userData)) {
      mesh.userData.visibleSplatCount = splatCount;
      // Append-fast-path bookkeeping (depth-sorting Phase 4 Stage 2): record
      // the truncate baked into the GPU texels and mark the GPU prefix intact.
      // A full rewrite re-establishes both, so the next commit may append; a
      // context restore clears gpuPrefixIntact to force a full rewrite.
      mesh.userData.committedTruncate = readTruncate(mesh);
      mesh.userData.gpuPrefixIntact = true;
      // Stamp the view-version this geometry was loaded for so the LOD registry
      // can distinguish "fresh for the current slice" from merely "ready" (a
      // re-slice overwrites the buffers in place above without flipping any
      // readiness flag). Shared with the points/lines commits via the helper.
      stampLoadedViewVersion(mesh.userData, loadedViewVersion);
      // Ladder-completeness stamp for the never-downgrade display gate
      // (see stamp-view-version.ts) — commit-synchronized with the count above.
      stampLadderComplete(mesh.userData);
      // Record the committed data reference — a later update returning the
      // SAME reference (memoized progressive concat) can then take the
      // stamp-only no-op path instead of re-projecting + re-uploading.
      setCommittedData(mesh, staged.sourceData);
    }

    if (splatCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing gsplats for ${staged.path} (no visible splats at current slice)`
      );
    }

    // Depth-sorting Phase 2: every non-noop commit bumps the node's sort
    // generation; order-dependent (`normal`) nodes additionally transfer
    // their projected centers to the SortWorker and get one sort from the
    // current camera pose. Runs LAST: the texture-write/bbox loops above
    // are the final main-thread readers of `centers3D`, and the transfer
    // detaches it (safe — the memoized-concat noop keys on `sourceData`).
    // The CLAMPED count keeps the SortWorker's permutation values inside
    // [0, textureCapacity) — the worker clamps its own count to it.
    noteGSplatsCommit(mesh, processed.centers3D, splatCount);
  } finally {
    bufferSession?.end();
  }
}
