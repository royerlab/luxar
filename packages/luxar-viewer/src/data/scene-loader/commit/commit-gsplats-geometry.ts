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
import { noteDepthSortCommit } from '../../../rendering/depth-sort-coordinator';
import { clampSplatCapacity } from '../../../rendering/element-texture-layout';
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
  setElementIdMap,
} from '../../../types/committed-data';
import { getPrefixParent, setPrefixParent } from '../../../types/prefix-lineage';
import type { StagedGSplatsCommit } from '../process/data-processor-gsplats';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../config/constants';

/**
 * Read the `uTruncate` uniform from the mesh material, falling back to
 * the default. Used at commit time for frustum-culling sizing —
 * duplicated from `data-processor-gsplats.ts` (where the worker
 * projection also needs it) so this file stays self-contained.
 */
function readTruncate(mesh: THREE.Mesh): number {
  return (
    (mesh.material as { uniforms?: { uTruncate?: { value: number } } })?.uniforms?.uTruncate
      ?.value ?? GSPLAT_DEFAULT_TRUNCATION_RADIUS
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

  const { processed } = staged;

  // SEMANTIC clamp at the commit choke point: the GPU writers below clamp
  // the WRITTEN splats to the per-node texture bound (element-texture-layout),
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
      // re-sort dispatched by noteDepthSortCommit below lands. Every guard is
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
      // The same-buffer prior splits in two on the count.
      //
      // Equal count: keep the permutation verbatim (`preserveOrdering`).
      //
      // CHANGED count: hand the adapter the previous count so it can REBUILD
      // the permutation over the new population (`repairSortedIndexForCount`)
      // instead of falling back to storage order. This closes exactly the gap
      // the third bullet above names — "a permutation of [0,prevCount) is not a
      // permutation of [0,count)" is true, and rebuilding it is cheaper than
      // giving it up. It is also the branch a timelapse actually takes: an nD
      // re-slice changes the resident count at almost every step, so the
      // equal-count guard alone never fired and every timepoint drew at least
      // one unsorted frame — a flash per timepoint under an order-dependent
      // blending mode (#2290). Measured on the `cloud` demo at a frozen camera
      // pose, as the fraction of sampled element pairs composited in correct
      // back-to-front order: storage order 0.617, repaired 0.858, a real sort
      // 1.000.
      //
      // The other three conjuncts are what make the buffer's contents
      // meaningful at all, and are unchanged.
      const sameBuffers = hadCommittedData && !attributesRebuilt && geometry === prevGeometry;
      const preserveOrdering = sameBuffers && prevCount === splatCount;
      const repairFromCount =
        sameBuffers && prevCount !== undefined && prevCount !== splatCount ? prevCount : undefined;
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
      // conjunct. Colors white-fill missing parts, while the progressive
      // GSplat loader requires every concatenated level to agree on label
      // presence and vocabulary. A valid append therefore preserves both
      // optional channels across the proven prefix lineage.
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
            choleskyFactors: processed.choleskyFactors3D,
            colors: processed.colors,
            colorComponents: processed.colorComponents,
            labelIndices: processed.labelIndices,
            bounds: processed.bounds,
          },
          splatCount,
          truncationRadius,
          { preserveOrdering, repairFromCount, fromInstance: canAppend ? (prevCount ?? 0) : 0 }
        );
      } catch (err) {
        // Defense-in-depth: a throwing write leaves the buffer content
        // unproven — clear the append-safety flag so the next commit
        // full-rewrites regardless of lineage (the lineage consume-and-
        // clear already forces this today; the flag makes the gate robust
        // against future loader/cache changes that re-stamp lineage).
        mesh.userData.gpuPrefixIntact = false;
        throw err;
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
        // Dispose a replaced NON-pool geometry (the creation-time
        // placeholder) — see the commit-points-geometry.ts twin. Its
        // splat texture (minimum-row alloc) would otherwise leak per
        // node per dataset switch.
        if (prevGeometry !== geometry && !prevGeometry.userData?.luxarPooled) {
          prevGeometry.dispose();
        }
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
      const sameMeshBuffers = hadCommittedData && mesh.geometry === prevGeometry;
      const preserveOrdering = sameMeshBuffers && prevCount === splatCount;
      const rebuilt = updateInstancedGSplatsMesh(
        mesh,
        {
          centers: processed.centers3D,
          choleskyFactors: processed.choleskyFactors3D,
          amplitudes: processed.amplitudes,
          colors: processed.colors,
          colorComponents: processed.colorComponents,
          labelIndices: processed.labelIndices,
          splatCount,
          bounds: processed.bounds,
        },
        { preserveOrdering }
      );
      syncGSplatMaterialWithGeometry(mesh);
      if (rebuilt) invalidateRenderObjectFor(mesh);
    }

    // (The committed color layout reaches the render material's
    // uHasElementAlpha gate via stampGSplatPresenceFlags at the texel
    // writers + syncGSplatMaterialWithGeometry above — the same
    // chokepoint decomposition as points/lines. The pick material has no
    // such uniform: picking stays brightness-as-depth, alpha-free.)

    if (isGSplatsUserData(mesh.userData)) {
      mesh.userData.visibleSplatCount = splatCount;
      mesh.userData.requestedElementCount = processed.splatCount;
      mesh.userData.droppedElementCount = processed.splatCount - splatCount;
      // Append-fast-path bookkeeping (depth-sorting Phase 4 Stage 2): record
      // the truncate baked into the GPU texels and mark the GPU prefix intact.
      // A full rewrite re-establishes both, so the next commit may append; a
      // context restore clears gpuPrefixIntact to force a full rewrite.
      mesh.userData.committedTruncate = readTruncate(mesh);
      mesh.userData.gpuPrefixIntact = true;
      mesh.userData.labelIndices = processed.labelIndices;
      mesh.userData.labelVocabulary = processed.labelVocabulary;
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
      // The slot → on-disk map is only known after projection, so it is a
      // MESH-level stamp written here rather than a field on the payload:
      // `staged.sourceData` may be a SliceCache-owned snapshot handed back by
      // reference on a cache hit, whose byte size was measured at store time —
      // mutating it would under-count the cache and break its never-mutated
      // invariant. Written in lockstep with `setCommittedData` above (no early
      // return between them) so the map always describes the buffers now on
      // the GPU, and cleared when this commit has none, so a previous commit's
      // map can never outlive the geometry it described. The stamp-only noop
      // branch at the top touches neither: its geometry is unchanged, so the
      // existing pair still describes exactly what the GPU holds.
      setElementIdMap(mesh, staged.processed.elementIds);
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
    noteDepthSortCommit(mesh, processed.centers3D, splatCount);
  } finally {
    bufferSession?.end();
  }
}
