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

import { findObjectByName } from '../../../utils/scene-graph-index';
import * as THREE from 'three';
import { updateInstancedGSplatsMesh } from '../../../rendering/gsplat-geometry';
import type { GeometryCommitHost } from './commit-host';
import { clampSplatCapacity } from '../../../rendering/element-texture-layout';
import { syncGSplatMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { isGSplatsUserData } from '../../../types/gsplats';
import { log, LogEmoji, Modules } from '../../../utils/log';
import type { UpdateSession } from '../../../profiling/update-profiler';
import { planInstancedOrdering } from './plan-instanced-ordering';
import { invalidateRenderObjectFor } from './invalidate-render-object';
import { stampLadderComplete, stampLoadedViewVersion } from './stamp-view-version';
import { markFirstCommit } from '../../../profiling/load-timeline';
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
  host: GeometryCommitHost,
  session: UpdateSession | undefined,
  loadedViewVersion: number
): void {
  const { rootGroup, gpuBufferPool, depthSort } = host;
  if (!rootGroup) return;

  const mesh = findObjectByName(rootGroup, staged.path) as THREE.Mesh;
  if (!mesh || !isGSplatsUserData(mesh.userData)) return;

  if (staged.noop) {
    // Stamp-only commit: the data reference matches what the GPU already
    // holds (see noop-commit.ts). Refresh the LOD freshness + ladder stamps
    // so the registry keeps treating this node as fresh; touch no geometry.
    stampLoadedViewVersion(mesh.userData, loadedViewVersion);
    stampLadderComplete(mesh.userData);
    markFirstCommit('gsplats');
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
  // A pool grow releases the old geometry without ending its held draw.
  // Preserve the count that was actually on screen.
  const prevDrawnCount = (prevGeometry as THREE.InstancedBufferGeometry).instanceCount;
  const hadCommittedData = hasCommittedData(mesh);
  const prevCount = mesh.userData.visibleSplatCount;

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      const geometry = gpuBufferPool.acquireGSplatsGeometry(staged.path, splatCount);
      const attributesRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      const truncationRadius = readTruncate(mesh);
      // Ordering + append gate, shared with points/lines
      // (plan-instanced-ordering.ts). The suffix-only upload needs the GPU
      // prefix intact (a WebGL context restore clears the flag) and the same
      // `truncate`: a material uniform outside the loader view state, whose
      // change restyles the prefix's frustum sizing. Unlike the points/lines
      // gates there is NO optional-field parity conjunct: colors white-fill
      // missing parts, and the progressive GSplat loader requires every
      // concatenated level to agree on label presence and vocabulary, so a
      // proven lineage already preserves both optional channels.
      const orderingOptions = planInstancedOrdering({
        geometry,
        prevGeometry,
        prevDrawnCount,
        hadCommittedData,
        prevCount,
        count: splatCount,
        attributesRebuilt,
        prefixParent: getPrefixParent(staged.sourceData),
        committedData: getCommittedData(mesh),
        prefixReusable:
          mesh.userData.gpuPrefixIntact === true &&
          mesh.userData.committedTruncate === truncationRadius,
      });
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
          orderingOptions
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
      // Consume-and-clear on this path too (prefix-lineage.ts retention
      // contract): no append gate reads the lineage here, and leaving it set
      // would keep the parent concat's CPU arrays pinned for the payload's life.
      setPrefixParent(staged.sourceData, null);
      // Non-pool path: a size change swaps in a fresh geometry+texture
      // pair — evict Three's cached RenderObject exactly like the pool
      // branch above (stale `vertexBuffers` on the WebGPU backend
      // otherwise) and rebind the materials' splat texture.
      //
      // Same preserve-ordering predicate as the pool branch (see the
      // comment there), minus the pool-reuse guards: nothing has swapped
      // `mesh.geometry` at this point (a size change swaps it INSIDE
      // updateInstancedGSplatsMesh, whose rebuild branch always writes
      // identity regardless of the flag — fresh geometries are
      // zero-filled), so geometry identity holds by construction and the
      // count is the only guard.
      const preserveOrdering = hadCommittedData && prevCount === splatCount;
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

    // `mesh.userData` is GSplats userData: the guard at the top of this
    // function returned otherwise, and nothing above reassigns it.
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
    markFirstCommit('gsplats');
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

    if (splatCount === 0) {
      log.verbose(
        LogEmoji.INFO,
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
    // SHARED buffers (retained by the post-projection stage cache and handed
    // to every later commit of this slice) must never be detached: hand the
    // coordinator a lazy COPY instead, paid only when the node actually sorts.
    const { centers3D } = processed;
    depthSort?.noteCommit(
      mesh,
      processed.sharedBuffers ? () => centers3D.slice(0, splatCount * 3) : centers3D,
      splatCount
    );
  } finally {
    bufferSession?.end();
  }
}
