/**
 * Lines geometry-commit concern, mirroring `commit-points-geometry.ts`.
 *
 * Keeps the synchronous Lines GPU commit in a focused module. The async
 * projection step stays in `data-processor-lines.ts` because it has no
 * Points counterpart.
 *
 * `StagedLinesCommit` (the staged-data shape carried between the async
 * processing and the synchronous commit) remains in
 * `data-processor-lines.ts` and is consumed here as a type import.
 *
 * @module data/scene-loader/commit/commit-lines-geometry
 */

import * as THREE from 'three';
import { isLinesUserData } from '../../../types/lines';
import { log, Modules } from '../../../utils/log';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import { updateInstancedLinesMesh } from '../../../rendering/line-geometry';
import { noteDepthSortCommit } from '../../../rendering/depth-sort-coordinator';
import { clampLineCapacity } from '../../../rendering/element-texture-layout';
import { syncLineMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { invalidateRenderObjectFor } from './invalidate-render-object';
import { stampLadderComplete, stampLoadedViewVersion } from './stamp-view-version';
import {
  getCommittedData,
  hasCommittedData,
  setCommittedData,
} from '../../../types/committed-data';
import { getPrefixParent, setPrefixParent } from '../../../types/prefix-lineage';
import type { LoadedLinesData } from '../../../types/lines';
import type { StagedLinesCommit } from '../process/data-processor-lines';

// Re-export so callers can import this name from the commit module while
// the implementation lives in the rendering layer (points parity).
export { syncLineMaterialWithGeometry };

/**
 * Synchronous GPU commit step: write the staged buffers into the
 * mesh's geometry, either via the GPU buffer pool (if enabled) or via
 * `updateInstancedLinesMesh`. Updates `visibleSegmentCount` on the
 * mesh's user-data and logs an info line on a zero-segment frame
 * (slice with no visible content).
 *
 * Must run synchronously inside the atomic commit stage — no async
 * operations allowed.
 */
export function commitLinesGeometry(
  staged: StagedLinesCommit,
  rootGroup: THREE.Group | null,
  gpuBufferPool: GPUBufferPool | null,
  session: UpdateSession | undefined,
  loadedViewVersion: number
): void {
  if (!rootGroup) return;

  const mesh = rootGroup.getObjectByName(staged.path) as THREE.Mesh;
  if (!mesh || !isLinesUserData(mesh.userData)) return;

  if (staged.noop) {
    // Stamp-only commit: the data reference matches what the GPU already
    // holds (see noop-commit.ts). Refresh the LOD freshness + ladder stamps
    // so the registry keeps treating this node as fresh; touch no geometry.
    stampLoadedViewVersion(mesh.userData, loadedViewVersion);
    stampLadderComplete(mesh.userData);
    return;
  }

  const { processed } = staged;

  // SEMANTIC clamp at the commit choke point (mirrors
  // commit-points-geometry.ts): the GPU writers below clamp the WRITTEN
  // segments to the per-node texture bound (element-texture-layout), so
  // every count this commit records — visibleSegmentCount, the
  // append-gate comparisons, the sort registration — must be the clamped
  // one. Otherwise a later sorted ordering over the recorded count would
  // carry aSortedIndex slot values ≥ the texture capacity, and those
  // entries would fetch out-of-bounds texels.
  const segmentCount = clampLineCapacity(processed.segmentCount);

  // Pre-commit state for the append predicate below. Captured BEFORE the
  // writers run: the pool branch reassigns `mesh.geometry`, and
  // `visibleSegmentCount` is overwritten in the success-only tail.
  const prevGeometry = mesh.geometry;
  const hadCommittedData = hasCommittedData(mesh);
  const prevCount = mesh.userData.visibleSegmentCount;

  // Lazy projected-midpoints provider for the depth-sort coordinator
  // (invoked only when the node actually registers — order-dependent
  // effective mode, non-empty, still the latest generation — so the
  // common additive path never pays the copy). Segment-midpoint keys are
  // the standard approximation (volumetric spec §Phase 4); artifacts
  // only when long segments interleave. MUST allocate fresh: the
  // coordinator TRANSFERS the returned buffer to the SortWorker, and the
  // processed arrays back the texel source (points parity — never hand
  // the transfer a view into live data).
  const sortCenters3 = (): Float32Array => {
    const starts = processed.startPositions;
    const ends = processed.endPositions;
    const out = new Float32Array(segmentCount * 3);
    for (let i = 0; i < out.length; i++) {
      out[i] = (starts[i] + ends[i]) * 0.5;
    }
    return out;
  };

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      // Acquire geometry from pool (capacity-aware: the fixed 6-texel
      // layout means any pooled lines geometry fits any lines node —
      // the interleaved era's scalar spec-set dimension is gone).
      const geometry = gpuBufferPool.acquireLinesGeometry(staged.path, segmentCount);
      const attributesRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      // Keep the previous depth-sort permutation on a same-node
      // same-count in-place recommit (timepoint scrub): a permutation of
      // [0,count) is a strictly-no-worse prior than storage order for
      // the ≥1 frame until the re-sort dispatched by noteDepthSortCommit
      // below lands. Guards mirror the points/gsplats twins.
      const preserveOrdering =
        hadCommittedData &&
        !attributesRebuilt &&
        geometry === prevGeometry &&
        prevCount === segmentCount;
      // Append fast path (depth-sorting Phase 4 Stage 2): when this commit
      // merely EXTENDS the segment prefix already on the GPU, write & upload
      // only the new `[prevCount, segmentCount)` suffix. Clipping is an
      // order-preserving drop + in-place endpoint clip (never splits or
      // reorders), so under an unchanged view state the longer projection's
      // first `prevCount` surviving segments are byte-identical to the
      // previous commit's whole output. For cap suppression this
      // additionally rests on `concatenateLinesData`
      // (lines-progressive-loader.ts) offset-adjusting each additive-LOD
      // level's segment indices into a DISJOINT vertex-index range
      // (`part.segments[i] + vertexOffset`): suppression joints are keyed
      // on SHARED vertex indices, so an appended level can never register
      // a joint on — and thereby alter — a prefix endpoint's suppression.
      // Conjuncts as in the points/gsplats
      // twins (see commit-gsplats-geometry.ts for the full rationale), plus:
      // - optional-field presence must MATCH the committed parent: a
      //   presence flip (e.g. the new level introduces colors) re-fills the
      //   prefix through the interpolation kernel, which need not be
      //   bit-exact with the constant default the prefix was committed
      //   with. (The fixed texel layout means the pool no longer rebuilds
      //   on a scalar-presence change — these presence conjuncts are now
      //   the SOLE guard for every optional field, scalars included.)
      const committed = getCommittedData(mesh) as LoadedLinesData | undefined;
      const canAppend =
        hadCommittedData &&
        !attributesRebuilt &&
        geometry === prevGeometry &&
        mesh.userData.gpuPrefixIntact === true &&
        segmentCount > (prevCount ?? 0) &&
        committed !== undefined &&
        getPrefixParent(staged.sourceData) !== undefined &&
        getPrefixParent(staged.sourceData) === committed &&
        !!staged.sourceData.colors === !!committed.colors &&
        // Color LAYOUT parity, not just presence: an append writes only
        // the suffix texels, so an RGB↔RGBA flip between levels would
        // strand the prefix's texel5.zw alphas at the other layout's
        // values (mirrors commit-points-geometry's conjunct).
        (staged.sourceData.colorComponents ?? 3) === (committed.colorComponents ?? 3) &&
        !!staged.sourceData.sharpness === !!committed.sharpness &&
        !!staged.sourceData.scalars === !!committed.scalars;
      // Consume-and-clear (see prefix-lineage.ts retention contract): the
      // lineage entry existed solely for the gate check above — clearing it
      // unpins the parent concat's CPU arrays. A retry after a throwing
      // write below reads `undefined` and full-rewrites, the safe direction.
      setPrefixParent(staged.sourceData, null);
      try {
        gpuBufferPool.updateLinesGeometry(geometry, processed, segmentCount, {
          preserveOrdering,
          fromInstance: canAppend ? (prevCount ?? 0) : 0,
        });
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
        // Rebind the geometry-owned line texture on the render + pick
        // materials (a pool acquire may hand back a different
        // geometry+texture pair). Idempotent on the common same-pair
        // commit.
        syncLineMaterialWithGeometry(mesh);
        if (attributesRebuilt) invalidateRenderObjectFor(mesh);
        // Dispose a replaced NON-pool geometry (the creation-time
        // placeholder) — see the commit-points-geometry.ts twin.
        if (prevGeometry !== geometry && !prevGeometry.userData?.luxarPooled) {
          prevGeometry.dispose();
        }
      }
    } else {
      // Non-pool path: a size change rebuilds a fresh exact-size
      // geometry+texture pair — evict Three's cached RenderObject and
      // rebind the texture exactly like the pool branch above.
      const rebuilt = updateInstancedLinesMesh(mesh, processed);
      syncLineMaterialWithGeometry(mesh);
      if (rebuilt) invalidateRenderObjectFor(mesh);
    }

    // SUCCESS-ONLY tail (a throwing write above propagates past this
    // point, mirroring the points/gsplats twins): freshness first —
    mesh.userData.visibleSegmentCount = segmentCount;
    // Append-fast-path bookkeeping: the buffer now holds this commit's
    // data in full (whether written fully or by suffix-extension), so the
    // next commit may append. A context restore clears this flag.
    mesh.userData.gpuPrefixIntact = true;
    // Slice-aware LOD freshness stamp (see commit-gsplats-geometry.ts).
    stampLoadedViewVersion(mesh.userData, loadedViewVersion);
    // Ladder-completeness stamp for the never-downgrade display gate.
    stampLadderComplete(mesh.userData);
    // Record the committed data reference — a later update returning the
    // SAME reference (memoized progressive concat) can then take the
    // stamp-only no-op path instead of re-projecting + re-uploading.
    setCommittedData(mesh, staged.sourceData);

    if (segmentCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing lines for ${staged.path} (no visible segments at current slice)`
      );
    }

    // Depth-sorting (lines integration): every non-noop commit bumps the
    // node's sort generation; order-dependent (effective `normal`) nodes
    // additionally register their segment midpoints with the SortWorker
    // and get one sort from the current camera pose. Runs LAST like the
    // points/gsplats twins, success-only, with the CLAMPED count so
    // permutation values stay inside [0, textureCapacity). The lazy
    // provider defers the O(N) midpoint computation to the sorted path.
    noteDepthSortCommit(mesh, sortCenters3, segmentCount);
  } finally {
    bufferSession?.end();
  }
}
