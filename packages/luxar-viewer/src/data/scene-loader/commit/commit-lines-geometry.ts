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

  // Pre-commit state for the append predicate below. Captured BEFORE the
  // writers run: the pool branch reassigns `mesh.geometry`, and
  // `visibleSegmentCount` is overwritten near the end of this function.
  const prevGeometry = mesh.geometry;
  const hadCommittedData = hasCommittedData(mesh);
  const prevCount = mesh.userData.visibleSegmentCount;

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      const hasScalars = !!(processed.startScalars && processed.endScalars);
      const geometry = gpuBufferPool.acquireLinesGeometry(
        staged.path,
        processed.segmentCount,
        hasScalars
      );
      // The scalar spec set is decided by the acquire (no lazy in-place
      // rebuild remains in updateLinesGeometry), so the acquire flag is
      // the complete rebuild signal.
      const acquireRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      // Append fast path (depth-sorting Phase 4 Stage 2): when this commit
      // merely EXTENDS the segment prefix already on the GPU, write & upload
      // only the new `[prevCount, segmentCount)` suffix. Clipping is an
      // order-preserving drop + in-place endpoint clip (never splits or
      // reorders), so under an unchanged view state the longer projection's
      // first `prevCount` surviving segments are byte-identical to the
      // previous commit's whole output. Conjuncts as in the points/gsplats
      // twins (see commit-gsplats-geometry.ts for the full rationale), plus:
      // - optional-field presence must MATCH the committed parent: a
      //   presence flip (e.g. the new level introduces colors) re-fills the
      //   prefix through the interpolation kernel, which need not be
      //   bit-exact with the constant default the prefix was committed with.
      //   (A scalars flip already forces acquireRebuilt via the spec set;
      //   colors/sharpness are invisible to it, hence the explicit check.)
      const committed = getCommittedData(mesh) as LoadedLinesData | undefined;
      const canAppend =
        hadCommittedData &&
        !acquireRebuilt &&
        geometry === prevGeometry &&
        mesh.userData.gpuPrefixIntact === true &&
        processed.segmentCount > (prevCount ?? 0) &&
        committed !== undefined &&
        getPrefixParent(staged.sourceData) !== undefined &&
        getPrefixParent(staged.sourceData) === committed &&
        !!staged.sourceData.colors === !!committed.colors &&
        !!staged.sourceData.sharpness === !!committed.sharpness &&
        !!staged.sourceData.scalars === !!committed.scalars;
      // Consume-and-clear (see prefix-lineage.ts retention contract): the
      // lineage entry existed solely for the gate check above — clearing it
      // unpins the parent concat's CPU arrays. A retry after a throwing
      // write below reads `undefined` and full-rewrites, the safe direction.
      setPrefixParent(staged.sourceData, null);
      try {
        gpuBufferPool.updateLinesGeometry(geometry, processed, processed.segmentCount, {
          fromInstance: canAppend ? (prevCount ?? 0) : 0,
        });
      } finally {
        // Ownership handoff must happen even if the update throws: the
        // acquire may have RELEASED the mesh's current geometry into the
        // free pool (grow / scalar-spec-mismatch path), so bailing out
        // before this assignment would leave the mesh rendering a
        // free-pooled geometry that the evictor can dispose — or another
        // node adopt — mid-render. See commit-points-geometry.ts.
        mesh.geometry = geometry;
        if (acquireRebuilt) invalidateRenderObjectFor(mesh);
        // Dispose a replaced NON-pool geometry (the creation-time
        // placeholder) — see the commit-points-geometry.ts twin.
        if (prevGeometry !== geometry && !prevGeometry.userData?.luxarPooled) {
          prevGeometry.dispose();
        }
      }
    } else {
      // Non-pool path: a size/spec-set change rebinds a fresh
      // InstancedInterleavedBuffer — evict Three's cached RenderObject
      // exactly like the pool branch above (stale `vertexBuffers` on
      // the WebGPU backend otherwise).
      const rebuilt = updateInstancedLinesMesh(mesh, processed);
      if (rebuilt) invalidateRenderObjectFor(mesh);
    }

    if (isLinesUserData(mesh.userData)) {
      mesh.userData.visibleSegmentCount = processed.segmentCount;
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
    }

    if (processed.segmentCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing lines for ${staged.path} (no visible segments at current slice)`
      );
    }
  } finally {
    bufferSession?.end();
  }
}
