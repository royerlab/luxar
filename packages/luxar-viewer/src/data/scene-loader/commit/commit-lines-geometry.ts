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
import type { CommittedDataUserData } from './noop-commit';
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
      try {
        gpuBufferPool.updateLinesGeometry(geometry, processed, processed.segmentCount);
      } finally {
        // Ownership handoff must happen even if the update throws: the
        // acquire may have RELEASED the mesh's current geometry into the
        // free pool (grow / scalar-spec-mismatch path), so bailing out
        // before this assignment would leave the mesh rendering a
        // free-pooled geometry that the evictor can dispose — or another
        // node adopt — mid-render. See commit-points-geometry.ts.
        mesh.geometry = geometry;
        if (acquireRebuilt) invalidateRenderObjectFor(mesh);
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
      // Slice-aware LOD freshness stamp (see commit-gsplats-geometry.ts).
      stampLoadedViewVersion(mesh.userData, loadedViewVersion);
      // Ladder-completeness stamp for the never-downgrade display gate.
      stampLadderComplete(mesh.userData);
      // Record the committed data reference — a later update returning the
      // SAME reference (memoized progressive concat) can then take the
      // stamp-only no-op path instead of re-projecting + re-uploading.
      (mesh.userData as CommittedDataUserData).committedData = staged.sourceData;
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
