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
import { stampLoadedViewVersion } from './stamp-view-version';
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
    // holds (see noop-commit.ts). Refresh the LOD freshness stamp so the
    // registry keeps treating this node as fresh; touch no geometry.
    stampLoadedViewVersion(mesh.userData, loadedViewVersion);
    return;
  }

  const { processed } = staged;

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      const geometry = gpuBufferPool.acquireLinesGeometry(staged.path, processed.segmentCount);
      // Capture the acquire's rebuild flag BEFORE updateLinesGeometry,
      // which may itself trigger a spec-set rebuild (lazy scalar
      // promotion) and OR onto the same flag.
      const acquireRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      gpuBufferPool.updateLinesGeometry(geometry, processed, processed.segmentCount);
      const updateRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      mesh.geometry = geometry;
      if (acquireRebuilt || updateRebuilt) invalidateRenderObjectFor(mesh);
    } else {
      updateInstancedLinesMesh(mesh, processed);
    }

    if (isLinesUserData(mesh.userData)) {
      mesh.userData.visibleSegmentCount = processed.segmentCount;
      // Slice-aware LOD freshness stamp (see commit-gsplats-geometry.ts).
      stampLoadedViewVersion(mesh.userData, loadedViewVersion);
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
