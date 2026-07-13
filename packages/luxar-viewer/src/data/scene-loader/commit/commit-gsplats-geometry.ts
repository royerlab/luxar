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
import type { GSplatsUserData } from '../../../types/gsplats';
import { log, Modules } from '../../../utils/log';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import { invalidateRenderObjectFor } from './invalidate-render-object';
import { stampLadderComplete, stampLoadedViewVersion } from './stamp-view-version';
import type { CommittedDataUserData } from './noop-commit';
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
  if (!mesh || mesh.userData?.nodeType !== 'gsplats') return;

  if (staged.noop) {
    // Stamp-only commit: the data reference matches what the GPU already
    // holds (see noop-commit.ts). Refresh the LOD freshness + ladder stamps
    // so the registry keeps treating this node as fresh; touch no geometry.
    stampLoadedViewVersion(mesh.userData as GSplatsUserData, loadedViewVersion);
    stampLadderComplete(mesh.userData);
    return;
  }

  const { processed, cholesky01, cholesky23, cholesky45 } = staged;

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      const geometry = gpuBufferPool.acquireGSplatsGeometry(staged.path, processed.splatCount);
      const attributesRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      const truncationRadius = readTruncate(mesh);
      gpuBufferPool.updateGSplatsGeometry(
        geometry,
        {
          centers3D: processed.centers3D,
          amplitudes: processed.amplitudes,
          cholesky01,
          cholesky23,
          cholesky45,
          colors: processed.colors,
          splatCount: processed.splatCount,
        },
        processed.splatCount,
        truncationRadius
      );
      mesh.geometry = geometry;
      // Pool rebuilt the geometry's InstancedInterleavedBuffer; evict
      // Three's cached RenderObject so its `vertexBuffers` set is
      // rebuilt against the new buffer next draw.
      if (attributesRebuilt) invalidateRenderObjectFor(mesh);
    } else {
      // Non-pool path: a size change rebinds a fresh
      // InstancedInterleavedBuffer — evict Three's cached RenderObject
      // exactly like the pool branch above (stale `vertexBuffers` on
      // the WebGPU backend otherwise).
      const rebuilt = updateInstancedGSplatsMesh(mesh, {
        centers: processed.centers3D,
        cholesky01,
        cholesky23,
        cholesky45,
        amplitudes: processed.amplitudes,
        colors: processed.colors,
        splatCount: processed.splatCount,
      });
      if (rebuilt) invalidateRenderObjectFor(mesh);
    }

    if (mesh.userData) {
      (mesh.userData as GSplatsUserData).visibleSplatCount = processed.splatCount;
      // Stamp the view-version this geometry was loaded for so the LOD registry
      // can distinguish "fresh for the current slice" from merely "ready" (a
      // re-slice overwrites the buffers in place above without flipping any
      // readiness flag). Shared with the points/lines commits via the helper.
      stampLoadedViewVersion(mesh.userData as GSplatsUserData, loadedViewVersion);
      // Ladder-completeness stamp for the never-downgrade display gate
      // (see stamp-view-version.ts) — commit-synchronized with the count above.
      stampLadderComplete(mesh.userData);
      // Record the committed data reference — a later update returning the
      // SAME reference (memoized progressive concat) can then take the
      // stamp-only no-op path instead of re-projecting + re-uploading.
      (mesh.userData as CommittedDataUserData).committedData = staged.sourceData;
    }

    if (processed.splatCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing gsplats for ${staged.path} (no visible splats at current slice)`
      );
    }
  } finally {
    bufferSession?.end();
  }
}
