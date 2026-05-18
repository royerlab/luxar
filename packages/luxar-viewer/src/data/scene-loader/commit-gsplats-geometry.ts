/**
 * GSplats geometry-commit concern, mirroring `commit-points-geometry.ts`
 * and `commit-lines-geometry.ts`.
 *
 * Extracted from `data-processor-gsplats.ts` in step 6 of the
 * god-object refactor — the per-type commit helpers now live in
 * dedicated files. The async projection step keeps its home in
 * `data-processor-gsplats.ts` because it has no Points counterpart.
 *
 * `StagedGSplatsCommit` remains in `data-processor-gsplats.ts` and is
 * consumed here as a type import.
 *
 * @module data/scene-loader/commit-gsplats-geometry
 */

import * as THREE from 'three';
import { updateInstancedGSplatsMesh } from '../../rendering/gsplat-geometry';
import type { GSplatsUserData } from '../../types/gsplats';
import { log, Modules } from '../../utils/log';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { GPUBufferPool } from '../../rendering/gpu-buffer-pool';
import { invalidateRenderObjectFor } from './invalidate-render-object';
import type { StagedGSplatsCommit } from './data-processor-gsplats';

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
  session?: UpdateSession
): void {
  if (!rootGroup) return;

  const mesh = rootGroup.getObjectByName(staged.path) as THREE.Mesh;
  if (!mesh || mesh.userData?.nodeType !== 'gsplats') return;

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
      updateInstancedGSplatsMesh(mesh, {
        centers: processed.centers3D,
        cholesky01,
        cholesky23,
        cholesky45,
        amplitudes: processed.amplitudes,
        colors: processed.colors,
        splatCount: processed.splatCount,
      });
    }

    if (mesh.userData) {
      (mesh.userData as GSplatsUserData).visibleSplatCount = processed.splatCount;
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
