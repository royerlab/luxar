/**
 * Points geometry-commit concern extracted from `scene-loader.ts`.
 *
 * Single function (`commitPointsGeometry`) matching the inline
 * `updatePointsGeometry` method on the SceneLoader class. The inline
 * version interleaved three concerns:
 *
 *   1. Find the THREE.Points by name in the root group.
 *   2. Update `userData.visiblePointCount` and log on empty data.
 *   3. Write the new attribute data into GPU buffers — either via the
 *      GPU buffer pool (zero allocations on reuse) or, with the pool
 *      disabled, dispose+recreate via NodeFactory.
 *
 * **Why is there no `data-processor-points.ts`?** Lines and gsplats
 * each have their own `data-processor-{lines,gsplats}.ts` running a
 * `process*Data` step (worker-driven nD → 3D projection + clipping)
 * before commit. Points has no equivalent module because the points
 * facade folds the equivalent projection into `loadPoints()` itself
 * — the data returned by the facade is already 3D-projected and
 * ready for GPU upload, so the orchestrator only needs to commit it.
 * That asymmetry is a real architectural split, not drift: points'
 * projection is single-pass and fits in the loader; lines clipping
 * and gsplats Cholesky-projection are per-frame transforms that
 * the orchestrator needs to schedule on a worker.
 *
 * Filename matches the single export. If a future per-frame points
 * processing step appears (e.g. nD intensity attenuation), this
 * file should grow into `data-processor-points.ts` matching the
 * other two.
 *
 * @module data/scene-loader/commit/commit-points-geometry
 */

import * as THREE from 'three';
import type { LoadedPointsData } from '../../data-loader-types';
import { isPointsUserData } from '../../../types/points';
import { stampLadderComplete, stampLoadedViewVersion } from './stamp-view-version';
import { isAlreadyCommitted } from './noop-commit';
import { setCommittedData } from '../../../types/committed-data';
import { log, Modules } from '../../../utils/log';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { NodeFactory } from '../../../rendering/node-factory';
import { syncPointMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { invalidateRenderObjectFor } from './invalidate-render-object';

// Re-export so callers can import this name from the commit module while
// the implementation lives in the rendering layer.
export { syncPointMaterialWithGeometry };

/**
 * Synchronous GPU commit step for a points node. Same behavior as the
 * inline `updatePointsGeometry`:
 *
 *   - early return if the rootGroup or the named THREE.Points is gone,
 *   - log + zero `visiblePointCount` on an empty-data frame,
 *   - GPU-buffer-pool path (zero alloc on reuse) when the pool is set,
 *   - dispose + recreate via `nodeFactory.createPointsGeometry`
 *     otherwise (the factory owns all dtype/bounds logic).
 */
export function commitPointsGeometry(
  path: string,
  data: LoadedPointsData,
  rootGroup: THREE.Group | null,
  gpuBufferPool: GPUBufferPool | null,
  nodeFactory: NodeFactory,
  session: UpdateSession | undefined,
  loadedViewVersion: number
): void {
  if (!rootGroup) return;

  const points = rootGroup.getObjectByName(path) as THREE.Mesh;
  // Parity with lines/gsplats commit helpers — verify the named
  // object actually IS a Points node (not e.g. a stray Group with the
  // same name). Guards against bugs where a placeholder of the wrong
  // type is attached at this path.
  if (!points || !isPointsUserData(points.userData)) return;

  // No-op fast path: the data reference matches what the GPU already holds
  // (memoized progressive concat — see noop-commit.ts). Points has no
  // separate process step, so unlike lines/gsplats the check lives here in
  // the commit helper, covering the atomic-commit, refinement, retry, and
  // lazy paths uniformly. Refresh only the LOD freshness stamp.
  if (isAlreadyCommitted(points, data)) {
    stampLoadedViewVersion(points.userData, loadedViewVersion);
    stampLadderComplete(points.userData);
    return;
  }

  if (data.pointCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `Clearing points for ${path} (no visible points at current slice)`
    );
  }

  // Type guard already passed in the early-return above.
  points.userData.visiblePointCount = data.pointCount;
  // Slice-aware LOD freshness stamp (see commit-gsplats-geometry.ts).
  stampLoadedViewVersion(points.userData, loadedViewVersion);
  // Ladder-completeness stamp for the never-downgrade display gate.
  stampLadderComplete(points.userData);

  // World-space radius footprint, shared by every commit path so the
  // boundingBox carries the rendered disc extent (the three-geometry
  // invariant — see create-points-node.ts). Uint8 radii normalize to
  // [0, max_radius]; Float32 radii are already world units, so max_radius
  // is the correct world-space max for both. No radii → 0.5 fill default.
  const attrs = points.userData.attrs;
  const maxRadius = (attrs?.max_radius as number | undefined) ?? 1.0;
  const footprintRadius = data.radii ? maxRadius : 0.5;

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      // Acquire geometry from pool (type-aware: matches capacity AND attribute types).
      const geometry = gpuBufferPool.acquirePointsGeometry(path, data, data.pointCount);
      // Pool rebuilt the geometry's InstancedInterleavedBuffer (grow,
      // pool swap, or fresh allocation). The mesh's cached RenderObject
      // in Three's WebGPURenderer still references the old buffer; the
      // helper dispatches a `dispose` event on the material to evict
      // that cache. No-op under WebGL2 / pre-init / no cached entry.
      const attributesRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      try {
        gpuBufferPool.updatePointsGeometry(geometry, data, data.pointCount);

        if (data.metadata.bounds) {
          geometry.boundingBox = data.metadata.bounds.clone();
          if (footprintRadius > 0) geometry.boundingBox.expandByScalar(footprintRadius);
          geometry.boundingSphere = new THREE.Sphere();
          geometry.boundingBox.getBoundingSphere(geometry.boundingSphere);
        }

        // propagate dtype-aware radius scale onto geometry userData and
        // immediately sync render + pick material uniforms. Without this,
        // a placeholder→real-data transition would leave radiusScale=1
        // even though Uint8 normalized radii should map to [0, max_radius].
        if (!geometry.userData) {
          geometry.userData = {};
        }
        geometry.userData.radiusScale = data.radii instanceof Uint8Array ? maxRadius : 1.0;
      } finally {
        // Ownership handoff must happen even if the update throws: the
        // acquire may have RELEASED the mesh's current geometry into the
        // free pool (grow / attribute-spec-mismatch path), so bailing out
        // before this assignment would leave the mesh rendering a
        // free-pooled geometry that the evictor can dispose — or another
        // node adopt — mid-render. On a throw the mesh shows one frame of
        // partially-written data instead; the skipped committedData stamp
        // below guarantees the next update re-uploads in full.
        points.geometry = geometry;
        syncPointMaterialWithGeometry(points);
        if (attributesRebuilt) invalidateRenderObjectFor(points);
      }
      // Record the committed data reference — a later update returning the
      // SAME reference (memoized progressive concat) takes the stamp-only
      // no-op path above instead of re-uploading.
      setCommittedData(points, data);
      return;
    }

    // Pool disabled: dispose and recreate unconditionally. Recreation
    // handles all the dtype logic (Uint8/Uint16 `normalized:true`,
    // Float16 widening, bounds/footprint, radiusScale userData) via
    // NodeFactory — the single owner of the plain
    // InstancedBufferAttribute layout points geometries use outside
    // the pool. (A historical same-count in-place branch assumed the
    // pool's interleaved layout and threw against factory-built
    // geometry; correctness over reuse on this non-default fallback.)
    const oldGeometry = points.geometry;
    if (oldGeometry) {
      oldGeometry.dispose();
    }
    // Pass max_radius so the rebuilt geometry bakes the correct
    // footprint into boundingBox (and the right dtype scale); omitting
    // it would default maxRadius=1.0 and clip large radii.
    points.geometry = nodeFactory.createPointsGeometry(data, maxRadius);
    // dispose+recreate path picks up new dtype-aware scales from
    // the freshly built geometry's userData.
    syncPointMaterialWithGeometry(points);
    // Fresh GPU buffers replaced the geometry: evict Three's cached
    // RenderObject (stale `vertexBuffers` on the WebGPU backend) —
    // same contract as the pool path's attributesRebuilt branch.
    invalidateRenderObjectFor(points);
    // Record the committed data reference (see the pool path above).
    setCommittedData(points, data);
  } finally {
    bufferSession?.end();
  }
}
