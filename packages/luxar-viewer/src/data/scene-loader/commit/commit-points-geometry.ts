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
 *      GPU buffer pool (zero allocations on reuse) or the in-place
 *      same-size path / dispose+create different-size path.
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
import {
  widenToFloat32,
  writeInterleavedAttribute,
} from '../../../rendering/interleaved-attributes';
import { isPointsUserData } from '../../../types/points';
import { stampLoadedViewVersion } from './stamp-view-version';
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
 *   - same-size in-place attribute update when count matches,
 *   - dispose + recreate via `nodeFactory.createPointsGeometry` when
 *     the count differs.
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

      points.geometry = geometry;
      syncPointMaterialWithGeometry(points);
      if (attributesRebuilt) invalidateRenderObjectFor(points);
      return;
    }

    // Pool disabled: try in-place reuse if the count matches; otherwise
    // dispose and recreate. Recreation handles all the dtype logic via
    // NodeFactory. Point attributes use a* names and
    // InstancedBufferAttribute storage.
    const oldGeometry = points.geometry;
    // Post-interleaving, the per-instance attributes on a points
    // geometry are `InterleavedBufferAttribute` views sharing one
    // `InstancedInterleavedBuffer`. `getAttribute(...).count` returns
    // the per-instance count from the underlying buffer's
    // `stride * arrayLength`, which is what we want either way.
    const oldCenterAttr = oldGeometry?.getAttribute(
      'aCenter'
    ) as THREE.InterleavedBufferAttribute | null;
    const oldCount = oldCenterAttr ? oldCenterAttr.count : 0;

    if (oldCount === data.pointCount && data.pointCount > 0) {
      // Recover the shared interleaved buffer from any view; every
      // per-instance attribute on a pooled points geometry points at
      // the same buffer.
      const buffer = oldCenterAttr!.data as THREE.InstancedInterleavedBuffer;
      const positionsF32 =
        data.positions instanceof Float32Array
          ? data.positions
          : widenToFloat32(data.positions as ArrayLike<number>);
      writeInterleavedAttribute(buffer, oldCenterAttr!.offset, 3, positionsF32, data.pointCount);

      const colorAttr = oldGeometry.getAttribute('aColor') as
        | THREE.InterleavedBufferAttribute
        | undefined;
      if (colorAttr && data.colors) {
        const widened =
          data.colors instanceof Float32Array
            ? data.colors
            : widenToFloat32(data.colors as ArrayLike<number>, 255);
        writeInterleavedAttribute(buffer, colorAttr.offset, 3, widened, data.pointCount);
      }

      const radiiAttr = oldGeometry.getAttribute('aRadius') as
        | THREE.InterleavedBufferAttribute
        | undefined;
      if (radiiAttr && data.radii) {
        const widened =
          data.radii instanceof Float32Array
            ? data.radii
            : widenToFloat32(data.radii as ArrayLike<number>, 255);
        writeInterleavedAttribute(buffer, radiiAttr.offset, 1, widened, data.pointCount);
      }

      const sharpAttr = oldGeometry.getAttribute('aSharpness') as
        | THREE.InterleavedBufferAttribute
        | undefined;
      if (sharpAttr && data.sharpness) {
        const widened =
          data.sharpness instanceof Float32Array
            ? data.sharpness
            : widenToFloat32(data.sharpness as ArrayLike<number>, 255);
        writeInterleavedAttribute(buffer, sharpAttr.offset, 1, widened, data.pointCount);
      }

      // The 'position' attribute holds the unit quad template, not the
      // per-point world positions — so THREE's computeBoundingBox()/
      // Sphere() would compute the quad's [-1,1]² bounds, not the
      // actual scene extent. Source the bounds from the loader metadata
      // instead (same pattern as the pool-enabled path above).
      if (data.metadata.bounds) {
        oldGeometry.boundingBox = data.metadata.bounds.clone();
        if (footprintRadius > 0) oldGeometry.boundingBox.expandByScalar(footprintRadius);
        oldGeometry.boundingSphere = new THREE.Sphere();
        oldGeometry.boundingBox.getBoundingSphere(oldGeometry.boundingSphere);
      }
      // Same-size in-place update: no setAttribute calls happened, so
      // Three's _maxInstanceCount cache does not need invalidation.
      const instanced = oldGeometry as THREE.InstancedBufferGeometry;
      instanced.instanceCount = data.pointCount;
      instanced.setDrawRange(0, 6);

      // in-place reuse — re-sync material scales in case dtype-
      // aware geometry userData changed since the last commit.
      syncPointMaterialWithGeometry(points);
    } else {
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
    }
  } finally {
    bufferSession?.end();
  }
}
