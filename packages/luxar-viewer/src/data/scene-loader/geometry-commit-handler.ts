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
 * Lines and gsplats commit helpers live alongside their respective
 * data-processor modules. They could move here for symmetry, but
 * keeping them next to the async process step keeps each pipeline's
 * stages easy to follow.
 *
 * @module data/scene-loader/geometry-commit-handler
 */

import * as THREE from 'three';
import type { LoadedPointsData } from '../data-loader-types';
import { isPointsUserData } from '../../types/points';
import { log, Modules } from '../../utils/log';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { GPUBufferPool } from '../../rendering/gpu-buffer-pool';
import type { NodeFactory } from '../node-factory';

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
  session?: UpdateSession
): void {
  if (!rootGroup) return;

  const points = rootGroup.getObjectByName(path) as THREE.Points;
  if (!points) return;

  if (data.pointCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `Clearing points for ${path} (no visible points at current slice)`
    );
  }

  if (isPointsUserData(points.userData)) {
    points.userData.visiblePointCount = data.pointCount;
  }

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      // Acquire geometry from pool (type-aware: matches capacity AND attribute types).
      const geometry = gpuBufferPool.acquirePointsGeometry(path, data, data.pointCount);
      gpuBufferPool.updatePointsGeometry(geometry, data, data.pointCount);

      if (data.metadata.bounds) {
        geometry.boundingBox = data.metadata.bounds.clone();
      }
      points.geometry = geometry;
      return;
    }

    // Pool disabled: try in-place reuse if the count matches; otherwise
    // dispose and recreate. Recreation handles all the dtype logic via
    // NodeFactory.
    const oldGeometry = points.geometry;
    const oldPositionAttr = oldGeometry?.getAttribute(
      'position'
    ) as THREE.BufferAttribute | null;
    const oldCount = oldPositionAttr ? oldPositionAttr.count : 0;

    if (oldCount === data.pointCount && data.pointCount > 0) {
      (oldPositionAttr!.array as Float32Array).set(data.positions as Float32Array);
      oldPositionAttr!.needsUpdate = true;

      const colorAttr = oldGeometry.getAttribute('color') as THREE.BufferAttribute;
      if (colorAttr && data.colors) {
        (colorAttr.array as ArrayLike<number> & { set: (a: ArrayLike<number>) => void }).set(
          data.colors
        );
        colorAttr.needsUpdate = true;
      }

      const radiiAttr = oldGeometry.getAttribute('radius') as THREE.BufferAttribute;
      if (radiiAttr && data.radii) {
        (radiiAttr.array as ArrayLike<number> & { set: (a: ArrayLike<number>) => void }).set(
          data.radii
        );
        radiiAttr.needsUpdate = true;
      }

      const sharpAttr = oldGeometry.getAttribute('sharpness') as THREE.BufferAttribute;
      if (sharpAttr && data.sharpness) {
        (sharpAttr.array as ArrayLike<number> & { set: (a: ArrayLike<number>) => void }).set(
          data.sharpness
        );
        sharpAttr.needsUpdate = true;
      }

      oldGeometry.computeBoundingBox();
      oldGeometry.computeBoundingSphere();
    } else {
      if (oldGeometry) {
        oldGeometry.dispose();
      }
      points.geometry = nodeFactory.createPointsGeometry(data);
    }
  } finally {
    bufferSession?.end();
  }
}
