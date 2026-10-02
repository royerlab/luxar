/**
 * The host references every geometry commit needs, bundled so the commit
 * helpers take one argument for them instead of a growing positional list.
 *
 * @module data/scene-loader/commit/commit-host
 */

import type * as THREE from 'three';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { DepthSortCoordinator } from '../../../rendering/depth-sort-coordinator';

/** What the SceneLoader hands each `commit*Geometry` helper. */
export interface GeometryCommitHost {
  /** The loader's scene root; the committed node is looked up by name under it. */
  rootGroup: THREE.Group | null;
  /** The pooled-geometry path, when the loader runs one. */
  gpuBufferPool: GPUBufferPool | null;
  /**
   * The host's depth-sort coordinator, which every non-noop commit reports to
   * (`noteCommit`). `null` leaves the node untracked: no back-to-front
   * ordering and no cross-node renderOrder.
   */
  depthSort: DepthSortCoordinator | null;
}
