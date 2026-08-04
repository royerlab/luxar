/**
 * Mesh geometry-commit concern — the Mesh sibling of
 * `commit-points-geometry.ts` / `commit-lines-geometry.ts` /
 * `commit-gsplats-geometry.ts`.
 *
 * Much shorter than the other three, and the reasons are structural rather than
 * "less finished":
 *
 * - **No GPU buffer pool.** The pool exists to recycle the instanced-quad
 *   attribute buffers whose capacity churns as a slice query returns different
 *   element counts. A mesh's vertex buffers are uploaded once per `displayDims`
 *   epoch and never resized, so there is nothing to recycle
 *   (`gpu-buffer-pool/pool-stats.ts` must keep listing exactly the three instanced
 *   types).
 * - **No depth-sort registration.** Per-triangle depth sorting is explicitly out
 *   of scope (`docs/specs/MESH_NODE_SPEC.md` §9); an opaque surface gets correct
 *   occlusion from the depth buffer, which is what the other three cannot do.
 * - **No capacity clamp.** Mesh's element ordinal is `gl_VertexID`, not an
 *   element-texture texel, so it is bounded by `MAX_MESH_VERTICES` at the loader's
 *   Stage-1 preflight instead of by texture dimensions here.
 *
 * What remains is: find the placeholder by name, update its geometry in place, and
 * apply the epoch's material `side`.
 *
 * @module data/scene-loader/commit/commit-mesh-geometry
 */

import type * as THREE from 'three';
import { log, Modules } from '../../../utils/log';
import { updateMeshGeometry } from '../../../rendering/mesh-geometry';
import { applyMeshSide } from '../../../rendering/node-factory/create-mesh-node';
import { stampLoadedViewVersion } from './stamp-view-version';
import { isMeshUserData } from '../../../types/mesh';
import type { StagedMeshCommit } from '../process/data-processor-mesh';
import type { UpdateSession } from '../../../profiling/update-profiler';

/** Host references the commit needs. */
export interface MeshCommitCtx {
  rootGroup: THREE.Group | null;
  currentVersion: number;
}

/**
 * Commit a staged mesh projection into the scene.
 *
 * Resolved by `rootGroup.getObjectByName(path)` rather than through the loader
 * registry, matching the sibling commits: the placeholder was attached at node-load
 * time, so the scene graph is the single source of truth for "where does this
 * node's geometry live", and a registry lookup could disagree with it after a
 * dataset switch.
 */
export function commitMeshGeometry(
  ctx: MeshCommitCtx,
  staged: StagedMeshCommit,
  _session?: UpdateSession,
  loadedViewVersion?: number
): void {
  const { rootGroup, currentVersion } = ctx;
  if (!rootGroup) return;

  const found = rootGroup.getObjectByName(staged.path);
  // Guarded rather than cast: `getObjectByName` searches by name across the whole
  // subtree, so a path collision or a dataset switch mid-commit can hand back an
  // object of another type. Writing mesh geometry into a points node would corrupt
  // it silently, whereas this reports and declines.
  if (!found || !isMeshUserData(found.userData)) {
    log.warning(
      Modules.SCENE_LOADER,
      `commitMeshGeometry: no mesh node named ${staged.path} in the scene graph`
    );
    return;
  }
  const object = found as THREE.Mesh;

  const { data, projected } = staged;
  updateMeshGeometry(object.geometry, {
    position: projected.position,
    indices: projected.indices,
    colors: data.colors,
    colorComponents: data.colorComponents,
    vertexCount: data.vertexCount,
  });

  // The epoch's side, which is NOT simply the node's `double_sided`: an odd-parity
  // reflection keeps single-sided (the index post-pass restored winding), while an
  // undecidable frame forces double-sided regardless of what was authored.
  applyMeshSide(object, projected.side);

  object.userData.visibleTriangleCount = projected.visibleFaceCount;
  object.userData.visibleVertexCount = projected.visibleVertexCount;

  stampLoadedViewVersion(object.userData, loadedViewVersion ?? currentVersion);

  if (projected.visibleFaceCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `No visible triangles for ${staged.path} at this slice — the surface's ` +
        'vertices all fall outside the nD slab'
    );
  }
}
