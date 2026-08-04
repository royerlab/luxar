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
import { invalidateRenderObjectFor } from './invalidate-render-object';
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
  const attributesRebuilt = updateMeshGeometry(object.geometry, {
    position: projected.position,
    indices: projected.indices,
    colors: data.colors,
    colorComponents: data.colorComponents,
    vertexCount: data.vertexCount,
    // The node's TOTAL faces, which sizes the index buffer's capacity — not the
    // visible count, which changes every slice move and would reallocate (and leak)
    // the index buffer each time.
    faceCount: data.faceCount,
  });

  // The epoch's side, which is NOT simply the node's `double_sided`: an odd-parity
  // reflection keeps single-sided (the index post-pass restored winding), while an
  // undecidable frame forces double-sided regardless of what was authored.
  applyMeshSide(object, projected.side);

  // A first-commit vertex-attribute rebind (position grow / color install) leaves
  // three's cached WebGPU RenderObject pointing at the old vertex buffers; evict it
  // so the next draw rebuilds from the current attributes. WebGPU-gated — a no-op on
  // the classic WebGL backend and in headless contexts. Same contract as the
  // points/lines/gsplats commits. A pure slice move rebinds nothing, so this is skipped.
  if (attributesRebuilt) invalidateRenderObjectFor(object);

  object.userData.visibleTriangleCount = projected.visibleFaceCount;
  object.userData.visibleVertexCount = projected.visibleVertexCount;
  // Bounds over the vertices the index actually references, for camera framing. The
  // geometry's own box spans the whole position buffer (correct to keep — conservative
  // bounds are safe for frustum culling), which would frame a moving 4D surface's
  // entire trajectory instead of the drawn slice (#1252).
  object.userData.visibleBounds = projected.visibleBounds;

  stampLoadedViewVersion(object.userData, loadedViewVersion ?? currentVersion);

  if (projected.visibleFaceCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `No visible triangles for ${staged.path} at this slice — the surface's ` +
        'vertices all fall outside the nD slab'
    );
  }
}
