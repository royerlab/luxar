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
 * - **No capacity clamp.** Mesh's element ordinal is `gl_VertexID`, not an
 *   element-texture texel, so it is bounded by `MAX_MESH_VERTICES` at the loader's
 *   Stage-1 preflight instead of by texture dimensions here.
 *
 * Depth-sort registration IS shared, though both halves of the payload differ from
 * the instanced types': a triangle's "center" is its vertex centroid, and the
 * resolved ordering permutes `geometry.index` rather than an `aSortedIndex`
 * indirection (`rendering/depth-sort-coordinator/triangle-ordering.ts`).
 *
 * What remains is: find the placeholder by name, update its geometry in place,
 * apply the epoch's material `side`, and register the epoch with the depth-sort
 * coordinator.
 *
 * @module data/scene-loader/commit/commit-mesh-geometry
 */

import type * as THREE from 'three';
import { log, Modules } from '../../../utils/log';
import { updateMeshGeometry } from '../../../rendering/mesh-geometry';
import { invalidateRenderObjectFor } from './invalidate-render-object';
import { applyMeshSide, applyMeshShading } from '../../../rendering/node-factory/create-mesh-node';
import { noteDepthSortCommit } from '../../../rendering/depth-sort-coordinator';
import { computeFaceCentroids } from '../../../rendering/depth-sort-coordinator/triangle-ordering';
import { stampLoadedViewVersion } from './stamp-view-version';
import { setCommittedData } from '../../../types/committed-data';
import { isMeshUserData, type MeshMetadata } from '../../../types/mesh';
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
    // Explicit rather than inferred from array identity: the position buffer is
    // reused across epochs, so identity no longer signals a displayDims change.
    positionChanged: projected.positionChanged,
    // The displayDims key these positions were extracted for. The geometry re-uploads
    // when this differs from the key it last uploaded, which repairs a superseded
    // commit whose projection advanced the loader's key but never uploaded (#1245).
    positionKey: projected.positionKey,
    indices: projected.indices,
    // Projected AABB over the INDEXED vertices — `computeMeshBounds` sets the
    // geometry's box and sphere from it, the same way `computeLineBounds` consumes the
    // lines projection's precomputed bounds (#1252).
    bounds: projected.bounds,
    colors: data.colors,
    colorComponents: data.colorComponents,
    // Uploaded once, like `color`: normals are authored in their own frame and are
    // never re-projected (§3.4 — a frame mismatch drops to the derivative normal
    // instead of re-deriving them), and scalars are view-independent by nature.
    normals: data.normals,
    scalars: data.scalars,
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

  // The epoch's shading variant, for the same reason and from the same event: stored
  // normals are only meaningful when `normal_dims` equals the displayed axes, so a
  // `displayDims` change can flip a smooth-shaded node onto the derivative fallback
  // and back (§3.4 / §6.2). Both are guarded on change, so a slice move costs
  // nothing here.
  applyMeshShading(object, object.userData.attrs as MeshMetadata, projected.storedNormalsUsable);

  // A first-commit vertex-attribute rebind (position grow / color install) leaves
  // three's cached WebGPU RenderObject pointing at the old vertex buffers; evict it
  // so the next draw rebuilds from the current attributes. WebGPU-gated — a no-op on
  // the classic WebGL backend and in headless contexts. Same contract as the
  // points/lines/gsplats commits. A pure slice move rebinds nothing, so this is skipped.
  if (attributesRebuilt) invalidateRenderObjectFor(object);

  object.userData.visibleTriangleCount = projected.visibleFaceCount;
  object.userData.visibleVertexCount = projected.visibleVertexCount;

  // The GPU now holds this data. Mesh has no memoized-concat noop path (the whole
  // node is resident, so there is no LOD concatenation to memoize and
  // `processMeshData` re-projects every update), so the stamp is not used as an
  // identity key here — it is used in the OTHER direction the contract describes:
  // its presence is what tells the depth-sort coordinator that this mesh's index
  // buffer still holds the commit whose ordering is being resolved, and its
  // absence is the demotion signal. Without it every resolved ordering would be
  // dropped as stale (see `types/committed-data.ts`).
  setCommittedData(object, data);

  // Register the epoch with the depth-sort coordinator. Unconditional — the
  // coordinator judges order-dependence off the LIVE material mode and releases
  // the node when it is commutative, which is also what keeps the generation
  // counter advancing so an in-flight sort from a superseded commit is dropped.
  //
  // The centroids are a THUNK for the same reason the points commit uses one: the
  // O(F) pass is paid only if the node actually registers (order-dependent, non
  // empty, still the latest generation). It must allocate fresh — the buffer is
  // transferred to the worker and detached.
  //
  // `projected.indices` is handed over as the permutation SOURCE. It is already a
  // fresh per-epoch copy (`projectMeshTo3D` copies out of its reused scratch), so
  // this retains a reference rather than paying for one, and the coordinator drops
  // it the moment the node stops sorting.
  noteDepthSortCommit(
    object,
    () => computeFaceCentroids(projected.position, projected.indices, projected.visibleFaceCount),
    projected.visibleFaceCount,
    projected.indices
  );

  stampLoadedViewVersion(object.userData, loadedViewVersion ?? currentVersion);

  if (projected.visibleFaceCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `No visible triangles for ${staged.path} at this slice — the surface's ` +
        'vertices all fall outside the nD slab'
    );
  }
}
