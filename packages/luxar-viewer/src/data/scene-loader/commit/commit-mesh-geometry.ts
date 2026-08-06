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
import { applyMeshSide, applyMeshShading } from '../../../rendering/node-factory/create-mesh-node';
import { normalModeDepthWrite } from '../../../rendering/blending-state';
import { stampLoadedViewVersion } from './stamp-view-version';
import { isMeshUserData, type MeshMetadata } from '../../../types/mesh';
import type { StagedMeshCommit } from '../process/data-processor-mesh';
import type { UpdateSession } from '../../../profiling/update-profiler';

/**
 * Nodes already warned about unsorted translucency, keyed by OBJECT IDENTITY.
 *
 * Module-scoped so the notice is once per node for the lifetime of the tab rather than
 * once per commit — this runs on EVERY slice move, and a per-call warning would flood
 * the console during a scrub.
 *
 * A `WeakSet` of the mesh itself rather than a `Set` of paths, which is what this was
 * and which was wrong in both directions: a dataset switch that reused a path left the
 * NEW node permanently suppressed, while the comment claimed the opposite ("a dataset
 * switch may re-warn"). Identity has neither problem — a replaced node is a different
 * object, and the entry disappears with the old one instead of leaking for the tab's
 * lifetime.
 */
let noticedTranslucency = new WeakSet<THREE.Mesh>();

/** Test seam: forget which nodes have been warned about. */
export function resetTranslucencyNoticesForTesting(): void {
  // A WeakSet cannot be enumerated or cleared, so replace it.
  noticedTranslucency = new WeakSet<THREE.Mesh>();
}

/**
 * Warn once per node when `normal` is combined with translucency (spec §6.3).
 *
 * §6.3 promises this warning and names it as the mitigation for the per-triangle
 * depth-sort exclusion (§9) — mesh sorts nothing, so triangles composite in index
 * order. `opaque`, mesh's default, is unaffected: it depth-tests and depth-writes, so
 * the depth buffer orders it correctly whatever the index order is.
 *
 * **The predicate is two independent clauses, and neither is `opacity < 1`.**
 *
 * The opacity arm reuses {@link normalModeDepthWrite} (the `>= 0.99` threshold)
 * rather than testing `< 1` directly. That threshold is where `normal` mode actually
 * turns `depthWrite` off, i.e. the observable onset of the artifact, and it is the
 * same predicate the material's own blending state keys on — so the warning and the
 * behaviour it warns about cannot drift apart.
 *
 * The per-vertex-RGBA arm is deliberately UNCONDITIONAL in opacity, because at
 * `opacity = 1` the failure is worse rather than absent: `depthWrite` is on while
 * `transparent` is true, so a translucent fragment writes depth and whatever is behind
 * it is depth-REJECTED. That is dropout, not mis-ordering, and the opacity arm cannot
 * see it.
 *
 * Both inputs are read LIVE off the material rather than from the authored attrs: the
 * Layers panel can switch a node into `normal` or drag its opacity long after load,
 * and `userData.blendingMode` is the RESOLVED mode (so an unsupported request that
 * already fell back to `opaque` stays silent). `uniforms.uOpacity.value` is the same
 * live-opacity read `ui/layers/layer-apply.ts` performs.
 */
export function noticeUnsortedTranslucency(object: THREE.Mesh, path: string): void {
  if (noticedTranslucency.has(object)) return;

  const material = object.material as
    (THREE.Material & { uniforms?: { uOpacity?: { value?: number } } }) | undefined;
  if (!material || Array.isArray(material)) return;
  if (material.userData?.blendingMode !== 'normal') return;

  const opacity = material.uniforms?.uOpacity?.value ?? 1.0;
  // Stamped at commit (below). NOT read off the geometry attribute: `mesh-geometry.ts`
  // pads RGB to RGBA for uint8/uint16, so `itemSize === 4` is true for plenty of meshes
  // that carry no authored alpha at all.
  const hasVertexAlpha = object.userData.meshColorComponents === 4;
  if (normalModeDepthWrite(opacity) && !hasVertexAlpha) return;

  noticedTranslucency.add(object);
  const cause = hasVertexAlpha
    ? `per-vertex RGBA alpha${normalModeDepthWrite(opacity) ? '' : ` and opacity ${opacity}`}`
    : `opacity ${opacity}`;
  log.warning(
    Modules.SCENE_LOADER,
    `Mesh ${path} renders translucent (${cause}) under blending_mode='normal', which ` +
      'is drawn WITHOUT per-triangle depth sorting (MESH_NODE_SPEC.md §6.3): triangles ' +
      'composite in index order, so faces may show through each other incorrectly. Use ' +
      "'opaque' (the mesh default, depth-correct at any opacity) unless the see-through " +
      'look is the point.'
  );
}

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

  // Warned here rather than at node creation because the two halves of the condition
  // are only both known here: the resolved mode lives on the material, while per-vertex
  // RGBA is a property of the LOADED arrays (`MeshMetadata` carries `has_colors`, not a
  // channel count) and so does not exist until the first commit.
  // Stamped so the Layers panel can evaluate the same predicate later: a mode or
  // opacity change after load has no access to `LoadedMeshData`.
  object.userData.meshColorComponents = data.colorComponents;
  noticeUnsortedTranslucency(object, staged.path);

  // A first-commit vertex-attribute rebind (position grow / color install) leaves
  // three's cached WebGPU RenderObject pointing at the old vertex buffers; evict it
  // so the next draw rebuilds from the current attributes. WebGPU-gated — a no-op on
  // the classic WebGL backend and in headless contexts. Same contract as the
  // points/lines/gsplats commits. A pure slice move rebinds nothing, so this is skipped.
  if (attributesRebuilt) invalidateRenderObjectFor(object);

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
