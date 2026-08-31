/**
 * Depth-shard meshes — splitting one order-dependent node's draw into `S`
 * contiguous depth ranges so shards of DIFFERENT nodes can interleave.
 *
 * Companion module of `depth-sort-coordinator.ts` (same facade-file +
 * same-name-directory pattern as `render-order.ts` / `triangle-ordering.ts`).
 * Full design: `docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md`.
 *
 * ## Why this exists
 *
 * A draw call is atomic in draw order, and three's only inter-object lever is
 * one `renderOrder` integer per `Object3D`. So two overlapping order-dependent
 * nodes cannot be composited correctly: the whole of one draws before the whole
 * of the other. A node's committed permutation is already back-to-front, though,
 * so any contiguous range of it is a depth interval — draw each range as its own
 * call and the ranges can be merged globally by depth.
 *
 * ## The shape, and why it is this shape
 *
 * - **The parent mesh IS shard 0.** It keeps its geometry, material, element
 *   texture and every `userData` stamp; only its `instanceCount` narrows to the
 *   shard size. A zero-instance parent was tried and is wrong:
 *   `scene/lod-fade.ts` is literally `if (obj.material) visit(root); else
 *   root.traverse(visit)` — "a mesh that has a material IS the whole node" — so
 *   cross-fade would skip every shard, and camera framing, blend warmup and
 *   scene stats all key off a non-zero `instanceCount`.
 * - **Shards 1..S-1 are CHILDREN**, not siblings, so visibility, world matrix
 *   and `traverse`-based enumeration inherit for free.
 * - **Shards share the parent's material OBJECT**, never a clone. That is what
 *   makes `lod-fade` and the LayersPanel correct, since both mutate the one
 *   shared material. `syncShardMaterials` re-asserts it because
 *   `ui/layers/layer-apply.ts` clone-on-first-use REPLACES `parent.material`.
 * - **Shards carry no `nodeType` / `visible*Count` stamps**, so every stats,
 *   monitor, warmup and picking traversal skips them by construction rather than
 *   double-counting. {@link effectiveInstanceCount} exists for the few readers
 *   that legitimately want the node's whole element count.
 * - **Shard attributes are SUBARRAY VIEWS of the parent's ordering arrays**, so
 *   there is one contiguous CPU array per slot and every existing sorted-index
 *   writer keeps writing it unchanged. Measured on all three renderer arms; the
 *   zero-copy `InterleavedBufferAttribute`-at-an-offset alternative fails WebGPU
 *   pipeline validation (spec §7.1), so do not "optimise" into it.
 *
 * ## Ownership
 *
 * Shards belong to the COORDINATOR, not to the scene graph. They have no scene
 * path, no `attrs`, no loader, no commit, no pool entry and no pick
 * registration — which is exactly what leaves every path that resolves a node by
 * `getObjectByName(path)` untouched. They are created and destroyed here, driven
 * from the commit and release hooks, and `releaseDepthShards` is the single
 * teardown point. This is deliberately NOT the partition-part precedent: parts
 * are real scene-graph nodes, and a shard can never be one.
 */

import * as THREE from 'three';
import { registerSortedIndexMirrors } from '../element-storage';
import { log, Modules } from '../../utils/log';

/** Marker stamped on a shard mesh's `userData` and on its geometry. */
export const DEPTH_SHARD_FLAG = 'depthShardOf';

interface ShardRecord {
  /** Shards 1..S-1, in permutation order. Shard 0 is the parent itself. */
  children: THREE.Mesh[];
  /** Total shards INCLUDING the parent, so `children.length + 1`. */
  shardCount: number;
  /** Elements per shard: `ceil(fullCount / shardCount)`. */
  shardSize: number;
  /** The node's whole element count, i.e. the parent's pre-shard `instanceCount`. */
  fullCount: number;
  /** The geometry the shards were built against (a pool swap invalidates them). */
  geometry: THREE.InstancedBufferGeometry;
}

/**
 * Per-node shard state. A WeakMap so a disposed mesh cannot be pinned alive by
 * this module — the release path is explicit, but a dataset switch that drops a
 * subtree without calling it must not leak.
 */
const shardRecords = new WeakMap<THREE.Mesh, ShardRecord>();

/** True when `object` is a shard mesh rather than a node's own render mesh. */
export function isDepthShard(object: THREE.Object3D): boolean {
  return (object.userData as { [DEPTH_SHARD_FLAG]?: unknown })[DEPTH_SHARD_FLAG] !== undefined;
}

/** True when `geometry` shares its attributes with a parent node's geometry. */
export function isDepthShardGeometry(geometry: THREE.BufferGeometry): boolean {
  return (geometry.userData as { depthShard?: boolean }).depthShard === true;
}

/**
 * The node's WHOLE element count, for readers that want "how many elements does
 * this node have" rather than "how many does this mesh draw".
 *
 * Sharding narrows the parent's `instanceCount` to one shard, so a reader that
 * sums `instanceCount` over meshes carrying a `nodeType` would under-report by
 * a factor of `S`. Returns the live `instanceCount` when the node is unsharded,
 * which keeps every existing call site's semantics identical.
 */
export function effectiveInstanceCount(mesh: THREE.Mesh): number {
  const record = shardRecords.get(mesh);
  if (record) return record.fullCount;
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry | undefined;
  return geometry?.isInstancedBufferGeometry ? geometry.instanceCount : 0;
}

/** Shards 1..S-1 of `mesh` (empty when unsharded). Shard 0 is `mesh` itself. */
export function depthShardChildren(mesh: THREE.Mesh): readonly THREE.Mesh[] {
  return shardRecords.get(mesh)?.children ?? [];
}

/** Number of shards including the parent; 1 when unsharded. */
export function depthShardCount(mesh: THREE.Mesh): number {
  return shardRecords.get(mesh)?.shardCount ?? 1;
}

/**
 * Build the shard geometry for range `[start, start + count)` of the parent's
 * ordering.
 *
 * Shares the base quad and the index buffer with the parent (same objects, so
 * one GPU buffer each), and takes SUBARRAY views of both sorted-index arrays so
 * no ordering bytes are copied. The views alias the parent's memory, so every
 * existing writer that fills the parent's array also fills these — they only
 * need their own update range (see `element-storage.ts`), because each view owns
 * a GPU buffer. `userData` is deliberately NOT shared; see below.
 */
function buildShardGeometry(
  parent: THREE.InstancedBufferGeometry,
  start: number,
  count: number
): THREE.InstancedBufferGeometry | null {
  const geometry = new THREE.InstancedBufferGeometry();
  for (const name in parent.attributes) {
    if (name === 'aSortedIndex' || name === 'aSortedIndexB') continue;
    geometry.setAttribute(name, parent.attributes[name]);
  }
  if (parent.index) geometry.setIndex(parent.index);

  for (const name of ['aSortedIndex', 'aSortedIndexB'] as const) {
    const source = parent.getAttribute(name) as THREE.InstancedBufferAttribute | undefined;
    if (!source) {
      if (name === 'aSortedIndex') return null; // no ordering to shard
      continue;
    }
    const array = source.array as Uint32Array;
    if (start + count > array.length) return null;
    const view = new THREE.InstancedBufferAttribute(array.subarray(start, start + count), 1);
    view.setUsage(source.usage);
    geometry.setAttribute(name, view);
  }

  geometry.instanceCount = count;
  geometry.setDrawRange(parent.drawRange.start, parent.drawRange.count);
  // Bounds are the WHOLE node's on purpose. A shard's own bounds would be
  // tighter, but they are a depth slab whose extent changes on every re-sort,
  // and frustum culling a shard by stale bounds would drop visible geometry.
  geometry.boundingBox = parent.boundingBox;
  geometry.boundingSphere = parent.boundingSphere;
  // Its OWN userData, deliberately not the parent's. Sharing that object would
  // (a) let this marker leak onto the parent and (b) double-count the element
  // texture, whose bytes are accounted by stashing it on `geometry.userData`
  // for `estimateGeometryBytes`. Nothing on the shard's draw path needs it: the
  // element texture and the active sorted-index slot both reach the shader as
  // MATERIAL uniforms, and the material is shared.
  geometry.userData = { depthShard: true };
  return geometry;
}

/**
 * Group the shards' ordering views by the parent attribute each one mirrors, in
 * the shape `element-storage`'s range fan-out consumes.
 */
function collectMirrors(
  parent: THREE.InstancedBufferGeometry,
  children: readonly THREE.Mesh[]
): Map<THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute[]> {
  const mirrors = new Map<THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute[]>();
  for (const name of ['aSortedIndex', 'aSortedIndexB'] as const) {
    const source = parent.getAttribute(name) as THREE.InstancedBufferAttribute | undefined;
    if (!source) continue;
    const views: THREE.InstancedBufferAttribute[] = [];
    for (const child of children) {
      const view = (child.geometry as THREE.InstancedBufferGeometry).getAttribute(name) as
        THREE.InstancedBufferAttribute | undefined;
      if (view) views.push(view);
    }
    if (views.length > 0) mirrors.set(source, views);
  }
  return mirrors;
}

/**
 * Bring `mesh`'s shard set to `shardCount` (1 = unsharded), returning the count
 * actually established.
 *
 * Idempotent: called on every commit, because a changed element count changes
 * the shard boundaries, and a pool swap changes the geometry the views alias.
 * Rebuilds wholesale rather than resizing — shard geometries hold no state worth
 * preserving, and a partial resize is where an aliasing bug would live.
 *
 * `fullCount` is the caller's authoritative element count for this commit, NOT
 * `geometry.instanceCount`: once sharded, the parent's instance count is one
 * shard, so reading it back would shrink the node by a factor of `S` on every
 * subsequent commit.
 */
export function syncDepthShards(mesh: THREE.Mesh, shardCount: number, fullCount: number): number {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry | undefined;
  if (!geometry?.isInstancedBufferGeometry) return 1;

  const existing = shardRecords.get(mesh);
  const wanted = Math.max(1, Math.trunc(shardCount));

  if (
    existing &&
    existing.shardCount === wanted &&
    existing.geometry === geometry &&
    existing.fullCount === fullCount
  ) {
    return existing.shardCount;
  }

  releaseDepthShards(mesh);

  if (wanted <= 1 || fullCount <= 1) {
    geometry.instanceCount = fullCount;
    return 1;
  }

  const shardSize = Math.ceil(fullCount / wanted);
  const children: THREE.Mesh[] = [];
  for (let s = 1; s < wanted; s++) {
    const start = s * shardSize;
    if (start >= fullCount) break;
    const count = Math.min(shardSize, fullCount - start);
    const shardGeometry = buildShardGeometry(geometry, start, count);
    if (!shardGeometry) {
      // No ordering attribute, or an undersized one — the node cannot be
      // sharded. Undo rather than draw a partial set, which would DROP the
      // elements the missing shards were responsible for.
      for (const child of children) {
        mesh.remove(child);
      }
      geometry.instanceCount = fullCount;
      log.warning(
        Modules.RENDERER,
        `depth shards: ${mesh.name || '(unnamed)'} has no shardable ordering attribute; drawing unsharded`
      );
      return 1;
    }
    const shard = new THREE.Mesh(shardGeometry, mesh.material);
    shard.name = `${mesh.name || 'node'}__depthShard${s}`;
    shard.frustumCulled = mesh.frustumCulled;
    shard.matrixAutoUpdate = false;
    // Identity local matrix: `updateMatrixWorld` still multiplies the parent's
    // world matrix by it, so the shard inherits the node's placement exactly.
    // Deliberately NOT copying the parent's userData — no `nodeType`, no
    // element-count stamps, so every stats/monitor traversal skips this mesh.
    shard.userData = { [DEPTH_SHARD_FLAG]: mesh.name || mesh.uuid };
    mesh.add(shard);
    children.push(shard);
  }

  // Shard 0 is the parent, narrowed to its own range.
  geometry.instanceCount = Math.min(shardSize, fullCount);

  // Tell `element-storage` which views mirror each ordering attribute, so the
  // two places that register an update range on the parent can translate it into
  // each view's local range. Without this the views' GPU buffers would keep the
  // permutation they were created with — a silent wrong ordering.
  registerSortedIndexMirrors(geometry, collectMirrors(geometry, children));

  shardRecords.set(mesh, {
    children,
    shardCount: children.length + 1,
    shardSize,
    fullCount,
    geometry,
  });
  return children.length + 1;
}

/**
 * Drop `mesh`'s shard children and restore it to one full draw.
 *
 * Their geometries are NOT disposed: they share every attribute object with the
 * parent, so disposing one would free the parent's GPU buffers out from under
 * it. Dropping the references is the whole teardown — the shard geometry itself
 * owns nothing.
 */
export function releaseDepthShards(mesh: THREE.Mesh): void {
  const record = shardRecords.get(mesh);
  if (!record) return;
  for (const child of record.children) {
    mesh.remove(child);
  }
  shardRecords.delete(mesh);
  // Clear the mirror registration off the geometry the views were built
  // against, not off the mesh's CURRENT one: a pool swap can have replaced it,
  // and leaving a registration behind would keep dirtying views nothing draws.
  registerSortedIndexMirrors(record.geometry, null);
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry | undefined;
  if (geometry?.isInstancedBufferGeometry) {
    geometry.instanceCount = record.fullCount;
  }
}

/**
 * Re-point every shard at the parent's CURRENT material.
 *
 * Idempotent and called per frame from the coordinator's pump, beside the
 * `uSortedIndexSlot` re-assert it already does. That is not belt-and-braces:
 * `ui/layers/layer-apply.ts` clone-on-first-use and `scene/lod-fade.ts` both
 * REPLACE `mesh.material` with a clone, and a shard still holding the previous
 * object would silently stop tracking every appearance edit. A pointer compare
 * per shard is immune to a mutation site nobody remembered to hook.
 */
export function syncShardMaterials(mesh: THREE.Mesh): void {
  const record = shardRecords.get(mesh);
  if (!record) return;
  for (const child of record.children) {
    if (child.material !== mesh.material) child.material = mesh.material;
  }
}
