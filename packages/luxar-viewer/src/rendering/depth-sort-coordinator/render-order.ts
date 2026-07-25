/**
 * Cross-node (inter-mesh) back-to-front `renderOrder` assignment — the
 * companion module of `depth-sort-coordinator.ts` (same facade-file +
 * same-name-directory pattern as `picking-system.ts` /
 * `picking-system/settle-scheduler.ts`).
 *
 * Depth sorting orders elements WITHIN a mesh; THREE orders transparent
 * MESHES by their `matrixWorld` origin — but every Luxar data mesh bakes
 * element centers into the geometry and shares the world origin, so
 * THREE's per-object sort key is identical for all parts and they draw
 * in fixed creation order, NOT back-to-front. This module gives THREE a
 * real signal via `renderOrder` (compared before z, ascending → lowest
 * drawn first), on ONE global integer scale across every visible
 * sorted-mode mesh the coordinator tracks (all three geometry types —
 * gsplats, points, and lines; needsDepthSort on the live mode).
 *
 * Per-frame protocol (driven by `evaluateDepthSortPerFrame`):
 * 1. {@link clearRenderOrderFrameState} at the top of the frame,
 * 2. {@link collectRenderOrderSlot} once per surviving sorted-mode mesh,
 * 3. {@link assignGlobalRenderOrder} after the loop.
 */

import * as THREE from 'three';
import type { BspTreeNode } from '../../types/partition-group';

// Per-frame scratch (no allocation on the hot path — the
// 'lod-group-selector' invariant). Allocated lazily on first use rather
// than at module load: several unit-test files partially mock 'three',
// and an import-time `new THREE.Matrix4()` would break every test that
// transitively imports this module.
interface RenderOrderScratch {
  /** Inverse of a partition wrapper's world matrix (camera → wrapper-local). */
  wrapperInv: THREE.Matrix4;
  /** Camera position in a partition wrapper's local space (BSP traversal). */
  eyeLocal: THREE.Vector3;
  /** Bounding-sphere center pushed through the model-view (view-z read). */
  center: THREE.Vector3;
}
let scratch: RenderOrderScratch | null = null;

/**
 * Per-frame cache of a partition wrapper's back-to-front part order, keyed by
 * the wrapper `THREE.Object3D`. Value maps a part index (the leaf `part` /
 * `child_index`) to its draw RANK — 0 = farthest, drawn first. `null` marks a
 * wrapper with no usable `bspTree` (→ the centroid fallback). Cleared at the
 * top of every frame via {@link clearRenderOrderFrameState} so it never
 * outlives a frame.
 */
const partitionRankCache = new Map<THREE.Object3D, Map<number, number> | null>();

/**
 * Nearest partition-wrapper ancestor of `mesh` plus the mesh's part index
 * (the `userData.partIndex` stamped on the wrapper's direct child by
 * `load-partition-group-node`). Returns `null` when `mesh` is not inside a
 * partition (a single-leaf gsplat scene) — the caller then uses the centroid
 * heuristic, a harmless no-op for one mesh.
 */
function bspPartOf(mesh: THREE.Object3D): { wrapper: THREE.Object3D; partIndex: number } | null {
  let child: THREE.Object3D = mesh;
  for (let o: THREE.Object3D | null = mesh.parent; o; child = o, o = o.parent) {
    if ((o.userData as { kind?: string }).kind === 'partition') {
      const partIndex = (child.userData as { partIndex?: number }).partIndex;
      return partIndex === undefined ? null : { wrapper: o, partIndex };
    }
  }
  return null;
}

/**
 * Emit the leaf part indices of a BSP tree in EXACT back-to-front order for an
 * eye at `eyeLocal` (the parts' own local space). At each split the eye is on
 * one side of the plane; everything on the far side draws before everything on
 * the near side (Fuchs–Kedem–Naylor painter's algorithm) — correct for any
 * camera pose, including inside the volume. `left` holds `coord < split`
 * (the near side when `eyeLocal[axis] < split`).
 */
function traverseBspBackToFront(node: BspTreeNode, eyeLocal: THREE.Vector3, out: number[]): void {
  if (node.part !== undefined) {
    out.push(node.part);
    return;
  }
  const eye = node.axis === 0 ? eyeLocal.x : node.axis === 1 ? eyeLocal.y : eyeLocal.z;
  if (eye < node.split) {
    // Eye on the small-coord (left) side → left is near, right is far.
    traverseBspBackToFront(node.right, eyeLocal, out);
    traverseBspBackToFront(node.left, eyeLocal, out);
  } else {
    traverseBspBackToFront(node.left, eyeLocal, out);
    traverseBspBackToFront(node.right, eyeLocal, out);
  }
}

/**
 * Back-to-front part RANK map for a partition wrapper (memoized per frame in
 * {@link partitionRankCache}). Transforms the camera into the wrapper's local
 * space once, traverses its `bspTree`, and numbers the resulting order
 * (0 = farthest). Returns `null` for a wrapper without a stored tree.
 */
function wrapperPartRanks(
  wrapper: THREE.Object3D,
  camPos: THREE.Vector3,
  s: RenderOrderScratch
): Map<number, number> | null {
  const cached = partitionRankCache.get(wrapper);
  if (cached !== undefined) return cached;

  const tree = (wrapper.userData as { bspTree?: BspTreeNode }).bspTree;
  if (!tree) {
    partitionRankCache.set(wrapper, null);
    return null;
  }
  wrapper.updateWorldMatrix(true, false);
  s.wrapperInv.copy(wrapper.matrixWorld).invert();
  s.eyeLocal.copy(camPos).applyMatrix4(s.wrapperInv);
  const order: number[] = [];
  traverseBspBackToFront(tree, s.eyeLocal, order);
  const ranks = new Map<number, number>();
  for (let rank = 0; rank < order.length; rank++) ranks.set(order[rank], rank);
  partitionRankCache.set(wrapper, ranks);
  return ranks;
}

/**
 * One order-pass entry per visible sorted-mode gsplat mesh, rebuilt every
 * frame (fresh array per frame — a grow-only pool would pin disposed
 * meshes across frames; counts are tens, matching the per-frame
 * allocations {@link wrapperPartRanks} already makes).
 */
interface OrderSlot {
  mesh: THREE.Mesh;
  /** Partition wrapper, or the mesh itself for a single-leaf node. */
  groupKey: THREE.Object3D;
  /** BSP painter rank within the wrapper (0 = farthest), or -1 when none. */
  partRank: number;
  /** View-space z of the bounding-sphere center (more negative = farther). */
  viewZ: number;
}
let orderSlots: OrderSlot[] = [];

/**
 * Reset the per-frame containers. Called FIRST in
 * `evaluateDepthSortPerFrame` — before any early-return — so a
 * disposed/dataset-switched frame can't leave the module-scoped cache
 * holding stale partition-wrapper subtrees alive; and from
 * `disposeDepthSort` (module-state reset completeness — an embedder that
 * disposes and re-inits in one page must not have the old scene pinned).
 */
export function clearRenderOrderFrameState(): void {
  partitionRankCache.clear();
  orderSlots = [];
}

/**
 * COLLECT half of the cross-node ordering: record one order slot for a
 * visible sorted-mode gsplat mesh. `mv` is the mesh's model-view matrix
 * and `camPos` the camera world position, both computed by the caller's
 * per-frame loop (shared with the re-sort trigger math).
 *
 * Collected slots are ASSIGNED after the loop by
 * {@link assignGlobalRenderOrder}: renderOrder is compared globally
 * across all transparent meshes, so per-wrapper BSP ranks and raw
 * view-z fallbacks must land on ONE comparable scale — mixing them
 * (two partitions, or partition + single leaf) previously drew every
 * negative-z leaf before every rank>=0 partition part regardless of
 * actual depth.
 */
export function collectRenderOrderSlot(
  mesh: THREE.Mesh,
  mv: THREE.Matrix4,
  camPos: THREE.Vector3
): void {
  if (!scratch) {
    scratch = {
      wrapperInv: new THREE.Matrix4(),
      eyeLocal: new THREE.Vector3(),
      center: new THREE.Vector3(),
    };
  }
  const bs = (mesh.geometry as THREE.BufferGeometry | undefined)?.boundingSphere;
  const part = bspPartOf(mesh);
  const ranks = part ? wrapperPartRanks(part.wrapper, camPos, scratch) : null;
  if (bs) {
    scratch.center.copy(bs.center).applyMatrix4(mv);
  }
  orderSlots.push({
    mesh,
    // Meshes sharing a partition wrapper form one order group; a
    // single-leaf mesh is its own group of one.
    groupKey: part ? part.wrapper : mesh,
    partRank: part && ranks ? (ranks.get(part.partIndex) ?? -1) : -1,
    // View-space z of the content centroid (negative in front of the
    // camera; MORE negative = farther). Without bounds — or with a
    // non-finite center (NaN input data propagates into the bbox) —
    // there is no depth reference; 0 keeps the mesh comparable
    // instead of poisoning the group-sort comparators with NaN.
    viewZ: bs && Number.isFinite(scratch.center.z) ? scratch.center.z : 0,
  });
}

/**
 * ASSIGN half of the cross-node ordering (runs after the collect loop).
 *
 * Every visible sorted-mode mesh lands on ONE global integer
 * renderOrder scale, farthest first:
 * 1. Slots group by partition wrapper (single leaves are groups of one).
 * 2. Groups order by the MEAN view-z of their members' content centroids —
 *    a documented approximation: exact inter-group ordering does not exist
 *    for arbitrarily interleaved groups, but wrappers/leaves are normally
 *    spatially disjoint datasets, and co-located overlapping layers have
 *    no meaningful cross order anyway.
 * 3. Within a group, BSP painter ranks order the parts where a stored
 *    tree exists (EXACT Fuchs–Kedem–Naylor order, any camera pose,
 *    including inside the volume — the #565 guarantee, preserved as the
 *    single-wrapper special case); otherwise members fall back to their
 *    own view-z (legacy partitions without a stored tree).
 * 4. Sequential global integers 0..M-1 are written to mesh.renderOrder.
 *
 * Transparent objects OUTSIDE the coordinator's sorted set (commutative
 * modes) keep renderOrder 0 and tie with the globally-farthest sorted
 * mesh (falling back to THREE's per-object z) — depth interleaving with
 * unsorted content stays out of scope, unchanged from the per-wrapper
 * scheme this replaces.
 */
export function assignGlobalRenderOrder(): void {
  const slots = orderSlots;
  orderSlots = [];
  if (slots.length === 0) return;

  // Group by wrapper/leaf identity (insertion order is stable).
  const groups = new Map<THREE.Object3D, { slots: OrderSlot[]; sumZ: number }>();
  for (const slot of slots) {
    const group = groups.get(slot.groupKey);
    if (group) {
      group.slots.push(slot);
      group.sumZ += slot.viewZ;
    } else {
      groups.set(slot.groupKey, { slots: [slot], sumZ: slot.viewZ });
    }
  }

  // Farthest group first (ascending mean view-z: more negative = farther).
  const ordered = [...groups.values()].sort(
    (a, b) => a.sumZ / a.slots.length - b.sumZ / b.slots.length
  );

  let nextRank = 0;
  for (const group of ordered) {
    // BSP ranks where both sides have one (a ranked wrapper ranks ALL its
    // members); view-z otherwise (rank-less legacy wrapper members).
    group.slots.sort((a, b) =>
      a.partRank >= 0 && b.partRank >= 0 ? a.partRank - b.partRank : a.viewZ - b.viewZ
    );
    for (const slot of group.slots) {
      slot.mesh.renderOrder = nextRank++;
    }
  }
}
