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
 *
 * `axisToComponent` maps a node's `axis` — a CENTER-COLUMN index of the stored
 * data — to the local x/y/z component that column is displayed as. The two
 * coincide only when `displayDims == [0, 1, 2]`; see {@link bspAxisToComponent}.
 */
function traverseBspBackToFront(
  node: BspTreeNode,
  eyeLocal: THREE.Vector3,
  axisToComponent: readonly number[],
  out: number[]
): void {
  if (node.part !== undefined) {
    out.push(node.part);
    return;
  }
  const component = axisToComponent[node.axis];
  const eye = component === 0 ? eyeLocal.x : component === 1 ? eyeLocal.y : eyeLocal.z;
  if (eye < node.split) {
    // Eye on the small-coord (left) side → left is near, right is far.
    traverseBspBackToFront(node.right, eyeLocal, axisToComponent, out);
    traverseBspBackToFront(node.left, eyeLocal, axisToComponent, out);
  } else {
    traverseBspBackToFront(node.left, eyeLocal, axisToComponent, out);
    traverseBspBackToFront(node.right, eyeLocal, axisToComponent, out);
  }
}

/**
 * Map each BSP split axis (a CENTER-COLUMN index of the stored data) to the
 * local x/y/z component that column is currently displayed as.
 *
 * The producer splits on the first up-to-three center columns, so a serialized
 * `axis` is 0/1/2 in STORED-COLUMN space (`partition.py::spatial_bsp_tree`).
 * `eyeLocal`, however, is in the wrapper's local 3D space, where x/y/z are
 * `displayDims[0..2]`. The two coincide only for `displayDims == [0, 1, 2]`;
 * a 4D scene displaying `[1, 2, 3]` would otherwise order along the wrong axis
 * — silently, since the result is still a valid permutation of the parts.
 *
 * Read from the LIVE dims rather than a value stamped at load: display dims can
 * change at runtime (nD navigation) while the stored tree stays valid, so a
 * load-time snapshot would go stale.
 *
 * @returns A center-column → component lookup, or `null` when any split axis is
 *   not currently displayed (its plane then carries no on-screen depth
 *   information, so the caller must fall back to the centroid heuristic).
 */
function bspAxisToComponent(tree: BspTreeNode): readonly number[] | null {
  const displayed = getDisplayDims?.();
  // No dims yet (or a 3-displayed identity map): the naive axis === component
  // reading is exactly right, and this is the overwhelmingly common case. Note
  // the app-layer accessor returns an EMPTY array before dims init, which must
  // read as "unknown" rather than as a zero-length mapping.
  if (!displayed || displayed.length === 0) return IDENTITY_AXIS_MAP;
  if (displayed.length === 3 && displayed[0] === 0 && displayed[1] === 1 && displayed[2] === 2) {
    return IDENTITY_AXIS_MAP;
  }

  // A split axis is usable only if that stored column is on screen.
  const map: number[] = [];
  for (let axis = 0; axis < 3; axis++) {
    map[axis] = displayed.indexOf(axis);
  }
  return bspTreeAxesAreMapped(tree, map) ? map : null;
}

/**
 * Live display-dims accessor, injected by `configureDepthSort` from the app
 * layer. NOT a direct `sceneDimsManager` import: `rendering/` must not depend
 * on `scene/` (`layer-rendering-no-upward`), and the same dependency inversion
 * already carries `getCamera` here and `getDisplayDims` into the LOD registry.
 */
let getDisplayDims: (() => readonly number[] | null) | null = null;

/** Wire the display-dims accessor (see {@link bspAxisToComponent}). */
export function setRenderOrderDisplayDimsAccessor(
  accessor: (() => readonly number[] | null) | null
): void {
  getDisplayDims = accessor;
}

/** Identity center-column → component map for the common `[0, 1, 2]` case. */
const IDENTITY_AXIS_MAP: readonly number[] = [0, 1, 2];

/** True when every split axis used by `tree` maps to a displayed component. */
function bspTreeAxesAreMapped(node: BspTreeNode, map: readonly number[]): boolean {
  if (node.part !== undefined) return true;
  if ((map[node.axis] ?? -1) < 0) return false;
  return bspTreeAxesAreMapped(node.left, map) && bspTreeAxesAreMapped(node.right, map);
}

/**
 * Back-to-front part RANK map for a partition wrapper (memoized per frame in
 * {@link partitionRankCache}). Transforms the camera into the wrapper's local
 * space once, traverses its `bspTree`, and numbers the resulting order
 * (0 = farthest). Returns `null` for a wrapper without a usable tree — no
 * stored tree, or split axes that aren't currently displayed.
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
  const axisToComponent = bspAxisToComponent(tree);
  if (axisToComponent === null) {
    // A split axis isn't on screen under the current display dims, so its
    // plane carries no depth information — fall back to the centroid
    // heuristic rather than ordering along the wrong axis.
    partitionRankCache.set(wrapper, null);
    return null;
  }

  const order: number[] = [];
  traverseBspBackToFront(tree, s.eyeLocal, axisToComponent, order);
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
  /** View-space bounding-sphere center (finite iff `radius >= 0`). */
  viewX: number;
  viewY: number;
  /** View-space bounding-sphere radius, or -1 without usable bounds. */
  radius: number;
}
let orderSlots: OrderSlot[] = [];

/** A per-frame order group: one partition wrapper or one single-leaf mesh. */
interface OrderGroup {
  slots: OrderSlot[];
  sumZ: number;
}

/**
 * Relative slack on the containment test: a sphere counts as containing
 * another when `dist + rInner <= rOuter * (1 + EPS)`, so a node whose
 * bounds graze the container's surface (footprint expansion, float
 * round-trip through the model-view) still registers as embedded.
 */
const CONTAINMENT_EPS = 1e-3;

/** Strict bounding-sphere containment of group `inner` inside group `outer`. */
function groupContains(
  outer: { x: number; y: number; z: number; r: number },
  inner: { x: number; y: number; z: number; r: number }
): boolean {
  // `r > 0` on both sides excludes bounds-less members (radius -1 sentinel
  // never aggregates above 0); strictly-greater radius makes every
  // containment edge point large → small, so the relation cannot cycle.
  if (outer.r <= 0 || inner.r <= 0 || outer.r <= inner.r) return false;
  const dx = outer.x - inner.x;
  const dy = outer.y - inner.y;
  const dz = outer.z - inner.z;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  return dist + inner.r <= outer.r * (1 + CONTAINMENT_EPS);
}

/**
 * Order groups farthest-first EXCEPT that a group whose bounding sphere
 * strictly contains another group's must draw before it (rationale in
 * {@link assignGlobalRenderOrder}). `byDepth` arrives sorted by mean
 * view-z; a priority topological pass (Kahn's algorithm, farthest ready
 * group emitted first) preserves that order wherever containment allows.
 * Group counts are tens, so the O(G²) edge scan is negligible next to
 * the per-frame BSP traversals this module already does.
 */
function orderGroupsWithContainment(byDepth: OrderGroup[]): OrderGroup[] {
  const n = byDepth.length;
  if (n < 2) return byDepth;

  // Enclosing sphere per group: centroid of member centers, radius the
  // max center-distance + member radius (exact for the dominant
  // single-leaf group-of-one case; a cheap upper bound for partitions).
  const spheres = byDepth.map((group) => {
    let x = 0;
    let y = 0;
    let z = 0;
    let count = 0;
    for (const slot of group.slots) {
      if (slot.radius < 0) continue;
      x += slot.viewX;
      y += slot.viewY;
      z += slot.viewZ;
      count++;
    }
    if (count === 0) return { x: 0, y: 0, z: 0, r: -1 };
    x /= count;
    y /= count;
    z /= count;
    let r = 0;
    for (const slot of group.slots) {
      if (slot.radius < 0) continue;
      const dx = slot.viewX - x;
      const dy = slot.viewY - y;
      const dz = slot.viewZ - z;
      r = Math.max(r, Math.sqrt(dx * dx + dy * dy + dz * dz) + slot.radius);
    }
    return { x, y, z, r };
  });

  // indegree[i] = number of groups that must draw before group i.
  const indegree = new Array<number>(n).fill(0);
  const containsEdges: number[][] = new Array(n);
  let anyEdge = false;
  for (let a = 0; a < n; a++) {
    const edges: number[] = [];
    for (let b = 0; b < n; b++) {
      if (a !== b && groupContains(spheres[a], spheres[b])) {
        edges.push(b);
        indegree[b]++;
        anyEdge = true;
      }
    }
    containsEdges[a] = edges;
  }
  if (!anyEdge) return byDepth;

  // Kahn's algorithm; among ready groups always emit the farthest first
  // (byDepth index order = depth order, so a linear min-scan suffices).
  const emitted = new Array<boolean>(n).fill(false);
  const ordered: OrderGroup[] = [];
  for (let step = 0; step < n; step++) {
    let pick = -1;
    for (let i = 0; i < n; i++) {
      if (!emitted[i] && indegree[i] === 0) {
        pick = i;
        break;
      }
    }
    // A cycle is impossible (edges point strictly large → small radius);
    // guard anyway so a future invariant break degrades to depth order
    // instead of dropping meshes from the rank pass.
    if (pick === -1) {
      for (let i = 0; i < n; i++) if (!emitted[i]) ordered.push(byDepth[i]);
      break;
    }
    emitted[pick] = true;
    ordered.push(byDepth[pick]);
    for (const b of containsEdges[pick]) indegree[b]--;
  }
  return ordered;
}

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
  // The camera transform is rigid, so the model-view scale IS the mesh's
  // world scale — a view-space radius stays comparable across meshes.
  const usable =
    bs !== null &&
    bs !== undefined &&
    Number.isFinite(scratch.center.x) &&
    Number.isFinite(scratch.center.y) &&
    Number.isFinite(scratch.center.z) &&
    Number.isFinite(bs.radius) &&
    bs.radius >= 0;
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
    viewZ: usable ? scratch.center.z : 0,
    viewX: usable ? scratch.center.x : 0,
    viewY: usable ? scratch.center.y : 0,
    radius: usable ? bs.radius * mv.getMaxScaleOnAxis() : -1,
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
 * 3. CONTAINMENT overrides depth: when one group's bounding sphere
 *    strictly contains another's (a small reference-marker node embedded
 *    inside a huge cloud), NO single per-mesh order integer is correct —
 *    the container's centroid sorts nearer than the embedded node for
 *    ~half of all camera orientations, and an order-dependent mode
 *    (volumetric/normal) drawn container-last multiplies the embedded
 *    node's pixels by the container's whole transmittance ≈ erases it.
 *    The container is forced to draw FIRST so embedded content composites
 *    on top: under-attenuating a marker is the lesser error vs. blinking
 *    it out entirely on camera orbit. Containment edges always point from
 *    a strictly larger to a strictly smaller sphere, so the relation is
 *    acyclic and the remaining freedom is still resolved farthest-first
 *    (a priority topological order).
 * 4. Within a group, BSP painter ranks order the parts where a stored
 *    tree exists (EXACT Fuchs–Kedem–Naylor order, any camera pose,
 *    including inside the volume — the #565 guarantee, preserved as the
 *    single-wrapper special case); otherwise members fall back to their
 *    own view-z (legacy partitions without a stored tree).
 * 5. Sequential global integers 0..M-1 are written to mesh.renderOrder.
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

  // Farthest group first (ascending mean view-z: more negative = farther),
  // then hoist strict bounding-sphere containers before their contents.
  const byDepth = [...groups.values()].sort(
    (a, b) => a.sumZ / a.slots.length - b.sumZ / b.slots.length
  );
  const ordered = orderGroupsWithContainment(byDepth);

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
