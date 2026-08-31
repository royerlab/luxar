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
 * sorted-mode mesh the coordinator tracks (all four geometry types —
 * gsplats, points, lines and mesh; needsDepthSort on the live mode).
 *
 * Per-frame protocol (driven by `evaluateDepthSortPerFrame`):
 * 1. {@link clearRenderOrderFrameState} at the top of the frame,
 * 2. {@link collectRenderOrderSlot} once per surviving sorted-mode mesh,
 * 3. {@link assignGlobalRenderOrder} after the loop.
 *
 * Compositing invariant (see `rendering/blending-state.ts`): `opaque` is the
 * only blending mode that escapes this sorted set entirely — it is the sole
 * mode with `transparent: false` — and it unconditionally sets
 * `depthWrite: true`. A backdrop must therefore be `opaque` to be reliably
 * composited-over by the transparent content this module orders in front of it.
 */

import * as THREE from 'three';
import type { BspTreeNode } from '../../types/partition-group';
import { log, Modules } from '../../utils/log';

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

/** Partition wrappers already diagnosed for an unusable display-axis mapping. */
const warnedAxisMappingWrappers = new WeakSet<THREE.Object3D>();

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
 *
 * NOTE for mesh partitions (`add_mesh(partition=…)`, spec §9.2): back-to-front is
 * *correct but pointless* for OPAQUE parts — it forfeits the front-to-back early-Z
 * rejection an opaque pass would rather have. Native mesh partitions now retain
 * their BSP because a translucent mesh genuinely needs this order; opaque meshes
 * accept the harmless ordering cost rather than carrying a second metadata policy.
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
 * A serialized `axis` is in STORED-COLUMN space. `eyeLocal`, however, is in the
 * wrapper's local 3D space, where x/y/z are `displayDims[0..2]`. The two
 * coincide only for `displayDims == [0, 1, 2]`; a 4D scene displaying
 * `[1, 2, 3]` needs column 3 mapped to z.
 *
 * Read from the LIVE dims rather than a value stamped at load: display dims can
 * change at runtime (nD navigation) while the stored tree stays valid, so a
 * load-time snapshot would go stale.
 *
 * @returns A center-column → component lookup, or `null` when any split axis
 *   cannot be mapped to a displayed component. Without display metadata only
 *   the conventional first three columns can be mapped safely.
 */
function bspAxisToComponent(
  tree: BspTreeNode,
  displayed: readonly number[] | null | undefined
): readonly number[] | null {
  // The app-layer accessor returns an EMPTY array before dims init. Treat that
  // as unknown: the conventional first-three mapping is usable only when every
  // split axis actually belongs to it.
  if (!displayed || displayed.length === 0) {
    return bspTreeAxesAreMapped(tree, IDENTITY_AXIS_MAP) ? IDENTITY_AXIS_MAP : null;
  }
  if (displayed.length === 3 && displayed[0] === 0 && displayed[1] === 1 && displayed[2] === 2) {
    return bspTreeAxesAreMapped(tree, IDENTITY_AXIS_MAP) ? IDENTITY_AXIS_MAP : null;
  }

  // A split axis is usable only if that stored column is on screen.
  const map: number[] = [];
  for (let component = 0; component < displayed.length; component++) {
    map[displayed[component]] = component;
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
  const displayed = getDisplayDims?.();
  const axisToComponent = bspAxisToComponent(tree, displayed);
  if (axisToComponent === null) {
    // A split axis isn't on screen under the current display dims, so its
    // plane carries no depth information — fall back to the centroid
    // heuristic rather than ordering along the wrong axis.
    if (!warnedAxisMappingWrappers.has(wrapper)) {
      warnedAxisMappingWrappers.add(wrapper);
      log.warning(
        Modules.RENDERER,
        `partition-kind group ${wrapper.name || '(unnamed)'} has a valid bsp_tree whose split axes are not all displayed (displayDims=[${displayed?.join(', ') ?? ''}]); falling back to centroid ordering`
      );
    }
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
  /** View-space bounding-sphere center (zeroed without usable bounds). */
  viewX: number;
  viewY: number;
  /**
   * View-space bounding-sphere radius; -1 without usable bounds, or when
   * the world scale overflows the finite local radius to non-finite (the
   * center then still carries a valid depth reference).
   */
  radius: number;
  /**
   * Depth-shard index within the node, 0 for the node's own mesh. Shards are
   * contiguous ranges of an ALREADY back-to-front permutation, so ascending
   * index IS back-to-front and this doubles as the within-node order key —
   * which is why the merge never needs to compare two shards of one node.
   */
  shardIndex: number;
  /**
   * The NODE's mesh, shared by every shard of it (so `mesh` for shard 0). The
   * group's internal sort keys on this rather than on `mesh`, which keeps a
   * node's shards adjacent and in index order no matter what their individual
   * depth keys say — including a shard with no usable bounds, whose key would
   * otherwise sort it to the near end of its own node.
   */
  memberKey: THREE.Mesh;
  /**
   * View-space z of the NODE's own bounding-sphere centre — identical for every
   * shard of it. This is the group's member-ordering key, so a group orders its
   * members exactly as it did before sharding existed; the per-shard `viewZ`
   * only ever positions a shard against OTHER groups.
   */
  memberViewZ: number;
}

/**
 * A node's depth shards, as the render-order pass needs them.
 *
 * Supplied by the coordinator rather than queried here: it already owns the
 * per-node sort state the bounds arrive on, and passing them keeps this module
 * free of any dependency on the shard machinery.
 */
export interface ShardOrderInput {
  /** Shard meshes 1..S-1; shard 0 is the node's own mesh. */
  meshes: readonly THREE.Mesh[];
  /** Shards with usable depth bounds. `<= 1` means "do not shard the order". */
  count: number;
  /** Per-shard LOCAL-space AABB minima, `[count * 3]`. */
  boundsMin: Float32Array;
  /** Per-shard LOCAL-space AABB maxima, `[count * 3]`. */
  boundsMax: Float32Array;
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

/** Enclosing sphere of two member spheres `a` and `b` (Ritter seed). */
function encloseTwoSpheres(
  a: { x: number; y: number; z: number; r: number },
  b: { x: number; y: number; z: number; r: number }
): { x: number; y: number; z: number; r: number } {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dz = b.z - a.z;
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  // One sphere already swallows the other (covers the concentric d===0 case,
  // so the division below is only reached with d > 0).
  if (d + b.r <= a.r) return { x: a.x, y: a.y, z: a.z, r: a.r };
  if (d + a.r <= b.r) return { x: b.x, y: b.y, z: b.z, r: b.r };
  const r = (d + a.r + b.r) / 2;
  const t = (r - a.r) / d; // center sits `r - a.r` along a → b
  return { x: a.x + dx * t, y: a.y + dy * t, z: a.z + dz * t, r };
}

/**
 * Tight enclosing sphere of a group's member spheres via Ritter's algorithm.
 * Each member is `{viewX, viewY, viewZ, radius}`; members with `radius < 0`
 * carry no usable bounds and are skipped. A group with no usable members
 * yields the `{r: -1}` sentinel (unchanged from the old bound); a single-member
 * group yields exactly that member's sphere (the dominant single-leaf case,
 * kept exact).
 *
 * Seed from an extreme member pair (farthest-of-farthest by center distance +
 * radius), then iteratively expand: for a member at distance `d` with radius
 * `rm`, when `d + rm > R` grow to `R' = (R + d + rm) / 2` and shift the center
 * toward the member by `R' - R`. Ritter hugs a cluster-plus-outlier layout far
 * tighter than the legacy centroid + max-reach bound, but on near-symmetric
 * layouts it OVERSHOOTS the exact circumsphere, so it is not universally
 * tighter — and its center is offset, so a smaller radius alone does not make
 * its containment relation a subset of the legacy one.
 * {@link orderGroupsWithContainment} therefore requires an edge to hold under
 * BOTH this and {@link centroidMaxReachSphere}.
 */
function groupEnclosingSphere(slots: readonly OrderSlot[]): {
  x: number;
  y: number;
  z: number;
  r: number;
} {
  const members = slots.filter((s) => s.radius >= 0);
  if (members.length === 0) return { x: 0, y: 0, z: 0, r: -1 };
  const first = members[0];
  if (members.length === 1) {
    return { x: first.viewX, y: first.viewY, z: first.viewZ, r: first.radius };
  }

  // Extreme-pair seed: farthest member from an arbitrary one, then the
  // farthest from that (center distance + the member's own radius = reach).
  const farthestFrom = (m: OrderSlot): OrderSlot => {
    let best = m;
    let bestReach = -Infinity;
    for (const s of members) {
      const dx = s.viewX - m.viewX;
      const dy = s.viewY - m.viewY;
      const dz = s.viewZ - m.viewZ;
      const reach = Math.sqrt(dx * dx + dy * dy + dz * dz) + s.radius;
      if (reach > bestReach) {
        bestReach = reach;
        best = s;
      }
    }
    return best;
  };
  const a = farthestFrom(first);
  const b = farthestFrom(a);
  const sphere = encloseTwoSpheres(
    { x: a.viewX, y: a.viewY, z: a.viewZ, r: a.radius },
    { x: b.viewX, y: b.viewY, z: b.viewZ, r: b.radius }
  );

  // Expand to cover any member sphere still poking outside the current bound.
  for (const s of members) {
    const dx = s.viewX - sphere.x;
    const dy = s.viewY - sphere.y;
    const dz = s.viewZ - sphere.z;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d + s.radius <= sphere.r) continue; // already covered
    const rNew = (sphere.r + d + s.radius) / 2;
    if (d > 0) {
      const t = (rNew - sphere.r) / d; // shift center toward s by (rNew - R)
      sphere.x += dx * t;
      sphere.y += dy * t;
      sphere.z += dz * t;
    }
    sphere.r = rNew;
  }
  return sphere;
}

/**
 * Legacy enclosing sphere: centroid of the usable member centers, radius the
 * max over members of (center-distance + member radius). Skips members with
 * `radius < 0` and yields the `{r: -1}` sentinel when none are usable; exact
 * for a single member. Paired with {@link groupEnclosingSphere}:
 * {@link orderGroupsWithContainment} requires a containment edge to hold under
 * both bounds, so the tighter Ritter sphere can only remove legacy false
 * edges, never introduce new off-center ones.
 */
function centroidMaxReachSphere(slots: readonly OrderSlot[]): {
  x: number;
  y: number;
  z: number;
  r: number;
} {
  let x = 0;
  let y = 0;
  let z = 0;
  let count = 0;
  for (const slot of slots) {
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
  for (const slot of slots) {
    if (slot.radius < 0) continue;
    const dx = slot.viewX - x;
    const dy = slot.viewY - y;
    const dz = slot.viewZ - z;
    r = Math.max(r, Math.sqrt(dx * dx + dy * dy + dz * dz) + slot.radius);
  }
  return { x, y, z, r };
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
function orderGroupsWithContainment(byDepth: OrderGroup[]): {
  ordered: OrderGroup[];
  /** Groups each key must finish drawing before; empty when no edge applies. */
  edges: Map<OrderGroup, OrderGroup[]>;
} {
  const n = byDepth.length;
  if (n < 2) return { ordered: byDepth, edges: EMPTY_EDGES };

  // Two valid enclosing spheres per group: the Ritter union
  // ({@link groupEnclosingSphere}) and the legacy centroid + max-reach bound
  // ({@link centroidMaxReachSphere}); `tight` is whichever has the smaller
  // radius (Ritter overshoots near-symmetric layouts, centroid overshoots
  // cluster-plus-outlier ones).
  //
  // A containment edge requires the relation to hold under BOTH spheres. The
  // tight sphere alone is not enough: a smaller Ritter sphere is OFF-CENTER,
  // so on some layouts it contains a disjoint neighbour the centroid sphere
  // never did — a NEW false-positive edge that could regress draw order on
  // scenes the old bound handled correctly. Requiring the legacy bound to
  // agree makes the edge set a strict SUBSET of the legacy relation (the
  // tight bound only ever REMOVES false edges — e.g. the biodiversity
  // cluster-plus-outlier one, Earth globe wrongly "inside" the spread-out
  // `All life` tiles — never relocates them), while a true containment (an
  // embedded node inside a member sphere) satisfies both bounds and is kept.
  // Acyclicity is preserved: every edge points strictly large → small in the
  // tight radius.
  const spheres = byDepth.map((group) => {
    const ritter = groupEnclosingSphere(group.slots);
    // No usable members → both bounds are the {r: -1} sentinel.
    if (ritter.r < 0) return { tight: ritter, legacy: ritter };
    const legacy = centroidMaxReachSphere(group.slots);
    return { tight: legacy.r <= ritter.r ? legacy : ritter, legacy };
  });

  // indegree[i] = number of groups that must draw before group i.
  const indegree = new Array<number>(n).fill(0);
  const containsEdges: number[][] = new Array(n);
  let anyEdge = false;
  for (let a = 0; a < n; a++) {
    const edges: number[] = [];
    for (let b = 0; b < n; b++) {
      if (
        a !== b &&
        groupContains(spheres[a].tight, spheres[b].tight) &&
        groupContains(spheres[a].legacy, spheres[b].legacy)
      ) {
        edges.push(b);
        indegree[b]++;
        anyEdge = true;
      }
    }
    containsEdges[a] = edges;
  }
  if (!anyEdge) return { ordered: byDepth, edges: EMPTY_EDGES };

  // Publish the edge set by GROUP identity, so a caller holding either the depth
  // order or the hoisted order can use it without index bookkeeping. The merge
  // needs it: inferring edges by diffing the two orderings would be a guess, not
  // the relation — Kahn's priority tie-breaking reorders groups that no edge
  // connects, and a transitive chain looks the same as a direct edge.
  const edgeMap = new Map<OrderGroup, OrderGroup[]>();
  for (let a = 0; a < n; a++) {
    if (containsEdges[a].length > 0) {
      edgeMap.set(
        byDepth[a],
        containsEdges[a].map((b) => byDepth[b])
      );
    }
  }

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
  return { ordered, edges: edgeMap };
}

/** Shared empty edge map for the no-containment case (no per-frame allocation). */
const EMPTY_EDGES: Map<OrderGroup, OrderGroup[]> = new Map();

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
  camPos: THREE.Vector3,
  shards?: ShardOrderInput
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
  const groupKey = part ? part.wrapper : mesh;
  const partRank = part && ranks ? (ranks.get(part.partIndex) ?? -1) : -1;

  // Sharded: one slot per shard, each positioned by its OWN re-projected box.
  // This is where cross-node interleaving becomes expressible at all — a shard
  // is a thin depth interval, so unlike a whole-node centroid it can be compared
  // against another node's interval meaningfully.
  if (shards && shards.count > 1) {
    const scale = mv.getMaxScaleOnAxis();
    // The node's own centroid depth, used as the member-ordering key on every
    // shard so the group orders its MEMBERS exactly as it did pre-sharding.
    let memberViewZ = 0;
    if (bs && Number.isFinite(bs.radius)) {
      scratch.center.copy(bs.center).applyMatrix4(mv);
      if (Number.isFinite(scratch.center.z)) memberViewZ = scratch.center.z;
    }
    for (let s = 0; s < shards.count; s++) {
      // Shard 0 is the node's own mesh; 1..S-1 are its shard children.
      const shardMesh = s === 0 ? mesh : shards.meshes[s - 1];
      if (!shardMesh) break;
      const b = s * 3;
      const minX = shards.boundsMin[b];
      const minY = shards.boundsMin[b + 1];
      const minZ = shards.boundsMin[b + 2];
      const maxX = shards.boundsMax[b];
      const maxY = shards.boundsMax[b + 1];
      const maxZ = shards.boundsMax[b + 2];
      // `min > max` on any axis is the empty-box sentinel (a shard with no
      // finite element there); a non-finite bound is a real ±Inf center. Both
      // mean "no usable depth reference", handled exactly as a missing bounding
      // sphere is: keep the slot comparable at 0 rather than poisoning the
      // comparators with NaN, and mark the radius unusable.
      const usable =
        minX <= maxX &&
        minY <= maxY &&
        minZ <= maxZ &&
        Number.isFinite(minX + minY + minZ + maxX + maxY + maxZ);
      if (usable) {
        scratch.center.set((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
        scratch.center.applyMatrix4(mv);
      }
      const dx = maxX - minX;
      const dy = maxY - minY;
      const dz = maxZ - minZ;
      const localRadius = usable ? Math.sqrt(dx * dx + dy * dy + dz * dz) / 2 : -1;
      const scaledRadius = usable ? localRadius * scale : -1;
      orderSlots.push({
        mesh: shardMesh,
        groupKey,
        partRank,
        viewZ: usable ? scratch.center.z : 0,
        viewX: usable ? scratch.center.x : 0,
        viewY: usable ? scratch.center.y : 0,
        radius: Number.isFinite(scaledRadius) ? scaledRadius : -1,
        shardIndex: s,
        memberKey: mesh,
        memberViewZ,
      });
    }
    return;
  }

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
  const scaledRadius = usable ? bs.radius * mv.getMaxScaleOnAxis() : -1;
  orderSlots.push({
    mesh,
    // Meshes sharing a partition wrapper form one order group; a
    // single-leaf mesh is its own group of one.
    groupKey,
    partRank,
    // View-space z of the content centroid (negative in front of the
    // camera; MORE negative = farther). Without bounds — or with a
    // non-finite center (NaN input data propagates into the bbox) —
    // there is no depth reference; 0 keeps the mesh comparable
    // instead of poisoning the group-sort comparators with NaN.
    viewZ: usable ? scratch.center.z : 0,
    viewX: usable ? scratch.center.x : 0,
    viewY: usable ? scratch.center.y : 0,
    radius: Number.isFinite(scaledRadius) ? scaledRadius : -1,
    shardIndex: 0,
    memberKey: mesh,
    memberViewZ: usable ? scratch.center.z : 0,
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
 *    tree ranks EVERY member (EXACT Fuchs–Kedem–Naylor order, any camera
 *    pose, including inside the volume — the #565 guarantee, preserved as
 *    the single-wrapper special case); otherwise the whole group falls
 *    back to member view-z (a partition with no stored tree, or one whose
 *    tree does not name every part).
 * 5. Sequential global integers 1..M are written to mesh.renderOrder.
 *
 * Transparent objects OUTSIDE the coordinator's sorted set (commutative
 * modes, plus empty parts that have never committed) keep renderOrder 0
 * and draw before the globally-farthest sorted mesh. Depth interleaving
 * with unsorted content stays out of scope, unchanged from the
 * per-wrapper scheme this replaces.
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
      // MEMBER depth, not shard depth: a single-leaf group's key is then exactly
      // the node's own centroid whether or not it is sharded, so splitting a
      // node cannot move it relative to the other groups.
      group.sumZ += slot.memberViewZ;
    } else {
      groups.set(slot.groupKey, { slots: [slot], sumZ: slot.memberViewZ });
    }
  }

  // Farthest group first (ascending mean view-z: more negative = farther),
  // then hoist strict bounding-sphere containers before their contents.
  const byDepth = [...groups.values()].sort(
    (a, b) => a.sumZ / a.slots.length - b.sumZ / b.slots.length
  );
  const { ordered, edges } = orderGroupsWithContainment(byDepth);

  // Reserve THREE's default 0 for meshes the coordinator does not track yet,
  // so an empty never-committed partition placeholder cannot alias a live rank.
  let nextRank = 1;

  // Order each group's slots internally FIRST, so a group is a monotone stream:
  // a sequence already in its own correct back-to-front order.
  for (const group of ordered) {
    sortGroupSlots(group);
  }

  // When any group is sharded the assignment becomes a k-way merge of those
  // streams; otherwise it stays exactly the group-major emission it has always
  // been (see {@link mergeOrderStreams} for why that split is deliberate).
  if (ordered.some(groupIsSharded)) {
    mergeOrderStreams(ordered, edges);
    return;
  }

  for (const group of ordered) {
    for (const slot of group.slots) {
      slot.mesh.renderOrder = nextRank++;
    }
  }
}

/**
 * Put one group's slots into that group's own correct back-to-front order.
 *
 * BSP ranks when EVERY member has one; view-z otherwise (a rank-less legacy
 * wrapper, or a partition with no stored tree).
 *
 * The all-or-nothing choice is made once per GROUP, not per comparison, and
 * that is load-bearing. Mixing the two keys makes the comparator
 * NON-TRANSITIVE — ranked pairs order by rank while any pair involving an
 * unranked member orders by depth, so a < b < c < a is representable — and
 * `Array.sort` is free to return anything for an inconsistent comparator,
 * including an order that violates both keys. A ranked wrapper normally ranks
 * all its members, but `ranks.get(partIndex) ?? -1` yields -1 for a part the
 * stored tree does not name, so the mixed case is reachable from data alone;
 * falling the whole group back to depth keeps the result a well-defined
 * (if approximate) order instead of an arbitrary one.
 *
 * The keys are MEMBER-level, not shard-level, and that is the point. A node's
 * shards must stay adjacent and in index order regardless of their individual
 * depth keys: their correct order is not a matter of comparison at all — they are
 * contiguous ranges of an already back-to-front permutation, so ascending index
 * IS back-to-front. Keying on the shard's own `viewZ` instead would let a shard
 * with no usable bounds (key 0, i.e. AT the camera) sort to the near end of its
 * own node, and would let float noise swap two adjacent slabs — which would make
 * the sharding buy nothing at all.
 */
function sortGroupSlots(group: OrderGroup): void {
  const everyMemberRanked = group.slots.every((slot) => slot.partRank >= 0);
  const primary = everyMemberRanked
    ? (a: OrderSlot, b: OrderSlot) => a.partRank - b.partRank
    : (a: OrderSlot, b: OrderSlot) => a.memberViewZ - b.memberViewZ;
  group.slots.sort((a, b) => {
    if (a.memberKey === b.memberKey) return a.shardIndex - b.shardIndex;
    const byPrimary = primary(a, b);
    if (byPrimary !== 0) return byPrimary;
    return a.shardIndex - b.shardIndex;
  });
}

/** True when any member of `group` was split into depth shards. */
function groupIsSharded(group: OrderGroup): boolean {
  return group.slots.some((slot) => slot.shardIndex > 0);
}

/**
 * Assign `renderOrder` by merging the groups as MONOTONE STREAMS rather than
 * emitting them one after another.
 *
 * Each group arrives already in its own correct order (see
 * {@link sortGroupSlots}) and is only ever consumed front-to-back, which is what
 * makes this safe: a partition wrapper's exact Fuchs–Kedem–Naylor part order and
 * a node's own shard order are preserved BY CONSTRUCTION, never re-derived from a
 * comparison. The merge is therefore robust to depth-key noise — a stale or
 * imprecise key can pick the wrong stream next, but can never mis-order one
 * node's own shards.
 *
 * **An UNSHARDED group is emitted atomically, keyed by its mean view-z** — which
 * is exactly the group-major behaviour that has always applied, and is not a
 * compatibility hack: a whole-node centroid is not a depth interval, so
 * interleaving two unsharded nodes by it would change draw order without
 * improving accuracy. Only a sharded group is consumed shard-by-shard, keyed by
 * the head shard's re-projected centroid. With nothing sharded this reduces to
 * the previous implementation exactly, integer for integer.
 *
 * Containment (PR #843) is preserved as a constraint on the merge: a contained
 * group's stream stays blocked until its container's stream is EXHAUSTED, so the
 * container still draws entirely first. That is Kahn's algorithm over streams,
 * with the farthest ready head emitted first — the same priority-topological
 * shape {@link orderGroupsWithContainment} already applies to whole groups.
 *
 * `ordered` is the group list after the containment hoist; `edges` is that
 * hoist's actual containment relation, keyed by group identity.
 */
function mergeOrderStreams(ordered: OrderGroup[], edges: Map<OrderGroup, OrderGroup[]>): void {
  const n = ordered.length;
  const cursors = new Array<number>(n).fill(0);
  const sharded = ordered.map(groupIsSharded);
  // Mean view-z per group: the atomic streams' selection key, identical to the
  // one the depth sort that produced `ordered` was built with.
  const meanZ = ordered.map(
    (g) => g.slots.reduce((sum, s) => sum + s.memberViewZ, 0) / Math.max(1, g.slots.length)
  );

  const indexOf = new Map<OrderGroup, number>();
  ordered.forEach((g, i) => indexOf.set(g, i));
  const blockedBy = new Array<number>(n).fill(0);
  const blocks: number[][] = Array.from({ length: n }, () => []);
  for (let a = 0; a < n; a++) {
    for (const contained of edges.get(ordered[a]) ?? []) {
      const b = indexOf.get(contained);
      if (b === undefined || b === a) continue;
      blocks[a].push(b);
      blockedBy[b]++;
    }
  }

  let nextRank = 1;
  let remaining = ordered.reduce((sum, g) => sum + g.slots.length, 0);
  while (remaining > 0) {
    // Farthest ready head first. Streams are tens at most, so a linear scan is
    // cheaper than a heap and keeps the tie-breaking obvious: on an exact tie the
    // lower stream index wins, which is depth order after the hoist.
    let pick = -1;
    let pickKey = Infinity;
    for (let i = 0; i < n; i++) {
      if (cursors[i] >= ordered[i].slots.length || blockedBy[i] > 0) continue;
      const key = sharded[i] ? ordered[i].slots[cursors[i]].viewZ : meanZ[i];
      if (key < pickKey) {
        pickKey = key;
        pick = i;
      }
    }

    if (pick === -1) {
      // Every remaining stream is blocked — impossible while containment edges
      // point strictly large → small radius, but guard so a future invariant
      // break degrades to the hoisted order instead of dropping meshes from the
      // rank pass (the same posture as the cycle guard in the group hoist).
      for (let i = 0; i < n; i++) {
        for (let c = cursors[i]; c < ordered[i].slots.length; c++) {
          ordered[i].slots[c].mesh.renderOrder = nextRank++;
        }
        cursors[i] = ordered[i].slots.length;
      }
      return;
    }

    if (sharded[pick]) {
      ordered[pick].slots[cursors[pick]].mesh.renderOrder = nextRank++;
      cursors[pick]++;
      remaining--;
    } else {
      // Atomic: the whole group at once, preserving today's group-major emission.
      for (const slot of ordered[pick].slots) {
        slot.mesh.renderOrder = nextRank++;
      }
      remaining -= ordered[pick].slots.length;
      cursors[pick] = ordered[pick].slots.length;
    }

    if (cursors[pick] >= ordered[pick].slots.length) {
      for (const b of blocks[pick]) blockedBy[b]--;
    }
  }
}
