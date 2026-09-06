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
 * Once-per-object memos for the three `layer_order` diagnostics. Same shape and
 * lifetime as {@link warnedAxisMappingWrappers}: a `WeakSet`/`WeakMap` keyed on
 * the scene object, so a warning cannot repeat every frame (this module runs on
 * EVERY frame, ungated by camera motion) and cannot pin a disposed subtree.
 */
const warnedMixedBandGroups = new WeakSet<THREE.Object3D>();
const warnedBandSplitContainers = new WeakMap<THREE.Object3D, WeakSet<THREE.Object3D>>();
const warnedBucketOrderGroups = new WeakSet<THREE.Object3D>();

/** A group whose members disagree on their band (spec D6 violated on disk). */
function warnMixedBandGroup(groupKey: THREE.Object3D, kept: number, seen: number): void {
  if (warnedMixedBandGroups.has(groupKey)) return;
  warnedMixedBandGroups.add(groupKey);
  log.warning(
    Modules.RENDERER,
    `Order group '${groupKey.name || '(unnamed)'}' has members on different layer orders ` +
      `(${kept} and ${seen}). A layer_order authored INSIDE a partition/LOD wrapper is refused ` +
      `at write time; this store carries one anyway. Using ${kept} for the whole group — the ` +
      'wrapper is the layer, and splitting it across bands would destroy its exact part order.'
  );
}

/**
 * An authored band split a containment relation the renderer would otherwise
 * have honoured. Not an error — it is the author overriding an inferred order —
 * but it is the one way a `layer_order` can make an embedded node nearly
 * disappear (the container's whole transmittance multiplies it), so it must be
 * diagnosable rather than mysterious.
 */
function warnBandSplitContainment(outer: OrderGroup, inner: OrderGroup): void {
  const outerKey = outer.slots[0]?.groupKey;
  const innerKey = inner.slots[0]?.groupKey;
  if (!outerKey || !innerKey) return;
  let seen = warnedBandSplitContainers.get(outerKey);
  if (!seen) {
    seen = new WeakSet<THREE.Object3D>();
    warnedBandSplitContainers.set(outerKey, seen);
  }
  if (seen.has(innerKey)) return;
  seen.add(innerKey);
  log.warning(
    Modules.RENDERER,
    `'${outerKey.name || '(unnamed)'}' (layer order ${outer.level}) spatially CONTAINS ` +
      `'${innerKey.name || '(unnamed)'}' (layer order ${inner.level}), but their authored orders ` +
      'put them in different bands, so the container-draws-first rule is not applied. That is ' +
      'the authored order winning, as intended — but if the inner layer looks washed out or ' +
      'vanishes, this is why: give both the same layer_order to restore the inferred ordering.'
  );
}

/**
 * An authored order that conflicts with THREE's fixed opaque-before-transparent
 * bucket order. `renderOrder` is compared only WITHIN a bucket, so an opaque
 * group cannot actually draw after a transparent group even when its authored
 * level is greater. Equal levels cover the mixed-band case: the groups are
 * meant to share one band, but the bucket split still separates them.
 */
function orderGroupBucket(group: OrderGroup): 'opaque' | 'transparent' | 'mixed' {
  let sawOpaque = false;
  let sawTransparent = false;
  for (const slot of group.slots) {
    const material = Array.isArray(slot.mesh.material) ? slot.mesh.material[0] : slot.mesh.material;
    if (material?.transparent) sawTransparent = true;
    else sawOpaque = true;
  }
  return sawOpaque && sawTransparent ? 'mixed' : sawOpaque ? 'opaque' : 'transparent';
}

function warnBucketOrderConflict(groups: OrderGroup[]): void {
  let hasUnwarnedAuthoredGroup = false;
  for (const group of groups) {
    const key = group.slots[0]?.groupKey;
    if (group.levelExplicit && key && !warnedBucketOrderGroups.has(key)) {
      hasUnwarnedAuthoredGroup = true;
      break;
    }
  }
  if (!hasUnwarnedAuthoredGroup) return;

  let conflictOpaque: OrderGroup | undefined;
  let conflictTransparent: OrderGroup | undefined;
  conflict: for (const opaqueGroup of groups) {
    if (!opaqueGroup.levelExplicit || orderGroupBucket(opaqueGroup) === 'transparent') continue;
    for (const transparentGroup of groups) {
      if (
        !transparentGroup.levelExplicit ||
        orderGroupBucket(transparentGroup) === 'opaque' ||
        opaqueGroup.level < transparentGroup.level
      ) {
        continue;
      }
      const opaqueKey = opaqueGroup.slots[0]?.groupKey;
      const transparentKey = transparentGroup.slots[0]?.groupKey;
      if (!opaqueKey && !transparentKey) continue;
      if (
        (!opaqueKey || warnedBucketOrderGroups.has(opaqueKey)) &&
        (!transparentKey || warnedBucketOrderGroups.has(transparentKey))
      ) {
        continue;
      }
      conflictOpaque = opaqueGroup;
      conflictTransparent = transparentGroup;
      break conflict;
    }
  }
  if (!conflictOpaque || !conflictTransparent) return;

  const opaqueKey = conflictOpaque.slots[0]?.groupKey;
  const transparentKey = conflictTransparent.slots[0]?.groupKey;
  if (opaqueKey) warnedBucketOrderGroups.add(opaqueKey);
  if (transparentKey) warnedBucketOrderGroups.add(transparentKey);
  log.warning(
    Modules.RENDERER,
    `Authored layer order puts an opaque group at ${conflictOpaque.level} and a transparent ` +
      `group at ${conflictTransparent.level}, but every opaque mesh draws before every ` +
      'transparent mesh. renderOrder is only compared within a bucket, so this ordering cannot be ' +
      'honoured. Use compatible blending modes if their relative order matters.'
  );
}

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
   * Authored cross-layer draw order (`LAYER_ORDER_SPEC.md`), `0` when
   * unset. Read ONCE here rather than in the comparator, which runs
   * O(G log G) times per frame.
   */
  level: number;
  /**
   * Whether the order was authored, as opposed to defaulted to 0. Ordering is
   * based only on `level`; this bit gates the cross-bucket diagnostic so an
   * inferred band is not reported as author intent (spec D2/D3).
   */
  levelExplicit: boolean;
  /**
   * The mesh must draw before everything else in its band — a transmissive
   * (glass) physical mesh, see {@link drawsBeforeEmissive}.
   */
  first: boolean;
  /**
   * The mesh must draw AFTER everything else in its band — a transmissive physical
   * mesh authored with `refract_data`, see {@link drawsAfterEmissive}. Never true
   * together with `first`.
   */
  last: boolean;
}
let orderSlots: OrderSlot[] = [];

/** A per-frame order group: one partition wrapper or one single-leaf mesh. */
interface OrderGroup {
  slots: OrderSlot[];
  sumZ: number;
  /**
   * The group's band. Taken from its FIRST member: composition gives every
   * part of a wrapper its wrapper's order, and authoring one strictly
   * inside a specialized group is refused (spec D6), so members normally
   * agree. A hand-edited store can disagree — that warns once and the first
   * member wins, which keeps the band deterministic either way.
   */
  level: number;
  /** True when the winning member's order was authored rather than defaulted. */
  levelExplicit: boolean;
  /** True when the group draws first in its band (its first member says so). */
  first: boolean;
  /** True when the group draws last in its band (its first member says so). */
  last: boolean;
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
function orderGroupsWithContainment(byDepth: OrderGroup[]): OrderGroup[] {
  const n = byDepth.length;
  if (n < 2) return byDepth;

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
        // A containment edge is honoured only WITHIN a band. Bands are hard
        // partitions of the draw order, so a cross-band edge is not merely
        // outranked — it is unrepresentable, and an authored level is the
        // author's stated intent overriding an inferred relation (spec D3).
        //
        // Dropping the edge is what makes the whole band scheme work with the
        // EXISTING Kahn pass and no restructuring: indegree then counts only
        // same-band predecessors, so every band always holds a zero-indegree
        // member, and the "lowest ready index first" rule below walks the
        // (level, meanZ)-sorted array band by band on its own.
        if (byDepth[a].level !== byDepth[b].level) {
          if (byDepth[b].level < byDepth[a].level) {
            warnBandSplitContainment(byDepth[a], byDepth[b]);
          }
          continue;
        }
        // A glass that must draw LAST in its band (`refract_data`) — or one that must
        // draw FIRST and would be the CONTENT here — states an intent the inferred
        // container-first rule must not undo. A lens enclosing a cluster is exactly
        // "container contains content", and hoisting the lens first would paint the
        // cluster crisp on top instead of refracting it. Same standing as an authored
        // band (spec D3), so no diagnostic.
        if (byDepth[a].last || byDepth[b].last || byDepth[b].first) continue;
        edges.push(b);
        indegree[b]++;
        anyEdge = true;
      }
    }
    containsEdges[a] = edges;
  }
  // No honoured edge → the incoming (level, meanZ) order already IS the answer,
  // bands included, since it is sorted band-first.
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
  const scaledRadius = usable ? bs.radius * mv.getMaxScaleOnAxis() : -1;
  const authored = authoredLayerOrder(mesh);
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
    radius: Number.isFinite(scaledRadius) ? scaledRadius : -1,
    level: authored ?? 0,
    levelExplicit: authored !== undefined,
    first: drawsBeforeEmissive(mesh),
    last: drawsAfterEmissive(mesh),
  });
}

/**
 * Whether a mesh asks to be drawn BEFORE the emissive data sharing its band.
 *
 * Stamped as `userData.drawBeforeEmissive` by the physical material's compositing
 * decision (`materials/mesh-physical/config.ts`) for a transmissive (glass) surface.
 * On WebGPU a transmissive mesh lives in the same transparent render list as every
 * point, line and splat layer and lands its fragments with alpha 1, so if a cluster's
 * centre sorted farther than the sphere the cluster would draw first and be painted
 * over; ordering the glass first keeps the spec §3.4 contract ("data in front and
 * behind stays visible, unrefracted") on both backends. Read off the MATERIAL's
 * `userData` (where the live `transmission` slider keeps it current), through the
 * record rather than the material type for the same layering reason as
 * {@link authoredLayerOrder}. A multi-material mesh never asks.
 */
export function drawsBeforeEmissive(mesh: THREE.Mesh): boolean {
  const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
  if (!material || Array.isArray(material)) return false;
  return (material.userData as { drawBeforeEmissive?: unknown }).drawBeforeEmissive === true;
}

/**
 * Whether a mesh asks to be drawn AFTER the emissive data sharing its band — the
 * Phase 3 inverse of {@link drawsBeforeEmissive}.
 *
 * Stamped as `userData.drawAfterEmissive` by the physical material's compositing
 * decision for glass authored with `refract_data` (spec §3.4). Such a glass must
 * sample a framebuffer that already holds the data: on WebGPU three's transmission
 * reads the framebuffer as it stands when the glass draws, so ranking it LAST in its
 * band IS the whole mechanism; on WebGL the post-processing pipeline splits the scene
 * pass instead and the rank is bookkeeping. Read off the MATERIAL's `userData` like
 * its twin (the live Layers-panel toggle keeps it current there). A multi-material
 * mesh never asks. The coordinator gives such a mesh a rank even when nothing else
 * would — the unranked emissive layers sit at renderOrder 0, and "after them" is not
 * a value 0 can express.
 */
export function drawsAfterEmissive(mesh: THREE.Mesh): boolean {
  const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
  if (!material || Array.isArray(material)) return false;
  return (material.userData as { drawAfterEmissive?: unknown }).drawAfterEmissive === true;
}

/**
 * The authored `layer_order` on a mesh, or `undefined` when unset.
 *
 * Read off the dedicated `userData.layerOrder` render-state slot, which every
 * `create-*-node.ts` stamps from the COMPOSED attrs record. Keeping it separate
 * from `userData.attrs` matters for lines/gsplats, where that record is the raw
 * leaf attrs object owned by the loaded scene graph.
 * A plain property read on purpose: `rendering/` may not import `data/`
 * (`layer-rendering-no-upward`, enforced by `pnpm check:layers`), which is why
 * this takes no type from the composer.
 *
 * Re-sanitised even though the composer already did: a mesh can reach the
 * coordinator without passing through `applyEffectiveAttrs` (synthetic scenes,
 * unit fixtures), and a non-finite level would poison the band comparator for
 * the whole frame.
 */
export function authoredLayerOrder(mesh: THREE.Mesh): number | undefined {
  const raw = (mesh.userData as { layerOrder?: unknown } | undefined)?.layerOrder;
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}

/**
 * ASSIGN half of the cross-node ordering (runs after the collect loop).
 *
 * Every visible sorted-mode mesh lands on ONE global integer
 * renderOrder scale, farthest first:
 * 1. Slots group by partition wrapper (single leaves are groups of one).
 * 2. Groups band by authored `layer_order`, lower first; unset means band 0.
 * 3. Within a band, groups order by the MEAN view-z of their members' content centroids —
 *    a documented approximation: exact inter-group ordering does not exist
 *    for arbitrarily interleaved groups, but wrappers/leaves are normally
 *    spatially disjoint datasets, and co-located overlapping layers have
 *    no meaningful cross order anyway.
 * 4. Within a band, CONTAINMENT overrides depth: when one group's bounding sphere
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
 * 5. Within a group, BSP painter ranks order the parts where a stored
 *    tree ranks EVERY member (EXACT Fuchs–Kedem–Naylor order, any camera
 *    pose, including inside the volume — the #565 guarantee, preserved as
 *    the single-wrapper special case); otherwise the whole group falls
 *    back to member view-z (a partition with no stored tree, or one whose
 *    tree does not name every part).
 * 6. Sequential global integers 1..M are written to mesh.renderOrder.
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
  const groups = new Map<THREE.Object3D, OrderGroup>();
  for (const slot of slots) {
    const group = groups.get(slot.groupKey);
    if (group) {
      group.slots.push(slot);
      group.sumZ += slot.viewZ;
      // The wrapper defines the band, so its members must agree. Disagreement
      // is only reachable from a store that authored a level strictly inside a
      // specialized group (refused at write time, tolerated on read): warn and
      // keep the first member's band.
      if (slot.level !== group.level) warnMixedBandGroup(slot.groupKey, group.level, slot.level);
    } else {
      groups.set(slot.groupKey, {
        slots: [slot],
        sumZ: slot.viewZ,
        level: slot.level,
        levelExplicit: slot.levelExplicit,
        first: slot.first,
        last: slot.last,
      });
    }
  }

  // Band first (lower level = farther = drawn first), then any group that must
  // draw first in its band (glass — see `drawsBeforeEmissive`), then any group that
  // must draw LAST in its band goes after the rest (refracting glass — see
  // `drawsAfterEmissive`), then farthest group within a band (ascending mean
  // view-z: more negative = farther), then hoist strict bounding-sphere containers
  // before their contents.
  //
  // With no level authored anywhere every group is band 0, and with no glass the
  // second and third terms are always 0, so this falls through to EXACTLY today's
  // comparator — which, together with the intra-band edge restriction below, is
  // what makes an unauthored scene bit-identical to before the feature existed.
  const byDepth = [...groups.values()].sort(
    (a, b) =>
      a.level - b.level ||
      Number(!a.first) - Number(!b.first) ||
      Number(a.last) - Number(b.last) ||
      a.sumZ / a.slots.length - b.sumZ / b.slots.length
  );
  const ordered = orderGroupsWithContainment(byDepth);
  warnBucketOrderConflict(ordered);

  // Reserve THREE's default 0 for meshes the coordinator does not track yet,
  // so an empty never-committed partition placeholder cannot alias a live rank.
  let nextRank = 1;
  for (const group of ordered) {
    // BSP ranks when EVERY member has one; view-z otherwise (a rank-less legacy
    // wrapper, or a partition with no stored tree).
    //
    // The all-or-nothing choice is made once per GROUP, not per comparison, and
    // that is load-bearing. Mixing the two keys makes the comparator
    // NON-TRANSITIVE — ranked pairs order by rank while any pair involving an
    // unranked member orders by depth, so a < b < c < a is representable — and
    // `Array.sort` is free to return anything for an inconsistent comparator,
    // including an order that violates both keys. A ranked wrapper normally ranks
    // all its members, but `ranks.get(partIndex) ?? -1` yields -1 for a part the
    // stored tree does not name, so the mixed case is reachable from data alone;
    // falling the whole group back to depth keeps the result a well-defined
    // (if approximate) order instead of an arbitrary one.
    const everyMemberRanked = group.slots.every((slot) => slot.partRank >= 0);
    group.slots.sort(
      everyMemberRanked ? (a, b) => a.partRank - b.partRank : (a, b) => a.viewZ - b.viewZ
    );
    for (const slot of group.slots) {
      slot.mesh.renderOrder = nextRank++;
    }
  }
}
