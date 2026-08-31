/**
 * How many depth shards each order-dependent node should be split into.
 *
 * Pure function of the tracked nodes' world-space bounds — no camera, no frame
 * state — which is what makes it cheap to re-evaluate and stable to apply:
 * whether two nodes overlap changes only when a node commits or the tracked set
 * changes, never as the camera moves. Design: `CROSS_NODE_DEPTH_ORDERING_SPEC.md`
 * §4.
 *
 * ## The measured reason this gates coarsely
 *
 * The cost of splitting draws was measured (spec §7) and it is **sublinear in
 * draw count and saturating**: at 2M splats, interleaving cost +5.5 ms at 128
 * interleaved draws and +6.4 ms at 512 — only ~1 ms apart despite 4× the draws,
 * while interleaving *at all* cost the first +5.5 ms. The dominant term follows
 * node ALTERNATION, not draw count, and is GPU-side (most plausibly
 * element-texture locality plus pipeline state).
 *
 * So the lever is **which nodes interleave**, not how finely they are sharded.
 * A node that overlaps nothing gets 1 and pays nothing; a node that qualifies
 * gets a generous count. Tuning the count down would forfeit accuracy without
 * recovering the cost. The draw budget exists only to bound the small linear
 * per-draw term, which dominates at low element counts — where it is also
 * cheap in absolute terms.
 */

import * as THREE from 'three';

/** One tracked order-dependent node, as the policy needs to see it. */
export interface ShardPolicyNode {
  mesh: THREE.Mesh;
  /** World-space bounding-sphere centre. */
  center: THREE.Vector3;
  /** World-space bounding-sphere radius; `< 0` when the node has no usable bounds. */
  radius: number;
  /**
   * Order-group identity — the partition wrapper, or the mesh itself for a
   * single leaf. Members of ONE group are already ordered exactly against each
   * other (BSP painter ranks), so overlapping only within a group buys nothing.
   */
  group: THREE.Object3D;
  /** The node's whole element count. */
  elements: number;
}

export interface ShardPolicyOptions {
  /** Master switch (`config.depthShards.enabled` ∧ `?depthShards` ≠ 0). */
  enabled: boolean;
  /** Shards an overlapping node gets, unless pinned or budget-limited. */
  shardsPerNode: number;
  /** Soft ceiling on the total interleaved draw count across the scene. */
  maxInterleavedDraws: number;
  /** Nodes below this element count are never split (the split cannot pay). */
  minElements: number;
  /** `?depthShards=N` — pins the per-node count, bypassing `shardsPerNode`. */
  pinnedShardsPerNode?: number;
}

/**
 * Decide each node's shard count. Every input node appears in the result, with 1
 * meaning "draw as one call", so a caller can drive `syncDepthShards` straight
 * from it without tracking which nodes it asked about.
 */
export function assignShardCounts(
  nodes: readonly ShardPolicyNode[],
  options: ShardPolicyOptions
): Map<THREE.Mesh, number> {
  const result = new Map<THREE.Mesh, number>();
  for (const node of nodes) result.set(node.mesh, 1);
  if (!options.enabled || nodes.length < 2) return result;

  const requested = Math.max(2, Math.trunc(options.pinnedShardsPerNode ?? options.shardsPerNode));
  if (!Number.isFinite(requested)) return result;

  // Rule 1: only a node whose bounds overlap a FOREIGN order-dependent node can
  // benefit. Same-group members (a partition's parts) are already exactly
  // ordered against each other, so their mutual overlap does not qualify —
  // which is spec §4's partition rule falling out of the group identity rather
  // than needing its own special case.
  const candidates: ShardPolicyNode[] = [];
  for (const node of nodes) {
    if (node.radius < 0 || node.elements < options.minElements) continue;
    if (
      nodes.some((other) => other !== node && other.group !== node.group && overlaps(node, other))
    )
      candidates.push(node);
  }
  if (candidates.length === 0) return result;

  // A node cannot have more shards than elements, and one element per shard is
  // the degenerate limit where the split stops meaning anything.
  const perNode = candidates.map((node) => Math.max(2, Math.min(requested, node.elements)));

  // Bound the linear per-draw term. Scaling every candidate by the same factor
  // keeps the allocation neutral between nodes — the alternative (dropping the
  // least-overlapping first) needs an overlap-extent measure that spec §4 rules
  // 3-4 describe but this pass does not yet compute, and guessing at it would be
  // worse than treating candidates equally.
  const total = perNode.reduce((sum, s) => sum + s, 0);
  const budget = Math.max(2, Math.trunc(options.maxInterleavedDraws));
  const scale = total > budget ? budget / total : 1;

  for (const [i, node] of candidates.entries()) {
    const scaled = scale < 1 ? Math.floor(perNode[i] * scale) : perNode[i];
    // Never below 2: a "sharded" node with one shard is just an unsharded node
    // paying the bookkeeping, and rounding a candidate down to 1 would silently
    // withdraw the ordering the overlap test said it needed.
    result.set(node.mesh, Math.max(2, Math.min(scaled, node.elements)));
  }
  return result;
}

/** World-space bounding-sphere intersection. */
function overlaps(a: ShardPolicyNode, b: ShardPolicyNode): boolean {
  if (b.radius < 0) return false;
  const dx = a.center.x - b.center.x;
  const dy = a.center.y - b.center.y;
  const dz = a.center.z - b.center.z;
  const reach = a.radius + b.radius;
  return dx * dx + dy * dy + dz * dz <= reach * reach;
}

/**
 * Total interleaved draws the assignment implies — the quantity the §7 budget is
 * stated in, surfaced for the monitor and the perf gate. Counts only nodes that
 * were actually split, since an unsharded node contributes the one draw it
 * always did.
 */
export function interleavedDrawCount(counts: Map<THREE.Mesh, number>): number {
  let total = 0;
  for (const count of counts.values()) if (count > 1) total += count;
  return total;
}
