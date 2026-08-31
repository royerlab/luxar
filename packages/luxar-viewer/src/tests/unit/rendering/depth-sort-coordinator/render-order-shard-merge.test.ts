/**
 * The cross-node depth merge, driven directly through render-order.ts's
 * three-function per-frame API.
 *
 * These tests are about what the sharding is FOR: two overlapping
 * order-dependent nodes interleaving instead of one drawing entirely before the
 * other. They exercise the merge with `S > 1`, which no other suite does — the
 * coordinator's own 79 `renderOrder` assertions all run at `S === 1` and are the
 * evidence that this change is behaviour-preserving there.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  assignGlobalRenderOrder,
  clearRenderOrderFrameState,
  collectRenderOrderSlot,
  type ShardOrderInput,
} from '../../../../rendering/depth-sort-coordinator/render-order';

/** Identity model-view: local coordinates ARE view coordinates. */
const IDENTITY_MV = new THREE.Matrix4();
const CAM_POS = new THREE.Vector3(0, 0, 0);

function makeMesh(name: string, center: THREE.Vector3, radius: number): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.boundingSphere = new THREE.Sphere(center.clone(), radius);
  const mesh = new THREE.Mesh(geometry, new THREE.Material());
  mesh.name = name;
  mesh.updateMatrixWorld();
  return mesh;
}

/**
 * Build the shard input for a node whose shards span view-z `zStart..zEnd`,
 * farthest first, plus the shard meshes to carry the ranks.
 *
 * Boxes are 1-unit wide in x/y and evenly divide the z span, which is what a
 * real re-projected shard box looks like: a thin slab across the node.
 */
function shardsSpanning(
  parent: THREE.Mesh,
  count: number,
  zStart: number,
  zEnd: number
): ShardOrderInput {
  const meshes: THREE.Mesh[] = [];
  const boundsMin = new Float32Array(count * 3);
  const boundsMax = new Float32Array(count * 3);
  const step = (zEnd - zStart) / count;
  for (let s = 0; s < count; s++) {
    const lo = zStart + s * step;
    const hi = lo + step;
    boundsMin.set([-0.5, -0.5, Math.min(lo, hi)], s * 3);
    boundsMax.set([0.5, 0.5, Math.max(lo, hi)], s * 3);
    if (s > 0) {
      const shard = new THREE.Mesh(parent.geometry, parent.material);
      shard.name = `${parent.name}__s${s}`;
      shard.userData.depthShardOf = parent.name;
      parent.add(shard);
      shard.updateMatrixWorld();
      meshes.push(shard);
    }
  }
  return { meshes, count, boundsMin, boundsMax };
}

/** Every mesh that received a rank, ordered by it (lowest = drawn first). */
function drawOrder(meshes: THREE.Mesh[]): string[] {
  const all: THREE.Mesh[] = [];
  for (const m of meshes) m.traverse((o) => o instanceof THREE.Mesh && all.push(o));
  return all
    .filter((m) => m.renderOrder > 0)
    .sort((a, b) => a.renderOrder - b.renderOrder)
    .map((m) => m.name);
}

describe('cross-node depth merge', () => {
  beforeEach(() => {
    clearRenderOrderFrameState();
  });

  it('INTERLEAVES two overlapping sharded nodes', () => {
    // The whole point. A spans view-z -40..-10 and B spans -35..-5, so their
    // depth intervals interleave; with one integer per node, one would have to
    // draw entirely before the other.
    const a = makeMesh('A', new THREE.Vector3(0, 0, -25), 15);
    const b = makeMesh('B', new THREE.Vector3(0, 0, -20), 15);
    collectRenderOrderSlot(a, IDENTITY_MV, CAM_POS, shardsSpanning(a, 4, -40, -10));
    collectRenderOrderSlot(b, IDENTITY_MV, CAM_POS, shardsSpanning(b, 4, -35, -5));
    assignGlobalRenderOrder();

    const order = drawOrder([a, b]);
    expect(order).toHaveLength(8);
    // Not node-major: B's first shard lands before A's last.
    const nodeSequence = order.map((n) => n[0]);
    expect(nodeSequence.join('')).not.toBe('AAAABBBB');
    expect(new Set(nodeSequence)).toEqual(new Set(['A', 'B']));
    // Farthest first overall.
    expect(order[0]).toBe('A');
  });

  it("never re-orders a node's OWN shards, whatever the keys say", () => {
    // The invariant that makes the merge robust: shards are contiguous ranges of
    // an already back-to-front permutation, so their order is positional and is
    // never derived from a comparison. Here the boxes are deliberately given
    // IDENTICAL depth (a degenerate slab), which would leave any key-based sort
    // free to return them in any order.
    const a = makeMesh('A', new THREE.Vector3(0, 0, -20), 10);
    const flat = shardsSpanning(a, 5, -20, -20);
    const b = makeMesh('B', new THREE.Vector3(0, 0, -20), 10);
    collectRenderOrderSlot(a, IDENTITY_MV, CAM_POS, flat);
    collectRenderOrderSlot(b, IDENTITY_MV, CAM_POS, shardsSpanning(b, 5, -30, -10));
    assignGlobalRenderOrder();

    const aRanks = [a, ...flat.meshes].map((m) => m.renderOrder);
    // Shard 0 (the parent) through shard 4, strictly increasing.
    for (let i = 1; i < aRanks.length; i++) {
      expect(aRanks[i], `shard ${i} after shard ${i - 1}`).toBeGreaterThan(aRanks[i - 1]);
    }
  });

  it('keeps an UNSHARDED node atomic even while another node is sharded', () => {
    // A whole-node centroid is not a depth interval, so interleaving an
    // unsharded node by it would change draw order without improving accuracy.
    // Its slots must stay contiguous.
    const sharded = makeMesh('S', new THREE.Vector3(0, 0, -20), 15);
    const plainA = makeMesh('P1', new THREE.Vector3(0, 0, -50), 5);
    const plainB = makeMesh('P2', new THREE.Vector3(0, 0, -3), 5);
    collectRenderOrderSlot(sharded, IDENTITY_MV, CAM_POS, shardsSpanning(sharded, 4, -35, -5));
    collectRenderOrderSlot(plainA, IDENTITY_MV, CAM_POS);
    collectRenderOrderSlot(plainB, IDENTITY_MV, CAM_POS);
    assignGlobalRenderOrder();

    const order = drawOrder([sharded, plainA, plainB]);
    // The far unsharded node first, the near one last, the shards in between.
    expect(order[0]).toBe('P1');
    expect(order[order.length - 1]).toBe('P2');
  });

  it('preserves a partition wrapper’s exact BSP part order when its parts are sharded', () => {
    // Flat-sorting shard intervals by depth would discard the exact
    // Fuchs–Kedem–Naylor order; consuming each group as a monotone STREAM cannot.
    const wrapper = new THREE.Group();
    wrapper.userData.kind = 'partition';
    // No bspTree, so parts fall back to view-z ordering within the group — which
    // is still a group-internal order the merge must not break.
    const p0 = makeMesh('p0', new THREE.Vector3(0, 0, -30), 5);
    const p1 = makeMesh('p1', new THREE.Vector3(0, 0, -20), 5);
    p0.userData.partIndex = 0;
    p1.userData.partIndex = 1;
    wrapper.add(p0, p1);
    wrapper.updateMatrixWorld(true);
    const foreign = makeMesh('F', new THREE.Vector3(0, 0, -25), 10);

    collectRenderOrderSlot(p0, IDENTITY_MV, CAM_POS, shardsSpanning(p0, 2, -35, -25));
    collectRenderOrderSlot(p1, IDENTITY_MV, CAM_POS, shardsSpanning(p1, 2, -25, -15));
    collectRenderOrderSlot(foreign, IDENTITY_MV, CAM_POS, shardsSpanning(foreign, 2, -30, -20));
    assignGlobalRenderOrder();

    // Within the wrapper, every p0 slot precedes every p1 slot.
    const p0Max = Math.max(p0.renderOrder, ...p0.children.map((c) => c.renderOrder));
    const p1Min = Math.min(p1.renderOrder, ...p1.children.map((c) => c.renderOrder));
    expect(p0Max).toBeLessThan(p1Min);
  });

  it('orders a group’s MEMBERS by node depth, not by whichever shard got compared', () => {
    // Two sharded members of one wrapper, with no BSP tree so member order falls
    // back to depth. Keying that comparison on a SHARD's own view-z would make it
    // non-transitive — different shards of the same member disagree — which
    // `Array.sort` may resolve any way it likes.
    //
    // Rigged so the two keys disagree outright: p1's node centre is much farther
    // (-50 vs -10), so p1 must draw first; but p0 owns the single farthest SHARD
    // of the whole scene (≈ -75), so a shard-keyed comparison puts p0 first.
    const wrapper = new THREE.Group();
    wrapper.userData.kind = 'partition';
    const p0 = makeMesh('p0', new THREE.Vector3(0, 0, -10), 5);
    const p1 = makeMesh('p1', new THREE.Vector3(0, 0, -50), 5);
    p0.userData.partIndex = 0;
    p1.userData.partIndex = 1;
    wrapper.add(p0, p1);
    wrapper.updateMatrixWorld(true);

    collectRenderOrderSlot(p0, IDENTITY_MV, CAM_POS, shardsSpanning(p0, 2, -100, -1));
    collectRenderOrderSlot(p1, IDENTITY_MV, CAM_POS, shardsSpanning(p1, 2, -52, -48));
    assignGlobalRenderOrder();

    const p1Max = Math.max(p1.renderOrder, ...p1.children.map((c) => c.renderOrder));
    const p0Min = Math.min(p0.renderOrder, ...p0.children.map((c) => c.renderOrder));
    expect(p1Max, 'the farther NODE draws first, whole').toBeLessThan(p0Min);
  });

  it('keeps the containment override: a container draws entirely before its contents', () => {
    // PR #843. A huge cloud containing a small embedded marker must draw FIRST
    // so the marker composites on top — under-attenuating it is the lesser error
    // against it blinking out on orbit. Blocking the contained STREAM until the
    // container's is exhausted is how that survives interleaving.
    const cloud = makeMesh('cloud', new THREE.Vector3(0, 0, -50), 40);
    const marker = makeMesh('marker', new THREE.Vector3(0, 0, -50), 2);
    collectRenderOrderSlot(cloud, IDENTITY_MV, CAM_POS, shardsSpanning(cloud, 4, -90, -10));
    collectRenderOrderSlot(marker, IDENTITY_MV, CAM_POS, shardsSpanning(marker, 4, -52, -48));
    assignGlobalRenderOrder();

    const cloudMax = Math.max(cloud.renderOrder, ...cloud.children.map((c) => c.renderOrder));
    const markerMin = Math.min(marker.renderOrder, ...marker.children.map((c) => c.renderOrder));
    expect(cloudMax).toBeLessThan(markerMin);
  });

  it('assigns a contiguous 1..M with no gaps or repeats', () => {
    const a = makeMesh('A', new THREE.Vector3(0, 0, -25), 15);
    const b = makeMesh('B', new THREE.Vector3(0, 0, -20), 15);
    const plain = makeMesh('P', new THREE.Vector3(0, 0, -60), 5);
    collectRenderOrderSlot(a, IDENTITY_MV, CAM_POS, shardsSpanning(a, 3, -40, -10));
    collectRenderOrderSlot(b, IDENTITY_MV, CAM_POS, shardsSpanning(b, 3, -35, -5));
    collectRenderOrderSlot(plain, IDENTITY_MV, CAM_POS);
    assignGlobalRenderOrder();

    const all: THREE.Mesh[] = [];
    for (const m of [a, b, plain]) m.traverse((o) => o instanceof THREE.Mesh && all.push(o));
    const ranks = all.map((m) => m.renderOrder).sort((x, y) => x - y);
    expect(ranks).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it('falls back to ONE whole-node interval when the shard count is not > 1', () => {
    // What the coordinator passes for the kernel's identity-ordering fallback:
    // the ranges exist but carry no depth meaning.
    const a = makeMesh('A', new THREE.Vector3(0, 0, -30), 10);
    const b = makeMesh('B', new THREE.Vector3(0, 0, -10), 10);
    const degenerate: ShardOrderInput = {
      meshes: [],
      count: 1,
      boundsMin: new Float32Array(3),
      boundsMax: new Float32Array(3),
    };
    collectRenderOrderSlot(a, IDENTITY_MV, CAM_POS, degenerate);
    collectRenderOrderSlot(b, IDENTITY_MV, CAM_POS, degenerate);
    assignGlobalRenderOrder();
    // Exactly today's behaviour: farthest node first, one integer each.
    expect(a.renderOrder).toBe(1);
    expect(b.renderOrder).toBe(2);
  });

  it('tolerates an EMPTY shard box without poisoning the order', () => {
    // A shard with no finite element on an axis carries min=+Inf / max=-Inf. It
    // must be merged as "no depth reference" (key 0), never as NaN — which would
    // make the comparators return garbage for every other shard too.
    const a = makeMesh('A', new THREE.Vector3(0, 0, -25), 15);
    const shards = shardsSpanning(a, 3, -40, -10);
    shards.boundsMin.set([Infinity, Infinity, Infinity], 3);
    shards.boundsMax.set([-Infinity, -Infinity, -Infinity], 3);
    const b = makeMesh('B', new THREE.Vector3(0, 0, -20), 15);
    collectRenderOrderSlot(a, IDENTITY_MV, CAM_POS, shards);
    collectRenderOrderSlot(b, IDENTITY_MV, CAM_POS, shardsSpanning(b, 3, -35, -5));
    assignGlobalRenderOrder();

    const all: THREE.Mesh[] = [];
    for (const m of [a, b]) m.traverse((o) => o instanceof THREE.Mesh && all.push(o));
    const ranks = all.map((m) => m.renderOrder).sort((x, y) => x - y);
    expect(ranks).toEqual([1, 2, 3, 4, 5, 6]);
    // And A's own shards are still in index order despite one having no bounds.
    const aRanks = [a, ...shards.meshes].map((m) => m.renderOrder);
    for (let i = 1; i < aRanks.length; i++) {
      expect(aRanks[i]).toBeGreaterThan(aRanks[i - 1]);
    }
  });
});
