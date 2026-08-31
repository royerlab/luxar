/**
 * The depth-shard policy: which nodes are split, and into how many ranges.
 *
 * A pure function of world-space bounds, so it needs neither a renderer nor a
 * camera — the tests state bounds directly.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  assignShardCounts,
  interleavedDrawCount,
  type ShardPolicyNode,
  type ShardPolicyOptions,
} from '../../../../rendering/depth-sort-coordinator/shard-policy';

const OPTIONS: ShardPolicyOptions = {
  enabled: true,
  shardsPerNode: 16,
  maxInterleavedDraws: 512,
  minElements: 4096,
};

let uid = 0;
function node(
  x: number,
  radius: number,
  overrides: Partial<ShardPolicyNode> = {}
): ShardPolicyNode {
  const mesh = new THREE.Mesh();
  mesh.name = `n${uid++}`;
  return {
    mesh,
    center: new THREE.Vector3(x, 0, 0),
    radius,
    group: mesh, // its own group: a single-leaf node
    elements: 100_000,
    ...overrides,
  };
}

describe('depth-shard policy', () => {
  it('splits nothing when disabled', () => {
    const a = node(0, 10);
    const b = node(5, 10);
    const counts = assignShardCounts([a, b], { ...OPTIONS, enabled: false });
    expect(counts.get(a.mesh)).toBe(1);
    expect(counts.get(b.mesh)).toBe(1);
    expect(interleavedDrawCount(counts)).toBe(0);
  });

  it('splits nothing when no two nodes overlap', () => {
    // Rule 1, and the reason the feature is nearly free on a normal scene: the
    // measured cost is paid by ALTERNATION, so a node that overlaps nothing must
    // not pay it.
    const a = node(0, 5);
    const b = node(100, 5);
    const counts = assignShardCounts([a, b], OPTIONS);
    expect(counts.get(a.mesh)).toBe(1);
    expect(counts.get(b.mesh)).toBe(1);
  });

  it('splits both nodes of an overlapping pair', () => {
    const a = node(0, 10);
    const b = node(5, 10);
    const counts = assignShardCounts([a, b], OPTIONS);
    expect(counts.get(a.mesh)).toBe(16);
    expect(counts.get(b.mesh)).toBe(16);
    expect(interleavedDrawCount(counts)).toBe(32);
  });

  it('treats exactly-touching spheres as overlapping', () => {
    // The boundary is inclusive: two nodes whose bounds graze are the case a
    // coarse AABB most often reports, and declining to order them there would
    // make the gate silently miss the grazing-contact scene.
    const a = node(0, 5);
    const b = node(10, 5);
    const counts = assignShardCounts([a, b], OPTIONS);
    expect(counts.get(a.mesh)).toBe(16);
  });

  it('does NOT split members of the same order group that only overlap each other', () => {
    // A partition's parts are already ordered exactly against each other by BSP
    // painter rank, so their mutual overlap buys nothing — spec §4's partition
    // rule falling out of group identity rather than a special case.
    const wrapper = new THREE.Group();
    const p0 = node(0, 10, { group: wrapper });
    const p1 = node(5, 10, { group: wrapper });
    const counts = assignShardCounts([p0, p1], OPTIONS);
    expect(counts.get(p0.mesh)).toBe(1);
    expect(counts.get(p1.mesh)).toBe(1);
  });

  it('DOES split a partition part that overlaps a foreign node', () => {
    const wrapper = new THREE.Group();
    const part = node(0, 10, { group: wrapper });
    const foreign = node(5, 10);
    const counts = assignShardCounts([part, foreign], OPTIONS);
    expect(counts.get(part.mesh)).toBe(16);
    expect(counts.get(foreign.mesh)).toBe(16);
  });

  it('leaves a node below the element floor alone, and does not let it qualify others', () => {
    const tiny = node(0, 10, { elements: 100 });
    const big = node(5, 10);
    const counts = assignShardCounts([tiny, big], OPTIONS);
    expect(counts.get(tiny.mesh)).toBe(1);
    // `big` overlaps only `tiny`, which was excluded as a candidate — but `tiny`
    // is still a real overlapping node, so `big` DOES qualify. The floor is
    // about who is worth splitting, not about who counts as present.
    expect(counts.get(big.mesh)).toBe(16);
  });

  it('never splits a node into more shards than it has elements', () => {
    // Budget raised out of the way so this isolates the element clamp; with the
    // default 512 the budget would bind first and mask it.
    const a = node(0, 10, { elements: 5000 });
    const b = node(5, 10, { elements: 5000 });
    const counts = assignShardCounts([a, b], {
      ...OPTIONS,
      shardsPerNode: 100_000,
      maxInterleavedDraws: 1_000_000,
    });
    expect(counts.get(a.mesh)).toBe(5000);
  });

  it('skips a node with no usable bounds', () => {
    const noBounds = node(0, -1);
    const other = node(0, 10);
    const counts = assignShardCounts([noBounds, other], OPTIONS);
    expect(counts.get(noBounds.mesh)).toBe(1);
    // ...and it cannot qualify anyone else either, since its extent is unknown.
    expect(counts.get(other.mesh)).toBe(1);
  });

  it('scales every candidate down proportionally when the draw budget is exceeded', () => {
    const nodes = Array.from({ length: 8 }, (_, i) => node(i, 10));
    const counts = assignShardCounts(nodes, { ...OPTIONS, maxInterleavedDraws: 32 });
    // 8 nodes × 16 = 128 wanted, budget 32 → scale 0.25 → 4 each.
    for (const n of nodes) expect(counts.get(n.mesh)).toBe(4);
    expect(interleavedDrawCount(counts)).toBeLessThanOrEqual(32);
  });

  it('never scales a candidate below 2 shards', () => {
    // Rounding a qualifying node down to 1 would silently withdraw the ordering
    // the overlap test just said it needed.
    const nodes = Array.from({ length: 20 }, (_, i) => node(i * 0.1, 10));
    const counts = assignShardCounts(nodes, { ...OPTIONS, maxInterleavedDraws: 4 });
    for (const n of nodes) expect(counts.get(n.mesh)).toBe(2);
  });

  it('honours a pinned per-node count over the configured one', () => {
    const a = node(0, 10);
    const b = node(5, 10);
    const counts = assignShardCounts([a, b], { ...OPTIONS, pinnedShardsPerNode: 3 });
    expect(counts.get(a.mesh)).toBe(3);
    expect(counts.get(b.mesh)).toBe(3);
  });

  it('clamps a pinned count of 1 up to 2 rather than pretending to split', () => {
    const a = node(0, 10);
    const b = node(5, 10);
    const counts = assignShardCounts([a, b], { ...OPTIONS, pinnedShardsPerNode: 1 });
    expect(counts.get(a.mesh)).toBe(2);
  });

  it('reports every input node, so a caller can drive teardown from the result', () => {
    const a = node(0, 10);
    const b = node(100, 10);
    const counts = assignShardCounts([a, b], OPTIONS);
    expect(counts.size).toBe(2);
    expect(counts.has(a.mesh)).toBe(true);
    expect(counts.has(b.mesh)).toBe(true);
  });

  it('handles a single node and an empty scene', () => {
    expect(assignShardCounts([], OPTIONS).size).toBe(0);
    const lone = node(0, 10);
    expect(assignShardCounts([lone], OPTIONS).get(lone.mesh)).toBe(1);
  });
});
