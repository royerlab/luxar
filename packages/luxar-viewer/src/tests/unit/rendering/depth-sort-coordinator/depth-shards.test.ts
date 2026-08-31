/**
 * Depth-shard mesh construction and teardown.
 *
 * Real THREE objects throughout, no GL: the properties under test are all
 * structural (which mesh draws which range, which buffer a view aliases, which
 * attribute is marked dirty), so a mocked renderer would test the mock.
 *
 * The suite is deliberately weighted toward `S > 1`. The whole rest of the
 * viewer's 13k unit tests already cover `S === 1`, which is production today —
 * so anything asserted only at `S === 1` here would be vacuous.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  depthShardChildren,
  depthShardCount,
  effectiveInstanceCount,
  isDepthShard,
  isDepthShardGeometry,
  releaseDepthShards,
  syncDepthShards,
  syncShardMaterials,
} from '../../../../rendering/depth-sort-coordinator/depth-shards';
import { writeSortedIndexIdentity } from '../../../../rendering/element-storage';

/**
 * A committed instanced node, mirroring what `attachElementStorage` produces:
 * a base quad, an index, and TWO distinct full-capacity ordering buffers.
 */
function makeNode(count: number, capacity = count): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    'aQuadCorner',
    new THREE.BufferAttribute(new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), 2)
  );
  geometry.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 2, 1, 3]), 1));
  for (const name of ['aSortedIndex', 'aSortedIndexB'] as const) {
    const attr = new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1);
    attr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, attr);
  }
  geometry.instanceCount = count;
  geometry.setDrawRange(0, 6);
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);
  geometry.boundingBox = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  const mesh = new THREE.Mesh(geometry, new THREE.Material());
  mesh.name = 'node';
  mesh.userData.nodeType = 'gsplats';
  mesh.userData.visibleSplatCount = count;
  return mesh;
}

const orderingOf = (mesh: THREE.Mesh, name = 'aSortedIndex'): THREE.InstancedBufferAttribute =>
  (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute(
    name
  ) as THREE.InstancedBufferAttribute;

describe('depth shards', () => {
  describe('shard set construction', () => {
    it('S=1 is the identity: no children, full instance count', () => {
      const mesh = makeNode(100);
      expect(syncDepthShards(mesh, 1, 100)).toBe(1);
      expect(mesh.children).toHaveLength(0);
      expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(100);
      expect(depthShardCount(mesh)).toBe(1);
    });

    it('the PARENT draws shard 0 and is never left at zero instances', () => {
      // Load-bearing: `lod-fade` treats a mesh WITH a material as the whole
      // node, and camera framing / blend warmup both bail on instanceCount <= 0.
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(25);
      expect(mesh.children).toHaveLength(3);
    });

    it('shard instance counts PARTITION the element count', () => {
      for (const [count, shards] of [
        [100, 4],
        [7, 3],
        [10, 3],
        [5, 5],
        [1000, 16],
      ] as const) {
        const mesh = makeNode(count);
        syncDepthShards(mesh, shards, count);
        const parent = (mesh.geometry as THREE.InstancedBufferGeometry).instanceCount;
        const kids = depthShardChildren(mesh).map(
          (c) => (c.geometry as THREE.InstancedBufferGeometry).instanceCount
        );
        expect(parent + kids.reduce((a, b) => a + b, 0), `count=${count} shards=${shards}`).toBe(
          count
        );
      }
    });

    it('shard views cover DISJOINT, contiguous, in-order ranges of the permutation', () => {
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      const parentArray = orderingOf(mesh).array as Uint32Array;
      const offsets = [
        0,
        ...depthShardChildren(mesh).map((c) => {
          const a = orderingOf(c).array as Uint32Array;
          return a.byteOffset / a.BYTES_PER_ELEMENT;
        }),
      ];
      const lengths = [
        (mesh.geometry as THREE.InstancedBufferGeometry).instanceCount,
        ...depthShardChildren(mesh).map((c) => (orderingOf(c).array as Uint32Array).length),
      ];
      // Every shard starts exactly where the previous one ended, and together
      // they cover [0, count) with no gap and no overlap.
      let cursor = 0;
      for (let s = 0; s < offsets.length; s++) {
        expect(offsets[s], `shard ${s} start`).toBe(cursor);
        cursor += lengths[s];
      }
      expect(cursor).toBe(100);
      // All views alias the SAME buffer as the parent — no ordering was copied.
      for (const child of depthShardChildren(mesh)) {
        expect((orderingOf(child).array as Uint32Array).buffer).toBe(parentArray.buffer);
      }
    });

    it('a shard view SEES writes made through the parent array', () => {
      // The property the whole layout rests on: one contiguous CPU array, so
      // every existing sorted-index writer keeps working untouched.
      const mesh = makeNode(8);
      syncDepthShards(mesh, 4, 8);
      const parentArray = orderingOf(mesh).array as Uint32Array;
      for (let i = 0; i < 8; i++) parentArray[i] = 100 + i;
      const last = depthShardChildren(mesh)[2];
      expect(Array.from(orderingOf(last).array as Uint32Array)).toEqual([106, 107]);
    });

    it('shards carry BOTH ordering slots, so the shared slot uniform stays valid', () => {
      const mesh = makeNode(20);
      syncDepthShards(mesh, 4, 20);
      for (const child of depthShardChildren(mesh)) {
        expect(orderingOf(child, 'aSortedIndex')).toBeDefined();
        expect(orderingOf(child, 'aSortedIndexB')).toBeDefined();
      }
    });

    it('shards share the base quad and index attribute OBJECTS', () => {
      const mesh = makeNode(20);
      const parent = mesh.geometry as THREE.InstancedBufferGeometry;
      syncDepthShards(mesh, 3, 20);
      for (const child of depthShardChildren(mesh)) {
        const geo = child.geometry as THREE.InstancedBufferGeometry;
        expect(geo.getAttribute('aQuadCorner')).toBe(parent.getAttribute('aQuadCorner'));
        expect(geo.index).toBe(parent.index);
        expect(geo.drawRange.count).toBe(parent.drawRange.count);
      }
    });

    it('declines to shard a geometry with no ordering attribute, drawing it whole', () => {
      // Partial sharding would DROP the elements the missing shards covered, so
      // the only safe answer is to draw unsharded.
      const mesh = makeNode(40);
      (mesh.geometry as THREE.InstancedBufferGeometry).deleteAttribute('aSortedIndex');
      expect(syncDepthShards(mesh, 4, 40)).toBe(1);
      expect(mesh.children).toHaveLength(0);
      expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(40);
    });

    it('declines to shard a single-element node', () => {
      const mesh = makeNode(1);
      expect(syncDepthShards(mesh, 8, 1)).toBe(1);
      expect(mesh.children).toHaveLength(0);
    });

    it('caps the shard count at the element count rather than making empty shards', () => {
      const mesh = makeNode(3);
      const established = syncDepthShards(mesh, 10, 3);
      expect(established).toBe(3);
      expect(mesh.children).toHaveLength(2);
      for (const child of depthShardChildren(mesh)) {
        expect((child.geometry as THREE.InstancedBufferGeometry).instanceCount).toBeGreaterThan(0);
      }
    });
  });

  describe('shards are invisible to node-level traversals', () => {
    it('carries no nodeType or element-count stamps', () => {
      // This is what makes scene-stats / visible-counts / the draw-order panel /
      // blend warmup skip shards by CONSTRUCTION instead of double-counting.
      const mesh = makeNode(40);
      syncDepthShards(mesh, 4, 40);
      for (const child of depthShardChildren(mesh)) {
        expect(child.userData.nodeType).toBeUndefined();
        expect(child.userData.visibleSplatCount).toBeUndefined();
        expect(child.userData.visiblePointCount).toBeUndefined();
        expect(child.userData.visibleSegmentCount).toBeUndefined();
        expect(isDepthShard(child)).toBe(true);
        expect(isDepthShardGeometry(child.geometry)).toBe(true);
      }
      expect(isDepthShard(mesh)).toBe(false);
      expect(isDepthShardGeometry(mesh.geometry)).toBe(false);
    });

    it('does not stamp the shard marker onto the parent geometry userData', () => {
      // A shard geometry must NOT share the parent's userData object: the
      // element texture's bytes are accounted by stashing it there, so sharing
      // would double-count it (and leak this marker upward).
      const mesh = makeNode(40);
      (mesh.geometry as THREE.InstancedBufferGeometry).userData.elementTexture = { fake: true };
      syncDepthShards(mesh, 4, 40);
      expect(mesh.geometry.userData.depthShard).toBeUndefined();
      for (const child of depthShardChildren(mesh)) {
        expect(child.geometry.userData).not.toBe(mesh.geometry.userData);
        expect(child.geometry.userData.elementTexture).toBeUndefined();
      }
    });

    it('effectiveInstanceCount reports the WHOLE node, sharded or not', () => {
      const mesh = makeNode(100);
      expect(effectiveInstanceCount(mesh)).toBe(100);
      syncDepthShards(mesh, 4, 100);
      expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(25);
      expect(effectiveInstanceCount(mesh)).toBe(100);
      releaseDepthShards(mesh);
      expect(effectiveInstanceCount(mesh)).toBe(100);
    });
  });

  describe('material identity', () => {
    it('shards share the parent material OBJECT, never a clone', () => {
      // `lod-fade` mutates the one material it finds on the node; a cloned
      // shard material would silently stop tracking cross-fade and every
      // LayersPanel edit.
      const mesh = makeNode(40);
      syncDepthShards(mesh, 4, 40);
      for (const child of depthShardChildren(mesh)) {
        expect(child.material).toBe(mesh.material);
      }
    });

    it('syncShardMaterials re-points shards after the parent material is REPLACED', () => {
      const mesh = makeNode(40);
      syncDepthShards(mesh, 4, 40);
      const replacement = new THREE.Material();
      mesh.material = replacement; // what layer-apply's clone-on-first-use does
      for (const child of depthShardChildren(mesh)) {
        expect(child.material).not.toBe(replacement);
      }
      syncShardMaterials(mesh);
      for (const child of depthShardChildren(mesh)) {
        expect(child.material).toBe(replacement);
      }
    });

    it('syncShardMaterials is a no-op on an unsharded node', () => {
      const mesh = makeNode(40);
      expect(() => syncShardMaterials(mesh)).not.toThrow();
    });
  });

  describe('update-range fan-out', () => {
    it('translates a parent-array write into each overlapping view LOCAL range', () => {
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      const children = depthShardChildren(mesh);
      for (const child of children) orderingOf(child).clearUpdateRanges();
      // `needsUpdate` is WRITE-ONLY on a THREE BufferAttribute (the setter bumps
      // `version`), so `version` is the readable proof it was set.
      const versionsBefore = children.map((c) => orderingOf(c).version);

      // `writeSortedIndexIdentity` registers `[0, 100)` on the parent, which
      // spans every shard.
      writeSortedIndexIdentity(mesh.geometry as THREE.InstancedBufferGeometry, 100);

      for (const [i, child] of children.entries()) {
        const view = orderingOf(child);
        expect(view.version, `shard ${i + 1} dirtied`).toBeGreaterThan(versionsBefore[i]);
        expect(view.updateRanges, `shard ${i + 1} ranges`).toEqual([{ start: 0, count: 25 }]);
      }
    });

    it('leaves views the write did NOT touch clean', () => {
      // The reason this maps ranges instead of dirtying everything: the chunked
      // apply exists to BOUND ordering upload per frame, and blanket dirtying
      // would multiply that bound by the shard count.
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      const children = depthShardChildren(mesh);
      orderingOf(mesh).clearUpdateRanges();
      for (const child of children) orderingOf(child).clearUpdateRanges();
      const versionsBefore = children.map((c) => orderingOf(c).version);

      // A write confined to the parent's own shard [0, 25).
      writeSortedIndexIdentity(mesh.geometry as THREE.InstancedBufferGeometry, 20);

      for (const [i, child] of children.entries()) {
        expect(orderingOf(child).updateRanges, `shard ${i + 1} untouched`).toEqual([]);
        expect(orderingOf(child).version, `shard ${i + 1} not dirtied`).toBe(versionsBefore[i]);
      }
    });

    it('stops fanning out once the shards are released', () => {
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      const child = depthShardChildren(mesh)[0];
      const view = orderingOf(child);
      releaseDepthShards(mesh);
      view.clearUpdateRanges();
      const versionBefore = view.version;
      writeSortedIndexIdentity(mesh.geometry as THREE.InstancedBufferGeometry, 100);
      expect(view.updateRanges).toEqual([]);
      expect(view.version).toBe(versionBefore);
    });
  });

  describe('re-sync and release', () => {
    it('rebuilds when the element count changes, without shrinking the node', () => {
      // The bug this pins: inferring the total from `geometry.instanceCount`
      // after sharding would read back ONE shard and shrink the node by a
      // factor of S on every subsequent commit.
      const mesh = makeNode(100, 200);
      syncDepthShards(mesh, 4, 100);
      expect(effectiveInstanceCount(mesh)).toBe(100);

      syncDepthShards(mesh, 4, 200);
      expect(effectiveInstanceCount(mesh)).toBe(200);
      const parent = (mesh.geometry as THREE.InstancedBufferGeometry).instanceCount;
      const kids = depthShardChildren(mesh).map(
        (c) => (c.geometry as THREE.InstancedBufferGeometry).instanceCount
      );
      expect(parent + kids.reduce((a, b) => a + b, 0)).toBe(200);
    });

    it('is idempotent for an unchanged request', () => {
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      const before = depthShardChildren(mesh).slice();
      syncDepthShards(mesh, 4, 100);
      expect(depthShardChildren(mesh)).toEqual(before);
    });

    it('RE-NARROWS the parent when a commit has reset its instance count', () => {
      // Every commit sets `geometry.instanceCount` to the whole element count
      // before calling here. An unchanged request takes the idempotent path, so
      // that path must still re-assert the narrowing — otherwise the parent draws
      // the ENTIRE node on top of its shards, i.e. every element twice, which in
      // an order-dependent mode is a visibly wrong composite and not a crash.
      // Found by the two-node E2E: 9375 elements drawn for a 5000-element node.
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
      expect(geometry.instanceCount).toBe(25);

      geometry.instanceCount = 100; // what a commit does
      syncDepthShards(mesh, 4, 100);

      expect(geometry.instanceCount).toBe(25);
      const total =
        geometry.instanceCount +
        depthShardChildren(mesh).reduce(
          (sum, c) => sum + (c.geometry as THREE.InstancedBufferGeometry).instanceCount,
          0
        );
      expect(total, 'every element drawn exactly once').toBe(100);
    });

    it('rebuilds when the pooled geometry is swapped underneath', () => {
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      const staleView = orderingOf(depthShardChildren(mesh)[0]);
      mesh.geometry = makeNode(100).geometry; // what a pool best-fit swap does
      syncDepthShards(mesh, 4, 100);
      const freshView = orderingOf(depthShardChildren(mesh)[0]);
      expect(freshView).not.toBe(staleView);
      expect((freshView.array as Uint32Array).buffer).toBe(
        (orderingOf(mesh).array as Uint32Array).buffer
      );
    });

    it('release restores one full draw and detaches every child', () => {
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      releaseDepthShards(mesh);
      expect(mesh.children).toHaveLength(0);
      expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(100);
      expect(depthShardCount(mesh)).toBe(1);
      expect(depthShardChildren(mesh)).toEqual([]);
    });

    it('release is idempotent and safe on an unsharded node', () => {
      const mesh = makeNode(100);
      expect(() => releaseDepthShards(mesh)).not.toThrow();
      syncDepthShards(mesh, 4, 100);
      releaseDepthShards(mesh);
      expect(() => releaseDepthShards(mesh)).not.toThrow();
      expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(100);
    });

    it('going back to S=1 through sync also restores one full draw', () => {
      const mesh = makeNode(100);
      syncDepthShards(mesh, 4, 100);
      expect(syncDepthShards(mesh, 1, 100)).toBe(1);
      expect(mesh.children).toHaveLength(0);
      expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(100);
    });
  });
});
