/**
 * Geometry-behaviour row `sortedAppendHold`: in a sorted mode, a commit that
 * EXTENDS the drawn population keeps drawing the previous population, in its
 * sorted order, until the grown ordering lands.
 *
 * Drawing the old back-to-front prefix with the appended elements after it in
 * storage order composites every new element over the prefix regardless of
 * depth, for a SortWorker round trip per ladder rung; a pool grow without a
 * seed is worse, drawing the whole node in storage order. GSplats held the
 * draw in both cases while Points and Lines did neither, because the rule was
 * written once per type. It is now one rule in two halves —
 * `planInstancedOrdering` (the commit decides append vs seeded grow from the
 * prefix lineage) and `writeInstancedCommitOrdering` (the pool adapter holds
 * the draw) — and this probe drives both, end to end, through the real
 * `SceneLoader` commit and a real `GPUBufferPool`: commit, land a sort, commit
 * an extension, and read what the geometry draws.
 *
 * Mesh is declared absent (no-op): its draw order IS its index buffer, which
 * a commit rewrites whole, so the extension draws its full population in
 * storage order at once and the next sort permutes it atomically
 * (`triangle-ordering.ts`). The mesh adapter ignores `extending` because
 * `mesh-progressive-loader.ts` stamps no prefix lineage; this probe asserts
 * the current whole-index rewrite, not a future lineage-triggered hold.
 */
import * as THREE from 'three';
import { expect } from 'vitest';

import { SceneLoader } from '../../../data/scene-loader';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import {
  getActiveSortedIndexAttribute,
  pumpSortedIndexOrderingApply,
  writeSortedIndexOrdering,
} from '../../../rendering/element-storage';
import { commitInternals, GEOMETRY_COMMITS, makeCommitScene } from '../../helpers/geometry-commits';
import { defineBehaviourConformance } from '../../_conformance/define-behaviour-conformance';

/** A loader over a real pool and a scene of one node per type. */
function pooledLoader(): { loader: SceneLoader; root: THREE.Group } {
  const loader = new SceneLoader({ enableMonitor: false });
  commitInternals(loader).lodGroupRegistry = null;
  commitInternals(loader)._gpuBufferPool = new GPUBufferPool();
  return { loader, root: makeCommitScene(loader) };
}

/** The permutation the draw samples: the active ordering's first `instanceCount` slots. */
function drawnOrder(geometry: THREE.InstancedBufferGeometry): number[] {
  const attr = getActiveSortedIndexAttribute(geometry)!;
  return Array.from((attr.array as Uint32Array).subarray(0, geometry.instanceCount));
}

defineBehaviourConformance('sortedAppendHold', {
  async holds(type) {
    const { loader, root } = pooledLoader();
    const adapter = GEOMETRY_COMMITS[type];
    const node = () => root.getObjectByName(adapter.path) as THREE.Mesh;

    await adapter.commit(loader, 4);
    // The SortWorker lands a back-to-front permutation over the four.
    const sorted = node().geometry as THREE.InstancedBufferGeometry;
    writeSortedIndexOrdering(sorted, new Uint32Array([3, 2, 1, 0]), 4);
    while (pumpSortedIndexOrderingApply(sorted).more) {
      /* one slice per pump; it swaps in on the last */
    }

    // The next ladder rung extends the drawn four to six.
    await adapter.commit(loader, 6, true);

    // Same buffers (append) or a grown geometry (seeded) — either way the draw
    // holds on the sorted four until the six's ordering lands.
    const held = node().geometry as THREE.InstancedBufferGeometry;
    expect(held.instanceCount, `${type} drew the appended suffix unsorted`).toBe(4);
    expect(drawnOrder(held)).toEqual([3, 2, 1, 0]);
  },
  enforced: {
    'no-op': async (type) => {
      const { loader, root } = pooledLoader();
      const adapter = GEOMETRY_COMMITS[type];
      await adapter.commit(loader, 4);
      const geometry = (root.getObjectByName(adapter.path) as THREE.Mesh).geometry;
      // A landed sort permutes the index in place.
      geometry.index!.array.set([9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]);

      await adapter.commit(loader, 6, true);

      // Nothing holds: the whole extension draws at once, in storage order.
      expect(geometry.drawRange.count).toBe(6 * 3);
      expect(Array.from(geometry.index!.array.subarray(0, 18))).toEqual(
        Array.from({ length: 18 }, (_, i) => i)
      );
    },
  },
});
