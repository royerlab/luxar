/**
 * Geometry-behaviour row `depthSortRegistration`: every type's commit hands
 * its node's element centers to the host's depth-sort coordinator.
 *
 * A triangle's center is its vertex centroid, a segment's its midpoint, a
 * point's or splat's its position — three floats per element either way, so
 * the probe asserts the same contract for all four: the coordinator hears
 * about the commit with the committed count, and the centers it is handed
 * (eagerly or as a lazy provider) cover exactly that many elements. The row is
 * also pinned to `GEOMETRY_CAPABILITIES.depthSortable` by the matrix
 * meta-check, since `layer-apply.ts` reads that flag to decide whether a
 * blending-mode switch must start or stop the sort.
 */
import { expect, vi } from 'vitest';

import { SceneLoader } from '../../../data/scene-loader';
import type { DepthSortCoordinator } from '../../../rendering/depth-sort-coordinator';
import { GEOMETRY_COMMITS, makeCommitScene } from '../../helpers/geometry-commits';
import { defineBehaviourConformance } from '../../_conformance/define-behaviour-conformance';

defineBehaviourConformance('depthSortRegistration', {
  async holds(type) {
    const loader = new SceneLoader({ enableMonitor: false });
    const noteCommit = vi.fn();
    loader.setDepthSortCoordinator({ noteCommit } as unknown as DepthSortCoordinator);
    const root = makeCommitScene(loader);
    const adapter = GEOMETRY_COMMITS[type];

    await adapter.commit(loader, 3);

    expect(noteCommit).toHaveBeenCalledTimes(1);
    const [node, centers, count] = noteCommit.mock.calls[0] as [
      unknown,
      Float32Array | (() => Float32Array),
      number,
    ];
    expect(node).toBe(root.getObjectByName(adapter.path));
    expect(count).toBe(3);
    const resolved = typeof centers === 'function' ? centers() : centers;
    expect(resolved).toBeInstanceOf(Float32Array);
    expect(resolved.length).toBe(3 * 3);
  },
});
