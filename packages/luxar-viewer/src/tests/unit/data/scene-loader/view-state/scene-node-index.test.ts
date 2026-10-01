/**
 * `SceneNodeIndex` must answer exactly what the root-down walk
 * (`computeWorldNdTransform` + a depth-first path search) answers, without
 * walking: one O(N) build after `buildSceneGraph`, then O(1) per lookup.
 */

import { describe, expect, it } from 'vitest';
import type { SceneNode } from '../../../../../data/data-loader-types';
import { SceneNodeIndex } from '../../../../../data/scene-loader/view-state/scene-node-index';
import { computeWorldNdTransform } from '../../../../../data/transforms/nd-transform';

function node(path: string, ndTransform?: object, children: SceneNode[] = []): SceneNode {
  return {
    path,
    type: children.length > 0 ? 'group' : 'points',
    attrs: (ndTransform ? { nd_transform: ndTransform } : {}) as SceneNode['attrs'],
    hasSpatialIndex: false,
    children,
  };
}

/** Root → groups with stacked affine + permutation transforms → leaves. */
function makeGraph(): SceneNode {
  return node('/', { t: { scale: 2, offset: 1 } }, [
    node('/a', { t: { scale: 0.5, offset: 3 } }, [
      node('/a/leaf'),
      node('/a/b', { c: { permutation: [1, 0, 2] } }, [node('/a/b/leaf', { t: { offset: -4 } })]),
    ]),
    node('/plain', undefined, [node('/plain/leaf')]),
    // A duplicate path: the first occurrence in pre-order wins, as in the walk.
    node('/a/leaf', { t: { scale: 9 } }),
  ]);
}

function allPaths(root: SceneNode): string[] {
  return [root.path, ...(root.children ?? []).flatMap(allPaths)];
}

describe('SceneNodeIndex', () => {
  it('matches computeWorldNdTransform for every path, including duplicates and unknowns', () => {
    const graph = makeGraph();
    const index = new SceneNodeIndex(graph);
    for (const path of [...allPaths(graph), '/missing']) {
      expect(index.worldNdTransform(path)).toEqual(computeWorldNdTransform(graph, path));
      expect(index.hasNdTransform(path)).toBe(
        Object.keys(computeWorldNdTransform(graph, path)).length > 0
      );
    }
    expect(index.node('/a/b/leaf')?.path).toBe('/a/b/leaf');
    expect(index.node('/a/leaf')).toBe(graph.children![0].children![0]);
    expect(index.node('/missing')).toBeNull();
    expect(index.size).toBe(7);
  });

  it('looks paths up without walking the graph', () => {
    let childReads = 0;
    const counted = (n: SceneNode): SceneNode => {
      const kids = (n.children ?? []).map(counted);
      return Object.defineProperty({ ...n }, 'children', {
        get: () => {
          childReads++;
          return kids;
        },
      });
    };
    // A wide partition: the walk's cost per lookup grows with the part count.
    const parts = Array.from({ length: 500 }, (_, i) => node(`/p/part_${i}`));
    const graph = counted(node('/', undefined, [node('/p', { t: { offset: 1 } }, parts)]));
    const index = new SceneNodeIndex(graph);
    const built = childReads;
    for (let i = 0; i < 500; i++) index.worldNdTransform(`/p/part_${i}`);
    expect(childReads).toBe(built);

    computeWorldNdTransform(graph, '/p/part_499');
    expect(childReads).toBeGreaterThan(built);
  });

  it('refuses a node reached twice, like the walk', () => {
    const shared = node('/shared');
    const graph = node('/', undefined, [
      node('/x', undefined, [shared]),
      node('/y', undefined, [shared]),
    ]);
    expect(() => new SceneNodeIndex(graph)).toThrow(/encountered twice/);
  });
});
