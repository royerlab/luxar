import { describe, expect, it, vi } from 'vitest';

import type { SceneGraphNode } from '../../../../types/data-monitor-types';
import { SceneGraphModel } from '../../../../ui/data-loading-monitor/scene-graph-model';

function leaf(
  path: string,
  type: 'points' | 'lines' | 'gsplats' | 'mesh',
  count: number
): SceneGraphNode {
  const countField = {
    points: 'pointCount',
    lines: 'segmentCount',
    gsplats: 'splatCount',
    mesh: 'faceCount',
  }[type];
  return { path, name: path.slice(1), type, children: [], [countField]: count };
}

describe('SceneGraphModel', () => {
  it('owns scene state while preserving root aliasing and substitutive LOD totals', () => {
    const model = new SceneGraphModel(vi.fn());
    const root: SceneGraphNode = {
      path: '/',
      name: 'Scene',
      type: 'scene',
      children: [
        leaf('/points', 'points', 10),
        leaf('/mesh', 'mesh', 12),
        {
          path: '/lod',
          name: 'lod',
          type: 'group',
          kind: 'lod',
          children: [leaf('/lod/coarse', 'gsplats', 20), leaf('/lod/fine', 'gsplats', 80)],
        },
      ],
    };

    model.setSceneGraph(root);
    model.updateDroppedElementCount(9);
    model.setSceneGraph(root);

    const state = model.getSceneGraph();
    expect(state.root).toBe(root);
    expect(state.totalNodes).toBe(6);
    expect(state.nodesByType).toEqual({ points: 1, lines: 0, gsplats: 2, mesh: 1 });
    expect(state.totalByType).toEqual({ points: 10, lines: 0, gsplats: 80, mesh: 12 });
    expect(state.visibleByType).toEqual(state.totalByType);
    expect(state.droppedElements).toBe(0);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -1, 1.5])(
    'rejects invalid element count %s',
    (count) => {
      const model = new SceneGraphModel(vi.fn());
      model.setSceneGraph(leaf('/points', 'points', count));
      expect(model.getSceneGraph().totalByType.points).toBe(0);
    }
  );

  it('merges per-path visible counts into all four geometry types', () => {
    // The visible-counts walk already stamps a mesh node's committed
    // `visibleTriangleCount` into the per-path map; mesh used to be skipped
    // here, dropping a number that had already been measured and leaving the
    // tree's mesh badge unable to say how much of the surface the slab indexes.
    const model = new SceneGraphModel(vi.fn());
    const root: SceneGraphNode = {
      path: '/',
      name: 'Scene',
      type: 'scene',
      children: [
        leaf('/points', 'points', 10),
        leaf('/lines', 'lines', 20),
        leaf('/splats', 'gsplats', 30),
        leaf('/mesh', 'mesh', 40),
      ],
    };
    model.setSceneGraph(root);
    model.updateVisibleCountsByPath(
      new Map([
        ['/points', 1],
        ['/lines', 2],
        ['/splats', 3],
        ['/mesh', 4],
      ])
    );
    model.syncVisibleCountsIntoTree();

    expect(model.getSceneGraphNodeByPath('/points')?.visiblePointCount).toBe(1);
    expect(model.getSceneGraphNodeByPath('/lines')?.visibleSegmentCount).toBe(2);
    expect(model.getSceneGraphNodeByPath('/splats')?.visibleSplatCount).toBe(3);
    expect(model.getSceneGraphNodeByPath('/mesh')?.visibleFaceCount).toBe(4);
  });

  it('clears a stale mesh visible count when its path leaves the walk', () => {
    // Paths absent from the latest walk reset to `undefined` — the tooltip must
    // not keep asserting a slice-dependent count that no longer holds.
    const model = new SceneGraphModel(vi.fn());
    model.setSceneGraph(leaf('/mesh', 'mesh', 40));
    model.updateVisibleCountsByPath(new Map([['/mesh', 4]]));
    model.syncVisibleCountsIntoTree();
    expect(model.getSceneGraphNodeByPath('/mesh')?.visibleFaceCount).toBe(4);

    model.updateVisibleCountsByPath(new Map());
    model.syncVisibleCountsIntoTree();
    expect(model.getSceneGraphNodeByPath('/mesh')?.visibleFaceCount).toBeUndefined();
  });

  it('rebuilds the path index when the root reference changes', () => {
    const model = new SceneGraphModel(vi.fn());
    const first = leaf('/points', 'points', 10);
    model.setSceneGraph({ path: '/', name: 'first', type: 'scene', children: [first] });
    model.updateVisibleCountsByPath(new Map([['/points', 7]]));
    model.syncVisibleCountsIntoTree();
    expect(first.visiblePointCount).toBe(7);

    const second = leaf('/points', 'points', 20);
    model.setSceneGraph({ path: '/', name: 'second', type: 'scene', children: [second] });
    model.updateVisibleCountsByPath(new Map([['/points', 3]]));
    model.syncVisibleCountsIntoTree();

    expect(second.visiblePointCount).toBe(3);
    expect(first.visiblePointCount).toBe(7);
    expect(model.getSceneGraphNodeByPath('/points')).toBe(second);

    model.updateVisibleCountsByPath(new Map());
    model.syncVisibleCountsIntoTree();
    expect(second.visiblePointCount).toBeUndefined();
  });

  it('resets with a fresh expanded set and supports explicit disposal clearing', () => {
    const markStructureDirty = vi.fn();
    const model = new SceneGraphModel(markStructureDirty);
    const initialExpanded = model.expandedNodes;
    model.setSceneGraph({
      path: '/',
      name: 'Scene',
      type: 'scene',
      children: [leaf('/child', 'points', 10)],
    });
    model.getSceneGraphNodeByPath('/child');
    model.toggleNodeExpansion('/child');
    expect(model.expandedNodes.has('/child')).toBe(true);

    model.resetSceneGraphState();

    expect(model.getSceneGraph().root).toBeNull();
    expect(model.expandedNodes).not.toBe(initialExpanded);
    expect(model.expandedNodes).toEqual(new Set(['/']));
    const indexState = model as unknown as {
      sceneGraphNodeIndex: Map<string, SceneGraphNode>;
      sceneGraphNodeIndexRoot: SceneGraphNode | null;
    };
    expect(indexState.sceneGraphNodeIndex.size).toBe(0);
    expect(indexState.sceneGraphNodeIndexRoot).toBeNull();
    model.clearExpandedNodes();
    expect(model.expandedNodes.size).toBe(0);
    expect(markStructureDirty).toHaveBeenCalledTimes(3);
  });
});
