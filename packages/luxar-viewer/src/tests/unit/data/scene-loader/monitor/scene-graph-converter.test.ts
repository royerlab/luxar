/**
 * Unit tests for the SceneNode → SceneGraphNode converter.
 */

import { describe, it, expect } from 'vitest';
import { convertToSceneGraphNode } from '../../../../../data/scene-loader/monitor/scene-graph-converter';
import type { SceneNode } from '../../../../../data/data-loader-types';

function makeNode(overrides: Partial<SceneNode> = {}): SceneNode {
  return {
    path: '/test',
    type: 'group',
    attrs: {},
    hasSpatialIndex: false,
    ...overrides,
  } as SceneNode;
}

describe('convertToSceneGraphNode — display name', () => {
  it('renames "/" to "Scene"', () => {
    expect(convertToSceneGraphNode(makeNode({ path: '/' })).name).toBe('Scene');
  });

  it('takes the last path segment as the display name', () => {
    expect(convertToSceneGraphNode(makeNode({ path: '/a/b/c' })).name).toBe('c');
  });

  it('handles a single-segment path', () => {
    expect(convertToSceneGraphNode(makeNode({ path: '/foo' })).name).toBe('foo');
  });

  it('falls back to the raw path when there are no segments after filtering', () => {
    // Path is empty string — split/filter leaves [], pop returns undefined,
    // so the fallback `|| node.path` gives the empty string back.
    expect(convertToSceneGraphNode(makeNode({ path: '' })).name).toBe('');
  });

  it('strips trailing slashes from the segment list', () => {
    expect(convertToSceneGraphNode(makeNode({ path: '/a/b/c/' })).name).toBe('c');
  });
});

describe('convertToSceneGraphNode — type coercion', () => {
  it('preserves valid types', () => {
    expect(convertToSceneGraphNode(makeNode({ type: 'points' })).type).toBe('points');
    expect(convertToSceneGraphNode(makeNode({ type: 'lines' })).type).toBe('lines');
    expect(convertToSceneGraphNode(makeNode({ type: 'gsplats' })).type).toBe('gsplats');
    expect(convertToSceneGraphNode(makeNode({ type: 'group' })).type).toBe('group');
  });

  it('treats "scene" as scene', () => {
    expect(convertToSceneGraphNode(makeNode({ type: 'scene' })).type).toBe('scene');
  });

  it('falls back to "scene" for empty type', () => {
    expect(convertToSceneGraphNode(makeNode({ type: '' as 'group' })).type).toBe('scene');
  });

  // previously a bare `as` cast let unknown runtime
  // types through into the `SceneGraphNode['type']` union and lied
  // to TypeScript about it. Whitelist guarantees the union is honest.
  it('falls back to "scene" for an unknown non-empty type', () => {
    expect(convertToSceneGraphNode(makeNode({ type: 'volume' as 'group' })).type).toBe('scene');
  });

  it('preserves "mesh" as a valid type', () => {
    expect(convertToSceneGraphNode(makeNode({ type: 'mesh' as 'group' })).type).toBe('mesh');
  });
});

describe('convertToSceneGraphNode — type-specific stats', () => {
  it('points node carries pointCount from n_points', () => {
    const node = makeNode({
      type: 'points',
      attrs: { n_points: 42 },
    });
    const graph = convertToSceneGraphNode(node);
    expect(graph.pointCount).toBe(42);
  });

  it('lines node carries segmentCount + vertexCount', () => {
    const node = makeNode({
      type: 'lines',
      attrs: { n_segments: 99, n_vertices: 100 },
    });
    const graph = convertToSceneGraphNode(node);
    expect(graph.segmentCount).toBe(99);
    expect(graph.vertexCount).toBe(100);
  });

  it('gsplats node carries splatCount from n_splats', () => {
    const node = makeNode({
      type: 'gsplats',
      attrs: { n_splats: 5000 },
    });
    const graph = convertToSceneGraphNode(node);
    expect(graph.splatCount).toBe(5000);
  });

  it('group node has no extra stats', () => {
    const graph = convertToSceneGraphNode(makeNode({ type: 'group' }));
    expect(graph.pointCount).toBeUndefined();
    expect(graph.segmentCount).toBeUndefined();
    expect(graph.splatCount).toBeUndefined();
  });

  it('passes hasSpatialIndex through unchanged', () => {
    const a = convertToSceneGraphNode(makeNode({ hasSpatialIndex: true }));
    const b = convertToSceneGraphNode(makeNode({ hasSpatialIndex: false }));
    expect(a.hasSpatialIndex).toBe(true);
    expect(b.hasSpatialIndex).toBe(false);
  });
});

describe('convertToSceneGraphNode — specialized groups (kind=lod / partition)', () => {
  it('carries kind + lodGroupChildCount for a kind=lod group', () => {
    const node = makeNode({
      path: '/lod',
      type: 'group',
      attrs: { kind: 'lod', display_type: 'gsplats' },
      children: [makeNode({ path: '/lod/a' }), makeNode({ path: '/lod/b' })],
    });
    const graph = convertToSceneGraphNode(node);
    expect(graph.kind).toBe('lod');
    expect(graph.displayType).toBe('gsplats');
    expect(graph.lodGroupChildCount).toBe(2);
    expect(graph.partCount).toBeUndefined();
    // type stays 'group' so the stats aggregator treats it as a container.
    expect(graph.type).toBe('group');
  });

  it('carries kind + partCount for a kind=partition group', () => {
    const node = makeNode({
      path: '/part',
      type: 'group',
      attrs: { kind: 'partition', display_type: 'points' },
      children: [
        makeNode({ path: '/part/0' }),
        makeNode({ path: '/part/1' }),
        makeNode({ path: '/part/2' }),
      ],
    });
    const graph = convertToSceneGraphNode(node);
    expect(graph.kind).toBe('partition');
    expect(graph.displayType).toBe('points');
    expect(graph.partCount).toBe(3);
    expect(graph.lodGroupChildCount).toBeUndefined();
  });

  it('ignores an unknown kind value', () => {
    const graph = convertToSceneGraphNode(
      makeNode({ type: 'group', attrs: { kind: 'something-else' } })
    );
    expect(graph.kind).toBeUndefined();
  });

  it('does not treat a leaf node with a stray kind attr as specialized', () => {
    // kind only resolves on group nodes (mirrors layer-state.ts).
    const graph = convertToSceneGraphNode(
      makeNode({ type: 'points', attrs: { kind: 'lod', n_points: 5 } })
    );
    expect(graph.kind).toBeUndefined();
    expect(graph.pointCount).toBe(5);
  });

  it('records additiveSublods for an additive-LOD leaf', () => {
    const graph = convertToSceneGraphNode(
      makeNode({ type: 'gsplats', attrs: { n_splats: 100, n_additive_sublods: 4 } })
    );
    expect(graph.additiveSublods).toBe(4);
  });

  it('omits additiveSublods when n_additive_sublods <= 1', () => {
    const graph = convertToSceneGraphNode(
      makeNode({ type: 'gsplats', attrs: { n_splats: 100, n_additive_sublods: 1 } })
    );
    expect(graph.additiveSublods).toBeUndefined();
  });
});

describe('convertToSceneGraphNode — recursion', () => {
  it('converts an empty children list to an empty children array', () => {
    expect(convertToSceneGraphNode(makeNode({ children: [] })).children).toEqual([]);
  });

  it('passes through an undefined children as an empty array', () => {
    expect(convertToSceneGraphNode(makeNode({ children: undefined })).children).toEqual([]);
  });

  it('recurses into children', () => {
    const node = makeNode({
      path: '/',
      type: 'scene',
      children: [
        makeNode({ path: '/a', type: 'points', attrs: { n_points: 10 } }),
        makeNode({ path: '/b', type: 'group' }),
      ],
    });
    const graph = convertToSceneGraphNode(node);
    expect(graph.children).toHaveLength(2);
    expect(graph.children[0].name).toBe('a');
    expect(graph.children[0].pointCount).toBe(10);
    expect(graph.children[1].name).toBe('b');
  });

  it('recurses to arbitrary depth', () => {
    const node = makeNode({
      path: '/',
      type: 'scene',
      children: [
        makeNode({
          path: '/grp',
          type: 'group',
          children: [makeNode({ path: '/grp/inner', type: 'gsplats', attrs: { n_splats: 7 } })],
        }),
      ],
    });
    const graph = convertToSceneGraphNode(node);
    const grandchild = graph.children[0].children[0];
    expect(grandchild.name).toBe('inner');
    expect(grandchild.type).toBe('gsplats');
    expect(grandchild.splatCount).toBe(7);
  });
});
