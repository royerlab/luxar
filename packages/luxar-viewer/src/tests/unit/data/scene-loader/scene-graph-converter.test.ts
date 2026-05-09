/**
 * Unit tests for the SceneNode → SceneGraphNode converter.
 */

import { describe, it, expect } from 'vitest';
import { convertToSceneGraphNode } from '../../../../data/scene-loader/scene-graph-converter';
import type { SceneNode } from '../../../../data/data-loader-types';

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
    expect(
      convertToSceneGraphNode(makeNode({ type: 'volume' as 'group' })).type
    ).toBe('scene');
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
          children: [
            makeNode({ path: '/grp/inner', type: 'gsplats', attrs: { n_splats: 7 } }),
          ],
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
