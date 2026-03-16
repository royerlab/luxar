/**
 * Tests for nD transform utilities (inverse-query approach).
 */
import { describe, it, expect } from 'vitest';
import {
  invertNdTransformForQuery,
  composeNdTransforms,
  computeWorldNdTransform,
} from '../../../data/nd-transform';
import type { NdTransformMap } from '../../../types/zarr';

describe('invertNdTransformForQuery', () => {
  it('should inverse affine transform on slice position', () => {
    // nd_transform: effective = 2 * raw + 10
    // inverse: raw = (effective - 10) / 2
    const ndTransform: NdTransformMap = {
      Time: { scale: 2.0, offset: 10.0 },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 50], // world slice: time=50
      [1e10, 1e10, 1e10, 5], // tolerance: time=5
      ndTransform,
      ['X', 'Y', 'Z', 'Time'],
      [0, 1, 2] // X,Y,Z are displayed
    );
    // local_time = (50 - 10) / 2 = 20
    expect(result.slicePosition[3]).toBeCloseTo(20.0);
    // local_tolerance = 5 / |2| = 2.5
    expect(result.tolerance[3]).toBeCloseTo(2.5);
    // Displayed dims unchanged
    expect(result.slicePosition[0]).toBe(0);
    expect(result.tolerance[0]).toBe(1e10);
  });

  it('should inverse permutation on slice position', () => {
    // perm = [2, 0, 1] means: local 0→world 2, local 1→world 0, local 2→world 1
    // inverse: world 0→local 1, world 1→local 2, world 2→local 0
    const ndTransform: NdTransformMap = {
      Channel: { permutation: [2, 0, 1] },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 0, 2], // world: channel=2
      [1e10, 1e10, 1e10, 5, 0.5],
      ndTransform,
      ['X', 'Y', 'Z', 'Time', 'Channel'],
      [0, 1, 2]
    );
    // world 2 → local 0 (since perm[0]=2, inverse[2]=0)
    expect(result.slicePosition[4]).toBe(0);
    // Tolerance unchanged for permutation
    expect(result.tolerance[4]).toBe(0.5);
  });

  it('should handle negative scale correctly', () => {
    const ndTransform: NdTransformMap = {
      Depth: { scale: -1.0, offset: 100.0 },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 75], // world depth=75
      [1e10, 1e10, 1e10, 10],
      ndTransform,
      ['X', 'Y', 'Z', 'Depth'],
      [0, 1, 2]
    );
    // local = (75 - 100) / (-1) = 25
    expect(result.slicePosition[3]).toBeCloseTo(25.0);
    // tolerance = 10 / |-1| = 10
    expect(result.tolerance[3]).toBeCloseTo(10.0);
  });

  it('should skip displayed dimensions', () => {
    const ndTransform: NdTransformMap = {
      X: { scale: 2.0, offset: 5.0 }, // This should be ignored (X is displayed)
      Time: { offset: 10.0 },
    };
    const result = invertNdTransformForQuery(
      [100, 0, 0, 50],
      [1e10, 1e10, 1e10, 5],
      ndTransform,
      ['X', 'Y', 'Z', 'Time'],
      [0, 1, 2]
    );
    // X unchanged (displayed)
    expect(result.slicePosition[0]).toBe(100);
    // Time: (50 - 10) / 1 = 40
    expect(result.slicePosition[3]).toBeCloseTo(40.0);
  });

  it('should handle identity transform (no entries)', () => {
    const result = invertNdTransformForQuery(
      [0, 0, 0, 50],
      [1e10, 1e10, 1e10, 5],
      {},
      ['X', 'Y', 'Z', 'Time'],
      [0, 1, 2]
    );
    expect(result.slicePosition).toEqual([0, 0, 0, 50]);
    expect(result.tolerance).toEqual([1e10, 1e10, 1e10, 5]);
  });

  it('should skip scale=0 (cannot invert)', () => {
    const ndTransform: NdTransformMap = {
      Time: { scale: 0, offset: 50.0 },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 50],
      [1e10, 1e10, 1e10, 5],
      ndTransform,
      ['X', 'Y', 'Z', 'Time'],
      [0, 1, 2]
    );
    // Unchanged — scale=0 is not invertible
    expect(result.slicePosition[3]).toBe(50);
    expect(result.tolerance[3]).toBe(5);
  });
});

describe('composeNdTransforms', () => {
  it('should compose affine transforms correctly', () => {
    // parent: y = 2*x + 100
    // child: y = 0.5*x + 10
    // composed: y = 2*(0.5*x + 10) + 100 = x + 120
    const parent: NdTransformMap = { Time: { scale: 2.0, offset: 100.0 } };
    const child: NdTransformMap = { Time: { scale: 0.5, offset: 10.0 } };
    const result = composeNdTransforms(parent, child);
    // scale = 2*0.5 = 1.0 (identity, elided)
    // offset = 2*10 + 100 = 120
    expect(result.Time).toBeDefined();
    expect((result.Time as { offset?: number }).offset).toBe(120);
  });

  it('should compose permutations correctly', () => {
    // parent: [2, 0, 1] → child: [1, 2, 0]
    // composed[i] = parent[child[i]]
    // i=0: child[0]=1, parent[1]=0 → 0
    // i=1: child[1]=2, parent[2]=1 → 1
    // i=2: child[2]=0, parent[0]=2 → 2
    const parent: NdTransformMap = { Ch: { permutation: [2, 0, 1] } };
    const child: NdTransformMap = { Ch: { permutation: [1, 2, 0] } };
    const result = composeNdTransforms(parent, child);
    expect((result.Ch as { permutation: number[] }).permutation).toEqual([0, 1, 2]);
  });

  it('should handle empty composition', () => {
    expect(composeNdTransforms()).toEqual({});
  });

  it('should handle different dimensions', () => {
    const t1: NdTransformMap = { Time: { offset: 10 } };
    const t2: NdTransformMap = { Channel: { permutation: [1, 0] } };
    const result = composeNdTransforms(t1, t2);
    expect(result.Time).toBeDefined();
    expect(result.Channel).toBeDefined();
  });
});

describe('computeWorldNdTransform', () => {
  // Helper to create a SceneNode
  interface TestNode {
    path: string;
    type: string;
    attrs: Record<string, any>;
    hasSpatialIndex: boolean;
    children: TestNode[];
  }

  function makeNode(path: string, ndTransform?: NdTransformMap, children?: TestNode[]): TestNode {
    return {
      path,
      type: 'group' as string,
      attrs: ndTransform ? { nd_transform: ndTransform } : {},
      hasSpatialIndex: false,
      children: children ?? [],
    };
  }

  it('should return empty for node without nd_transform', () => {
    const root = makeNode('/', undefined, [makeNode('/A')]);
    expect(computeWorldNdTransform(root, '/A')).toEqual({});
  });

  it('should return node own nd_transform', () => {
    const root = makeNode('/', undefined, [makeNode('/A', { Time: { offset: 10 } })]);
    const result = computeWorldNdTransform(root, '/A');
    expect(result).toEqual({ Time: { offset: 10 } });
  });

  it('should compose parent + child nd_transforms', () => {
    const root = makeNode('/', undefined, [
      makeNode('/G', { Time: { offset: 100 } }, [makeNode('/G/P', { Time: { scale: 2 } })]),
    ]);
    // G: offset=100, P: scale=2
    // composed: parent(child(x)) = (2*x) + 100 → scale=2, offset=100
    const result = computeWorldNdTransform(root, '/G/P');
    expect((result.Time as { scale?: number }).scale).toBe(2);
    expect((result.Time as { offset?: number }).offset).toBe(100);
  });

  it('should inherit parent nd_transform when child has none', () => {
    const root = makeNode('/', undefined, [
      makeNode('/G', { Time: { offset: 50 } }, [
        makeNode('/G/P'), // no nd_transform
      ]),
    ]);
    const result = computeWorldNdTransform(root, '/G/P');
    expect(result).toEqual({ Time: { offset: 50 } });
  });

  it('should include root nd_transform for descendants', () => {
    const root = makeNode('/', { Time: { scale: 3 } }, [
      makeNode('/A', undefined, [makeNode('/A/B')]),
    ]);
    const result = computeWorldNdTransform(root, '/A/B');
    expect(result).toEqual({ Time: { scale: 3 } });
  });

  it('should NOT include sibling transforms (backtracking)', () => {
    const root = makeNode('/', undefined, [
      makeNode('/Sibling', { Time: { offset: 999 } }),
      makeNode('/Target'),
    ]);
    // Sibling's transform should be backtracked, not included
    const result = computeWorldNdTransform(root, '/Target');
    expect(result).toEqual({});
  });

  it('should return empty for non-existent path', () => {
    const root = makeNode('/', { Time: { offset: 10 } }, [makeNode('/A')]);
    const result = computeWorldNdTransform(root, '/NonExistent');
    expect(result).toEqual({});
  });

  it('should handle root as target', () => {
    const root = makeNode('/', { Time: { offset: 10 } });
    const result = computeWorldNdTransform(root, '/');
    expect(result).toEqual({ Time: { offset: 10 } });
  });
});
