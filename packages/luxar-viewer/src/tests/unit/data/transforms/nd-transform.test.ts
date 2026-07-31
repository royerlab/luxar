/**
 * Tests for nD transform utilities (inverse-query approach).
 */
import { describe, it, expect } from 'vitest';
import {
  invertNdTransformForQuery,
  composeNdTransforms,
  computeWorldNdTransform,
} from '../../../../data/transforms/nd-transform';
import type { NdTransformMap } from '../../../../types/zarr';

/**
 * Name-only dimension metadata for `invertNdTransformForQuery`. Omitting
 * `discrete` keeps every dimension continuous, so the no-preimage rule (which
 * only applies to discrete dims) never fires — these cases predate it and must
 * keep their original expectations. Discrete cases pass explicit metadata.
 */
const dims = (names: string[]) => names.map((name) => ({ name }));

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
      dims(['X', 'Y', 'Z', 'Time']),
      [0, 1, 2] // X,Y,Z are displayed
    );
    // local_time = (50 - 10) / 2 = 20
    expect(result.slicePosition[3]).toBeCloseTo(20.0, 5);
    // local_tolerance = 5 / |2| = 2.5
    expect(result.tolerance[3]).toBeCloseTo(2.5, 5);
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
      dims(['X', 'Y', 'Z', 'Time', 'Channel']),
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
      dims(['X', 'Y', 'Z', 'Depth']),
      [0, 1, 2]
    );
    // local = (75 - 100) / (-1) = 25
    expect(result.slicePosition[3]).toBeCloseTo(25.0, 5);
    // tolerance = 10 / |-1| = 10
    expect(result.tolerance[3]).toBeCloseTo(10.0, 5);
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
      dims(['X', 'Y', 'Z', 'Time']),
      [0, 1, 2]
    );
    // X unchanged (displayed)
    expect(result.slicePosition[0]).toBe(100);
    // Time: (50 - 10) / 1 = 40
    expect(result.slicePosition[3]).toBeCloseTo(40.0, 5);
  });

  it('should handle identity transform (no entries)', () => {
    const result = invertNdTransformForQuery(
      [0, 0, 0, 50],
      [1e10, 1e10, 1e10, 5],
      {},
      dims(['X', 'Y', 'Z', 'Time']),
      [0, 1, 2]
    );
    expect(result.slicePosition).toEqual([0, 0, 0, 50]);
    expect(result.tolerance).toEqual([1e10, 1e10, 1e10, 5]);
  });

  // BOUNDARY [P5]: three non-displayed dims set simultaneously — affine,
  // categorical permutation, and a second affine — each inverted
  // independently by the per-dimension loop.
  it('inverts mixed affine + permutation + affine dims independently in one query', () => {
    const ndTransform: NdTransformMap = {
      // affine: effective = 2*raw + 10 → raw = (eff-10)/2
      Time: { scale: 2.0, offset: 10.0 },
      // permutation: perm[0]=2,[1]=0,[2]=1 → inverse maps world 2→local 0
      Channel: { permutation: [2, 0, 1] },
      // affine: effective = -4*raw + 100 → raw = (eff-100)/(-4)
      Depth: { scale: -4.0, offset: 100.0 },
    };
    // Dim order: X, Y, Z displayed; Time, Channel, Depth hidden.
    const result = invertNdTransformForQuery(
      [0, 0, 0, 50, 2, 60], // world: time=50, channel=2, depth=60
      [1e10, 1e10, 1e10, 8, 0.5, 12], // tolerance per dim
      ndTransform,
      dims(['X', 'Y', 'Z', 'Time', 'Channel', 'Depth']),
      [0, 1, 2]
    );

    // Time affine: (50 - 10) / 2 = 20; tol = 8 / |2| = 4
    expect(result.slicePosition[3]).toBeCloseTo(20.0, 5);
    expect(result.tolerance[3]).toBeCloseTo(4.0, 5);

    // Channel permutation: world 2 → local 0; tolerance unchanged
    expect(result.slicePosition[4]).toBe(0);
    expect(result.tolerance[4]).toBe(0.5);

    // Depth affine: (60 - 100) / (-4) = 10; tol = 12 / |-4| = 3
    expect(result.slicePosition[5]).toBeCloseTo(10.0, 5);
    expect(result.tolerance[5]).toBeCloseTo(3.0, 5);

    // Displayed dims untouched.
    expect(result.slicePosition[0]).toBe(0);
    expect(result.tolerance[0]).toBe(1e10);
  });

  it('should skip scale=0 (cannot invert)', () => {
    const ndTransform: NdTransformMap = {
      Time: { scale: 0, offset: 50.0 },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 50],
      [1e10, 1e10, 1e10, 5],
      ndTransform,
      dims(['X', 'Y', 'Z', 'Time']),
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

// ════════════════════════════════════════════════════════════════════════════
// data.md G5 fix [P5]: boundary cases — these are the cases that would let
// production silently corrupt data if the inverse-query implementation
// drifted. They are first-class boundary tests, not coverage filler.
// ════════════════════════════════════════════════════════════════════════════

describe('invertNdTransformForQuery — boundary cases (data.md G5)', () => {
  it('does not modify the world index when it is out of the permutation range (negative)', () => {
    // Source guard: if (worldIndex >= 0 && worldIndex < inversePerm.length).
    // A mutation that dropped the negative-side guard would silently write
    // `undefined` into localSlice.
    const ndTransform: NdTransformMap = {
      Channel: { permutation: [2, 0, 1] }, // length 3 → valid worldIndex in [0,3)
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, -1], // world channel=-1 (out of range)
      [1e10, 1e10, 1e10, 0],
      ndTransform,
      dims(['X', 'Y', 'Z', 'Channel']),
      [0, 1, 2]
    );
    // Out-of-range → slicePosition unchanged from input.
    expect(result.slicePosition[3]).toBe(-1);
    // Sanity: didn't write `undefined`.
    expect(result.slicePosition[3]).toBeDefined();
  });

  it('does not modify the world index when it is out of the permutation range (overflow)', () => {
    const ndTransform: NdTransformMap = {
      Channel: { permutation: [2, 0, 1] },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 99], // world channel=99 (out of range)
      [1e10, 1e10, 1e10, 0],
      ndTransform,
      dims(['X', 'Y', 'Z', 'Channel']),
      [0, 1, 2]
    );
    expect(result.slicePosition[3]).toBe(99);
    expect(result.slicePosition[3]).toBeDefined();
  });

  it('treats a single-element permutation as identity', () => {
    // perm=[0] is the only valid length-1 permutation.
    const ndTransform: NdTransformMap = {
      Channel: { permutation: [0] },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 0],
      [1e10, 1e10, 1e10, 0],
      ndTransform,
      dims(['X', 'Y', 'Z', 'Channel']),
      [0, 1, 2]
    );
    expect(result.slicePosition[3]).toBe(0);
  });

  it('treats an empty permutation as no-op (out of range for any input)', () => {
    // perm=[] is degenerate but the function should not crash; every world
    // index is out of range so the slice value passes through unchanged.
    const ndTransform: NdTransformMap = {
      Channel: { permutation: [] },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 5],
      [1e10, 1e10, 1e10, 0],
      ndTransform,
      dims(['X', 'Y', 'Z', 'Channel']),
      [0, 1, 2]
    );
    expect(result.slicePosition[3]).toBe(5);
  });

  it('skips the dimension when scale === 0 (cannot invert)', () => {
    // Source guard: if (scale === 0) continue;
    const ndTransform: NdTransformMap = {
      Time: { scale: 0, offset: 10 },
    };
    const result = invertNdTransformForQuery(
      [0, 0, 0, 42],
      [1e10, 1e10, 1e10, 5],
      ndTransform,
      dims(['X', 'Y', 'Z', 'Time']),
      [0, 1, 2]
    );
    // Position and tolerance must pass through unchanged (no NaN/Infinity).
    expect(result.slicePosition[3]).toBe(42);
    expect(result.tolerance[3]).toBe(5);
    expect(Number.isFinite(result.slicePosition[3])).toBe(true);
    expect(Number.isFinite(result.tolerance[3])).toBe(true);
  });
});

describe('invertNdTransformForQuery — the no-preimage rule on discrete dims', () => {
  /** X,Y,Z displayed; Frame a discrete ordinal on the integer grid. */
  const frameDims = [
    { name: 'X' },
    { name: 'Y' },
    { name: 'Z' },
    { name: 'Frame', discrete: true, step: 1 },
  ];
  const DISPLAYED = [0, 1, 2];
  /** Discrete dims are queried with an exact (zero) tolerance. */
  const exactTol = [1e10, 1e10, 1e10, 0];

  const invert = (world: number, entry: NdTransformMap['Frame'], d = frameDims, tol = exactTol) =>
    invertNdTransformForQuery([0, 0, 0, world], tol, { Frame: entry }, d, DISPLAYED);

  it('reports no preimage when scale=2 lands the query between two categories', () => {
    // world 7 → local 3.5. The half-step membership gate would otherwise admit
    // BOTH local 3 and local 4 — two categories from other world slices.
    const result = invert(7, { scale: 2 });
    expect(result.slicePosition[3]).toBeCloseTo(3.5, 6);
    expect(result.noPreimage).toBe(true);
  });

  it('reports a preimage when scale=2 lands the query exactly on a category', () => {
    const result = invert(8, { scale: 2 });
    expect(result.slicePosition[3]).toBeCloseTo(4, 6);
    expect(result.noPreimage).toBe(false);
  });

  it('catches non-half off-grid queries too (scale=3 → local 2.333)', () => {
    // A strict/half-open membership boundary would NOT catch this one: local 2
    // is only 0.333 away and would sneak through the half-step window.
    const result = invert(7, { scale: 3 });
    expect(result.noPreimage).toBe(true);
    expect(invert(6, { scale: 3 }).noPreimage).toBe(false);
  });

  it('catches a fractional offset', () => {
    expect(invert(7, { offset: 0.5 }).noPreimage).toBe(true);
    expect(invert(7, { offset: 5 }).noPreimage).toBe(false);
  });

  it('honours the dimension step when it is not 1', () => {
    const stepped = [
      { name: 'X' },
      { name: 'Y' },
      { name: 'Z' },
      { name: 'Frame', discrete: true, step: 0.5 },
    ];
    // scale=2 → local 3.5, which IS on the 0.5 grid.
    expect(invert(7, { scale: 2 }, stepped).noPreimage).toBe(false);
    // local 3.25 is not.
    expect(invert(6.5, { scale: 2 }, stepped).noPreimage).toBe(true);
  });

  it('never fires on a CONTINUOUS dimension (fractional slices are legitimate)', () => {
    // Same name so the transform still applies — only `discrete` differs.
    const continuousDims = [{ name: 'X' }, { name: 'Y' }, { name: 'Z' }, { name: 'Frame' }];
    const result = invert(7, { scale: 2 }, continuousDims, [1e10, 1e10, 1e10, 5]);
    expect(result.slicePosition[3]).toBeCloseTo(3.5, 6);
    expect(result.noPreimage).toBe(false);
  });

  it('never fires on an extend_to_all dimension (it is not being sliced)', () => {
    // The extend_to_all sentinel tolerance means "ignore this axis"; a node that
    // extends the very dim it transforms must keep rendering.
    const extended = [1e10, 1e10, 1e10, 1e10];
    const result = invert(7, { scale: 2 }, frameDims, extended);
    expect(result.noPreimage).toBe(false);
  });

  it('never fires on a categorical permutation (a bijection always has a preimage)', () => {
    const catDims = [
      { name: 'X' },
      { name: 'Y' },
      { name: 'Z' },
      { name: 'Channel', discrete: true, step: 1 },
    ];
    for (const world of [0, 1, 2]) {
      const result = invertNdTransformForQuery(
        [0, 0, 0, world],
        exactTol,
        { Channel: { permutation: [2, 0, 1] } },
        catDims,
        DISPLAYED
      );
      expect(result.noPreimage).toBe(false);
    }
  });

  it('never fires for an identity-scale offset transform (the common case)', () => {
    for (const world of [0, 5, 7, 15]) {
      expect(invert(world, { offset: 5 }).noPreimage).toBe(false);
      expect(invert(world, { offset: -3 }).noPreimage).toBe(false);
      expect(invert(world, { scale: -1, offset: 15 }).noPreimage).toBe(false);
    }
  });

  it('does not fire from float error on an exactly-representable inverse', () => {
    // (world - offset) / scale must not drift off-grid for the values a real
    // dataset uses. 0.1-style scales are the classic float hazard.
    const result = invert(70, { scale: 0.1 });
    expect(result.slicePosition[3]).toBeCloseTo(700, 6);
    expect(result.noPreimage).toBe(false);
  });
});

describe('composeNdTransforms — boundary cases (data.md G5)', () => {
  it('composes three affine transforms in root-first order', () => {
    // root: scale=2, offset=1 → effective_root(x) = 2x + 1
    // mid: scale=3, offset=0 → effective_mid(x) = 3x
    // leaf: scale=1, offset=4 → effective_leaf(x) = x + 4
    //
    // Composition (parent applied outermost):
    //   y = leaf(x) = x + 4
    //   y = mid(y) = 3(x+4) = 3x + 12
    //   y = root(y) = 2(3x+12) + 1 = 6x + 25
    //
    // So composed: scale=6, offset=25.
    const root: NdTransformMap = { Time: { scale: 2, offset: 1 } };
    const mid: NdTransformMap = { Time: { scale: 3, offset: 0 } };
    const leaf: NdTransformMap = { Time: { scale: 1, offset: 4 } };

    const result = composeNdTransforms(root, mid, leaf);
    expect((result.Time as { scale: number; offset: number }).scale).toBeCloseTo(6, 5);
    expect((result.Time as { scale: number; offset: number }).offset).toBeCloseTo(25, 5);
  });

  it('composes three permutations correctly', () => {
    // root: [1, 0, 2] (swap 0/1)
    // mid:  [2, 1, 0] (swap 0/2)
    // leaf: [0, 2, 1] (swap 1/2)
    //
    // For each leaf output index i, walk inward:
    //   i=0: leaf[0]=0 → mid[0]=2 → root[2]=2
    //   i=1: leaf[1]=2 → mid[2]=0 → root[0]=1
    //   i=2: leaf[2]=1 → mid[1]=1 → root[1]=0
    // Expected: [2, 1, 0]
    const root: NdTransformMap = { Ch: { permutation: [1, 0, 2] } };
    const mid: NdTransformMap = { Ch: { permutation: [2, 1, 0] } };
    const leaf: NdTransformMap = { Ch: { permutation: [0, 2, 1] } };

    const result = composeNdTransforms(root, mid, leaf);
    expect((result.Ch as { permutation: number[] }).permutation).toEqual([2, 1, 0]);
  });

  it('identity composes with anything to yield the other (algebraic identity)', () => {
    // P12 / H1 connection: identity composition is a property worth pinning.
    const identity: NdTransformMap = {}; // empty = identity
    const t: NdTransformMap = { Time: { scale: 7, offset: 3 } };

    expect(composeNdTransforms(identity, t)).toEqual(t);
    expect(composeNdTransforms(t, identity)).toEqual(t);
  });

  it('skips mixed affine+permutation under the same dim (no silent corruption)', () => {
    // Source guard: if (!allPerm && !allAffine) continue;
    // Mutation that dropped this guard would produce a nonsensical hybrid.
    const t1: NdTransformMap = { Time: { permutation: [1, 0] } };
    const t2: NdTransformMap = { Time: { scale: 2 } };

    const result = composeNdTransforms(t1, t2);
    // Time is dropped entirely because the entries are type-inconsistent.
    expect(result.Time).toBeUndefined();
  });
});

// ════════════════════════════════════════════════════════════════════════════
// MED-5 (production-bug worklist): `computeWorldNdTransform.findPath`
// mutated a shared `chain` array without guarding against repeated node
// references. A malformed scene graph that contains the same node twice
// (cycle or shared subtree) used to double-push the same nd_transform,
// causing the chain to be composed twice — silent corruption.
// ════════════════════════════════════════════════════════════════════════════

describe('computeWorldNdTransform — cycle / shared-reference safety (MED-5)', () => {
  interface TestNode {
    path: string;
    type: string;
    attrs: Record<string, any>;
    hasSpatialIndex: boolean;
    children: TestNode[];
  }

  it('throws a clear error on a self-referential cycle instead of looping silently', () => {
    // Construct a cycle: root → A → A (self-ref). Without the visited
    // guard this would recurse without termination; with the guard, it
    // throws on the second visit.
    const A: TestNode = {
      path: '/A',
      type: 'group',
      attrs: { nd_transform: { Time: { offset: 7 } } },
      hasSpatialIndex: false,
      children: [],
    };
    A.children.push(A); // self-cycle
    const root: TestNode = {
      path: '/',
      type: 'group',
      attrs: {},
      hasSpatialIndex: false,
      children: [A],
    };
    expect(() => computeWorldNdTransform(root, '/NonExistent')).toThrow(/malformed scene graph/i);
  });

  it('throws when a node is reachable via two distinct paths (shared reference)', () => {
    // Same node referenced as a child of two different parents — would
    // double-compose its nd_transform without the visited set.
    const shared: TestNode = {
      path: '/Shared',
      type: 'group',
      attrs: { nd_transform: { Time: { scale: 3 } } },
      hasSpatialIndex: false,
      children: [],
    };
    const root: TestNode = {
      path: '/',
      type: 'group',
      attrs: {},
      hasSpatialIndex: false,
      children: [shared, shared], // listed twice under the same parent
    };
    expect(() => computeWorldNdTransform(root, '/Target')).toThrow(/malformed scene graph/i);
  });
});
