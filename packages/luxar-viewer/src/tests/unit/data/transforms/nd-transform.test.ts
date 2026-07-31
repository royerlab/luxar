/**
 * Tests for nD transform utilities (inverse-query approach).
 */
import { describe, it, expect } from 'vitest';
import {
  invertNdTransformForQuery,
  composeNdTransforms,
  computeWorldNdTransform,
} from '../../../../data/transforms/nd-transform';
import type { QueryDimensionInfo } from '../../../../data/transforms/nd-transform';
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
  const frameDims: QueryDimensionInfo[] = [
    { name: 'X' },
    { name: 'Y' },
    { name: 'Z' },
    { name: 'Frame', discrete: true, step: 1 },
  ];
  const DISPLAYED = [0, 1, 2];
  /** Discrete dims are queried with an exact (zero) tolerance. */
  const exactTol = [1e10, 1e10, 1e10, 0];

  const invert = (
    world: number,
    entry: NdTransformMap['Frame'],
    d: QueryDimensionInfo[] = frameDims,
    tol = exactTol
  ) => invertNdTransformForQuery([0, 0, 0, world], tol, { Frame: entry }, d, DISPLAYED);

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

  it('does NOT blank a fractional offset — round(k + 0.4) = k has a preimage', () => {
    // The rule implements the spec's forward round(), not exact inverse-on-grid
    // alignment. `offset: 0.4` maps every local k to world k, so every world
    // value has a preimage and the node must keep rendering. (Testing
    // inverse-on-grid here would blank it permanently.)
    for (const world of [0, 1, 5, 7, 15]) {
      const r = invert(world, { offset: 0.4 });
      expect(r.noPreimage).toBe(false);
      expect(r.slicePosition[3]).toBeCloseTo(world, 9); // snapped to local k = world
    }
    expect(invert(7, { offset: 5 }).noPreimage).toBe(false);
  });

  it('does NOT blank a fractional scale that still has a preimage', () => {
    // scale 1.2: round(1.2·1) = 1, so world 1 resolves to local 1.
    const r = invert(1, { scale: 1.2 });
    expect(r.noPreimage).toBe(false);
    expect(r.slicePosition[3]).toBeCloseTo(1, 9);
    // world 6 = round(1.2·5); world 2 is the image of no integer k
    // (1.2·1 → 1, 1.2·2 → 2.4 → 2 ✓) so 2 DOES resolve, to local 2.
    expect(invert(6, { scale: 1.2 }).slicePosition[3]).toBeCloseTo(5, 9);
    expect(invert(2, { scale: 1.2 }).noPreimage).toBe(false);
  });

  it('snaps the query onto the resolved grid point (kills midpoint ties)', () => {
    // scale 2 at world 8 resolves to local 4 exactly, so the downstream
    // half-step window brackets only local 4 — no neighbour can tie in.
    const r = invert(8, { scale: 2 });
    expect(r.slicePosition[3]).toBe(4);
    // A lossy downsample (|scale| < 1) picks the nearest representative.
    const half = invert(3, { scale: 0.5 });
    expect(half.noPreimage).toBe(false);
    expect(half.slicePosition[3]).toBeCloseTo(6, 9); // round(0.5·6) = 3
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

  it('reads the extend sentinel BEFORE rescaling it (large scales)', () => {
    // Order-of-operations pin: the inverse divides the tolerance by |scale|, so
    // for scale > 10 the 1e10 sentinel drops BELOW the 1e9 floor. Testing the
    // rescaled value would lose the exemption and wrongly blank an extended
    // node. Only a large scale exposes it — scale 2 stays above the floor
    // either way, so the earlier test cannot catch this regression.
    const extended = [1e10, 1e10, 1e10, 1e10];
    const result = invert(7, { scale: 1e4 }, frameDims, extended);
    // NOTE: the eroded tolerance below is CURRENT behaviour, not desirable
    // behaviour — downstream `>= 1e9` extend checks also lose the sentinel at
    // scale > 10. That erosion is a separate pre-existing issue; this test only
    // pins that the no-preimage rule reads the sentinel before it happens.
    expect(result.tolerance[3]).toBeLessThan(1e9);
    expect(result.noPreimage).toBe(false); // ...the exemption still applied
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
    // ...nor for a scale whose inverse is not exactly representable at all.
    for (const world of [1, 5, 11]) {
      expect(invert(world, { scale: 1 / 3 }).noPreimage).toBe(false); // local = 3·world
    }
  });

  it('tolerates a degenerate step (missing / zero / negative / NaN) by falling back to 1', () => {
    for (const step of [undefined, 0, -1, NaN]) {
      const degenerate = [
        { name: 'X' },
        { name: 'Y' },
        { name: 'Z' },
        { name: 'Frame', discrete: true, step },
      ];
      const onGrid = invert(8, { scale: 2 }, degenerate);
      const offGrid = invert(7, { scale: 2 }, degenerate);
      expect(Number.isFinite(onGrid.slicePosition[3])).toBe(true);
      expect(onGrid.noPreimage).toBe(false);
      expect(offGrid.noPreimage).toBe(true);
    }
  });

  it('handles negative local positions and a negative non-unit scale', () => {
    // offset 5 puts the query below zero for T < 5. Still ON the grid — that
    // the data has nothing there is the loader's business, not the rule's.
    expect(invert(0, { offset: 5 }).slicePosition[3]).toBe(-5);
    expect(invert(0, { offset: 5 }).noPreimage).toBe(false);
    // scale -2: local = T / -2, so odd T lands on a negative half-integer.
    expect(invert(7, { scale: -2 }).noPreimage).toBe(true);
    expect(invert(8, { scale: -2 }).noPreimage).toBe(false);
    expect(invert(8, { scale: -2 }).slicePosition[3]).toBe(-4);
  });

  it('stays exact at large indices (no float-error false positives)', () => {
    for (const world of [1e3, 1e5, 1e7]) {
      expect(invert(world, { scale: 2 }).noPreimage).toBe(false); // even → integer
      expect(invert(world + 1, { scale: 2 }).noPreimage).toBe(true); // odd → .5
    }
  });

  it('fires on a THREE-level composed chain, not just a single entry', () => {
    // root {scale 2} ∘ mid {offset 1} ∘ leaf {scale 3} composes (root-first) to
    // scale 6, offset 2 → local = (T-2)/6. Only T ≡ 2 (mod 6) has a preimage.
    const composed = composeNdTransforms(
      { Frame: { scale: 2 } },
      { Frame: { offset: 1 } },
      { Frame: { scale: 3 } }
    );
    expect(composed.Frame).toEqual({ scale: 6, offset: 2 });
    for (const world of [2, 8, 14]) {
      expect(invert(world, composed.Frame).noPreimage).toBe(false);
    }
    for (const world of [0, 1, 3, 7, 9, 13]) {
      expect(invert(world, composed.Frame).noPreimage).toBe(true);
    }
  });

  it('still fires when a DIFFERENT dimension is the extended one', () => {
    // Frame is transformed and sliced; Channel is the extended one. The
    // exemption is per-dimension, so Frame must not inherit Channel's pass.
    const twoHidden = [
      { name: 'X' },
      { name: 'Y' },
      { name: 'Z' },
      { name: 'Frame', discrete: true, step: 1 },
      { name: 'Channel', discrete: true, step: 1 },
    ];
    const result = invertNdTransformForQuery(
      [0, 0, 0, 7, 1],
      [1e10, 1e10, 1e10, 0, 1e10], // Channel extended, Frame exact
      { Frame: { scale: 2 } },
      twoHidden,
      [0, 1, 2]
    );
    expect(result.noPreimage).toBe(true);
  });

  it('fires if ANY transformed discrete dim is off-grid (not just the first)', () => {
    const twoHidden = [
      { name: 'X' },
      { name: 'Y' },
      { name: 'Z' },
      { name: 'A', discrete: true, step: 1 },
      { name: 'B', discrete: true, step: 1 },
    ];
    // A lands on-grid, B does not.
    const result = invertNdTransformForQuery(
      [0, 0, 0, 8, 7],
      [1e10, 1e10, 1e10, 0, 0],
      { A: { scale: 2 }, B: { scale: 2 } },
      twoHidden,
      [0, 1, 2]
    );
    expect(result.slicePosition[3]).toBe(4);
    expect(result.slicePosition[4]).toBe(3.5);
    expect(result.noPreimage).toBe(true);
  });

  it('ignores a transform for a dimension with no metadata entry', () => {
    // dimensions shorter than slicePosition (a truncated/mismatched scene):
    // the loop bails on that index instead of throwing.
    const short = [{ name: 'X' }, { name: 'Y' }, { name: 'Z' }];
    const result = invertNdTransformForQuery(
      [0, 0, 0, 7],
      [1e10, 1e10, 1e10, 0],
      { Frame: { scale: 2 } },
      short,
      [0, 1, 2]
    );
    expect(result.slicePosition[3]).toBe(7); // untouched
    expect(result.noPreimage).toBe(false);
  });

  it('fails SAFE on a non-finite slice position (renders nothing, not everything)', () => {
    // NaN can reach a query through a malformed animation state. `|NaN -
    // round(NaN)| <= eps` is false, so the rule reports no preimage and the
    // node clears — matching the lines kernel's non-finite policy (#806).
    expect(invert(NaN, { scale: 2 }).noPreimage).toBe(true);
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
