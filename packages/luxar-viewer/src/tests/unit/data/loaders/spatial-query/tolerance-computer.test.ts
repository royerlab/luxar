import { describe, it, expect } from 'vitest';
import { computeTolerance, type DimensionInfo } from '../../../../../data/loaders';

const DISPLAYED = 1e10;

describe('computeTolerance — common rules', () => {
  it('sets infinite tolerance for displayed dimensions across all geometry types', () => {
    for (const type of ['points', 'lines', 'gsplats', 'mesh'] as const) {
      const tol = computeTolerance(type, [0, 1, 2], 4);
      expect(tol[0]).toBe(DISPLAYED);
      expect(tol[1]).toBe(DISPLAYED);
      expect(tol[2]).toBe(DISPLAYED);
      // Pin the actual sentinel constant (1e10), not just the local alias, so a
      // mutated DISPLAYED_TOLERANCE in the source is caught.
      expect(tol[0]).toBe(1e10);
      // Dimension 3 is hidden here: its tolerance must be far below the
      // displayed sentinel (kills a swap of displayed/hidden branches).
      expect(tol[3]).toBeLessThan(1e9);
    }
  });

  it('returns an array of the requested length', () => {
    expect(computeTolerance('points', [0, 1, 2], 5)).toHaveLength(5);
    expect(computeTolerance('lines', [], 7)).toHaveLength(7);
    expect(computeTolerance('gsplats', [0], 3)).toHaveLength(3);
  });
});

describe('computeTolerance — points', () => {
  it('uses maxRadius for hidden spatial dimensions', () => {
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, { maxRadius: 5.0 });
    expect(tol[3]).toBe(5.0);
  });

  it('falls back to default maxRadius (1.0) when not supplied', () => {
    const tol = computeTolerance('points', [0, 1, 2], 4);
    expect(tol[3]).toBe(1.0);
  });

  it('uses the shared quarter-cell tolerance for hidden dims flagged non-spatial via spatialExtendDims', () => {
    // Dimension 3 is non-spatial (discrete-like) per the per-dimension flag array.
    // No dimension metadata is supplied, so the step falls back to 1 → 0.25.
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true, false],
    });
    expect(tol[3]).toBe(0.25);
  });

  it('uses maxRadius for hidden dimensions flagged spatial via spatialExtendDims', () => {
    // Same shape, but dimension 3 is flagged spatial (true): it must use
    // maxRadius, not the discrete quarter-cell value. This pins the true branch of
    // the spatialExtendDims flag so a flipped flag-check is caught.
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true, true],
    });
    expect(tol[3]).toBe(5.0);
    expect(tol[3]).not.toBe(0.25);
  });

  it('treats dimensions beyond spatialExtendDims length as spatial', () => {
    const tol = computeTolerance('points', [0, 1, 2], 5, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true],
    });
    expect(tol[3]).toBe(5.0);
    expect(tol[4]).toBe(5.0);
  });
});

describe('computeTolerance — lines', () => {
  const dims: DimensionInfo[] = [
    { discrete: false },
    { discrete: false },
    { discrete: false },
    { discrete: true, step: 1.0 },
  ];

  it('returns 0 for hidden spatial dimensions (segment bounds already include line width)', () => {
    const tol = computeTolerance('lines', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.25); // discrete, 0.25 × step (step 1)
    const spatialOnly: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false },
    ];
    const tol2 = computeTolerance('lines', [0, 1, 2], 4, spatialOnly);
    expect(tol2[3]).toBe(0);
  });

  it('uses 0.25 × step for discrete hidden dimensions', () => {
    const tol = computeTolerance('lines', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.25);
  });

  it('scales the discrete tolerance with a larger step (step 2 → 0.5)', () => {
    const dimsStep2: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 2.0 },
    ];
    const tol = computeTolerance('lines', [0, 1, 2], 4, dimsStep2);
    expect(tol[3]).toBe(0.5); // 0.25 × 2
  });

  it('falls back to a quarter-cell (0.25) for discrete hidden dimensions without a step', () => {
    const dimsNoStep: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true },
    ];
    const tol = computeTolerance('lines', [0, 1, 2], 4, dimsNoStep);
    expect(tol[3]).toBe(0.25);
  });

  // Regression (deep-double-check): the lines PROJECTION/CLIPPING path is a
  // MEMBERSHIP gate, not a fetch reach. When the quarter-cell query rule
  // landed (ee0971f3) it silently halved the lines visibility slab from
  // step/2 to step/4 — cross-category whiskers shrank and off-grid vertices
  // in the (0.25, 0.5]×step band vanished while identical points/gsplats
  // stayed visible. `discreteRole: 'membership'` restores the half-cell gate.
  describe('membership role (projection clipping slab)', () => {
    it('uses the half-cell (0.5 × step) for discrete hidden dims under discreteRole=membership', () => {
      const tol = computeTolerance('lines', [0, 1, 2], 4, dims, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(0.5); // 0.5 × step (step 1) — NOT the 0.25 query reach
    });

    it('scales the membership slab with the step (step 2 → 1.0)', () => {
      const dimsStep2: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true, step: 2.0 },
      ];
      const tol = computeTolerance('lines', [0, 1, 2], 4, dimsStep2, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(1.0);
    });

    it('keeps spatial dims at 0 and displayed dims infinite under the membership role', () => {
      const tol = computeTolerance('lines', [0, 1, 2], 4, dims, {
        discreteRole: 'membership',
      });
      expect(tol[0]).toBe(1e10);
      expect(tol[3]).toBe(0.5);
      const spatialOnly: DimensionInfo[] = [{ discrete: false }, { discrete: false }];
      const tol2 = computeTolerance('lines', [0], 2, spatialOnly, {
        discreteRole: 'membership',
      });
      expect(tol2[1]).toBe(0);
    });

    it('falls back to a half-cell of a unit step (0.5) when the discrete dim has no step', () => {
      const dimsNoStep: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true },
      ];
      const tol = computeTolerance('lines', [0, 1, 2], 4, dimsNoStep, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(0.5);
    });

    it('three-geometry parity: lines membership slab equals the half-step gate points/gsplats use', () => {
      // Points membership: absolute 0.5 on the unit grid
      // (effective-radius-calculator.ts); gsplats projection: step × 0.5
      // (workers/data-worker/projection). Lines must match at step 1.
      const tol = computeTolerance('lines', [0, 1, 2], 4, dims, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(0.5);
    });
  });
});

describe('computeTolerance — gsplats', () => {
  it('uses step × defaultTolerance for hidden continuous dimensions', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 0.5 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.5 * 3.0); // default 3 sigma
  });

  it('honors gsplatsDefaultTolerance override', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 0.25 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims, { gsplatsDefaultTolerance: 5.0 });
    expect(tol[3]).toBe(0.25 * 5.0);
  });

  it('uses 0.25 × step for hidden discrete dimensions', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 1.0 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.25);
  });

  it('scales the discrete tolerance with a larger step (step 2 → 0.5)', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 2.0 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.5); // 0.25 × 2
  });

  it('falls back to defaultTolerance when no step is available', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(3.0);
  });
});

describe('computeTolerance — mesh', () => {
  const HIDDEN_DISCRETE: DimensionInfo[] = [
    { discrete: false },
    { discrete: false },
    { discrete: false },
    { discrete: true, step: 1.0 },
  ];

  it('uses the HALF-cell membership rule for discrete dims, not the quarter-cell query reach', () => {
    // Mesh is whole-node resident, so this slab is a per-element visibility gate
    // applied after fetch — like the lines projection-clipping slab — not a
    // chunk-fetch reach. The quarter-cell default is deliberately < 0.5 x step and
    // would drop on-grid geometry.
    const tol = computeTolerance('mesh', [0, 1, 2], 4, HIDDEN_DISCRETE);
    expect(tol[3]).toBe(0.5);
    // Explicitly NOT the value the other three get by default.
    expect(tol[3]).not.toBe(computeTolerance('gsplats', [0, 1, 2], 4, HIDDEN_DISCRETE)[3]);
  });

  it('scales the half-cell with the step', () => {
    const dims = [...HIDDEN_DISCRETE.slice(0, 3), { discrete: true, step: 4.0 }];
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims)[3]).toBe(2.0);
  });

  it('ignores discreteRole — mesh has no query role to serve', () => {
    // The other three switch behaviour on this option. Mesh has no spatial index and
    // issues no range query, so membership is the only rule it has; honouring a
    // 'query' role here would quietly hand a fetch reach to the one caller that is
    // asking about visibility.
    for (const role of ['query', 'membership'] as const) {
      expect(
        computeTolerance('mesh', [0, 1, 2], 4, HIDDEN_DISCRETE, { discreteRole: role })[3]
      ).toBe(0.5);
    }
  });

  it('gives a hidden CONTINUOUS dim one cell by default — emphatically not Lines’ 0', () => {
    // THE trap this arm exists for. Lines can use 0 because segment clipping
    // interpolates through the slab; mesh culls whole triangles with no
    // interpolation, so 0 reduces membership to exact float equality with the slice
    // plane and the node renders NOTHING.
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 2.0 },
    ];
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims)[3]).toBe(2.0);
    expect(computeTolerance('lines', [0, 1, 2], 4, dims)[3]).toBe(0);
  });

  it('never returns 0 for a hidden spatial dim, whatever the dimension metadata', () => {
    // The invariant behind the trap above, asserted across the shapes a store can
    // actually present: absent metadata, no step, and a zero step.
    const shapes: (DimensionInfo[] | undefined)[] = [
      undefined,
      [{ discrete: false }, { discrete: false }, { discrete: false }, { discrete: false }],
      [{ discrete: false }, { discrete: false }, { discrete: false }, { discrete: false, step: 0 }],
    ];
    for (const dims of shapes) {
      expect(computeTolerance('mesh', [0, 1, 2], 4, dims)[3]).toBeGreaterThan(0);
    }
  });

  it('honours meshSlabTolerance for the continuous arm', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 1.0 },
    ];
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims, { meshSlabTolerance: 3.0 })[3]).toBe(3.0);
  });
});
