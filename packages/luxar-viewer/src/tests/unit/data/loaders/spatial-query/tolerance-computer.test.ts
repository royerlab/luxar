import { describe, it, expect } from 'vitest';
import { computeTolerance, type DimensionInfo } from '../../../../../data/loaders';

const DISPLAYED = 1e10;

describe('computeTolerance — common rules', () => {
  it('sets infinite tolerance for displayed dimensions across all geometry types', () => {
    for (const type of ['points', 'lines', 'gsplats'] as const) {
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

  it('uses 0.5 for hidden dimensions flagged non-spatial via spatialExtendDims', () => {
    // Dimension 3 is non-spatial (discrete-like) per the per-dimension flag array
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true, false],
    });
    expect(tol[3]).toBe(0.5);
  });

  it('uses maxRadius for hidden dimensions flagged spatial via spatialExtendDims', () => {
    // Same shape, but dimension 3 is flagged spatial (true): it must use
    // maxRadius, not the 0.5 discrete value. This pins the true branch of the
    // spatialExtendDims flag so a flipped flag-check is caught.
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true, true],
    });
    expect(tol[3]).toBe(5.0);
    expect(tol[3]).not.toBe(0.5);
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
    expect(tol[3]).toBe(0.5); // discrete, step/2
    const spatialOnly: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false },
    ];
    const tol2 = computeTolerance('lines', [0, 1, 2], 4, spatialOnly);
    expect(tol2[3]).toBe(0);
  });

  it('uses step/2 for discrete hidden dimensions', () => {
    const tol = computeTolerance('lines', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.5);
  });

  it('falls back to 0.5 for discrete hidden dimensions without a step', () => {
    const dimsNoStep: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true },
    ];
    const tol = computeTolerance('lines', [0, 1, 2], 4, dimsNoStep);
    expect(tol[3]).toBe(0.5);
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

  it('uses 0.5 for hidden discrete dimensions', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 1.0 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.5);
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
