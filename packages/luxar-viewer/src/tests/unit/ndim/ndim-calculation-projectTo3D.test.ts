/**
 * Tests for ndim calculation inside the REAL `projectPointsTo3D`.
 *
 * Previous version (~522 lines) tested LOCAL re-implementations of the ndim
 * calculation and the display-dim extraction loop, not the production
 * function. Any drift between the local copy and the source went undetected
 * (delme/test-audit-luxar-viewer.src/ndim.md, C1/C2).
 *
 * This rewrite imports `projectPointsTo3D` directly and pins the
 * load-bearing contract through it: ndim is computed from the positions
 * array length (positions.length / totalPoints) rather than from
 * `ctx.chunkIndex?.metadata.ndim`. The chunkIndex value is used only as a
 * fallback when totalPoints === 0.
 */
import { describe, it, expect } from 'vitest';
import { projectPointsTo3D, type ProjectionContext } from '../../../data/points/projection';
import type { ViewState, PointRange } from '../../../data/data-loader-types';
import type { PointsMetadata } from '../../../types/points';

function makeContext(overrides: Partial<ProjectionContext> = {}): ProjectionContext {
  const nodeAttrs: PointsMetadata = {
    type: 'points',
    n_points: 0,
  } as PointsMetadata;
  return {
    chunkIndex: null,
    effectiveRadiusConfig: null,
    accumulator: null,
    nodeAttrs,
    ...overrides,
  };
}

const viewState3D: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
};

const viewState4D: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

describe('projectPointsTo3D — ndim is computed from positions array length', () => {
  it('3D positions: positions.length / totalPoints === 3, output positions3D has 3 components per point', () => {
    // 4 points, 3D: total 12 elements.
    const positions = new Float32Array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const ranges: PointRange[] = [{ start: 0, end: 4 }];
    const result = projectPointsTo3D(
      positions,
      null,
      null,
      null,
      viewState3D,
      ranges,
      makeContext()
    );

    // Output has exactly 3 components per point.
    expect(result.pointCount).toBe(4);
    expect(result.positions.length).toBe(4 * 3);
    expect(result.ndim).toBe(3);
    // The first point projects [0,0,0] → [0,0,0]; the second [1,2,3].
    expect(Array.from(result.positions.slice(0, 6))).toEqual([0, 0, 0, 1, 2, 3]);
  });

  it('4D positions: positions.length / totalPoints === 4, output extracts displayDims 0,1,2', () => {
    // 3 points, 4D: total 12 elements. Layout per point: [x, y, z, t].
    const positions = new Float32Array([1, 2, 3, 100, 4, 5, 6, 200, 7, 8, 9, 300]);
    const ranges: PointRange[] = [{ start: 0, end: 3 }];
    const result = projectPointsTo3D(
      positions,
      null,
      null,
      null,
      viewState4D,
      ranges,
      makeContext()
    );

    expect(result.pointCount).toBe(3);
    expect(result.positions.length).toBe(3 * 3);
    expect(result.ndim).toBe(4);
    // 4D → 3D extraction with displayDims [0,1,2]: t dimension is dropped.
    expect(Array.from(result.positions)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('5D positions: positions.length / totalPoints === 5, displayDims [0,2,4] picks the right slots', () => {
    // 2 points, 5D: layout per point [d0, d1, d2, d3, d4].
    const positions = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    const ranges: PointRange[] = [{ start: 0, end: 2 }];
    const viewState5D: ViewState = {
      displayDims: [0, 2, 4],
      slicePosition: [0, 0, 0, 0, 0],
      tolerance: [0, 1, 0, 1, 0],
    };
    const result = projectPointsTo3D(
      positions,
      null,
      null,
      null,
      viewState5D,
      ranges,
      makeContext()
    );

    expect(result.pointCount).toBe(2);
    expect(result.ndim).toBe(5);
    // Point 1: positions[0,2,4] = [10, 30, 50]; point 2: positions[5,7,9] = [60, 80, 100].
    expect(Array.from(result.positions)).toEqual([10, 30, 50, 60, 80, 100]);
  });

  it('totalPoints=0 falls back to ctx.chunkIndex?.metadata.ndim (not positions.length / 0)', () => {
    // Empty positions + totalPoints=0: ndim should come from chunkIndex.
    const positions = new Float32Array(0);
    const ranges: PointRange[] = [];
    const ctx = makeContext({
      // Stub chunkIndex with metadata.ndim = 7 to verify the fallback path.
      chunkIndex: { metadata: { ndim: 7 } } as ProjectionContext['chunkIndex'],
    });
    const result = projectPointsTo3D(positions, null, null, null, viewState3D, ranges, ctx);
    expect(result.pointCount).toBe(0);
    expect(result.positions.length).toBe(0);
    // The contract under test: ndim flows from chunkIndex.metadata.ndim
    // when positions is empty. A mutant in projection.ts that returned a
    // hardcoded 3 instead of chunkIndex?.metadata.ndim would survive
    // without this assertion.
    expect(result.ndim).toBe(7);
  });

  it('positions.length not divisible by totalPoints throws fast (MED-14)', () => {
    // 10 points but only 39 elements → ndim = round(39/10) = 4 but 10*4 ≠ 39.
    // Production code formerly logged an error and proceeded with the wrong
    // ndim, producing silently-corrupted positions3D downstream. MED-14
    // fix throws to make the encoding-metadata bug visible at the boundary.
    const positions = new Float32Array(39);
    const ranges: PointRange[] = [{ start: 0, end: 10 }];
    expect(() =>
      projectPointsTo3D(positions, null, null, null, viewState4D, ranges, makeContext())
    ).toThrow(/size mismatch/i);
  });
});

// [ndim.md/G][P5] Boundary tests for projectPointsTo3D corner cases.
// Round 8 follow-up — extends coverage to single-point, displayDims.length=0,
// and displayDims-with-out-of-range entries.
describe('projectPointsTo3D — corner cases', () => {
  it('single-point input: positions3D has length 3 and value matches displayDims selection', () => {
    // 1 point, 5D: pick dims [1, 3, 0].
    const positions = new Float32Array([10, 20, 30, 40, 50]);
    const ranges: PointRange[] = [{ start: 0, end: 1 }];
    const viewState: ViewState = {
      displayDims: [1, 3, 0],
      slicePosition: [0, 0, 0, 0, 0],
      tolerance: [0, 0, 0, 0, 0],
    };
    const result = projectPointsTo3D(positions, null, null, null, viewState, ranges, makeContext());
    expect(result.pointCount).toBe(1);
    expect(result.positions.length).toBe(3);
    // displayDims=[1,3,0] picks values [20, 40, 10] from the source point.
    expect(Array.from(result.positions)).toEqual([20, 40, 10]);
  });

  it('displayDims.length=0 (no displayed dim): output positions are all zero', () => {
    // 2 points, 3D, no displayed dims → output is all zeros (source
    // fills the remaining `j < 3` slots with 0; with displayDims.length=0
    // ALL 3 slots are zero-filled).
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    const ranges: PointRange[] = [{ start: 0, end: 2 }];
    const viewState: ViewState = {
      displayDims: [],
      slicePosition: [0, 0, 0],
      tolerance: [0, 0, 0],
    };
    const result = projectPointsTo3D(positions, null, null, null, viewState, ranges, makeContext());
    expect(result.pointCount).toBe(2);
    expect(result.positions.length).toBe(6);
    expect(Array.from(result.positions)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('displayDims.length=1 (one displayed dim): output X is that dim, Y/Z are 0', () => {
    // 2 points, 4D, displayDims=[2] → output is [d2, 0, 0] per point.
    const positions = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const ranges: PointRange[] = [{ start: 0, end: 2 }];
    const viewState: ViewState = {
      displayDims: [2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    };
    const result = projectPointsTo3D(positions, null, null, null, viewState, ranges, makeContext());
    expect(result.pointCount).toBe(2);
    expect(Array.from(result.positions)).toEqual([30, 0, 0, 70, 0, 0]);
  });

  it('multi-range input: pointCount sums across ranges (covers reduce path)', () => {
    // 4 points, 3D — two disjoint ranges of size 1 + 1 = 2 visible.
    // Note: projectPointsTo3D uses totalPoints to derive ndim from the
    // whole positions array, but pointCount in the result reflects only
    // the ranges' sum — pin the contract.
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    const ranges: PointRange[] = [
      { start: 0, end: 1 },
      { start: 1, end: 2 },
    ];
    const result = projectPointsTo3D(
      positions,
      null,
      null,
      null,
      viewState3D,
      ranges,
      makeContext()
    );
    // pointCount comes from ranges.sum (2), positions array is 2*3 = 6.
    expect(result.pointCount).toBe(2);
    expect(result.positions.length).toBe(6);
  });
});
