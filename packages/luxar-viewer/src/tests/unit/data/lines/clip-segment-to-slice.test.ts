// [data.md/O2][P10] Split from `lines-clipping.test.ts`: clipSegmentToSlice
// concrete cases + property-based invariants + small math helpers
// (lerp / lerpVec3 / distance3D) used by the clipper. Projection-pipeline
// tests live alongside in `project-lines-to-3d.test.ts` and
// `project-lines-wasm-parity.test.ts`.
import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { clipSegmentToSlice, lerp, lerpVec3, distance3D } from '../../../../data/lines/projection';

describe('clipSegmentToSlice', () => {
  // Default 3D display setup: display dims [0, 1, 2] (XYZ)
  const displayDims3D = [0, 1, 2];
  const slicePos3D = [0, 0, 0];
  const tolerance3D = [1e10, 1e10, 1e10]; // All dimensions visible in 3D

  // 4D setup: display dims [0, 1, 2], slice on dim 3
  const displayDims4D = [0, 1, 2];
  const slicePos4D = [0, 0, 0, 5]; // At dim3 = 5
  const tolerance4D = [1e10, 1e10, 1e10, 0.5]; // 0.5 tolerance on dim3

  describe('Case A: Both endpoints IN slice', () => {
    it('should return full segment when both endpoints within tolerance', () => {
      const p1 = [0, 0, 0, 5.2]; // Within +-0.5 of slicePos4D[3]=5
      const p2 = [10, 10, 10, 4.8];

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
      expect(result.p1).toEqual([0, 0, 0]); // Projected to 3D
      expect(result.p2).toEqual([10, 10, 10]);
    });

    it('should handle 3D data with all dimensions visible', () => {
      const p1 = [1, 2, 3];
      const p2 = [4, 5, 6];

      const result = clipSegmentToSlice(p1, p2, slicePos3D, tolerance3D, displayDims3D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
      expect(result.p1).toEqual([1, 2, 3]);
      expect(result.p2).toEqual([4, 5, 6]);
    });
  });

  describe('Case B: P1 IN, P2 OUT', () => {
    it('should clip P2 to slice boundary', () => {
      const p1 = [0, 0, 0, 5]; // IN (dim3 = 5, within 5+-0.5)
      const p2 = [10, 10, 10, 10]; // OUT (dim3 = 10, outside 5+-0.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0); // P1 unchanged
      expect(result.t2).toBeLessThan(1); // P2 clipped

      // t where dim3 crosses 5.5 (upper boundary)
      // p1[3] + t*(p2[3]-p1[3]) = 5.5
      // 5 + t*(10-5) = 5.5
      // t = 0.1
      expect(result.t2).toBeCloseTo(0.1, 5);

      // Interpolated 3D position at t=0.1
      expect(result.p2[0]).toBeCloseTo(1, 5); // 0 + 0.1*10
      expect(result.p2[1]).toBeCloseTo(1, 5);
      expect(result.p2[2]).toBeCloseTo(1, 5);
    });
  });

  describe('Case C: P1 OUT, P2 IN', () => {
    it('should clip P1 to slice boundary', () => {
      const p1 = [0, 0, 0, 0]; // OUT (dim3 = 0, outside 5+-0.5)
      const p2 = [10, 10, 10, 5]; // IN (dim3 = 5, within 5+-0.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBeGreaterThan(0); // P1 clipped
      expect(result.t2).toBe(1); // P2 unchanged

      // t where dim3 crosses 4.5 (lower boundary)
      // 0 + t*(5-0) = 4.5
      // t = 0.9
      expect(result.t1).toBeCloseTo(0.9, 5);

      // Interpolated 3D position at t=0.9
      expect(result.p1[0]).toBeCloseTo(9, 5);
      expect(result.p1[1]).toBeCloseTo(9, 5);
      expect(result.p1[2]).toBeCloseTo(9, 5);
    });
  });

  describe('Case D: Both OUT, opposite sides', () => {
    it('should clip both endpoints when segment crosses slice', () => {
      const p1 = [0, 0, 0, 0]; // OUT below (dim3 = 0)
      const p2 = [10, 10, 10, 10]; // OUT above (dim3 = 10)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBeGreaterThan(0); // P1 clipped
      expect(result.t2).toBeLessThan(1); // P2 clipped

      // t1 where dim3 crosses 4.5
      // 0 + t*(10-0) = 4.5 -> t = 0.45
      expect(result.t1).toBeCloseTo(0.45, 5);

      // t2 where dim3 crosses 5.5
      // 0 + t*(10-0) = 5.5 -> t = 0.55
      expect(result.t2).toBeCloseTo(0.55, 5);
    });

    // [R11/B-G2][P5/P8] Mirror the prior Case-D test with the segment
    // running in the opposite direction (dv < 0 in the source's
    // intersection branch). The `else` arm at the dv-sign split is
    // separate code; a regression that mutated `Math.max` ↔ `Math.min`
    // in just one arm would survive the original test. Pin both
    // visibility and the swapped (1-t2', 1-t1') symmetry.
    it('clips both endpoints when segment crosses slice in the reverse direction', () => {
      const p1 = [10, 10, 10, 10]; // OUT above (dim3 = 10)
      const p2 = [0, 0, 0, 0]; // OUT below (dim3 = 0)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBeGreaterThan(0);
      expect(result.t2).toBeLessThan(1);

      // Reversed segment: t1 now hits the upper boundary (5.5).
      // 10 + t*(0 - 10) = 5.5 -> 10 - 10t = 5.5 -> t = 0.45
      expect(result.t1).toBeCloseTo(0.45, 5);
      // t2 hits the lower boundary (4.5).
      // 10 + t*(0 - 10) = 4.5 -> 10 - 10t = 4.5 -> t = 0.55
      expect(result.t2).toBeCloseTo(0.55, 5);
    });
  });

  describe('Case E: Both OUT, same side', () => {
    it('should return invisible when both endpoints below slice', () => {
      const p1 = [0, 0, 0, 0]; // OUT (dim3 = 0)
      const p2 = [10, 10, 10, 2]; // OUT (dim3 = 2, still below 4.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(false);
    });

    it('should return invisible when both endpoints above slice', () => {
      const p1 = [0, 0, 0, 8]; // OUT (dim3 = 8)
      const p2 = [10, 10, 10, 10]; // OUT (dim3 = 10, both above 5.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(false);
    });
  });

  describe('Edge cases', () => {
    it('should handle segment exactly on slice boundary', () => {
      const p1 = [0, 0, 0, 4.5]; // ON lower boundary
      const p2 = [10, 10, 10, 5.5]; // ON upper boundary

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBeCloseTo(0, 5);
      expect(result.t2).toBeCloseTo(1, 5);
    });

    it('should handle segment parallel to slice dimension', () => {
      const p1 = [0, 0, 0, 5]; // All dim3 = 5
      const p2 = [10, 10, 10, 5];

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
    });

    it('should handle 2D display (padding to 3D)', () => {
      const displayDims2D = [0, 1]; // Only X, Y
      const p1 = [0, 0, 5]; // Z = 5, but we're slicing on it
      const p2 = [10, 10, 5];
      const slicePos = [0, 0, 5];
      const tolerance = [1e10, 1e10, 0.5]; // Slicing on Z

      const result = clipSegmentToSlice(p1, p2, slicePos, tolerance, displayDims2D);

      expect(result.visible).toBe(true);
      expect(result.p1).toEqual([0, 0, 0]); // 2D padded to 3D
      expect(result.p2).toEqual([10, 10, 0]);
    });

    it('should handle multiple slice dimensions', () => {
      // 5D case: display [0,1,2], slice on [3,4]
      const displayDims5D = [0, 1, 2];
      const slicePos5D = [0, 0, 0, 5, 10];
      const tolerance5D = [1e10, 1e10, 1e10, 0.5, 1.0];

      // Segment inside both slice dimensions
      const p1 = [0, 0, 0, 5.2, 10.5]; // Both within tolerance
      const p2 = [10, 10, 10, 4.8, 9.5];

      const result = clipSegmentToSlice(p1, p2, slicePos5D, tolerance5D, displayDims5D);

      expect(result.visible).toBe(true);
      expect(result.p1).toEqual([0, 0, 0]);
      expect(result.p2).toEqual([10, 10, 10]);
    });

    it('should clip in multiple dimensions', () => {
      // Segment that crosses slice in both non-display dimensions
      const displayDims5D = [0, 1, 2];
      const slicePos5D = [0, 0, 0, 5, 10];
      const tolerance5D = [1e10, 1e10, 1e10, 0.5, 1.0];

      const p1 = [0, 0, 0, 4, 8]; // Both outside
      const p2 = [10, 10, 10, 6, 12]; // Both outside

      const result = clipSegmentToSlice(p1, p2, slicePos5D, tolerance5D, displayDims5D);

      // Both cross through valid ranges - should be visible
      expect(result.visible).toBe(true);
      expect(result.t1).toBeGreaterThan(0);
      expect(result.t2).toBeLessThan(1);
    });

    // [P5] BOUNDARY: zero-length segment (p1 === p2 in all dims). Each hidden
    // dim sees v1 === v2, so for an in-slice point both are IN → `continue`;
    // dv === 0 never reaches the intersection branch. The result is the
    // degenerate "full segment" with t1=0, t2=1 and visible=true.
    it('zero-length segment fully inside the slice stays visible with t1=0, t2=1', () => {
      const p1 = [3, 4, 5, 5]; // dim3 = 5 == slicePos4D[3], inside tolerance
      const p2 = [3, 4, 5, 5]; // identical point

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
      // 3D-projected endpoints both equal the (degenerate) point.
      expect(result.p1).toEqual([3, 4, 5]);
      expect(result.p2).toEqual([3, 4, 5]);
    });

    // [P5] BOUNDARY: zero-length segment OUTSIDE the slice. Both endpoints are
    // on the same side (Case E) → invisible, regardless of dv being zero.
    it('zero-length segment outside the slice is invisible (Case E, both same side)', () => {
      const p1 = [0, 0, 0, 100]; // dim3 = 100, far above 5.5
      const p2 = [0, 0, 0, 100];

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(false);
    });

    // [P5] BOUNDARY: near-parallel segment with |dv| just BELOW the 1e-10
    // threshold. The parallel-handling branch (`if (Math.abs(dv) < 1e-10)
    // continue;`) skips the intersection math. Both endpoints are inside the
    // slice (dim3 ≈ 5), so the segment renders un-clipped (t1=0, t2=1).
    it('near-parallel segment (|dv| < 1e-10) takes the parallel branch — no clipping', () => {
      const dv = 1e-11; // strictly below the 1e-10 threshold
      const p1 = [0, 0, 0, 5];
      const p2 = [10, 10, 10, 5 + dv]; // dim3 barely changes; both within ±0.5

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
      // Display dims un-clipped → full projected segment.
      expect(result.p1).toEqual([0, 0, 0]);
      expect(result.p2).toEqual([10, 10, 10]);
    });

    it('should treat missing tolerance entries as zero', () => {
      const displayDims = [0, 1, 2];
      const slicePos = [0, 0, 0, 0];
      const tolerance = [1e10, 1e10, 1e10]; // No entry for dim 3

      const p1 = [0, 0, 0, 0];
      const p2 = [10, 10, 10, 0];

      const result = clipSegmentToSlice(p1, p2, slicePos, tolerance, displayDims);

      expect(result.visible).toBe(true);
    });
  });
});

describe('lerp', () => {
  it('should interpolate between values', () => {
    expect(lerp(0, 10, 0)).toBe(0);
    expect(lerp(0, 10, 1)).toBe(10);
    expect(lerp(0, 10, 0.5)).toBe(5);
    expect(lerp(0, 10, 0.25)).toBe(2.5);
  });

  it('should handle negative values', () => {
    expect(lerp(-10, 10, 0.5)).toBe(0);
    expect(lerp(-10, -5, 0.5)).toBe(-7.5);
  });
});

describe('lerpVec3', () => {
  it('should interpolate between 3D vectors', () => {
    const a = [0, 0, 0];
    const b = [10, 20, 30];

    expect(lerpVec3(a, b, 0)).toEqual([0, 0, 0]);
    expect(lerpVec3(a, b, 1)).toEqual([10, 20, 30]);
    expect(lerpVec3(a, b, 0.5)).toEqual([5, 10, 15]);
  });
});

describe('distance3D', () => {
  it('should calculate Euclidean distance', () => {
    expect(distance3D([0, 0, 0], [1, 0, 0])).toBe(1);
    expect(distance3D([0, 0, 0], [3, 4, 0])).toBe(5); // 3-4-5 triangle
    expect(distance3D([0, 0, 0], [1, 1, 1])).toBeCloseTo(Math.sqrt(3), 5);
  });

  it('should handle negative coordinates', () => {
    expect(distance3D([-1, 0, 0], [1, 0, 0])).toBe(2);
    expect(distance3D([0, -2, 0], [0, 2, 0])).toBe(4);
  });
});

// ============================================================================
// [data.md/H3][P12] Property tests for clipSegmentToSlice
//
// Three algebraic invariants of the nD slice-clipper that hold for every
// (p1, p2, slicePos, tolerance, displayDims) tuple where the segment is
// visible:
//
//   1. Endpoint symmetry: clipSegmentToSlice(p1, p2, ...) and
//      clipSegmentToSlice(p2, p1, ...) describe the same 3D segment
//      (endpoints possibly swapped) — order of vertices must not change
//      the geometric outcome.
//
//   2. In-slice endpoints: when both p1 and p2 already lie strictly
//      inside the per-hidden-dim slice tolerance, the result must be
//      "visible" with t1=0, t2=1, and the 3D endpoints equal to the
//      projection of (p1, p2) onto displayDims (no spurious clipping).
//
//   3. Interpolation consistency: the returned 3D endpoints must equal
//      the displayDims-projection of `p1 + t * (p2 - p1)` for t = t1
//      (start) and t = t2 (end). This pins the parameterization
//      contract that downstream attribute-interpolation depends on.
// ============================================================================

describe('clipSegmentToSlice (property tests)', () => {
  // Tight numeric arbitrary: 3D coordinates in [-100, 100], no NaN/Infinity.
  // Math.fround is required because fc.float emits 32-bit floats and the
  // production code does float32 math via Float32Array downstream.
  const f = Math.fround;
  const coordArb = fc.float({
    min: f(-100),
    max: f(100),
    noNaN: true,
    noDefaultInfinity: true,
  });

  it('[H3] endpoint symmetry: swap(p1,p2) -> swap(result.p1,result.p2), same visibility', () => {
    // 4D inputs: 3 display dims + 1 hidden dim. Both endpoints inside
    // the slice (tolerance is large) so visibility is guaranteed; this
    // isolates the symmetry property from the visibility branch.
    fc.assert(
      fc.property(
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        (x1, y1, z1, w1, x2, y2, z2, w2) => {
          const p1 = [x1, y1, z1, w1];
          const p2 = [x2, y2, z2, w2];
          const slicePos = [0, 0, 0, 0];
          // Large tolerance on hidden dim => both endpoints always IN
          const tolerance = [1e10, 1e10, 1e10, 1e10];
          const displayDims = [0, 1, 2];

          const r12 = clipSegmentToSlice(p1, p2, slicePos, tolerance, displayDims);
          const r21 = clipSegmentToSlice(p2, p1, slicePos, tolerance, displayDims);

          expect(r12.visible).toBe(r21.visible);
          if (!r12.visible) return;

          // Reversed-input result must describe the same segment with
          // endpoints swapped. We compare 3D-projected endpoints.
          for (let i = 0; i < 3; i++) {
            expect(r21.p1[i]).toBeCloseTo(r12.p2[i], 4);
            expect(r21.p2[i]).toBeCloseTo(r12.p1[i], 4);
          }
        }
      ),
      { numRuns: 60, seed: 0x5eed }
    );
  });

  it('[H3] in-slice endpoints: t1=0, t2=1, 3D endpoints = projection(p1,p2)', () => {
    // When both endpoints lie inside the slice along every hidden dim,
    // no clipping should happen.
    fc.assert(
      fc.property(
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        coordArb,
        (x1, y1, z1, x2, y2, z2) => {
          const p1 = [x1, y1, z1, 0]; // hidden-dim value = slice center
          const p2 = [x2, y2, z2, 0];
          const slicePos = [0, 0, 0, 0];
          const tolerance = [1e10, 1e10, 1e10, 0.5];
          const displayDims = [0, 1, 2];

          const r = clipSegmentToSlice(p1, p2, slicePos, tolerance, displayDims);

          expect(r.visible).toBe(true);
          expect(r.t1).toBe(0);
          expect(r.t2).toBe(1);
          // The 3D-projected endpoints must equal p1/p2 restricted to display dims.
          expect(r.p1[0]).toBeCloseTo(x1, 4);
          expect(r.p1[1]).toBeCloseTo(y1, 4);
          expect(r.p1[2]).toBeCloseTo(z1, 4);
          expect(r.p2[0]).toBeCloseTo(x2, 4);
          expect(r.p2[1]).toBeCloseTo(y2, 4);
          expect(r.p2[2]).toBeCloseTo(z2, 4);
        }
      ),
      { numRuns: 60, seed: 0x5eed }
    );
  });

  it('[H3] interpolation consistency: result.p1 == proj(p1 + t1*(p2-p1)), result.p2 == proj(p1 + t2*(p2-p1))', () => {
    // The t1, t2 fields are how downstream interpolation reproduces
    // per-vertex attribute values at the clipped endpoints. Pin the
    // contract that the 3D endpoints really do match lerp(p1, p2, t)
    // restricted to displayDims for both t1 and t2.
    fc.assert(
      fc.property(
        coordArb,
        coordArb,
        coordArb,
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        coordArb,
        coordArb,
        coordArb,
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        (x1, y1, z1, w1, x2, y2, z2, w2) => {
          const p1 = [x1, y1, z1, w1];
          const p2 = [x2, y2, z2, w2];
          const slicePos = [0, 0, 0, 0];
          const tolerance = [1e10, 1e10, 1e10, 0.5];
          const displayDims = [0, 1, 2];

          const r = clipSegmentToSlice(p1, p2, slicePos, tolerance, displayDims);
          if (!r.visible) return;

          // Reconstruct lerp(p1, p2, t1) and lerp(p1, p2, t2) along display dims.
          const lerpAt = (t: number, dim: number) => p1[dim] + t * (p2[dim] - p1[dim]);
          for (let i = 0; i < 3; i++) {
            const d = displayDims[i];
            expect(r.p1[i]).toBeCloseTo(lerpAt(r.t1, d), 3);
            expect(r.p2[i]).toBeCloseTo(lerpAt(r.t2, d), 3);
          }
        }
      ),
      { numRuns: 80, seed: 0x5eed }
    );
  });
});
